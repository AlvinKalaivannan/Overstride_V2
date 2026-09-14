"""Phase 4B -- the three things a demo needs that AUC does not give you.

Everything in this project so far is reported as AUC. A demo shell cannot be
built on AUC: it has to emit a decision or a number to a person, and that
requires knowing what happens at a THRESHOLD, whether the probability attached
to it means anything, and whether the same runner gets the same answer twice.
None of the three has been measured. Phase 6 should not start until they are.

  1. OPERATING POINT   sensitivity / specificity / PPV / NPV at thresholds,
                       from out-of-fold scores only.
  2. CALIBRATION       Brier score, calibration slope and intercept, reliability
                       bins. An uncalibrated probability shown to a user is
                       worse than no probability.
  3. REPEATABILITY     74 subjects have two or more sessions. Does the model
                       give the same answer on both? This is the closest thing
                       available to test-retest reliability, and it is the
                       question a user actually asks.

  4. CONVENTION SENSITIVITY. phase3_angles.py flagged that keypoint three-point
     angles and Ferber Cardan angles are NOT the same quantity, and that the
     mismatch "must be handled in phase 4 rather than assumed away". Phase 4
     injects three-point-angle RESIDUALS into Cardan curves, which assumes error
     magnitude transfers between conventions. That assumption is not directly
     testable with these datasets -- no subject has both -- so instead this
     bounds it: scale the residuals and find how far off the transfer would have
     to be before the verdict changes.

Run: .venv/Scripts/python.exe scripts/phase4b_operating_point.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_eval import check_no_group_leakage, ci, make_folds, make_model
from phase4_real_delta import (LATERAL_RATIO, PCA_PER_CHANNEL,  # noqa: E402
                               build_features, load_bank, load_limb_cohort)
from phase5_limb import JOINTS3  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase4b_operating_point.json"
WAVE2 = ["knee", "hip"]
N_REALIZATIONS = 3


def oof_predictions(frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                    folds: list[dict], nch: int) -> np.ndarray:
    """Out-of-fold probability per session, averaged over the 5 seeds.

    Each session is in the test set exactly once per seed, so every session gets
    5 genuinely out-of-fold scores. Averaging them is the honest analogue of
    the averaged AUC reported elsewhere -- and it never uses a score from a
    fold where that subject was in training.
    """
    cols = list(frame.columns)
    acc = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for f in folds:
        tr, te = f["train"], f["test"]
        model, needs_groups = make_model("logit", cols, [], "explicit",
                                         PCA_PER_CHANNEL * nch)
        kw = {"clf__groups": groups[tr]} if needs_groups else {}
        model.fit(frame.iloc[tr], y[tr], **kw)
        acc[te] += model.predict_proba(frame.iloc[te])[:, 1]
        cnt[te] += 1
    assert cnt.min() > 0, "a session was never held out"
    return acc / cnt


def operating_points(y: np.ndarray, p: np.ndarray) -> list[dict]:
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1 - fpr
    out = []
    picks = [("Youden-optimal", int(np.argmax(tpr - fpr))),
             ("specificity >= 0.90", int(np.argmin(np.abs(spec - 0.90)))),
             ("sensitivity >= 0.90", int(np.argmin(np.abs(tpr - 0.90)))),
             ("threshold 0.50", int(np.argmin(np.abs(thr - 0.50))))]
    prev = y.mean()
    for name, i in picks:
        se, sp = float(tpr[i]), float(spec[i])
        ppv = se * prev / max(se * prev + (1 - sp) * (1 - prev), 1e-9)
        npv = sp * (1 - prev) / max(sp * (1 - prev) + (1 - se) * prev, 1e-9)
        out.append({"point": name, "threshold": float(thr[i]),
                    "sensitivity": se, "specificity": sp,
                    "ppv": float(ppv), "npv": float(npv),
                    "balanced_accuracy": float((se + sp) / 2)})
    return out


def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    """Slope and intercept of the logit-linear recalibration.

    slope 1 / intercept 0 is perfect. slope < 1 means the probabilities are too
    extreme for the evidence behind them, which is the failure mode that matters
    when a number is shown to a user.
    """
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    slope = float(LogisticRegression(C=1e9, max_iter=5000)
                  .fit(lp.reshape(-1, 1), y).coef_[0][0])
    inter = float(LogisticRegression(C=1e9, max_iter=5000)
                  .fit(np.zeros((len(y), 1)), y).intercept_[0])
    bins = np.quantile(p, np.linspace(0, 1, 6))
    bins[-1] += 1e-9
    rel = []
    for i in range(len(bins) - 1):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum():
            rel.append({"n": int(m.sum()), "mean_pred": float(p[m].mean()),
                        "observed": float(y[m].mean())})
    return {"brier": float(brier_score_loss(y, p)),
            "brier_baseline": float(brier_score_loss(y, np.full_like(p, y.mean()))),
            "calibration_slope": slope, "calibration_intercept": inter,
            "reliability_bins": rel}


def repeatability(meta: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> dict:
    """Same subject, different session -- same answer?"""
    df = pd.DataFrame({"sub_id": meta["sub_id"].to_numpy(), "y": y, "p": p})
    multi = df.groupby("sub_id").filter(lambda g: len(g) >= 2)
    same_side = multi.groupby("sub_id").filter(lambda g: g["y"].nunique() == 1)
    calls, scores = [], []
    for _, g in same_side.groupby("sub_id"):
        pv = g["p"].to_numpy()
        for a in range(len(pv)):
            for b in range(a + 1, len(pv)):
                calls.append(int((pv[a] > 0.5) == (pv[b] > 0.5)))
                scores.append((pv[a], pv[b]))
    s = np.array(scores)
    return {"n_subjects_multi_session": int(multi["sub_id"].nunique()),
            "n_subjects_same_side": int(same_side["sub_id"].nunique()),
            "n_session_pairs": len(calls),
            "binary_agreement": float(np.mean(calls)) if calls else float("nan"),
            "score_corr": float(np.corrcoef(s[:, 0], s[:, 1])[0, 1])
            if len(s) > 2 else float("nan"),
            "mean_abs_score_diff": float(np.abs(s[:, 0] - s[:, 1]).mean())
            if len(s) else float("nan")}


def main() -> int:
    mean, pos, signs, meta, y, groups = load_limb_cohort()
    B, pairs, pair_ratio = load_bank()
    lateral = pairs[pair_ratio >= LATERAL_RATIO]
    folds = make_folds(y, groups)
    check_no_group_leakage(folds, groups)
    rng0 = np.random.default_rng(20260810)
    far_is_right = rng0.random(len(meta)) < 0.5
    print(f"cohort {len(meta)} sessions / {meta['sub_id'].nunique()} subjects, "
          f"chance {y.mean():.3f}; {len(folds)} folds\n")

    kw = dict(mean=mean, pos=pos, signs=signs, B=B, far_is_right=far_is_right)
    configs = {
        "clean mocap (upper bound)":
            dict(k=101, seed=0, perturb=False, pool=pairs),
        "deployment: lateral bank, 30 fps":
            dict(k=9, seed=700, perturb=True, pool=lateral),
    }

    out: dict = {"configs": {}}
    for name, cfg in configs.items():
        frame = build_features(joints=WAVE2, **cfg, **kw)
        p = oof_predictions(frame, y, groups, folds, nch=2)
        auc = float(roc_auc_score(y, p))
        print(f"=== {name} ===")
        print(f"  pooled out-of-fold AUC {auc:.3f}")
        ops = operating_points(y, p)
        print(f"  {'operating point':<22}{'thr':>7}{'sens':>7}{'spec':>7}"
              f"{'PPV':>7}{'NPV':>7}{'bal acc':>9}")
        for o in ops:
            print(f"  {o['point']:<22}{o['threshold']:>7.3f}"
                  f"{o['sensitivity']:>7.3f}{o['specificity']:>7.3f}"
                  f"{o['ppv']:>7.3f}{o['npv']:>7.3f}"
                  f"{o['balanced_accuracy']:>9.3f}")
        cal = calibration(y, p)
        print(f"  Brier {cal['brier']:.4f} vs {cal['brier_baseline']:.4f} "
              f"baseline | calibration slope {cal['calibration_slope']:.3f} "
              f"(1.0 = perfect)")
        for b in cal["reliability_bins"]:
            print(f"    n={b['n']:>4}  mean score {b['mean_pred']:.3f}  "
                  f"observed {b['observed']:.3f}")
        rep = repeatability(meta, y, p)
        print(f"  repeatability: {rep['n_session_pairs']} session pairs from "
              f"{rep['n_subjects_same_side']} subjects | "
              f"binary agreement {rep['binary_agreement']:.3f} | "
              f"score r {rep['score_corr']:+.3f}\n")
        out["configs"][name] = {"auc_pooled_oof": auc, "operating_points": ops,
                                "calibration": cal, "repeatability": rep}

    # --- 4. how wrong could the angle-convention transfer be? --------------
    print("=== CONVENTION SENSITIVITY: residual magnitude scaled ===")
    print("  Ferber curves are Cardan angles; the residuals are three-point")
    print("  angle errors. Magnitude transfer is an assumption -- this bounds it.\n")
    print(f"  {'scale':>6}{'realized err':>14}{'AUC':>8}   CI")
    scale_rows = []
    for scale in (0.5, 1.0, 1.5, 2.0, 3.0):
        realized: dict[str, float] = {}
        per_fold = []
        for rep in range(N_REALIZATIONS):
            frame = build_features(joints=WAVE2, k=9, seed=700 + rep,
                                   perturb=True, pool=lateral, scale=scale,
                                   realized=realized, tag=f"s{scale}", **kw)
            aucs = []
            for f in folds:
                tr, te = f["train"], f["test"]
                model, ng = make_model("logit", list(frame.columns), [],
                                       "explicit", PCA_PER_CHANNEL * 2)
                kw2 = {"clf__groups": groups[tr]} if ng else {}
                model.fit(frame.iloc[tr], y[tr], **kw2)
                aucs.append(roc_auc_score(
                    y[te], model.predict_proba(frame.iloc[te])[:, 1]))
            per_fold.append(np.array(aucs))
        pf = np.mean(per_fold, axis=0)
        lo, hi = ci(pf)
        print(f"  {scale:>6.1f}{realized[f's{scale}']:>13.2f}d{pf.mean():>8.3f}"
              f"   [{lo:.3f}, {hi:.3f}]"
              f"{'' if lo > 0.5 else '   <- CI touches chance'}")
        scale_rows.append({"scale": scale,
                           "realized_error_deg": realized[f"s{scale}"],
                           "auc_mean": float(pf.mean()), "ci": [lo, hi],
                           "survives": bool(lo > 0.5)})
    out["convention_sensitivity"] = scale_rows

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
