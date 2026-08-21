"""Phase 5D -- can abstention make a 0.61 model usable?

Phases 5B and 5C established that the injured-limb signal cannot be raised: a
different measurement modality hits the identical ceiling, the learning curve is
saturated, and nothing tried moved it. This is the remaining option -- stop
trying to raise the signal and instead REFUSE TO ANSWER when the model is not
confident.

  coverage   fraction of scans the system answers at all
  selective  accuracy among only those it chose to answer
             accuracy

The question a demo actually needs answered: is there a coverage at which the
accuracy is high enough to show a person, and is the resulting system still
useful at that coverage?

NO LABEL LEAKAGE IN THE SELECTION
  Selection is by the model's OWN confidence |p - 0.5|, never by the label, and
  the top-X% is taken WITHIN EACH TEST FOLD independently. Coverage is a design
  parameter, not a fitted threshold, so nothing is tuned on the evaluation data.

THE CONTROL THAT MAKES THIS INTERPRETABLE
  A selective-accuracy curve that rises as coverage falls looks like a result but
  can be an artifact of the procedure. So the identical sweep is run on the
  PROVENANCE-ONLY model, which phases 4/5 verified sits at chance on this task.
  Its curve must stay flat. If it climbs, the method is manufacturing the effect
  and every number here is void.

Run: .venv/Scripts/python.exe scripts/phase5d_selective.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_cohort import PROVENANCE_FEATURES  # noqa: E402
from phase1_eval import (check_no_group_leakage, ci,  # noqa: E402
                         make_folds, make_model)
from phase4_real_delta import (LATERAL_RATIO, PCA_PER_CHANNEL,  # noqa: E402
                               build_features, load_bank, load_limb_cohort)
from phase5_limb import JOINTS3, SAGITTAL_PLANE, mirror_signs  # noqa: E402
from phase5c_features import stride_features  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase5d_selective.json"
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
WAVE2 = ["knee", "hip"]


def fold_scores(frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                folds: list[dict], nch: int | None) -> list[tuple]:
    """(test_idx, score) per fold. Kept per fold so selection can happen
    inside a fold rather than across the pooled set."""
    out = []
    cols = list(frame.columns)
    for f in folds:
        tr, te = f["train"], f["test"]
        model, ng = make_model("logit", cols, [], "explicit",
                               PCA_PER_CHANNEL * nch if nch else None)
        kw = {"clf__groups": groups[tr]} if ng else {}
        model.fit(frame.iloc[tr], y[tr], **kw)
        out.append((te, model.predict_proba(frame.iloc[te])[:, 1]))
    return out


def selective_curve(per_fold: list[tuple], y: np.ndarray,
                    groups: np.ndarray) -> list[dict]:
    """Accuracy among the most-confident X% of each test fold."""
    rows = []
    for cov in COVERAGES:
        accs, bals, aucs, kept_n = [], [], [], 0
        pooled_y, pooled_p = [], []
        for te, p in per_fold:
            k = max(2, int(round(cov * len(te))))
            conf = np.abs(p - 0.5)
            sel = np.argsort(-conf)[:k]          # most confident first
            ys, ps = y[te][sel], p[sel]
            kept_n += k
            pred = (ps > 0.5).astype(int)
            accs.append((pred == ys).mean())
            if len(np.unique(ys)) == 2:
                se = (pred[ys == 1] == 1).mean()
                sp = (pred[ys == 0] == 0).mean()
                bals.append((se + sp) / 2)
                aucs.append(roc_auc_score(ys, ps))
            pooled_y.append(ys)
            pooled_p.append(ps)
        py = np.concatenate(pooled_y)
        a = np.asarray(accs)
        lo, hi = ci(a)
        rows.append({
            "coverage": cov,
            "mean_kept_per_fold": kept_n / len(per_fold),
            "n_unique_sessions": int(round(cov * len(y))),
            "accuracy": float(a.mean()),
            "accuracy_ci": [lo, hi],
            "balanced_accuracy": float(np.mean(bals)) if bals else float("nan"),
            "auc_on_kept": float(np.mean(aucs)) if aucs else float("nan"),
            "base_rate_kept": float(py.mean()),
        })
    return rows


def show(name: str, rows: list[dict], chance: float) -> None:
    print(f"\n  {name}")
    print(f"    {'coverage':>9}{'n/fold':>8}{'accuracy':>10}   {'95% CI':<17}"
          f"{'bal acc':>9}{'AUC':>7}{'lift':>8}  beats base?")
    for r in rows:
        lift = r["accuracy"] - max(chance, 1 - chance)
        lo, hi = r["accuracy_ci"]
        beats = lo > max(chance, 1 - chance)
        print(f"    {r['coverage']:>8.0%}{r['mean_kept_per_fold']:>8.0f}"
              f"{r['accuracy']:>10.3f}   [{lo:.3f}, {hi:.3f}]  "
              f"{r['balanced_accuracy']:>9.3f}{r['auc_on_kept']:>7.3f}"
              f"{lift:>+8.3f}  {'YES' if beats else 'no'}")


def main() -> int:
    mean, pos, signs, meta, y, groups = load_limb_cohort()
    folds = make_folds(y, groups)
    check_no_group_leakage(folds, groups)
    chance = float(y.mean())
    majority = max(chance, 1 - chance)
    print(f"{len(folds)} folds, groups=sub_id | chance {chance:.3f}, "
          f"always-guess-majority accuracy {majority:.3f}\n")
    _ = mirror_signs({"mean": mean}, pos)

    B, pairs, pair_ratio = load_bank()
    lateral = pairs[pair_ratio >= LATERAL_RATIO]
    rng0 = np.random.default_rng(20260810)
    far_is_right = rng0.random(len(meta)) < 0.5
    kw = dict(mean=mean, pos=pos, signs=signs, B=B, far_is_right=far_is_right)

    # sagittal reference
    cols, blocks = [], []
    for j in JOINTS3:
        s = signs[(j, SAGITTAL_PLANE)]
        blocks.append(mean[:, pos[f"ang_R_{j}_p{SAGITTAL_PLANE}"], :]
                      - s * mean[:, pos[f"ang_L_{j}_p{SAGITTAL_PLANE}"], :])
        cols += [f"d_{j}_t{t:03d}" for t in range(101)]
    sag = pd.DataFrame(np.concatenate(blocks, axis=1), columns=cols)

    print("building stride-distribution features (phase 5C best set)")
    stf = stride_features(meta, signs)

    configs = {
        "clean mocap, limbsag (upper bound)": (sag, 3),
        "clean mocap, limbsag + strides (5C best)":
            (pd.concat([sag, stf], axis=1), None),
        "deployment: wave2, side-on bank, 30 fps":
            (build_features(joints=WAVE2, k=9, seed=700, perturb=True,
                            pool=lateral, **kw), 2),
        "CONTROL: provenance only (must stay FLAT)":
            (meta[PROVENANCE_FEATURES].astype(float), None),
    }

    print("\n=== SELECTIVE CLASSIFICATION ===")
    print("  selection is by |p - 0.5| within each test fold; labels are never "
          "used to select")
    out: dict = {"chance": chance, "majority_accuracy": majority,
                 "n_sessions": int(len(meta)), "curves": {}}
    curves = {}
    for name, (frame, nch) in configs.items():
        per_fold = fold_scores(frame, y, groups, folds, nch)
        rows = selective_curve(per_fold, y, groups)
        curves[name] = rows
        out["curves"][name] = rows
        show(name, rows, chance)

    # --- did the control stay flat? ---------------------------------------
    ctrl = curves["CONTROL: provenance only (must stay FLAT)"]
    drift = max(abs(r["accuracy"] - ctrl[0]["accuracy"]) for r in ctrl)
    verdict = ("FLAT, method is sound" if drift < 0.08 else
               "NOT FLAT -- the procedure is manufacturing the effect, void")
    print(f"\n  control drift across the sweep: {drift:.3f} -> {verdict}")
    out["control_drift"] = float(drift)

    # --- multi-session averaging ------------------------------------------
    # 72 subjects have two or more sessions with the same injured side. If the
    # error is partly independent between sessions, averaging their scores
    # should beat a single scan.
    print("\n=== MULTI-SESSION AVERAGING ===")
    best_frame, best_nch = configs["clean mocap, limbsag (upper bound)"]
    per_fold = fold_scores(best_frame, y, groups, folds, best_nch)
    acc_s = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for te, p in per_fold:
        acc_s[te] += p
        cnt[te] += 1
    oof = acc_s / cnt

    df = pd.DataFrame({"sub_id": meta["sub_id"].to_numpy(), "y": y, "p": oof})
    multi = df.groupby("sub_id").filter(
        lambda g: len(g) >= 2 and g["y"].nunique() == 1)
    single_acc = float(((multi["p"] > 0.5).astype(int) == multi["y"]).mean())
    avg = multi.groupby("sub_id").agg(y=("y", "first"), p=("p", "mean"))
    avg_acc = float(((avg["p"] > 0.5).astype(int) == avg["y"]).mean())
    print(f"  {avg.shape[0]} subjects with >=2 same-side sessions "
          f"({len(multi)} sessions)")
    print(f"    single-session accuracy on those sessions : {single_acc:.3f}")
    print(f"    accuracy after averaging per subject      : {avg_acc:.3f}  "
          f"({avg_acc - single_acc:+.3f})")
    out["multi_session"] = {"n_subjects": int(avg.shape[0]),
                            "n_sessions": int(len(multi)),
                            "single_session_accuracy": single_acc,
                            "averaged_accuracy": avg_acc,
                            "gain": avg_acc - single_acc}

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
