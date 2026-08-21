"""Phase 5C -- three attempts to raise the injured-limb signal.

SCOPE, set by phase 5B's result rather than by ambition
  5B pre-registered that a flat severity dose-response would mean Stage 2 gets
  "scoped down to the cheap items only". It came out flat (Spearman +0.044,
  CI spanning zero) and, worse, ANTI-monotone by stratum. The dv_r probe then
  landed at 0.610 -- identical to the sagittal waveform model, from a completely
  different measurement modality.

  So the expensive stride-pair augmentation (R3b in the plan) is NOT run. What is
  run is everything cheap, plus ONE genuinely different information axis:
  within-session stride DISTRIBUTIONS, which no earlier phase touched and which
  neither 5B probe covers -- both were session-level aggregates.

R1  velocity asymmetry. waveforms_mean.npy carries 24 velocity channels that
    phase 5 never read (it touches only `ang_*`). Free.
R2  per-condition models. ITBS n=100, PFPS n=81. Underpowered by construction,
    BH-corrected, hypothesis-generating only.
R3a within-session stride distributions, streamed from waveforms_steps/*.mat.
    L and R have UNEQUAL stride counts (median 29 each but equal in only 25% of
    sessions), so this is distribution-vs-distribution, not paired.

Every feature set is scored on the SAME 25 folds as phases 4/5, so every delta is
paired. ALL sets tried are reported, not the best -- running ten and reporting
the winner is how a spurious result gets manufactured here.

Run: .venv/Scripts/python.exe scripts/phase5c_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_eval import (check_no_group_leakage, evaluate,  # noqa: E402
                         fmt_result, make_folds, make_model, paired_delta)
from phase4_real_delta import load_limb_cohort  # noqa: E402
from phase5b_ceiling import derive_signs, negative_controls  # noqa: E402
from phase5_limb import JOINTS3, SAGITTAL_PLANE, mirror_signs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
STEPS = DERIVED / "waveforms_steps"
OUT = REPO / "results" / "phase5c_features.json"

PCA_PER_CHANNEL = 3
N_BLOCKS = 5
MIN_CONDITION_N = 35
VEL_JOINTS = ["ankle", "knee", "hip", "pelvis"]


def subject_bootstrap_ci(y: np.ndarray, score: np.ndarray, groups: np.ndarray,
                         n_boot: int = 2000, seed: int = 31) -> tuple:
    """AUC CI from resampling SUBJECTS, which is the independent unit here.

    WHY NOT A FOLD-LEVEL TEST. The plan called for BH-corrected p-values across
    the family. A Wilcoxon over the 25 fold AUCs was implemented first and
    discarded: the 25 folds are 5 seeds x 5 splits of ONE dataset, so they are
    heavily correlated and the test treats them as independent. It returned
    q = 0.000 for ITB syndrome while that condition's own cross-fold CI ran
    [0.410, 0.986] -- a p-value and an interval cannot both be right, and the
    interval is the one this project has used throughout. Resampling subjects
    respects the clustering and agrees with the intervals.
    """
    rng = np.random.default_rng(seed)
    subs = np.unique(groups)
    idx = {s: np.flatnonzero(groups == s) for s in subs}
    from sklearn.metrics import roc_auc_score
    out = []
    for _ in range(n_boot):
        pick = rng.choice(subs, len(subs), replace=True)
        sel = np.concatenate([idx[s] for s in pick])
        if len(np.unique(y[sel])) < 2:
            continue
        out.append(roc_auc_score(y[sel], score[sel]))
    if len(out) < 100:
        return float("nan"), float("nan")
    return tuple(float(v) for v in np.percentile(out, [2.5, 97.5]))


def oof_scores(frame: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
               folds: list[dict]) -> np.ndarray:
    """Out-of-fold score per session, averaged over the seeds."""
    acc = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for f in folds:
        tr, te = f["train"], f["test"]
        model, ng = make_model("logit", list(frame.columns), [], "explicit",
                               PCA_PER_CHANNEL * 3)
        kw = {"clf__groups": groups[tr]} if ng else {}
        model.fit(frame.iloc[tr], y[tr], **kw)
        acc[te] += model.predict_proba(frame.iloc[te])[:, 1]
        cnt[te] += 1
    assert cnt.min() > 0, "a session was never held out"
    return acc / cnt


def stride_features(meta: pd.DataFrame, signs: dict) -> pd.DataFrame:
    """Within-session stride-distribution asymmetry, streamed one file at a time.

    CLAUDE.md forbids loading all session files at once. Each .mat holds
    (101, n_steps, 3) per channel; L and R stride counts differ, so the limbs are
    compared as DISTRIBUTIONS over strides rather than differenced pairwise.

    Per channel and per stance block: differences in location (mean, median),
    in spread (SD, IQR), and the Wasserstein distance between the two stride
    sets. Location and spread terms are signed so they can indicate WHICH limb;
    the Wasserstein term is unsigned magnitude only.
    """
    edges = np.linspace(0, 101, N_BLOCKS + 1).astype(int)
    rows, cols, missing = [], None, 0
    for n, (sub, fn) in enumerate(zip(meta["sub_id"], meta["filename"])):
        path = STEPS / f"{sub}__{Path(fn).stem}.mat"
        if not path.exists():
            rows.append(None)
            missing += 1
            continue
        m = sio.loadmat(path, squeeze_me=True)
        feat, names = [], []
        for j in JOINTS3:
            for p in range(3):
                s = signs[(j, p)]
                Lc = np.atleast_3d(m[f"ang_L_{j}"])[:, :, p]     # (101, nL)
                Rc = np.atleast_3d(m[f"ang_R_{j}"])[:, :, p]     # (101, nR)
                Lc = s * Lc
                for b in range(N_BLOCKS):
                    a, z = edges[b], edges[b + 1]
                    lv = np.nanmean(Lc[a:z, :], axis=0)
                    rv = np.nanmean(Rc[a:z, :], axis=0)
                    lv = lv[np.isfinite(lv)]
                    rv = rv[np.isfinite(rv)]
                    if len(lv) < 3 or len(rv) < 3:
                        feat += [np.nan] * 5
                    else:
                        feat += [
                            rv.mean() - lv.mean(),
                            np.median(rv) - np.median(lv),
                            rv.std() - lv.std(),
                            (np.subtract(*np.percentile(rv, [75, 25]))
                             - np.subtract(*np.percentile(lv, [75, 25]))),
                            wasserstein_distance(rv, lv),
                        ]
                    names += [f"st_{k}_{j}_p{p}_b{b}" for k in
                              ("dmean", "dmed", "dsd", "diqr", "wass")]
        cols = cols or names
        rows.append(feat)
        if (n + 1) % 200 == 0:
            print(f"    streamed {n + 1}/{len(meta)} sessions")
    if missing:
        print(f"    WARNING: {missing} stride files missing -- rows are NaN and "
              "will be median-imputed inside the fold")
    filled = [r if r is not None else [np.nan] * len(cols) for r in rows]
    return pd.DataFrame(filled, columns=cols)


def main() -> int:
    mean, pos, signs, meta, y, groups = load_limb_cohort()
    folds = make_folds(y, groups)
    check_no_group_leakage(folds, groups)
    print(f"{len(folds)} folds, groups=sub_id\n")

    results: dict = {}
    out: dict = {"n_sessions": int(len(meta)), "chance": float(y.mean())}
    out["neg_control_max_dev"] = negative_controls(meta, y, folds, groups,
                                                   results)
    _ = mirror_signs({"mean": mean}, pos)

    def score(name, frame, nch=None):
        f = frame.copy()
        f["label"] = y
        cols = [c for c in frame.columns]
        r = evaluate(name, "logit", f, cols, [], folds, groups,
                     pca_components=PCA_PER_CHANNEL * nch if nch else None)
        results[name] = r
        print("  " + fmt_result(r))
        return r

    # ---------------- reference ------------------------------------------
    sag_cols, sag_blocks = [], []
    for j in JOINTS3:
        s = signs[(j, SAGITTAL_PLANE)]
        L = mean[:, pos[f"ang_L_{j}_p{SAGITTAL_PLANE}"], :]
        R = mean[:, pos[f"ang_R_{j}_p{SAGITTAL_PLANE}"], :]
        sag_blocks.append(R - s * L)
        sag_cols += [f"d_{j}_t{t:03d}" for t in range(101)]
    sag = pd.DataFrame(np.concatenate(sag_blocks, axis=1), columns=sag_cols)
    print("=== reference ===")
    ref = score("limbsag_mean (phase 5 reference)", sag, 3)

    family: list[tuple[str, dict]] = []

    # ---------------- R1: velocity asymmetry ------------------------------
    print("\n=== R1: velocity asymmetry (24 channels phase 5 never read) ===")
    vnames = [f"{j}_p{p}" for j in VEL_JOINTS for p in range(3)]
    VL = np.stack([mean[:, pos[f"vel_L_{n}"], :].mean(axis=1) for n in vnames], 1)
    VR = np.stack([mean[:, pos[f"vel_R_{n}"], :].mean(axis=1) for n in vnames], 1)
    vsign = derive_signs(VL, VR, vnames, "velocity")
    vblocks, vcols = [], []
    for i, n in enumerate(vnames):
        s = vsign[i]
        vblocks.append(mean[:, pos[f"vel_R_{n}"], :]
                       - s * mean[:, pos[f"vel_L_{n}"], :])
        vcols += [f"dv_{n}_t{t:03d}" for t in range(101)]
    vel = pd.DataFrame(np.concatenate(vblocks, axis=1), columns=vcols)
    family.append(("limbvel (24 velocity channels)", score(
        "limbvel (24 velocity channels)", vel, 12)))
    both = pd.concat([sag, vel], axis=1)
    family.append(("limbsag + limbvel", score("limbsag + limbvel", both, 15)))

    # ---------------- R3a: stride distributions ---------------------------
    print("\n=== R3a: within-session stride distributions ===")
    print("  streaming waveforms_steps/*.mat one file at a time")
    stf = stride_features(meta, signs)
    print(f"  {stf.shape[1]} features "
          f"(9 channels x {N_BLOCKS} blocks x 5 statistics)")
    family.append(("stride distributions", score("stride distributions", stf)))
    family.append(("limbsag + stride distributions", score(
        "limbsag + stride distributions", pd.concat([sag, stf], axis=1))))

    # ---------------- deltas + BH over the secondary family ---------------
    print("\n=== all feature sets vs the phase 5 reference ===")
    print("  Decision rule is the project's established one: the PAIRED delta CI")
    print("  across the shared 25 folds excluding zero. See subject_bootstrap_ci")
    print("  for why a fold-level p-value was computed and then discarded.\n")
    print(f"  {'feature set':<34}{'AUC':>7}   CI               dAUC   dAUC CI"
          f"           excl 0")
    rows = []
    for name, r in family:
        d = paired_delta(r, ref)
        rows.append({"name": name, "auc_mean": r["auc_mean"],
                     "ci": list(r["auc_ci"]), "delta": d["delta_mean"],
                     "delta_ci": list(d["delta_ci"]),
                     "excludes_zero": d["excludes_zero"]})
        print(f"  {name:<34}{r['auc_mean']:>7.3f}   "
              f"[{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]   "
              f"{d['delta_mean']:+.3f}   [{d['delta_ci'][0]:+.3f}, "
              f"{d['delta_ci'][1]:+.3f}]   "
              f"{'YES' if d['excludes_zero'] else 'no'}")
    out["feature_sets"] = rows

    # ---------------- R2: per condition -----------------------------------
    print("\n=== R2: per condition (underpowered, hypothesis-generating) ===")
    cond = meta["SpecInjury"].astype(str).str.strip().str.lower()
    counts = cond.value_counts()
    keep = [c for c, n in counts.items()
            if n >= MIN_CONDITION_N and c not in ("pain", "other", "nan", "")]
    print(f"  conditions with n >= {MIN_CONDITION_N}, excluding the "
          f"uninformative 'pain' ({counts.get('pain', 0)}) and "
          f"'other' ({counts.get('other', 0)}) labels\n")
    crows = []
    for c in keep:
        m = (cond == c).to_numpy()
        ys, gs = y[m], groups[m]
        if len(np.unique(ys)) < 2:
            continue
        fs = make_folds(ys, gs)
        check_no_group_leakage(fs, gs)
        fr = sag[m].reset_index(drop=True)
        fr2 = fr.copy()
        fr2["label"] = ys
        r = evaluate(f"limbsag | {c}", "logit", fr2, list(fr.columns), [], fs,
                     gs, pca_components=PCA_PER_CHANNEL * 3)
        oof = oof_scores(fr, ys, gs, fs)
        blo, bhi = subject_bootstrap_ci(ys, oof, gs)
        crows.append({"condition": c, "n": int(m.sum()),
                      "n_subjects": int(len(np.unique(gs))),
                      "auc_mean": r["auc_mean"], "ci": list(r["auc_ci"]),
                      "boot_ci": [blo, bhi],
                      "survives": bool(r["auc_ci"][0] > 0.5 and blo > 0.5)})
    print(f"  {'condition':<32}{'n':>5}{'subj':>6}{'AUC':>8}   {'fold CI':<17}"
          f"{'subject-bootstrap CI':<22}survives")
    for row in sorted(crows, key=lambda r: -r["auc_mean"]):
        print(f"  {row['condition']:<32}{row['n']:>5}{row['n_subjects']:>6}"
              f"{row['auc_mean']:>8.3f}   "
              f"[{row['ci'][0]:.3f}, {row['ci'][1]:.3f}]    "
              f"[{row['boot_ci'][0]:.3f}, {row['boot_ci'][1]:.3f}]        "
              f"{'YES' if row['survives'] else 'no'}")
    n_surv = sum(r["survives"] for r in crows)
    print(f"\n  {n_surv} of {len(crows)} conditions survive both intervals.")
    out["per_condition"] = crows

    # ---------------- does the best set actually behave better? -----------
    # An AUC gain that does not improve REPEATABILITY does not address the
    # question that motivated this phase. Phase 4B measured the reference at
    # binary agreement 0.744 across 258 same-subject session pairs; that stands
    # until something beats it.
    print("\n=== repeatability of the best-scoring set vs the reference ===")
    print("  (phase 4B: reference = 0.744 agreement, score r +0.647)\n")
    from phase4b_operating_point import calibration, repeatability
    best = max(rows, key=lambda r: r["auc_mean"])
    frames = {"limbsag_mean (reference)": sag,
              best["name"]: {"limbvel (24 velocity channels)": vel,
                             "limbsag + limbvel": both,
                             "stride distributions": stf,
                             "limbsag + stride distributions":
                                 pd.concat([sag, stf], axis=1)}[best["name"]]}
    rep_out = {}
    for nm, fr in frames.items():
        oof = oof_scores(fr, y, groups, folds)
        rep = repeatability(meta, y, oof)
        cal = calibration(y, oof)
        rep_out[nm] = {"repeatability": rep, "calibration": cal}
        print(f"  {nm}")
        print(f"    binary agreement {rep['binary_agreement']:.3f} | "
              f"score r {rep['score_corr']:+.3f} | "
              f"{rep['n_session_pairs']} pairs from "
              f"{rep['n_subjects_same_side']} subjects")
        print(f"    calibration slope {cal['calibration_slope']:.3f} | "
              f"Brier {cal['brier']:.4f} vs {cal['brier_baseline']:.4f}")
    out["repeatability"] = rep_out

    out["results"] = {n: {"auc_mean": r["auc_mean"], "auc_ci": r["auc_ci"],
                          "auc_per_fold": r["auc"].tolist()}
                      for n, r in results.items()}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
