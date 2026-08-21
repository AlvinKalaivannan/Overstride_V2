"""Phase 4 -- the injured-limb signal under MEASURED monocular error.

WHAT PHASE 4 CAN AND CANNOT BE, stated plainly
  CLAUDE.md's phase 4 is "video-derived features through the phase 2 model". Run
  literally that is impossible with these datasets: the Ferber archive is
  marker-based mocap with NO video, and AthleticsPose has video-derived poses but
  no injury labels. No subject exists with both.

  What is possible, and is a real advance on phase 2: phase 2 degraded the Ferber
  waveforms with ASSUMED Gaussian noise. Phase 3 MEASURED what monocular pose
  actually does. This phase injects the measured residual curves -- real
  magnitude, real temporal structure, real systematic bias -- and reports the
  resulting delta AUC.

  So the noise is video-derived and empirical rather than invented. It is not an
  end-to-end video pipeline, and the report must not imply otherwise.

ACQUISITION ORDER MATTERS
  The camera samples k frames of stance, the pose estimator errs AT THOSE FRAMES,
  and only then is the curve reconstructed to 101 points. An earlier version
  added the residual at 101 points and decimated afterwards, which let
  decimation low-pass away error a real 30 fps camera actually delivers -- it
  flattered the exact condition this phase exists to test. Each condition now
  also reports its REALIZED mean |degraded - clean|, so the framerate rows can be
  compared on measured error rather than on a nominal setting.

TWO VIDEO-CONSTRAINED FEATURE SETS
  Phase 3 can only supply hip and knee: the H36M-17 skeleton has no toe keypoint,
  so ankle dorsiflexion is unavailable from monocular pose.
    wave3_hk  -- ankle kept clean, hip/knee perturbed. OPTIMISTIC: it credits the
                 model with an ankle channel a camera cannot actually deliver.
    wave2     -- ankle dropped entirely. The honest video-recoverable set.

Run: .venv/Scripts/python.exe scripts/phase4_real_delta.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_cohort import build_cohort  # noqa: E402
from phase1_eval import (check_no_group_leakage, ci, evaluate,  # noqa: E402
                         fmt_result, make_folds, paired_delta)
from phase5_limb import JOINTS3, SAGITTAL_PLANE, mirror_signs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
OUT = REPO / "results" / "phase4_real_delta.json"

N_POINTS = 101
PCA_PER_CHANNEL = 3
N_REALIZATIONS = 3
FPS_POINTS = {"mocap": 101, "60fps": 18, "30fps": 9}


def to_k(curve: np.ndarray, k: int) -> np.ndarray:
    """(n, 101) -> (n, k): the samples a camera at this rate actually captures."""
    if k >= N_POINTS:
        return curve
    src = np.linspace(0.0, 1.0, N_POINTS)
    dst = np.linspace(0.0, 1.0, k)
    return np.stack([np.interp(dst, src, c) for c in curve])


def to_101(sampled: np.ndarray, k: int) -> np.ndarray:
    """(n, k) -> (n, 101): reconstruct the time-normalised curve."""
    if k >= N_POINTS:
        return sampled
    src = np.linspace(0.0, 1.0, N_POINTS)
    dst = np.linspace(0.0, 1.0, k)
    return np.stack([np.interp(src, dst, c) for c in sampled])


def main() -> int:
    bank = np.load(DERIVED / "phase4_error_bank.npz", allow_pickle=True)
    B, BL = bank["curves"], bank["limb"]     # (n, 2, 101) hip/knee, near|far
    print(f"error bank: {B.shape}, near={int((BL=='near').sum())} "
          f"far={int((BL=='far').sum())}, |err| {np.abs(B).mean():.2f} deg")

    idx = pd.read_parquet(DERIVED / "waveforms_var_index.parquet")
    idx["sub_id"] = idx["sub_id"].astype(str)
    idx["filename"] = idx["filename"].astype(str)
    ch = (DERIVED / "waveform_channels.txt").read_text(encoding="utf-8").split("\n")
    pos = {c: i for i, c in enumerate(ch)}

    full = pd.read_parquet(DERIVED / "waveforms_index.parquet")
    full["sub_id"] = full["sub_id"].astype(str)
    full["filename"] = full["filename"].astype(str)
    order = {k: i for i, k in enumerate(full["sub_id"] + "|" + full["filename"])}
    sel = np.array([order[k] for k in (idx["sub_id"] + "|" + idx["filename"])])
    mean = np.load(DERIVED / "waveforms_mean.npy")[sel]

    signs = mirror_signs({"mean": mean}, pos)

    cohort, _oa, _n = build_cohort()
    cohort["sub_id"] = cohort["sub_id"].astype(str)
    cohort["filename"] = cohort["filename"].astype(str)
    meta = idx.merge(cohort, on=["sub_id", "filename"], how="inner")
    keep = ((meta["label"] == 1)
            & meta["InjSide"].isin(["Right", "Left"])).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    mean = mean[keep]
    y = (meta["InjSide"] == "Right").astype(int).to_numpy()
    groups = meta["sub_id"].to_numpy()
    print(f"cohort: {len(meta)} unilateral sessions / {meta['sub_id'].nunique()} "
          f"subjects, chance {y.mean():.3f}")

    folds = make_folds(y, groups)
    check_no_group_leakage(folds, groups)

    rng0 = np.random.default_rng(20260810)
    far_is_right = rng0.random(len(meta)) < 0.5

    # Ferber sagittal plane is index 2 (verified in phase 1C). The error bank
    # supplies hip and knee only.
    JOINT_TO_BANK = {"hip": 0, "knee": 1}

    near_pool = np.flatnonzero(BL == "near")
    far_pool = np.flatnonzero(BL == "far")
    realized: dict[str, float] = {}

    def build(k: int, seed: int, joints: list[str], perturb: bool,
              tag: str = "") -> pd.DataFrame:
        """Degrade in ACQUISITION ORDER, as phase 2 does.

        The camera samples k frames of stance, the pose estimator makes its error
        AT THOSE FRAMES, and only then is the curve reconstructed to 101 points.
        Adding the residual at 101 points and decimating afterwards would let
        decimation low-pass away error a real 30 fps camera actually delivers --
        it understates the very condition this phase exists to test.
        """
        rng = np.random.default_rng(seed)
        cols, blocks, errs = [], [], []
        for j in joints:
            s = signs[(j, SAGITTAL_PLANE)]
            clean = {side: mean[:, pos[f"ang_{side}_{j}_p{SAGITTAL_PLANE}"], :]
                     for side in ("L", "R")}
            got = {}
            for side in ("L", "R"):
                sampled = to_k(clean[side], k).copy()
                if perturb and j in JOINT_TO_BANK:
                    bi = JOINT_TO_BANK[j]
                    for i in range(len(sampled)):
                        # Independent residual per limb, matched to whether that
                        # limb is the near or the far one for this session, and
                        # resampled to the k frames the camera captured.
                        far = far_is_right[i] if side == "R" else not far_is_right[i]
                        pool = far_pool if far else near_pool
                        sampled[i] += to_k(B[rng.choice(pool), bi][None], k)[0]
                got[side] = to_101(sampled, k)
                errs.append(np.abs(got[side] - clean[side]).mean())
            blocks.append(got["R"] - s * got["L"])
            cols += [f"d_{j}_t{t:03d}" for t in range(N_POINTS)]
        if tag:
            realized[tag] = float(np.mean(errs))
        return pd.DataFrame(np.concatenate(blocks, axis=1), columns=cols)

    results: dict = {}

    def score(tag: str, frame: pd.DataFrame, nch: int) -> dict:
        f = frame.copy()
        f["label"] = y
        r = evaluate(tag, "logit", f, list(frame.columns), [], folds, groups,
                     pca_components=PCA_PER_CHANNEL * nch)
        results[tag] = r
        print("  " + fmt_result(r))
        return r

    print("\n=== baselines: clean mocap ===")
    ref3 = score("wave3 clean (mocap)", build(101, 0, JOINTS3, False), 3)
    ref2 = score("wave2 clean (hip+knee)", build(101, 0, ["knee", "hip"], False), 2)

    print("\n=== with MEASURED monocular error ===")
    summary = []
    for fps, k in FPS_POINTS.items():
        for name, joints, nch, ref in (("wave3_hk", JOINTS3, 3, ref3),
                                       ("wave2", ["knee", "hip"], 2, ref2)):
            reps = [score(f"{name} + real error @ {fps} | rep{rep}",
                          build(k, 700 + rep, joints, True, f"{name}|{fps}"), nch)
                    for rep in range(N_REALIZATIONS)]
            # Every realization is scored on the SAME 25 folds, so averaging
            # fold-by-fold gives 25 realization-averaged fold scores. The CI and
            # the paired delta both come from those, so the AUC, the CI and the
            # dAUC in one row all describe the same quantity. Reporting a mean
            # AUC beside a single realization's CI would not reconcile.
            per_fold = np.mean([r["auc"] for r in reps], axis=0)
            agg = {"name": f"{name} @ {fps}", "auc": per_fold}
            lo, hi = ci(per_fold)
            d = paired_delta(agg, ref)
            summary.append({"features": name, "fps": fps,
                            "auc_mean": float(per_fold.mean()),
                            "auc_spread": float(np.ptp([r["auc_mean"] for r in reps])),
                            "ci": [lo, hi],
                            "survives": bool(lo > 0.5),
                            "delta_vs_clean": d["delta_mean"],
                            "delta_ci": list(d["delta_ci"]),
                            "delta_excludes_zero": d["excludes_zero"],
                            "realized_error_deg": realized[f"{name}|{fps}"]})

    print("\n=== PHASE 4 VERDICT: real delta AUC ===")
    print(f"  clean mocap wave3 {ref3['auc_mean']:.3f}   wave2 {ref2['auc_mean']:.3f}")
    print(f"  {'features':<10}{'fps':>8}{'err':>7}{'AUC':>8}{'spread':>8}   CI  "
          f"            CI>0.5  AUC>=.60   dAUC  dAUC CI")
    for s in summary:
        lo, hi = s["ci"]
        dlo, dhi = s["delta_ci"]
        # The phase 5 bar was BOTH legs: mean AUC >= 0.60 and CI excluding 0.5.
        print(f"  {s['features']:<10}{s['fps']:>8}{s['realized_error_deg']:>6.2f}d"
              f"{s['auc_mean']:>8.3f}"
              f"{s['auc_spread']:>8.3f}   [{lo:.3f}, {hi:.3f}]  "
              f"{'yes' if s['survives'] else 'NO ':>6}  "
              f"{'yes' if s['auc_mean'] >= 0.60 else 'NO ':>8}  "
              f"{s['delta_vs_clean']:+.3f}  [{dlo:+.3f}, {dhi:+.3f}]"
              f"{'  *' if s['delta_excludes_zero'] else ''}")

    OUT.write_text(json.dumps({
        "n_sessions": int(len(meta)), "chance": float(y.mean()),
        "bank_mae_deg": float(np.abs(B).mean()),
        "realized_error_deg": realized,
        "clean": {"wave3": ref3["auc_mean"], "wave2": ref2["auc_mean"]},
        "summary": summary,
        "results": {n: {"auc_mean": r["auc_mean"], "auc_ci": r["auc_ci"],
                        "auc_per_fold": r["auc"].tolist()}
                    for n, r in results.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
