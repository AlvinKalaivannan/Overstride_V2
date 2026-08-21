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

RESIDUALS ARE DRAWN AS CLIP PAIRS, AND VIEWPOINT IS A CONDITION
  Phase 3B established two things this phase depends on. First, AthleticsPose is
  NOT near-frontal -- that claim came from comparing 44 PIXELS against a 200-250
  mm reference. Half the held-out clips are near-lateral, median 48.7 deg out of
  the image plane. Second, the far-limb penalty does not grow with how side-on
  the view is (+0.07 deg extrapolated to a fully lateral view).

  So two things change here:
    - near and far residuals are drawn as a PAIR FROM THE SAME CLIP, preserving
      the common-mode error a single camera induces on both limbs. Independent
      draws destroy that, and this feature is a LEFT-RIGHT DIFFERENCE, so
      common-mode error is exactly what cancels. Independent draws overstate the
      damage.
    - a LATERAL-ONLY bank (view ratio >= 0.85) is run as its own condition. That
      is the geometry Overstride actually prescribes -- a side-on camera -- and
      phase 3B found it is also where the pose estimator is most accurate.

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

from phase1_cohort import (CONTROL_CLEAN_CATEGORICAL,  # noqa: E402
                           CONTROL_CLEAN_NUMERIC, PROVENANCE_FEATURES,
                           STRUCTURE_FEATURES, build_cohort)
from phase1_eval import (check_no_group_leakage, ci, evaluate,  # noqa: E402
                         fmt_result, make_folds, paired_delta)
from phase3b_viewpoint import clip_geometry  # noqa: E402
from phase5_limb import JOINTS3, SAGITTAL_PLANE, mirror_signs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
OUT = REPO / "results" / "phase4_real_delta.json"

N_POINTS = 101
PCA_PER_CHANNEL = 3
N_REALIZATIONS = 3
FPS_POINTS = {"mocap": 101, "60fps": 18, "30fps": 9}
LATERAL_RATIO = 0.85     # phase 3B: >= this is the near-lateral half of the data


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


# Ferber sagittal plane is index 2 (verified in phase 1C). The error bank
# supplies hip and knee only.
JOINT_TO_BANK = {"hip": 0, "knee": 1}


def load_bank(verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual curves, plus near/far index pairs and each pair's view ratio.

    Each source clip contributes exactly one near row and one far row. Pairing
    them preserves the common-mode error a single camera puts on both limbs,
    which is precisely what cancels in an R - L difference.
    """
    bank = np.load(DERIVED / "phase4_error_bank.npz", allow_pickle=True)
    B, BL, BC = bank["curves"], bank["limb"], bank["clip"]
    geo = clip_geometry()
    ratio_of = {f"{a}/{s}/{c}": g["view_ratio"] for (a, s, c), g in geo.items()}
    by_clip: dict[str, dict[str, int]] = {}
    for i, (clip, limb) in enumerate(zip(BC, BL)):
        by_clip.setdefault(str(clip), {})[str(limb)] = i
    keep = [(c, d) for c, d in by_clip.items() if "near" in d and "far" in d]
    pairs = np.array([[d["near"], d["far"]] for _c, d in keep])
    pair_ratio = np.array([ratio_of.get(c, np.nan) for c, _d in keep])
    if verbose:
        print(f"error bank: {B.shape}, near={int((BL=='near').sum())} "
              f"far={int((BL=='far').sum())}, |err| {np.abs(B).mean():.2f} deg")
        lat = pairs[pair_ratio >= LATERAL_RATIO]
        print(f"paired clips: {len(pairs)} (near+far from the same capture)")
        print(f"  view ratio: median {np.nanmedian(pair_ratio):.3f}, "
              f"near-lateral (>= {LATERAL_RATIO}) {len(lat)} clips")
        print(f"  |err| all {np.abs(B[pairs.ravel()]).mean():.2f} deg | "
              f"lateral-only {np.abs(B[lat.ravel()]).mean():.2f} deg")
    return B, pairs, pair_ratio


def build_features(*, mean: np.ndarray, pos: dict, signs: dict,
                   joints: list[str], k: int, seed: int, perturb: bool,
                   pool: np.ndarray, B: np.ndarray, far_is_right: np.ndarray,
                   scale: float = 1.0, realized: dict | None = None,
                   tag: str = "") -> pd.DataFrame:
    """Degrade in ACQUISITION ORDER, as phase 2 does.

    The camera samples k frames of stance, the pose estimator makes its error AT
    THOSE FRAMES, and only then is the curve reconstructed to 101 points. Adding
    the residual at 101 points and decimating afterwards would let decimation
    low-pass away error a real 30 fps camera actually delivers -- it understates
    the very condition this phase exists to test.

    scale multiplies the residual magnitude. It exists for phase 4B's bound on
    the Cardan / three-point angle convention transfer, and is 1.0 everywhere
    else.
    """
    rng = np.random.default_rng(seed)
    cols, blocks, errs = [], [], []
    # One clip drawn per SESSION, shared by both limbs and both joints, so the
    # near/far pairing and the hip/knee correlation both survive.
    draw = rng.integers(0, len(pool), len(mean)) if perturb else None
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
                    # Column 0 of the pair is the near row, column 1 the far
                    # row, from one capture. Resampled to the k frames the
                    # camera actually took.
                    far = far_is_right[i] if side == "R" else not far_is_right[i]
                    row = pool[draw[i]][1 if far else 0]
                    sampled[i] += scale * to_k(B[row, bi][None], k)[0]
            got[side] = to_101(sampled, k)
            errs.append(np.abs(got[side] - clean[side]).mean())
        blocks.append(got["R"] - s * got["L"])
        cols += [f"d_{j}_t{t:03d}" for t in range(N_POINTS)]
    if tag and realized is not None:
        realized[tag] = float(np.mean(errs))
    return pd.DataFrame(np.concatenate(blocks, axis=1), columns=cols)


def load_limb_cohort() -> tuple:
    """The 818 unilateral-injury sessions, identical to phases 2 and 5."""
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
    struct = pd.read_parquet(DERIVED / "session_structure.parquet")
    for f in (cohort, struct):
        f["sub_id"] = f["sub_id"].astype(str)
        f["filename"] = f["filename"].astype(str)
    meta = (idx.merge(cohort, on=["sub_id", "filename"], how="inner")
            .merge(struct, on=["sub_id", "filename"], how="inner"))
    keep = ((meta["label"] == 1)
            & meta["InjSide"].isin(["Right", "Left"])).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    mean = mean[keep]
    y = (meta["InjSide"] == "Right").astype(int).to_numpy()
    groups = meta["sub_id"].to_numpy()
    print(f"cohort: {len(meta)} unilateral sessions / {meta['sub_id'].nunique()} "
          f"subjects, chance {y.mean():.3f}")
    return mean, pos, signs, meta, y, groups


def main() -> int:
    B, pairs, pair_ratio = load_bank()
    BANKS = {"all views": pairs,
             "lateral only": pairs[pair_ratio >= LATERAL_RATIO]}
    mean, pos, signs, meta, y, groups = load_limb_cohort()

    folds = make_folds(y, groups)
    check_no_group_leakage(folds, groups)

    rng0 = np.random.default_rng(20260810)
    far_is_right = rng0.random(len(meta)) < 0.5

    realized: dict[str, float] = {}
    common = dict(mean=mean, pos=pos, signs=signs, B=B,
                  far_is_right=far_is_right)

    def build(k: int, seed: int, joints: list[str], perturb: bool,
              tag: str = "", pool: np.ndarray | None = None) -> pd.DataFrame:
        return build_features(joints=joints, k=k, seed=seed, perturb=perturb,
                              pool=pairs if pool is None else pool,
                              realized=realized, tag=tag, **common)

    results: dict = {}

    def score(tag: str, frame: pd.DataFrame, nch: int) -> dict:
        f = frame.copy()
        f["label"] = y
        r = evaluate(tag, "logit", f, list(frame.columns), [], folds, groups,
                     pca_components=PCA_PER_CHANNEL * nch)
        results[tag] = r
        print("  " + fmt_result(r))
        return r

    # --- negative controls, re-asserted rather than inherited ---------------
    # CLAUDE.md requires a demographics-only control and a provenance-only
    # baseline alongside every kinematic score. On this within-subject task all
    # three are constant within a session and so MUST sit at chance; any
    # departure would mean the setup leaks and every number below is void.
    print("\n=== negative controls (must be ~0.5) ===")
    base = meta.copy()
    base["label"] = y
    nc = {}
    for name, num, cat in (("NEG demographics", CONTROL_CLEAN_NUMERIC,
                            CONTROL_CLEAN_CATEGORICAL),
                           ("NEG provenance", PROVENANCE_FEATURES, []),
                           ("NEG structure", STRUCTURE_FEATURES, [])):
        missing = [c for c in num + cat if c not in base.columns]
        if missing:
            raise SystemExit(f"{name}: missing columns {missing} -- cannot "
                             "assert the control, so not proceeding")
        r = evaluate(name, "logit", base, num, cat, folds, groups)
        nc[name] = r
        results[name] = r
        print("  " + fmt_result(r))
    worst = max(abs(r["auc_mean"] - 0.5) for r in nc.values())
    print(f"  gate: max |AUC-0.5| = {worst:.3f} -> "
          f"{'PASS' if worst < 0.08 else 'FAIL'}")
    assert worst < 0.08, "negative control is not at chance -- setup leaks"

    print("\n=== baselines: clean mocap ===")
    ref3 = score("wave3 clean (mocap)", build(101, 0, JOINTS3, False), 3)
    ref2 = score("wave2 clean (hip+knee)", build(101, 0, ["knee", "hip"], False), 2)

    print("\n=== with MEASURED monocular error ===")
    summary = []
    for bank_name, pool in BANKS.items():
      for fps, k in FPS_POINTS.items():
        for name, joints, nch, ref in (("wave3_hk", JOINTS3, 3, ref3),
                                       ("wave2", ["knee", "hip"], 2, ref2)):
            tag = f"{name}|{fps}|{bank_name}"
            reps = [score(f"{name} + {bank_name} @ {fps} | rep{rep}",
                          build(k, 700 + rep, joints, True, tag, pool), nch)
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
            summary.append({"features": name, "fps": fps, "bank": bank_name,
                            "auc_mean": float(per_fold.mean()),
                            "auc_spread": float(np.ptp([r["auc_mean"] for r in reps])),
                            "ci": [lo, hi],
                            "survives": bool(lo > 0.5),
                            "delta_vs_clean": d["delta_mean"],
                            "delta_ci": list(d["delta_ci"]),
                            "delta_excludes_zero": d["excludes_zero"],
                            "realized_error_deg": realized[tag]})

    print("\n=== PHASE 4 VERDICT: real delta AUC ===")
    print(f"  clean mocap wave3 {ref3['auc_mean']:.3f}   wave2 {ref2['auc_mean']:.3f}")
    print(f"  {'features':<10}{'bank':>14}{'fps':>7}{'err':>7}{'AUC':>8}"
          f"{'spread':>8}   CI              CI>0.5  AUC>=.60   dAUC  dAUC CI")
    for s in summary:
        lo, hi = s["ci"]
        dlo, dhi = s["delta_ci"]
        # The phase 5 bar was BOTH legs: mean AUC >= 0.60 and CI excluding 0.5.
        print(f"  {s['features']:<10}{s['bank']:>14}{s['fps']:>7}"
              f"{s['realized_error_deg']:>6.2f}d"
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
