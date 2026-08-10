"""Phase 2 -- does the within-subject signal survive what a camera does to it?

Phase 5 found the project's only measurable kinematic signal: which limb is
injured, AUC 0.610 [0.532, 0.686], with every confound verified at chance, and no
loss from restricting to the three sagittal channels a side-on camera can see.

Two things a real camera does were never tested:

  1. TEMPORAL RESOLUTION. Stance is ~0.295 s, so 30 fps captures ~9 frames of
     stance against mocap's 101 normalized points -- an 11x reduction.
  2. FAR-LIMB OCCLUSION. The task is built on R - L differences, and a side-on
     camera sees one limb clearly and the other through the body.

The degradation is applied in ACQUISITION order, per limb, mimicking the real
chain: decimate to the frames a camera captures -> add pose noise at those
samples -> low-pass (the pipeline filters at 10 Hz) -> re-interpolate to 101
points. Only then are the limbs differenced.

HONEST LIMIT: baseline 0.610 against a 0.5 floor leaves ~0.11 of dynamic range
against fold CIs near +/-0.08. This reports an OPERATING-POINT VERDICT, not a
precise curve; most cells will have overlapping intervals.

Run: .venv/Scripts/python.exe scripts/phase2_degradation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_cohort import (  # noqa: E402
    CONTROL_CLEAN_CATEGORICAL,
    CONTROL_CLEAN_NUMERIC,
    STRUCTURE_FEATURES,
    build_cohort,
)
from phase1_eval import (  # noqa: E402
    check_no_group_leakage,
    evaluate,
    fmt_result,
    make_folds,
    paired_delta,
)
from phase5_limb import JOINTS3, SAGITTAL_PLANE, mirror_signs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
OUT = REPO / "results" / "phase2_degradation.json"

N_POINTS = 101
PCA_PER_CHANNEL = 3
LOWPASS_HZ = 10.0          # matches the pipeline's 4th-order Butterworth
N_REALIZATIONS = 2         # noise is stochastic; average and report the spread

# Frames of stance a camera captures, from the measured median stance of 0.295 s.
FPS_TO_POINTS = {"mocap": N_POINTS, "240fps": 71, "120fps": 35,
                 "60fps": 18, "30fps": 9}


def degrade(curves: np.ndarray, k: int, sigma: float, stance_s: np.ndarray,
            rng: np.random.Generator) -> np.ndarray:
    """Simulate acquiring a 101-point curve with k camera frames and pose noise.

    curves: (n_sessions, 101) for one channel of one limb.
    Returns the same shape, reconstructed from the degraded acquisition.
    """
    if k >= N_POINTS and sigma == 0.0:
        return curves.copy()

    n = curves.shape[0]
    src = np.linspace(0.0, 1.0, N_POINTS)
    dst = np.linspace(0.0, 1.0, k)
    out = np.empty_like(curves)

    for i in range(n):
        # 1. decimate to what the camera actually samples
        sampled = np.interp(dst, src, curves[i])
        # 2. pose estimation error at those samples
        if sigma > 0:
            sampled = sampled + rng.normal(0.0, sigma, size=k)
        # 3. low-pass at the acquired rate. Gaussian rather than Butterworth:
        #    filtfilt cannot pad a 9-sample signal, and the -3 dB point is set to
        #    match LOWPASS_HZ given this session's effective sampling rate.
        if sigma > 0 and k >= 5:
            fs = k / max(stance_s[i], 1e-3)
            sd_samples = fs / (2.0 * np.pi * LOWPASS_HZ)
            if sd_samples > 0.3:
                sampled = gaussian_filter1d(sampled, sd_samples, mode="nearest")
        # 4. reconstruct to 101 points (pchip, as gait_steps.m does)
        out[i] = PchipInterpolator(dst, sampled)(src) if k > 3 else \
            np.interp(src, dst, sampled)
    return out


def main() -> int:
    idx = pd.read_parquet(DERIVED / "waveforms_var_index.parquet")
    idx["sub_id"] = idx["sub_id"].astype(str)
    idx["filename"] = idx["filename"].astype(str)
    ch = (DERIVED / "waveform_channels.txt").read_text(encoding="utf-8").split("\n")
    pos = {c: i for i, c in enumerate(ch)}

    full_idx = pd.read_parquet(DERIVED / "waveforms_index.parquet")
    full_idx["sub_id"] = full_idx["sub_id"].astype(str)
    full_idx["filename"] = full_idx["filename"].astype(str)
    order = {k: i for i, k in enumerate(full_idx["sub_id"] + "|" + full_idx["filename"])}
    sel = np.array([order[k] for k in (idx["sub_id"] + "|" + idx["filename"])])
    mean = np.load(DERIVED / "waveforms_mean.npy")[sel]

    signs = mirror_signs({"mean": mean}, pos)

    cohort, _oa, _n = build_cohort()
    struct = pd.read_parquet(DERIVED / "session_structure.parquet")
    dvr = pd.read_parquet(DERIVED / "dvr_features.parquet")
    for f in (cohort, struct, dvr):
        f["sub_id"] = f["sub_id"].astype(str)
        f["filename"] = f["filename"].astype(str)

    meta = (idx.merge(cohort, on=["sub_id", "filename"], how="inner")
            .merge(struct, on=["sub_id", "filename"], how="inner")
            .merge(dvr[["sub_id", "filename", "left_STANCE_TIME",
                        "right_STANCE_TIME"]], on=["sub_id", "filename"],
                   how="inner"))
    keep = ((meta["label"] == 1) & meta["InjSide"].isin(["Right", "Left"])).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    mean = mean[keep]

    stance = meta[["left_STANCE_TIME", "right_STANCE_TIME"]].replace(0, np.nan) \
        .mean(axis=1).fillna(0.295).to_numpy()
    y = (meta["InjSide"] == "Right").astype(int).to_numpy()
    groups = meta["sub_id"].to_numpy()
    print(f"cohort: {len(meta)} unilateral sessions / {meta['sub_id'].nunique()} "
          f"subjects, Right-injured rate {y.mean():.3f}")
    print(f"stance time: median {np.median(stance):.3f} s\n")

    # Far/near limb assignment -- fixed seed, INDEPENDENT of injury side.
    rng0 = np.random.default_rng(20260810)
    far_is_right = rng0.random(len(meta)) < 0.5
    ct = pd.crosstab(far_is_right, y, normalize="index")
    dev = float(abs(ct[1] - y.mean()).max())
    print(f"far/near assignment vs injured side: max deviation {dev:.3f} "
          f"-> {'independent OK' if dev < 0.06 else 'NOT INDEPENDENT - stop'}")
    assert dev < 0.06, "far-limb assignment correlates with injured side"

    folds = make_folds(y, groups)
    check_no_group_leakage(folds, groups)
    print(f"{len(folds)} folds, groups=sub_id\n")

    # --- conditions --------------------------------------------------------
    conditions = [(f"fps {name} (k={k}), no noise", k, 0.0, 0.0)
                  for name, k in FPS_TO_POINTS.items()]
    conditions += [("101 pts, sigma near/far 2/2", N_POINTS, 2.0, 2.0),
                   ("101 pts, sigma near/far 2/4", N_POINTS, 2.0, 4.0),
                   ("101 pts, sigma near/far 2/8", N_POINTS, 2.0, 8.0),
                   ("120fps + sigma 2/2", 35, 2.0, 2.0),
                   ("60fps + sigma 2/4", 18, 2.0, 4.0),
                   ("30fps + sigma 2/8", 9, 2.0, 8.0)]
    # Extended band. Published monocular knee-flexion MAE against marker-based
    # mocap is 14.1-25.8 deg for generic models on clinical gait (Sci Rep 2025),
    # well beyond the 8 deg ceiling above. Sport-fine-tuned checkpoints should do
    # much better, but phase 3 must be readable against the surface wherever it
    # lands, so the sweep is extended to cover the published band.
    conditions += [("101 pts, sigma near/far 2/12", N_POINTS, 2.0, 12.0),
                   ("101 pts, sigma near/far 2/16", N_POINTS, 2.0, 16.0),
                   ("101 pts, sigma near/far 2/20", N_POINTS, 2.0, 20.0),
                   ("101 pts, sigma both 8/8", N_POINTS, 8.0, 8.0),
                   ("101 pts, sigma both 15/15", N_POINTS, 15.0, 15.0),
                   ("30fps + sigma 2/16", 9, 2.0, 16.0),
                   ("30fps + sigma 8/8", 9, 8.0, 8.0),
                   ("30fps + sigma 15/15", 9, 15.0, 15.0)]

    def build(k: int, s_near: float, s_far: float, seed: int,
              joints: list[str], planes: list[int]) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        cols, blocks = [], []
        for j in joints:
            for p in planes:
                L = mean[:, pos[f"ang_L_{j}_p{p}"], :]
                R = mean[:, pos[f"ang_R_{j}_p{p}"], :]
                sig_R = np.where(far_is_right, s_far, s_near)
                sig_L = np.where(far_is_right, s_near, s_far)
                # Sigma differs per session (far vs near limb), so degrade each
                # distinct sigma group in one pass.
                Rd = np.empty_like(R)
                Ld = np.empty_like(L)
                for uniq in np.unique(sig_R):
                    m = sig_R == uniq
                    Rd[m] = degrade(R[m], k, float(uniq), stance[m], rng)
                for uniq in np.unique(sig_L):
                    m = sig_L == uniq
                    Ld[m] = degrade(L[m], k, float(uniq), stance[m], rng)
                d = Rd - signs[(j, p)] * Ld
                blocks.append(d)
                cols += [f"d_{j}_p{p}_t{t:03d}" for t in range(N_POINTS)]
        return pd.DataFrame(np.concatenate(blocks, axis=1), columns=cols)

    results: dict = {}

    def score(tag: str, frame: pd.DataFrame, feats: list[str], ncf: int) -> dict:
        f = frame.copy()
        f["label"] = y
        r = evaluate(tag, "logit", f, feats, [], folds, groups,
                     pca_components=PCA_PER_CHANNEL * ncf)
        results[tag] = r
        print("  " + fmt_result(r))
        return r

    # --- negative controls, re-asserted at baseline ------------------------
    print("=== negative controls (must be ~0.5) ===")
    base = meta.copy()
    base["label"] = y
    nc = {}
    for name, num, cat in (("NEG demographics", CONTROL_CLEAN_NUMERIC,
                            CONTROL_CLEAN_CATEGORICAL),
                           ("NEG provenance", ["year", "yrs_missing",
                                               "lvl_missing"], []),
                           ("NEG structure", STRUCTURE_FEATURES, [])):
        r = evaluate(name, "logit", base, num, cat, folds, groups)
        nc[name] = r
        results[name] = r
        print("  " + fmt_result(r))
    worst = max(abs(r["auc_mean"] - 0.5) for r in nc.values())
    print(f"  gate: max |AUC-0.5| = {worst:.3f} -> "
          f"{'PASS' if worst < 0.08 else 'FAIL'}\n")

    # --- sagittal (video-recoverable) across conditions ---------------------
    print("=== limbsag (3 sagittal channels) across degradation ===")
    summary = []
    for label, k, s_near, s_far in conditions:
        aucs, tags = [], []
        reps = 1 if (s_near == 0 and s_far == 0) else N_REALIZATIONS
        for rep in range(reps):
            frame = build(k, s_near, s_far, 900 + rep, JOINTS3, [SAGITTAL_PLANE])
            feats = list(frame.columns)
            r = score(f"limbsag | {label} | rep{rep}", frame, feats, 3)
            aucs.append(r["auc_mean"])
            tags.append(f"limbsag | {label} | rep{rep}")
        summary.append({"condition": label, "k": k, "sigma_near": s_near,
                        "sigma_far": s_far, "n_reps": reps,
                        "auc_mean": float(np.mean(aucs)),
                        "auc_spread": float(np.ptp(aucs)) if reps > 1 else 0.0,
                        "tags": tags})

    # --- limb9 reference at three points ------------------------------------
    print("\n=== limb9 (9 channels) reference points ===")
    for label, k, s_near, s_far in [("mocap", N_POINTS, 0.0, 0.0),
                                    ("60fps no noise", 18, 0.0, 0.0),
                                    ("60fps + sigma 2/4", 18, 2.0, 4.0)]:
        frame = build(k, s_near, s_far, 900, JOINTS3, [0, 1, 2])
        score(f"limb9 | {label}", frame, list(frame.columns), 9)

    # --- verdict ------------------------------------------------------------
    print("\n=== OPERATING-POINT VERDICT ===")
    baseline = next(s for s in summary if s["condition"].startswith("fps mocap"))
    print(f"  baseline (mocap, no noise): AUC {baseline['auc_mean']:.3f}")
    ref = results[baseline["tags"][0]]
    print(f"  {'condition':<34}{'AUC':>7}{'spread':>8}   CI            "
          f"CI>0.5  dAUC vs mocap")
    for s in summary:
        r0 = results[s["tags"][0]]
        lo, hi = r0["auc_ci"]
        d = paired_delta(r0, ref)
        print(f"  {s['condition']:<34}{s['auc_mean']:>7.3f}{s['auc_spread']:>8.3f}"
              f"   [{lo:.3f}, {hi:.3f}]  {'yes' if lo > 0.5 else 'no ':<6}"
              f"  {d['delta_mean']:+.3f}")
        s["ci"] = [lo, hi]
        s["survives"] = bool(lo > 0.5)
        s["delta_vs_mocap"] = d["delta_mean"]

    payload = {"n_sessions": int(len(meta)),
               "n_subjects": int(meta["sub_id"].nunique()),
               "chance": float(y.mean()),
               "far_near_independence_dev": dev,
               "negative_control_max_dev": float(worst),
               "summary": summary,
               "results": {n: {"auc_mean": r["auc_mean"], "auc_ci": r["auc_ci"],
                               "ap_mean": r["ap_mean"],
                               "auc_per_fold": r["auc"].tolist()}
                           for n, r in results.items()}}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
