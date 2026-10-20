"""Phase 13 arm 1 -- does filtering the 2D before lifting help?

Pre-registration: `results/phase13.md`, frozen 2026-10-19, including the stated
expectation that this is small: 13.21 px of the 16.34 px 2D error RMS is a
CONSTANT bias no filter touches, and a perfect low-pass removes 1.39 px.

It runs anyway for two reasons. It is one pass per cutoff. And it settles a
discrepancy nobody had noticed: `scripts/video_kinematics.py:smooth` low-passes
the 2D at 10 Hz before lifting, while the 3.509 deg baseline those angles are
compared against is measured on UNFILTERED released detections. The tool has
been filtering all along and it was never validated.

Everything is paired: every cutoff is scored on the same clips as the unfiltered
baseline, so the comparison is a per-subject difference rather than two
independent means. At three held-out subjects that distinction is the difference
between a usable interval and a useless one.

Near and far limb are reported separately, never pooled.

Run: .venv/Scripts/python.exe scripts/phase13_filter.py [--clips N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AP = REPO / "data" / "athleticspose"
sys.path.insert(0, str(AP / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import DATA, load_model  # noqa: E402
from phase7_inference_fix import CKPT_FOR, clip_list, lift  # noqa: E402
from phase12_scan import subject_bootstrap_ci  # noqa: E402

OUT = REPO / "results" / "phase13_filter.json"

DET = "coco"                 # the deployable detector
PAD, WINDOWS = "edge", "overlap"
FS = 100.0                   # AthleticsPose capture rate, Timebase.Frequency
CUTOFFS = [6.0, 10.0, 15.0]  # pre-registered

# A fourth arm, ADDED after the smoke test and labelled as such, because the
# pre-registration raised the question and the three cutoffs above do not answer
# it: what does the filter the TOOL applies actually cost? `video_kinematics.py`
# low-passes at 10 Hz, but on ~30 fps footage, where sigma = fs / (2 pi f_c)
# = 30 / (2 pi * 10) = 0.48 frames -- a far gentler filter than 10 Hz at this
# dataset's 100 Hz. Sigma in frames is the dimensionless quantity, so the tool's
# setting is reproducible here by sigma rather than by cutoff.
TOOL_SIGMA_AT_30FPS = 0.48
JOINTS = ("hip", "knee")
LIMBS = ("near", "far")


def smooth2d_sigma(kp: np.ndarray, sigma: float) -> np.ndarray:
    """The same filter, parametrised by sigma in FRAMES rather than by cutoff.

    Sigma is what the filter actually does; a cutoff in Hz only means something
    once the sampling rate is fixed. 10 Hz is sigma 1.59 frames at this
    dataset's 100 Hz and sigma 0.48 at a phone's 30 fps -- the same line of code,
    three times the smoothing.
    """
    from scipy.ndimage import gaussian_filter1d

    if sigma <= 0.0:
        return kp
    out = kp.copy()
    out[:, :, :2] = gaussian_filter1d(kp[:, :, :2], sigma, axis=0, mode="nearest")
    return out


def smooth2d(kp: np.ndarray, cutoff: float, fs: float = FS) -> np.ndarray:
    """Zero-phase Gaussian low-pass on the x,y tracks only.

    Identical in form to `video_kinematics.smooth`, which is the point: this
    measures the filter the tool already applies. The confidence channel is left
    alone -- it is not a trajectory, and the lifter reads it as a per-frame
    weight.

    NON-CAUSAL: it uses future frames, so it cannot run in a stream without a
    lag of roughly 3 sigma. Recorded in the pre-registration.
    """
    from scipy.ndimage import gaussian_filter1d

    sd = fs / (2.0 * np.pi * cutoff)
    out = kp.copy()
    out[:, :, :2] = gaussian_filter1d(kp[:, :, :2], sd, axis=0, mode="nearest")
    return out


def paired(base: dict[str, list[float]], arm: dict[str, list[float]]) -> dict:
    """Per-subject mean difference, arm minus base. Negative is better."""
    per = {}
    for s in sorted(base):
        b, a = np.asarray(base[s]), np.asarray(arm[s])
        n = min(len(b), len(a))
        per[s] = float(np.mean(a[:n] - b[:n]))
    vals = list(per.values())
    lo, hi = subject_bootstrap_ci(vals)
    return {"delta_deg": float(np.mean(vals)) if vals else float("nan"),
            "ci95": [lo, hi],
            "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi)
                                  and (lo > 0 or hi < 0)),
            "per_subject": per}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=0, help="0 = every held-out clip")
    args = ap.parse_args()

    model = load_model(CKPT_FOR[DET], "base")
    if model is None:
        raise SystemExit(f"checkpoint missing: {CKPT_FOR[DET]}")
    clips = clip_list(args.clips or None)
    # label -> sigma in frames
    sigmas = {"unfiltered": 0.0}
    for c in CUTOFFS:
        sigmas[f"{c:.0f} Hz"] = FS / (2.0 * np.pi * c)
    sigmas[f"tool @30fps (sd {TOOL_SIGMA_AT_30FPS})"] = TOOL_SIGMA_AT_30FPS
    arms = list(sigmas)
    print(f"=== phase 13 arm 1: {len(clips)} clips, {len(arms)} arms ===")
    print(f"  {DET} detections + {PAD} padding + {WINDOWS} windows, "
          f"filtering at {FS:.0f} Hz\n")

    # per arm, per limb, per joint, per subject: a list of clip means
    acc: dict = {a: {lb: {j: defaultdict(list) for j in JOINTS} for lb in LIMBS}
                 for a in arms}
    for i, (action, subj, stem, _n) in enumerate(clips):
        gt3d = np.asarray(np.load(
            DATA / "gt_markers3d_by_cam" / action / subj / f"{stem}.npz")["markers_h36m"])
        det = np.load(DATA / f"det_markers2d_by_cam_{DET}" / action / subj
                      / f"{stem}.npy").astype(float)
        n = min(len(gt3d), len(det))
        if n < 20:
            continue
        gt3d, det = gt3d[:n], det[:n]
        near = ("l" if gt3d[:, H36M["l_hip"], 2].mean()
                < gt3d[:, H36M["r_hip"], 2].mean() else "r")
        far = "r" if near == "l" else "l"
        ga = {"near": sagittal_angles(gt3d, near),
              "far": sagittal_angles(gt3d, far)}

        for arm in arms:
            x = smooth2d_sigma(det, sigmas[arm])
            est = lift(model, x, PAD, WINDOWS)
            ea = {"near": sagittal_angles(est, near),
                  "far": sagittal_angles(est, far)}
            for limb in LIMBS:
                for j in JOINTS:
                    e = np.abs(ea[limb][j] - ga[limb][j])
                    if np.all(np.isfinite(e)):
                        acc[arm][limb][j][subj].append(float(e.mean()))
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(clips)} clips")

    def pooled(arm: str, limb: str, j: str) -> float:
        v = [x for s in acc[arm][limb][j] for x in acc[arm][limb][j][s]]
        return float(np.mean(v)) if v else float("nan")

    def mean_per_subject(arm: str, limb: str) -> dict[str, list[float]]:
        """Per subject, the per-clip mean of hip and knee."""
        out: dict[str, list[float]] = {}
        for s in acc[arm][limb]["hip"]:
            h, k = acc[arm][limb]["hip"][s], acc[arm][limb]["knee"][s]
            m = min(len(h), len(k))
            out[s] = [0.5 * (h[i] + k[i]) for i in range(m)]
        return out

    out = {"detector": DET, "padding": PAD, "windows": WINDOWS,
           "capture_fps": FS, "cutoffs_hz": CUTOFFS,
           "sigma_frames": {k: round(v, 3) for k, v in sigmas.items()},
           "n_clips": len(clips),
           "pre_registration": "results/phase13.md, frozen 2026-10-19",
           "filter_is_causal": False,
           "arms": {}}
    for arm in arms:
        e = {"limbs": {}}
        for limb in LIMBS:
            e["limbs"][limb] = {
                "hip_pooled": pooled(arm, limb, "hip"),
                "knee_pooled": pooled(arm, limb, "knee"),
                "mean_pooled": 0.5 * (pooled(arm, limb, "hip")
                                      + pooled(arm, limb, "knee"))}
            if arm != "unfiltered":
                e["limbs"][limb]["paired_vs_unfiltered"] = paired(
                    mean_per_subject("unfiltered", limb),
                    mean_per_subject(arm, limb))
        out["arms"][arm] = e

    print(f"\n=== VERDICT ===")
    print(f"  {'arm':<26}{'limb':>6}{'hip':>8}{'knee':>8}{'mean':>8}"
          f"{'paired delta':>15}{'95% CI':>20}")
    for arm in arms:
        for limb in LIMBS:
            e = out["arms"][arm]["limbs"][limb]
            if arm == "unfiltered":
                print(f"  {arm:<26}{limb:>6}{e['hip_pooled']:>8.3f}"
                      f"{e['knee_pooled']:>8.3f}{e['mean_pooled']:>8.3f}"
                      f"{'baseline':>15}")
            else:
                d = e["paired_vs_unfiltered"]
                lo, hi = d["ci95"]
                flag = "excludes 0" if d["excludes_zero"] else "spans 0"
                print(f"  {arm:<26}{limb:>6}{e['hip_pooled']:>8.3f}"
                      f"{e['knee_pooled']:>8.3f}{e['mean_pooled']:>8.3f}"
                      f"{d['delta_deg']:>+15.3f}   [{lo:>+6.3f}, {hi:>+6.3f}] {flag}")

    wins = [(arm, limb) for arm in arms[1:] for limb in LIMBS
            if out["arms"][arm]["limbs"][limb]["paired_vs_unfiltered"]["delta_deg"] < 0
            and out["arms"][arm]["limbs"][limb]["paired_vs_unfiltered"]["excludes_zero"]]
    out["gate_met_for"] = [{"arm": a, "limb": lb} for a, lb in wins]
    print(f"\n  gate met (improves, CI excludes zero): "
          f"{wins if wins else 'NOTHING'}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
