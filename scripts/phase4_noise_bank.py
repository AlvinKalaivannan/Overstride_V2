"""Phase 4 step 1 -- build an EMPIRICAL error bank from phase 3.

Phase 2 injected Gaussian noise at assumed magnitudes. Phase 3 measured what
monocular pose actually does. This extracts the real residual (estimate minus
ground truth) as a time-normalised curve, so phase 4 can perturb the Ferber
waveforms with measured error instead of an assumed distribution.

Why residual CURVES and not a sigma
  A Gaussian sigma throws away two things the real error has: temporal structure
  (the residual drifts across stance rather than jittering independently) and a
  systematic offset (phase 3 found knee bias of -8 to -12 deg for the generic
  checkpoint). Both matter here, because the phase 5 feature is a LEFT-RIGHT
  DIFFERENCE -- error that is common to both limbs cancels, error that is
  limb-specific does not.

Only the fine-tuned checkpoint is run: it is the one whose error level
(3.4 deg) sits inside the survivable region, so it is the operating point worth
propagating. Running all three would triple the cost for no decision value.

Output: data/derived/phase4_error_bank.npz
  curves  (n, 2, 101)  residual in degrees, channels [hip, knee]
  limb    (n,)         'near' | 'far'
  clip    (n,)         source clip id, so draws can avoid reusing one clip twice

Run: .venv/Scripts/python.exe scripts/phase4_noise_bank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
AP = REPO / "data" / "athleticspose"
sys.path.insert(0, str(AP / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import (CLIP, DATA, TEST_SUBJECTS, load_model,  # noqa: E402
                          predict)

OUT = REPO / "data" / "derived" / "phase4_error_bank.npz"
N_POINTS = 101
ACTIONS = ["running", "sprint"]
MIN_FRAMES = 20          # need enough to time-normalise meaningfully


def resample(series: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    src = np.linspace(0.0, 1.0, len(series))
    return np.interp(np.linspace(0.0, 1.0, n), src, series)


def main() -> int:
    model = load_model("motionagformer-b-ath-det-ft-v1.ckpt", "base")
    if model is None:
        raise SystemExit("fine-tuned checkpoint missing")

    curves, limbs, clips = [], [], []
    n_clips = 0

    for action in ACTIONS:
        adir = DATA / "det_markers2d_by_cam_ft" / action
        if not adir.is_dir():
            continue
        for subj in TEST_SUBJECTS:
            sdir = adir / subj
            if not sdir.is_dir():
                continue
            for det_file in sorted(sdir.glob("*.npy")):
                gt_file = (DATA / "gt_markers3d_by_cam" / action / subj
                           / f"{det_file.stem}.npz")
                if not gt_file.exists():
                    continue
                det2d = np.load(det_file).astype(np.float64)
                gt3d = np.asarray(np.load(gt_file)["markers_h36m"])
                n = min(len(det2d), len(gt3d))
                if n < MIN_FRAMES:
                    continue
                det2d, gt3d = det2d[:n], gt3d[:n]

                with torch.no_grad():
                    est = predict(model, det2d, gt3d)

                # near = smaller root-relative depth; sign verified in phase 3
                # via corr(z_L - z_R, conf_L - conf_R) = -0.664
                near = ("l" if gt3d[:, H36M["l_hip"], 2].mean()
                        < gt3d[:, H36M["r_hip"], 2].mean() else "r")

                for side in ("l", "r"):
                    ea = sagittal_angles(est, side)
                    ga = sagittal_angles(gt3d, side)
                    res = []
                    ok = True
                    for joint in ("hip", "knee"):
                        r = ea[joint] - ga[joint]
                        if not np.all(np.isfinite(r)):
                            ok = False
                            break
                        res.append(resample(r))
                    if not ok:
                        continue
                    curves.append(np.stack(res))          # (2, 101)
                    limbs.append("near" if side == near else "far")
                    clips.append(f"{action}/{subj}/{det_file.stem}")
                n_clips += 1

    C = np.stack(curves).astype(np.float32)
    L = np.array(limbs)
    K = np.array(clips)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, curves=C, limb=L, clip=K)

    print(f"\n{n_clips} clips -> {len(C)} residual curves  {C.shape}")
    for j, name in enumerate(("hip", "knee")):
        a = np.abs(C[:, j, :])
        print(f"  {name:5} MAE {a.mean():5.2f} deg | bias {C[:, j, :].mean():+5.2f}"
              f" | within-curve sd {C[:, j, :].std(axis=1).mean():5.2f}")
    for lab in ("near", "far"):
        m = L == lab
        print(f"  {lab:5} n={m.sum():4}  MAE {np.abs(C[m]).mean():5.2f} deg")
    print(f"\nwrote {OUT.relative_to(REPO)} ({OUT.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
