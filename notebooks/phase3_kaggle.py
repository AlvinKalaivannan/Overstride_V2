"""Phase 3 on Kaggle -- video/2D -> 3D kinematics, and the error phase 2 needs.

This file is written as ordered CELLS. Paste each block between the `# %% CELL`
markers into its own Kaggle notebook cell, or run `jupytext --to notebook` on it.

BEFORE RUNNING
  - Kaggle notebook -> Settings -> Accelerator: GPU T4 x2 (or P100)
  - Kaggle notebook -> Settings -> Internet: ON  (needed to fetch the repo,
    dataset and checkpoints)
  - No Ferber data here. CLAUDE.md forbids the archive leaving the laptop, and
    nothing in this phase needs it.

WHAT THIS PRODUCES
  results/phase3_angle_errors.json -- sagittal joint-angle error by joint and by
  NEAR/FAR limb, plus the systematic/random split. Download it and run
  scripts/phase3_report.py locally.

DESIGN NOTE
  Cells 2-4 INSPECT the data before anything computes. The AthleticsPose array
  layout has not been verified from here, and writing a rigid pipeline against an
  unverified schema is how phase 0's traps got made. Look first.
"""

# %% CELL 1 -- environment
import subprocess
import sys

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv"], capture_output=True, text=True).stdout)
print("python", sys.version)

# %% CELL 2 -- fetch repo, dataset and checkpoints
# AthleticsPose: CC BY-NC-SA 4.0, non-commercial research only (CLAUDE.md
# already records this constraint). Dataset + 3 checkpoints ship together:
#   - trained on AthleticsPose      (sport fine-tuned)
#   - trained on Human3.6M          (generic baseline)
#   - trained on AthletePose3D      (other sport dataset)
# The generic-vs-fine-tuned contrast is the point: it brackets the realistic
# operating point rather than assuming one.
import os
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(WORK)
if not (WORK / "AthleticsPose").exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/SZucchini/AthleticsPose.git"], check=True)
os.chdir(WORK / "AthleticsPose")
subprocess.run(["bash", "-lc",
                'curl -L -o data.zip '
                '"https://github.com/SZucchini/AthleticsPose/releases/latest/'
                'download/data.zip" && unzip -q -o data.zip'], check=True)
print(subprocess.run(["bash", "-lc", "find . -maxdepth 3 -name '*.pth' -o "
                      "-maxdepth 3 -name '*.ckpt' | head -20"],
                     capture_output=True, text=True).stdout)

# %% CELL 3 -- INSPECT the dataset before assuming anything about it
import numpy as np

root = Path("data")
print("top level:", sorted(p.name for p in root.iterdir())[:20])
for pat in ("*.npz", "*.npy", "*.pkl", "*.json"):
    for f in sorted(root.rglob(pat))[:6]:
        print(f"  {f.relative_to(root)}  {f.stat().st_size/1e6:.1f} MB")

sample = next(iter(sorted(root.rglob("*.npz"))), None)
if sample is not None:
    z = np.load(sample, allow_pickle=True)
    print("\nkeys:", list(z.keys()))
    for k in list(z.keys())[:10]:
        try:
            print(f"  {k}: {np.asarray(z[k]).shape} {np.asarray(z[k]).dtype}")
        except Exception as e:
            print(f"  {k}: <{e}>")

# %% CELL 4 -- what do the actions/subjects/cameras look like?
# Confirm: which events are present, framerate, number of cameras, and whether
# running/sprinting is among them. Phase 2 assumed ~0.295 s stance; check the
# sampling here matches something comparable.
# (Fill in against the keys printed by CELL 3 -- do not guess.)

# %% CELL 5 -- run the released evaluation + prediction pipeline
subprocess.run(["bash", "-lc", "pip install -q uv && uv sync --frozen || "
                "pip install -q -r requirements.txt || true"], check=False)
subprocess.run(["bash", "-lc",
                "uv run python scripts/evaluate.py evaluation=default || "
                "python scripts/evaluate.py evaluation=default"], check=False)
subprocess.run(["bash", "-lc",
                "uv run python scripts/predict.py prediction=from_2d_markers "
                "prediction.input.marker_type=det_ft || "
                "python scripts/predict.py prediction=from_2d_markers "
                "prediction.input.marker_type=det_ft"], check=False)
print(subprocess.run(["bash", "-lc",
                      "find data -name 'predictions*' -maxdepth 3 | head"],
                     capture_output=True, text=True).stdout)

# %% CELL 6 -- joint angles + the decomposition phase 2 needs
# scripts/phase3_angles.py from the Overstride repo. Upload it as a Kaggle
# dataset, or paste its contents into a cell. It has no Ferber dependency.
sys.path.insert(0, "/kaggle/input/overstride-scripts")
from phase3_angles import (error_decomposition, sagittal_angles,  # noqa: E402
                           summarise)

def near_far_limb(kp_world: np.ndarray, cam_pos: np.ndarray) -> str:
    """Which limb is nearer the camera. Returns 'l' or 'r'.

    Uses mean hip-to-camera distance over the clip. If camera extrinsics are not
    available, this must NOT be guessed -- the near/far split is the entire point
    of the phase, and a wrong assignment would invert the result.
    """
    from phase3_angles import H36M
    dl = np.linalg.norm(kp_world[:, H36M["l_hip"]] - cam_pos, axis=-1).mean()
    dr = np.linalg.norm(kp_world[:, H36M["r_hip"]] - cam_pos, axis=-1).mean()
    return "l" if dl < dr else "r"

rows = []
# for each clip:  est_kp (n,17,3), gt_kp (n,17,3), cam_pos (3,), stride_ids
#     near = near_far_limb(gt_kp, cam_pos)
#     for side in ('l','r'):
#         e = sagittal_angles(est_kp, side); g = sagittal_angles(gt_kp, side)
#         for joint in ('hip','knee'):          # ankle needs a toe keypoint
#             d = error_decomposition(e[joint], g[joint], stride_ids)
#             d.update(joint=joint, limb='near' if side == near else 'far',
#                      clip=clip_id, checkpoint=ckpt_name)
#             rows.append(d)

# %% CELL 7 -- export the small result file (this is all that leaves Kaggle)
import json

out = Path("/kaggle/working/phase3_angle_errors.json")
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(summarise(rows))
print(f"wrote {out}  ({out.stat().st_size/1e3:.1f} KB)")
