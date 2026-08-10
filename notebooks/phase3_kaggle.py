"""Phase 3 on Kaggle -- monocular 3D kinematics error, in the form phase 2 needs.

Written as ordered CELLS (`# %% CELL n`). Run `python scripts/make_phase3_notebook.py`
to emit notebooks/phase3_kaggle.ipynb for direct upload to Kaggle.

BEFORE RUNNING
  Kaggle -> Settings -> Accelerator: GPU T4 x2 or P100
  Kaggle -> Settings -> Internet: ON   (repo, dataset and checkpoints are fetched)
  No Ferber data here. CLAUDE.md keeps the archive on the laptop; nothing in this
  phase needs it.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
  AthleticsPose does not release the original videos (anonymisation). What ships
  is 2D marker detections plus 3D ground truth, so this evaluates the
  **2D -> 3D lifting** stage using 2D from a real detector (`det_ft`). Detector
  error is therefore included; raw video decoding and person detection are not.
  That is the dominant error source for monocular 3D pose, but the result is a
  LOWER BOUND on a full in-the-wild pipeline and must be reported as one.

  Second, and more important: the released predictor denormalises each clip using
  a scale derived from GROUND TRUTH 3D. A deployed system has no such scale. The
  errors below are therefore optimistic, and phase 4 cannot assume this scale is
  available.

WHAT PHASE 2 NEEDS BACK
  Sagittal hip/knee angle error, split NEAR vs FAR limb, plus the
  systematic/random split. Phase 2 found the signal survives 30 fps alone and
  20 deg of far-limb error alone, but dies at 30 fps + 16 deg. The operating
  point decides whether phase 4 is worth starting.
"""

# %% CELL 1 -- environment
# Deliberately non-fatal. Kaggle halts the whole notebook on the first exception,
# so a missing nvidia-smi must report rather than raise -- otherwise a machine
# configuration problem looks identical to a code bug.
import shutil
import subprocess
import sys

print("python", sys.version.split()[0])

if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv"], capture_output=True, text=True).stdout)
else:
    print("nvidia-smi NOT on PATH")

try:
    import torch
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
          "| device count:", torch.cuda.device_count())
    HAS_GPU = torch.cuda.is_available()
except Exception as exc:  # noqa: BLE001
    print("torch unavailable:", exc)
    HAS_GPU = False

if not HAS_GPU:
    print("\n*** NO GPU ALLOCATED ***\n"
          "enable_gpu was set in the kernel metadata, so the likely causes are:\n"
          "  - the Kaggle account is not phone-verified (GPU and Internet both\n"
          "    require it, and are silently withheld otherwise)\n"
          "  - the weekly GPU quota (~30 h) is exhausted\n"
          "Inference will fall back to CPU: slower, but the angle error this phase\n"
          "measures is identical either way.")

# %% CELL 2 -- fetch repo, dataset and the three checkpoints
# CC BY-NC-SA 4.0, non-commercial research only -- already recorded in CLAUDE.md.
import os
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(WORK)
if not (WORK / "AthleticsPose").exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/SZucchini/AthleticsPose.git"], check=True)
os.chdir(WORK / "AthleticsPose")

# `make setup` builds the uv venv and downloads data + checkpoints. Kaggle
# already has torch, so fall back to plain pip if uv/CUDA pinning fights it.
subprocess.run(["bash", "-lc", "pip install -q uv"], check=False)
r = subprocess.run(["bash", "-lc", "make download || "
                    'curl -L -o data.zip "https://github.com/SZucchini/'
                    'AthleticsPose/releases/latest/download/data.zip" '
                    "&& unzip -q -o data.zip"], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
print(subprocess.run(["bash", "-lc", "find . -name '*.ckpt' -o -name '*.pth' | head"],
                     capture_output=True, text=True).stdout)

# %% CELL 3 -- INSPECT before computing anything
# The layout below is expected, not verified from outside Kaggle. Confirm it
# rather than trusting it -- writing a pipeline against a guessed schema is how
# phase 0's traps were made.
import numpy as np

root = Path("data")
print("top level:", sorted(p.name for p in root.iterdir()))
npz = sorted(root.rglob("*.npz"))
print(f"\n{len(npz)} npz files; first few:")
for f in npz[:5]:
    print("   ", f.relative_to(root))

if npz:
    z = np.load(npz[0], allow_pickle=True)
    print("\nkeys in", npz[0].name, "->", list(z.keys()))
    for k in z.keys():
        a = np.asarray(z[k])
        print(f"   {k:<24} {str(a.shape):<18} {a.dtype}")

# THE decisive check: are camera extrinsics present? Without them the near/far
# limb split cannot be computed, and it must NOT be guessed -- a wrong
# assignment would invert the headline result.
CAM_KEYS = [k for k in (z.keys() if npz else [])
            if any(t in k.lower() for t in ("cam", "extrinsic", "rt", "proj"))]
print("\ncamera-like keys:", CAM_KEYS or "NONE FOUND -- see CELL 6")

# %% CELL 4 -- which events, how many subjects, what framerate
# configs/data/{running,sd_sprint,hurdle,racewalk,discus_shotput,all}.yaml exist,
# so running and sprinting are both available. Restrict to those two: they are
# the motions phase 2's stance-phase model applies to.
print(subprocess.run(["bash", "-lc", "cat configs/data/running.yaml; echo ---; "
                      "cat configs/data/sd_sprint.yaml; echo ---; "
                      "cat configs/prediction/from_2d_markers.yaml"],
                     capture_output=True, text=True).stdout)

# %% CELL 5 -- evaluate all three checkpoints (MPJPE), then predict
# The generic-vs-fine-tuned contrast is the point: it brackets the operating
# point instead of assuming one.
for cfg, extra in (("default", ""),               # trained on AthleticsPose
                   ("h36m_pretrained", ""),       # generic baseline
                   ("ap3d_pretrained", "model=small")):
    cmd = f"uv run python scripts/evaluate.py evaluation={cfg} {extra}"
    r = subprocess.run(["bash", "-lc", cmd + " || " + cmd.replace("uv run ", "")],
                       capture_output=True, text=True)
    print(f"\n===== {cfg} =====\n{r.stdout[-1500:]}{r.stderr[-600:]}")

cmd = ("uv run python scripts/predict.py prediction=from_2d_markers "
       "prediction.input.marker_type=det_ft")
r = subprocess.run(["bash", "-lc", cmd + " || " + cmd.replace("uv run ", "")],
                   capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-600:])
print(subprocess.run(["bash", "-lc", "find data -name 'predictions' -type d"],
                     capture_output=True, text=True).stdout)

# %% CELL 6 -- joint angles, near/far split, error decomposition
# Upload scripts/phase3_angles.py as a Kaggle dataset named `overstride-scripts`.
# Its H36M index map was checked against athleticspose/statics/joints.py and
# matches (PELVIS, R_HIP, R_KNEE, R_ANKLE, L_HIP, L_KNEE, L_ANKLE, ...).
sys.path.insert(0, "/kaggle/input/overstride-scripts")
from phase3_angles import (H36M, error_decomposition,  # noqa: E402
                           sagittal_angles, summarise)

PRED_DIR = Path("data/AthleticsPoseDataset/predictions")
rows = []

def near_side(gt_kp: np.ndarray, cam_pos: np.ndarray | None) -> str | None:
    """'l' or 'r' -- whichever hip is nearer the camera. None if unknowable."""
    if cam_pos is None:
        return None
    dl = np.linalg.norm(gt_kp[:, H36M["l_hip"]] - cam_pos, axis=-1).mean()
    dr = np.linalg.norm(gt_kp[:, H36M["r_hip"]] - cam_pos, axis=-1).mean()
    return "l" if dl < dr else "r"

for pred_file in sorted(PRED_DIR.rglob("*.npy")):
    est = np.load(pred_file)                       # (T, 17, 3)
    gt_file = next((f for f in npz if f.stem in pred_file.stem
                    or pred_file.stem in f.stem), None)
    if gt_file is None:
        continue
    z = np.load(gt_file, allow_pickle=True)
    gt = np.asarray(z["markers_h36m"])             # (T, 17, 3)
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]

    cam_pos = np.asarray(z[CAM_KEYS[0]]) if CAM_KEYS else None
    if cam_pos is not None and cam_pos.size >= 3:
        cam_pos = cam_pos.reshape(-1)[:3]
    near = near_side(gt, cam_pos)

    for side in ("l", "r"):
        e, g = sagittal_angles(est, side), sagittal_angles(gt, side)
        for joint in ("hip", "knee"):     # ankle needs a toe keypoint -> NaN
            d = error_decomposition(e[joint], g[joint])
            d.update(joint=joint, clip=pred_file.stem, side=side,
                     limb=("unknown" if near is None
                           else ("near" if side == near else "far")))
            rows.append(d)

print(f"{len(rows)} rows from {len(set(r['clip'] for r in rows))} clips")
if not CAM_KEYS:
    print("\nWARNING: no camera extrinsics found. The near/far split -- the number "
          "phase 2 actually needs -- is NOT computed. Report as unavailable "
          "rather than guessing; a wrong assignment would invert the result.")
print(summarise(rows))

# %% CELL 7 -- export the few KB that leave Kaggle
import json

out = Path("/kaggle/working/phase3_angle_errors.json")
out.write_text(json.dumps({"rows": rows,
                           "near_far_available": bool(CAM_KEYS),
                           "caveats": [
                               "2D->3D lifting only; original videos not released",
                               "per-clip denormalisation uses GT 3D scale -> optimistic",
                               "ankle unavailable: no toe keypoint in H36M-17"]},
                          indent=2), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size/1e3:.1f} KB) -- download this")
