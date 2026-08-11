"""Phase 3 -- monocular 3D kinematics error, measured. Runs locally on CPU.

WHY LOCAL, NOT KAGGLE
  CLAUDE.md put phase 3 on Kaggle because CUDA was assumed necessary. Two Kaggle
  runs were given neither GPU nor Internet (the account is not phone-verified),
  so that route is blocked. But GPU was only ever a speed convenience: the
  quantity this phase measures -- joint-angle error between predicted and
  ground-truth 3D -- is identical on CPU. CLAUDE.md forbids solutions REQUIRING a
  local GPU; CPU inference requires none, and the Ferber archive is not involved.

WHAT IS MEASURED
  MotionAGFormer lifts detected 2D markers to 3D. We compare predicted vs
  ground-truth sagittal hip/knee angles on held-out subjects, and report the
  error in the form phase 2 needs: split by NEAR vs FAR limb relative to the
  camera, plus a systematic/random decomposition.

HELD-OUT SUBJECTS ONLY
  configs/data/running.yaml designates S11, S13, S16 as test. The fine-tuned
  checkpoint was trained on the others, so scoring it on those would flatter it.

KNOWN OPTIMISM, carried into the report
  - 2D -> 3D lifting only; original videos are not released, so video decode and
    person detection error are excluded. This is a LOWER BOUND.
  - The released pipeline denormalises each clip using a scale derived from GT
    3D. A deployed system has no such scale.

Run: .venv/Scripts/python.exe scripts/phase3_infer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
AP = REPO / "data" / "athleticspose"
sys.path.insert(0, str(AP / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from athleticspose.models.MotionAGFormer.model import MotionAGFormer  # noqa: E402
from athleticspose.utils import normalize_kpts  # noqa: E402

from phase3_angles import H36M, error_decomposition, sagittal_angles  # noqa: E402

DATA = AP / "data" / "AthleticsPoseDataset"
CKPT = AP / "checkpoints"
OUT = REPO / "results" / "phase3_angle_errors.json"

TEST_SUBJECTS = ["S11", "S13", "S16"]        # configs/data/running.yaml
ACTIONS = ["running", "sprint"]
CLIP = 81

MODELS = {
    "ath-det-ft (fine-tuned)": ("motionagformer-b-ath-det-ft-v1.ckpt", "base"),
    "h36m (generic)": ("motionagformer-b-h36m.pth", "base"),
    "ap3d (other sport)": ("motionagformer-s-ap3d.pth", "small"),
}
ARCH = {"base": dict(n_layers=16, dim_in=3, dim_feat=128, num_heads=8,
                     neighbour_num=2, n_frames=CLIP),
        "small": dict(n_layers=26, dim_in=3, dim_feat=64, num_heads=8,
                      neighbour_num=2, n_frames=CLIP)}


def load_model(fname: str, arch: str) -> MotionAGFormer | None:
    path = CKPT / fname
    if not path.exists():
        print(f"  missing checkpoint {fname}")
        return None
    model = MotionAGFormer(**ARCH[arch])
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob)
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 20:
        print(f"  {fname}: {len(missing)} missing keys -- architecture mismatch, "
              "skipping rather than reporting a meaningless error")
        return None
    print(f"  {fname}: loaded ({len(missing)} missing, {len(unexpected)} unexpected)")
    model.eval()
    return model


def camera_positions() -> dict[str, np.ndarray]:
    """Map camera_params file stem -> (n_cam, 3) camera positions in world mm."""
    out = {}
    for f in sorted((DATA / "camera_params").glob("*.json")):
        cams = json.loads(f.read_text(encoding="utf-8"))["Cameras"]
        out[f.stem] = np.array([[c["Transform"]["x"], c["Transform"]["y"],
                                 c["Transform"]["z"]] for c in cams])
    return out


@torch.no_grad()
def predict(model: MotionAGFormer, det2d: np.ndarray, gt3d: np.ndarray) -> np.ndarray:
    """Replicate the released From2DMarkersPredictor for one clip."""
    t = det2d.shape[0]
    coords = np.concatenate([det2d[:, :, :2],
                             np.zeros_like(det2d[:, :, :1])], axis=-1)
    coords_norm, _ = normalize_kpts(coords)
    x = np.concatenate([coords_norm[:, :, :2], det2d[:, :, 2:3]], axis=-1)
    _, gt_scale = normalize_kpts(gt3d)

    # Sequences longer than the 81-frame window are split into consecutive
    # windows and stitched, as the released predictor does. Padding the tail and
    # trimming afterwards keeps the model input shape fixed.
    out = np.empty((t, 17, 3), dtype=np.float64)
    for start in range(0, t, CLIP):
        chunk = x[start:start + CLIP]
        valid = len(chunk)
        if valid < CLIP:
            chunk = np.pad(chunk, [(0, CLIP - valid), (0, 0), (0, 0)])
        inp = torch.from_numpy(chunk[None]).float()
        out[start:start + valid] = model(inp)[0].numpy()[:valid]
    return out * gt_scale


def main() -> int:
    cams = camera_positions()
    print(f"camera_params: {len(cams)} sessions")
    rows: list[dict] = []
    mpjpe_by_model: dict[str, list[float]] = {}

    for label, (fname, arch) in MODELS.items():
        print(f"\n=== {label} ===")
        model = load_model(fname, arch)
        if model is None:
            continue
        mpjpes: list[float] = []
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
                    if n < 10:
                        continue
                    det2d, gt3d = det2d[:n], gt3d[:n]

                    est = predict(model, det2d, gt3d)

                    # MPJPE, root-relative, as the paper reports it
                    e = est - est[:, :1]
                    g = gt3d - gt3d[:, :1]
                    mpjpes.append(float(np.mean(np.linalg.norm(e - g, axis=-1))))
                    n_clips += 1

                    # Near/far limb. "by_cam" means organised by camera, NOT
                    # expressed in camera coordinates: y is vertical and z is
                    # ROOT-RELATIVE DEPTH (the pelvis sits at exactly z = 0).
                    #
                    # The sign was verified rather than assumed, because getting
                    # it backwards would invert the headline. Across 592 held-out
                    # clips, corr(z_L - z_R, conf_L - conf_R) = -0.664 using the
                    # 2D detector's own confidence channel: the deeper limb is
                    # detected less confidently, as an occluded limb must be.
                    # So larger z = farther, and near = smaller z.
                    zl = gt3d[:, H36M["l_hip"], 2].mean()
                    zr = gt3d[:, H36M["r_hip"], 2].mean()
                    near = "l" if zl < zr else "r"

                    for side in ("l", "r"):
                        ea, ga = sagittal_angles(est, side), sagittal_angles(gt3d, side)
                        for joint in ("hip", "knee"):
                            d = error_decomposition(ea[joint], ga[joint])
                            if not np.isfinite(d["mae_deg"]):
                                continue
                            d.update(joint=joint, model=label, action=action,
                                     subject=subj, clip=det_file.stem, side=side,
                                     limb="near" if side == near else "far")
                            rows.append(d)

        mpjpe_by_model[label] = mpjpes
        if mpjpes:
            print(f"  {n_clips} clips | MPJPE {np.mean(mpjpes):.1f} mm "
                  f"(median {np.median(mpjpes):.1f})")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows,
        "mpjpe_mm": {k: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                         "n_clips": len(v)} for k, v in mpjpe_by_model.items() if v},
        "test_subjects": TEST_SUBJECTS,
        "actions": ACTIONS,
        "caveats": [
            "2D->3D lifting only; original videos not released -> LOWER BOUND",
            "per-clip denormalisation uses GT 3D scale -> optimistic",
            "ankle unavailable: H36M-17 has no toe keypoint",
            "held-out subjects only (S11,S13,S16) per configs/data/running.yaml",
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
