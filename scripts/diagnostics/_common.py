"""Shared loading for the phase 15 diagnostics.

Deliberately thin. The point of putting it here is that all three diagnostics
reach MoVi through `phase15_checks`, so no diagnostic ever holds a calibration
array -- `project_reference` takes a camera NAME and hands back pixels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "athleticspose" / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

MOVI = REPO / "data" / "movi" / "raw" / "round_f"
MOCAP = MOVI / "mocap"


def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--subject", default="Subject_1")
    ap.add_argument("--camera", default="PG1", choices=["PG1", "PG2"],
                    help="calibrated cameras only; CP ships none")
    ap.add_argument("--frames", type=int, default=0,
                    help="0 = the whole jogging window")
    return ap


def load(subject: str, camera: str, frames: int):
    """(gt_window, est_free_inputs) for one subject/camera jogging window.

    Returns a dict with the pieces every diagnostic needs, and never any
    calibration value.
    """
    from phase15_checks import load_v3d, to_h36m
    from phase15_movi import CAMERAS, MOCAP_PER_VIDEO, jogging_window

    mat = MOCAP / f"F_v3d_{subject}.mat"
    if not mat.exists():
        raise SystemExit(f"mocap missing: {mat}")
    video = MOVI / CAMERAS[camera]["pattern"].format(s=subject.split("_")[-1])
    if not video.exists():
        raise SystemExit(f"video missing: {video}")

    v = load_v3d(mat)
    ji, f30, f120 = jogging_window(v)
    gt_world = to_h36m(v)

    n_v = int(f30[1] - f30[0] + 1)
    if frames:
        n_v = min(n_v, frames)
    mk = int(f120[0]) + MOCAP_PER_VIDEO * np.arange(n_v)
    mk = mk[mk < gt_world.shape[0]]
    return {"mat": mat, "video": video, "subject": subject, "camera": camera,
            "jogging_index": ji, "a30": int(f30[0]), "n_v": len(mk),
            "mocap_idx": mk, "gt": gt_world[mk], "gt_world": gt_world,
            "f120_start": int(f120[0])}


def lifter():
    from phase3_infer import load_model
    from phase7_inference_fix import CKPT_FOR
    m = load_model(CKPT_FOR["coco"], "base")
    if m is None:
        raise SystemExit(f"checkpoint missing: {CKPT_FOR['coco']}")
    return m


def lift_2d(model, kp2d: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """The frozen lifting path: resample up, lift, resample back."""
    from phase7_inference_fix import lift
    from video_kinematics import MODEL_FPS, resample
    n = len(kp2d)
    n_model = max(int(round(n * MODEL_FPS / fps)), 2)
    return resample(lift(model, resample(kp2d, n_model), "edge", "overlap"), n)
