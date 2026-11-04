"""Each lifter checkpoint's input contract, and a checker that enforces it.

Pre-registered in `results/phase15f.md`. Three defects in this project came
from feeding a checkpoint an input that looked right and was on the wrong
scale, and none of them raised:

* **30.07 deg** on MoVi -- `detect_coco17` wrote torchvision's raw
  `keypoints_scores` (mean 12.4) into a channel the checkpoint trained at
  **0.8676**. Knee flexion read +66 deg at r = +0.02.
* **0.60-1.63 deg** on AthleticsPose -- `gt_markers2d_by_cam` ships a constant
  **1.0** in that channel (phase 15E).
* a whole clip of skeletons at the image origin, because `np.nan_to_num` turned
  undetected frames into zeros and the lifter's 50%-overlap windows smeared
  them into their neighbours (phase 15, D5).

An input channel's **distribution** is part of the interface. This module
writes each contract down and checks a prepared array against it **before** any
lift, so a scale error is a reported flag instead of a number nobody questions.

**Window length is read from the weights, not from a config.** Each checkpoint's
`graph_temporal.mixer.batch_norm` is a BatchNorm over the temporal node
dimension, so its shape *is* `n_frames`. All three checkpoints on disk report
**81**, which corrects the 243 that `configs/h36m/MotionAGFormer-base.yaml` and
the upstream README's ambiguous "243 | 81" column imply for `b-h36m`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Contract:
    """One checkpoint's input contract, as established from repo/config/paper."""

    name: str
    arch: str                     # phase3_infer.ARCH key
    joints: str                   # joint order
    norm: str                     # "person_scale" | "image_res"
    conf_value: float | None      # what to put in channel 2
    conf_label: str               # where that value comes from
    train_fps: float | None       # rate the model saw motion at
    n_frames: int                 # window length, READ FROM THE WEIGHTS
    determined: int               # contract items established, out of 5
    undetermined: tuple[str, ...] = field(default=())
    include_in_primary: bool = True
    note: str = ""


# Measured on the tune set, 60 clips / 131,121 values (video_kinematics).
ATH_CONF_MEAN = 0.8676

CONTRACTS: dict[str, Contract] = {
    "b-ath-det-coco": Contract(
        name="motionagformer-b-ath-det-coco-v1.ckpt", arch="base",
        joints="h36m17", norm="person_scale",
        conf_value=ATH_CONF_MEAN,
        conf_label="measured AthleticsPose generic-COCO detector scores, "
                   "mean 0.8676 sd 0.1164",
        train_fps=100.0, n_frames=81, determined=5,
        note="the deployed checkpoint; in-domain on AthleticsPose"),
    "b-h36m": Contract(
        name="motionagformer-b-h36m.pth", arch="base",
        joints="h36m17", norm="image_res",
        conf_value=1.0,
        conf_label="MotionBERT datareader_h36m fills ones when confidence is "
                   "absent; the Stacked Hourglass distribution it actually "
                   "trained on is UNDETERMINED (pkl unavailable)",
        train_fps=50.0, n_frames=81, determined=4,
        undetermined=("confidence distribution",),
        note="upstream TaatiTeam/MotionAGFormer, trained on H36M at 50 Hz"),
    "s-ap3d": Contract(
        name="motionagformer-s-ap3d.pth", arch="small",
        joints="h36m17", norm="person_scale",
        conf_value=ATH_CONF_MEAN,
        conf_label="NOT its own contract -- the AthleticsPose repo's, which is "
                   "the one documented way to feed this file "
                   "(evaluation=ap3d_pretrained model=small)",
        train_fps=100.0, n_frames=81, determined=3,
        undetermined=("normalization", "confidence convention"),
        include_in_primary=False,
        note="EXCLUDED FROM THE PRIMARY: normalization and confidence are not "
             "stated in its repo, config or paper, and its training rate is "
             "mixed (120 FPS running, 60 FPS otherwise) with none given"),
}


def normalize(kp: np.ndarray, contract: Contract,
              res: tuple[int, int] | None = None) -> np.ndarray:
    """Prepare (T,17,>=2) pixel keypoints as `contract` requires.

    person_scale -- AthleticsPose `utils.normalize_kpts`: subtract joint 0,
    divide x,y by max|xy|, then re-attach the confidence channel RAW (the
    released code normalizes a zero-filled third channel and overwrites it).

    image_res -- MotionBERT `datareader_h36m.read_2d`:
    `xy / res_w * 2 - [1, res_h/res_w]`, which keeps absolute image position.
    """
    xy = np.asarray(kp[:, :, :2], dtype=float)
    if contract.norm == "person_scale":
        c = xy - xy[:, 0:1, :]
        scale = np.max(np.abs(c))
        c = c / max(scale, 1e-9)
    elif contract.norm == "image_res":
        if res is None:
            raise ValueError("image_res normalization needs (res_w, res_h)")
        w, h = float(res[0]), float(res[1])
        c = np.empty_like(xy)
        c[:, :, 0] = xy[:, :, 0] / w * 2.0 - 1.0
        c[:, :, 1] = xy[:, :, 1] / w * 2.0 - h / w
    else:
        raise ValueError(f"unknown normalization {contract.norm!r}")
    conf = np.full(c.shape[:2] + (1,), float(contract.conf_value))
    return np.concatenate([c, conf], axis=-1)


def check(x: np.ndarray, contract: Contract,
          res: tuple[int, int] | None = None) -> list[str]:
    """Flags for a PREPARED input array. Empty list means it satisfies the
    contract. Never raises -- a caller reports what came back."""
    flags: list[str] = []

    if x.ndim != 3 or x.shape[1:] != (17, 3):
        flags.append(f"shape {tuple(x.shape)} is not (T, 17, 3)")
        return flags
    if not np.isfinite(x).all():
        flags.append(f"{int((~np.isfinite(x)).sum())} non-finite values")
    if x.dtype.kind != "f":
        flags.append(f"dtype {x.dtype} is not floating point")

    xy = x[:, :, :2]
    fin = np.isfinite(xy)
    if fin.any():
        if contract.norm == "person_scale":
            lim = np.nanmax(np.abs(xy[fin])) if fin.any() else 0.0
            if lim > 1.0 + 1e-6:
                flags.append(f"person_scale: max|xy| = {lim:.4f} exceeds 1.0")
        elif contract.norm == "image_res":
            if res is None:
                flags.append("image_res: no resolution given, cannot check")
            else:
                w, h = float(res[0]), float(res[1])
                xmax = float(np.nanmax(np.abs(xy[:, :, 0])))
                ymax = float(np.nanmax(np.abs(xy[:, :, 1])))
                if xmax > 1.0 + 1e-3:
                    flags.append(f"image_res: max|x| = {xmax:.4f} exceeds 1.0, "
                                 f"so x is outside the image")
                if ymax > h / w + 1e-3:
                    flags.append(f"image_res: max|y| = {ymax:.4f} exceeds "
                                 f"res_h/res_w = {h / w:.4f}")

    conf = x[:, :, 2]
    if contract.conf_value is not None:
        lo, hi = float(np.nanmin(conf)), float(np.nanmax(conf))
        if lo < 0.0 or hi > 1.0:
            flags.append(f"confidence outside [0, 1]: [{lo:.4f}, {hi:.4f}] -- "
                         f"this is the 30.07 deg MoVi defect's signature")
        target = float(contract.conf_value)
        mean = float(np.nanmean(conf))
        if abs(mean - target) > 0.05:
            flags.append(f"confidence mean {mean:.4f} is not the contract's "
                         f"{target:.4f} ({contract.conf_label})")

    allzero = int((np.abs(x[:, :, :2]).sum(axis=(1, 2)) == 0).sum())
    if allzero:
        flags.append(f"{allzero} all-zero frames -- the phase 15 D5 sentinel "
                     f"that the 50%-overlap windows smear into neighbours")
    return flags


def summarize(x: np.ndarray) -> dict:
    """Channel statistics, for the record beside the flags."""
    return {"n_frames": int(len(x)),
            "x_min": float(np.nanmin(x[:, :, 0])),
            "x_max": float(np.nanmax(x[:, :, 0])),
            "y_min": float(np.nanmin(x[:, :, 1])),
            "y_max": float(np.nanmax(x[:, :, 1])),
            "conf_mean": float(np.nanmean(x[:, :, 2])),
            "conf_min": float(np.nanmin(x[:, :, 2])),
            "conf_max": float(np.nanmax(x[:, :, 2]))}
