"""Segment inclinations against gravity. Added BESIDE the frozen angle code.

Pre-registered in `results/phase15d.md`, committed alone in e19980f.

`phase3_angles.sagittal_angles` is NOT touched and keeps producing every
pre-registered number. These functions measure the same skeleton against a
different reference direction:

    thigh inclination   hip   -> knee    against vertical
    shank inclination   knee  -> ankle   against vertical
    trunk lean          pelvis-> thorax  against vertical

all in the sagittal plane, which is the plane perpendicular to the hip-to-hip
axis. Positive is anterior.

Why: the hip angle is referenced to pelvis->thorax, and phase 15C found the
lifter does not place its own thorax on the same anatomical point in two
datasets -- CLAV to 0.750 deg on AthleticsPose, 87-144 mm away on MoVi. Gravity
is the one reference direction that means the same thing in every lab.

THE KNEE IDENTITY. `sagittal_angles` computes knee as signed_angle(thigh, shank)
= atan2(shank.fwd, shank.up) - atan2(thigh.fwd, thigh.up), so

    knee == shank inclination - thigh inclination

and that difference is invariant to rotating (fwd, up) inside the plane. So the
knee comes out identical whether `up` is the trunk axis or gravity, which is
both why the knee is immune to the thorax definition and a test this module must
pass (`tests/test_segment_angles.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_angles import H36M  # noqa: E402

SEGMENTS = {"thigh": ("hip", "knee"), "shank": ("knee", "ankle")}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def as_frames(up: np.ndarray, n: int) -> np.ndarray:
    """Broadcast a single vertical to one per frame."""
    up = np.asarray(up, dtype=float)
    if up.ndim == 1:
        up = np.tile(up, (n, 1))
    return _unit(up)


def sagittal_basis(kp: np.ndarray, up: np.ndarray):
    """(fwd, up) spanning the sagittal plane, from the hip axis and a vertical.

    The plane's normal is the hip-to-hip axis, exactly as in `sagittal_angles`.
    Two things differ from that function, both deliberate:

    * which direction inside the plane is "up" -- here the supplied vertical,
      orthogonalised against the hip axis, rather than pelvis->thorax;
    * the SIGN of forward. `sagittal_angles` uses `cross(ml, up)`, which points
      POSTERIOR in a world with x right, y anterior, z up. That basis is kept
      untouched inside the frozen function, but these new quantities are
      documented as anterior-positive, so they use `cross(up, ml)`. Verified:
      with ml = +x and up = +z, `cross(up, ml)` = +y = anterior.
    """
    u = as_frames(up, len(kp))
    ml = _unit(kp[:, H36M["r_hip"]] - kp[:, H36M["l_hip"]])
    fwd = _unit(np.cross(u, ml))          # anterior
    up2 = _unit(np.cross(ml, fwd))        # back to (approximately) the vertical
    return fwd, up2


def inclination(kp: np.ndarray, a: str, b: str, up: np.ndarray,
                from_down: bool = True) -> np.ndarray:
    """Sagittal inclination of a -> b, in degrees, anterior-positive.

    `from_down=True` measures from the DOWNWARD vertical, which is what a limb
    segment wants: a thigh hanging straight down reads 0 and one swung forward
    reads positive. `from_down=False` measures from up, which is what the trunk
    wants, since it points upward and an upright trunk should read 0.

    Both were checked against a synthetic skeleton built with known angles: a
    thigh at +30, a shank at -15 and a trunk at +10 read back exactly those
    numbers (`tests/test_segment_angles.py`).
    """
    fwd, up2 = sagittal_basis(kp, up)
    v = kp[:, H36M[b]] - kp[:, H36M[a]]
    ref = -up2 if from_down else up2
    return np.degrees(np.arctan2(np.sum(v * fwd, axis=-1),
                                 np.sum(v * ref, axis=-1)))


def wrap(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def segment_angles(kp: np.ndarray, side: str, up: np.ndarray) -> dict:
    """thigh, shank and trunk inclinations, plus the knee rebuilt from them.

    `knee_from_segments` exists to be checked against
    `sagittal_angles(kp, side)["knee"]`; it is not a new measurement.
    """
    thigh = inclination(kp, f"{side}_hip", f"{side}_knee", up, from_down=True)
    shank = inclination(kp, f"{side}_knee", f"{side}_ankle", up, from_down=True)
    trunk = inclination(kp, "pelvis", "thorax", up, from_down=False)
    return {"thigh": thigh, "shank": shank, "trunk": trunk,
            "knee_from_segments": wrap(shank - thigh)}


# --------------------------------------------------------------------------
# What the level-camera assumption costs. Reported, never used in the estimate.
def camera_tilt(R_cv: np.ndarray, world_up=(0.0, 0.0, 1.0)) -> dict:
    """Pitch, roll and total tilt of a camera, in degrees.

    `R_cv` maps world -> camera for column vectors, so its ROWS are the camera's
    x, y and z axes expressed in world coordinates.

      pitch  elevation of the optical axis (camera z) above horizontal
      roll   elevation of the camera x axis above horizontal, i.e. rotation
             about the optical axis
      tilt   total angle between the image's "up" (-camera y) and world up --
             this is what the level-camera assumption actually gets wrong
    """
    e = _unit(np.asarray(world_up, float)[None])[0]
    cam_x, cam_y, cam_z = R_cv[0], R_cv[1], R_cv[2]
    img_up = -cam_y
    return {
        "pitch_deg": float(np.degrees(np.arcsin(np.clip(cam_z @ e, -1, 1)))),
        "roll_deg": float(np.degrees(np.arcsin(np.clip(cam_x @ e, -1, 1)))),
        "tilt_deg": float(np.degrees(np.arccos(np.clip(
            img_up @ e / (np.linalg.norm(img_up) * np.linalg.norm(e)),
            -1, 1)))),
    }


def level_camera_error(kp_ref: np.ndarray, R_cv: np.ndarray,
                       world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Per-frame additive error, in degrees, from assuming the camera is level.

    An inclination is measured in the sagittal plane against an assumed "up".
    If the assumed up and the true up differ, every inclination in that frame is
    offset by the SIGNED ANGLE BETWEEN THEM INSIDE THE PLANE -- the out-of-plane
    part does not affect a sagittal angle. So this is the exact additive term,
    not an approximation.

    `kp_ref` supplies the hip axis (the plane's normal) in the reference frame;
    `R_cv` carries the image up into that same frame.
    """
    e = _unit(np.asarray(world_up, float)[None])[0]
    img_up_world = -R_cv[1]
    fwd_true, up_true = sagittal_basis(kp_ref, e)
    _fa, up_assumed = sagittal_basis(kp_ref, img_up_world)
    # signed angle from the assumed vertical to the true one, inside the plane
    return np.degrees(np.arctan2(np.sum(up_assumed * fwd_true, axis=-1),
                                 np.sum(up_assumed * up_true, axis=-1)))
