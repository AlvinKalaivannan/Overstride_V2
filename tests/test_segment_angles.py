"""Segment inclinations: the knee identity, the sign convention, known values.

Pre-registered in `results/phase15d.md`. The first test is the one that matters:
if `shank - thigh` does not reproduce the frozen `sagittal_angles` knee, the new
definitions are wrong and no result built on them means anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from phase15d_segments import (camera_tilt, inclination,  # noqa: E402
                               level_camera_error, segment_angles, wrap)
from phase3_angles import H36M, sagittal_angles  # noqa: E402

UP = np.array([0.0, 0.0, 1.0])


def _skeleton(n=40, thigh_deg=0.0, shank_deg=0.0, trunk_deg=0.0, seed=0):
    """A skeleton with KNOWN sagittal inclinations, built in a z-up world.

    x is to the subject's right, y is anterior, z is up. The hip axis is along
    x, so the sagittal plane is y-z and `fwd = cross(ml, up)` points along +y.
    Each segment is placed at its requested angle from straight-down, rotated
    toward +y, which is what the sign convention calls positive.
    """
    rng = np.random.default_rng(seed)
    kp = np.zeros((n, 17, 3))
    hw = 100.0
    kp[:, H36M["l_hip"]] = [-hw / 2, 0.0, 900.0]
    kp[:, H36M["r_hip"]] = [+hw / 2, 0.0, 900.0]
    kp[:, H36M["pelvis"]] = [0.0, 0.0, 900.0]

    def place(origin, length, deg):
        t = np.radians(deg)
        # straight down is -z; positive deg rotates toward +y
        return origin + np.array([0.0, length * np.sin(t), -length * np.cos(t)])

    for side in ("l", "r"):
        hip = kp[:, H36M[f"{side}_hip"]][0]
        knee = place(hip, 400.0, thigh_deg)
        ankle = place(knee, 380.0, shank_deg)
        kp[:, H36M[f"{side}_knee"]] = knee
        kp[:, H36M[f"{side}_ankle"]] = ankle
    # trunk goes UP from the pelvis, so a positive lean is +y with +z
    t = np.radians(trunk_deg)
    kp[:, H36M["thorax"]] = kp[:, H36M["pelvis"]][0] + np.array(
        [0.0, 450.0 * np.sin(t), 450.0 * np.cos(t)])
    kp[:, H36M["spine"]] = 0.5 * (kp[:, H36M["pelvis"]]
                                  + kp[:, H36M["thorax"]])
    # the joints these tests do not use still need to be finite
    for name in ("neck", "head", "l_shoulder", "r_shoulder", "l_elbow",
                 "r_elbow", "l_wrist", "r_wrist"):
        kp[:, H36M[name]] = kp[:, H36M["thorax"]] + rng.normal(0, 20, (n, 3))
    return kp


# ------------------------------------------------------- the knee identity ---
@pytest.mark.parametrize("thigh_deg,shank_deg", [
    (0.0, 0.0), (20.0, -10.0), (-15.0, 35.0), (45.0, 5.0), (-30.0, -40.0)])
def test_shank_minus_thigh_reproduces_the_frozen_knee_angle(thigh_deg, shank_deg):
    """knee == shank inclination - thigh inclination, to float tolerance.

    Asserted against `sagittal_angles`, which is frozen and produces every
    pre-registered number. Note the ORDER: shank minus thigh, not thigh minus
    shank.
    """
    kp = _skeleton(thigh_deg=thigh_deg, shank_deg=shank_deg, trunk_deg=12.0)
    for side in ("l", "r"):
        frozen = sagittal_angles(kp, side)["knee"]
        rebuilt = segment_angles(kp, side, UP)["knee_from_segments"]
        assert np.allclose(wrap(frozen - rebuilt), 0.0, atol=1e-6), (
            f"side {side}: frozen {frozen[0]:.6f} vs rebuilt "
            f"{rebuilt[0]:.6f}")


def test_the_knee_identity_holds_for_ANY_vertical():
    """The identity must not depend on which direction is called up.

    That invariance is the structural reason the knee is immune to the thorax
    definition, so it is asserted directly: a tilted, even an absurd, vertical
    must leave the rebuilt knee unchanged.
    """
    kp = _skeleton(thigh_deg=25.0, shank_deg=-18.0, trunk_deg=8.0)
    ref = segment_angles(kp, "l", UP)["knee_from_segments"]
    for up in ([0.0, 0.3, 1.0], [0.0, -0.5, 1.0], [0.0, 1.0, 0.2]):
        got = segment_angles(kp, "l", np.array(up))["knee_from_segments"]
        assert np.allclose(wrap(ref - got), 0.0, atol=1e-6), f"up={up}"


# ------------------------------------------------------ the sign convention ---
def test_zero_means_straight_down_the_vertical():
    kp = _skeleton(thigh_deg=0.0, shank_deg=0.0, trunk_deg=0.0)
    a = segment_angles(kp, "l", UP)
    assert np.allclose(a["thigh"], 0.0, atol=1e-6)
    assert np.allclose(a["shank"], 0.0, atol=1e-6)
    assert np.allclose(a["trunk"], 0.0, atol=1e-6)


@pytest.mark.parametrize("deg", [10.0, 25.0, 40.0, -10.0, -35.0])
def test_inclination_recovers_the_angle_it_was_built_with(deg):
    """A synthetic skeleton with known inclinations reads them back."""
    kp = _skeleton(thigh_deg=deg, shank_deg=-deg, trunk_deg=deg / 2.0)
    a = segment_angles(kp, "l", UP)
    assert np.allclose(a["thigh"], deg, atol=1e-6)
    assert np.allclose(a["shank"], -deg, atol=1e-6)
    assert np.allclose(a["trunk"], deg / 2.0, atol=1e-6)


def test_positive_is_anterior():
    """Forward of vertical is positive, which is the documented convention."""
    kp = _skeleton(thigh_deg=30.0)
    thigh = segment_angles(kp, "l", UP)["thigh"]
    knee_y = kp[0, H36M["l_knee"], 1]
    hip_y = kp[0, H36M["l_hip"], 1]
    assert knee_y > hip_y, "the test skeleton should swing the knee anterior"
    assert thigh[0] > 0, "an anterior segment must read positive"


def test_the_hip_axis_sets_the_plane_so_a_mirrored_subject_flips_sign():
    """Swapping the hips mirrors the sagittal plane's forward direction."""
    kp = _skeleton(thigh_deg=22.0)
    a = segment_angles(kp, "l", UP)["thigh"]
    kp2 = kp.copy()
    kp2[:, H36M["l_hip"]], kp2[:, H36M["r_hip"]] = (
        kp[:, H36M["r_hip"]].copy(), kp[:, H36M["l_hip"]].copy())
    b = segment_angles(kp2, "l", UP)["thigh"]
    assert np.allclose(a, -b, atol=1e-6)


# ------------------------------------------------ the level-camera accounting ---
def _side_on_camera(roll_deg: float = 0.0) -> np.ndarray:
    """A level side-on camera, optionally rolled about its optical axis.

    Rows of R_cv are the camera axes in world. Side-on is the geometry this
    project prescribes and the one MoVi's PG1 has, and it is also the only
    geometry in which a camera roll lands inside the SAGITTAL plane -- for a
    camera looking along the direction of travel, a roll tilts the image
    vertical in the frontal plane instead and costs a sagittal angle nothing.
    """
    cam_z = np.array([1.0, 0.0, 0.0])        # optical axis along +x (side-on)
    cam_y = np.array([0.0, 0.0, -1.0])       # image down = world -z
    cam_x = np.cross(cam_y, cam_z)           # right-handed
    t = np.radians(roll_deg)
    cx = np.cos(t) * cam_x + np.sin(t) * cam_y
    cy = -np.sin(t) * cam_x + np.cos(t) * cam_y
    return np.vstack([cx, cy, cam_z])


def test_a_level_camera_costs_nothing():
    """With the image vertical equal to gravity the error must be exactly 0."""
    R = _side_on_camera(0.0)
    assert abs(camera_tilt(R)["tilt_deg"]) < 1e-6
    kp = _skeleton(thigh_deg=15.0)
    assert np.allclose(level_camera_error(kp, R), 0.0, atol=1e-6)


@pytest.mark.parametrize("roll", [3.0, 7.0, -5.0])
def test_a_rolled_side_on_camera_costs_exactly_its_roll(roll):
    """For a side-on camera the roll tilts the image vertical inside the
    sagittal plane, so it is an exact additive offset on every inclination."""
    R = _side_on_camera(roll)
    kp = _skeleton(thigh_deg=15.0)
    err = level_camera_error(kp, R)
    assert np.allclose(np.abs(err), abs(roll), atol=1e-6), (
        f"roll {roll} -> error {err[0]:.4f}")
    assert abs(camera_tilt(R)["tilt_deg"] - abs(roll)) < 1e-6


def test_the_trunk_referenced_hip_decomposes_into_thigh_plus_trunk():
    """hip (trunk-referenced) == thigh inclination + trunk lean, exactly.

    This is the identity that makes phase 15D worth doing: the frozen hip angle
    is measured from the trunk, so it necessarily carries BOTH the limb's
    orientation and the trunk's. Referencing each to gravity separates them, and
    the two parts must add back to the original.

    Verified on real data in `results/phase15d.json`: the residual is ~1e-15 on
    all 12 MoVi subject-arm rows. Asserted here on a synthetic skeleton so the
    identity is pinned rather than observed.
    """
    for thigh_deg, trunk_deg in ((0.0, 0.0), (25.0, 10.0), (-15.0, 20.0),
                                 (40.0, -8.0), (-30.0, -25.0)):
        kp = _skeleton(thigh_deg=thigh_deg, shank_deg=-12.0,
                       trunk_deg=trunk_deg)
        for side in ("l", "r"):
            seg = segment_angles(kp, side, UP)
            hip_trunk_ref = sagittal_angles(kp, side)["hip"]
            got = wrap(seg["thigh"] + seg["trunk"])
            assert np.allclose(wrap(hip_trunk_ref - got), 0.0, atol=1e-6), (
                f"thigh {thigh_deg} trunk {trunk_deg} side {side}: "
                f"hip {hip_trunk_ref[0]:.6f} vs thigh+trunk {got[0]:.6f}")
