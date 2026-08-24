"""Invariants of the angle extraction. These are the things that would silently
corrupt every downstream number if they broke."""
import numpy as np
import pytest
from phase3_angles import H36M, sagittal_angles


def _synthetic_knee_sweep(n=200, scale=1.0):
    """A leg with the knee flexing 0->90 deg, built by construction."""
    kp = np.zeros((n, 17, 3))
    kp[:, H36M["pelvis"]] = [0, 0, 0]
    kp[:, H36M["thorax"]] = [0, 0, 500]
    kp[:, H36M["l_hip"]] = [-100, 0, 0]
    kp[:, H36M["r_hip"]] = [100, 0, 0]
    kp[:, H36M["r_knee"]] = [100, 0, -400]
    ang = np.linspace(0, 90, n)
    kp[:, H36M["r_ankle"], 0] = 100
    kp[:, H36M["r_ankle"], 1] = -400 * np.sin(np.radians(ang))
    kp[:, H36M["r_ankle"], 2] = -400 - 400 * np.cos(np.radians(ang))
    return kp * scale, ang


def test_recovers_a_known_knee_sweep():
    kp, truth = _synthetic_knee_sweep()
    got = np.abs(sagittal_angles(kp, "r")["knee"])
    assert np.max(np.abs(got - truth)) < 0.5


@pytest.mark.parametrize("scale", [0.001, 0.1, 7.0, 1000.0])
def test_angles_are_scale_invariant(scale):
    """The video tool lifts WITHOUT the ground-truth-derived denormalisation
    phase 3 used. That is only sound because angles do not depend on scale."""
    a, _ = _synthetic_knee_sweep(scale=1.0)
    b, _ = _synthetic_knee_sweep(scale=scale)
    for joint in ("hip", "knee"):
        x, y = sagittal_angles(a, "r")[joint], sagittal_angles(b, "r")[joint]
        assert np.max(np.abs(x - y)) < 1e-6, f"{joint} changed with scale {scale}"


def test_ankle_is_nan_not_fabricated():
    """H36M-17 has no toe keypoint, so ankle dorsiflexion is unavailable. It must
    come back NaN rather than a plausible-looking wrong number."""
    kp, _ = _synthetic_knee_sweep()
    assert np.all(np.isnan(sagittal_angles(kp, "r")["ankle"]))


def test_angles_ignore_global_rotation():
    """Angles are defined in the body's own frame, so a rotated camera must not
    change them -- this is what lets a hand-held clip be usable at all."""
    kp, _ = _synthetic_knee_sweep()
    th = np.radians(37.0)
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    rot = kp @ R.T
    for joint in ("hip", "knee"):
        x, y = sagittal_angles(kp, "r")[joint], sagittal_angles(rot, "r")[joint]
        assert np.max(np.abs(x - y)) < 1e-6, f"{joint} changed under rotation"
