"""Phase 14 invariants.

Two of these are identities, not approximations: if the per-segment
decomposition is real, the thigh's direction error IS the hip angle error, and
the knee angle error IS the difference of the shank and thigh errors. They are
what separates a decomposition from a relabelling, and they are asserted before
any confidence is fitted.

The third guards the rule the whole phase rests on: every confidence input must
be computable from the video alone, which in practice means it must not change
when the image scale or the estimate's arbitrary scale changes.
"""
import numpy as np
import pytest

from phase3_angles import sagittal_angles
from phase14_confidence import (
    FEATURES,
    SEGMENTS,
    body_frame,
    features_from_video,
    realized_errors,
    seg_angle,
    wrap,
)


@pytest.fixture
def pair():
    """Two unrelated plausible skeletons: an 'estimate' and a 'ground truth'."""
    rng = np.random.default_rng(14)
    gt = rng.normal(size=(120, 17, 3)) * 100.0
    est = gt + rng.normal(size=(120, 17, 3)) * 12.0
    return est, gt


def test_thigh_segment_error_is_exactly_the_hip_angle_error(pair):
    """hip_flex = wrap(seg_angle(thigh) - 180), so the constant cancels."""
    est, gt = pair
    errs = realized_errors(est, gt)
    for side, name in (("l", "thigh_l"), ("r", "thigh_r")):
        hip_err = np.abs(wrap(sagittal_angles(est, side)["hip"]
                              - sagittal_angles(gt, side)["hip"]))
        assert np.allclose(errs[name], hip_err, atol=1e-9), side


def test_knee_angle_error_is_the_shank_error_minus_the_thigh_error(pair):
    """A knee angle that looks fine because two segment errors cancelled is the
    case this decomposition exists to make visible."""
    est, gt = pair
    for side in ("l", "r"):
        thigh = wrap(seg_angle(est, f"{side}_hip", f"{side}_knee")
                     - seg_angle(gt, f"{side}_hip", f"{side}_knee"))
        shank = wrap(seg_angle(est, f"{side}_knee", f"{side}_ankle")
                     - seg_angle(gt, f"{side}_knee", f"{side}_ankle"))
        knee = wrap(sagittal_angles(est, side)["knee"]
                    - sagittal_angles(gt, side)["knee"])
        assert np.allclose(wrap(knee - (shank - thigh)), 0.0, atol=1e-9), side


def test_pelvis_and_trunk_cannot_have_a_sagittal_error(pair):
    """They define the frame, so they have no component in it. This is why they
    are reported as not measurable rather than estimated."""
    est, _ = pair
    fwd, up = body_frame(est)
    pelvis_dir = est[:, 1] - est[:, 4]              # r_hip - l_hip
    trunk_dir = est[:, 8] - est[:, 0]               # thorax - pelvis
    pelvis_dir /= np.linalg.norm(pelvis_dir, axis=-1, keepdims=True)
    trunk_dir /= np.linalg.norm(trunk_dir, axis=-1, keepdims=True)
    # the pelvis has no sagittal component at all
    assert np.allclose(np.sum(pelvis_dir * fwd, axis=-1), 0.0, atol=1e-9)
    assert np.allclose(np.sum(pelvis_dir * up, axis=-1), 0.0, atol=1e-9)
    # and the trunk is the vertical, so its sagittal angle is identically zero
    assert np.allclose(np.sum(trunk_dir * fwd, axis=-1), 0.0, atol=1e-9)
    assert set(SEGMENTS) == {"thigh_l", "thigh_r", "shank_l", "shank_r"}


def test_every_feature_is_invariant_to_both_arbitrary_scales():
    """The image scale is unknown and the lifter's output is scale-free, so a
    feature that moves when either is rescaled is reading something the clip
    does not actually determine."""
    rng = np.random.default_rng(7)
    det = np.concatenate([rng.normal(size=(60, 17, 2)) * 50.0 + 300.0,
                          rng.uniform(0.3, 1.0, size=(60, 17, 1))], axis=-1)
    est = rng.normal(size=(60, 17, 3)) * 100.0
    blend = rng.uniform(0.5, 1.0, size=60)

    a, near_a, _ = features_from_video(det, est, blend)
    scaled = det.copy()
    scaled[:, :, :2] *= 5.0
    b, near_b, _ = features_from_video(scaled, est * 3.0, blend)

    assert near_a == near_b
    for seg in SEGMENTS:
        for f in FEATURES:
            assert np.allclose(a[seg][f], b[seg][f], atol=1e-9), (seg, f)


def test_features_never_see_ground_truth():
    """Enforced by the signature, which is the point of the split: adding a
    ground-truth feature would mean changing this call."""
    import inspect

    sig = inspect.signature(features_from_video)
    assert list(sig.parameters) == ["det2d", "est", "blend"]
    sig = inspect.signature(realized_errors)
    assert list(sig.parameters) == ["est", "gt"]
