"""Phase 12 scan invariants.

The gate is measured on scale-invariant ratios, so the two things that must hold
are that lengths are recovered correctly and that ratios really are invariant.
The third is that the scale reconstruction mirrors the upstream normalisation it
is undoing, because an absolute-mm figure derived from a mismatched scale would
look plausible and be wrong.
"""
import numpy as np
import pytest

from phase12_scan import (
    ANATOMICAL,
    RATIOS,
    SEGMENTS,
    assert_no_subject_leakage,
    bone_lengths,
    cv,
    norm_scale_of,
    ratios_of,
    scan,
    subject_bootstrap_ci,
)
from phase3_angles import H36M


def _skeleton(n=50, thigh=400.0, shank=380.0, pelvis=200.0, trunk=500.0, scale=1.0):
    """A still skeleton with segment lengths known by construction, in mm."""
    kp = np.zeros((n, 17, 3))
    kp[:, H36M["pelvis"]] = [0.0, 0.0, 0.0]
    kp[:, H36M["l_hip"]] = [-pelvis / 2, 0.0, 0.0]
    kp[:, H36M["r_hip"]] = [+pelvis / 2, 0.0, 0.0]
    kp[:, H36M["l_knee"]] = [-pelvis / 2, -thigh, 0.0]
    kp[:, H36M["r_knee"]] = [+pelvis / 2, -thigh, 0.0]
    kp[:, H36M["l_ankle"]] = [-pelvis / 2, -thigh - shank, 0.0]
    kp[:, H36M["r_ankle"]] = [+pelvis / 2, -thigh - shank, 0.0]
    kp[:, H36M["thorax"]] = [0.0, trunk, 0.0]
    return kp * scale


def test_recovers_known_segment_lengths():
    got = scan(bone_lengths(_skeleton()))
    for name, want in [("thigh_l", 400.0), ("thigh_r", 400.0),
                       ("shank_l", 380.0), ("shank_r", 380.0),
                       ("pelvis", 200.0), ("trunk", 500.0)]:
        assert abs(got[name] - want) < 1e-9, f"{name}: {got[name]} != {want}"


@pytest.mark.parametrize("scale", [0.001, 0.1, 7.0, 1000.0])
def test_ratios_are_scale_invariant_but_lengths_are_not(scale):
    """The reason the gate is on ratios: the lifter output carries no scale.

    Ratios must be identical under any scale; absolute lengths must not be, or
    the test would be asserting the wrong thing.
    """
    base = scan(bone_lengths(_skeleton()))
    other = scan(bone_lengths(_skeleton(scale=scale)))
    r_base, r_other = ratios_of(base), ratios_of(other)
    for k in RATIOS:
        assert abs(r_base[k] - r_other[k]) < 1e-9, f"ratio {k} moved with scale"
    assert not np.isclose(base["thigh_l"], other["thigh_l"]), (
        "absolute length did not change with scale -- the fixture is wrong")


def test_scan_median_survives_a_corrupted_frame():
    """One mis-detected frame must not move the body model."""
    kp = _skeleton(n=51)
    clean = scan(bone_lengths(kp))
    kp[25, H36M["l_knee"]] = [5000.0, 5000.0, 5000.0]
    dirty = scan(bone_lengths(kp))
    assert abs(dirty["thigh_l"] - clean["thigh_l"]) < 1e-9
    # and the per-frame series does see it, so the corruption was real
    assert bone_lengths(kp)["thigh_l"].max() > 1000.0


def test_norm_scale_mirrors_the_upstream_normalisation():
    """Absolute mm is reconstructed by undoing normalize_kpts; it must match.

    If these drift apart, every millimetre figure in the report is silently
    wrong by a constant factor while still looking anatomically plausible.
    """
    from athleticspose.utils import normalize_kpts

    rng = np.random.default_rng(0)
    det = rng.uniform(-500, 500, size=(40, 17, 3))
    det[:, :, 2] = rng.uniform(0, 1, size=(40, 17))        # confidence channel
    coords = np.concatenate([det[:, :, :2], np.zeros_like(det[:, :, :1])], axis=-1)
    _, upstream = normalize_kpts(coords)
    assert abs(norm_scale_of(det) - float(upstream)) < 1e-12


def test_foot_is_declared_unavailable_not_estimated():
    """H36M-17 has no toe keypoint, so no foot segment may be estimated."""
    assert not any("foot" in k for k in SEGMENTS), (
        "a foot segment appeared in the estimated set; H36M-17 cannot support it")
    assert not any("foot" in k for k in RATIOS)
    # but mocap can measure it, so ground truth must still be reported
    assert "foot_l" in ANATOMICAL and "foot_r" in ANATOMICAL


def test_bone_lengths_rejects_the_wrong_skeleton():
    with pytest.raises(ValueError):
        bone_lengths(np.zeros((10, 84, 3)))     # raw mocap, not H36M
    with pytest.raises(ValueError):
        bone_lengths(np.zeros((17, 3)))


def test_leakage_check_actually_catches_a_shared_subject():
    assert_no_subject_leakage([["S07", "S09"], ["S11"]])      # clean
    with pytest.raises(AssertionError):
        assert_no_subject_leakage([["S07", "S09"], ["S09"]])  # S09 in both


def test_bootstrap_resamples_subjects_and_needs_at_least_two():
    lo, hi = subject_bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert lo < 3.0 < hi
    assert np.isnan(subject_bootstrap_ci([1.0])[0]), (
        "a CI over one subject must be nan, not a point")


def test_cv_is_a_percentage_and_undefined_for_one_value():
    assert abs(cv([100.0, 100.0, 100.0])) < 1e-12
    assert abs(cv([90.0, 110.0]) - 100.0 * np.std([90.0, 110.0], ddof=1) / 100.0) < 1e-9
    assert np.isnan(cv([5.0]))


def test_h36m_mapping_indices_are_inside_the_84_marker_set():
    from athleticspose.statics.joints import joint_index_h36m_to_mocap

    for joint, idx in joint_index_h36m_to_mocap.items():
        assert 0 <= joint < 17
        assert idx, f"H36M joint {joint} maps to no marker"
        assert all(0 <= i < 84 for i in idx), f"joint {joint} indexes outside 84"
