"""Phase 15 must refuse an angle error for a camera it cannot align.

The frozen pre-registration (`results/phase15.md`, commit 64fb0bc) says a CP
clip carries an MAE "if their synchronisation to mocap can be verified from the
dataset's own documentation ... if it cannot, they are reported as traces with
no error figure".

It cannot be verified. `Camera Parameters.tar` covers PG1 and PG2 only, and
`flags30` is indexed to the PG recording: CP is 1920x1080 at 30.009 fps with
4,369 frames for Subject_1 against PG's 800x600 at 30.000 fps with 3,441, so CP
frame k is not the same instant as PG frame k. Scoring CP on the PG window gave
r = -0.077 hip and +0.022 knee -- no relationship at all.

These tests pin the refusal structurally, so the 41.863 deg / 61.204 deg figures
can never come back by someone re-enabling a flag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "data" / "athleticspose" / "repo"))


def _movi():
    try:
        import phase15_movi
    except Exception as e:                      # torch/cv2 absent in some envs
        pytest.skip(f"phase15_movi not importable: {e}")
    return phase15_movi


def test_only_the_point_grey_cameras_are_marked_calibrated():
    """CP must be declared uncalibrated; that flag is what drives the refusal."""
    m = _movi()
    assert m.CAMERAS["PG1"]["calibrated"] is True
    assert m.CAMERAS["PG2"]["calibrated"] is True
    for name, spec in m.CAMERAS.items():
        if name.startswith("CP"):
            assert spec["calibrated"] is False, (
                f"{name} is a hand-held phone camera and MoVi ships no "
                f"calibration for it")


def test_run_clip_refuses_before_it_can_decode_an_uncalibrated_camera():
    """The refusal must come BEFORE any scoring, not be filtered afterwards.

    Proven by handing `run_clip` a video path that does not exist: if it
    refused first, it returns the refusal record; if it tried to decode, it
    would raise SystemExit from `decode_range`.
    """
    m = _movi()
    mats = sorted((m.MOCAP).glob("F_v3d_*.mat"))
    if not mats:
        pytest.skip("MoVi round F mocap not on disk")

    missing = m.MOVI / "this_video_does_not_exist.mp4"
    assert not missing.exists()

    row = m.run_clip("Subject_1", "CP1", mats[0], missing,
                     det_model=None, lift_model=None, max_frames=0)

    assert row is not None
    assert row["mae_refused"] is True
    assert row["err"] is None
    assert "calibration" in row["mae_refused_because"]
    assert "flags30" in row["mae_refused_because"]


def test_a_refused_clip_can_never_reach_an_aggregate_mean():
    """`aggregate` must drop err=None rows rather than average around them."""
    m = _movi()
    good = {"subject": "Subject_1", "camera": "PG1",
            "err": {"near": {"hip": 10.0, "knee": 8.0, "mean": 9.0},
                    "far": {"hip": 11.0, "knee": 10.0, "mean": 10.5}}}
    refused = {"subject": "Subject_1", "camera": "CP1", "mae_refused": True,
               "err": None}

    agg = m.aggregate([good, refused])

    assert "PG1" in agg
    assert "CP1" not in agg, ("a camera with no valid clip must not appear in "
                              "the results at all")


def test_the_committed_result_carries_no_cp_error_figure():
    """The published JSON must not contain an MAE for an uncalibrated camera."""
    f = REPO / "results" / "phase15_movi.json"
    if not f.exists():
        pytest.skip("results/phase15_movi.json not present")
    d = json.loads(f.read_text(encoding="utf-8"))

    for cam in d["by_camera"]:
        assert not cam.startswith("CP"), (
            f"{cam} has an error figure in by_camera; the frozen clause says "
            f"an uncalibrated camera is traces only")

    for clip in d["clips"]:
        if clip["camera"].startswith("CP"):
            assert clip.get("mae_refused") is True
            assert clip.get("err") is None


def test_the_gate_reports_a_failing_check_whatever_the_bar_is():
    """The gate must surface a failed check, independent of which bar is set.

    An earlier version of this test asserted that `Subject_11` fails check 1.
    That was a consequence of the bar as originally pre-registered, and the D7
    re-registration made it false -- so the test broke for the right reason and
    is rewritten to pin the MECHANISM rather than one subject's status under one
    bar. See `results/phase15.md` D6/D7.
    """
    m = _movi()
    mats = sorted(m.MOCAP.glob("F_v3d_*.mat"))
    if not mats:
        pytest.skip("MoVi round F mocap not on disk")

    ok, chk = m.preregistered_checks(mats[0])
    # whatever the verdict, the shape of the answer is the contract
    assert isinstance(ok, bool)
    assert isinstance(chk, dict)
    assert "failed_checks" in chk
    assert ok == (not chk["failed_checks"]), (
        "the boolean must agree with the list of failures")


def test_run_clip_refuses_any_subject_whose_checks_fail(monkeypatch):
    """Forced failure: no MAE, no decoding, and the reason carried through.

    Uses a stubbed gate so the test does not depend on any real subject failing
    -- which is exactly the coupling that broke the previous version.
    """
    m = _movi()
    mats = sorted(m.MOCAP.glob("F_v3d_*.mat"))
    if not mats:
        pytest.skip("MoVi round F mocap not on disk")

    monkeypatch.setattr(
        m, "preregistered_checks",
        lambda mat: (False, {"failed_checks": {"check1_units": "1 (stubbed)"},
                             "gap_sentinel": None}))

    # a video path that does not exist: if the gate did NOT run first, decoding
    # would raise SystemExit instead of returning a refusal record
    missing = m.MOVI / "no_such_video.avi"
    assert not missing.exists()

    row = m.run_clip("Subject_X", "PG1", mats[0], missing,
                     det_model=None, lift_model=None, max_frames=0)

    assert row["mae_refused"] is True
    assert row["err"] is None
    assert "check1_units" in row["failed_checks"]
    assert "pre-registered check" in row["mae_refused_because"]


def test_the_committed_result_excludes_every_failing_subject():
    f = REPO / "results" / "phase15_movi.json"
    if not f.exists():
        pytest.skip("results/phase15_movi.json not present")
    d = json.loads(f.read_text(encoding="utf-8"))

    excluded = {e["subject"] for e in d.get("no_error_figure", [])
                if e.get("failed_checks")}
    for cam, block in d["by_camera"].items():
        for limb in ("near", "far"):
            scored = set(block["limbs"][limb]["per_subject"])
            assert not (scored & excluded), (
                f"{cam}/{limb} averages a subject that failed a "
                f"pre-registered check: {scored & excluded}")
