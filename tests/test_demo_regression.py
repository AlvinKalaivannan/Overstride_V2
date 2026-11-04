"""M1: the demo's processing path must reproduce the sealed test's numbers.

`docs/phase16-demo-design.md` section 6. The demo pins its components precisely
so that its error budget is the sealed test's measured behaviour. **That claim
is only worth anything if the demo path actually produces the sealed numbers**,
so this test runs `overstride.analyze` end to end on a spent MoVi subject and
compares against `results/phase15_sealed.json`.

`Subject_17` is the primary case: the best-behaved sealed subject (knee bias
+0.562 deg, thigh bias -0.052 deg), so a regression shows as a clear break
rather than hiding inside a subject that is already poor. `Subject_2` -- the
worst, knee bias +12.106 -- guards the other end.

Every MoVi test here is marked `local` and SKIPS when `data/movi/` is absent,
so a clean checkout and CI still pass. **No MoVi pixel data is written, printed
or asserted on** -- only angles.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from overstride.analyze import analyze_frames, reference_angles  # noqa: E402
from overstride.preprocess import (SHORT_SIDE,  # noqa: E402
                                   detector_working_resolution,
                                   resize_short_side)

SEALED = REPO / "results" / "phase15_sealed.json"
MOVI = REPO / "data" / "movi" / "raw" / "round_f"
CAM = "PG1"
MOCAP_PER_VIDEO = 4

# Committed in results/phase15_sealed.json, quoted in phase15_sealed.md.
#
# Keyed by ANATOMICAL SIDE, not by near/far. The sealed test used the camera to
# decide which limb was nearer; the demo has no camera and CANNOT -- it agrees
# with the camera on only 9 of 11 MoVi subjects, with no signal separating the
# failures (results/phase16a_nearlimb.json). So the demo reports left/right and
# this test asserts left/right, mapping the sealed near/far through the sealed
# result's own `near_limb` field.
EXPECTED = {
    # sealed near_limb is "l" for both, so sealed "near" == LEFT here
    "Subject_17": {"side": "l", "knee_from_segments": 5.130, "thigh": 3.274,
                   "sealed_column": "near",
                   "n_video": 221, "n_detected": 221},
    "Subject_2": {"side": "l", "knee_from_segments": 12.964, "thigh": 8.032,
                  "sealed_column": "near",
                  "n_video": 157, "n_detected": 157},
}
ATOL = 1e-3

movi_only = pytest.mark.skipif(
    not (MOVI / f"F_{CAM}_Subject_17_L.avi").exists(),
    reason="MoVi not on disk; this is a local-only regression test")


# ----------------------------------------------------------- resize [A1] ----

@pytest.mark.parametrize("w,h", [(800, 600), (1280, 720), (1920, 1080),
                                 (640, 480), (1080, 1920)])
def test_resize_sets_the_detectors_working_resolution_to_600(w, h):
    """[A1] The resize must control `min_size`, which IS the short side.

    This is the whole point of the step: the sealed test ran at short side 600,
    so every clip must reach the detector at short side 600.
    """
    frames = [np.zeros((h, w, 3), np.uint8) for _ in range(3)]
    out = resize_short_side(frames)
    assert detector_working_resolution(out) == SHORT_SIDE
    assert min(out[0].shape[:2]) == SHORT_SIDE
    # aspect ratio preserved to within a pixel of rounding
    assert abs((out[0].shape[1] / out[0].shape[0]) - (w / h)) < 0.01


def test_resize_is_identity_on_the_sealed_condition():
    """800x600 is already short side 600, so the step must not touch it.

    This is why the regression test below is a valid check that adding the
    resize changed no measured number.
    """
    frames = [np.random.default_rng(0).integers(0, 255, (600, 800, 3),
                                                dtype=np.uint8)]
    out = resize_short_side(frames)
    assert out[0].shape == frames[0].shape
    assert np.array_equal(out[0], frames[0])


def test_resize_handles_an_empty_list():
    assert resize_short_side([]) == []


# -------------------------------------------- the section 6 regression ------

def _sealed(subject: str) -> dict:
    d = json.loads(SEALED.read_text(encoding="utf-8"))
    return d["per_subject"][subject]


def _movi_reference(subject: str):
    """The subject's jogging window: frames, fps, reference angles, mask."""
    from phase15_checks import angle_mask, load_v3d, to_h36m
    from phase15_movi import MOCAP, decode_range, jogging_window

    v = load_v3d(MOCAP / f"F_v3d_{subject}.mat")
    _ji, f30, f120 = jogging_window(v)
    ref_all = to_h36m(v)
    frames, fps, _m = decode_range(MOVI / f"F_{CAM}_{subject}_L.avi",
                                   int(f30[0]), int(f30[1]))
    mk = int(f120[0]) + MOCAP_PER_VIDEO * np.arange(len(frames))
    mk = mk[mk < ref_all.shape[0]]
    frames = list(frames)[:len(mk)]
    ref = ref_all[mk]
    return frames, float(fps), ref, angle_mask


@movi_only
@pytest.mark.parametrize("subject", ["Subject_17", "Subject_2"])
def test_demo_path_reproduces_the_sealed_numbers(subject):
    """The demo path, end to end, must give the committed sealed MAEs."""
    from phase15_checks import load_v3d  # noqa: F401  (import guard)
    from phase3_infer import load_model
    from phase7_inference_fix import CKPT_FOR

    want = EXPECTED[subject]
    frames, fps, ref, angle_mask = _movi_reference(subject)
    lifter = load_model(CKPT_FOR["coco"], "base")
    assert lifter is not None, "checkpoint missing"

    res = analyze_frames(frames, fps, lifter)

    # the resize must be a no-op here: PG1 is 800x600
    assert res.detector_min_size == SHORT_SIDE
    assert res.resized_from is None
    assert res.n_frames == want["n_video"], "frame count changed"
    assert int(res.detected.sum()) == want["n_detected"], "detection changed"

    ref_ang = reference_angles(ref, np.array([0.0, 0.0, 1.0]))
    side = want["side"]                      # ANATOMICAL side, not near/far
    mask = angle_mask(ref[res.detected], side)

    for quantity, expected in (("knee_from_segments",
                                want["knee_from_segments"]),
                               ("thigh", want["thigh"])):
        est = res.series(side, quantity)[res.detected]
        gt = ref_ang[side][quantity][res.detected]
        ok = np.isfinite(est) & np.isfinite(gt) & mask
        got = float(np.abs(est[ok] - gt[ok]).mean())
        assert got == pytest.approx(expected, abs=ATOL), (
            f"{subject} {quantity}: demo path gives {got:.4f}, sealed test "
            f"committed {expected:.4f}. The demo is no longer the pipeline the "
            f"error budget describes.")


@movi_only
def test_sealed_expectations_match_the_committed_json():
    """Guard the test's own constants against the committed result.

    Also checks the near/far -> left/right mapping is right: the sealed column
    quoted must be the one whose limb is the anatomical side asserted above.
    """
    for subject, want in EXPECTED.items():
        sealed = _sealed(subject)
        cell = sealed["arms"]["detected"][want["sealed_column"]]
        assert cell["knee_from_segments"]["mae"] == pytest.approx(
            want["knee_from_segments"], abs=1e-3)
        assert cell["thigh"]["mae"] == pytest.approx(want["thigh"], abs=1e-3)
        assert sealed["n_frames"] == want["n_video"]
        # the mapping: sealed "near" means the sealed near_limb's anatomical side
        mapped = (sealed["near_limb"] if want["sealed_column"] == "near"
                  else ("r" if sealed["near_limb"] == "l" else "l"))
        assert mapped == want["side"], (
            f"{subject}: sealed column {want['sealed_column']} is side "
            f"{mapped}, test asserts {want['side']}")


# ------------------------------------- the near-limb finding, pinned -------

def test_the_demo_refuses_to_claim_a_near_limb():
    """The demo must not expose a near/far claim.

    `scripts/phase16a_nearlimb.py` measured the estimate at 9/11 against the
    camera truth, with NO signal separating the failures. Accessing
    `near_limb` therefore raises rather than returning a 9/11 guess.
    """
    from overstride.analyze import (NEAR_LIMB_AGREEMENT,  # noqa: PLC0415
                                    NEAR_LIMB_IS_RELIABLE, AnalysisResult)
    assert NEAR_LIMB_IS_RELIABLE is False
    assert NEAR_LIMB_AGREEMENT == pytest.approx(9 / 11)

    res = AnalysisResult(fps=30.0, n_frames=1, width=800, height=600,
                         resized_from=None, detector_min_size=600,
                         near_limb_estimate="l",
                         detected=np.array([True]))
    with pytest.raises(AttributeError, match="does not claim a near limb"):
        _ = res.near_limb
    # and the budget always quotes the conservative column
    assert res.budget_limb() == "far"
    assert res.near_limb_estimate == "l"      # still available as a diagnostic


@movi_only
def test_the_near_limb_measurement_is_recorded():
    """The 9/11 result must stay in the repo, so a future 'fix' has a baseline."""
    f = REPO / "results" / "phase16a_nearlimb.json"
    assert f.exists(), "run scripts/phase16a_nearlimb.py"
    d = json.loads(f.read_text(encoding="utf-8"))
    assert d["summary"]["n_agree"] == 9
    assert d["summary"]["n_subjects"] == 11
    # the key fact: no separator exists
    sep = d["summary"]["est_sep_over_width"]
    assert sep["disagree_max"] > sep["agree_min"], (
        "a disagreeing subject must still sit above an agreeing one -- that "
        "overlap is why no threshold works")


# -------------------------------------- [A1] the 1080p timing ceiling -------

@pytest.mark.slow
@movi_only
def test_a_1080p_clip_runs_within_the_re_estimated_time():
    """[A1] A 1080p clip, after the resize, must hit the measured per-frame cost.

    `results/phase16a_timing.json` measured 680.4 ms/frame for a 1080p clip
    resized to short side 600. The ceiling here is 900 ms/frame -- 1.3x the
    measurement -- so this fails on a regression (for instance the resize
    silently not being applied, which costs 2.13x) but not on machine noise.
    """
    import time

    from phase15_movi import decode_range
    from phase3_infer import load_model
    from phase7_inference_fix import CKPT_FOR

    cp1 = MOVI / "F_CP1_Subject_1.mp4"
    if not cp1.exists():
        pytest.skip("CP1 clip not on disk")

    n = 12
    frames, fps, _m = decode_range(cp1, 1500, 1500 + n)
    frames = list(frames)
    assert min(frames[0].shape[:2]) == 1080, "expected a 1080p source"

    lifter = load_model(CKPT_FOR["coco"], "base")
    analyze_frames(frames[:2], fps, lifter)          # warm-up
    t0 = time.perf_counter()
    res = analyze_frames(frames, fps, lifter)
    ms = 1000.0 * (time.perf_counter() - t0) / len(frames)

    assert res.detector_min_size == SHORT_SIDE, "the resize was not applied"
    assert res.resized_from == (1920, 1080)
    assert ms < 900.0, (
        f"{ms:.0f} ms/frame exceeds the 900 ms ceiling; "
        f"results/phase16a_timing.json measured 680 ms/frame for this case")
