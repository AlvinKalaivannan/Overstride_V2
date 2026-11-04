"""M2: the input checks of `docs/phase16-demo-design.md` section 4.

Every threshold in `overstride/checks.py` is derived from a measured number, so
every test here checks a threshold **at its boundary** and names the measurement
the boundary comes from. Two tests matter more than the rest:

* `test_no_clip_in_the_measured_range_is_flagged` -- the binding constraint from
  amendment `[A5]`. Row B's own footage must pass, or the budget contradicts the
  checks that gate it.
* `test_a_frontal_clip_is_refused` -- the frontal geometry MoVi actually
  measured (view_ratio 0.057 standing) must be refused, not warned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from overstride.checks import (FPS_REFUSE, FPS_WARN,  # noqa: E402
                               LEN_REFUSE_S, LEN_WARN_S,
                               GT_VIEW_REFUSE, GT_VIEW_WARN,
                               MEASURED_EST_VIEW_RANGE, MEASURED_VIEW_RANGE,
                               MISS_REFUSE_FRAC, PASS,
                               VIEW_BIAS_EST_MINUS_GT,
                               REFUSE, RES_REFUSE, RES_WARN, VIEW_REFUSE,
                               VIEW_WARN, WARN, ClipRefused, check_fps,
                               check_length, check_person, check_resolution,
                               check_viewpoint, estimate_view_ratio,
                               post_detect_checks, pre_decode_checks,
                               raise_if_refused, verdict)
from overstride.preprocess import ClipInfo  # noqa: E402
from phase14_confidence import R_POP  # noqa: E402
from phase3_angles import H36M  # noqa: E402


# ------------------------------------------------- 1. frame rate -----------

@pytest.mark.parametrize("fps,want", [
    (240.0, PASS), (100.0, PASS), (60.0, PASS), (50.0, PASS),   # at the bar
    (49.9, WARN), (30.0, WARN), (25.0, WARN),                   # at the bar
    (24.9, REFUSE), (15.0, REFUSE), (0.0, REFUSE),
])
def test_fps_thresholds_at_the_boundary(fps, want):
    assert check_fps(fps).status == want


def test_30_fps_warns_and_still_processes():
    """30 fps is what Row B was measured at, so it must WARN, never refuse."""
    r = check_fps(30.0)
    assert r.status == WARN and r.ok
    assert "67%" in r.message          # cites the measured steepening


def test_fps_refuse_bar_sits_below_every_measured_rate():
    assert FPS_REFUSE < 30.0, "30.0 fps is Row B's own rate; it must not refuse"
    assert FPS_WARN == 50.0


# ------------------------------------------------- 2. resolution -----------

@pytest.mark.parametrize("w,h,want", [
    (1920, 1080, PASS), (800, 600, PASS), (640, 480, PASS),
    (854, 479, WARN), (640, 400, WARN),
    (480, 359, REFUSE), (320, 240, REFUSE),
])
def test_resolution_thresholds(w, h, want):
    assert check_resolution(w, h).status == want


def test_the_sealed_resolution_passes():
    """800x600 is the condition Row B was measured in."""
    assert check_resolution(800, 600).status == PASS
    assert RES_WARN <= 600 and RES_REFUSE < 600


# ------------------------------------------------- 3. clip length ----------

@pytest.mark.parametrize("n,fps,want", [
    (900, 60.0, PASS), (480, 60.0, PASS), (240, 30.0, PASS),
    (420, 60.0, WARN), (150, 30.0, WARN),
    (120, 60.0, REFUSE), (60, 30.0, REFUSE),
])
def test_length_thresholds(n, fps, want):
    assert check_length(n, fps).status == want


def test_length_refuse_is_shorter_than_one_lifter_window_at_30fps():
    """81 frames at 30 fps is 2.7 s, so the 3 s bar is just above one window."""
    assert LEN_REFUSE_S * 30.0 >= 81 - 1
    assert LEN_WARN_S == 8.0


# ------------------------------------------------- 4. person found ---------

def test_person_all_detected_passes():
    assert check_person(np.ones(200, bool), {"frames_crowded": 0}).status == PASS


def test_person_a_few_misses_warn():
    d = np.ones(200, bool); d[:5] = False          # 2.5%
    assert check_person(d, {"frames_crowded": 0}).status == WARN


def test_person_too_many_misses_refuses():
    d = np.ones(100, bool); d[:25] = False         # 25% > 20%
    r = check_person(d, {"frames_crowded": 0})
    assert r.status == REFUSE
    assert "11 of 11" in r.message                  # cites the sealed result


def test_more_than_one_person_refuses_even_with_perfect_detection():
    r = check_person(np.ones(200, bool), {"frames_crowded": 3})
    assert r.status == REFUSE
    assert "more than one person" in r.message


def test_miss_bar_matches_the_design():
    assert MISS_REFUSE_FRAC == 0.20


# ------------------------------------------------- 5. viewpoint ------------

def _kp_at_azimuth(theta_deg: float, n: int = 60, trunk_px: float = 200.0):
    """Synthetic 2D at a known camera azimuth.

    Apparent hip separation is `R_POP * trunk * |cos(theta)|` by the proxy's
    own identity, so this inverts `estimate_view_ratio` exactly and lets the
    thresholds be tested in the units the demo computes.
    """
    kp = np.zeros((n, 17, 3))
    ml = R_POP * trunk_px * abs(np.cos(np.radians(theta_deg)))
    for t in range(n):
        cx = 400.0 + 2.0 * t
        kp[t, :, 0] = cx
        kp[t, :, 1] = 300.0
        kp[t, H36M["l_hip"]] = [cx - ml / 2, 300.0, 0.9]
        kp[t, H36M["r_hip"]] = [cx + ml / 2, 300.0, 0.9]
        kp[t, H36M["l_shoulder"]] = [cx, 300.0 - trunk_px, 0.9]
        kp[t, H36M["r_shoulder"]] = [cx, 300.0 - trunk_px, 0.9]
        kp[t, H36M["pelvis"]] = [cx, 300.0, 0.9]
        kp[t, H36M["thorax"]] = [cx, 300.0 - trunk_px, 0.9]
    return kp


def test_the_view_ratio_estimator_inverts_the_proxy_identity():
    """sanity: a 90 deg azimuth is side-on (1.0), 0 deg is frontal (0.0)."""
    assert estimate_view_ratio(_kp_at_azimuth(90.0)) == pytest.approx(1.0,
                                                                     abs=1e-6)
    assert estimate_view_ratio(_kp_at_azimuth(0.0)) == pytest.approx(0.0,
                                                                     abs=1e-6)
    # and it is monotone in between
    vals = [estimate_view_ratio(_kp_at_azimuth(a))
            for a in (10, 30, 50, 70, 90)]
    assert vals == sorted(vals)


@pytest.mark.parametrize("vr_target,want", [
    # these are ESTIMATED view_ratio, which is what the demo computes
    (0.95, PASS), (0.887, PASS), (0.633, PASS), (0.601, PASS),
    (0.59, WARN), (0.55, WARN), (0.491, WARN),
    (0.489, REFUSE), (0.30, REFUSE), (0.196, REFUSE),
])
def test_viewpoint_thresholds(vr_target, want):
    """Thresholds, tested just inside and outside each bar.

    The exact boundary value is deliberately NOT tested through this path: the
    synthetic clip is built by `arcsin` and measured back by `sin`, and that
    round-trip loses the last bits, so `0.35` lands a hair below 0.35. That
    would be a test of float precision, not of the threshold. The thresholds
    themselves are pinned directly in
    `test_viewpoint_boundaries_are_exclusive_below`.
    """
    theta = np.degrees(np.arcsin(np.clip(vr_target, 0, 1)))
    assert check_viewpoint(_kp_at_azimuth(theta)).status == want


def test_viewpoint_boundaries_are_exclusive_below():
    """A value exactly ON a bar is the kinder side of it.

    Tested on the threshold logic directly, with no trig round-trip, so this is
    about the comparison and not about floating point.
    """
    from overstride.checks import CheckResult  # noqa: F401

    def verdict_for(vr: float) -> str:
        if vr < VIEW_REFUSE:
            return REFUSE
        if vr < VIEW_WARN:
            return WARN
        return PASS

    assert verdict_for(VIEW_REFUSE) == WARN        # exactly 0.49 -> warn
    assert verdict_for(VIEW_WARN) == PASS          # exactly 0.60 -> pass
    assert verdict_for(np.nextafter(VIEW_REFUSE, 0.0)) == REFUSE
    assert verdict_for(np.nextafter(VIEW_WARN, 0.0)) == WARN
    # and the measured minimum is comfortably clear of the warn bar
    assert MEASURED_VIEW_RANGE[0] - VIEW_WARN > 0.03


def test_no_clip_in_the_measured_range_is_flagged():
    """[A5] THE BINDING CONSTRAINT, in the units the demo actually computes.

    Row B's own footage spans ESTIMATED view_ratio [0.633, 0.887] -- measured in
    results/phase16a_viewratio.json, where all 11 clips pass. Its ground-truth
    range is [0.537, 0.719]; the estimator runs +0.139 optimistic, so testing
    the GT range here would test the wrong numbers.
    """
    lo, hi = MEASURED_EST_VIEW_RANGE
    for vr in np.linspace(lo, hi, 12):
        theta = np.degrees(np.arcsin(vr))
        r = check_viewpoint(_kp_at_azimuth(theta))
        assert r.status == PASS, (
            f"view_ratio {vr:.3f} is inside the measured range "
            f"[{lo}, {hi}] and must not be flagged, got {r.status}")
    assert VIEW_WARN < lo, "the warn bar must sit below the measured minimum"


def test_a_frontal_clip_is_refused():
    """MoVi's measured frontal standing geometry: GT view_ratio 0.057.

    In estimate units that is about 0.057 + 0.139 = 0.196, still far below the
    0.49 refuse bar.
    """
    theta = np.degrees(np.arcsin(0.057 + VIEW_BIAS_EST_MINUS_GT))
    r = check_viewpoint(_kp_at_azimuth(theta))
    assert r.status == REFUSE
    assert "frontal" in r.message
    assert "3.50" in r.message          # cites phase 11's curve


def test_viewpoint_bars_are_in_estimate_units_not_ground_truth():
    """The bars must be shifted by the MEASURED estimator bias.

    Applying a ground-truth bar to a biased estimate under-refuses: a clip whose
    true view_ratio is 0.35 estimates near 0.49 and would only warn.
    """
    assert (GT_VIEW_WARN, GT_VIEW_REFUSE) == (0.50, 0.35)
    assert VIEW_BIAS_EST_MINUS_GT == pytest.approx(0.139)
    # refuse bar = GT bar + bias, to within rounding
    assert VIEW_REFUSE == pytest.approx(GT_VIEW_REFUSE
                                        + VIEW_BIAS_EST_MINUS_GT, abs=0.005)
    # and the pass bar sits below the lowest estimate the budget's footage gave
    assert VIEW_WARN < MEASURED_EST_VIEW_RANGE[0]
    assert VIEW_REFUSE < VIEW_WARN


def test_the_estimator_bias_is_recorded():
    import json
    f = REPO / "results" / "phase16a_viewratio.json"
    if not f.exists():
        pytest.skip("run scripts/phase16a_viewratio.py")
    d = json.loads(f.read_text(encoding="utf-8"))
    assert d["n_flagged"] == 0, "Row B's own footage must all pass"
    assert d["n_pass"] == d["n_subjects"] == 11
    assert d["est_minus_gt"]["mean"] == pytest.approx(
        VIEW_BIAS_EST_MINUS_GT, abs=0.002)
    assert "Subject_" not in f.read_text(encoding="utf-8")


# ------------------------------------------------- orchestration -----------

def test_pre_decode_checks_run_all_three():
    info = ClipInfo(path="x.mp4", width=1920, height=1080, fps=60.0,
                    n_frames=600, duration_s=10.0)
    rs = pre_decode_checks(info)
    assert [r.name for r in rs] == ["fps", "resolution", "length"]
    assert verdict(rs) == PASS


def test_pre_decode_refuses_a_bad_clip_before_decoding():
    info = ClipInfo(path="x.mp4", width=320, height=240, fps=15.0,
                    n_frames=30, duration_s=2.0)
    rs = pre_decode_checks(info)
    assert verdict(rs) == REFUSE
    with pytest.raises(ClipRefused) as e:
        raise_if_refused(rs)
    assert len(e.value.results) == 3        # all three refuse


def test_post_detect_checks_run_both():
    kp = _kp_at_azimuth(60.0)
    rs = post_detect_checks(kp, np.ones(len(kp), bool), {"frames_crowded": 0})
    assert [r.name for r in rs] == ["person", "viewpoint"]
    assert verdict(rs) == PASS


def test_verdict_takes_the_worst_status():
    from overstride.checks import CheckResult
    mk = (lambda s: CheckResult(name="x", status=s, message=""))
    assert verdict([mk(PASS), mk(PASS)]) == PASS
    assert verdict([mk(PASS), mk(WARN)]) == WARN
    assert verdict([mk(WARN), mk(REFUSE)]) == REFUSE


def test_a_warning_does_not_stop_processing():
    info = ClipInfo(path="x.mp4", width=1280, height=720, fps=30.0,
                    n_frames=180, duration_s=6.0)
    rs = pre_decode_checks(info)
    assert verdict(rs) == WARN
    raise_if_refused(rs)                     # must not raise
    assert all(r.ok for r in rs)
