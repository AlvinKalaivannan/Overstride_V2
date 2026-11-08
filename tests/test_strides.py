"""M3: gait-cycle segmentation, on synthetic signals and against its own bars.

No MoVi value appears here. The signals are built in the test, so the right
answer is known by construction; the measured behaviour on real footage lives
in `results/phase16a_strides.json` and the constants are pinned against it at
the bottom of this file.

`docs/phase16-demo-design.md` `[M3]`: the cycle boundary is **peak swing-phase
knee flexion**, not heel strike, because H36M-17 has no foot keypoint and the
standard substitute needs a direction of travel that this footage does not
give. These tests assert the demo never claims otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from overstride.strides import (CYCLE_SIGNAL, MAX_COUNT_DIFF,  # noqa: E402
                                MIN_KNEE_SWING_DEG, MIN_PLAUSIBLE_SHARE,
                                MIN_STRIDES, N_POINTS, STRIDE_S_MAX,
                                STRIDE_S_MIN, Strides, cv, cycle_events,
                                dominant_period, fill_short_gaps,
                                normalize_stride, peaks_per_stride, segment,
                                stride_average, swing_amplitude)

FPS = 30.0


def knee_signal(n_cycles: float, period_s: float = 0.75, fps: float = FPS,
                amp: float = 60.0, phase: float = 0.0,
                offset: float = 20.0) -> np.ndarray:
    """A knee-flexion-shaped trace with a known number of cycles."""
    n = int(round(n_cycles * period_s * fps))
    t = np.arange(n) / fps
    return offset + amp * 0.5 * (1 - np.cos(2 * np.pi * (t / period_s + phase)))


def both_legs(n_cycles: float = 8, **kw) -> dict:
    """`AnalysisResult.angles`-shaped input: the legs half a cycle apart."""
    return {"l": {CYCLE_SIGNAL: knee_signal(n_cycles, **kw)},
            "r": {CYCLE_SIGNAL: knee_signal(n_cycles, phase=0.5, **kw)}}


# ------------------------------------------------- the trigger itself ------

@pytest.mark.parametrize("n_cycles", [4, 6, 8, 12])
def test_the_trigger_finds_one_event_per_cycle(n_cycles):
    ev = cycle_events(knee_signal(n_cycles), FPS)
    # the first maximum is at t=period/2, so a clip of n cycles holds n events
    assert ev.size == pytest.approx(n_cycles, abs=1)


def test_the_events_are_the_maxima():
    sig = knee_signal(6)
    ev = cycle_events(sig, FPS)
    for i in ev:
        lo, hi = max(i - 5, 0), min(i + 6, sig.size)
        assert sig[i] == pytest.approx(sig[lo:hi].max(), abs=1e-9)


def test_a_secondary_bump_does_not_retrigger():
    """Stance flexion is a real second bump per cycle; it must not count.

    This is the whole reason for a hysteresis band rather than peak picking:
    a small bump cannot cross back below the low edge, so the trigger stays
    latched and the cycle is counted once.
    """
    sig = knee_signal(6)
    t = np.arange(sig.size) / FPS
    bumped = sig + 0.08 * 60.0 * np.sin(2 * np.pi * t / 0.75 * 2) ** 2
    assert cycle_events(bumped, FPS).size == cycle_events(sig, FPS).size


def test_a_flat_signal_yields_no_cycles():
    assert cycle_events(np.full(120, 30.0), FPS).size == 0


def test_a_changing_cadence_does_not_break_the_trigger():
    """No period estimate is used, so a runner who slows is still segmented."""
    fast = knee_signal(4, period_s=0.62)
    slow = knee_signal(4, period_s=0.95)
    ev = cycle_events(np.r_[fast, slow], FPS)
    assert ev.size >= 7, f"only {ev.size} events across 8 cycles"


# ----------------------------------------------------- the gap filling -----

def test_a_short_nan_gap_is_filled():
    x = knee_signal(6)
    holed = x.copy()
    holed[40:43] = np.nan                  # 3 frames = 0.1 s
    out = fill_short_gaps(holed, FPS)
    assert np.isfinite(out[40:43]).all()
    assert out[40:43] == pytest.approx(x[40:43], abs=6.0)


def test_a_long_nan_gap_is_left_alone():
    x = knee_signal(8)
    holed = x.copy()
    holed[40:60] = np.nan                  # 20 frames = 0.67 s
    assert not np.isfinite(fill_short_gaps(holed, FPS)[40:60]).any()


def test_filling_a_gap_does_not_invent_a_cycle():
    """The bug this exists for: one NaN frame splits the signal and the
    trigger re-arms mid-cycle, counting it twice."""
    x = knee_signal(6)
    holed = x.copy()
    holed[::17] = np.nan
    assert cycle_events(holed, FPS).size == cycle_events(x, FPS).size


def test_fill_handles_all_nan_and_no_nan():
    assert np.isnan(fill_short_gaps(np.full(10, np.nan), FPS)).all()
    x = knee_signal(3)
    assert np.array_equal(fill_short_gaps(x, FPS), x)


# --------------------------------------------------- the segmentation ------

def test_a_clean_clip_segments_and_passes_its_own_check():
    st = segment(both_legs(8), FPS)
    assert st.ok, st.reasons
    assert st.reasons == []
    assert abs(st.count["l"] - st.count["r"]) <= MAX_COUNT_DIFF
    assert st.n_used["l"] >= MIN_STRIDES
    for side in ("l", "r"):
        for lo, hi in st.used[side]:
            assert STRIDE_S_MIN <= (hi - lo) / FPS <= STRIDE_S_MAX


def test_the_boundary_is_named_and_is_not_a_heel_strike():
    st = segment(both_legs(8), FPS)
    assert st.boundary == "peak swing-phase knee flexion"
    assert "heel" not in st.boundary.lower()


def test_too_few_cycles_fails_the_check_and_says_why():
    st = segment(both_legs(2), FPS)
    assert not st.ok
    assert any("usable stride" in r for r in st.reasons)


def test_a_knee_that_does_not_swing_fails_the_check():
    small = both_legs(8, amp=MIN_KNEE_SWING_DEG * 0.5)
    st = segment(small, FPS)
    assert not st.ok
    assert any("does not look like running" in r for r in st.reasons)


def test_legs_disagreeing_on_the_count_fails_the_check():
    """Both legs take the same number of steps, so this cannot be right.

    Same clip LENGTH on both legs -- 6.0 s -- with 8 cycles on one and 12 on
    the other. Changing only the period would keep the counts equal and test
    nothing.
    """
    angles = both_legs(8, period_s=0.75)          # 8 x 0.75 = 6.0 s
    angles["r"][CYCLE_SIGNAL] = knee_signal(12, period_s=0.50)   # also 6.0 s
    st = segment(angles, FPS)
    assert not st.ok
    assert any("disagree on the cycle count" in r for r in st.reasons)


def test_implausible_cycle_durations_fail_the_share_check():
    """A signal cycling far too fast is not gait, and the share bar says so."""
    angles = {s: {CYCLE_SIGNAL: knee_signal(20, period_s=0.25)}
              for s in ("l", "r")}
    st = segment(angles, FPS)
    assert not st.ok
    assert any("plausible" in r for r in st.reasons)
    assert st.plausible_share("l") < MIN_PLAUSIBLE_SHARE


def test_undetected_frames_are_excluded():
    angles = both_legs(8)
    n = angles["l"][CYCLE_SIGNAL].size
    detected = np.ones(n, bool)
    detected[n // 2:] = False
    st = segment(angles, FPS, detected=detected)
    assert st.count["l"] < segment(angles, FPS).count["l"]


def test_duration_cv_is_reported_and_never_gates():
    """A runner who decelerates has a high CV and nothing is wrong."""
    angles = {s: {CYCLE_SIGNAL: np.r_[knee_signal(4, period_s=0.62),
                                      knee_signal(4, period_s=1.05)]}
              for s in ("l", "r")}
    st = segment(angles, FPS)
    assert np.isfinite(st.duration_cv("l"))
    assert st.duration_cv("l") > 0.15
    assert not any("CV" in r for r in st.reasons)


def test_cadence_is_plausible_for_the_synthetic_clip():
    st = segment(both_legs(8, period_s=0.75), FPS)
    assert st.cadence_spm == pytest.approx(160.0, abs=12.0)


def test_an_empty_strides_object_answers_without_raising():
    st = Strides(fps=FPS)
    assert st.count == {"l": 0, "r": 0}
    assert np.isnan(st.cadence_spm)
    assert np.isnan(st.duration_cv("l"))
    assert np.isnan(st.plausible_share("r"))


# ------------------------------------------------- the stride averages -----

def test_normalize_stride_returns_the_cycle_at_101_points():
    sig = knee_signal(6)
    out = normalize_stride(sig, 10, 32)
    assert out.shape == (N_POINTS,)
    assert out[0] == pytest.approx(sig[10])
    assert out[-1] == pytest.approx(sig[32])


def test_identical_strides_average_to_themselves_with_zero_spread():
    """A 0.80 s period is exactly 24 frames at 30 fps, so every stride is
    sampled identically and the spread must be numerically zero."""
    sig = knee_signal(8, period_s=0.80)
    st = segment(both_legs(8, period_s=0.80), FPS)
    avg = stride_average(sig, st.used["l"])
    assert avg["n"] == len(st.used["l"]) >= MIN_STRIDES
    assert avg["mean"].shape == (N_POINTS,)
    assert float(np.nanmax(avg["sd"])) < 1e-6, "identical strides, no spread"


def test_a_non_integer_period_leaves_only_sampling_jitter():
    """0.75 s is 22.5 frames, so strides are sampled at different offsets.

    The residual spread is frame quantization, not a segmentation error, and it
    must stay small enough that the report's spread band means something.
    """
    sig = knee_signal(8, period_s=0.75)
    st = segment(both_legs(8, period_s=0.75), FPS)
    sd = float(np.nanmax(stride_average(sig, st.used["l"])["sd"]))
    assert 0.0 < sd < 0.05 * 60.0


def test_stride_average_of_nothing_is_nan_not_an_error():
    avg = stride_average(knee_signal(4), [])
    assert avg["n"] == 0 and np.isnan(avg["mean"]).all()


def test_the_stride_sd_is_not_the_measurement_error():
    """Documented separation: the SD is repeatability, the budget is error."""
    import overstride.strides as m
    assert "NOT the measurement error" in m.stride_average.__doc__


def test_peaks_per_stride_and_cv():
    sig = knee_signal(8)
    st = segment(both_legs(8), FPS)
    pk = peaks_per_stride(sig, st.used["l"])
    assert pk.size == len(st.used["l"])
    assert np.allclose(pk, pk[0], atol=1.0)
    assert cv(pk) < 0.02
    assert np.isnan(cv([1.0]))
    assert np.isnan(cv([0.0, 0.0]))


def test_swing_amplitude_and_dominant_period():
    assert swing_amplitude(knee_signal(8, amp=60.0)) == pytest.approx(
        60.0, rel=0.15)
    assert dominant_period(knee_signal(8, period_s=0.75), FPS) == pytest.approx(
        0.75 * FPS, abs=2)
    assert np.isnan(dominant_period(np.zeros(200), FPS)), "no periodicity"
    assert np.isnan(dominant_period(np.arange(5.0), FPS)), "too short"


# ------------------------------------- the measured constants, pinned ------

RESULT = REPO / "results" / "phase16a_strides.json"


def _result():
    if not RESULT.exists():
        pytest.skip("run scripts/phase16a_strides.py")
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_the_bars_match_what_was_measured():
    d = _result()
    bars = d["bars_in_overstride_strides"]
    assert bars["MIN_STRIDES"] == MIN_STRIDES
    assert bars["MAX_COUNT_DIFF"] == MAX_COUNT_DIFF
    assert bars["MIN_KNEE_SWING_DEG"] == MIN_KNEE_SWING_DEG
    assert bars["MIN_PLAUSIBLE_SHARE"] == MIN_PLAUSIBLE_SHARE
    assert bars["STRIDE_S_RANGE"] == [STRIDE_S_MIN, STRIDE_S_MAX]

    gt = d["ground_truth"]
    # each bar must sit OUTSIDE the measured spread, or it rejects real gait
    assert min(gt["knee_swing_amplitude_deg"]) > MIN_KNEE_SWING_DEG
    assert gt["mean_stride_duration_s"][0] > STRIDE_S_MIN
    assert gt["mean_stride_duration_s"][1] < STRIDE_S_MAX


def test_the_legs_agreed_on_the_count_for_every_measured_subject():
    """The ground-truth-free consistency check, which is why this ships."""
    d = _result()
    gt = d["ground_truth"]
    assert gt["subjects_agreeing_within_1"] == d["n_subjects"]


def test_heel_strike_was_tried_measured_and_rejected():
    d = _result()
    rej = d["rejected_candidate"]
    assert "Zeni" in rej["method"]
    assert rej["n_readable"] < rej["n_subjects"], (
        "if the travel direction were readable, heel strike would be the "
        "boundary and this module would be wrong")
    assert rej["max_cycle_count_difference_vs_shipped"] >= 2


def test_the_stride_average_is_worth_showing():
    """M3's second bar: stride-to-stride spread below between-subject spread."""
    d = _result()
    k = d["knee_peak_spread"]
    assert k["within_subject_cv_median"] < k["across_subject_cv"]
    assert k["bar_met"] is True


def test_the_strides_json_is_aggregate_only():
    raw = RESULT.read_text(encoding="utf-8") if RESULT.exists() else ""
    if not raw:
        pytest.skip("run scripts/phase16a_strides.py")
    assert "Subject_" not in raw, "no per-subject MoVi value in the repo"


# ------------------------------ M3's bar, recorded as failed ---------------

def test_the_count_bar_is_recorded_as_failed_not_redefined():
    """2 of 3 clips, so the design's consequence applies: per-frame default."""
    from overstride.strides import (STRIDE_AVERAGING_IS_DEFAULT,
                                    STRIDE_COUNT_BAR, STRIDE_COUNT_BAR_MET,
                                    STRIDE_COUNT_CLIPS,
                                    STRIDE_COUNT_CLIPS_PASSED)
    assert STRIDE_COUNT_BAR == "+/-1 stride on 3 clips"
    assert (STRIDE_COUNT_CLIPS_PASSED, STRIDE_COUNT_CLIPS) == (2, 3)
    assert STRIDE_COUNT_BAR_MET is False
    assert STRIDE_AVERAGING_IS_DEFAULT is False, (
        "the design's stated consequence of M3 failing is per-frame traces "
        "only; stride averaging must be opt-in")


def test_the_recorded_bar_matches_the_measurement():
    from overstride.strides import (STRIDE_COUNT_BAR_MET,
                                    STRIDE_COUNT_CLIPS,
                                    STRIDE_COUNT_CLIPS_PASSED,
                                    STRIDE_SELFCHECK_PASSED_N)
    d = _result()["estimate_vs_ground_truth"]
    assert d["clips_within_1_stride"] == STRIDE_COUNT_CLIPS_PASSED
    assert d["n_clips"] == STRIDE_COUNT_CLIPS
    assert d["bar_met"] is STRIDE_COUNT_BAR_MET
    assert d["clips_passing_their_own_check"] == STRIDE_SELFCHECK_PASSED_N
    assert STRIDE_SELFCHECK_PASSED_N < STRIDE_COUNT_CLIPS, (
        "the self-check's screening value rests on one clip and must not be "
        "claimed as evidence")
