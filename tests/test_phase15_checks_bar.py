"""Check 1's femur bar must do its job: catch a unit error, not a tall subject.

The bar was re-registered on 2026-11-02 (`results/phase15.md` D7). Its purpose
is detecting a 10x or 1000x misreading of millimetres, which is the trap that
cost this project two wrong claims on AthleticsPose's `p2mm` factor. These tests
pin the bar to that purpose, so it cannot drift back into being an anatomical
outlier filter -- and they are written from the DERIVATION, with no MoVi
subject's measurement anywhere in them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

# The plausible range for the measured quantity |mean(ASIS,PSIS) -> KJC|, from
# the derivation in phase15_checks: a population-general femur:stature ratio of
# 26.5-27.0% over a deliberately over-wide adult stature span of 1450-2100 mm,
# plus the 40-122 mm pelvis-landmark-to-hip-joint-centre offset.
PLAUSIBLE_LO = 0.265 * 1450 + 40.0        # 424.25
PLAUSIBLE_HI = 0.270 * 2100 + 122.0       # 689.0


def _bar():
    try:
        from phase15_checks import FEMUR_MM
    except Exception as e:                  # torch/cv2 absent in some envs
        pytest.skip(f"phase15_checks not importable: {e}")
    return FEMUR_MM


def test_the_bar_accepts_the_whole_plausible_human_range():
    """No real adult may be rejected for being tall or short."""
    lo, hi = _bar()
    assert lo <= PLAUSIBLE_LO, (
        f"bar lower bound {lo} excludes the shortest plausible adult "
        f"({PLAUSIBLE_LO:.1f} mm)")
    assert hi >= PLAUSIBLE_HI, (
        f"bar upper bound {hi} excludes the tallest plausible adult "
        f"({PLAUSIBLE_HI:.1f} mm)")


@pytest.mark.parametrize("factor", [0.001, 0.1])
def test_the_bar_rejects_every_undersized_unit_error(factor):
    """Metres (1/1000) and centimetres (1/10) must fail across the range."""
    lo, _hi = _bar()
    for true_mm in (PLAUSIBLE_LO, 500.0, PLAUSIBLE_HI):
        assert true_mm * factor < lo, (
            f"a length of {true_mm:.0f} mm read at {factor}x "
            f"({true_mm * factor:.3f}) is not rejected by the lower bound {lo}")


@pytest.mark.parametrize("factor", [10.0, 1000.0])
def test_the_bar_rejects_every_oversized_unit_error(factor):
    """A 10x or 1000x misreading must fail across the range."""
    _lo, hi = _bar()
    for true_mm in (PLAUSIBLE_LO, 500.0, PLAUSIBLE_HI):
        assert true_mm * factor > hi, (
            f"a length of {true_mm:.0f} mm read at {factor}x "
            f"({true_mm * factor:.0f}) is not rejected by the upper bound {hi}")


def test_there_is_real_headroom_on_both_sides():
    """A factor-of-ten error must not sit anywhere near the bar.

    Otherwise the check passes these tests while being one unlucky subject away
    from either a false accept or a false reject.
    """
    lo, hi = _bar()
    assert PLAUSIBLE_HI * 10.0 / hi >= 3.0, "too little headroom above"
    assert lo / (PLAUSIBLE_LO * 0.1) >= 3.0, "too little headroom below"


def test_the_superseded_bar_is_kept_in_the_record():
    """The old bar stays visible, so the change is auditable rather than quiet."""
    from phase15_checks import FEMUR_MM_OLD_PREREGISTERED
    assert FEMUR_MM_OLD_PREREGISTERED == (350.0, 550.0)
    assert _bar() != FEMUR_MM_OLD_PREREGISTERED


def test_the_old_bar_would_have_failed_its_own_purpose_test():
    """Why the change was needed, asserted rather than claimed in prose.

    The old (350, 550) bar rejected part of the plausible human range outright,
    which is how two subjects were excluded for being tall.
    """
    from phase15_checks import FEMUR_MM_OLD_PREREGISTERED
    _lo, hi = FEMUR_MM_OLD_PREREGISTERED
    assert hi < PLAUSIBLE_HI, (
        "the old bar is expected to exclude tall-but-real adults; if this "
        "fails, the derivation changed and D7 needs revisiting")
