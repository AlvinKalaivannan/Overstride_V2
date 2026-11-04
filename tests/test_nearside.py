"""The direction-of-travel near-side rule, and the flag that overrides it.

The demo cannot INFER which limb is nearer -- 9 of 11 against calibrated truth,
with no signal separating the failures. So `--near-side` is required and the
geometric rule only ever warns.

The rule itself is exact geometry and is tested here on synthetic clips, where
the travel direction is known by construction. Its behaviour on real footage is
measured in `scripts/phase16a_travelrule.py`: **6/11 with abstentions counted as
misses, 6 of 6 when confident**, and the constants are pinned against the
committed JSON below.
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

from overstride.nearside import (MIN_TRAVEL_PER_FRAME,  # noqa: E402
                                 MIRRORING_NOTICE,
                                 TRAVEL_RULE_CONFIDENT_CORRECT,
                                 TRAVEL_RULE_CONFIDENT_N, TRAVEL_RULE_HITS,
                                 TRAVEL_RULE_IS_A_DEFAULT, TRAVEL_RULE_N,
                                 normalize_side, reconcile,
                                 suggest_from_travel, travel_direction)
from phase3_angles import H36M  # noqa: E402


def _clip(dx_px: float, n: int = 60, hip_px: float = 40.0):
    """A synthetic runner translating `dx_px` per frame."""
    kp = np.zeros((n, 17, 3))
    for t in range(n):
        cx = 400.0 + dx_px * t
        kp[t, :, 0] = cx
        kp[t, :, 1] = 300.0
        kp[t, :, 2] = 0.9
        kp[t, H36M["l_hip"], 0] = cx - hip_px / 2
        kp[t, H36M["r_hip"], 0] = cx + hip_px / 2
        kp[t, H36M["pelvis"], 0] = cx
        kp[t, H36M["thorax"], 1] = 300.0 - 200.0
    return kp


# ---------------------------------------------------------- the geometry ----

def test_rightward_travel_means_the_right_side_faces_the_camera():
    """left = superior x anterior; for rightward travel that is +z, i.e. far."""
    s = suggest_from_travel(_clip(+4.0))
    assert s.side == "r" and s.confident
    assert "rightward" in s.reason


def test_leftward_travel_means_the_left_side_faces_the_camera():
    s = suggest_from_travel(_clip(-4.0))
    assert s.side == "l" and s.confident
    assert "leftward" in s.reason


def test_the_rule_is_antisymmetric():
    a = suggest_from_travel(_clip(+4.0))
    b = suggest_from_travel(_clip(-4.0))
    assert {a.side, b.side} == {"l", "r"}
    assert a.dx_per_frame == pytest.approx(-b.dx_per_frame, abs=1e-9)


def test_travel_is_scale_free():
    """Measured in hip widths, so resolution and distance must not matter."""
    small = travel_direction(_clip(+4.0, hip_px=40.0))[0]
    big = travel_direction(_clip(+8.0, hip_px=80.0))[0]
    assert small == pytest.approx(big, rel=1e-9)


# ------------------------------------------------------- the abstention -----

def test_a_stationary_runner_gets_no_suggestion():
    s = suggest_from_travel(_clip(0.0))
    assert s.side is None and not s.confident
    assert "ambiguous" in s.reason


def test_travel_below_the_floor_abstains():
    dx_px = 0.5 * MIN_TRAVEL_PER_FRAME * 40.0      # half the floor
    s = suggest_from_travel(_clip(dx_px))
    assert s.side is None and not s.confident


def test_travel_above_the_floor_answers():
    dx_px = 2.0 * MIN_TRAVEL_PER_FRAME * 40.0
    s = suggest_from_travel(_clip(dx_px))
    assert s.side == "r" and s.confident


def test_too_few_frames_abstains():
    assert suggest_from_travel(_clip(+4.0, n=2)).side is None


# --------------------------------------------------- the flag always wins ---

def test_the_flag_wins_and_agreement_is_silent():
    out = reconcile("right", suggest_from_travel(_clip(+4.0)))
    assert out["near_side"] == "r" and out["far_side"] == "l"
    assert out["source"].startswith("--near-side")
    assert out["warnings"] == []


def test_disagreement_warns_but_does_not_override():
    out = reconcile("left", suggest_from_travel(_clip(+4.0)))   # rule says r
    assert out["near_side"] == "l", "the flag must win"
    assert out["far_side"] == "r"
    assert len(out["warnings"]) == 1
    w = out["warnings"][0]
    assert "DISAGREEMENT" in w
    assert "mirror" in w.lower()            # points at the likely cause
    assert "Proceeding with --near-side as given" in w


def test_abstention_warns_without_contradicting():
    out = reconcile("left", suggest_from_travel(_clip(0.0)))
    assert out["near_side"] == "l"
    assert len(out["warnings"]) == 1
    assert "could not suggest" in out["warnings"][0]


@pytest.mark.parametrize("flag,want", [
    ("left", "l"), ("LEFT", "l"), ("l", "l"), (" Left ", "l"),
    ("right", "r"), ("RIGHT", "r"), ("r", "r"),
])
def test_normalize_side_accepts_the_obvious_forms(flag, want):
    assert normalize_side(flag) == want


@pytest.mark.parametrize("bad", ["near", "far", "", "up", "1", "lefft", None])
def test_normalize_side_rejects_everything_else(bad):
    with pytest.raises(ValueError, match="must be left or right"):
        normalize_side(bad)


# ---------------------------------------------------------- mirroring -------

def test_the_mirroring_notice_says_what_it_breaks_and_what_to_do():
    n = MIRRORING_NOTICE
    assert "cannot be detected" in n
    assert "REAR camera" in n
    # it must say the LABELS break, not just the suggestion
    assert "LEFT" in n and "RIGHT" in n
    assert "front-facing" in n


# ------------------------------------------- the measured constants ---------

def test_the_rule_is_not_a_default_because_it_is_below_11_of_11():
    assert TRAVEL_RULE_HITS == 6 and TRAVEL_RULE_N == 11
    assert TRAVEL_RULE_HITS < TRAVEL_RULE_N
    assert TRAVEL_RULE_IS_A_DEFAULT is False, (
        "below 11/11 the design review requires warning-only, never a default")


def test_the_confidence_floor_is_clean_and_the_rule_never_erred_when_confident():
    assert TRAVEL_RULE_CONFIDENT_CORRECT == TRAVEL_RULE_CONFIDENT_N == 6
    f = REPO / "results" / "phase16a_travelrule.json"
    if not f.exists():
        pytest.skip("run scripts/phase16a_travelrule.py")
    d = json.loads(f.read_text(encoding="utf-8"))
    assert d["n_hits"] == TRAVEL_RULE_HITS
    assert d["n_subjects"] == TRAVEL_RULE_N
    assert d["confident_accuracy"] == "6/6"
    # the floor separates cleanly -- unlike every near-limb separator
    conf_lo = d["confident_abs_dx_range"][0]
    abst_hi = d["abstained_abs_dx_range"][1]
    assert conf_lo > abst_hi, (
        f"confident |dx| starts at {conf_lo} and abstained reaches {abst_hi}; "
        f"these must NOT overlap for the floor to mean anything")
    assert abst_hi < MIN_TRAVEL_PER_FRAME <= conf_lo


def test_the_travel_json_is_aggregate_only():
    f = REPO / "results" / "phase16a_travelrule.json"
    if not f.exists():
        pytest.skip("run scripts/phase16a_travelrule.py")
    raw = f.read_text(encoding="utf-8")
    assert "Subject_" not in raw, "no per-subject MoVi value in the repo"
    assert "subjects" not in json.loads(raw)
