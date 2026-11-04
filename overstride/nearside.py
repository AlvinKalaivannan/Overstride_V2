"""Which anatomical side faces the camera, and a geometric rule that suggests it.

**The user declares it.** `--near-side left|right` is required, because the demo
measured its own inference at only **9 of 11** against calibrated ground truth
with no signal separating the failures (`results/phase16a_nearlimb.json`).

This module provides a *suggestion* from image geometry, which is checked
against the calibrated near limb on all 11 spent MoVi subjects
(`scripts/phase16a_travelrule.py`). **It never overrides the flag.**

## The rule

Work in the OpenCV camera frame: **x right, y down, z into the scene**. That
triple is right-handed, since `x_hat x y_hat = z_hat`.

* world up in that frame is `u = (0, -1, 0)` (up on screen);
* the anatomical identity is `anterior x left = superior`, so
  `left = superior x anterior`;
* a runner travelling **rightward** in the image has `anterior = (+1, 0, 0)`.

Then

    left = u x f = (0,-1,0) x (1,0,0) = (0, 0, +1)

which is **+z, away from the camera**. So:

| travel across the image | side facing the camera |
|---|---|
| **rightward** (`dx > 0`) | **RIGHT** |
| **leftward** (`dx < 0`)  | **LEFT** |

## The assumption this rests on, which cannot be checked from the image

The rule inverts under **mirroring**. A front-facing phone camera usually
mirrors, and mirrored footage breaks more than this rule: the 2D detector
assigns left/right from learned appearance, so in a mirrored frame it labels the
anatomical left limb as the right one. **The anatomical labels themselves are
wrong, not just the near-side suggestion.**

There is no reliable image-only test for mirroring -- a mirrored runner looks
exactly like an unmirrored runner facing the other way. So the demo **states the
assumption and warns**, the filming instructions say *use the rear camera*, and
nothing here pretends to detect it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phase3_angles import H36M

# Measured against the calibrated near limb on the 11 spent MoVi PG1 subjects
# (scripts/phase16a_travelrule.py, results/phase16a_travelrule.json):
#
#   hit rate, abstentions counted as misses   6/11
#   accuracy when CONFIDENT                   6/6
#   abstained (travel too small to read)      5/11
#   if forced to answer on every clip         8/11
#
# The rule is never wrong when it speaks, and the confidence floor is clean:
# confident clips carry |dx| 0.0102-0.0310 hip widths/frame, abstained clips
# 0.0012-0.0095, and those ranges DO NOT OVERLAP. 6/11 is a floor set by the
# test footage -- MoVi subjects jog within a lab volume rather than crossing
# the camera's view, so five clips carry no readable direction of travel.
#
# Below 11/11, so per the design review this is a WARNING ONLY and never a
# default. It never overrides --near-side.
TRAVEL_RULE_HITS = 6
TRAVEL_RULE_N = 11
TRAVEL_RULE_CONFIDENT_CORRECT = 6
TRAVEL_RULE_CONFIDENT_N = 6
TRAVEL_RULE_IS_A_DEFAULT = False

# Below this, the runner is not crossing the frame decisively enough for the
# travel direction to mean anything. In units of hip width per frame, so it is
# scale-free. Chosen to sit well below every measured MoVi clip.
MIN_TRAVEL_PER_FRAME = 0.01


@dataclass(frozen=True)
class TravelSuggestion:
    side: str | None            # "l" / "r" / None when travel is ambiguous
    dx_per_frame: float         # median signed pelvis motion, hip widths/frame
    confident: bool
    reason: str


def travel_direction(kp2d: np.ndarray) -> tuple[float, float]:
    """Median signed horizontal pelvis motion per frame, in hip widths.

    Scale-free, so it does not depend on resolution or distance. Uses the
    detected 2D only -- no depth, no calibration.
    """
    kp2d = np.asarray(kp2d, dtype=float)
    ok = np.abs(kp2d[:, :, :2]).sum(axis=(1, 2)) > 0
    if ok.sum() < 3:
        return float("nan"), float("nan")
    pel = kp2d[ok][:, H36M["pelvis"], 0]
    hipw = np.abs(kp2d[ok][:, H36M["l_hip"], 0]
                  - kp2d[ok][:, H36M["r_hip"], 0])
    scale = float(np.nanmedian(hipw))
    if not np.isfinite(scale) or scale < 1e-6:
        # a near-perfect side-on view can collapse apparent hip width; fall
        # back on trunk length, which the camera does not foreshorten
        scale = float(np.nanmedian(np.abs(
            kp2d[ok][:, H36M["thorax"], 1] - kp2d[ok][:, H36M["pelvis"], 1])))
    if not np.isfinite(scale) or scale < 1e-6:
        return float("nan"), float("nan")
    d = np.diff(pel)
    return float(np.nanmedian(d) / scale), scale


def suggest_from_travel(kp2d: np.ndarray) -> TravelSuggestion:
    """Suggest the near side from the direction of travel. Never authoritative."""
    dx, _scale = travel_direction(kp2d)
    if not np.isfinite(dx):
        return TravelSuggestion(None, float("nan"), False,
                                "could not measure travel direction")
    if abs(dx) < MIN_TRAVEL_PER_FRAME:
        return TravelSuggestion(
            None, dx, False,
            f"the runner barely crosses the frame "
            f"({abs(dx):.4f} hip widths/frame, under "
            f"{MIN_TRAVEL_PER_FRAME}); direction of travel is ambiguous")
    # rightward travel -> RIGHT side faces the camera (see the module docstring)
    side = "r" if dx > 0 else "l"
    return TravelSuggestion(
        side, dx, True,
        f"runner travels {'rightward' if dx > 0 else 'leftward'} across the "
        f"image ({dx:+.4f} hip widths/frame), so the "
        f"{'right' if dx > 0 else 'left'} side faces the camera -- assuming "
        f"the footage is not mirrored")


def normalize_side(flag: str) -> str:
    """'left'/'l'/'LEFT' -> 'l'. Raises on anything else."""
    v = str(flag).strip().lower()
    if v in ("l", "left"):
        return "l"
    if v in ("r", "right"):
        return "r"
    raise ValueError(f"--near-side must be left or right, got {flag!r}")


def reconcile(declared: str, suggestion: TravelSuggestion) -> dict:
    """Combine the user's flag with the geometric suggestion.

    **The flag always wins.** The suggestion only ever produces a warning.
    """
    near = normalize_side(declared)
    out = {"near_side": near,
           "far_side": "r" if near == "l" else "l",
           "source": "--near-side (user declared)",
           "suggestion": suggestion.side,
           "suggestion_reason": suggestion.reason,
           "dx_per_frame": (None if not np.isfinite(suggestion.dx_per_frame)
                            else round(suggestion.dx_per_frame, 5)),
           "warnings": []}
    if suggestion.side is None:
        out["warnings"].append(
            f"could not suggest a near side from geometry: "
            f"{suggestion.reason}. Using --near-side as given.")
    elif suggestion.side != near:
        out["warnings"].append(
            f"DISAGREEMENT: you declared --near-side "
            f"{'left' if near == 'l' else 'right'}, but the "
            f"{TRAVEL_RULE_HITS}/{TRAVEL_RULE_N} direction-of-travel rule "
            f"suggests {'left' if suggestion.side == 'l' else 'right'} "
            f"({suggestion.reason}). Check which side of the runner the camera "
            f"was on, and that the clip is NOT mirrored (front-facing phone "
            f"cameras usually mirror). Proceeding with --near-side as given.")
    return out


MIRRORING_NOTICE = (
    "Mirroring cannot be detected from the image. A mirrored clip -- typically "
    "from a front-facing phone camera -- looks exactly like an unmirrored clip "
    "of a runner going the other way, and it corrupts more than the near-side "
    "suggestion: the 2D detector assigns left and right from learned "
    "appearance, so in a mirrored frame it labels the anatomical LEFT limb as "
    "the RIGHT one. Film with the REAR camera. This report assumes the footage "
    "is not mirrored.")
