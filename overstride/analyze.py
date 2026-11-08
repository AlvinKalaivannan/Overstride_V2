"""The demo's processing path. Stages 1, 3, 3b, 4-8 of the design doc's section 5.

**Every kinematic step is imported from the frozen measured pipeline.** This
module decides the ORDER and carries the per-frame results; it computes no
angle, no normalization and no window of its own. The design doc's section 1
pins the components, and `tests/test_demo_regression.py` asserts that this path
reproduces the sealed test's committed numbers for a spent MoVi subject.

  1  probe                 overstride.preprocess.probe
  3  decode                phase15_movi.decode_range  (or any frame list)
  3b resize                overstride.preprocess.resize_short_side   [A1]
  4  detect 2D             video_kinematics.detect_coco17(crop=True)
  6  reorder + confidence  coco17_to_h36m17 -> scores_to_train_scale
  7  lift                  phase15b_conventions.lift_2d  (edge pad, overlap)
  8  angles                phase15d_segments.segment_angles with EST_UP

Stage 5 (input checks on the detected 2D) is M2. Stride segmentation is M3.
**The trunk-referenced hip is not computed here at all** -- design doc `[A4]`:
its error is a learned convention, so it is not a measurement, and it is absent
from the report rather than labelled inside it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from phase15b_conventions import lift_2d  # noqa: E402
from phase15d_segments import segment_angles  # noqa: E402
from phase3_angles import H36M  # noqa: E402
from video_kinematics import (coco17_to_h36m17, detect_coco17,  # noqa: E402
                              scores_to_train_scale)

from overstride.nearside import (MIRRORING_NOTICE,  # noqa: E402
                                  TravelSuggestion, normalize_side,
                                  reconcile, suggest_from_travel)
from overstride.preprocess import (SHORT_SIDE, probe,  # noqa: E402
                                   resize_short_side)

# The level-camera vertical in the lifter's output frame. Verified for the
# deployed checkpoint in phase 15D, and re-verified for all three checkpoints
# on disk in phase 15F (all output up = axis 1, negative).
EST_UP = np.array([0.0, -1.0, 0.0])

PRIMARY = ("knee_from_segments", "thigh")
EXPERIMENTAL = ("trunk",)
REPORTED = PRIMARY + EXPERIMENTAL

# Legs are reported as LEFT and RIGHT, which the detector distinguishes
# unambiguously. NEAR and FAR are NOT claimed -- see `near_limb_estimate`.
SIDES = ("l", "r")

# `scripts/phase16a_nearlimb.py`, 11 spent MoVi PG1 subjects: the near limb
# inferred from the lifted 3D's hip depth agrees with the calibrated truth on
# only 9 of 11, and NO available signal separates the hits from the misses, so
# it cannot even be gated. Phase 14's "592 of 592" was AthleticsPose and does
# not transfer. **That inference is therefore not used for anything.**
#
# The near side is DECLARED by the user via --near-side, and
# `overstride.nearside.suggest_from_travel` offers a geometric suggestion that
# only ever warns. See results/phase16a_nearlimb.json and
# results/phase16a_travelrule.json.
NEAR_LIMB_FROM_DEPTH_AGREEMENT = 9 / 11
NEAR_LIMB_FROM_DEPTH_IS_RELIABLE = False


@dataclass
class AnalysisResult:
    """Per-frame angles for both legs, plus what the pipeline saw."""

    fps: float
    n_frames: int
    width: int
    height: int
    resized_from: tuple[int, int] | None
    detector_min_size: int
    near_side: str                       # "l"/"r" -- DECLARED by the user
    far_side: str                        # the other one
    near_limb_from_depth: str            # internal only, 9/11 -- never used
    detected: np.ndarray                 # (T,) bool
    kp2d: np.ndarray | None = None       # detected 2D, H36M-17 order
    travel: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    angles: dict = field(default_factory=dict)
    est3d: np.ndarray | None = None
    detect_meta: dict = field(default_factory=dict)

    def series(self, side: str, quantity: str) -> np.ndarray:
        return self.angles[side][quantity]

    @property
    def near_limb(self):
        """Deliberately absent: the demo does not INFER a near limb.

        Raising rather than returning is the point -- the depth inference is
        only 9/11 correct. Use `near_side`, which the user declared.
        """
        raise AttributeError(
            "the demo does not infer a near limb (9 of 11 against calibrated "
            "truth, no signal separating the failures -- see "
            "results/phase16a_nearlimb.json). Use `near_side`, which comes "
            "from the required --near-side flag.")

    def limb(self, which: str) -> str:
        """'near' or 'far' -> 'l' or 'r', from the DECLARED side."""
        if which == "near":
            return self.near_side
        if which == "far":
            return self.far_side
        raise ValueError(which)

    def budget_limb(self, side: str) -> str:
        """Which error-budget column applies to an anatomical side.

        With the near side declared, each leg gets its OWN column: Row B's near
        column for the near leg, far for the far leg. They differ by a measured
        +1.971 deg on the thigh, so this is the precision the flag buys back.
        """
        return "near" if side == self.near_side else "far"

    def mae_against(self, ref_angles: dict, side: str, quantity: str,
                    mask: np.ndarray | None = None) -> float:
        """MAE against a reference. Tests and internal checks only -- the demo
        no ground truth."""
        est = self.series(side, quantity)
        ref = ref_angles[side][quantity]
        n = min(len(est), len(ref))
        est, ref = est[:n], ref[:n]
        ok = np.isfinite(est) & np.isfinite(ref)
        if mask is not None:
            ok &= np.asarray(mask)[:n]
        if not ok.any():
            return float("nan")
        return float(np.abs(est[ok] - ref[ok]).mean())


def analyze_frames(frames, fps: float, lifter, near_side: str,
                   detector=None, resize: bool = True) -> AnalysisResult:
    """Stages 3b-8 on an in-memory frame list.

    `near_side` is REQUIRED -- "left" or "right", the side of the runner the
    camera was on. The demo cannot infer it (9/11), so the user declares it and
    the geometric suggestion only warns.

    `lifter` is the MotionAGFormer checkpoint; `detector` is the Keypoint R-CNN
    (None lets `detect_coco17` build it). The two models are separate and are
    passed separately, deliberately.
    """
    near = normalize_side(near_side)
    frames = list(frames)
    if not frames:
        raise ValueError("no frames")
    h0, w0 = frames[0].shape[:2]

    # --- 3b. resize into the condition the budget was measured in --------
    if resize:
        frames = resize_short_side(frames, SHORT_SIDE)
    h, w = frames[0].shape[:2]
    resized_from = (w0, h0) if (w, h) != (w0, h0) else None

    # --- 4. detect, 6. reorder + confidence ------------------------------
    coco, meta = detect_coco17(frames, crop=True, model=detector)
    kp = coco17_to_h36m17(coco)
    detected = np.abs(kp[:, :, :2]).sum(axis=(1, 2)) > 0
    prepared = np.nan_to_num(scores_to_train_scale(kp))

    # --- 7. lift ----------------------------------------------------------
    est = lift_2d(lifter, prepared, float(fps))

    # The near limb from the ESTIMATE's own hip depth. INTERNAL ONLY --
    # 9/11 against calibrated truth, with no signal separating the failures.
    # Nothing user-facing depends on it.
    near_depth = ("l" if np.nanmean(est[:, H36M["l_hip"], 2])
                  < np.nanmean(est[:, H36M["r_hip"], 2]) else "r")

    # the geometric suggestion, from the 2D alone. It only ever warns.
    travel = reconcile(near, suggest_from_travel(kp))

    # --- 8. angles --------------------------------------------------------
    angles = {}
    for side in SIDES:
        a = segment_angles(est, side, EST_UP)
        angles[side] = {q: a[q] for q in REPORTED}

    return AnalysisResult(
        fps=float(fps), n_frames=len(frames), width=w, height=h,
        resized_from=resized_from, detector_min_size=int(min(h, w)),
        near_side=near, far_side=travel["far_side"],
        near_limb_from_depth=near_depth, kp2d=kp, travel=travel,
        warnings=list(travel["warnings"]) + [MIRRORING_NOTICE],
        detected=detected, angles=angles, est3d=est, detect_meta=meta)


def analyze_clip(path, lifter, near_side: str, start: int = 0,
                 end: int | None = None, detector=None,
                 resize: bool = True) -> AnalysisResult:
    """Stages 1, 3, then `analyze_frames`. `near_side` is required."""
    from phase15_movi import decode_range
    info = probe(path)
    lo = int(start)
    hi = int(end) if end is not None else info.n_frames
    frames, fps, _meta = decode_range(Path(path), lo, hi)
    return analyze_frames(list(frames), float(fps), lifter, near_side,
                          detector=detector, resize=resize)


def reference_angles(ref3d: np.ndarray, up) -> dict:
    """Reference angles from a dataset's own 3D. **Regression test only** --
    the demo has no ground truth. `up` is the reference world vertical (MoVi:
    axis 2, per phase 15 check 2)."""
    out = {}
    for side in SIDES:
        a = segment_angles(np.asarray(ref3d), side, up)
        out[side] = {q: a[q] for q in REPORTED}
    return out
