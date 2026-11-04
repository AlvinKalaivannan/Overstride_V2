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
# estimated from the lifted 3D's hip depth agrees with the camera-derived truth
# on only 9 of 11, and NO available signal separates the hits from the misses.
# Estimated depth separation over width runs down to 0.0354 among the agreeing
# subjects while a disagreeing one sits higher at 0.0549; per-frame sign
# consistency is inverted (agree min 0.476, disagree max 0.688); and the
# ground-truth view_ratio does not separate them either (0.554 agrees, 0.665
# fails). Phase 14's "592 of 592" was AthleticsPose and does not transfer.
NEAR_LIMB_AGREEMENT = 9 / 11
NEAR_LIMB_IS_RELIABLE = False


@dataclass
class AnalysisResult:
    """Per-frame angles for both legs, plus what the pipeline saw."""

    fps: float
    n_frames: int
    width: int
    height: int
    resized_from: tuple[int, int] | None
    detector_min_size: int
    near_limb_estimate: str              # "l"/"r" -- UNRELIABLE, see below
    detected: np.ndarray                 # (T,) bool
    angles: dict = field(default_factory=dict)
    est3d: np.ndarray | None = None
    detect_meta: dict = field(default_factory=dict)

    def series(self, side: str, quantity: str) -> np.ndarray:
        return self.angles[side][quantity]

    @property
    def near_limb(self):
        """Deliberately absent. Use `near_limb_estimate` and read its caveat.

        Raising here rather than returning a value is the point: a caller that
        wants near/far must confront that it is only 9/11 correct.
        """
        raise AttributeError(
            "the demo does not claim a near limb: the estimate agrees with "
            "the camera on only 9 of 11 MoVi subjects and no signal separates "
            "the failures (results/phase16a_nearlimb.json). Report left/right, "
            "and quote the FAR-limb error budget for both legs.")

    def budget_limb(self) -> str:
        """Which error-budget column to quote: always 'far', the worse one.

        Near and far differ by a measured +1.971 deg on the thigh (sealed test,
        CI excluding zero, 5 of 5 subjects). Since the demo cannot tell which
        leg is which, quoting 'near' would understate the error on whichever
        leg is actually far. 'far' is conservative and never wrong by more than
        the measured gap.
        """
        return "far"

    def mae_against(self, ref_angles: dict, side: str, quantity: str,
                    mask: np.ndarray | None = None) -> float:
        """MAE against a reference. Tests and diagnostics only -- the demo has
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


def analyze_frames(frames, fps: float, lifter, detector=None,
                   resize: bool = True) -> AnalysisResult:
    """Stages 3b-8 on an in-memory frame list.

    `lifter` is the MotionAGFormer checkpoint; `detector` is the Keypoint R-CNN
    (None lets `detect_coco17` build it). The two models are separate and are
    passed separately, deliberately.
    """
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

    # The near limb from the ESTIMATE's own hip depth. Kept as a DIAGNOSTIC
    # only: measured at 9/11 against the camera truth on MoVi, with no signal
    # that separates the failures. Nothing user-facing may depend on it.
    near_est = ("l" if np.nanmean(est[:, H36M["l_hip"], 2])
                < np.nanmean(est[:, H36M["r_hip"], 2]) else "r")

    # --- 8. angles --------------------------------------------------------
    angles = {}
    for side in SIDES:
        a = segment_angles(est, side, EST_UP)
        angles[side] = {q: a[q] for q in REPORTED}

    return AnalysisResult(
        fps=float(fps), n_frames=len(frames), width=w, height=h,
        resized_from=resized_from, detector_min_size=int(min(h, w)),
        near_limb_estimate=near_est, detected=detected, angles=angles,
        est3d=est, detect_meta=meta)


def analyze_clip(path, lifter, start: int = 0, end: int | None = None,
                 detector=None, resize: bool = True) -> AnalysisResult:
    """Stages 1, 3, then `analyze_frames`."""
    from phase15_movi import decode_range
    info = probe(path)
    lo = int(start)
    hi = int(end) if end is not None else info.n_frames
    frames, fps, _meta = decode_range(Path(path), lo, hi)
    return analyze_frames(list(frames), float(fps), lifter,
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
