"""Probe a clip, and resize it into the condition the error budget was measured in.

**The resize is condition-matching, not a speed optimisation** (design doc
`[A1]`). `video_kinematics.detect_coco17` sets `min_size = min(h, w)`, so the
input's **short side is the detector's working resolution**. Every number in the
error budget's Row B was measured on MoVi PG1 at **800x600**, short side
**600**. A 1920x1080 clip would run the detector at `min_size=1080`, changing
the pixel scale of every keypoint -- and phase 7 section 5 and phase 15F both
measured that this lifter is sensitive to exactly that.

So: short side -> 600 px, always, whatever comes in. It is also 2.13x faster on
1080p (`results/phase16a_timing.json`), which is a consequence and not the
reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# The sealed test's input short side. MoVi PG1 is 800x600, so this is a NO-OP
# on the footage the budget was measured on -- which is what makes the
# regression test in tests/test_demo_regression.py a valid check of the resize
# being harmless.
SHORT_SIDE = 600


@dataclass(frozen=True)
class ClipInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def needs_resize(self) -> bool:
        return self.short_side != SHORT_SIDE


def probe(path: str | Path) -> ClipInfo:
    """Stage 1: read a clip's properties without decoding it."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"cannot open {path}")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return ClipInfo(path=str(path), width=w, height=h, fps=fps, n_frames=n,
                    duration_s=(n / fps if fps > 0 else float("nan")))


def resize_short_side(frames, target: int = SHORT_SIDE) -> list[np.ndarray]:
    """Stage 3b: scale so the SHORT side is `target`, aspect ratio preserved.

    Identity when the short side already matches, so this is a no-op on the
    800x600 footage the budget was measured on.
    """
    frames = list(frames)
    if not frames:
        return frames
    h, w = frames[0].shape[:2]
    s = target / float(min(h, w))
    if abs(s - 1.0) < 1e-9:
        return frames
    nw, nh = int(round(w * s)), int(round(h * s))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    return [cv2.resize(f, (nw, nh), interpolation=interp) for f in frames]


def detector_working_resolution(frames) -> int:
    """What `detect_coco17` will pass as `min_size` for these frames.

    Exposed so a test can assert the resize actually controls the detector's
    working resolution, rather than asserting on the resize in isolation.
    """
    h, w = np.asarray(frames[0]).shape[:2]
    return int(min(h, w))
