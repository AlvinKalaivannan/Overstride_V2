"""Phase 16 step 0 -- where do the 2,790 ms per frame actually go?

Phase 16's gate is >= 20 fps end to end, a 50 ms per-frame budget. The lifter
uses 11.2 ms of it. The 2D detector was timed at **2,790 ms/frame at 640x480**
and 3,280 ms at 1280x720 -- 56-66x over the whole budget by itself, which is the
one gate in the measurement track that is out by an order of magnitude.

`results/phase07.md` §5 then showed that **swapping the detector is not free**:
the lifter has learned its detector's error distribution, so handing the shipped
checkpoint cleaner-than-trained 2D costs up to +1.65 deg. A different detector
needs a matched lifter, and a matched lifter needs that detector's 2D on data
with 3D ground truth -- which AthleticsPose cannot provide, because it releases
no video.

So before reaching for a new architecture, this asks how much of the current
cost is the *model* and how much is how it is being *called*. Everything here
keeps `keypointrcnn_resnet50_fpn` and its weights exactly as they are, so
anything it finds is a free speed-up with **no distribution shift and no
retraining**:

  NATIVE RESOLUTION  torchvision's detector resizes every frame so its shortest
                     side is `min_size=800` before the backbone sees it. A
                     640x480 frame is therefore upscaled to 1067x800 -- 2.8x the
                     pixels -- purely by default.
  PERSON CROP        it is a two-stage detector run over the whole frame. The
                     runner occupies a fraction of it; the standard pose-pipeline
                     answer is to find the box once and run on a tight crop.
  BATCHING           the current loop calls the model one frame at a time.
  THREADS            CPU scaling is not free; more threads is not always faster.

Timed on synthetic frames: detector cost is a function of input size and
architecture, not of image content, and using noise keeps this runnable with no
video on disk. Keypoint *accuracy* is not measured here and cannot be -- see
above.

Run: .venv/Scripts/python.exe scripts/phase16_detector_speed.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase16_detector_speed.json"

BUDGET_MS = 50.0        # phase 16's >= 20 fps gate, end to end
LIFTER_MS = 11.2        # measured; leaves ~38 ms for everything else
PERSON_SCORE = 0.90


def frames(n: int, h: int, w: int, seed: int = 16) -> list[torch.Tensor]:
    """Synthetic frames. Detector cost tracks input size, not content."""
    rng = np.random.default_rng(seed)
    return [torch.from_numpy(rng.random((3, h, w), dtype=np.float32))
            for _ in range(n)]


def build(min_size: int | None, max_size: int | None):
    from torchvision.models.detection import (KeypointRCNN_ResNet50_FPN_Weights,
                                              keypointrcnn_resnet50_fpn)
    kw = {"weights": KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
          "box_score_thresh": 0.5}
    if min_size is not None:
        kw["min_size"] = min_size
    if max_size is not None:
        kw["max_size"] = max_size
    det = keypointrcnn_resnet50_fpn(**kw)
    det.eval()
    return det


@torch.no_grad()
def timed(det, fs: list[torch.Tensor], batch: int = 1,
          warmup: int = 1) -> float:
    """Mean ms per frame, after a warm-up pass that is not counted."""
    for i in range(warmup):
        det([fs[0]])
    t0 = time.perf_counter()
    for i in range(0, len(fs), batch):
        det(fs[i:i + batch])
    return 1000.0 * (time.perf_counter() - t0) / len(fs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()
    n = args.frames

    rows = []

    def record(label: str, ms: float, note: str = "") -> None:
        rows.append({"config": label, "ms_per_frame": ms,
                     "fps": 1000.0 / ms,
                     "x_over_budget": ms / (BUDGET_MS - LIFTER_MS),
                     "note": note})
        print(f"  {label:<44}{ms:>9.1f} ms{1000.0 / ms:>8.2f} fps"
              f"{ms / (BUDGET_MS - LIFTER_MS):>9.1f}x")

    print(f"=== detector cost, {n} frames each, {torch.get_num_threads()} threads ===")
    print(f"  budget: {BUDGET_MS:.0f} ms/frame total, {LIFTER_MS:.1f} ms is the "
          f"lifter, so ~{BUDGET_MS - LIFTER_MS:.0f} ms for the detector\n")
    print(f"  {'configuration':<44}{'ms/frame':>12}{'fps':>8}{'over':>9}")

    # 1. exactly what the tool does today: default min_size=800
    det_default = build(None, None)
    record("current path, 640x480 frame (min_size=800)",
           timed(det_default, frames(n, 480, 640)),
           "torchvision upscales the shortest side to 800")
    record("current path, 1280x720 frame (min_size=800)",
           timed(det_default, frames(n, 720, 1280)))

    # 2. same weights, told not to upscale
    det_native = build(480, 854)
    record("native resolution, 640x480 (min_size=480)",
           timed(det_native, frames(n, 480, 640)),
           "same weights, no upscaling")

    # 3. same weights, half resolution
    det_small = build(320, 568)
    record("half resolution, 320x240 input (min_size=320)",
           timed(det_small, frames(n, 240, 320)))

    # 4. person crop: the standard pose-pipeline answer
    det_crop = build(256, 256)
    record("tight person crop, 192x256 (min_size=256)",
           timed(det_crop, frames(n, 256, 192)),
           "box found once and tracked, keypoints on the crop")

    # 5. batching, at the crop size
    record("crop + batch of 4",
           timed(det_crop, frames(8, 256, 192), batch=4))

    # 6. thread scaling at the crop size
    for t in (4, 8, 16):
        torch.set_num_threads(t)
        record(f"crop, {t} threads", timed(det_crop, frames(n, 256, 192)))
    torch.set_num_threads(16)

    best = min(rows, key=lambda r: r["ms_per_frame"])
    worst = next(r for r in rows if r["config"].startswith("current path, 640"))
    out = {"budget_ms": BUDGET_MS, "lifter_ms": LIFTER_MS,
           "threads": torch.get_num_threads(),
           "frames_per_config": n,
           "note": "synthetic frames; detector cost tracks input size, not "
                   "content. Keypoint accuracy is NOT measured here -- "
                   "AthleticsPose releases no video, so it cannot be.",
           "rows": rows,
           "speedup_vs_current": worst["ms_per_frame"] / best["ms_per_frame"],
           "best": best["config"]}

    print(f"\n=== VERDICT ===")
    print(f"  current  {worst['ms_per_frame']:.0f} ms  ->  best "
          f"{best['ms_per_frame']:.0f} ms  ({out['speedup_vs_current']:.1f}x) "
          f"with the SAME weights")
    short = best["ms_per_frame"] / (BUDGET_MS - LIFTER_MS)
    if short > 1.0:
        print(f"  still {short:.1f}x over the detector's share of the budget, so "
              f"a different architecture is still needed for 20 fps.")
    else:
        print(f"  that fits the budget, with no retraining and no distribution "
              f"shift.")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
