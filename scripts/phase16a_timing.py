"""Measure the demo's real per-frame cost on REAL MoVi frames, not synthetic ones.

`results/phase16_detector_speed.json` timed `keypointrcnn_resnet50_fpn` on
SYNTHETIC frames, and its own note says so: *"synthetic frames; detector cost
tracks input size, not content."* That is fine for comparing call paths and
wrong for estimating a demo's runtime, because a synthetic frame contains no
person, so the crop tracker never engages and the second-stage keypoint head
sees no detections.

This measures the deployed path -- `video_kinematics.detect_coco17(crop=True)`
-- on decoded MoVi frames at three input sizes:

  1. PG1 native, 800x600          -- EXACTLY the sealed test's input condition
  2. CP1 native, 1920x1080        -- a phone-resolution clip, unresized
  3. CP1 resized to short side 600 -- the demo's fixed preprocessing step

`detect_coco17` sets `min_size=min(h, w)`, so the input's short side IS the
detector's working resolution: 1080 against 600 is 3.24x the pixels. Resizing to
short side 600 is therefore not a speed hack -- it puts a clip of any resolution
into the *same* condition the error budget was measured under.

No MoVi pixel data leaves `data/`; only timings are written.

Run: .venv/Scripts/python.exe scripts/phase16a_timing.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from phase15_movi import MOVI, decode_range  # noqa: E402
from video_kinematics import detect_coco17  # noqa: E402

OUT = REPO / "results" / "phase16a_timing.json"
SHORT_SIDE = 600          # the sealed test's input short side (PG1 is 800x600)
N_TIMED = 24              # frames timed per configuration, after a warm-up
N_WARM = 4


def resize_short_side(frames: list, target: int = SHORT_SIDE) -> list:
    """Scale so the SHORT side is `target`, preserving aspect ratio."""
    h, w = frames[0].shape[:2]
    s = target / float(min(h, w))
    if abs(s - 1.0) < 1e-6:
        return frames
    nw, nh = int(round(w * s)), int(round(h * s))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    return [cv2.resize(f, (nw, nh), interpolation=interp) for f in frames]


def time_config(name: str, frames: list, note: str) -> dict:
    h, w = frames[0].shape[:2]
    detect_coco17(frames[:N_WARM], crop=True, model=None)        # warm-up
    t0 = time.perf_counter()
    kp, meta = detect_coco17(frames[:N_TIMED], crop=True, model=None)
    dt = time.perf_counter() - t0
    ms = 1000.0 * dt / min(N_TIMED, len(frames))
    row = {"config": name, "width": int(w), "height": int(h),
           "detector_min_size": int(min(h, w)),
           "ms_per_frame": round(ms, 1), "fps": round(1000.0 / ms, 3),
           "n_frames_timed": int(min(N_TIMED, len(frames))),
           "frames_with_person": int(meta.get("frames_with_person", -1)),
           "re_detections": int(meta.get("re_detections", -1)),
           "megapixels": round(w * h / 1e6, 3), "note": note}
    print(f"  {name:<34}{w}x{h:<6} min_size {min(h,w):<5}"
          f"{ms:>9.1f} ms/frame  {1000.0/ms:>6.2f} fps  "
          f"person {row['frames_with_person']}/{row['n_frames_timed']}")
    return row


def main() -> int:
    torch.set_num_threads(torch.get_num_threads())
    print(f"=== phase 16A timing, REAL MoVi frames, "
          f"{torch.get_num_threads()} threads ===")
    print(f"    {N_TIMED} frames per configuration after {N_WARM} warm-up\n")

    rows = []

    pg1 = MOVI / "F_PG1_Subject_17_L.avi"
    if pg1.exists():
        f, fps, _ = decode_range(pg1, 1377, 1377 + N_TIMED + N_WARM + 2)
        rows.append(time_config("PG1 native (sealed condition)", list(f),
                                "exactly the input the sealed test measured"))
    else:
        print(f"  missing {pg1.name}")

    cp1 = MOVI / "F_CP1_Subject_1.mp4"
    if cp1.exists():
        f, fps, _ = decode_range(cp1, 1500, 1500 + N_TIMED + N_WARM + 2)
        full = list(f)
        rows.append(time_config("CP1 native, unresized", full,
                                "phone resolution; min_size=1080 is 3.24x the "
                                "pixels of 800x600"))
        rows.append(time_config("CP1 resized to short side 600",
                                resize_short_side(full),
                                "the demo's fixed preprocessing step: puts any "
                                "clip into the sealed test's condition"))
    else:
        print(f"  missing {cp1.name}")

    out = {"what": ("Per-frame cost of the deployed detection path on REAL "
                    "MoVi frames. Supersedes the synthetic-frame figures in "
                    "results/phase16_detector_speed.json for runtime "
                    "estimation."),
           "why_not_the_synthetic_figures": (
               "phase16_detector_speed.json timed synthetic frames, which "
               "contain no person, so the crop tracker never engages and the "
               "keypoint head sees no detections. Good for comparing call "
               "paths, wrong for a runtime estimate."),
           "detector": "keypointrcnn_resnet50_fpn, COCO weights, crop=True, "
                       "min_size = short side of the input",
           "resize_rule": f"short side -> {SHORT_SIDE} px before detection",
           "threads": int(torch.get_num_threads()),
           "n_frames_timed": N_TIMED, "n_warmup": N_WARM,
           "lifter_ms_per_frame": 11.2,
           "lifter_source": "results/phase16_detector_speed.json",
           "no_movi_pixels_written": True,
           "rows": rows}
    if len(rows) >= 2:
        base = next((r for r in rows if "PG1 native" in r["config"]), None)
        full = next((r for r in rows if r["config"] == "CP1 native, unresized"),
                    None)
        res = next((r for r in rows if "short side" in r["config"]), None)
        if base and full and res:
            out["ratios"] = {
                "cp1_native_over_pg1": round(full["ms_per_frame"]
                                             / base["ms_per_frame"], 2),
                "cp1_resized_over_pg1": round(res["ms_per_frame"]
                                              / base["ms_per_frame"], 2),
                "resize_speedup_on_1080p": round(full["ms_per_frame"]
                                                 / res["ms_per_frame"], 2),
                "pixel_ratio_1080_over_600": round((1920 * 1080)
                                                   / (800 * 600), 2)}
            print(f"\n  1080p unresized is {out['ratios']['cp1_native_over_pg1']}x "
                  f"the PG1 cost ({out['ratios']['pixel_ratio_1080_over_600']}x "
                  f"the pixels)")
            print(f"  resizing a 1080p clip to short side 600 is "
                  f"{out['ratios']['resize_speedup_on_1080p']}x faster and "
                  f"matches the measured condition")

    # ---- the demo's runtime estimate, from these numbers -----------------
    if rows:
        ms = next((r["ms_per_frame"] for r in rows
                   if "short side" in r["config"]), rows[0]["ms_per_frame"])
        pg = next((r["ms_per_frame"] for r in rows
                   if "PG1 native" in r["config"]), ms)
        est = {}
        for secs, fps_in in ((10, 60), (10, 30), (15, 60)):
            n = secs * fps_in
            n_model = round(n * 100.0 / fps_in)
            det = n * ms / 1000.0
            lift = n_model * 11.2 / 1000.0
            est[f"{secs}s_at_{fps_in}fps"] = {
                "frames": n, "model_frames": n_model,
                "detect_s": round(det, 1), "lift_s": round(lift, 1),
                "decode_render_s": 8.0,
                "total_s": round(det + lift + 8.0, 1),
                "total_min": round((det + lift + 8.0) / 60.0, 2),
                "x_real_time": round((det + lift + 8.0) / secs, 1)}
        out["runtime_estimate_after_resize"] = est
        out["runtime_basis_ms_per_frame"] = ms
        out["pg1_native_ms_per_frame"] = pg
        print(f"\n=== runtime after the resize step, at {ms:.0f} ms/frame ===")
        for k, v in est.items():
            print(f"  {k:<16} {v['frames']:>4} frames -> detect "
                  f"{v['detect_s']:>6.1f}s + lift {v['lift_s']:>5.1f}s "
                  f"= {v['total_s']:>6.1f}s ({v['total_min']:.1f} min, "
                  f"{v['x_real_time']:.0f}x real time)")

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
