"""`overstride-analyze clip.mp4 --near-side left --out report.html`

The demo's command line. `docs/phase16-demo-design.md` section 5: a CLI that
writes a self-contained HTML report, and **no server**, because processing a
clip takes minutes rather than milliseconds and because a program with no
sockets cannot leak a frame.

    1  probe      fps, resolution, duration, frame count
    2  checks     section 4 checks 1-3; refuse BEFORE decoding
    3  decode     the analysis window only
    3b resize     short side -> 600 px, matching the measured condition
    4  detect     Keypoint R-CNN on a tracked person crop
    5  checks     checks 4-5 on the detected 2D; refuse here
    6  prepare    coco17_to_h36m17 -> scores_to_train_scale
    7  lift       81-frame overlapping windows -> 3D
    8  angles     gravity-referenced segment angles
    9  strides    opt-in; M3's count bar was NOT met, so OFF by default
    10 render     one self-contained .html

**`--near-side` is required.** The pipeline inferred the near limb correctly on
only 9 of 11 calibrated clips with no signal separating the failures, so the
user declares which side of the runner the camera was on. Get it wrong and the
two error columns are swapped.

**Deviation from the design, recorded.** Section 5 specifies the analysis
window as "the longest contiguous stretch in which the runner is fully in
frame, capped at 8 s". Finding that stretch requires detection, which is 95%
of the runtime, so choosing the window from it would mean paying for the whole
clip to decide which 8 s to pay for. The default is therefore **the middle 8
seconds**, where a runner filmed crossing the frame is most likely to be fully
in view, and `--window START END` overrides it. The error budget is unaffected:
the window has the same frame rate and resolution as the clip.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from overstride.analyze import analyze_frames  # noqa: E402
from overstride.budget import load_budget  # noqa: E402
from overstride.checks import (ClipRefused, post_detect_checks,  # noqa: E402
                               pre_decode_checks, raise_if_refused, verdict)
from overstride.nearside import normalize_side  # noqa: E402
from overstride.preprocess import probe  # noqa: E402
from overstride.report import write_report  # noqa: E402
from overstride.strides import (STRIDE_COUNT_BAR,  # noqa: E402
                                STRIDE_COUNT_CLIPS,
                                STRIDE_COUNT_CLIPS_PASSED, segment)

WINDOW_S = 8.0


def choose_window(info, window=None) -> tuple[int, int]:
    """The analysis window, as (first frame, last frame + 1)."""
    if window:
        return int(window[0]), int(window[1])
    cap = int(round(WINDOW_S * info.fps))
    if info.n_frames <= cap:
        return 0, info.n_frames
    start = (info.n_frames - cap) // 2
    return start, start + cap


def decode(path: Path, lo: int, hi: int):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    frames = []
    for _ in range(hi - lo):
        good, img = cap.read()
        if not good:
            break
        frames.append(img)
    cap.release()
    return frames, fps


def _print_checks(checks) -> None:
    for c in checks:
        print(f"  {c.status.upper():<7}{c.name:<12}{c.message}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="overstride-analyze",
        description="Measure running kinematics from one side-on clip.")
    ap.add_argument("clip", type=Path)
    ap.add_argument("--near-side", required=True, choices=["left", "right"],
                    help="which side of the runner the camera was on. "
                         "REQUIRED: the pipeline cannot infer it (9 of 11)")
    ap.add_argument("--out", type=Path, default=Path("report.html"))
    ap.add_argument("--window", type=int, nargs=2, metavar=("START", "END"),
                    help="frame range to analyse; default is the middle 8 s")
    ap.add_argument("--stride-average", action="store_true",
                    help=f"opt in to stride averaging. M3's count bar "
                         f"({STRIDE_COUNT_BAR}) came out "
                         f"{STRIDE_COUNT_CLIPS_PASSED} of "
                         f"{STRIDE_COUNT_CLIPS}, so per-frame traces are the "
                         f"default")
    ap.add_argument("--debug", action="store_true",
                    help="also write report-debug.html with internal "
                         "detail. Never part of a normal run, never "
                         "published")
    ap.add_argument("--no-resize", action="store_true",
                    help="skip the resize to short side 600. This leaves the "
                         "condition the error budget was measured in, so the "
                         "report says so")
    args = ap.parse_args(argv)

    near = normalize_side(args.near_side)
    if not args.clip.exists():
        raise SystemExit(f"no such clip: {args.clip}")

    t0 = time.perf_counter()
    info = probe(args.clip)
    print(f"{args.clip.name}: {info.n_frames} frames, {info.fps:.3g} fps, "
          f"{info.width}x{info.height}, {info.duration_s:.1f} s")

    pre = pre_decode_checks(info)
    _print_checks(pre)
    try:
        raise_if_refused(pre)
    except ClipRefused as e:
        print(f"\nREFUSED before decoding: {e}")
        print("No numbers are emitted. Nothing was processed.")
        return 2

    lo, hi = choose_window(info, args.window)
    print(f"  analysing frames {lo}-{hi} ({(hi - lo) / info.fps:.1f} s)")
    frames, fps = decode(args.clip, lo, hi)
    if not frames:
        raise SystemExit("decoded no frames")

    from phase3_infer import load_model
    from phase7_inference_fix import CKPT_FOR
    lifter = load_model(CKPT_FOR["coco"], "base")
    if lifter is None:
        raise SystemExit("the lifter checkpoint is missing; refusing to run")

    print(f"  detecting and lifting {len(frames)} frames "
          f"(~0.7 s per frame, so this takes minutes)")
    res = analyze_frames(frames, fps or info.fps, lifter,
                         near_side=args.near_side, resize=not args.no_resize)

    post = post_detect_checks(res.kp2d, res.detected, res.detect_meta)
    _print_checks(post)
    try:
        raise_if_refused(post)
    except ClipRefused as e:
        print(f"\nREFUSED after detection: {e}")
        print("No numbers are emitted.")
        return 2

    strides = None
    if args.stride_average:
        strides = segment(res.angles, res.fps, detected=res.detected)
        print(f"  strides: left {strides.count['l']}, "
              f"right {strides.count['r']}, "
              f"usable {strides.ok}")
        for r in strides.reasons:
            print(f"    - {r}")

    out = write_report(args.out, res, strides=strides, checks=pre + post,
                       budget=load_budget(), clip_name=args.clip.name,
                       frames=frames)
    took = time.perf_counter() - t0
    print(f"\nwrote {out}  ({out.stat().st_size / 1024:.0f} KB) in "
          f"{took / 60:.1f} min  [{verdict(pre + post)}]")

    if args.debug:
        dbg = out.with_name(out.stem + "-debug.html")
        _write_debug(dbg, res, near)
        print(f"wrote {dbg}  (internal, not a measurement, never published)")
    return 0


def _write_debug(path: Path, res, near: str) -> None:
    """The internal detail file. Separate, and never a normal run.

    The trunk-referenced joint excluded from the report lives here and only
    here: `[A4]`, its error is a learned convention, so it is not a
    measurement and it is absent from `report.html` entirely rather than
    labelled inside it.
    """
    from phase3_angles import sagittal_angles

    rows = []
    for side in ("l", "r"):
        a = sagittal_angles(res.est3d, side)
        for name, series in a.items():
            v = np.asarray(series, float)
            v = v[np.isfinite(v)]
            rows.append(f"<tr><td>{side}</td><td>{name}</td>"
                        f"<td>{v.min():.2f}</td><td>{v.max():.2f}</td>"
                        f"<td>{v.mean():.2f}</td></tr>")
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Overstride internal detail</title></head><body>"
        "<h1>Internal detail - NOT a measurement</h1>"
        "<p>Produced only by <code>--debug</code>. These quantities are "
        "excluded from the report because their error is a learned "
        "convention rather than a measurement limit: an H36M-trained lifter "
        "that has never seen running, this lab or this camera beats this "
        "pipeline on them on held-out footage. Do not publish this file.</p>"
        "<table border='1'><tr><th>side</th><th>quantity</th><th>min</th>"
        f"<th>max</th><th>mean</th></tr>{''.join(rows)}</table>"
        f"<p>near side declared: {near}</p></body></html>",
        encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
