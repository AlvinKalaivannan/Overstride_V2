"""A procedurally generated running clip. No dataset frame, anywhere.

`docs/phase16-demo-design.md` section 6 forbids a MoVi frame from leaving
`data/` and section 7 forbids any dataset frame from appearing in a report,
screenshot or published artefact. But a renderer needs something to render, and
an example report needs something to look at.

So this builds a runner from nothing: joint angle traces, forward kinematics
into a 3D skeleton, a pinhole projection to 2D, and a stick figure drawn on a
plain background. **Every pixel is generated here**, so a report made from it
can be opened, shown and committed without touching any dataset's licence.

**What it does and does not exercise.** It drives the report from stage 8
onward -- the real `segment_angles`, the real stride segmentation, the real
budget and the real renderer -- with the angles known by construction, so the
renderer can be tested against a right answer. It does **not** exercise the 2D
detector or the lifter: a stick figure is not a person and Keypoint R-CNN will
not find one. The detector and lifter are covered by
`tests/test_demo_regression.py` on real footage instead.

Run: .venv/Scripts/python.exe scripts/synthetic_clip.py --out report.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from overstride.analyze import EST_UP, REPORTED, AnalysisResult  # noqa: E402
from phase15d_segments import segment_angles  # noqa: E402
from phase3_angles import H36M  # noqa: E402

# Segment lengths in millimetres, from AddBiomechanics-style proportions for a
# 1.75 m runner. Plausibility only -- nothing here is a measurement.
TRUNK, THIGH, SHANK = 520.0, 420.0, 400.0
HIP_W, SHOULDER_W = 180.0, 360.0
UPPER_ARM, FOREARM = 300.0, 260.0
HEAD = 230.0

# The camera. Side-on, level, 9 m away with a longish lens -- because the
# geometry has to be filmable. A phone 3 m from a runner sees 2.7 m of ground,
# which a 2.6 m/s runner crosses in ONE second, so the runner cannot stay in
# frame for the 8 s window the design asks for. Real side-on footage is
# therefore filmed from further back and panned, which is what this models: the
# camera tracks the runner and leaves a slow residual drift across the frame.
#
# Getting this wrong the first time was instructive: with the camera at 3 m the
# runner left the frame, the pelvis reached 22 m off-axis, and the two hips --
# separated only in DEPTH -- acquired a 193 px apparent separation purely from
# perspective. `check_viewpoint` then REFUSED the clip as near-frontal, which
# was the correct answer to the geometry it was given.
FOCAL, DEPTH = 1800.0, 9000.0
FRAME_W, FRAME_H = 800, 600

# The runner travels LEFTWARD, so the LEFT side faces the camera. That is the
# geometry `overstride.nearside` derives, and keeping the synthetic clip
# consistent with it means the near-side flag can be checked here too.
NEAR_SIDE = "left"


def _angles(t: float, period: float, phase: float):
    """Thigh and shank inclination for one leg, anterior-positive, degrees.

    Jogging-shaped: the thigh swings about 10 deg of mean flexion, and the knee
    flexes from nearly straight at contact to about 65 deg in swing. The knee
    angle this produces is `shank - thigh` by construction, which is exactly
    what `segment_angles` recomputes, so the test has a right answer.
    """
    w = 2.0 * np.pi * (t / period + phase)
    thigh = 10.0 + 25.0 * np.sin(w)
    knee = 35.0 - 32.0 * np.cos(w + 0.6)
    return thigh, thigh + knee


def skeleton(n_frames: int = 540, fps: float = 60.0, period: float = 0.75,
             speed_mm_s: float = 300.0) -> np.ndarray:
    """A (T, 17, 3) H36M skeleton in the camera frame: x right, y down, z depth.

    Forward kinematics in the sagittal plane. With the left hip nearer the
    camera (negative z) the sagittal basis puts anterior at -x, so the runner
    travels leftward across the image -- the direction that makes LEFT the near
    side under the geometric rule.
    """
    t = np.arange(n_frames) / fps
    kp = np.zeros((n_frames, 17, 3))

    down = np.array([0.0, 1.0, 0.0])
    ant = np.array([-1.0, 0.0, 0.0])          # anterior, for a leftward runner
    up = -down

    def place(origin, length, deg, ref):
        """`length` along a direction `deg` anterior of `ref`."""
        a = np.radians(deg)[:, None]
        return origin + length * (np.cos(a) * ref + np.sin(a) * ant)

    # the pelvis: drifts leftward relative to the panning camera, with a
    # two-per-cycle vertical bob
    px = 0.5 * speed_mm_s * t.max() - speed_mm_s * t
    py = -60.0 * np.cos(4.0 * np.pi * t / period)
    pelvis = np.stack([px, py, np.zeros_like(t)], axis=1)
    kp[:, H36M["pelvis"]] = pelvis

    # the trunk, leaning forward about 8 deg as a runner does
    lean = 8.0 + 1.5 * np.sin(4.0 * np.pi * t / period)
    thorax = place(pelvis, TRUNK, lean, up)
    kp[:, H36M["thorax"]] = thorax
    kp[:, H36M["spine"]] = 0.5 * (pelvis + thorax)
    kp[:, H36M["neck"]] = place(thorax, 0.35 * HEAD, lean, up)
    kp[:, H36M["head"]] = place(thorax, HEAD, lean, up)

    for side, phase, z in (("l", 0.0, -HIP_W / 2), ("r", 0.5, +HIP_W / 2)):
        hip = pelvis + np.array([0.0, 0.0, z])
        hip[:, 1] += 20.0
        kp[:, H36M[f"{side}_hip"]] = hip
        th, sh = _angles(t, period, phase)
        knee = place(hip, THIGH, th, down)
        kp[:, H36M[f"{side}_knee"]] = knee
        kp[:, H36M[f"{side}_ankle"]] = place(knee, SHANK, sh, down)

        # arms swing opposite the leg on the same side
        shoulder = thorax + np.array([0.0, 0.0, z * SHOULDER_W / HIP_W])
        kp[:, H36M[f"{side}_shoulder"]] = shoulder
        arm = -0.55 * (th - 10.0)
        elbow = place(shoulder, UPPER_ARM, arm, down)
        kp[:, H36M[f"{side}_elbow"]] = elbow
        kp[:, H36M[f"{side}_wrist"]] = place(elbow, FOREARM, arm + 35.0, down)
    return kp


def project(kp3d: np.ndarray) -> np.ndarray:
    """Pinhole projection to (T, 17, 3): u, v, confidence."""
    z = DEPTH + kp3d[:, :, 2]
    u = FRAME_W / 2 + FOCAL * kp3d[:, :, 0] / z
    v = FRAME_H * 0.52 + FOCAL * kp3d[:, :, 1] / z
    conf = np.full(u.shape, 0.87)
    return np.stack([u, v, conf], axis=2)


BONES = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
         (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15),
         (15, 16))
LEFT = {(0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13)}


def frames(kp2d: np.ndarray) -> list:
    """Draw the stick figure. Every pixel is generated here."""
    import cv2

    out = []
    ground = int(np.nanmax(kp2d[:, (H36M["l_ankle"], H36M["r_ankle"]), 1])
                 + 0.02 * FRAME_H)
    for f in range(kp2d.shape[0]):
        img = np.full((FRAME_H, FRAME_W, 3), 232, np.uint8)
        img[ground:] = 212
        # tick marks on the ground, so motion is visible frame to frame
        for x in range(-200, FRAME_W + 200, 120):
            xx = int(x - (kp2d[f, H36M["pelvis"], 0] % 120))
            if 0 <= xx < FRAME_W:
                cv2.line(img, (xx, ground), (xx, ground + 14), (196, 196, 196),
                         2, cv2.LINE_AA)
        cv2.line(img, (0, ground), (FRAME_W, ground), (170, 170, 170), 2,
                 cv2.LINE_AA)
        for a, b in BONES:
            pa = tuple(np.round(kp2d[f, a, :2]).astype(int))
            pb = tuple(np.round(kp2d[f, b, :2]).astype(int))
            colour = (92, 128, 168) if (a, b) in LEFT else (70, 78, 86)
            cv2.line(img, pa, pb, colour, 7, cv2.LINE_AA)
        for j in range(17):
            cv2.circle(img, tuple(np.round(kp2d[f, j, :2]).astype(int)), 5,
                       (40, 44, 48), -1, cv2.LINE_AA)
        out.append(img)
    return out


def synthetic_result(n_frames: int = 540, fps: float = 60.0,
                     period: float = 0.75, with_frames: bool = True):
    """An `AnalysisResult` for a clip that contains no dataset pixel.

    Returns `(result, frames, kp3d)`. The angles are computed by the frozen
    `segment_angles`, so they are the real quantity, not a stand-in.
    """
    kp3d = skeleton(n_frames, fps, period)
    kp2d = project(kp3d)
    angles = {s: {q: segment_angles(kp3d, s, EST_UP)[q] for q in REPORTED}
              for s in ("l", "r")}
    res = AnalysisResult(
        fps=float(fps), n_frames=n_frames, width=FRAME_W, height=FRAME_H,
        resized_from=None, detector_min_size=min(FRAME_W, FRAME_H),
        near_side="l", far_side="r", near_limb_from_depth="l",
        detected=np.ones(n_frames, bool), kp2d=kp2d,
        travel={"near_side": "l", "far_side": "r",
                "source": "--near-side (user declared)", "suggestion": "l",
                "suggestion_reason": "runner travels leftward across the image",
                "warnings": []},
        warnings=[], angles=angles, est3d=kp3d,
        detect_meta={"synthetic": True})
    return res, (frames(kp2d) if with_frames else None), kp3d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=REPO / "figures" / "example-report.html")
    ap.add_argument("--frames", type=int, default=540)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--stride-average", action="store_true",
                    help="opt in to stride averaging; M3's count bar was NOT "
                         "met (2 of 3 clips), so per-frame traces are the "
                         "default")
    ap.add_argument("--mp4", type=Path, default=None,
                    help="also write the generated clip as a video")
    args = ap.parse_args()

    from overstride.checks import post_detect_checks, pre_decode_checks
    from overstride.preprocess import ClipInfo
    from overstride.report import write_report
    from overstride.strides import segment

    res, imgs, _kp3d = synthetic_result(args.frames, args.fps)
    info = ClipInfo(path=Path("synthetic-runner.mp4"), fps=args.fps,
                    width=FRAME_W, height=FRAME_H, n_frames=args.frames,
                    duration_s=args.frames / args.fps)
    checks = (pre_decode_checks(info)
              + post_detect_checks(res.kp2d, res.detected, res.detect_meta))
    strides = segment(res.angles, res.fps) if args.stride_average else None

    out = write_report(args.out, res, strides=strides, checks=checks,
                       clip_name="synthetic-runner.mp4 (generated, not filmed)",
                       frames=imgs)
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB)")
    for c in checks:
        print(f"  {c.status:<7}{c.name:<12}{c.message[:88]}")
    if strides is not None:
        print(f"  strides: L{strides.count['l']} R{strides.count['r']}, "
              f"ok={strides.ok} {strides.reasons}")

    if args.mp4:
        import cv2
        w = cv2.VideoWriter(str(args.mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                            args.fps, (FRAME_W, FRAME_H))
        for img in imgs:
            w.write(img)
        w.release()
        print(f"wrote {args.mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
