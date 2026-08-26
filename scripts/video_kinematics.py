"""Sagittal hip and knee kinematics from a side-on running video.

WHAT THIS IS
  Video in, hip and knee flexion traces out, with a stated error budget. It is a
  KINEMATICS EXTRACTOR.

WHAT THIS IS NOT
  It says nothing about injury, and no output may be read that way. Phases 4B and
  5D of this project established that a per-runner injury or injured-limb verdict
  is not supportable from these signals -- see results/phase04b.md and
  results/phase05d.md. Do not add one.

THE PIPELINE
  decode -> Keypoint R-CNN COCO-17 -> temporal smoothing -> resample to the
  model's capture rate -> MotionAGFormer 2D-to-3D lift -> sagittal angles

  Three things here differ from the research path in scripts/phase3_infer.py, all
  measured in results/phase07.md rather than assumed:

  EDGE PADDING       phase 3 padded short windows with ALL-ZERO keypoints. Zeros
                     are out of distribution for a temporal transformer; edge
                     replication keeps the padded region on the manifold.
  OVERLAPPED WINDOWS phase 3 used consecutive windows with no shared context. A
                     10 s phone clip is 4-8 windows against the test set's usual
                     2, so seams matter far more here. Windows overlap 50% and
                     blend with a triangular weight.
  RATE MATCHING      the model's receptive field is 81 FRAMES, and the capture
                     rate it was trained on is ~120 fps (recovered from stride
                     cadence in phase 7A). A 30 fps clip has 4x the motion per
                     frame. Keypoints are interpolated up to the model rate
                     before lifting, and angles resampled back afterwards.

  Lifting is SCALE-FREE. phase 3 denormalised with a scale derived from
  ground-truth 3D, which real video does not have -- but the angles are computed
  in the body's own frame and are scale-invariant, so the step is unnecessary.

Run: .venv/Scripts/python.exe scripts/video_kinematics.py --video CLIP.mp4 --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
AP = REPO / "data" / "athleticspose"
sys.path.insert(0, str(AP / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from phase3_angles import sagittal_angles  # noqa: E402
from phase3_infer import load_model  # noqa: E402
from phase7_inference_fix import lift  # noqa: E402

MODEL_FPS = 120.0          # phase 7A, recovered from stride cadence
PERSON_SCORE = 0.90
LOWPASS_HZ = 10.0          # matches the Ferber pipeline's own filter
PLAUSIBLE = {"hip": (-40.0, 60.0), "knee": (-170.0, 20.0)}


def decode(path: Path, every: int, max_frames: int) -> tuple[list, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return frames, fps / every


def detect_coco17(frames: list) -> tuple[np.ndarray, dict]:
    """COCO-17 keypoints per frame from torchvision Keypoint R-CNN.

    Its keypoint order is already exactly what the lifter expects: nose, eyes,
    ears, shoulders, elbows, wrists, hips, knees, ankles.
    """
    from torchvision.models.detection import (KeypointRCNN_ResNet50_FPN_Weights,
                                              keypointrcnn_resnet50_fpn)
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    det = keypointrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.5)
    det.eval()

    kp = np.zeros((len(frames), 17, 3), dtype=np.float64)
    found, crowded = 0, 0
    with torch.no_grad():
        for i, bgr in enumerate(frames):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            out = det([t])[0]
            scores = out["scores"].numpy()
            if len(scores) == 0:
                continue
            if int((scores > PERSON_SCORE).sum()) > 1:
                crowded += 1
            best = int(np.argmax(scores))
            if scores[best] < 0.5:
                continue
            kp[i, :, :2] = out["keypoints"][best].numpy()[:, :2]
            kp[i, :, 2] = out["keypoints_scores"][best].numpy()
            found += 1
            if (i + 1) % 25 == 0:
                print(f"    detected {i + 1}/{len(frames)} frames")
    return kp, {"frames_with_person": found, "frames_crowded": crowded}


def smooth(kp: np.ndarray, fps: float) -> np.ndarray:
    """Low-pass the 2D tracks. Keypoint R-CNN is frame-independent and jitters;
    the lifter was trained on temporally coherent input."""
    from scipy.ndimage import gaussian_filter1d
    sd = fps / (2.0 * np.pi * LOWPASS_HZ)
    if sd <= 0.3:
        return kp
    out = kp.copy()
    out[:, :, :2] = gaussian_filter1d(kp[:, :, :2], sd, axis=0, mode="nearest")
    return out


def resample(seq: np.ndarray, n_out: int) -> np.ndarray:
    src = np.linspace(0.0, 1.0, len(seq))
    dst = np.linspace(0.0, 1.0, n_out)
    flat = seq.reshape(len(seq), -1)
    got = np.stack([np.interp(dst, src, flat[:, c]) for c in range(flat.shape[1])], 1)
    return got.reshape(n_out, *seq.shape[1:])


def check(angles: dict, kp: np.ndarray, meta: dict, n: int) -> list[str]:
    """No ground truth is available, so flag implausible output rather than
    emitting it silently."""
    warn = []
    if meta["frames_with_person"] < n:
        warn.append(f"no person detected in {n - meta['frames_with_person']} of "
                    f"{n} frames")
    if meta["frames_crowded"]:
        warn.append(f"MORE THAN ONE PERSON in {meta['frames_crowded']} frames -- "
                    "this tool does not track across people; use a single-subject "
                    "clip")
    conf = float(np.mean(kp[:, 11:17, 2]))
    if conf < 3.0:
        warn.append(f"low lower-body keypoint confidence (mean {conf:.2f})")
    for side in ("l", "r"):
        for joint, (lo, hi) in PLAUSIBLE.items():
            v = angles[side][joint]
            frac = float(np.mean((v < lo) | (v > hi)))
            if frac > 0.05:
                warn.append(f"{side} {joint}: {100 * frac:.0f}% of frames outside "
                            f"the plausible range [{lo:.0f}, {hi:.0f}] deg")
    lv, rv = angles["l"]["knee"], angles["r"]["knee"]
    if not (np.any(np.isfinite(lv)) and np.any(np.isfinite(rv))):
        return warn + ["knee traces are entirely non-finite"]
    lk, rk = lv - np.nanmean(lv), rv - np.nanmean(rv)
    if np.all(np.isfinite(lk)) and np.all(np.isfinite(rk)) and len(lk) > 20:
        r = float(np.corrcoef(lk, rk)[0, 1])
        if r > 0.5:
            warn.append(f"left and right knee traces move together (r={r:+.2f}); "
                        "in running they should alternate -- likely a tracking "
                        "failure")
    return warn


def error_budget() -> list[str]:
    b = ["lifting error 3.4 deg mean sagittal MAE -- measured on AthleticsPose "
         "held-out subjects, ITS OWN fine-tuned detections, a track capture rig"]
    # Prefer the full-n measurement. The 200-clip file is a long-weighted
    # subsample kept for the record; quoting it here would overstate the cost.
    p7 = next((f for f in (REPO / "results" / "phase7_inference_full.json",
                           REPO / "results" / "phase7_inference.json")
               if f.exists()), None)
    if p7 is not None:
        d = json.loads(p7.read_text(encoding="utf-8"))
        b.append(f"+ generic COCO detector penalty "
                 f"{d['generic_detector_cost_deg']:+.2f} deg -- measured in "
                 "phase 7A by swapping the fine-tuned detections for generic "
                 f"COCO ones on the same {d['n_clips']} clips")
    b += ["+ UNQUANTIFIED Keypoint R-CNN error on your footage -- AthleticsPose "
          "does not release its source videos, so there is no video with ground "
          "truth to measure this against",
          "+ UNQUANTIFIED domain gap -- the checkpoint was fine-tuned on a track "
          "rig; phone video is a different domain",
          "ankle dorsiflexion is unavailable: H36M-17 has no toe keypoint",
          "NON-COMMERCIAL ONLY -- AthleticsPose CC BY-NC-SA 4.0, AthletePose3D "
          "research use only"]
    return b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--every", type=int, default=1, help="use every Kth frame")
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("NOTE: this extracts KINEMATICS. It says nothing about injury.\n")
    print(f"decoding {args.video}")
    frames, fps = decode(args.video, args.every, args.max_frames)
    if len(frames) < 20:
        raise SystemExit(f"only {len(frames)} frames decoded; need at least 20")
    h, w = frames[0].shape[:2]
    print(f"  {len(frames)} frames at {fps:.1f} fps, {w}x{h}")

    print("detecting 2D keypoints (Keypoint R-CNN, COCO-17)")
    kp, meta = detect_coco17(frames)

    # Refuse rather than emit. A CSV of angles computed from frames with no
    # subject in them looks exactly like a real result, which is worse than no
    # output at all.
    seen = meta["frames_with_person"]
    if seen < 0.5 * len(frames):
        print(f"\nABORTED: a person was detected in only {seen} of "
              f"{len(frames)} frames.")
        print("  Nothing was written -- angles from these frames would be "
              "meaningless.")
        print("  This tool needs a single runner, filmed side-on, in frame "
              "throughout.")
        return 2
    kp = smooth(kp, fps)

    n_model = max(81, int(round(len(kp) * MODEL_FPS / fps)))
    print(f"resampling {len(kp)} frames @ {fps:.0f} fps -> {n_model} @ "
          f"{MODEL_FPS:.0f} fps (the rate the lifter was trained at)")
    kp_model = resample(kp, n_model)

    print("lifting to 3D (edge padding, overlapped windows, scale-free)")
    # Keypoint R-CNN is a general-purpose COCO detector, so the matched
    # checkpoint is the COCO one -- not the fine-tuned-detector weights. Pairing
    # them wrongly is gap 1 in docs/AGENT_CONTEXT.md.
    model = load_model("motionagformer-b-ath-det-coco-v1.ckpt", "base")
    if model is None:
        raise SystemExit("checkpoint missing -- see docs for the download")
    est = lift(model, kp_model, pad="edge", windows="overlap")

    angles = {}
    for side in ("l", "r"):
        a = sagittal_angles(est, side)
        angles[side] = {j: resample(a[j][:, None], len(kp))[:, 0]
                        for j in ("hip", "knee")}

    warns = check(angles, kp, meta, len(frames))

    stem = args.video.stem
    import csv
    with (args.out / f"{stem}_angles.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "time_s", "left_hip_deg", "left_knee_deg",
                     "right_hip_deg", "right_knee_deg"])
        for i in range(len(kp)):
            wr.writerow([i, round(i / fps, 4)]
                        + [round(float(angles[s][j][i]), 3)
                           for s in ("l", "r") for j in ("hip", "knee")])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(kp)) / fps
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, joint in zip(axes, ("hip", "knee")):
        ax.plot(t, angles["l"][joint], color="#1f6fb4", lw=1.8, label="left")
        ax.plot(t, angles["r"][joint], color="#c0392b", lw=1.8, label="right")
        ax.set_ylabel(f"{joint} flexion (deg)")
        ax.grid(alpha=0.18)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"Sagittal kinematics — {stem}", fontsize=12)
    fig.text(0.01, 0.015, "Kinematics only. This says nothing about injury. "
             "Lifting error >=3.4 deg plus unquantified detection error on this "
             "footage — see the JSON for the full budget.",
             fontsize=7.4, color="0.35")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.13)
    fig.savefig(args.out / f"{stem}_kinematics.png", dpi=200)

    budget = error_budget()
    (args.out / f"{stem}_run.json").write_text(json.dumps({
        "video": str(args.video), "generated": datetime.now(timezone.utc).isoformat(),
        "frames": len(kp), "fps": fps, "resolution": [w, h],
        "model_fps": MODEL_FPS, "detector": "torchvision keypointrcnn_resnet50_fpn",
        "lifter": "motionagformer-b-ath-det-coco-v1",
        "padding": "edge", "windows": "overlap-50pct", "scale_free": True,
        "detection": meta, "warnings": warns, "error_budget": budget,
        "not_an_injury_tool": "Kinematics only. Phases 4B and 5D of this project "
                              "found a per-runner injury verdict unsupportable.",
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {args.out}/{stem}_angles.csv, _kinematics.png, _run.json")
    print("\n=== ERROR BUDGET ===")
    for b in budget:
        print(f"  - {b}")
    if warns:
        print("\n=== WARNINGS ===")
        for w_ in warns:
            print(f"  ! {w_}")
    else:
        print("\nno plausibility warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
