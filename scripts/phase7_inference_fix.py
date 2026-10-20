"""Phase 7A -- is the lifting path sound, and what does a generic detector cost?

Building a tool that runs on real video exposed three properties of
`phase3_infer.predict()` that were never examined, two of which affect the
published 3.4 deg:

  ZERO PADDING     every sequence is padded to a multiple of 81 frames with
                   ALL-ZERO keypoints -- a mean of 14.9 frames of 81 across the
                   held-out clips. Zeros are out of distribution for a temporal
                   transformer and can reach valid frames through attention.
  HARD WINDOWS     windows are consecutive with no shared context, so the model
                   sees nothing across a boundary. Held-out clips typically reach
                   2 windows; a 10 s phone clip at 60 fps reaches 8.
  FRAME RATE       the receptive field is 81 FRAMES, not seconds, and the capture
                   rate is undocumented. Section A recovers it from the data.

And one thing the video tool needs that is measurable with data already on disk:
`det_markers2d_by_cam_coco` holds GENERIC COCO detections for the same clips as
the fine-tuned `det_markers2d_by_cam_ft`. The gap between them bounds what using
an off-the-shelf COCO detector costs -- which is exactly what torchvision's
Keypoint R-CNN is.

Design: one pass over the clips, every configuration scored on the SAME clips, so
each fix is a paired delta rather than two independent means.

Run: .venv/Scripts/python.exe scripts/phase7_inference_fix.py [--clips N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
AP = REPO / "data" / "athleticspose"
sys.path.insert(0, str(AP / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from athleticspose.utils import normalize_kpts  # noqa: E402

from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import CLIP, DATA, TEST_SUBJECTS, load_model  # noqa: E402

OUT = REPO / "results" / "phase7_inference.json"
ACTIONS = ["running", "sprint"]

# The release ships one checkpoint PER INPUT TYPE, all identical architecture
# (11,721,795 params). Pairing COCO detections with the fine-tuned-detector
# weights conflates "worse detector" with "wrong weights for that detector",
# which is what the first version of this experiment did.
CKPT_FOR = {"ft": "motionagformer-b-ath-det-ft-v1.ckpt",
            "coco": "motionagformer-b-ath-det-coco-v1.ckpt"}

# baseline == exactly what phase 3 did
CONFIGS = [
    ("baseline (phase 3)",      "ft",   "zero", "consecutive"),
    ("edge padding",            "ft",   "edge", "consecutive"),
    ("overlapped windows",      "ft",   "zero", "overlap"),
    ("both fixes",              "ft",   "edge", "overlap"),
    ("generic COCO detector",   "coco", "zero", "consecutive"),
    ("generic COCO + fixes",    "coco", "edge", "overlap"),
]


def pad_to(chunk: np.ndarray, n: int, mode: str) -> np.ndarray:
    """Extend a short chunk to n frames.

    'zero' is what phase 3 does. 'edge' replicates the last real frame, which
    keeps the padded region on the manifold the model was trained on instead of
    handing it a pose with every joint at the origin.
    """
    short = n - len(chunk)
    if short <= 0:
        return chunk
    if mode == "zero":
        return np.pad(chunk, [(0, short), (0, 0), (0, 0)])
    return np.pad(chunk, [(0, short), (0, 0), (0, 0)], mode="edge")


@torch.no_grad()
def lift(model, det2d: np.ndarray, pad: str, windows: str,
         scale: float = 1.0, with_weights: bool = False):
    """2D COCO-17 -> 3D. Scale-free by default; angles do not depend on it.

    with_weights also returns the per-frame blend weight each output frame was
    built from: the sum of the triangular weights of the windows covering it.
    A frame at a window centre carries ~1.0, a frame at a seam less, and an
    edge-padded frame least of all -- which is a video-only signal for phase 14,
    and the reason it is read out here rather than reimplemented beside this
    function where it could drift from it.
    """
    t = det2d.shape[0]
    coords = np.concatenate([det2d[:, :, :2],
                             np.zeros_like(det2d[:, :, :1])], axis=-1)
    coords_norm, _ = normalize_kpts(coords)
    x = np.concatenate([coords_norm[:, :, :2], det2d[:, :, 2:3]], axis=-1)

    if windows == "consecutive":
        out = np.empty((t, 17, 3), dtype=np.float64)
        for start in range(0, t, CLIP):
            chunk = x[start:start + CLIP]
            valid = len(chunk)
            inp = torch.from_numpy(pad_to(chunk, CLIP, pad)[None]).float()
            out[start:start + valid] = model(inp)[0].numpy()[:valid]
        if with_weights:
            # every frame is covered exactly once, with no blending
            return out * scale, np.ones(t)
        return out * scale

    # 50%-overlap sliding windows, blended with a triangular weight so there is
    # no seam and every interior frame is covered by two windows.
    stride = CLIP // 2
    acc = np.zeros((t, 17, 3))
    wts = np.zeros((t, 1, 1))
    starts = list(range(0, max(t - CLIP, 0) + 1, stride))
    if starts and starts[-1] + CLIP < t:
        starts.append(t - CLIP)
    if not starts:
        starts = [0]
    tri = 1.0 - np.abs(np.linspace(-1.0, 1.0, CLIP))
    tri = np.maximum(tri, 1e-3).reshape(CLIP, 1, 1)
    for s in starts:
        chunk = x[s:s + CLIP]
        valid = len(chunk)
        inp = torch.from_numpy(pad_to(chunk, CLIP, pad)[None]).float()
        got = model(inp)[0].numpy()
        acc[s:s + valid] += got[:valid] * tri[:valid]
        wts[s:s + valid] += tri[:valid]
    lifted = (acc / np.maximum(wts, 1e-9)) * scale
    if with_weights:
        return lifted, wts.reshape(t)
    return lifted


def clip_list(limit: int | None) -> list[tuple]:
    """Held-out clips present in BOTH detector sources, longest first so the
    windowing configurations are actually exercised."""
    out = []
    for action in ACTIONS:
        for subj in TEST_SUBJECTS:
            d = DATA / "det_markers2d_by_cam_ft" / action / subj
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.npy")):
                coco = DATA / "det_markers2d_by_cam_coco" / action / subj / f.name
                gt = DATA / "gt_markers3d_by_cam" / action / subj / f"{f.stem}.npz"
                if coco.exists() and gt.exists():
                    out.append((action, subj, f.stem, len(np.load(f))))
    out.sort(key=lambda r: -r[3])
    if limit:
        # half the longest clips, half sampled across the rest, so both the
        # multi-window and the single-padded-window paths are represented
        half = limit // 2
        rest = out[half:]
        step = max(1, len(rest) // max(limit - half, 1))
        out = out[:half] + rest[::step][:limit - half]
    return out


def infer_capture_fps() -> dict:
    """Recover the capture rate from the data, since it is undocumented.

    Ankle vertical position oscillates once per stride for that limb. Counting
    those cycles gives strides/frame; dividing a plausible running cadence by it
    gives frames/second. Reported as a range rather than a single number, since
    the cadence assumption is the weak link.
    """
    per_clip = []
    for action in ACTIONS:
        for subj in TEST_SUBJECTS:
            d = DATA / "gt_markers3d_by_cam" / action / subj
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.npz"))[:40]:
                g = np.asarray(np.load(f)["markers_h36m"])
                if len(g) < 60:
                    continue
                y = g[:, H36M["r_ankle"], 1].astype(float)
                y = y - y.mean()
                # cycles = zero crossings / 2
                crossings = int(np.sum(np.diff(np.signbit(y)) != 0))
                if crossings >= 2:
                    per_clip.append(crossings / 2.0 / len(g))
    spf = float(np.median(per_clip))          # strides per frame
    return {"n_clips": len(per_clip), "strides_per_frame": spf,
            "fps_at_cadence_2.5": 2.5 / spf, "fps_at_cadence_3.0": 3.0 / spf}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=200,
                    help="0 = every held-out clip")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated config indices, e.g. 0,3")
    ap.add_argument("--out", type=str, default="",
                    help="output filename under results/; defaults by scope so "
                         "a full-n run never clobbers the 200-clip result")
    args = ap.parse_args()

    print("=== A. capture rate, recovered from the data ===")
    fps = infer_capture_fps()
    print(f"  {fps['n_clips']} clips | {fps['strides_per_frame']:.4f} strides/frame")
    print(f"  => {fps['fps_at_cadence_2.5']:.0f}-{fps['fps_at_cadence_3.0']:.0f} fps "
          "for a running cadence of 2.5-3.0 strides/s")
    print(f"  81-frame window covers "
          f"{81 / fps['fps_at_cadence_3.0']:.2f}-{81 / fps['fps_at_cadence_2.5']:.2f} s\n")

    global CONFIGS
    if args.only:
        CONFIGS = [CONFIGS[int(i)] for i in args.only.split(",")]
    clips = clip_list(args.clips or None)
    n_multi = sum(1 for c in clips if c[3] > CLIP)
    print(f"=== B. inference-path ablation on {len(clips)} held-out clips ===")
    print(f"  {n_multi} exceed the 81-frame window (multi-window path)\n")

    models = {}
    for det in sorted({c[1] for c in CONFIGS}):
        models[det] = load_model(CKPT_FOR[det], "base")
        if models[det] is None:
            raise SystemExit(f"checkpoint missing: {CKPT_FOR[det]}")
        print(f"  {det:5} -> {CKPT_FOR[det]}")

    err = {name: {limb: {"hip": [], "knee": []} for limb in ("near", "far")}
           for name, *_ in CONFIGS}
    # Per-clip rows, carrying the subject id. The aggregate means below are
    # unchanged; this exists because a mean with no rows behind it cannot be
    # given a subject-grouped confidence interval, and the phase 13 gate asks
    # for one. Clips are not independent -- 8 cameras share a trial -- so the
    # interval must resample SUBJECTS, which needs the subject recorded here.
    per_clip: list[dict] = []
    for i, (action, subj, stem, _n) in enumerate(clips):
        gt3d = np.asarray(np.load(
            DATA / "gt_markers3d_by_cam" / action / subj / f"{stem}.npz")["markers_h36m"])
        srcs = {}
        for tag, folder in (("ft", "det_markers2d_by_cam_ft"),
                            ("coco", "det_markers2d_by_cam_coco")):
            srcs[tag] = np.load(DATA / folder / action / subj / f"{stem}.npy").astype(float)
        n = min(len(gt3d), min(len(v) for v in srcs.values()))
        if n < 20:
            continue
        gt3d = gt3d[:n]
        near = ("l" if gt3d[:, H36M["l_hip"], 2].mean()
                < gt3d[:, H36M["r_hip"], 2].mean() else "r")
        far = "r" if near == "l" else "l"
        # The published figures are the NEAR limb, and the aggregate below still
        # is -- nothing about them changes here. The FAR limb is scored
        # alongside because the measurement track forbids reporting the two
        # pooled, and the phase 13 gate asks the far limb to "improve or stay
        # within its CI" -- which needs a far-limb baseline to compare against.
        # There was none.
        ga = {"near": sagittal_angles(gt3d, near),
              "far": sagittal_angles(gt3d, far)}

        for name, det, pad, win in CONFIGS:
            est = lift(models[det], srcs[det][:n], pad, win)
            ea = {"near": sagittal_angles(est, near),
                  "far": sagittal_angles(est, far)}
            row = {"config": name, "action": action, "subject": subj,
                   "clip": stem, "near": near, "n_frames": int(n)}
            for limb in ("near", "far"):
                for joint in ("hip", "knee"):
                    d = np.abs(ea[limb][joint] - ga[limb][joint])
                    ok = bool(np.all(np.isfinite(d)))
                    row[f"{limb}_{joint}_mae"] = float(d.mean()) if ok else None
                    if ok:
                        err[name][limb][joint].append(float(d.mean()))
            per_clip.append(row)
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(clips)} clips")

    base = CONFIGS[0][0]
    rows = []
    for name, det, pad, win in CONFIGS:
        hip = float(np.mean(err[name]["near"]["hip"]))
        knee = float(np.mean(err[name]["near"]["knee"]))
        mean = (hip + knee) / 2
        fhip = float(np.mean(err[name]["far"]["hip"]))
        fknee = float(np.mean(err[name]["far"]["knee"]))
        rows.append({"config": name, "detector": det, "padding": pad,
                     "windows": win, "hip_mae": hip, "knee_mae": knee,
                     "mean_mae": mean, "n": len(err[name]["near"]["hip"]),
                     "far_hip_mae": fhip, "far_knee_mae": fknee,
                     "far_mean_mae": (fhip + fknee) / 2,
                     "n_far": len(err[name]["far"]["hip"])})
    bmean = rows[0]["mean_mae"]

    print(f"\n=== VERDICT ===")
    print(f"  {'configuration':<24}{'detector':>9}{'pad':>7}{'windows':>13}"
          f"{'hip':>8}{'knee':>8}{'mean':>8}{'vs base':>10}"
          f"{'far hip':>9}{'far knee':>10}{'far mean':>10}")
    for r in rows:
        print(f"  {r['config']:<24}{r['detector']:>9}{r['padding']:>7}"
              f"{r['windows']:>13}{r['hip_mae']:>8.2f}{r['knee_mae']:>8.2f}"
              f"{r['mean_mae']:>8.2f}{r['mean_mae'] - bmean:>+10.2f}"
              f"{r['far_hip_mae']:>9.2f}{r['far_knee_mae']:>10.2f}"
              f"{r['far_mean_mae']:>10.2f}")
    print("  near and far are reported separately, never pooled.")

    print(f"\n  phase 3 published 3.4 deg on the full 592 clips.")
    print(f"  baseline here on {rows[0]['n']} clips: {bmean:.2f} deg")
    # look these up by name -- --only subsets CONFIGS, so positional indices
    # are not stable
    by_name = {r["config"]: r for r in rows}
    fixes = gap = float("nan")
    if "both fixes" in by_name:
        fixes = by_name["both fixes"]["mean_mae"] - bmean
        print(f"  effect of the inference fixes: {fixes:+.2f} deg -> "
              + ("worth adopting" if abs(fixes) >= 0.15 else "negligible"))
    if "generic COCO detector" in by_name:
        gap = by_name["generic COCO detector"]["mean_mae"] - bmean
        print(f"  cost of a generic COCO detector: {gap:+.2f} deg "
              f"({by_name['generic COCO detector']['mean_mae']:.2f} vs {bmean:.2f})")

    # The 200-clip subsample result is published. A different scope is a
    # different measurement and gets a different file.
    if args.out:
        out_path = OUT.with_name(args.out)
    elif args.only:
        out_path = OUT.with_name(OUT.stem + "_partial.json")
    elif args.clips == 0:
        out_path = OUT.with_name(OUT.stem + "_full.json")
    else:
        out_path = OUT
    out_path.write_text(json.dumps({"capture_fps": fps, "n_clips": len(clips),
                               "n_multiwindow": n_multi, "configs": rows,
                               "n_subjects": len({c[1] for c in clips}),
                               "fix_effect_deg": fixes,
                               "generic_detector_cost_deg": gap}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out_path.relative_to(REPO)}")
    rows_path = out_path.with_name(out_path.stem + "_rows.json")
    rows_path.write_text(json.dumps(
        {"source": "scripts/phase7_inference_fix.py",
         "n_clips": len(clips), "n_rows": len(per_clip),
         "note": "one row per (config, clip); a null MAE is a non-finite angle "
                 "trace, excluded from that config mean exactly as before",
         "rows": per_clip}, indent=2), encoding="utf-8")
    print(f"wrote {rows_path.relative_to(REPO)} ({len(per_clip)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
