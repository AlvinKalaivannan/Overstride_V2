"""Phase 15 -- the frozen pipeline on BML MoVi, a lab it has never seen.

Nothing here is new estimation code. Every stage is the function the deployable
tool already runs, in the configuration frozen in `results/phase15.md` (commit
64fb0bc, written before a single MoVi byte was on disk):

    decode -> detect_coco17(crop=True) -> coco17_to_h36m17 -> NO smoothing
           -> resample to MODEL_FPS -> lift(edge, overlap, det-coco ckpt)
           -> sagittal_angles

`CLAUDE.md`: *"There is no pass/fail gate. A loss on held-out data is a result
and goes in the README exactly as prominently as a win."*

Round F only. Round S's export has no hip, knee or pelvis markers -- see
`docs/data-inventory.md` 8.5 and `scripts/phase15_checks.py`.

TIME ALIGNMENT IS EXACT, NOT ESTIMATED. MoVi ships `flags30` and `flags120`:
the same movement boundaries in video and in mocap frames, measured at a ratio
of 3.9981. So video frame k of the jogging window corresponds to mocap frame
`f120[0] + 4k`, and the estimate is scored against the ground truth AT THOSE
FRAMES. The reference 3D is never resampled, smoothed or re-derived, per
`CLAUDE.md`. That also satisfies the pre-registered 30 Hz control by
construction: the comparison already happens on a 30 Hz grid.

Run: .venv/Scripts/python.exe scripts/phase15_movi.py [--max-frames N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data" / "athleticspose" / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import load_model  # noqa: E402
from phase7_inference_fix import CKPT_FOR, lift  # noqa: E402
from phase12_scan import subject_bootstrap_ci  # noqa: E402
from phase15_checks import (angle_mask, camera_pose, load_v3d,  # noqa: E402
                            to_camera, to_h36m)
from video_kinematics import (LOWPASS_HZ, MODEL_FPS, coco17_to_h36m17,  # noqa: E402
                              detect_coco17, resample, scores_to_train_scale)

MOVI = REPO / "data" / "movi" / "raw" / "round_f"
MOCAP = MOVI / "mocap"
OUT = REPO / "results" / "phase15_movi.json"

DET, PAD, WINDOWS = "coco", "edge", "overlap"
JOINTS = ("hip", "knee")
LIMBS = ("near", "far")
MOCAP_PER_VIDEO = 4          # flags120 / flags30, measured 3.9981

# AthleticsPose deployable baseline, for the side-by-side. NOT "the same clips":
# a different dataset, camera, population and movement.
AP_NEAR, AP_FAR = 3.5093, 3.6177

CAMERAS = {
    "PG1": {"pattern": "F_PG1_Subject_{s}_L.avi", "calibrated": True},
    "PG2": {"pattern": "F_PG2_Subject_{s}_L.avi", "calibrated": True},
    "CP1": {"pattern": "F_CP1_Subject_{s}.mp4", "calibrated": False},
}


def decode_range(path: Path, a: int, b: int) -> tuple[list, float, dict]:
    """Decode video frames [a, b] inclusive, and read the rate off the file.

    `video_kinematics.decode` always starts at frame 0 and takes no range, and
    a MoVi round runs the whole 21-movement protocol, so decoding the lot to
    use 206 frames of it would cost minutes per clip for nothing.

    Read sequentially and discard, rather than seeking: `CAP_PROP_POS_FRAMES`
    on these containers lands on the nearest keyframe, which would shift the
    whole clip against the mocap by an unknown number of frames -- and the point
    of using `flags30` is that the alignment is exact.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    meta = {"fps_from_file": float(cap.get(cv2.CAP_PROP_FPS)),
            "n_frames_from_file": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    frames, i = [], 0
    while i <= b:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= a:
            frames.append(frame)
        i += 1
    cap.release()
    meta["n_decoded"] = len(frames)
    meta["range_requested"] = [a, b]
    return frames, meta["fps_from_file"], meta


def jogging_window(v: dict) -> tuple[int, np.ndarray, np.ndarray]:
    """(index, flags30, flags120) for this subject's jogging trial.

    The index is READ PER SUBJECT. `motions_list` order is randomised: jogging
    is 18 for subject 1 and 16 for subject 10, and `running_in_spot` -- which is
    excluded, being running without flight-phase translation -- moves from 7 to
    19. A hardcoded index would silently measure a different movement.
    """
    idx = [i for i, m in enumerate(v["motions"])
           if m.strip().lower() == "jogging"]
    if not idx:
        raise SystemExit(f"{v['subject_id']}: no jogging trial")
    i = idx[0]
    return i, v["flags30"][i], v["flags120"][i]


def near_limb(gt_cam: np.ndarray | None, est: np.ndarray) -> tuple[str, str]:
    """Which limb is nearer the camera. Depth is axis 2.

    With calibration the ground truth answers it. Without (the hand-held phone
    cameras ship none), the ESTIMATE answers it -- phase 14 found the near limb
    recoverable from the estimate alone on 592 of 592 AthleticsPose clips, and
    that is recorded wherever it is used rather than assumed silently.
    """
    src = gt_cam if gt_cam is not None else est
    lo = np.nanmean(src[:, H36M["l_hip"], 2])
    hi = np.nanmean(src[:, H36M["r_hip"], 2])
    near = "l" if lo < hi else "r"
    return near, ("r" if near == "l" else "l")


def run_clip(subject: str, cam: str, mat: Path, video: Path, det_model,
             lift_model, max_frames: int) -> dict | None:
    v = load_v3d(mat)
    ji, f30, f120 = jogging_window(v)
    gt_world = to_h36m(v)

    a30, b30 = int(f30[0]), int(f30[1])
    if max_frames:
        b30 = min(b30, a30 + max_frames - 1)
    t0 = time.time()
    frames, fps, vmeta = decode_range(video, a30, b30)
    if len(frames) < 20:
        print(f"    {cam}: only {len(frames)} frames decoded, skipping")
        return None

    kp_coco, dmeta = detect_coco17(frames, crop=True, model=det_model)
    kp = coco17_to_h36m17(kp_coco)
    assert LOWPASS_HZ == 0.0, "phase 13 measured 2D smoothing harmful"

    # A frame where the detector found nobody is left as ALL ZEROS by
    # `detect_coco17`. That is a sentinel, exactly like MoVi's own gap zeros,
    # and the lifter will happily lift it into a skeleton at the origin. The
    # deployable tool is protected by `video_kinematics.check`, which refuses to
    # write output when it cannot find the subject; this script does its own
    # scoring, so it has to drop those frames itself. They are excluded with
    # their n reported, never interpolated.
    detected = np.abs(kp[:, :, :2]).sum(axis=(1, 2)) > 0

    # Channel 2 onto the scale the lifter was trained against. torchvision's
    # raw keypoint scores run ~14x high and destroyed the estimate: 30 deg with
    # r = +0.02, against 4.5 deg for the same clip's projected reference 2D.
    # See the TRAIN_CONF_MEAN comment in video_kinematics.
    raw_conf = kp[:, :, 2].copy()
    kp = scores_to_train_scale(kp)

    n_v = len(frames)
    n_model = max(int(round(n_v * MODEL_FPS / max(fps, 1e-6))), 2)
    est_model = lift(lift_model, resample(kp, n_model), PAD, WINDOWS)
    est = resample(est_model, n_v)                   # back onto the video grid

    # Ground truth at the EXACT corresponding mocap frames. Never resampled.
    mk = int(f120[0]) + MOCAP_PER_VIDEO * np.arange(n_v)
    keep_mk = (mk < gt_world.shape[0]) & detected
    mk, est = mk[keep_mk], est[keep_mk]
    gt = gt_world[mk]
    if len(mk) < 20:
        print(f"    {cam}: only {len(mk)} frames with a detected subject of "
              f"{n_v} -- refusing to score this clip")
        return {"subject": subject, "camera": cam, "video": video.name,
                "refused": "detector found the subject in too few frames",
                "n_video_frames": n_v, "n_detected": int(detected.sum()),
                "detector_meta": dmeta, "video_meta": vmeta}

    gt_cam = None
    if CAMERAS[cam]["calibrated"]:
        R_cv, t_cv, _c = camera_pose(cam)
        gt_cam = to_camera(gt, R_cv, t_cv)
    near, far = near_limb(gt_cam, est)

    err: dict = {}
    n_scored = {}
    for limb, side in (("near", near), ("far", far)):
        m = angle_mask(gt, side)
        ga = sagittal_angles(gt, side)
        ea = sagittal_angles(est, side)
        err[limb] = {}
        for j in JOINTS:
            gv, ev = ga[j][m], ea[j][m]
            ok = np.isfinite(gv) & np.isfinite(ev)
            d = np.abs(gv - ev)[ok]
            err[limb][j] = float(d.mean()) if d.size else float("nan")
            # An MAE on its own cannot distinguish a degraded measurement from
            # an unrelated one. Pearson r against the reference says which:
            # near 0 means the trace carries no signal, and that is a different
            # result from a large but correlated error.
            err[limb][f"{j}_r"] = (
                float(np.corrcoef(gv[ok], ev[ok])[0, 1])
                if ok.sum() > 5 and np.std(gv[ok]) > 1e-6
                and np.std(ev[ok]) > 1e-6 else float("nan"))
            err[limb][f"{j}_gt_range_deg"] = (
                [round(float(gv[ok].min()), 1), round(float(gv[ok].max()), 1)]
                if ok.any() else [float("nan")] * 2)
        n_scored[limb] = int(m.sum())

    dt = time.time() - t0
    row = {"subject": subject, "camera": cam, "jogging_index": ji,
           "video": video.name, "video_meta": vmeta,
           "fps_from_file": fps, "n_video_frames": n_v,
           "n_frames_scored": n_scored,
           "near_limb": near, "near_limb_from": (
               "ground truth (calibrated)" if gt_cam is not None
               else "the estimate (no calibration for this camera)"),
           "mocap_frames": [int(mk[0]), int(mk[-1])],
           "detector_meta": dmeta,
           "n_detected": int(detected.sum()),
           "raw_conf_range": [round(float(raw_conf[raw_conf > 0].min()), 3),
                              round(float(raw_conf.max()), 3)],
           "detection_rate": round(float(detected.mean()), 4),
           "err": err, "seconds": round(dt, 1),
           "ms_per_frame": round(1000 * dt / max(n_v, 1), 1)}
    for limb in LIMBS:
        row["err"][limb]["mean"] = float(np.nanmean(
            [err[limb][j] for j in JOINTS]))
    print(f"    {cam}: {int(detected.sum())}/{n_v} detected @ {fps:.3f} fps  "
          f"near hip {err['near']['hip']:.3f} knee {err['near']['knee']:.3f} "
          f"| far hip {err['far']['hip']:.3f} knee {err['far']['knee']:.3f}  "
          f"(r hip {err['near']['hip_r']:+.2f} knee "
          f"{err['near']['knee_r']:+.2f}, {row['ms_per_frame']:.0f} ms/frame)")
    return row


def aggregate(rows: list[dict]) -> dict:
    """Per camera: subject means, then a subject-bootstrap CI over them."""
    out: dict = {}
    by_cam: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if "err" in r:
            by_cam[r["camera"]].append(r)
    for cam, rs in by_cam.items():
        e: dict = {"n_subjects": len({r["subject"] for r in rs}),
                   "n_clips": len(rs), "limbs": {}}
        for limb in LIMBS:
            per_subject: dict[str, list[float]] = defaultdict(list)
            for r in rs:
                per_subject[r["subject"]].append(r["err"][limb]["mean"])
            means = [float(np.nanmean(v)) for v in per_subject.values()]
            lo, hi = subject_bootstrap_ci(means)
            e["limbs"][limb] = {
                "mae_deg": float(np.nanmean(means)),
                "ci95": [lo, hi],
                "per_subject": {k: float(np.nanmean(v))
                                for k, v in per_subject.items()},
                "per_joint": {j: float(np.nanmean(
                    [r["err"][limb][j] for r in rs])) for j in JOINTS}}
        out[cam] = e
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="e.g. Subject_1; default every mocap file present")
    ap.add_argument("--cameras", nargs="*", default=list(CAMERAS))
    ap.add_argument("--max-frames", type=int, default=0,
                    help="smoke test: cap the jogging window")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    mats = sorted(MOCAP.glob("F_v3d_*.mat"))
    if args.subjects:
        want = set(args.subjects)
        mats = [m for m in mats if m.stem.split("_v3d_")[-1] in want]
    if not mats:
        print(f"no mocap under {MOCAP}")
        return 2

    lift_model = load_model(CKPT_FOR[DET], "base")
    if lift_model is None:
        raise SystemExit(f"checkpoint missing: {CKPT_FOR[DET]}")
    det_model = None

    print(f"=== phase 15: MoVi round F, {len(mats)} subjects, "
          f"cameras {args.cameras} ===")
    print(f"  frozen pipeline: {DET} detections, {PAD} padding, {WINDOWS} "
          f"windows, LOWPASS_HZ={LOWPASS_HZ}, MODEL_FPS={MODEL_FPS}")
    print(f"  AthleticsPose deployable baseline: {AP_NEAR:.4f} near / "
          f"{AP_FAR:.4f} far -- a different dataset, not the same clips\n")

    rows = []
    for mat in mats:
        subject = mat.stem.split("_v3d_")[-1]
        print(f"  {subject}:")
        for cam in args.cameras:
            video = MOVI / CAMERAS[cam]["pattern"].format(
                s=subject.split("_")[-1])
            if not video.exists():
                print(f"    {cam}: {video.name} not downloaded, skipping")
                continue
            # det_model stays None on purpose: `detect_coco17` builds the
            # detector with min_size set to the frame's own shortest side, so a
            # model reused across cameras of different resolutions would be
            # running at the wrong scale. PG is 800x600, CP is phone video.
            r = run_clip(subject, cam, mat, video, det_model, lift_model,
                         args.max_frames)
            if r:
                rows.append(r)

    if not rows:
        print("\nnothing scored")
        return 1
    agg = aggregate(rows)
    out = {"dataset": "BML MoVi round F", "doi": "doi:10.5683/SP2/JRHDRN",
           "role": "held-out test, opened 2026-11-02",
           "pre_registration": "results/phase15.md, frozen 64fb0bc",
           "pipeline": {"detector": DET, "padding": PAD, "windows": WINDOWS,
                        "lowpass_hz": LOWPASS_HZ, "model_fps": MODEL_FPS,
                        "checkpoint": CKPT_FOR[DET]},
           "movement": "jogging only; running_in_spot excluded",
           "alignment": ("exact: flags30/flags120 give the same boundaries in "
                         "both rates, so video frame k scores against mocap "
                         "frame f120[0] + 4k; the reference 3D is never "
                         "resampled"),
           "athleticspose_baseline": {"near": AP_NEAR, "far": AP_FAR,
                                      "note": "different dataset, camera, "
                                              "population and movement"},
           "by_camera": agg, "clips": rows}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== RESULT ===")
    print(f"  {'camera':<7}{'limb':>6}{'hip':>9}{'knee':>9}{'mean':>9}"
          f"{'95% CI':>22}{'n':>5}")
    for cam, e in agg.items():
        for limb in LIMBS:
            d = e["limbs"][limb]
            lo, hi = d["ci95"]
            print(f"  {cam:<7}{limb:>6}{d['per_joint']['hip']:>9.3f}"
                  f"{d['per_joint']['knee']:>9.3f}{d['mae_deg']:>9.3f}"
                  f"   [{lo:>6.3f}, {hi:>6.3f}]{e['n_subjects']:>5}")
    print(f"\n  AthleticsPose, for reference only: {AP_NEAR:.3f} near / "
          f"{AP_FAR:.3f} far")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
