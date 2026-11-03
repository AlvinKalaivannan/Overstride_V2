"""Is it a temporal offset? Lag scan, plus the tool's own plausibility bands.

A large MAE with a near-zero correlation has two common causes: the two signals
are misaligned in time, or they are unrelated. A lag scan separates them. If
the clip is shifted, correlation peaks sharply at some non-zero lag; if the
signals are unrelated, there is no peak anywhere.

This is what ruled misalignment OUT for the confidence-scale bug: lag 0 gave
r = +0.022 hip and +0.167 knee, and no lag from -12 to +12 video frames did
better. The second half of the script is why: the estimate's own knee flexion
reached +66 deg, outside `video_kinematics.PLAUSIBLE["knee"] = (-170, 20)`,
which says the estimate was broken rather than displaced.

DIAGNOSTIC ONLY. No calibration value and no participant attribute is printed
or saved -- what it prints is angular range and correlation, which is
measurement of the method.

Run: .venv/Scripts/python.exe scripts/diagnostics/diag_align.py --frames 150
"""

from __future__ import annotations

import numpy as np

from _common import lift_2d, lifter, load, parser  # noqa: E402

LAGS = range(-12, 13)          # video frames; 1 frame = 4 mocap frames = 33 ms


def main() -> int:
    ap = parser(__doc__.splitlines()[0])
    ap.add_argument("--raw-confidence", action="store_true",
                    help="reproduce the bug: pass torchvision's raw keypoint "
                         "scores through instead of rescaling them")
    args = ap.parse_args()

    from phase15_checks import angle_mask, camera_pose, to_camera, to_h36m
    from phase15_movi import MOCAP_PER_VIDEO, decode_range
    from phase3_angles import H36M, sagittal_angles
    from video_kinematics import (PLAUSIBLE, coco17_to_h36m17, detect_coco17,
                                  scores_to_train_scale)

    d = load(args.subject, args.camera, args.frames)
    print(f"=== alignment: {d['subject']} / {d['camera']}, "
          f"{d['n_v']} frames ===")

    frames, fps, vmeta = decode_range(d["video"], d["a30"],
                                      d["a30"] + d["n_v"] - 1)
    kp = coco17_to_h36m17(detect_coco17(frames, crop=True, model=None)[0])
    xy = kp[:, :, :2]
    print(f"  2D skeleton: {np.median(xy[:, :, 1].max(1) - xy[:, :, 1].min(1)):.0f} px tall "
          f"in a {vmeta['height']} px frame")
    raw = kp[:, :, 2].copy()
    print(f"  raw detector score channel: [{raw[raw > 0].min():.2f}, "
          f"{raw.max():.2f}] -- the lifter was trained on ~0.87")
    if not args.raw_confidence:
        kp = scores_to_train_scale(kp)
    else:
        print("  !! passing RAW scores through, reproducing the bug")

    est = lift_2d(lifter(), np.nan_to_num(kp), fps)

    gt = d["gt"][:len(est)]
    R_cv, t_cv, _c = camera_pose(args.camera)
    gc = to_camera(gt, R_cv, t_cv)
    near = ("l" if np.nanmean(gc[:, H36M["l_hip"], 2])
            < np.nanmean(gc[:, H36M["r_hip"], 2]) else "r")

    print(f"\n  ranges, near limb {near} -- does the ESTIMATE even stay "
          f"anatomical?")
    for j in ("hip", "knee"):
        m = angle_mask(gt, near)
        g = sagittal_angles(gt, near)[j][m]
        e = sagittal_angles(est, near)[j][m]
        lo, hi = PLAUSIBLE[j]
        flag = "" if (e.min() >= lo and e.max() <= hi) else "   <-- OUTSIDE"
        print(f"    {j:<5} GT  [{g.min():7.1f}, {g.max():7.1f}]")
        print(f"    {j:<5} EST [{e.min():7.1f}, {e.max():7.1f}]  "
              f"plausible band ({lo:.0f}, {hi:.0f}){flag}")

    print(f"\n  lag scan (video frames; 1 frame = {MOCAP_PER_VIDEO} mocap "
          f"frames = 33 ms)")
    gt_world = d["gt_world"]
    for j in ("hip", "knee"):
        scan = []
        for lag in LAGS:
            mk = d["f120_start"] + MOCAP_PER_VIDEO * (np.arange(len(est)) + lag)
            keep = (mk >= 0) & (mk < gt_world.shape[0])
            ga = gt_world[mk[keep]]
            m = angle_mask(ga, near)
            g = sagittal_angles(ga, near)[j][m]
            e = sagittal_angles(est[keep], near)[j][m]
            f = np.isfinite(g) & np.isfinite(e)
            if f.sum() > 20 and g[f].std() > 1e-6 and e[f].std() > 1e-6:
                scan.append((lag, float(np.corrcoef(g[f], e[f])[0, 1])))
        if not scan:
            print(f"    {j:<5} too few frames to scan")
            continue
        best = max(scan, key=lambda t: abs(t[1]))
        at0 = dict(scan).get(0, float("nan"))
        print(f"    {j:<5} lag 0: r = {at0:+.3f}    best: lag {best[0]:+d}, "
              f"r = {best[1]:+.3f}")
        verdict = ("a real shift" if best[0] != 0 and abs(best[1]) > abs(at0) + 0.15
                   else "no temporal shift")
        print(f"          -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
