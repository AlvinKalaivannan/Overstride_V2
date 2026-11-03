"""Is the detector or the joint-order conversion at fault? Per-joint pixels.

Once `diag_bisect.py` has shown the back half is fine, the 2D is the suspect.
Both the converted detections and the projected reference are H36M-ordered
(n, 17, 3) arrays, so they can be compared joint by joint in pixels. A
permutation shows up immediately as one or two joints being enormous while the
rest are small; a bad detector shows up as everything being large.

This is what cleared `coco17_to_h36m17`. Every H36M joint matched the COCO slot
it should, and the converted 2D measured 11.8 px mean error -- BETTER than
AthleticsPose's own 12.18-12.28 px -- with hips at 21-22 px (the definitional
mean(ASI,PSI) offset no image detector can mark), knees 5.9/6.3 px and ankles
4.9/5.2 px. The 2D was never the problem, which is what pointed at the
confidence channel.

DIAGNOSTIC ONLY. No calibration value is printed or saved.

Run: .venv/Scripts/python.exe scripts/diagnostics/diag_2d.py --frames 150
"""

from __future__ import annotations

import numpy as np

from _common import load, parser  # noqa: E402


def main() -> int:
    args = parser(__doc__.splitlines()[0]).parse_args()
    from phase15_checks import project_reference
    from phase15_movi import decode_range
    from phase3_angles import H36M
    from video_kinematics import COCO, coco17_to_h36m17, detect_coco17

    d = load(args.subject, args.camera, args.frames)
    print(f"=== 2D error: {d['subject']} / {d['camera']}, "
          f"{d['n_v']} frames ===")

    frames, fps, vmeta = decode_range(d["video"], d["a30"],
                                      d["a30"] + d["n_v"] - 1)
    print(f"  video: {vmeta['width']}x{vmeta['height']} @ {fps:.3f} fps, "
          f"{len(frames)} decoded")
    kp_coco, dmeta = detect_coco17(frames, crop=True, model=None)
    kp = coco17_to_h36m17(kp_coco)
    print(f"  detector: {dmeta['frames_with_person']}/{len(frames)} found, "
          f"{dmeta['frames_on_crop']} on a crop, "
          f"{dmeta['full_frame_redetections']} re-detections")

    ref = project_reference(d["gt"][:len(frames)], args.camera)[:, :, :2]
    inv = {i: n for n, i in H36M.items()}

    print(f"\n  {'H36M joint':<13}{'converted':>11}{'raw COCO slot':>15}")
    print("  " + "-" * 38)
    conv, raw = [], []
    for j in range(17):
        dc = np.nanmedian(np.linalg.norm(kp[:, j, :2] - ref[:, j], axis=-1))
        dr = np.nanmedian(np.linalg.norm(kp_coco[:, j, :2] - ref[:, j], axis=-1))
        conv.append(dc)
        raw.append(dr)
        print(f"  {inv[j]:<13}{dc:>11.1f}{dr:>15.1f}")
    print("  " + "-" * 38)
    print(f"  {'MEAN':<13}{np.mean(conv):>11.1f}{np.mean(raw):>15.1f}")
    print(f"\n  AthleticsPose's own detectors, for reference: 12.18-12.28 px "
          f"(results/phase07.md section 5)")

    # The permutation test: which COCO slot is each H36M joint nearest?
    print("\n  nearest COCO slot for each H36M joint of the reference")
    print("  (if the conversion is right, the lower-limb joints map to "
          "themselves)")
    bad = 0
    for name in ("r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
                 "l_shoulder", "r_shoulder"):
        j = H36M[name]
        dist = [float(np.nanmedian(np.linalg.norm(kp_coco[:, c, :2] - ref[:, j],
                                                  axis=-1)))
                for c in range(17)]
        b = int(np.argmin(dist))
        cname = [k for k, v in COCO.items() if v == b][0]
        mark = "" if cname == name else "   <-- MISMATCH"
        bad += cname != name
        print(f"    {name:<12} -> COCO[{b:>2}] {cname:<11}{dist[b]:>7.1f} px"
              f"{mark}")
    print(f"\n  {'conversion looks correct' if not bad else f'{bad} MISMATCHES'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
