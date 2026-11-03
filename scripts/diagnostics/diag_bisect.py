"""Which half of the pipeline is broken? Feed the lifter a perfect 2D.

An end-to-end error says the pipeline is wrong but not WHERE. To localise it,
hand the back half a perfect version of its own input: project the reference 3D
through the camera, which gives H36M-ordered pixels by construction -- no
detector, no COCO conversion, no joint-order question.

This is what settled the confidence-scale bug. The detected path was giving
30 deg with r = +0.02; this gave 4.49 deg near and 5.47 deg far on the same
clip, matching the 5.16 deg `results/phase07.md` section 5 had independently
measured for perfect 2D into this same checkpoint on AthleticsPose. One run
cleared the lifter, the ground-truth mapping, the projection, the time
alignment and the angle code, which left the detector's input as the only
candidate.

DIAGNOSTIC ONLY. Nothing is selected on this output -- selecting anything on a
held-out set would invalidate phase 15. No calibration value is printed or
saved; projection goes through `project_reference`, which takes a camera name.

Run: .venv/Scripts/python.exe scripts/diagnostics/diag_bisect.py
"""

from __future__ import annotations

import numpy as np

from _common import lift_2d, lifter, load, parser  # noqa: E402


def main() -> int:
    args = parser(__doc__.splitlines()[0]).parse_args()
    from phase15_checks import (angle_mask, camera_pose, project_reference,
                                to_camera)
    from phase3_angles import H36M, sagittal_angles

    d = load(args.subject, args.camera, args.frames)
    print(f"=== bisection: {d['subject']} / {d['camera']}, "
          f"jogging = motions_list[{d['jogging_index']}], "
          f"{d['n_v']} video frames ===")

    gt = d["gt"]
    perfect = project_reference(gt, args.camera)
    ok = np.isfinite(perfect).all(axis=(1, 2))
    perfect, gt = perfect[ok], gt[ok]
    print(f"  {ok.sum()} of {len(ok)} frames have a complete projected skeleton")
    hgt = np.nanmax(perfect[:, :, 1], axis=1) - np.nanmin(perfect[:, :, 1], axis=1)
    print(f"  projected skeleton height: {np.nanmedian(hgt):.0f} px")

    est = lift_2d(lifter(), perfect)

    R_cv, t_cv, _c = camera_pose(args.camera)
    gc = to_camera(gt, R_cv, t_cv)
    near = ("l" if np.nanmean(gc[:, H36M["l_hip"], 2])
            < np.nanmean(gc[:, H36M["r_hip"], 2]) else "r")

    print(f"\n  PERFECT 2D, near limb {near}")
    print(f"  {'limb':<6}{'joint':<6}{'MAE':>9}{'r':>8}   signed bias")
    out = {}
    for side in ("l", "r"):
        tag = "near" if side == near else "far"
        m = angle_mask(gt, side)
        for j in ("hip", "knee"):
            g = sagittal_angles(gt, side)[j][m]
            e = sagittal_angles(est, side)[j][m]
            f = np.isfinite(g) & np.isfinite(e)
            mae = float(np.abs(g - e)[f].mean())
            r = float(np.corrcoef(g[f], e[f])[0, 1])
            bias = float((e - g)[f].mean())
            out[(tag, j)] = mae
            print(f"  {tag:<6}{j:<6}{mae:>9.3f}{r:>+8.3f}   {bias:>+8.3f} deg")
    for tag in ("near", "far"):
        print(f"  {tag} mean: {0.5 * (out[(tag, 'hip')] + out[(tag, 'knee')]):.3f} deg")

    print("\n  reference points, both measured elsewhere, not fitted here:")
    print("    perfect 2D into this checkpoint on AthleticsPose: 5.16 deg "
          "(results/phase07.md section 5)")
    print("    the deployable detected path on AthleticsPose:    3.509 near / "
          "3.618 far")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
