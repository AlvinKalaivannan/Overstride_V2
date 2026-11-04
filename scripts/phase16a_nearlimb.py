"""Can the demo tell which limb is nearer the camera, from video alone?

The error budget reports near and far limb separately, and they differ by a
measured **+1.971 deg** on the thigh (sealed test, CI excluding zero, 5 of 5
subjects). So a demo that mislabels near and far shows the user the wrong row.

Phase 14 reported the near limb "identifiable from the estimate alone, 592 of
592 clips" -- on **AthleticsPose**. The M1 regression test found it **fails on
MoVi Subject_2**, which is the least side-on sealed subject (`view_ratio`
0.537, the lowest of the five). This measures how often it fails across all 11
spent MoVi PG1 subjects, against the camera-derived truth.

Note the azimuth proxy cannot help: it recovers |cos(theta)|, a **magnitude**,
so it says how side-on the view is and not which side faces the camera.

Output: results/phase16a_nearlimb.json. No MoVi pixel data is written.

Run: .venv/Scripts/python.exe scripts/phase16a_nearlimb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from overstride.analyze import analyze_frames  # noqa: E402
from phase11_viewpoint import view_ratio  # noqa: E402
from phase15_checks import (camera_pose, load_v3d, to_camera,  # noqa: E402
                            to_h36m)
from phase15_movi import (CAMERAS, MOCAP, MOCAP_PER_VIDEO, MOVI,  # noqa: E402
                          decode_range, jogging_window)
from phase3_angles import H36M  # noqa: E402
from phase3_infer import load_model  # noqa: E402
from phase7_inference_fix import CKPT_FOR  # noqa: E402

OUT = REPO / "results" / "phase16a_nearlimb.json"
CAM = "PG1"
SUBJECTS = ("Subject_1", "Subject_10", "Subject_11", "Subject_12",
            "Subject_13", "Subject_15", "Subject_16", "Subject_17",
            "Subject_18", "Subject_19", "Subject_2")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject-out", type=Path, default=None,
                    help="optional path OUTSIDE the repo for per-subject rows")
    args = ap.parse_args()
    R_cv, t_cv, _s = camera_pose(CAM)
    lifter = load_model(CKPT_FOR["coco"], "base")
    if lifter is None:
        raise SystemExit("checkpoint missing")

    print("=== near limb from video alone, against the camera truth ===")
    print(f"  {'subject':<12}{'GT near':>8}{'est near':>9}{'agree':>7}"
          f"{'GT view_ratio':>15}{'est |dz| / width':>18}")
    rows = []
    for subject in SUBJECTS:
        mat = MOCAP / f"F_v3d_{subject}.mat"
        video = MOVI / CAMERAS[CAM]["pattern"].format(s=subject.split("_")[-1])
        if not (mat.exists() and video.exists()):
            print(f"  {subject}: missing files, skipped")
            continue
        v = load_v3d(mat)
        _ji, f30, f120 = jogging_window(v)
        ref_all = to_h36m(v)
        frames, fps, _m = decode_range(video, int(f30[0]), int(f30[1]))
        mk = int(f120[0]) + MOCAP_PER_VIDEO * np.arange(len(frames))
        mk = mk[mk < ref_all.shape[0]]
        frames = list(frames)[:len(mk)]
        ref = ref_all[mk]

        gc = to_camera(ref, R_cv, t_cv)
        gt_near = ("l" if np.nanmean(gc[:, H36M["l_hip"], 2])
                   < np.nanmean(gc[:, H36M["r_hip"], 2]) else "r")
        ok = np.isfinite(gc[:, [H36M["l_hip"], H36M["r_hip"]]]).all(axis=(1, 2))
        vr = float(view_ratio(gc[ok])) if ok.sum() > 5 else float("nan")

        res = analyze_frames(frames, float(fps), lifter)
        est = res.est3d
        lz = float(np.nanmean(est[:, H36M["l_hip"], 2]))
        rz = float(np.nanmean(est[:, H36M["r_hip"], 2]))
        width = float(np.nanmean(np.linalg.norm(
            est[:, H36M["l_hip"]] - est[:, H36M["r_hip"]], axis=-1)))
        sep = abs(lz - rz) / max(width, 1e-9)
        # how consistently the per-frame sign agrees with the clip-level call
        per_frame = np.where(est[:, H36M["l_hip"], 2]
                             < est[:, H36M["r_hip"], 2], "l", "r")
        consistency = float((per_frame == res.near_limb).mean())

        agree = res.near_limb == gt_near
        rows.append({"subject": subject, "gt_near": gt_near,
                     "est_near": res.near_limb, "agree": bool(agree),
                     "gt_view_ratio": round(vr, 4),
                     "est_depth_sep_over_width": round(sep, 4),
                     "per_frame_sign_consistency": round(consistency, 3),
                     "n_frames": int(len(frames))})
        print(f"  {subject:<12}{gt_near:>8}{res.near_limb:>9}"
              f"{('yes' if agree else 'NO'):>7}{vr:>15.4f}{sep:>18.4f}")

    n = len(rows)
    n_ok = sum(1 for r in rows if r["agree"])
    bad = [r for r in rows if not r["agree"]]
    print(f"\n  agreement: {n_ok}/{n} subjects")
    if bad:
        print("  disagreeing subjects, with their viewpoint:")
        for r in bad:
            print(f"    {r['subject']:<12} GT view_ratio {r['gt_view_ratio']:.3f}"
                  f"  est sep/width {r['est_depth_sep_over_width']:.3f}"
                  f"  sign consistency {r['per_frame_sign_consistency']:.2f}")

    sep_ok = [r["est_depth_sep_over_width"] for r in rows if r["agree"]]
    sep_bad = [r["est_depth_sep_over_width"] for r in rows if not r["agree"]]
    cons_ok = [r["per_frame_sign_consistency"] for r in rows if r["agree"]]
    cons_bad = [r["per_frame_sign_consistency"] for r in rows if not r["agree"]]
    summary = {
        "n_subjects": n, "n_agree": n_ok,
        "agreement_fraction": round(n_ok / max(n, 1), 3),
        "est_sep_over_width": {
            "agree_min": round(min(sep_ok), 4) if sep_ok else None,
            "disagree_max": round(max(sep_bad), 4) if sep_bad else None},
        "per_frame_sign_consistency": {
            "agree_min": round(min(cons_ok), 3) if cons_ok else None,
            "disagree_max": round(max(cons_bad), 3) if cons_bad else None},
    }
    print(f"\n  estimated depth separation / width:")
    print(f"    agreeing subjects    min {summary['est_sep_over_width']['agree_min']}")
    print(f"    disagreeing subjects max {summary['est_sep_over_width']['disagree_max']}")
    print(f"  per-frame sign consistency:")
    print(f"    agreeing subjects    min {summary['per_frame_sign_consistency']['agree_min']}")
    print(f"    disagreeing subjects max {summary['per_frame_sign_consistency']['disagree_max']}")

    OUT.write_text(json.dumps({
        "what": ("Can the demo identify the near limb from video alone? "
                 "Estimated-3D hip depth against the camera-derived truth, "
                 "11 spent MoVi PG1 subjects."),
        "why": ("the error budget reports near and far separately and they "
                "differ by a measured +1.971 deg on the thigh; mislabelling "
                "shows the user the wrong row"),
        "azimuth_proxy_cannot_help": ("it recovers |cos(theta)|, a magnitude, "
                                      "so it gives how side-on the view is and "
                                      "not which side faces the camera"),
        "camera": CAM, "summary": summary,
        "aggregate_only": ("no per-subject MoVi value is written here -- same "
                           "standard as the scoping files"),
        "per_subject": (f"{len(rows)} rows, not committed; use "
                        f"--per-subject-out PATH outside the repo")},
        indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}  (aggregate only)")
    if args.per_subject_out:
        dest = Path(args.per_subject_out).resolve()
        if REPO in dest.parents or dest.parent == REPO:
            print("  REFUSED: --per-subject-out is inside the repository")
            return 3
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"  per-subject rows -> {dest}  (outside the repo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
