"""Does direction of travel across the image predetermine which side faces the camera?

The demo cannot INFER the near limb from the lifted 3D -- 9 of 11 against
calibrated truth, with no signal separating the failures
(`results/phase16a_nearlimb.json`). So the near side is declared by the user via
`--near-side`, and `overstride.nearside.suggest_from_travel` offers a geometric
suggestion that can only warn.

This measures that suggestion's hit rate against the **calibrated** near limb on
all 11 spent MoVi PG1 subjects. The rule, derived in `overstride/nearside.py`:

    rightward travel across the image  ->  RIGHT side faces the camera
    leftward  travel across the image  ->  LEFT  side faces the camera

from `left = superior x anterior` in the OpenCV camera frame. Travel direction
comes from the **detected 2D only** -- no depth, no calibration -- exactly as
the demo will compute it.

All 11 subjects are SPENT; this establishes no held-out claim. Output is
AGGREGATE ONLY: no per-subject MoVi value is written to the repository.

Run: .venv/Scripts/python.exe scripts/phase16a_travelrule.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from overstride.nearside import suggest_from_travel  # noqa: E402
from phase15_checks import (camera_pose, load_v3d, to_camera,  # noqa: E402
                            to_h36m)
from phase15_movi import (CAMERAS, MOCAP, MOCAP_PER_VIDEO, MOVI,  # noqa: E402
                          decode_range, jogging_window)
from phase3_angles import H36M  # noqa: E402
from video_kinematics import coco17_to_h36m17, detect_coco17  # noqa: E402

OUT = REPO / "results" / "phase16a_travelrule.json"
CAM = "PG1"
SUBJECTS = ("Subject_1", "Subject_10", "Subject_11", "Subject_12",
            "Subject_13", "Subject_15", "Subject_16", "Subject_17",
            "Subject_18", "Subject_19", "Subject_2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject-out", type=Path, default=None,
                    help="optional path OUTSIDE the repo for per-subject rows")
    args = ap.parse_args()

    R_cv, t_cv, _s = camera_pose(CAM)
    print("=== direction-of-travel rule vs the CALIBRATED near limb ===")
    print("    rightward travel -> RIGHT near;  leftward -> LEFT near\n")
    print(f"  {'subject':<12}{'calibrated':>11}{'rule':>6}{'hit':>5}"
          f"{'dx (hipw/frame)':>17}{'confident':>11}")

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
        calibrated = ("l" if np.nanmean(gc[:, H36M["l_hip"], 2])
                      < np.nanmean(gc[:, H36M["r_hip"], 2]) else "r")

        kp = coco17_to_h36m17(detect_coco17(frames, crop=True, model=None)[0])
        sug = suggest_from_travel(kp)
        hit = sug.side == calibrated
        rows.append({"subject": subject, "calibrated": calibrated,
                     "rule": sug.side, "hit": bool(hit),
                     "dx_per_frame": (None
                                      if not np.isfinite(sug.dx_per_frame)
                                      else round(sug.dx_per_frame, 5)),
                     "confident": bool(sug.confident)})
        print(f"  {subject:<12}{calibrated:>11}{str(sug.side):>6}"
              f"{('yes' if hit else 'NO'):>5}"
              f"{sug.dx_per_frame:>17.5f}{str(sug.confident):>11}")

    n = len(rows)
    hits = sum(1 for r in rows if r["hit"])
    conf = [r for r in rows if r["confident"]]
    dx_abs = [abs(r["dx_per_frame"]) for r in rows
              if r["dx_per_frame"] is not None]
    print(f"\n  HIT RATE: {hits}/{n}")
    print(f"  confident suggestions: {len(conf)}/{n}")
    if dx_abs:
        print(f"  |dx| per frame: min {min(dx_abs):.4f}  "
              f"median {float(np.median(dx_abs)):.4f}  max {max(dx_abs):.4f}")

    out = {
        "what": ("Hit rate of the direction-of-travel near-side rule against "
                 "the calibrated near limb, 11 spent MoVi PG1 subjects."),
        "rule": ("rightward travel across the image -> RIGHT side faces the "
                 "camera; leftward -> LEFT. From left = superior x anterior in "
                 "the OpenCV camera frame (x right, y down, z into scene)."),
        "travel_measured_from": "the detected 2D only -- no depth, no calibration",
        "diagnostic_only": ("all 11 subjects are spent; this establishes no "
                            "held-out claim"),
        "aggregate_only": ("no per-subject MoVi value is written here, same "
                           "standard as the scoping files"),
        "mirroring": ("the rule inverts under mirroring, which cannot be "
                      "detected from the image; the demo warns and the filming "
                      "instructions say to use the rear camera"),
        "n_subjects": n, "n_hits": hits,
        "hit_rate": round(hits / max(n, 1), 4),
        "n_confident": len(conf),
        "abs_dx_per_frame": ({"min": round(min(dx_abs), 5),
                              "median": round(float(np.median(dx_abs)), 5),
                              "max": round(max(dx_abs), 5)}
                             if dx_abs else None),
        "n_misses": n - hits,
    }
    if hits < n:
        # describe misses WITHOUT naming subjects
        miss_dx = [abs(r["dx_per_frame"]) for r in rows
                   if not r["hit"] and r["dx_per_frame"] is not None]
        out["miss_abs_dx_range"] = ([round(min(miss_dx), 5),
                                     round(max(miss_dx), 5)]
                                    if miss_dx else None)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
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
