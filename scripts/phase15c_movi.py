"""Phase 15C PART 3 -- apply the chosen thorax to MoVi, once.

The definition was chosen on AthleticsPose by `scripts/phase15c_thorax.py`,
under the rule fixed in `results/phase15c.md` (commit 49428c0). **No MoVi
subject took part in choosing it.** This script reads the winner out of
`results/phase15c.json` and applies it -- it does not decide anything.

Reported BESIDE the original numbers as "thorax-matched (15C)". The phase 15
headline does not change. The sealed test set is not opened. The angle code and
the lifter are untouched.

Run: .venv/Scripts/python.exe scripts/phase15c_movi.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data" / "athleticspose" / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from phase15_checks import (angle_mask, camera_pose, load_v3d,  # noqa: E402
                             to_camera, to_h36m)
from phase15_movi import (CAMERAS, MOCAP, MOCAP_PER_VIDEO, MOVI,  # noqa: E402
                           decode_range, jogging_window)
from phase15b_conventions import lift_2d, project, score  # noqa: E402
from phase3_angles import H36M  # noqa: E402
from phase3_infer import load_model  # noqa: E402
from phase7_inference_fix import CKPT_FOR  # noqa: E402
from video_kinematics import (coco17_to_h36m17, detect_coco17,  # noqa: E402
                               scores_to_train_scale)

AP_JSON = REPO / "results" / "phase15c.json"
OUT = REPO / "results" / "phase15c_movi.json"
CAM = "PG1"
RULE_SUBJECTS = ("Subject_1", "Subject_10", "Subject_12", "Subject_13")
DIAGNOSTIC_ONLY = ("Subject_11", "Subject_15")

# The candidates' MoVi equivalents, fixed in the pre-registration before any
# scoring. Only the winner is used; the rest are here so the mapping that was
# committed is visible in the code that applies it.
CANDIDATE_IN_MOVI = {
    "CLAV": ("CLAV",),
    "C7": ("C7",),
    "STRN": ("STRN",),                      # AP calls this marker Xiphoid
    "mid_CLAV_C7": ("CLAV", "C7"),
    "mean_Xiphoid_T10": ("STRN", "T10"),
}


def thorax_matched(v: dict, chosen: str) -> np.ndarray:
    """The original MoVi reference with ONLY the thorax joint replaced.

    Everything else -- pelvis, hips, knees, ankles -- is exactly
    `phase15_checks.to_h36m`, so the one changed quantity is the one the phase
    decided. Gap handling is unchanged: `markerGaps == 1` means missing, the
    marker becomes NaN, and `np.mean` propagates.
    """
    ref = to_h36m(v).copy()
    names = CANDIDATE_IN_MOVI[chosen]
    idx = {n: i for i, n in enumerate(v["marker_names"])}
    missing = [n for n in names if n not in idx]
    if missing:
        raise KeyError(f"{chosen}: markers absent from this export: {missing}")
    loc = v["marker_location"].copy()
    loc[np.asarray(v["marker_gaps"]) == 1] = np.nan
    ref[:, H36M["thorax"]] = np.mean([loc[:, idx[n]] for n in names], axis=0)
    return ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not AP_JSON.exists():
        raise SystemExit(f"{AP_JSON} missing -- run phase15c_thorax.py first")
    apd = json.loads(AP_JSON.read_text(encoding="utf-8"))
    chosen = apd["step3_decision"]["chosen"]
    print(f"=== phase 15C part 3: thorax := {chosen} ===")
    print(f"  chosen on AthleticsPose: {apd['step3_decision']['reason']}")
    print(f"  MoVi markers used: {CANDIDATE_IN_MOVI[chosen]}")
    print(f"  no MoVi subject took part in that choice\n")

    model = load_model(CKPT_FOR["coco"], "base")
    if model is None:
        raise SystemExit("checkpoint missing")

    mats = {p.stem.split("_v3d_")[-1]: p
            for p in sorted(MOCAP.glob("F_v3d_*.mat"))}
    rows: dict = {}
    for subject in RULE_SUBJECTS + DIAGNOSTIC_ONLY:
        if subject not in mats:
            continue
        video = MOVI / CAMERAS[CAM]["pattern"].format(s=subject.split("_")[-1])
        if not video.exists():
            continue
        print(f"--- {subject} ---")
        v = load_v3d(mats[subject])
        _ji, f30, f120 = jogging_window(v)
        orig_all = to_h36m(v)
        match_all = thorax_matched(v, chosen)

        frames, fps, vmeta = decode_range(video, int(f30[0]), int(f30[1]))
        mk = int(f120[0]) + MOCAP_PER_VIDEO * np.arange(len(frames))
        mk = mk[mk < orig_all.shape[0]]
        frames = frames[:len(mk)]
        ref = {"original": orig_all[mk], "thorax_matched": match_all[mk]}

        thorax_shift = float(np.nanmean(np.linalg.norm(
            ref["original"][:, H36M["thorax"]]
            - ref["thorax_matched"][:, H36M["thorax"]], axis=1)))

        R_cv, t_cv, _c = camera_pose(CAM)
        gc = to_camera(ref["original"], R_cv, t_cv)
        near = ("l" if np.nanmean(gc[:, H36M["l_hip"], 2])
                < np.nanmean(gc[:, H36M["r_hip"], 2]) else "r")

        kp = coco17_to_h36m17(detect_coco17(frames, crop=True, model=None)[0])
        det_ok = np.abs(kp[:, :, :2]).sum(axis=(1, 2)) > 0
        kp = scores_to_train_scale(kp)
        est_det = lift_2d(model, np.nan_to_num(kp), fps)

        p2d = project(ref["original"], CAM)
        keep = np.isfinite(p2d).all(axis=(1, 2))
        est_perf = lift_2d(model, p2d[keep], fps)

        grid: dict = {}
        for arm, (est, sel) in (("detected", (est_det, det_ok)),
                                ("perfect", (est_perf, keep))):
            grid[arm] = {}
            for rname, rr in ref.items():
                grid[arm][rname] = score(est[sel] if arm == "detected" else est,
                                         rr[sel], near)
                s = grid[arm][rname]
                print(f"  {arm:<9}{rname:<16} near hip "
                      f"MAE {s['near']['hip']['mae']:6.3f} "
                      f"bias {s['near']['hip']['bias']:+7.3f} "
                      f"sd {s['near']['hip']['sd']:5.2f} "
                      f"r {s['near']['hip']['r']:+.3f} | knee "
                      f"MAE {s['near']['knee']['mae']:6.3f} "
                      f"r {s['near']['knee']['r']:+.3f}")
        rows[subject] = {"near_limb": near, "n_frames": len(mk),
                         "in_rule_set": subject in RULE_SUBJECTS,
                         "thorax_shift_mm": thorax_shift, "grid": grid}

    out = {"what": f"Phase 15C part 3: MoVi re-scored with thorax := {chosen}, "
                   f"chosen on AthleticsPose only. Reported BESIDE the "
                   f"originals. The phase 15 headline is unchanged.",
           "chosen_thorax": chosen,
           "chosen_on": "AthleticsPose S11/S13/S16, running and sprint",
           "movi_markers": list(CANDIDATE_IN_MOVI[chosen]),
           "camera": CAM, "subjects": rows}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
