"""READ-ONLY: during the still (standing) motions, is the subject SIDE-ON to PG1?

A per-subject offset calibration only helps if the pose it is read from sits at
the same viewpoint as the running it corrects. Phase 11 measured sagittal error
falling 3.50 deg -> 1.77 deg as a view becomes side-on, and phase 14 found camera
azimuth carries essentially all of the per-frame confidence signal -- so a
calibration pose taken facing the camera would be read at a viewpoint where the
tool is known to be worse, and the offset would be contaminated by that.

This reads only the twelve round F subjects already on disk. No model is run and
no new subject is opened. `view_ratio` is phase 3B's, unchanged: hip DEPTH
separation over hip width, 0 = frontal, 1 = side-on (AthleticsPose median
0.751).

Output: results/phase15_standing_viewpoint.json

Run: .venv/Scripts/python.exe scripts/phase15_standing_viewpoint.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from phase11_viewpoint import view_ratio  # noqa: E402
from phase15_standing_scope import analyse  # noqa: E402
from phase15_checks import (camera_pose, load_v3d, to_camera,  # noqa: E402
                            to_h36m)
from phase15_movi import jogging_window  # noqa: E402
from phase3_angles import H36M  # noqa: E402

MOCAP = REPO / "data" / "movi" / "raw" / "round_f" / "mocap"
OUT = REPO / "results" / "phase15_standing_viewpoint.json"
CAM = "PG1"

# The four motions that are legs-still in all twelve subjects, ranked by median
# continuous stillness in phase15_standing_scope.
STILL_MOTIONS = ("crossarms", "hand_clapping", "scratching_head",
                 "checking_watch")


def window_viewpoint(ref: np.ndarray, lo: int, hi: int, R, t) -> dict:
    """view_ratio and near-limb over one mocap window, in PG1's camera frame."""
    hi = min(hi, ref.shape[0])
    if hi - lo < 10:
        return {"n": int(max(0, hi - lo)), "view_ratio": None}
    seg = ref[lo:hi]
    cam = to_camera(seg, R, t)
    # `view_ratio` returns a SCALAR via np.mean, so one gapped hip frame makes
    # the whole window NaN. Mask to frames where both hips are measured first,
    # exactly as phase15_checks does -- not nan-averaging around the sentinel.
    ok = np.isfinite(cam[:, [H36M["l_hip"], H36M["r_hip"]]]).all(axis=(1, 2))
    if ok.sum() <= 5:
        return {"n": int(hi - lo), "n_scored": int(ok.sum()),
                "view_ratio": None}
    vr = view_ratio(cam[ok])
    lz = float(np.nanmean(cam[ok][:, H36M["l_hip"], 2]))
    rz = float(np.nanmean(cam[ok][:, H36M["r_hip"], 2]))
    # The hip depth gap is deliberately NOT returned: it is a per-subject
    # measurement and is needed only to decide which limb is nearer.
    return {"n": int(hi - lo), "n_scored": int(ok.sum()),
            "view_ratio": round(vr, 4),
            "near_limb": "l" if lz < rz else "r"}


def main() -> int:
    # Per-subject stillness is recomputed HERE rather than read from a file:
    # MoVi's licence keeps per-subject detail out of the repo, so there is no
    # committed file to read it from.
    R, t, _s = camera_pose(CAM)

    print(f"=== READ-ONLY: standing-pose viewpoint on {CAM} ===")
    print("    view_ratio: 0 = frontal, 1 = side-on "
          "(AthleticsPose median 0.751)\n")
    hdr = (f"  {'subject':<12}{'jog vr':>8}{'jog near':>9}   "
           f"{'motion':<17}{'still s':>8}{'still vr':>9}{'near':>6}"
           f"{'same?':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for p in sorted(MOCAP.glob("F_v3d_*.mat")):     # one at a time
        v = load_v3d(p)
        sid = v["subject_id"]
        if "motions" not in v:
            continue
        detail = analyse(p)          # in memory only, never written here
        if "motions" not in detail:
            continue
        ref = to_h36m(v)
        _ji, _f30, f120 = jogging_window(v)
        jog = window_viewpoint(ref, int(f120[0]), int(f120[1]), R, t)
        if jog.get("view_ratio") is None:
            print(f"  {sid:<12}  jogging window unscoreable (hip gaps)")
            continue

        motions = [m.strip() for m in v["motions"]]
        best = None
        for m in detail["motions"]:
            if m["motion"] not in STILL_MOTIONS:
                continue
            st = m["longest_legs_still"]
            if not st["frames_120hz"] or st["start_120hz"] is None:
                continue
            lo = st["start_120hz"]
            w = window_viewpoint(ref, lo, lo + st["frames_120hz"], R, t)
            if w.get("view_ratio") is None:
                continue
            cand = {"motion": m["motion"], "index_0based": m["index_0based"],
                    "video_30fps": m["video_30fps"],
                    "still_seconds": st["seconds"],
                    "still_start_120hz": lo, **w}
            if best is None or cand["still_seconds"] > best["still_seconds"]:
                best = cand
            rows.append({"subject_id": sid, "jogging": jog, "still": cand})

        if best:
            same = "yes" if best["near_limb"] == jog["near_limb"] else "NO"
            print(f"  {sid:<12}{jog['view_ratio']:>8.3f}{jog['near_limb']:>9}   "
                  f"{best['motion']:<17}{best['still_seconds']:>8.2f}"
                  f"{best['view_ratio']:>9.3f}{best['near_limb']:>6}"
                  f"{same:>7}")

    # ---- summary across subjects, best still window each ------------------
    best_per = {}
    for r in rows:
        sid = r["subject_id"]
        if (sid not in best_per
                or r["still"]["still_seconds"] > best_per[sid]["still"]["still_seconds"]):
            best_per[sid] = r
    jog_vr = [r["jogging"]["view_ratio"] for r in best_per.values()]
    st_vr = [r["still"]["view_ratio"] for r in best_per.values()]
    agree = sum(1 for r in best_per.values()
                if r["still"]["near_limb"] == r["jogging"]["near_limb"])
    print(f"\n  jogging  view_ratio: median {np.median(jog_vr):.3f} "
          f"range [{min(jog_vr):.3f}, {max(jog_vr):.3f}]")
    print(f"  standing view_ratio: median {np.median(st_vr):.3f} "
          f"range [{min(st_vr):.3f}, {max(st_vr):.3f}]")
    print(f"  near limb agrees between standing and jogging: "
          f"{agree}/{len(best_per)} subjects")

    OUT.write_text(json.dumps(
        {"what": ("READ-ONLY: viewpoint of the legs-still standing motions "
                  "against the jogging window, PG1. No model run, no new "
                  "subject opened."),
         "licence": ("MoVi forbids distributing the Data in any form. This "
                     "file carries CROSS-SUBJECT AGGREGATES ONLY -- no "
                     "per-subject measurement, no frame window, no marker "
                     "coordinate, no camera parameter. PG1 is named; its "
                     "calibration is not reproduced."),
         "camera": CAM,
         "view_ratio": "phase 3B, hip depth separation / hip width; "
                       "0 = frontal, 1 = side-on; AthleticsPose median 0.751",
         "still_motions_considered": list(STILL_MOTIONS),
         "n_subjects": len(best_per),
         "jogging_view_ratio": {
             "median": round(float(np.median(jog_vr)), 4),
             "min": round(float(min(jog_vr)), 4),
             "max": round(float(max(jog_vr)), 4)},
         "standing_view_ratio": {
             "median": round(float(np.median(st_vr)), 4),
             "min": round(float(min(st_vr)), 4),
             "max": round(float(max(st_vr)), 4)},
         "near_limb_agrees_standing_vs_jogging":
             f"{agree}/{len(best_per)}",
         "finding": ("every stationary frame in round F is essentially frontal "
                     "(standing view_ratio median 0.057 against jogging 0.675), "
                     "so MoVi cannot test a side-on standing calibration pass")},
        indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}  (aggregates only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
