"""Phase 15D parts 2-3: run G1 on AthleticsPose, G2 on MoVi, apply both gates.

Pre-registered in `results/phase15d.md`, committed alone in e19980f. The gates
are evaluated in code against the thresholds fixed there.

The frozen angle code is not touched; `phase15d_segments` sits beside it.

Run: .venv/Scripts/python.exe scripts/phase15d_run.py [--ap-clips N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data" / "athleticspose" / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from athleticspose.statics.joints import mocap_joints_name_to_index as MK  # noqa: E402
from athleticspose.utils import mocap_to_h36m  # noqa: E402
from phase12_scan import subject_bootstrap_ci  # noqa: E402
from phase15_checks import (angle_mask, camera_pose, load_v3d,  # noqa: E402
                             to_camera, to_h36m)
from phase15_movi import (CAMERAS, MOCAP, MOCAP_PER_VIDEO, MOVI,  # noqa: E402
                           decode_range, jogging_window)
from phase15b_conventions import lift_2d, project  # noqa: E402
from phase15c_thorax import (LOWER, align_per_frame, clip_list,  # noqa: E402
                              gt3d_mm)
from phase15d_segments import (camera_tilt, level_camera_error,  # noqa: E402
                                segment_angles)
from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import DATA, TEST_SUBJECTS, load_model  # noqa: E402
from phase7_inference_fix import CKPT_FOR, lift  # noqa: E402
from video_kinematics import (coco17_to_h36m17, detect_coco17,  # noqa: E402
                               scores_to_train_scale)

OUT = REPO / "results" / "phase15d.json"
CAM = "PG1"

# The level-camera assumption, in the lifter's output frame. Verified in the
# pre-registration: head - pelvis has mean y of -0.4635, and the 2D puts the
# head at smaller image y (the usual y-down convention).
EST_UP = np.array([0.0, -1.0, 0.0])

G1_BAR = 4.118          # AthleticsPose hip MAE with the CLAV thorax, 15C step 4
RULE_SUBJECTS = ("Subject_1", "Subject_10", "Subject_12", "Subject_13")
DIAGNOSTIC_ONLY = ("Subject_11", "Subject_15")
QUANTITIES = ("thigh", "shank", "trunk", "knee_from_segments")


def stats(est_v: np.ndarray, ref_v: np.ndarray) -> dict:
    ok = np.isfinite(est_v) & np.isfinite(ref_v)
    if ok.sum() < 5:
        return {k: float("nan") for k in ("mae", "bias", "sd", "r")} | {"n": 0}
    d = (est_v - ref_v)[ok]
    return {"mae": float(np.abs(d).mean()), "bias": float(d.mean()),
            "sd": float(d.std(ddof=1)), "n": int(ok.sum()),
            "r": (float(np.corrcoef(ref_v[ok], est_v[ok])[0, 1])
                  if ref_v[ok].std() > 1e-9 and est_v[ok].std() > 1e-9
                  else float("nan"))}


def world_up_athleticspose(raw: np.ndarray) -> np.ndarray:
    """Which world axis is up, from the posture of the markers themselves."""
    v = np.nanmean(raw[:, MK["Vertex"]] - raw[:, MK["PELVIS"]], axis=0)
    ax = int(np.argmax(np.abs(v)))
    out = np.zeros(3)
    out[ax] = np.sign(v[ax])
    return out


def run_g1(model, limit: int | None) -> dict:
    """G1: thigh-inclination MAE on AthleticsPose S11/S13/S16."""
    clips = clip_list(limit)
    print(f"=== G1: AthleticsPose, {len(clips)} length-matched clips, "
          f"{list(TEST_SUBJECTS)} ===")
    per = {q: defaultdict(list) for q in QUANTITIES}
    hip_per = defaultdict(list)
    knee_check = []
    for i, (action, subj, stem, rawp) in enumerate(clips):
        raw = np.load(rawp).astype(float)
        gt = gt3d_mm(DATA / "gt_markers3d_by_cam" / action / subj / f"{stem}.npz")
        det = np.load(DATA / "det_markers2d_by_cam_coco" / action / subj
                      / f"{stem}.npy").astype(float)
        n = min(len(raw), len(gt), len(det))
        if n < 30:
            continue
        raw, gt, det = raw[:n], gt[:n], det[:n]

        # the world vertical, carried into the released camera frame by the
        # per-frame rotation already validated in phase 15C
        wup = world_up_athleticspose(raw)
        rebuilt = mocap_to_h36m(raw)
        _rc, tfs = align_per_frame(rebuilt, gt, list(range(17)), scale=False)
        ref_up = np.stack([R @ wup for R, _t, _s in tfs])

        est = lift(model, det, "edge", "overlap")
        for side in ("l", "r"):
            m = angle_mask(gt, side)
            ea = segment_angles(est, side, EST_UP)
            ga = segment_angles(gt, side, ref_up)
            for q in QUANTITIES:
                s = stats(ea[q][m], ga[q][m])
                if s["n"]:
                    per[q][subj].append(s)
            # the trunk-referenced hip, for the comparison G1's bar comes from
            hs = stats(sagittal_angles(est, side)["hip"][m],
                       sagittal_angles(gt, side)["hip"][m])
            if hs["n"]:
                hip_per[subj].append(hs)
            # the knee identity, on real data this time
            k1 = ga["knee_from_segments"][m]
            k2 = sagittal_angles(gt, side)["knee"][m]
            dd = np.abs(((k1 - k2) + 180) % 360 - 180)
            knee_check.append(float(np.nanmax(dd)))
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(clips)}")

    def agg(d):
        subj_means = {s: float(np.mean([x["mae"] for x in v]))
                      for s, v in d.items()}
        vals = list(subj_means.values())
        lo, hi = subject_bootstrap_ci(vals)
        return {"mae": float(np.mean(vals)), "ci95": [lo, hi],
                "per_subject": subj_means,
                "bias": float(np.mean([x["bias"] for v in d.values()
                                       for x in v])),
                "n_subjects": len(subj_means)}

    out = {"n_clips": len(clips),
           "knee_identity_max_abs_deviation_deg": float(np.max(knee_check)),
           "quantities": {q: agg(per[q]) for q in QUANTITIES if per[q]},
           "trunk_referenced_hip": agg(hip_per)}
    print(f"\n  knee identity on real data: max |deviation| "
          f"{out['knee_identity_max_abs_deviation_deg']:.2e} deg")
    print(f"  {'quantity':<22}{'MAE':>8}{'bias':>9}{'CI95':>22}")
    for q in QUANTITIES:
        if q not in out["quantities"]:
            continue
        a = out["quantities"][q]
        print(f"  {q:<22}{a['mae']:>8.3f}{a['bias']:>9.3f}"
              f"   [{a['ci95'][0]:>6.3f}, {a['ci95'][1]:>6.3f}]")
    a = out["trunk_referenced_hip"]
    print(f"  {'(trunk-ref hip)':<22}{a['mae']:>8.3f}{a['bias']:>9.3f}"
          f"   [{a['ci95'][0]:>6.3f}, {a['ci95'][1]:>6.3f}]")
    return out


def run_g2(model) -> dict:
    """G2: MoVi, detected and perfect 2D, plus the level-camera accounting."""
    print(f"\n=== G2: MoVi {CAM} ===")
    R_cv, _t, _s = camera_pose(CAM)
    tilt = camera_tilt(R_cv)
    print(f"  {CAM} camera: pitch {tilt['pitch_deg']:+.3f} deg, "
          f"roll {tilt['roll_deg']:+.3f} deg, total tilt "
          f"{tilt['tilt_deg']:.3f} deg")

    mats = {p.stem.split("_v3d_")[-1]: p
            for p in sorted(MOCAP.glob("F_v3d_*.mat"))}
    rows: dict = {}
    for subject in RULE_SUBJECTS + DIAGNOSTIC_ONLY:
        if subject not in mats:
            continue
        video = MOVI / CAMERAS[CAM]["pattern"].format(s=subject.split("_")[-1])
        if not video.exists():
            continue
        print(f"\n  --- {subject} ---")
        v = load_v3d(mats[subject])
        _ji, f30, f120 = jogging_window(v)
        ref_all = to_h36m(v)                       # MoVi WORLD coordinates
        frames, fps, _vm = decode_range(video, int(f30[0]), int(f30[1]))
        mk = int(f120[0]) + MOCAP_PER_VIDEO * np.arange(len(frames))
        mk = mk[mk < ref_all.shape[0]]
        frames = frames[:len(mk)]
        ref = ref_all[mk]
        ref_up = np.array([0.0, 0.0, 1.0])         # world vertical, check 2

        gc = to_camera(ref, R_cv, _t)
        near = ("l" if np.nanmean(gc[:, H36M["l_hip"], 2])
                < np.nanmean(gc[:, H36M["r_hip"], 2]) else "r")

        kp = coco17_to_h36m17(detect_coco17(frames, crop=True, model=None)[0])
        det_ok = np.abs(kp[:, :, :2]).sum(axis=(1, 2)) > 0
        est_det = lift_2d(model, np.nan_to_num(scores_to_train_scale(kp)), fps)

        p2d = project(ref, CAM)
        keep = np.isfinite(p2d).all(axis=(1, 2))
        est_perf = lift_2d(model, p2d[keep], fps)

        lce = level_camera_error(ref, R_cv)
        entry = {"near_limb": near, "n_frames": len(mk),
                 "in_rule_set": subject in RULE_SUBJECTS,
                 "level_camera_error_deg": {
                     "mean": float(np.nanmean(lce)),
                     "sd": float(np.nanstd(lce, ddof=1)),
                     "min": float(np.nanmin(lce)),
                     "max": float(np.nanmax(lce))},
                 "arms": {}}
        for arm, (est, sel) in (("detected", (est_det, det_ok)),
                                ("perfect", (est_perf, keep))):
            e = est[sel] if arm == "detected" else est
            r = ref[sel]
            entry["arms"][arm] = {}
            for limb, side in (("near", near),
                               ("far", "r" if near == "l" else "l")):
                m = angle_mask(r, side)
                ea = segment_angles(e, side, EST_UP)
                ga = segment_angles(r, side, ref_up)
                cell = {q: stats(ea[q][m], ga[q][m]) for q in QUANTITIES}
                cell["trunk_referenced_hip"] = stats(
                    sagittal_angles(e, side)["hip"][m],
                    sagittal_angles(r, side)["hip"][m])
                entry["arms"][arm][limb] = cell
            c = entry["arms"][arm]["near"]
            print(f"    {arm:<9} thigh MAE {c['thigh']['mae']:6.3f} "
                  f"bias {c['thigh']['bias']:+7.3f} r {c['thigh']['r']:+.3f} | "
                  f"trunk MAE {c['trunk']['mae']:6.3f} "
                  f"bias {c['trunk']['bias']:+7.3f} | "
                  f"hip(trunk-ref) bias {c['trunk_referenced_hip']['bias']:+7.3f}")
        rows[subject] = entry
    return {"camera_tilt": tilt, "subjects": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ap-clips", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    model = load_model(CKPT_FOR["coco"], "base")
    if model is None:
        raise SystemExit("checkpoint missing")

    g1 = run_g1(model, args.ap_clips or None)
    g2 = run_g2(model)

    # ---- gates, applied mechanically --------------------------------------
    thigh_mae = g1["quantities"]["thigh"]["mae"]
    g1v = {"rule": f"thigh-inclination MAE on AthleticsPose no worse than "
                   f"{G1_BAR} deg",
           "thigh_mae": thigh_mae, "bar": G1_BAR,
           "ci95": g1["quantities"]["thigh"]["ci95"],
           "passed": bool(thigh_mae <= G1_BAR)}

    per = {}
    for s in RULE_SUBJECTS:
        if s not in g2["subjects"]:
            per[s] = {"available": False}
            continue
        c = g2["subjects"][s]["arms"]["detected"]["near"]
        t, h = abs(c["thigh"]["bias"]), abs(c["trunk_referenced_hip"]["bias"])
        per[s] = {"available": True, "thigh_abs_bias": t,
                  "hip_abs_bias": h, "smaller": bool(t < h)}
    n_ok = sum(1 for x in per.values() if x.get("smaller"))
    g2v = {"rule": "thigh-inclination |bias| smaller than trunk-referenced hip "
                   "|bias| on at least 3 of the 4 rule subjects "
                   "(detected arm, near limb)",
           "per_subject": per, "n_smaller": n_ok, "n_required": 3,
           "passed": bool(n_ok >= 3)}

    print(f"\n=== GATES, against the thresholds fixed in e19980f ===")
    print(f"  G1  thigh MAE {thigh_mae:.3f} vs bar {G1_BAR}  -> "
          f"{'PASS' if g1v['passed'] else 'FAIL'}")
    print(f"  G2  thigh |bias| smaller on {n_ok} of 4  -> "
          f"{'PASS' if g2v['passed'] else 'FAIL'}")

    out = {"what": "Phase 15D: segment inclinations against gravity, beside "
                   "the frozen trunk-referenced angles. The phase 15 headline "
                   "is unchanged.",
           "pre_registration": "results/phase15d.md, committed alone in e19980f",
           "estimate_vertical": "level-camera assumption, up = (0,-1,0) in the "
                                "lifter's output frame",
           "reference_vertical": {
               "athleticspose": "world vertical of raw_markers_in_world, "
                                "carried into the released camera frame by the "
                                "phase 15C per-frame rotation",
               "movi": "world axis 2 (phase 15 check 2)"},
           "g1": g1, "g2": g2,
           "gates": {"g1": g1v, "g2": g2v}}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
