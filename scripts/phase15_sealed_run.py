"""Phase 15 SEALED TEST -- the one run on subjects that have never been opened.

Pre-registered in `results/phase15_sealed.md`, committed alone and PUSHED as
`268f28b` before a single sealed byte was downloaded. Subjects, camera, primary
quantity, secondary quantities, the diagnostic arm and every comparison figure
were fixed by that commit.

THIS SCRIPT TOUCHES NO FROZEN FILE. The six blobs pinned by the pre-registration
-- video_kinematics, phase15_movi, phase15d_segments, phase3_angles,
phase15_checks, phase7_inference_fix -- are imported, never edited. The code
path below is `phase15d_run.run_g2` applied to the sealed subjects, with the
reporting the pre-registration asks for.

There are NO GATES. The numbers are reported as they come out.

Run: .venv/Scripts/python.exe scripts/phase15_sealed_run.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from phase12_scan import N_BOOT, SEED, subject_bootstrap_ci  # noqa: E402
from phase15_checks import (angle_mask, camera_pose, load_v3d,  # noqa: E402
                            to_camera, to_h36m)
from phase15_movi import (CAMERAS, MOCAP, MOCAP_PER_VIDEO, MOVI,  # noqa: E402
                          decode_range, jogging_window)
from phase15b_conventions import lift_2d, project  # noqa: E402
from phase15d_run import EST_UP, stats  # noqa: E402
from phase15d_segments import camera_tilt, level_camera_error  # noqa: E402
from phase15d_segments import segment_angles  # noqa: E402
from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import load_model  # noqa: E402
from phase7_inference_fix import CKPT_FOR  # noqa: E402
from video_kinematics import (coco17_to_h36m17, detect_coco17,  # noqa: E402
                              scores_to_train_scale)

OUT = REPO / "results" / "phase15_sealed.json"
CAM = "PG1"

# Fixed by the pre-registration, before download.
SEALED = ("Subject_16", "Subject_17", "Subject_18", "Subject_19", "Subject_2")

# The pre-registration's comparison set: the diagnostic subjects, detected 2D,
# PG1, same bootstrap. These are QUOTED from `results/phase15_sealed.md`, which
# computed them into commit 268f28b so they cannot shift. They are not
# recomputed here.
DIAGNOSTIC_COMPARISON = {
    "knee": {"near": (9.269, [7.556, 11.326], 8.143),
             "far": (9.725, [7.938, 11.889], 8.157)},
    "thigh": {"near": (6.282, [5.270, 7.128], -5.473),
              "far": (6.487, [5.387, 7.844], -5.601)},
    "trunk": {"near": (7.174, [5.634, 8.095], -5.352),
              "far": (7.169, [5.632, 8.086], -5.349)},
    "trunk_referenced_hip": {"near": (12.270, [9.732, 13.926], -10.825),
                             "far": (11.874, [9.002, 14.423], -10.951)},
    "combined_hip_knee": {"near": (10.769, [9.330, 12.570], None),
                          "far": (10.799, [9.964, 11.787], None)},
}

# PRIMARY: the two quantities phases 15C/15D established mean the same thing in
# two labs. SECONDARY: what phase 15's own pre-registration named.
PRIMARY = ("knee_from_segments", "thigh")
SECONDARY = ("trunk_referenced_hip", "trunk")
LABEL = {"knee_from_segments": "knee flexion",
         "thigh": "thigh inclination (gravity)",
         "trunk": "trunk lean",
         "trunk_referenced_hip": "hip (trunk-referenced)",
         "shank": "shank inclination (gravity)",
         "combined_hip_knee": "combined hip+knee mean"}


def run_subject(model, subject: str, mat: Path) -> dict:
    """`phase15d_run.run_g2`'s body, for one sealed subject. No new logic."""
    video = MOVI / CAMERAS[CAM]["pattern"].format(s=subject.split("_")[-1])
    if not video.exists():
        return {"available": False, "why": "no PG1 video"}
    R_cv, t, _s = camera_pose(CAM)

    v = load_v3d(mat)
    _ji, f30, f120 = jogging_window(v)
    ref_all = to_h36m(v)                            # MoVi WORLD coordinates
    frames, fps, _vm = decode_range(video, int(f30[0]), int(f30[1]))
    mk = int(f120[0]) + MOCAP_PER_VIDEO * np.arange(len(frames))
    mk = mk[mk < ref_all.shape[0]]
    frames = frames[:len(mk)]
    ref = ref_all[mk]
    ref_up = np.array([0.0, 0.0, 1.0])              # world vertical, check 2

    gc = to_camera(ref, R_cv, t)
    near = ("l" if np.nanmean(gc[:, H36M["l_hip"], 2])
            < np.nanmean(gc[:, H36M["r_hip"], 2]) else "r")

    kp = coco17_to_h36m17(detect_coco17(frames, crop=True, model=None)[0])
    det_ok = np.abs(kp[:, :, :2]).sum(axis=(1, 2)) > 0
    est_det = lift_2d(model, np.nan_to_num(scores_to_train_scale(kp)), fps)

    p2d = project(ref, CAM)
    keep = np.isfinite(p2d).all(axis=(1, 2))        # never zero-filled
    est_perf = lift_2d(model, p2d[keep], fps)

    lce = level_camera_error(ref, R_cv)
    entry = {"available": True, "near_limb": near, "n_frames": int(len(mk)),
             "n_frames_detected": int(det_ok.sum()),
             "n_frames_perfect_2d": int(keep.sum()),
             "fps_read_from_file": float(fps),
             "level_camera_error_deg": {
                 "mean": float(np.nanmean(lce)),
                 "sd": float(np.nanstd(lce, ddof=1)),
                 "min": float(np.nanmin(lce)),
                 "max": float(np.nanmax(lce))},
             "arms": {}}

    for arm, (est, sel) in (("detected", (est_det, det_ok)),
                            ("perfect_2d_DIAGNOSTIC", (est_perf, keep))):
        e = est if arm != "detected" else est[sel]
        r = ref[sel]
        entry["arms"][arm] = {}
        for limb, side in (("near", near), ("far", "r" if near == "l" else "l")):
            m = angle_mask(r, side)
            ea = segment_angles(e, side, EST_UP)
            ga = segment_angles(r, side, ref_up)
            cell = {q: stats(ea[q][m], ga[q][m])
                    for q in ("thigh", "shank", "trunk", "knee_from_segments")}
            cell["trunk_referenced_hip"] = stats(
                sagittal_angles(e, side)["hip"][m],
                sagittal_angles(r, side)["hip"][m])
            # the phase 15 headline's form: per-subject mean of hip and knee
            cell["combined_hip_knee"] = {
                "mae": float(np.mean([cell["trunk_referenced_hip"]["mae"],
                                      cell["knee_from_segments"]["mae"]]))}
            # the pre-registered internal-consistency identity
            cell["identity"] = {
                "hip_bias": cell["trunk_referenced_hip"]["bias"],
                "thigh_bias_plus_trunk_bias": (cell["thigh"]["bias"]
                                               + cell["trunk"]["bias"]),
                "residual": (cell["trunk_referenced_hip"]["bias"]
                             - cell["thigh"]["bias"] - cell["trunk"]["bias"])}
            entry["arms"][arm][limb] = cell
        c = entry["arms"][arm]["near"]
        print(f"    {arm:<22} knee {c['knee_from_segments']['mae']:6.3f} "
              f"thigh {c['thigh']['mae']:6.3f} "
              f"trunk {c['trunk']['mae']:6.3f} "
              f"hip {c['trunk_referenced_hip']['mae']:6.3f}   "
              f"identity residual {c['identity']['residual']:+.2e}")
    return entry


def pool(rows: dict, arm: str, quantity: str, limb: str) -> dict:
    """Unweighted mean of per-subject MAE; CI by resampling SUBJECTS."""
    per, bias = {}, {}
    for s, e in rows.items():
        if not e.get("available"):
            continue
        c = e["arms"][arm][limb][quantity]
        if not np.isfinite(c["mae"]):
            continue
        per[s] = c["mae"]
        if "bias" in c:
            bias[s] = c["bias"]
    vals = list(per.values())
    if not vals:
        return {"n_subjects": 0}
    lo, hi = subject_bootstrap_ci(vals)
    out = {"mae": float(np.mean(vals)), "ci95": [lo, hi],
           "per_subject": per, "n_subjects": len(vals),
           "n_boot": N_BOOT, "seed": SEED}
    if bias:
        out["bias"] = float(np.mean(list(bias.values())))
        out["per_subject_bias"] = bias
        out["sd_of_subject_maes"] = (float(np.std(vals, ddof=1))
                                     if len(vals) > 1 else float("nan"))
    return out


def table(title: str, rows: dict, arm: str, quantities, comparison: bool) -> str:
    out = [f"\n=== {title} ===",
           f"  {'quantity':<30}{'limb':<6}{'MAE':>8}{'95% CI':>20}"
           f"{'bias':>9}" + ("      diagnostic-set MAE (quoted)"
                             if comparison else "")]
    for q in quantities:
        for limb in ("near", "far"):
            a = pool(rows, arm, q, limb)
            if not a.get("n_subjects"):
                continue
            line = (f"  {LABEL.get(q, q):<30}{limb:<6}{a['mae']:>8.3f}"
                    f"   [{a['ci95'][0]:>6.3f}, {a['ci95'][1]:>6.3f}]"
                    f"{a.get('bias', float('nan')):>+9.3f}")
            if comparison:
                key = ("combined_hip_knee" if q == "combined_hip_knee"
                       else {"knee_from_segments": "knee",
                             "thigh": "thigh", "trunk": "trunk",
                             "trunk_referenced_hip":
                                 "trunk_referenced_hip"}.get(q))
                cmp_ = DIAGNOSTIC_COMPARISON.get(key, {}).get(limb)
                if cmp_:
                    line += f"      {cmp_[0]:>7.3f}  ({a['mae'] - cmp_[0]:+.3f})"
            out.append(line)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    model = load_model(CKPT_FOR["coco"], "base")
    if model is None:
        raise SystemExit("checkpoint missing")

    mats = {p.stem.split("_v3d_")[-1]: p
            for p in sorted(MOCAP.glob("F_v3d_*.mat"))}
    R_cv, _t, _s = camera_pose(CAM)
    tilt = camera_tilt(R_cv)
    print(f"=== PHASE 15 SEALED TEST -- {CAM}, {len(SEALED)} subjects, "
          f"run once ===")
    print(f"  pre-registration: results/phase15_sealed.md, pushed as 268f28b")
    print(f"  {CAM} camera: pitch {tilt['pitch_deg']:+.3f} deg, "
          f"roll {tilt['roll_deg']:+.3f} deg, tilt {tilt['tilt_deg']:.3f} deg")

    rows: dict = {}
    for subject in SEALED:
        if subject not in mats:
            rows[subject] = {"available": False, "why": "no mocap"}
            continue
        print(f"\n  --- {subject} ---")
        rows[subject] = run_subject(model, subject, mats[subject])

    arms = ["detected", "perfect_2d_DIAGNOSTIC"]
    txt = [table("PRIMARY -- detected 2D, the cross-lab claim", rows,
                 "detected", PRIMARY, True),
           table("SECONDARY -- the originally pre-registered quantities", rows,
                 "detected", SECONDARY + ("combined_hip_knee",), True),
           table("DIAGNOSTIC -- projected reference 2D, NEVER pooled into the "
                 "primary", rows, "perfect_2d_DIAGNOSTIC",
                 PRIMARY + SECONDARY, False)]
    for t in txt:
        print(t)

    print("\n=== the pre-registered identity, per subject "
          "(detected, near limb) ===")
    print(f"  {'subject':<12}{'hip bias':>10}{'thigh+trunk':>13}"
          f"{'residual':>12}")
    ident = {}
    for s, e in rows.items():
        if not e.get("available"):
            continue
        i = e["arms"]["detected"]["near"]["identity"]
        ident[s] = i
        print(f"  {s:<12}{i['hip_bias']:>+10.4f}"
              f"{i['thigh_bias_plus_trunk_bias']:>+13.4f}"
              f"{i['residual']:>+12.2e}")

    out = {
        "what": ("Phase 15 SEALED TEST: the frozen pipeline run once on five "
                 "MoVi subjects that had never been opened."),
        "pre_registration": ("results/phase15_sealed.md, committed alone and "
                             "pushed as 268f28b before any sealed byte was "
                             "downloaded"),
        "gates": "none -- there is no pass/fail threshold on any sealed number",
        "camera": CAM, "camera_tilt": tilt,
        "subjects_sealed": list(SEALED),
        "exclusions": [s for s, e in rows.items() if not e.get("available")],
        "checks": "results/phase15_sealed_checks.json -- all five pass 1/2/5/6",
        "estimate_vertical": ("level-camera assumption, up = (0,-1,0) in the "
                              "lifter's output frame"),
        "reference_vertical": "MoVi world axis 2 (phase 15 check 2)",
        "bootstrap": {"n_boot": N_BOOT, "seed": SEED,
                      "resampled": "subjects, not clips or frames"},
        "per_subject": rows,
        "pooled": {arm: {q: {limb: pool(rows, arm, q, limb)
                             for limb in ("near", "far")}
                         for q in (PRIMARY + SECONDARY
                                   + ("shank", "combined_hip_knee"))}
                   for arm in arms},
        "primary": list(PRIMARY),
        "secondary": list(SECONDARY) + ["combined_hip_knee", "identity"],
        "diagnostic_comparison_quoted_from_prereg": DIAGNOSTIC_COMPARISON,
        "identity_near_detected": ident,
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
