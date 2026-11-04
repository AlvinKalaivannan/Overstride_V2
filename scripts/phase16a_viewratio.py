"""Is the ESTIMATED view_ratio on the same scale as the thresholds?

The design doc's `[A5]` viewpoint bars -- PASS >= 0.50, WARN 0.35-0.50, REFUSE
below 0.35 -- are derived from **ground-truth** `view_ratio` measured from
calibrated 3D: MoVi PG1 jogging spans [0.536, 0.720]. The demo has no
calibration and computes `overstride.checks.estimate_view_ratio` from the
detected 2D via phase 14's azimuth proxy.

Phase 14 measured that proxy at **Pearson 0.974** against the ground-truth depth
metric -- but correlation is not scale. An absolute threshold needs the scale to
be right, so this measures the estimate against ground truth on all 11 spent
MoVi PG1 subjects and answers one question:

> **Would the demo's own check have flagged the footage Row B was measured on?**

If the estimate is biased low, the 0.50 bar would warn on Row B's own clips --
exactly the incoherence amendment `[A5]` was written to remove -- and the bar
must then be restated in estimate units.

Aggregate output only; no per-subject MoVi value is written to the repository.

Run: .venv/Scripts/python.exe scripts/phase16a_viewratio.py
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

from overstride.checks import (VIEW_REFUSE, VIEW_WARN,  # noqa: E402
                               check_viewpoint, estimate_view_ratio)
from phase11_viewpoint import view_ratio  # noqa: E402
from phase15_checks import (camera_pose, load_v3d, to_camera,  # noqa: E402
                            to_h36m)
from phase15_movi import (CAMERAS, MOCAP, MOCAP_PER_VIDEO, MOVI,  # noqa: E402
                          decode_range, jogging_window)
from phase3_angles import H36M  # noqa: E402
from video_kinematics import coco17_to_h36m17, detect_coco17  # noqa: E402

OUT = REPO / "results" / "phase16a_viewratio.json"
CAM = "PG1"
SUBJECTS = ("Subject_1", "Subject_10", "Subject_11", "Subject_12",
            "Subject_13", "Subject_15", "Subject_16", "Subject_17",
            "Subject_18", "Subject_19", "Subject_2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject-out", type=Path, default=None)
    args = ap.parse_args()

    R_cv, t_cv, _s = camera_pose(CAM)
    print("=== estimated view_ratio vs ground truth, 11 spent subjects ===")
    print(f"    bars: PASS >= {VIEW_WARN}, REFUSE < {VIEW_REFUSE}\n")
    print(f"  {'subject':<12}{'GT vr':>8}{'est vr':>9}{'err':>8}{'verdict':>9}")

    rows = []
    for subject in SUBJECTS:
        mat = MOCAP / f"F_v3d_{subject}.mat"
        video = MOVI / CAMERAS[CAM]["pattern"].format(s=subject.split("_")[-1])
        if not (mat.exists() and video.exists()):
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
        ok = np.isfinite(gc[:, [H36M["l_hip"], H36M["r_hip"]]]).all(axis=(1, 2))
        gt = float(view_ratio(gc[ok])) if ok.sum() > 5 else float("nan")

        kp = coco17_to_h36m17(detect_coco17(frames, crop=True, model=None)[0])
        est = estimate_view_ratio(kp)
        verdict = check_viewpoint(kp).status
        rows.append({"subject": subject, "gt": round(gt, 4),
                     "est": round(est, 4), "err": round(est - gt, 4),
                     "verdict": verdict})
        print(f"  {subject:<12}{gt:>8.4f}{est:>9.4f}{est - gt:>+8.4f}"
              f"{verdict:>9}")

    gts = np.array([r["gt"] for r in rows])
    ests = np.array([r["est"] for r in rows])
    errs = ests - gts
    n_pass = sum(1 for r in rows if r["verdict"] == "pass")
    n_flag = len(rows) - n_pass
    r_p = (float(np.corrcoef(gts, ests)[0, 1]) if len(rows) > 2
           else float("nan"))

    print(f"\n  BINDING CHECK: {n_pass}/{len(rows)} of Row B's own footage "
          f"passes its own viewpoint check")
    if n_flag:
        print(f"  *** {n_flag} clip(s) the budget was measured on would be "
              f"FLAGGED -- the bar must be restated in estimate units ***")
    print(f"  est - gt: mean {errs.mean():+.4f}, sd {errs.std(ddof=1):.4f}, "
          f"range [{errs.min():+.4f}, {errs.max():+.4f}]")
    print(f"  Pearson(gt, est) = {r_p:.4f}  (phase 14 reported 0.974 for the "
          f"proxy against its own GT metric)")
    print(f"  est range [{ests.min():.4f}, {ests.max():.4f}]  "
          f"vs gt [{gts.min():.4f}, {gts.max():.4f}]")

    out = {
        "what": ("Does the ESTIMATED view_ratio sit on the same scale as the "
                 "ground-truth thresholds? 11 spent MoVi PG1 subjects."),
        "why": ("the [A5] bars are in ground-truth units; the demo computes an "
                "estimate from the detected 2D, and an absolute threshold "
                "needs the scale, not just the correlation"),
        "estimator": ("overstride.checks.estimate_view_ratio -- phase 14's "
                      "azimuth proxy, sqrt(1 - (ratio/R_POP)^2)"),
        "bars": {"pass_at_or_above": VIEW_WARN, "refuse_below": VIEW_REFUSE},
        "aggregate_only": "no per-subject MoVi value is written here",
        "diagnostic_only": "all 11 subjects are spent",
        "n_subjects": len(rows),
        "n_pass": n_pass, "n_flagged": n_flag,
        "binding_check": (f"{n_pass}/{len(rows)} of the footage Row B was "
                          f"measured on passes its own viewpoint check"),
        "est_minus_gt": {"mean": round(float(errs.mean()), 4),
                         "sd": round(float(errs.std(ddof=1)), 4),
                         "min": round(float(errs.min()), 4),
                         "max": round(float(errs.max()), 4)},
        "pearson_gt_est": round(r_p, 4),
        "gt_range": [round(float(gts.min()), 4), round(float(gts.max()), 4)],
        "est_range": [round(float(ests.min()), 4), round(float(ests.max()), 4)],
    }
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
