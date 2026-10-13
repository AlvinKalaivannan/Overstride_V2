"""Phase 12 aggregation: the gate verdict from results/phase12_scan.json.

Separate from the scan itself so the verdict can be recomputed without re-lifting
1,686 clips. Reads per-clip rows and applies the bars pre-registered in
results/phase12.md. It does not choose them.

One consequence of the lifter being scale-free that is easy to get wrong:
stability must also be measured on RATIOS, not on raw lengths. Each clip is
normalised by its own max|xy|, so raw scale-free lengths carry an arbitrary
per-clip factor and their CV would measure the normalisation, not the body.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from phase12_scan import cv, subject_bootstrap_ci  # noqa: E402

ACC_BAR = 3.0      # per cent, pre-registered
CV_BAR = 3.0       # per cent, pre-registered


def _mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _verdict(value: float, bar: float) -> bool | None:
    """met / not met / undefined. A nan is missing data, never a failure."""
    return None if not np.isfinite(value) else bool(value <= bar)


def _label(v: bool | None) -> str:
    return "n/a" if v is None else ("met" if v else "NOT met")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", type=Path, default=REPO / "results" / "phase12_scan.json")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "phase12_gate.json")
    args = ap.parse_args()

    d = json.loads(args.scan.read_text(encoding="utf-8"))
    rows = d["rows"]
    ratios = list(d["ratios"])
    segments = list(d["segments"])
    print(f"phase 12 gate: {len(rows)} clips, {d['n_subjects']} subjects, "
          f"groups={d['groups']}")

    # ---------------------------------------------------------- accuracy -----
    # per subject first, then across subjects. Never pooled over clips: 8
    # cameras share a trial, so clips are not independent.
    acc: dict[str, dict] = {}
    for r_name in ratios:
        by_subj = defaultdict(list)
        for row in rows:
            v = row["ratio_err_pct"].get(r_name)
            if v is not None and np.isfinite(v):
                by_subj[row["subject"]].append(v)
        per_subj = [_mean(v) for v in by_subj.values()]
        m = _mean(per_subj)
        lo, hi = subject_bootstrap_ci(per_subj)
        acc[r_name] = {"mean_err_pct": m, "ci": [lo, hi],
                       "n_subjects": len(per_subj),
                       "meets_bar": _verdict(m, ACC_BAR)}

    # ------------------------------------------------- near vs far limb ------
    # Mandatory: a scan that recovers the near limb and fails the far one is a
    # failed scan, and a pooled mean would hide it.
    nf: dict[str, dict] = {}
    for limb in ("near", "far"):
        by_subj = defaultdict(list)
        for row in rows:
            for r_name, v in row["ratio_err_pct"].items():
                side = r_name[-1]                      # 'l' or 'r'
                if side not in ("l", "r"):
                    continue
                is_near = (side == row["near"])
                if (limb == "near") == is_near and np.isfinite(v):
                    by_subj[row["subject"]].append(v)
        per_subj = [_mean(v) for v in by_subj.values()]
        lo, hi = subject_bootstrap_ci(per_subj)
        nf[limb] = {"mean_err_pct": _mean(per_subj), "ci": [lo, hi],
                    "n_subjects": len(per_subj)}

    # --------------------------------------------------------- stability -----
    # One scan per trial = median of that trial's cameras. Then CV across a
    # subject's trials (the gate), and CV across cameras within a trial.
    per_trial: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["action"], row["subject"], row["trial"])
        for r_name in ratios:
            v = row["est_ratios"].get(r_name)
            if v is not None and np.isfinite(v):
                per_trial[key][r_name].append(v)

    stab: dict[str, dict] = {}
    for r_name in ratios:
        across_trials, across_cams = defaultdict(list), []
        for (_, subj, _), vals in per_trial.items():
            v = vals.get(r_name, [])
            if not v:
                continue
            across_trials[subj].append(float(statistics.median(v)))
            if len(v) >= 2:
                across_cams.append(cv(v))
        subj_cv = [cv(v) for v in across_trials.values() if len(v) >= 2]
        m = _mean(subj_cv)
        lo, hi = subject_bootstrap_ci(subj_cv)
        stab[r_name] = {
            "cv_across_trials_pct": m, "ci": [lo, hi], "n_subjects": len(subj_cv),
            "cv_across_cameras_pct": _mean(across_cams),
            "meets_bar": _verdict(m, CV_BAR),
        }

    # -------------------------------------------------------------- kill -----
    # Does collapsing a clip to one number beat reading a single frame? Compare
    # across-trial CV of the clip median against within-clip frame-to-frame CV.
    within = {s: _mean([row["per_frame_cv_pct"].get(s, float("nan")) for row in rows])
              for s in segments}
    kill = {
        "within_clip_per_frame_cv_pct": within,
        "across_trial_cv_pct_mean": _mean([stab[r]["cv_across_trials_pct"]
                                           for r in ratios]),
        "within_clip_cv_pct_mean": _mean(list(within.values())),
    }
    kill["scan_adds_nothing"] = bool(
        np.isfinite(kill["across_trial_cv_pct_mean"])
        and np.isfinite(kill["within_clip_cv_pct_mean"])
        and kill["across_trial_cv_pct_mean"] >= kill["within_clip_cv_pct_mean"])

    # ------------------------------------- absolute mm, anatomy check only ---
    mm: dict[str, dict] = {}
    for s in segments:
        est = [row["est_mm"].get(s) for row in rows if row.get("est_mm")]
        est = [x for x in est if x is not None and np.isfinite(x)]
        gt_h = [row["gt_h36m_mm"][s] for row in rows if s in row["gt_h36m_mm"]]
        gt_a = [row["gt_anatomical_mm"].get(s) for row in rows]
        gt_a = [x for x in gt_a if x is not None and np.isfinite(x)]
        mm[s] = {"est_median_mm": float(np.median(est)) if est else float("nan"),
                 "gt_h36m_median_mm": float(np.median(gt_h)) if gt_h else float("nan"),
                 "gt_anatomical_median_mm": float(np.median(gt_a)) if gt_a
                 else float("nan")}

    # ------------------------------------------------- shape bias in mm -----
    # Why the ratios fail. Per subject then across, never pooled: a median of
    # est over all clips divided by a median of gt over all clips is a different
    # estimator and gives a different answer.
    bias: dict[str, dict] = {}
    for s in segments:
        by_subj = defaultdict(list)
        for row in rows:
            e = (row.get("est_mm") or {}).get(s)
            g = row["gt_h36m_mm"].get(s)
            if e is not None and g:
                by_subj[row["subject"]].append(100.0 * (e - g) / g)
        per_subj = [_mean(v) for v in by_subj.values()]
        lo, hi = subject_bootstrap_ci(per_subj)
        bias[s] = {"bias_pct": _mean(per_subj), "ci": [lo, hi],
                   "sd_across_subjects": float(np.std(
                       [x for x in per_subj if np.isfinite(x)], ddof=1))
                   if len(per_subj) > 1 else float("nan"),
                   "n_subjects": len(per_subj)}
    _limb = [s for s in segments if s.startswith(("thigh", "shank"))]
    bias["_limb_mean_pct"] = _mean([bias[s]["bias_pct"] for s in _limb])

    # --------------------------------------------- L/R symmetry in the GT ---
    sym = {}
    for subj in d["subjects"]:
        r = [row for row in rows if row["subject"] == subj]
        if not r:
            continue
        a = r[0]["gt_anatomical_mm"]
        for seg in ("thigh", "shank", "foot"):
            lk, rk = f"{seg}_l", f"{seg}_r"
            if lk in a and rk in a and a[rk]:
                sym.setdefault(seg, {})[subj] = float(
                    100.0 * abs(a[lk] - a[rk]) / ((a[lk] + a[rk]) / 2))

    n_acc = sum(1 for v in acc.values() if v["meets_bar"] is True)
    n_stab = sum(1 for v in stab.values() if v["meets_bar"] is True)
    und_acc = sum(1 for v in acc.values() if v["meets_bar"] is None)
    und_stab = sum(1 for v in stab.values() if v["meets_bar"] is None)
    payload = {
        "bars": {"accuracy_pct": ACC_BAR, "cv_pct": CV_BAR,
                 "source": "results/phase12.md, pre-registered 2026-10-13"},
        "n_clips": len(rows), "n_subjects": d["n_subjects"], "groups": d["groups"],
        "accuracy": acc, "near_far": nf, "stability": stab,
        "kill_criterion": kill, "absolute_mm": mm, "shape_bias_pct": bias,
        "gt_lr_asymmetry_pct": sym,
        "verdict": {
            "accuracy_ratios_meeting_bar": f"{n_acc}/{len(ratios)}",
            "stability_ratios_meeting_bar": f"{n_stab}/{len(ratios)}",
            "undefined_accuracy": und_acc,
            "undefined_stability": und_stab,
            "gate_met": bool(n_acc == len(ratios) and n_stab == len(ratios)),
        },
    }
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    # ------------------------------------------------------------- print -----
    print(f"\naccuracy (bar {ACC_BAR}% -- mean |ratio error|, per subject then across)")
    print(f"  {'ratio':18} {'err%':>7} {'95% CI':>18}  bar")
    for r_name in ratios:
        v = acc[r_name]
        print(f"  {r_name:18} {v['mean_err_pct']:7.2f} "
              f"[{v['ci'][0]:7.2f},{v['ci'][1]:7.2f}]  "
              f"{_label(v['meets_bar'])}")
    print(f"\nnear vs far limb")
    for limb in ("near", "far"):
        v = nf[limb]
        print(f"  {limb:5} {v['mean_err_pct']:7.2f}%  "
              f"[{v['ci'][0]:7.2f},{v['ci'][1]:7.2f}]  n={v['n_subjects']}")
    print(f"\nstability (bar {CV_BAR}% CV across a subject's trials)")
    print(f"  {'ratio':18} {'trialCV%':>9} {'camCV%':>8}  bar")
    for r_name in ratios:
        v = stab[r_name]
        print(f"  {r_name:18} {v['cv_across_trials_pct']:9.2f} "
              f"{v['cv_across_cameras_pct']:8.2f}  "
              f"{_label(v['meets_bar'])}")
    print(f"\nkill criterion")
    print(f"  across-trial CV of the clip median : "
          f"{kill['across_trial_cv_pct_mean']:.2f}%")
    print(f"  within-clip frame-to-frame CV      : "
          f"{kill['within_clip_cv_pct_mean']:.2f}%")
    print(f"  scan adds nothing: {kill['scan_adds_nothing']}")
    print(f"\nabsolute mm (anatomy check, NOT a gate)")
    print(f"  {'segment':10} {'est':>8} {'gt_h36m':>9} {'gt_anat':>9}")
    for s in segments:
        v = mm[s]
        print(f"  {s:10} {v['est_median_mm']:8.1f} {v['gt_h36m_median_mm']:9.1f} "
              f"{v['gt_anatomical_median_mm']:9.1f}")
    print("\nshape bias vs H36M GT, per subject then across (why the ratios fail)")
    print(f"  {'segment':10} {'bias%':>8} {'95% CI':>18} {'sd across subj':>15}")
    for s_ in segments:
        v = bias[s_]
        print(f"  {s_:10} {v['bias_pct']:+8.2f} "
              f"[{v['ci'][0]:+7.2f},{v['ci'][1]:+7.2f}] "
              f"{v['sd_across_subjects']:15.2f}")
    print(f"  limb segments mean: {bias['_limb_mean_pct']:+.2f}%")
    print("\nmocap L/R asymmetry, flagged above 5%")
    for seg, per in sorted(sym.items()):
        bad = {k: round(x, 1) for k, x in sorted(per.items()) if x > 5.0}
        print(f"  {seg:6} max {max(per.values()):5.1f}%   over 5%: {bad or 'none'}")
    print(f"\nVERDICT  accuracy {n_acc}/{len(ratios)}   stability {n_stab}/{len(ratios)}"
          f"   gate met: {payload['verdict']['gate_met']}")
    try:
        shown: Path | str = args.out.relative_to(REPO)
    except ValueError:
        shown = args.out
    print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
