"""Every number quoted in README.md, pulled from results/*.json.

The README is prose, so its figures could drift from the artifacts they describe.
This prints them from source. Re-run it after any phase is re-run; if a number
here disagrees with the README, the README is wrong.

Run: .venv/Scripts/python.exe scripts/phase6_numbers.py
"""

from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"


def L(n):
    return json.loads((RES / n).read_text(encoding="utf-8"))


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


p2, p3, p3b = L("phase2_degradation.json"), L("phase3_angle_errors.json"), L("phase3b_viewpoint.json")
p4, p4b = L("phase4_real_delta.json"), L("phase4b_operating_point.json")
p5, p5b, p5c, p5d = (L("phase5_limb.json"), L("phase5b_ceiling.json"),
                     L("phase5c_features.json"), L("phase5d_selective.json"))
p1d, pre = L("phase1d_percondition.json"), L("phase2_realized_error.json")

print("== COHORT ==")
print(f"  {p4['n_sessions']} unilateral sessions / {p2['n_subjects']} subjects, "
      f"chance {p4['chance']:.3f}")

print("\n== SCREENING (phase 1D) ==")
tests = p1d["all_tests"]
pos = [t for t in tests if t.get("survives_bh")] if isinstance(tests[0], dict) else []
print(f"  {p1d['n_tests']} pre-registered tests, surviving BH: {len(pos)}")

print("\n== WITHIN-SUBJECT LIMB (phase 5) ==")
r5 = p5["results"]
for k in ("limbsag_mean [matched]", "NEG provenance", "NEG demographics",
          "NEG structure", "DominantLeg"):
    print(f"  {k:<28} {r5[k]['auc_mean']:.3f} {ci(r5[k]['auc_ci'])}")

print("\n== PHASE 3: monocular error ==")
for k, v in p3["mpjpe_mm"].items():
    print(f"  {k:<26} MPJPE {v['mean']:.1f} mm (median {v['median']:.1f})")
for k, v in p3b["models"].items():
    print(f"  {k:<26} near {v['near_mae']:.2f}° far {v['far_mae']:.2f}° "
          f"penalty {v['penalty']:+.2f}° | at fully lateral {v['penalty_at_ratio_1']:+.2f}°")
g = p3b["geometry"]
print(f"  viewpoint: ratio median {g['view_ratio_median']:.3f} "
      f"= {g['out_of_plane_deg_median']:.1f}° out of plane; "
      f"femur {g['femur_mm']:.0f} mm, pelvis {g['pelvis_mm']:.0f} mm")

print("\n== PHASE 2: nominal sigma vs realized error ==")
for k in ("k101_s2", "k101_s8", "k101_s15", "k9_s0", "k9_s8", "k9_s15"):
    print(f"  {k:<10} {pre[k]:.2f}°")

print("\n== PHASE 4: real delta ==")
print(f"  error bank mean |err| {p4['bank_mae_deg']:.2f}°")
print(f"  clean wave2 {p4['clean']['wave2']:.3f}  wave3 {p4['clean']['wave3']:.3f}")
for s in p4["summary"]:
    if s["features"] == "wave2":
        print(f"  wave2 {s['bank']:<13} {s['fps']:<6} err {s['realized_error_deg']:.2f}° "
              f"AUC {s['auc_mean']:.3f} {ci(s['ci'])} dAUC {s['delta_vs_clean']:+.3f}")

print("\n== PHASE 4B: operating point ==")
for name, c in p4b["configs"].items():
    print(f"  {name}")
    print(f"    pooled OOF AUC {c['auc_pooled_oof']:.3f}")
    for o in c["operating_points"][:3]:
        print(f"      {o['point']:<22} sens {o['sensitivity']:.3f} "
              f"spec {o['specificity']:.3f} PPV {o['ppv']:.3f}")
    cal, rep = c["calibration"], c["repeatability"]
    print(f"    calibration slope {cal['calibration_slope']:.3f} | "
          f"Brier {cal['brier']:.4f} vs {cal['brier_baseline']:.4f}")
    print(f"    repeatability {rep['binary_agreement']:.3f} over "
          f"{rep['n_session_pairs']} pairs / {rep['n_subjects_same_side']} subjects")

print("\n== PHASE 5B: ceiling diagnostics ==")
d1 = p5b["D1"]
print(f"  D1 severity rho {d1['spearman_rho']:+.4f} {ci(d1['spearman_ci'])} "
      f"p={d1['spearman_p']:.3f} | partial {d1['partial_rho']:+.4f}")
for k, v in d1["strata"].items():
    print(f"     rank {v['rank']} {k:<38} n={v['n']:>4} AUC {v['auc_mean']:.3f}")
d2 = p5b["D2"]
print(f"  D2 dv_r {d2['n_metrics']} metrics: {d2['dvr_auc']:.3f} {ci(d2['dvr_ci'])} "
      f"| dAUC {d2['delta_dvr_vs_sag']:+.3f} {ci(d2['delta_dvr_ci'])}")
d3 = p5b["D3"]
# derived rather than read, so this works against JSONs written before the
# field existed; agrees with phase5b_ceiling.py's own calculation
_inc = d3["curve"][-1]["auc_mean"] - d3["curve"][-2]["auc_mean"]
print("  D3 learning curve: " + " -> ".join(f"{c['auc_mean']:.3f}" for c in d3["curve"])
      + f" | final increment {_inc:+.3f}")

print("\n== PHASE 5C: attempts ==")
for f in p5c["feature_sets"]:
    print(f"  {f['name']:<34} {f['auc_mean']:.3f} {ci(f['ci'])} "
          f"dAUC {f['delta']:+.3f} {ci(f['delta_ci'])} "
          f"{'EXCL 0' if f['excludes_zero'] else ''}")
print(f"  conditions surviving both intervals: "
      f"{sum(c['survives'] for c in p5c['per_condition'])} / {len(p5c['per_condition'])}")
for c in sorted(p5c["per_condition"], key=lambda r: -r["auc_mean"]):
    print(f"     {c['condition']:<32} n={c['n']:>4} AUC {c['auc_mean']:.3f} "
          f"fold {ci(c['ci'])} boot {ci(c['boot_ci'])}")
for k, v in p5c.get("repeatability", {}).items():
    print(f"  repeat {k:<36} agreement {v['repeatability']['binary_agreement']:.3f}")

print("\n== PHASE 5D: selective classification ==")
print(f"  majority-guess accuracy {p5d['majority_accuracy']:.3f}, "
      f"control drift {p5d['control_drift']:.3f}")
for name, rows in p5d["curves"].items():
    hit = [r for r in rows if r["coverage"] in (1.0, 0.5, 0.3)]
    print(f"  {name}")
    for r in hit:
        print(f"     coverage {r['coverage']:.0%}  acc {r['accuracy']:.3f} "
              f"{ci(r['accuracy_ci'])}")
m = p5d["multi_session"]
print(f"  multi-session: single {m['single_session_accuracy']:.3f} -> "
      f"averaged {m['averaged_accuracy']:.3f} ({m['gain']:+.3f}) "
      f"over {m['n_subjects']} subjects")
