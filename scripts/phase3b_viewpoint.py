"""Phase 3B -- does the far-limb penalty grow as the view becomes side-on?

WHY THIS EXISTS
  results/phase03.md reported the far-limb penalty as "computed but
  uninformative", on the grounds that AthleticsPose contains only near-frontal
  views: "the median separation is 44 mm ... against ~200-250 mm for a true
  side-on view". results/phase04.md then carried that forward as the project's
  largest open gap.

  That comparison was wrong. `markers_h36m` is stored in PIXELS, not millimetres
  -- each npz carries a per-frame `p2mm` factor, and the released evaluation code
  divides by it (`y_sample / p2mm_sample[:, None, None]`, applied to all three
  axes) to reach millimetres. So 44 was compared against 200-250 mm.

  The unit-free quantity is what matters anyway:

      view ratio = |z_L - z_R| / ||p_L - p_R||

  the fraction of the subject's OWN hip-to-hip vector that lies along the camera
  depth axis. 0 = perfectly frontal, 1 = perfectly lateral, and arcsin of it is
  the angle of the pelvis out of the image plane. Being a ratio of two lengths in
  the same units, it is immune to the pixel/mm question entirely.

WHAT IS MEASURED
  The far-limb penalty (far MAE - near MAE) per clip, regressed on that clip's
  view ratio. If the penalty is flat across the range the dataset does span, then
  occlusion is not driving sagittal angle error, and extrapolating to a fully
  lateral view is defensible. If it climbs, phase 2's far-limb sweep is the right
  worry and the gap is real.

Run: .venv/Scripts/python.exe scripts/phase3b_viewpoint.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from phase3_angles import H36M  # noqa: E402

DATA = REPO / "data" / "athleticspose" / "data" / "AthleticsPoseDataset"
ROWS = REPO / "results" / "phase3_angle_errors.json"
OUT = REPO / "results" / "phase3b_viewpoint.json"
TEST = ["S11", "S13", "S16"]


def clip_geometry() -> dict[tuple[str, str, str], dict]:
    """View ratio and true-mm scale for every held-out clip."""
    geo = {}
    for action in ("running", "sprint"):
        for subj in TEST:
            d = DATA / "gt_markers3d_by_cam" / action / subj
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.npz")):
                z = np.load(f)
                g = np.asarray(z["markers_h36m"])
                if len(g) < 20:
                    continue
                p2mm = np.asarray(z["p2mm"], dtype=float).ravel()
                lh, rh = g[:, H36M["l_hip"]], g[:, H36M["r_hip"]]
                width = np.linalg.norm(lh - rh, axis=-1)
                depth = np.abs(lh[:, 2] - rh[:, 2])
                ratio = float(np.mean(depth / np.maximum(width, 1e-9)))
                n = min(len(p2mm), len(g))
                # mm = px / p2mm, per frame, exactly as the released evaluator does
                mm = float(np.mean(width[:n] / np.maximum(p2mm[:n], 1e-9)))
                depth_mm = float(np.mean(depth[:n] / np.maximum(p2mm[:n], 1e-9)))
                femur_mm = float(np.mean(
                    np.linalg.norm(g[:n, H36M["l_hip"]] - g[:n, H36M["l_knee"]],
                                   axis=-1) / np.maximum(p2mm[:n], 1e-9)))
                geo[(action, subj, f.stem)] = {
                    "view_ratio": ratio,
                    "out_of_plane_deg": float(np.degrees(np.arcsin(min(ratio, 1.0)))),
                    "pelvis_mm": mm, "depth_sep_mm": depth_mm,
                    "femur_mm": femur_mm,
                }
    return geo


def main() -> int:
    blob = json.loads(ROWS.read_text(encoding="utf-8"))
    rows = blob["rows"] if isinstance(blob, dict) else blob
    geo = clip_geometry()
    print(f"{len(geo)} held-out clips with geometry\n")

    r = np.array([g["view_ratio"] for g in geo.values()])
    print("=== ANATOMY CHECK: are the millimetres real? ===")
    print(f"  femur              {np.mean([g['femur_mm'] for g in geo.values()]):7.1f} mm"
          "   (anatomical 390-460)")
    print(f"  inter-hip-centre   {np.mean([g['pelvis_mm'] for g in geo.values()]):7.1f} mm"
          "   (anatomical 170-200 -- H36M hips are JOINT CENTRES, not iliac crests)")
    print(f"  hip depth sep      {np.mean([g['depth_sep_mm'] for g in geo.values()]):7.1f} mm"
          "   (phase 3 reported 44 -- that was pixels)")

    print("\n=== VIEWPOINT: how side-on is this dataset really? ===")
    print(f"  view ratio  median {np.median(r):.3f}  mean {r.mean():.3f}  "
          f"p10 {np.percentile(r,10):.3f}  p90 {np.percentile(r,90):.3f}")
    print(f"  out of plane  median {np.degrees(np.arcsin(np.median(r))):.1f} deg"
          f"   (0 = frontal, 90 = fully lateral)")
    for lo, hi, name in ((0.0, 0.3, "near-frontal"), (0.3, 0.6, "oblique"),
                         (0.6, 0.85, "strongly oblique"), (0.85, 1.01, "near-lateral")):
        m = (r >= lo) & (r < hi)
        print(f"    {name:<18} ratio [{lo:.2f},{hi:.2f})  {m.sum():>4} clips "
              f"({100*m.mean():5.1f}%)")

    # --- far-limb penalty vs view ratio ------------------------------------
    print("\n=== FAR-LIMB PENALTY vs VIEWPOINT ===")
    print("  If occlusion drives sagittal error, penalty must climb with ratio.\n")
    out: dict = {"geometry": {"n_clips": len(geo),
                              "view_ratio_median": float(np.median(r)),
                              "out_of_plane_deg_median": float(
                                  np.degrees(np.arcsin(np.median(r)))),
                              "femur_mm": float(np.mean(
                                  [g["femur_mm"] for g in geo.values()])),
                              "pelvis_mm": float(np.mean(
                                  [g["pelvis_mm"] for g in geo.values()])),
                              "depth_sep_mm": float(np.mean(
                                  [g["depth_sep_mm"] for g in geo.values()]))},
                 "models": {}}

    models = sorted({x["model"] for x in rows})
    for model in models:
        per_clip: dict[tuple, dict[str, list]] = {}
        for x in rows:
            if x["model"] != model:
                continue
            key = (x["action"], x["subject"], x["clip"])
            per_clip.setdefault(key, {"near": [], "far": []})
            per_clip[key][x["limb"]].append(x["mae_deg"])

        ratios, penalties, nears, fars = [], [], [], []
        for key, d in per_clip.items():
            if key not in geo or not d["near"] or not d["far"]:
                continue
            ratios.append(geo[key]["view_ratio"])
            nears.append(np.mean(d["near"]))
            fars.append(np.mean(d["far"]))
            penalties.append(np.mean(d["far"]) - np.mean(d["near"]))
        ratios = np.array(ratios)
        penalties = np.array(penalties)
        if len(ratios) < 20:
            continue

        rho = float(np.corrcoef(ratios, penalties)[0, 1])
        slope, intercept = np.polyfit(ratios, penalties, 1)
        # Slope CI by clip bootstrap -- clips are the independent unit here.
        rng = np.random.default_rng(7)
        boot = [np.polyfit(ratios[i], penalties[i], 1)[0]
                for i in (rng.integers(0, len(ratios), len(ratios))
                          for _ in range(2000))]
        lo, hi = np.percentile(boot, [2.5, 97.5])

        print(f"  {model}")
        print(f"    n={len(ratios)} clips | near {np.mean(nears):5.2f} deg | "
              f"far {np.mean(fars):5.2f} deg | penalty {penalties.mean():+5.2f} deg")
        print(f"    corr(penalty, view ratio) = {rho:+.3f}")
        print(f"    slope = {slope:+.2f} deg per unit ratio  "
              f"[{lo:+.2f}, {hi:+.2f}]  "
              f"{'EXCLUDES ZERO' if lo > 0 or hi < 0 else 'includes zero'}")
        print(f"    extrapolated to a fully lateral view (ratio 1.0): "
              f"{intercept + slope:+.2f} deg penalty")
        qs = np.percentile(ratios, [25, 50, 75])
        print(f"    {'ratio bin':<18}{'n':>5}{'near':>8}{'far':>8}{'penalty':>9}")
        edges = [ratios.min(), *qs, ratios.max() + 1e-9]
        for i in range(4):
            m = (ratios >= edges[i]) & (ratios < edges[i + 1])
            if m.sum():
                print(f"    [{edges[i]:.2f}, {edges[i+1]:.2f})".ljust(22)
                      + f"{m.sum():>3}{np.array(nears)[m].mean():>8.2f}"
                      f"{np.array(fars)[m].mean():>8.2f}"
                      f"{penalties[m].mean():>9.2f}")
        print()
        out["models"][model] = {
            "n_clips": len(ratios), "near_mae": float(np.mean(nears)),
            "far_mae": float(np.mean(fars)), "penalty": float(penalties.mean()),
            "corr_penalty_ratio": rho, "slope_per_unit_ratio": float(slope),
            "slope_ci": [float(lo), float(hi)],
            "penalty_at_ratio_1": float(intercept + slope),
        }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
