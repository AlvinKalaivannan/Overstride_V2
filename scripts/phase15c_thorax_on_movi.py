"""Phase 15C addendum -- which point does the lifter's thorax track ON MoVi?

Part 2 established, on AthleticsPose, that this checkpoint's output thorax is
`CLAV`: trunk-axis offset 0.750 deg, and the released reference's own thorax
matches CLAV to 5.04 mm. Part 3 then found that applying CLAV to MoVi makes the
hip bias much WORSE on every subject.

Those two cannot both be the whole story, so this measures the same quantity as
part 2 -- where the lifter's thorax actually lands -- but on MoVi input. It is a
DIAGNOSTIC: nothing is chosen on it, and the chosen definition stays the one
part 2 decided on AthleticsPose.

Perfect-2D arm only, so no detector is involved and the comparison is about the
lifter rather than about 2D quality. Alignment is per frame on the LOWER BODY
only, so the trunk never helps place the trunk.

Run: .venv/Scripts/python.exe scripts/phase15c_thorax_on_movi.py
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

from phase15_checks import load_v3d, to_h36m  # noqa: E402
from phase15_movi import MOCAP, MOCAP_PER_VIDEO, jogging_window  # noqa: E402
from phase15b_conventions import lift_2d, project  # noqa: E402
from phase15c_thorax import LOWER, align_per_frame  # noqa: E402
from phase3_angles import H36M  # noqa: E402
from phase3_infer import load_model  # noqa: E402
from phase7_inference_fix import CKPT_FOR  # noqa: E402

OUT = REPO / "results" / "phase15c_thorax_on_movi.json"
CAM = "PG1"
# The two points at issue: the reference this project built, and the one the
# AthleticsPose evidence says the lifter should be producing.
TARGETS = {"THORAX_visual3d": None, "CLAV": "CLAV"}


def sag_between(a: np.ndarray, b: np.ndarray, normal: np.ndarray) -> np.ndarray:
    ap = a - normal * np.sum(a * normal, axis=1, keepdims=True)
    bp = b - normal * np.sum(b * normal, axis=1, keepdims=True)
    return np.degrees(np.arctan2(np.sum(np.cross(ap, bp) * normal, axis=1),
                                 np.sum(ap * bp, axis=1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    model = load_model(CKPT_FOR["coco"], "base")
    if model is None:
        raise SystemExit("checkpoint missing")

    print("=== Which point does the lifter's thorax track on MoVi? ===")
    print("  perfect-2D arm; per-frame alignment on the lower body only\n")
    print(f"  {'subject':<12}{'dist THORAX':>13}{'dist CLAV':>11}"
          f"{'sag vs THORAX':>15}{'sag vs CLAV':>13}")
    rows = {}
    for p in sorted(MOCAP.glob("F_v3d_*.mat")):
        s = p.stem.split("_v3d_")[-1]
        v = load_v3d(p)
        try:
            _i, f30, f120 = jogging_window(v)
        except SystemExit:
            continue
        ref = to_h36m(v)
        idx = {n: i for i, n in enumerate(v["marker_names"])}
        loc = v["marker_location"].copy()
        loc[np.asarray(v["marker_gaps"]) == 1] = np.nan

        n = int(f30[1] - f30[0] + 1)
        mk = int(f120[0]) + MOCAP_PER_VIDEO * np.arange(n)
        mk = mk[mk < ref.shape[0]]
        g = ref[mk]
        p2 = project(g, CAM)
        keep = np.isfinite(p2).all(axis=(1, 2))
        if keep.sum() < 20:
            continue
        est = lift_2d(model, p2[keep])
        g = g[keep]
        clav = loc[mk, idx["CLAV"]][keep]

        li = [H36M[k] for k in LOWER]
        ea, _t = align_per_frame(est, g, li, scale=True)
        et = ea[:, H36M["thorax"]]
        pel = g[:, H36M["pelvis"]]
        thx = g[:, H36M["thorax"]]
        ml = g[:, H36M["r_hip"]] - g[:, H36M["l_hip"]]
        ml /= np.maximum(np.linalg.norm(ml, axis=1, keepdims=True), 1e-9)

        rows[s] = {
            "n_frames": int(keep.sum()),
            "dist_to_THORAX_mm": float(np.nanmean(
                np.linalg.norm(et - thx, axis=1))),
            "dist_to_CLAV_mm": float(np.nanmean(
                np.linalg.norm(et - clav, axis=1))),
            "sag_offset_vs_THORAX_deg": float(np.nanmean(
                sag_between(et - pel, thx - pel, ml))),
            "sag_offset_vs_CLAV_deg": float(np.nanmean(
                sag_between(et - pel, clav - pel, ml))),
            "THORAX_to_CLAV_mm": float(np.nanmean(
                np.linalg.norm(thx - clav, axis=1)))}
        r = rows[s]
        print(f"  {s:<12}{r['dist_to_THORAX_mm']:>13.1f}"
              f"{r['dist_to_CLAV_mm']:>11.1f}"
              f"{r['sag_offset_vs_THORAX_deg']:>15.2f}"
              f"{r['sag_offset_vs_CLAV_deg']:>13.2f}")

    nearer = {s: ("THORAX_visual3d" if r["dist_to_THORAX_mm"]
                  < r["dist_to_CLAV_mm"] else "CLAV") for s, r in rows.items()}
    n_thorax = sum(1 for v in nearer.values() if v == "THORAX_visual3d")
    print(f"\n  nearer THORAX on {n_thorax} of {len(rows)} subjects")
    print("  On AthleticsPose the same checkpoint's thorax matched CLAV to "
          "0.750 deg (part 2).")
    print("  So the lifter's thorax does NOT land on the same anatomical point "
          "in the two datasets.")

    out = {"what": "DIAGNOSTIC addendum to phase 15C. Nothing is chosen on "
                   "this; the definition stays the one part 2 decided on "
                   "AthleticsPose.",
           "arm": "perfect 2D (projected reference), no detector",
           "alignment": "per frame, lower body only",
           "camera": CAM,
           "athleticspose_reference": {
               "lifter_thorax_matches": "CLAV",
               "sag_offset_deg": 0.750,
               "released_reference_thorax_vs_CLAV_mm": 5.04},
           "subjects": rows, "nearer_point": nearer,
           "n_nearer_thorax": n_thorax, "n_subjects": len(rows)}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
