"""Phase 15B verdicts and Results section, applied MECHANICALLY.

The outcome rules were fixed in `results/phase15b.md` and committed alone in
b13b48f before anything ran. They are evaluated here in code, from the JSON,
rather than read off a table by eye -- so the verdict cannot be shaded by what
would be convenient.

Run: .venv/Scripts/python.exe scripts/phase15b_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
JSON = REPO / "results" / "phase15b.json"
MD = REPO / "results" / "phase15b.md"

RULE_SUBJECTS = ("Subject_1", "Subject_10", "Subject_12", "Subject_13")
H1_SUBJECTS = ("Subject_12", "Subject_13")
ARMS = ("detected", "perfect_original", "perfect_convention_matched")
REFS = ("original", "recipe")


def cell(d: dict, subj: str, arm: str, ref: str, limb: str, joint: str) -> dict:
    return d["subjects"][subj]["grid"][arm][ref][limb][joint]


def verdict_h1(d: dict) -> dict:
    """Rule: raises near-hip r to at least the detected arm's r on BOTH S12 and
    S13, AND moves the near-hip bias toward the detected arm's bias."""
    per = {}
    for s in H1_SUBJECTS:
        if s not in d["subjects"]:
            per[s] = {"available": False}
            continue
        det = cell(d, s, "detected", "original", "near", "hip")
        p0 = cell(d, s, "perfect_original", "original", "near", "hip")
        p1 = cell(d, s, "perfect_convention_matched", "original", "near", "hip")
        r_ok = bool(np.isfinite(p1["r"]) and np.isfinite(det["r"])
                    and p1["r"] >= det["r"])
        moved = bool(abs(p1["bias"] - det["bias"]) < abs(p0["bias"] - det["bias"]))
        per[s] = {"available": True,
                  "detected_r": det["r"], "perfect_r": p0["r"],
                  "matched_r": p1["r"], "r_reaches_detected": r_ok,
                  "detected_bias": det["bias"], "perfect_bias": p0["bias"],
                  "matched_bias": p1["bias"], "bias_moved_toward": moved,
                  "passes": bool(r_ok and moved)}
    avail = [v for v in per.values() if v.get("available")]
    supported = bool(avail) and all(v["passes"] for v in avail) and len(avail) == len(H1_SUBJECTS)
    return {"rule": "convention-matched perfect 2D raises near-hip r to >= the "
                    "detected arm's r on BOTH Subject_12 and Subject_13, AND "
                    "moves the near-hip bias toward the detected arm's bias",
            "per_subject": per,
            "supported": supported}


def verdict_h2(d: dict) -> dict:
    """Rule: the recipe-matched reference cuts perfect-2D hip |bias| by at least
    half on at least 3 of the 4 original diagnostic subjects."""
    per = {}
    for s in RULE_SUBJECTS:
        if s not in d["subjects"]:
            per[s] = {"available": False}
            continue
        a = cell(d, s, "perfect_original", "original", "near", "hip")
        b = cell(d, s, "perfect_original", "recipe", "near", "hip")
        halved = bool(np.isfinite(b["bias"]) and np.isfinite(a["bias"])
                      and abs(b["bias"]) <= 0.5 * abs(a["bias"]))
        per[s] = {"available": True,
                  "bias_original_ref": a["bias"], "bias_recipe_ref": b["bias"],
                  "abs_reduction_factor": (abs(a["bias"]) / abs(b["bias"])
                                           if b["bias"] else float("inf")),
                  "halved": halved}
    n_ok = sum(1 for v in per.values() if v.get("halved"))
    return {"rule": "recipe-matched reference cuts perfect-2D near-hip |bias| "
                    "by >= half on >= 3 of the 4 original diagnostic subjects",
            "per_subject": per, "n_halved": n_ok, "n_required": 3,
            "supported": bool(n_ok >= 3)}


def verdict_h3(d: dict) -> dict:
    """Rule: posture explains R^2 >= 0.25 of hip error variance in at least 3
    of the 4 original diagnostic subjects."""
    per = {}
    for s in RULE_SUBJECTS:
        if s not in d["subjects"]:
            per[s] = {"available": False}
            continue
        h3 = d["subjects"][s]["h3"]
        r2 = h3.get("r2", float("nan"))
        per[s] = {"available": True, "r2": r2, "n": h3.get("n"),
                  "terms": h3.get("terms"),
                  "meets": bool(np.isfinite(r2) and r2 >= 0.25)}
    n_ok = sum(1 for v in per.values() if v.get("meets"))
    return {"rule": "posture explains R^2 >= 0.25 of near-hip error variance "
                    "in >= 3 of the 4 original diagnostic subjects",
            "per_subject": per, "n_meeting": n_ok, "n_required": 3,
            "supported": bool(n_ok >= 3)}


def detector_cost(d: dict, arm: str, ref: str) -> dict:
    """detected - perfect, using the arm and reference the verdicts support."""
    out = {"perfect_arm_used": arm, "reference_used": ref, "per_subject": {}}
    for s in d["subjects"]:
        try:
            det = d["subjects"][s]["grid"]["detected"][ref]
            per = d["subjects"][s]["grid"][arm][ref]
        except KeyError:
            continue
        out["per_subject"][s] = {
            limb: {"detected": det[limb]["mean"], "perfect": per[limb]["mean"],
                   "cost": det[limb]["mean"] - per[limb]["mean"]}
            for limb in ("near", "far")}
    for limb in ("near", "far"):
        v = [x[limb]["cost"] for x in out["per_subject"].values()
             if np.isfinite(x[limb]["cost"])]
        out[f"{limb}_mean"] = float(np.mean(v)) if v else float("nan")
        out[f"{limb}_range"] = [float(np.min(v)), float(np.max(v))] if v else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=JSON)
    args = ap.parse_args()
    d = json.loads(args.json.read_text(encoding="utf-8"))

    h1, h2, h3 = verdict_h1(d), verdict_h2(d), verdict_h3(d)

    # Which perfect-2D arm does the evidence support, and which reference?
    arm = "perfect_convention_matched" if h1["supported"] else "perfect_original"
    ref = "recipe" if h2["supported"] else "original"
    cost = detector_cost(d, arm, ref)

    print("=== VERDICTS, against the rules fixed in b13b48f ===")
    for name, v in (("H1 input convention", h1), ("H2 reference convention", h2),
                    ("H3 within-subject posture", h3)):
        print(f"  {name:<28} "
              f"{'SUPPORTED' if v['supported'] else 'REJECTED'}")
    print(f"\n  detector cost recomputed with arm={arm}, reference={ref}")
    print(f"    near mean {cost['near_mean']:.3f}  range {cost['near_range']}")
    print(f"    far  mean {cost['far_mean']:.3f}  range {cost['far_range']}")

    out = dict(d)
    out["verdicts"] = {"h1": h1, "h2": h2, "h3": h3}
    out["detector_cost_recomputed"] = cost
    args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nverdicts written into {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
