"""The frozen measurement baseline, with the confidence interval its own gate asks for.

`CLAUDE.md` requires every accuracy claim in the measurement track to be a delta
against the frozen phase 7 pipeline, reported "with subject-level bootstrap CIs",
and the phase 13 gate asks for an interval that excludes zero. Neither was
possible: `results/phase7_inference_full.json` carried a mean and `n` per
configuration and nothing else -- no spread, no per-clip rows, no subject ids.

This script turns the per-clip rows that `phase7_inference_fix.py` now emits into
the baseline as it should be quoted:

  * per configuration, NEAR and FAR limb separately -- never pooled, per the
    measurement-track rule;
  * a subject-level mean and a percentile CI from resampling SUBJECTS, because
    8 cameras share one trial and clips are not independent;
  * a PAIRED delta against the deployable configuration, since every
    configuration is scored on the same clips -- the paired interval is the one
    a gate should be read against, not the overlap of two marginal intervals;
  * the pooled clip mean alongside, which is the estimator the published figures
    used, so the two can be compared rather than silently swapped.

It computes nothing new from the model: it is a re-aggregation of rows.

Run: .venv/Scripts/python.exe scripts/phase7_baseline_ci.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from phase12_scan import N_BOOT, SEED, subject_bootstrap_ci  # noqa: E402

ROWS = REPO / "results" / "phase7_inference_full_rows.json"
OUT = REPO / "results" / "phase7_baseline_ci.json"

# The configuration the tool actually runs (video_kinematics.py:241,244): the
# generic COCO checkpoint with edge padding and overlapped windows. Every delta
# is taken against this, not against the historical phase 3 row.
DEPLOYABLE = "generic COCO + fixes"
HISTORICAL = "baseline (phase 3)"
LIMBS = ("near", "far")
JOINTS = ("hip", "knee")


def per_subject(rows: list[dict], limb: str, joint: str | None) -> dict[str, float]:
    """Mean error per subject. joint=None means the hip/knee mean per clip.

    A clip contributes only if every value it needs is finite, so the near and
    far columns can rest on slightly different clip sets -- recorded rather than
    hidden, since a dropped clip is usually a degenerate far limb.
    """
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if joint is None:
            vals = [r.get(f"{limb}_{j}_mae") for j in JOINTS]
            if any(v is None for v in vals):
                continue
            acc[r["subject"]].append(float(np.mean(vals)))
        else:
            v = r.get(f"{limb}_{joint}_mae")
            if v is None:
                continue
            acc[r["subject"]].append(float(v))
    return {s: float(np.mean(v)) for s, v in sorted(acc.items())}


def pooled(rows: list[dict], limb: str, joint: str) -> tuple[float, int]:
    """The published estimator: the plain mean over clips, subjects ignored."""
    v = [r[f"{limb}_{joint}_mae"] for r in rows
         if r.get(f"{limb}_{joint}_mae") is not None]
    return (float(np.mean(v)) if v else float("nan")), len(v)


def summarise(rows: list[dict], limb: str, joint: str | None) -> dict:
    subj = per_subject(rows, limb, joint)
    vals = list(subj.values())
    lo, hi = subject_bootstrap_ci(vals)
    out = {"n_subjects": len(vals),
           "subject_mean": float(np.mean(vals)) if vals else float("nan"),
           "ci95": [lo, hi],
           "per_subject": subj}
    if joint is not None:
        out["pooled_clip_mean"], out["n_clips"] = pooled(rows, limb, joint)
    return out


def paired_delta(rows_a: list[dict], rows_b: list[dict], limb: str) -> dict:
    """b minus a, per subject, on the clips both configurations scored.

    Negative means b is better. Configurations see identical clips by
    construction, so this is a paired comparison and its interval is far tighter
    than the two marginals would suggest.
    """
    def key(r: dict) -> tuple:
        return (r["action"], r["subject"], r["clip"])

    a = {key(r): r for r in rows_a}
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows_b:
        ar = a.get(key(r))
        if ar is None:
            continue
        va = [ar.get(f"{limb}_{j}_mae") for j in JOINTS]
        vb = [r.get(f"{limb}_{j}_mae") for j in JOINTS]
        if any(v is None for v in va + vb):
            continue
        acc[r["subject"]].append(float(np.mean(vb)) - float(np.mean(va)))
    subj = {s: float(np.mean(v)) for s, v in sorted(acc.items())}
    vals = list(subj.values())
    lo, hi = subject_bootstrap_ci(vals)
    return {"n_subjects": len(vals),
            "subject_mean_delta": float(np.mean(vals)) if vals else float("nan"),
            "ci95": [lo, hi],
            "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi)
                                  and (lo > 0 or hi < 0)),
            "per_subject": subj}


def main() -> int:
    if not ROWS.exists():
        raise SystemExit(
            f"missing {ROWS.relative_to(REPO)} -- run\n"
            "  .venv/Scripts/python.exe scripts/phase7_inference_fix.py "
            "--clips 0 --out phase7_inference_full.json")
    data = json.loads(ROWS.read_text(encoding="utf-8"))
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in data["rows"]:
        by_config[r["config"]].append(r)

    configs = list(by_config)
    if DEPLOYABLE not in by_config:
        raise SystemExit(f"{DEPLOYABLE!r} not among {configs}")

    out = {"source": str(ROWS.relative_to(REPO)),
           "n_boot": N_BOOT, "seed": SEED,
           "deployable_config": DEPLOYABLE,
           "historical_config": HISTORICAL,
           "estimator_note": "subject_mean averages within subject first, so it "
                             "is not the published pooled clip mean; both are "
                             "reported",
           "configs": {}}

    print(f"=== baseline, {data['n_rows']} rows over {data['n_clips']} clips ===")
    print(f"  {'configuration':<24}{'limb':>6}{'hip':>8}{'knee':>8}"
          f"{'mean':>8}{'95% CI':>18}{'pooled':>9}")
    for name in configs:
        rows = by_config[name]
        entry: dict = {"limbs": {}}
        for limb in LIMBS:
            e = {"mean": summarise(rows, limb, None)}
            for joint in JOINTS:
                e[joint] = summarise(rows, limb, joint)
            entry["limbs"][limb] = e
            lo, hi = e["mean"]["ci95"]
            pooled_mean = float(np.mean([e[j]["pooled_clip_mean"] for j in JOINTS]))
            print(f"  {name:<24}{limb:>6}{e['hip']['subject_mean']:>8.2f}"
                  f"{e['knee']['subject_mean']:>8.2f}{e['mean']['subject_mean']:>8.2f}"
                  f"   [{lo:>5.2f}, {hi:>5.2f}]{pooled_mean:>9.2f}")
        if name != DEPLOYABLE:
            entry["paired_vs_deployable"] = {
                limb: paired_delta(by_config[DEPLOYABLE], rows, limb)
                for limb in LIMBS}
        out["configs"][name] = entry

    print(f"\n=== paired deltas vs the deployable baseline ({DEPLOYABLE}) ===")
    print(f"  negative = better than deployable. {N_BOOT} subject bootstraps.")
    for name in configs:
        if name == DEPLOYABLE:
            continue
        for limb in LIMBS:
            d = out["configs"][name]["paired_vs_deployable"][limb]
            lo, hi = d["ci95"]
            flag = "excludes 0" if d["excludes_zero"] else "spans 0"
            print(f"  {name:<24}{limb:>6}{d['subject_mean_delta']:>+8.2f}"
                  f"   [{lo:>+5.2f}, {hi:>+5.2f}]  {flag}")

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
