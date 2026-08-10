"""Phase 1C Step A -- repair the benchmark.

Phase 1A's control (AUC 0.700) is inflated by a missingness leak and loses to a
provenance-only model. Phase 1B compared dv_r against that inflated number, which
is the wrong yardstick. This script fixes it:

  - CONTROL_CLEAN: demographics minus the two leak-carrying columns.
  - The paired delta dv_r vs CONTROL_CLEAN, which was never computed.
  - CONTROL_CLEAN + structure: the adversarial bar. File metadata alone scores
    0.757, so beating demographics is not sufficient evidence of physiology.

Also reports everything on the 28-channel stratum -- uniform marker set, uniform
sampling rate, one protocol -- which is the primary cohort for phase 1C Step C.

Run: .venv/Scripts/python.exe scripts/phase1c_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_cohort import (  # noqa: E402
    CONTROL_CATEGORICAL,
    CONTROL_CLEAN_CATEGORICAL,
    CONTROL_CLEAN_NUMERIC,
    CONTROL_NUMERIC,
    STRUCTURE_FEATURES,
    build_cohort,
)
from phase1_eval import (  # noqa: E402
    SEEDS,
    check_no_group_leakage,
    evaluate,
    fmt_delta,
    fmt_result,
    make_folds,
    paired_delta,
)

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
OUT = REPO / "results" / "phase1c_benchmark.json"


def load_joined() -> tuple[pd.DataFrame, list[str], list[str]]:
    cohort, _oa, _notes = build_cohort()
    feats = pd.read_parquet(DERIVED / "dvr_features.parquet")
    struct = pd.read_parquet(DERIVED / "session_structure.parquet")
    df = (cohort.merge(feats, on=["sub_id", "filename"], how="inner")
                .merge(struct, on=["sub_id", "filename"], how="inner")
                .reset_index(drop=True))

    dv_all = [c for c in feats.columns if c not in ("sub_id", "filename")]
    # "Live" = not 100% exact-zero. Label-independent and purely structural, so
    # selecting on it outside the fold cannot leak the outcome.
    zero_rate = (df[dv_all] == 0).mean()
    dv_live = [c for c in dv_all if zero_rate[c] < 1.0]
    return df, dv_all, dv_live


def run_suite(df: pd.DataFrame, dv_all: list[str], dv_live: list[str],
              tag: str, results: dict) -> dict:
    y = df["label"].to_numpy()
    groups = df["sub_id"].to_numpy()
    folds = make_folds(y, groups)
    check_no_group_leakage(folds, groups)
    print(f"\n=== {tag}: {len(df)} sessions, {df['sub_id'].nunique()} subjects, "
          f"positive rate {y.mean():.3f}, {len(folds)} folds ===")

    specs = [
        ("CONTROL_CLEAN", CONTROL_CLEAN_NUMERIC, CONTROL_CLEAN_CATEGORICAL),
        ("control as CLAUDE.md (leaky)", CONTROL_NUMERIC, CONTROL_CATEGORICAL),
        ("CONTROL_CLEAN + structure",
         CONTROL_CLEAN_NUMERIC + STRUCTURE_FEATURES, CONTROL_CLEAN_CATEGORICAL),
        ("provenance-only", ["year", "yrs_missing", "lvl_missing"], []),
        ("dv_r all cols", dv_all, []),
        ("dv_r live cols", dv_live, []),
        ("dv_r live + CONTROL_CLEAN",
         dv_live + CONTROL_CLEAN_NUMERIC, CONTROL_CLEAN_CATEGORICAL),
    ]
    local = {}
    for name, num, cat in specs:
        for kind in ["logit", "hgb"]:
            r = evaluate(f"{name} ({kind}) [{tag}]", kind, df, num, cat,
                         folds, groups)
            local[f"{name}|{kind}"] = r
            results[r["name"]] = r
            print("  " + fmt_result(r))

    def best(prefix: str) -> dict:
        cands = [r for k, r in local.items() if k.split("|")[0] == prefix]
        return max(cands, key=lambda r: r["auc_mean"])

    print(f"\n  --- paired deltas [{tag}] ---")
    deltas = []
    for a, b in [("dv_r live cols", "CONTROL_CLEAN"),
                 ("dv_r all cols", "CONTROL_CLEAN"),
                 ("dv_r live cols", "CONTROL_CLEAN + structure"),
                 ("dv_r live + CONTROL_CLEAN", "CONTROL_CLEAN"),
                 ("CONTROL_CLEAN", "provenance-only"),
                 ("control as CLAUDE.md (leaky)", "CONTROL_CLEAN")]:
        d = paired_delta(best(a), best(b))
        deltas.append(d)
        print("  " + fmt_delta(d))
    return {"tag": tag, "n": int(len(df)),
            "n_subjects": int(df["sub_id"].nunique()),
            "positive_rate": float(y.mean()), "deltas": deltas}


def main() -> int:
    df, dv_all, dv_live = load_joined()
    print(f"joined {len(df)} sessions; dv_r columns {len(dv_all)} "
          f"({len(dv_live)} live, {len(dv_all) - len(dv_live)} entirely zero)")
    print(f"CONTROL_CLEAN = {CONTROL_CLEAN_NUMERIC + CONTROL_CLEAN_CATEGORICAL}")
    print(f"seeds {SEEDS}, groups=sub_id")

    results: dict = {}
    payload = {"n_dv_all": len(dv_all), "n_dv_live": len(dv_live), "suites": []}

    payload["suites"].append(run_suite(df, dv_all, dv_live, "full", results))

    s28 = df[df["n_marker_channels"] == 28].reset_index(drop=True)
    payload["suites"].append(run_suite(s28, dv_all, dv_live, "28ch", results))

    payload["results"] = {n: {"auc_mean": r["auc_mean"], "auc_ci": r["auc_ci"],
                              "ap_mean": r["ap_mean"],
                              "n_features": r["n_features"],
                              "auc_per_fold": r["auc"].tolist()}
                          for n, r in results.items()}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
