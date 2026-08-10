"""Phase 1B -- does kinematic data track the collection wave?

Phase 1A established that collection year predicts injury at AUC 0.720, because
the dataset concatenates studies with injury rates from 0.00 to 0.93. That makes
provenance a laundering channel: any feature that encodes *which study* a session
came from inherits label signal that has nothing to do with the runner.

This script asks four questions, in order of how much they matter:

  Q1  Does session STRUCTURE (marker count, landmarks, rate) track the wave?
  Q2  Can dv_r KINEMATICS recover the collection year?          <-- the crux
  Q3  Do dv_r kinematics beat the control / provenance on injury?
  Q4  Do they still beat it *within* a wave-balanced cohort?

If Q2 is negative, the provenance problem is confined to the metadata and a
kinematic result can be believed. If Q2 is positive, no kinematic score in this
project is interpretable without wave control.

Run: .venv/Scripts/python.exe scripts/phase1b_wave_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_cohort import (  # noqa: E402
    CONTROL_CATEGORICAL,
    CONTROL_NUMERIC,
    build_cohort,
)
from phase1_eval import (  # noqa: E402
    check_no_group_leakage,
    evaluate,
    fmt_delta,
    fmt_result,
    make_folds,
    paired_delta,
)

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
OUT = REPO / "results" / "phase1b_wave.json"

STRUCTURE_FEATURES = ["n_marker_channels", "n_joints_landmarks",
                      "n_neutral_markers", "n_frames", "hz_r", "size_mb"]


def wave_recoverability(X: pd.DataFrame, year: np.ndarray, groups: np.ndarray,
                        label: str) -> dict:
    """Can these features tell you which year the session was collected?

    Multiclass, grouped 5-fold. Compared against always-guessing the modal year.
    A model that beats that baseline substantially is encoding provenance.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=11)
    accs, base = [], []
    for tr, te in cv.split(X, year, groups):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(random_state=0)),
        ])
        pipe.fit(X.iloc[tr], year[tr])
        accs.append((pipe.predict(X.iloc[te]) == year[te]).mean())
        modal = pd.Series(year[tr]).mode()[0]
        base.append((year[te] == modal).mean())
    accs, base = np.asarray(accs), np.asarray(base)
    print(f"  {label:<34} year-accuracy {accs.mean():.3f} "
          f"(majority baseline {base.mean():.3f}, "
          f"lift {accs.mean() - base.mean():+.3f})")
    return {"features": label, "accuracy": float(accs.mean()),
            "baseline": float(base.mean()),
            "lift": float(accs.mean() - base.mean())}


def main() -> int:
    if not (DERIVED / "dvr_features.parquet").exists():
        raise SystemExit("run scripts/phase1b_extract.py first")

    cohort, _oa, notes = build_cohort()
    feats = pd.read_parquet(DERIVED / "dvr_features.parquet")
    struct = pd.read_parquet(DERIVED / "session_structure.parquet")

    df = cohort.merge(feats, on=["sub_id", "filename"], how="inner")
    df = df.merge(struct, on=["sub_id", "filename"], how="inner")
    df = df.reset_index(drop=True)
    print(f"joined: {len(df)} sessions of {len(cohort)} cohort "
          f"({df['sub_id'].nunique()} subjects)\n")

    dv_cols = [c for c in feats.columns if c not in ("sub_id", "filename")]

    # --- dv_r hygiene: is 0 a value or a sentinel? -------------------------
    print("=== dv_r zero-rate (phase 0 flagged 0 as possibly meaning missing) ===")
    zero_rate = (df[dv_cols] == 0).mean().sort_values(ascending=False)
    suspect = zero_rate[zero_rate > 0.20]
    print(f"  {len(dv_cols)} dv_r columns; {len(suspect)} are >20% exact-zero")
    for c, r in suspect.head(10).items():
        print(f"    {c:<44} {r:.1%}")
    # Treated as sentinel -> NaN, so the imputer handles them like any absence.
    df_clean = df.copy()
    for c in suspect.index:
        df_clean[c] = df_clean[c].replace(0.0, np.nan)
    print(f"  -> {len(suspect)} column(s) masked to NaN before modelling\n")

    y = df_clean["label"].to_numpy()
    groups = df_clean["sub_id"].to_numpy()
    year = df_clean["year"].to_numpy()
    folds = make_folds(y, groups)
    check_no_group_leakage(folds, groups)

    results, payload = {}, {}

    # --- Q1: does session structure track the wave? ------------------------
    print("=== Q1: session structure vs collection wave ===")
    print(pd.crosstab(df_clean["n_marker_channels"], df_clean["year"]).to_string())
    print()
    print("marker channels vs label:")
    print(pd.crosstab(df_clean["n_marker_channels"], df_clean["label"],
                      normalize="index").to_string())
    print()

    # --- Q2: can kinematics recover the year? THE CRUX ---------------------
    print("=== Q2: can these features recover the collection year? ===")
    rec = []
    rec.append(wave_recoverability(df_clean[STRUCTURE_FEATURES], year, groups,
                                   "session structure"))
    rec.append(wave_recoverability(df_clean[dv_cols], year, groups,
                                   "dv_r kinematics"))
    rec.append(wave_recoverability(df_clean[CONTROL_NUMERIC], year, groups,
                                   "demographics (numeric)"))
    payload["wave_recoverability"] = rec
    print()

    # --- Q3: injury classification -----------------------------------------
    print("=== Q3: injury classification on the joined cohort ===")
    for kind in ["logit", "hgb"]:
        for name, cols in [("provenance-only", ["year", "yrs_missing",
                                                "lvl_missing"]),
                           ("structure-only", STRUCTURE_FEATURES),
                           ("dv_r kinematics", dv_cols)]:
            r = evaluate(f"{name} ({kind})", kind, df_clean, cols, [], folds, groups)
            results[r["name"]] = r
            print("  " + fmt_result(r))
        r = evaluate(f"demographics ({kind})", kind, df_clean, CONTROL_NUMERIC,
                     CONTROL_CATEGORICAL, folds, groups)
        results[r["name"]] = r
        print("  " + fmt_result(r))
        r = evaluate(f"dv_r + demographics ({kind})", kind, df_clean,
                     dv_cols + CONTROL_NUMERIC, CONTROL_CATEGORICAL, folds, groups)
        results[r["name"]] = r
        print("  " + fmt_result(r))
    print()

    best = lambda pre: max((r for n, r in results.items() if n.startswith(pre)),  # noqa: E731
                           key=lambda r: r["auc_mean"])
    kin, ctrl, prov = best("dv_r kinematics"), best("demographics ("), best("provenance")
    print("=== Q3 paired deltas ===")
    deltas = [paired_delta(kin, ctrl), paired_delta(kin, prov),
              paired_delta(best("dv_r + demographics"), ctrl)]
    for d in deltas:
        print("  " + fmt_delta(d))
    payload["deltas"] = deltas
    print()

    # --- Q4: within a wave-balanced cohort ---------------------------------
    print("=== Q4: restricted to 2012-2016 (both classes present) ===")
    sub = df_clean[df_clean["year"].between(2012, 2016)].reset_index(drop=True)
    y4, g4 = sub["label"].to_numpy(), sub["sub_id"].to_numpy()
    folds4 = make_folds(y4, g4)
    check_no_group_leakage(folds4, g4)
    print(f"  n={len(sub)} sessions, {sub['sub_id'].nunique()} subjects, "
          f"positive rate {sub['label'].mean():.3f}")
    w = {}
    for kind in ["logit", "hgb"]:
        w[f"kin-{kind}"] = evaluate(f"dv_r ({kind}) [2012-16]", kind, sub, dv_cols,
                                    [], folds4, g4)
        w[f"ctrl-{kind}"] = evaluate(f"demographics ({kind}) [2012-16]", kind, sub,
                                     CONTROL_NUMERIC, CONTROL_CATEGORICAL,
                                     folds4, g4)
        w[f"prov-{kind}"] = evaluate(f"provenance ({kind}) [2012-16]", kind, sub,
                                     ["year", "yrs_missing", "lvl_missing"], [],
                                     folds4, g4)
    for r in w.values():
        results[r["name"]] = r
        print("  " + fmt_result(r))
    bk = max((r for k, r in w.items() if k.startswith("kin")), key=lambda r: r["auc_mean"])
    bc = max((r for k, r in w.items() if k.startswith("ctrl")), key=lambda r: r["auc_mean"])
    bp = max((r for k, r in w.items() if k.startswith("prov")), key=lambda r: r["auc_mean"])
    d4 = [paired_delta(bk, bc), paired_delta(bk, bp)]
    for d in d4:
        print("  " + fmt_delta(d))
    payload["deltas_2012_16"] = d4

    payload["results"] = {n: {"auc_mean": r["auc_mean"], "auc_ci": r["auc_ci"],
                              "ap_mean": r["ap_mean"], "n_features": r["n_features"],
                              "auc_per_fold": r["auc"].tolist()}
                          for n, r in results.items()}
    payload["n_joined"] = int(len(df))
    payload["n_dv_cols"] = len(dv_cols)
    payload["n_zero_masked"] = int(len(suspect))
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
