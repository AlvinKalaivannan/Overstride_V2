"""Phase 1A -- the demographics-only control, and the provenance-only baseline.

CLAUDE.md's kill criterion: build the control FIRST. Everything downstream is a
delta against it.

Phase 0 added a second requirement: metadata missingness tracks the data
collection wave, and the waves differ in injury mix. So the control must itself
be measured against a baseline fitted on nothing but paperwork. If the control
cannot clearly beat that, it is detecting which study a session came from and no
kinematic delta measured against it means anything.

Run: .venv/Scripts/python.exe scripts/phase1a_control.py
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
    CONTROL_NUMERIC,
    build_cohort,
    feature_availability,
)
from phase1_eval import (  # noqa: E402
    N_REPEATS,
    N_SPLITS,
    SEEDS,
    build_preprocessor,
    check_no_group_leakage,
    evaluate,
    fmt_delta,
    fmt_result,
    make_folds,
    paired_delta,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase1a_control.json"


def main() -> int:
    cohort, oa, notes = build_cohort()
    y = cohort["label"].to_numpy()
    groups = cohort["sub_id"].to_numpy()

    print(f"cohort: {notes['n_sessions']} sessions, {notes['n_subjects']} subjects, "
          f"positive rate {notes['positive_rate']:.3f}")
    print(f"CV: StratifiedGroupKFold(n_splits={N_SPLITS}) x {N_REPEATS} seeds "
          f"{SEEDS}, groups=sub_id\n")

    folds = make_folds(y, groups)
    check_no_group_leakage(folds, groups)
    print(f"{len(folds)} folds generated; no-leakage assertion passed.\n")

    results = {}

    # --- the provenance baseline: paperwork only, no physiology -------------
    # Decomposed, because "collection year" and "was the form filled in" are two
    # different leaks and the fix for each is different.
    prov_sets = {
        "provenance-only": ["year", "yrs_missing", "lvl_missing"],
        "year alone": ["year"],
        "missingness flags alone": ["yrs_missing", "lvl_missing"],
    }
    for label, cols in prov_sets.items():
        for kind in ["logit", "hgb"]:
            r = evaluate(f"{label} ({kind})", kind, cohort, cols, [], folds, groups)
            results[r["name"]] = r
            print(fmt_result(r))
    print()

    # --- the demographics control, both categorical-missing treatments ------
    for kind in ["logit", "hgb"]:
        for cm in ["explicit", "mode"]:
            r = evaluate(f"demographics ({kind}, {cm} missing)", kind, cohort,
                         CONTROL_NUMERIC, CONTROL_CATEGORICAL, folds, groups,
                         categorical_missing=cm)
            results[r["name"]] = r
            print(fmt_result(r))
    print()

    # --- ablation: does the control survive without the leaky columns? ------
    # YrsRunning (28.4% missing) and Level (11.4%) are the two whose absence
    # tracks the collection wave. If the control collapses without them, its
    # score was substantially provenance.
    clean_num = [c for c in CONTROL_NUMERIC if c != "YrsRunning"]
    clean_cat = [c for c in CONTROL_CATEGORICAL if c != "Level"]
    for kind in ["logit", "hgb"]:
        r = evaluate(f"demographics minus leaky cols ({kind})", kind, cohort,
                     clean_num, clean_cat, folds, groups)
        results[r["name"]] = r
        print(fmt_result(r))
    print()

    # --- speed alone, for context -------------------------------------------
    r = evaluate("speed_r alone (logit)", "logit", cohort, ["speed_r"], [],
                 folds, groups)
    results[r["name"]] = r
    print(fmt_result(r))
    print()

    # --- the comparison that decides whether phase 1 can proceed ------------
    print("=== paired deltas vs the provenance baseline ===")
    prov_best = max(
        (r for n, r in results.items() if n.startswith("provenance")),
        key=lambda r: r["auc_mean"])
    print(f"(strongest provenance baseline: {prov_best['name']} "
          f"AUC {prov_best['auc_mean']:.3f})")
    deltas = []
    for name, r in results.items():
        if name.startswith("provenance"):
            continue
        d = paired_delta(r, prov_best)
        deltas.append(d)
        print("  " + fmt_delta(d))

    ctrl_best = max(
        (r for n, r in results.items() if n.startswith("demographics (")),
        key=lambda r: r["auc_mean"])
    print(f"\n=== the control ===\n{fmt_result(ctrl_best)}")
    print("This is the number every kinematic result is a delta against.")
    d_ctrl = paired_delta(ctrl_best, prov_best)
    print("  vs provenance: " + fmt_delta(d_ctrl))

    # --- interpretability: if the control works, say why in one sentence ----
    print("\n=== control coefficients (single fit on all data, for reading only,")
    print("    never scored -- every scored fit above was fold-internal) ===")
    from sklearn.linear_model import LogisticRegression
    pre = build_preprocessor(CONTROL_NUMERIC, CONTROL_CATEGORICAL)
    Xt = pre.fit_transform(cohort[CONTROL_NUMERIC + CONTROL_CATEGORICAL])
    lr = LogisticRegression(max_iter=5000).fit(Xt, y)
    names = pre.get_feature_names_out()
    order = np.argsort(-np.abs(lr.coef_[0]))
    for i in order[:10]:
        print(f"    {names[i]:<28} {lr.coef_[0][i]:+.3f}")

    # --- sensitivity: drop the degenerate 2017 wave -------------------------
    print("\n=== sensitivity: excluding the 2017 wave (all-uninjured, blank forms) ===")
    sub = cohort[cohort["year"] != 2017].reset_index(drop=True)
    y2, g2 = sub["label"].to_numpy(), sub["sub_id"].to_numpy()
    folds2 = make_folds(y2, g2)
    check_no_group_leakage(folds2, g2)
    print(f"  n={len(sub)} sessions, positive rate {sub['label'].mean():.3f}")
    sens = {}
    for kind in ["logit", "hgb"]:
        sens[f"prov-{kind}"] = evaluate(f"provenance-only ({kind}) [no 2017]", kind,
                                        sub, ["year", "yrs_missing", "lvl_missing"],
                                        [], folds2, g2)
        sens[f"demo-{kind}"] = evaluate(f"demographics ({kind}) [no 2017]", kind,
                                        sub, CONTROL_NUMERIC, CONTROL_CATEGORICAL,
                                        folds2, g2)
    for r in sens.values():
        results[r["name"]] = r
        print("  " + fmt_result(r))
    # Compare like with like: strongest provenance vs strongest demographics.
    best_prov2 = max((r for k, r in sens.items() if k.startswith("prov")),
                     key=lambda r: r["auc_mean"])
    best_demo2 = max((r for k, r in sens.items() if k.startswith("demo")),
                     key=lambda r: r["auc_mean"])
    print("  " + fmt_delta(paired_delta(best_demo2, best_prov2)))

    # --- is a wave-controlled cohort viable at all? -------------------------
    # If the label is confounded with collection wave, the way out is to compare
    # within waves. That is only possible for waves carrying both classes.
    print("\n=== wave-controlled cohort: years 2012-2016 (both classes present) ===")
    wave = cohort[cohort["year"].between(2012, 2016)].reset_index(drop=True)
    y3, g3 = wave["label"].to_numpy(), wave["sub_id"].to_numpy()
    folds3 = make_folds(y3, g3)
    check_no_group_leakage(folds3, g3)
    print(f"  n={len(wave)} sessions, {wave['sub_id'].nunique()} subjects, "
          f"positive rate {wave['label'].mean():.3f}")
    wave_res = {}
    for kind in ["logit", "hgb"]:
        wave_res[f"prov-{kind}"] = evaluate(
            f"provenance-only ({kind}) [2012-16]", kind, wave,
            ["year", "yrs_missing", "lvl_missing"], [], folds3, g3)
        wave_res[f"demo-{kind}"] = evaluate(
            f"demographics ({kind}) [2012-16]", kind, wave,
            CONTROL_NUMERIC, CONTROL_CATEGORICAL, folds3, g3)
    for r in wave_res.values():
        results[r["name"]] = r
        print("  " + fmt_result(r))
    bp3 = max((r for k, r in wave_res.items() if k.startswith("prov")),
              key=lambda r: r["auc_mean"])
    bd3 = max((r for k, r in wave_res.items() if k.startswith("demo")),
              key=lambda r: r["auc_mean"])
    d_wave = paired_delta(bd3, bp3)
    print("  " + fmt_delta(d_wave))
    print("  -> if this delta is positive and excludes zero, a wave-controlled")
    print("     analysis is viable and phase 1B has a valid benchmark.")

    print("\n=== osteoarthritis, held out separately (CLAUDE.md) ===")
    print(f"  {len(oa)} sessions, {oa['sub_id'].nunique()} subjects, "
          f"injured rate {oa['label'].mean():.3f}")
    print("  Not modelled: the held-out OA set has no uninjured sessions, so it")
    print("  has no classification task in it. Reported as a count only.")

    payload = {
        "notes": notes,
        "availability": feature_availability(cohort).to_dict("records"),
        "cv": {"n_splits": N_SPLITS, "seeds": SEEDS, "groups": "sub_id",
               "n_folds": len(folds)},
        "results": {n: {"auc_mean": r["auc_mean"], "auc_ci": r["auc_ci"],
                        "ap_mean": r["ap_mean"], "n_features": r["n_features"],
                        "auc_per_fold": r["auc"].tolist(),
                        "chosen_C": r["chosen_C"]}
                    for n, r in results.items()},
        "deltas_vs_provenance": deltas,
        "control_vs_provenance": d_ctrl,
        "best_control": ctrl_best["name"],
        "best_provenance": prov_best["name"],
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
