"""Phase 9 (A1) step 3 -- does the limb signal survive a change of gait mode?

Phase 5 found the project's only signal-carrying task: identify WHICH limb is
injured, within a session, from sagittal limb differences. It tops out at 0.610
on running. This asks whether that is a property of RUNNING or of GAIT.

THE DESIGN IS PAIRED, AND THAT IS THE POINT
-------------------------------------------
Comparing the walking cohort to the running cohort would confound gait mode with
population: osteoarthritis is 20.1% of walking sessions against 0.8% of running,
so the two cohorts are different people. Restricting to subjects present in BOTH
gait modes makes gait mode the only thing that changes within a subject -- the
phase 5 logic applied one level up. OA is excluded regardless (pre-registered in
phase9_walk_cohort.py, decided before any walking curve existed).

PRE-REGISTERED, BEFORE ANY WALKING AUC WAS COMPUTED
---------------------------------------------------
  3 hypothesis tests, in this order of interest:
      H1 (PRIMARY)  limbsag_mean [matched]  -- the exact config that gave 0.610
      H2            limb9_mean   [matched]
      H3            limb15_mean  [matched]
  Correction: HOLM-BONFERRONI across those 3.
  Bar, inherited from phase 5 unchanged: mean AUC >= 0.60 AND the cross-fold CI
  excludes 0.5.

  This is a new family of tests on a dataset that has already produced 60
  failures. Declaring the count in advance is what stops it becoming 61 attempts
  reported as one success.

FIVE NEGATIVE CONTROLS, NOT FOUR
--------------------------------
Provenance, demographics, structure and dominant leg are inherited from phase 5.
They are re-asserted on the walking subset rather than assumed -- a control that
cleared on one cohort has said nothing about another.

The fifth is specific to this cohort and the plan did not name it: SAMPLING RATE.
Walking mixes 120 Hz and 200 Hz where running was almost entirely 200 Hz (1,822
vs 10). Rate is constant within a session, so like demographics it CANNOT
indicate which limb is injured -- but if it scores above chance, the limb
differences are carrying an acquisition artefact and every kinematic number here
is void. Decimation is exactly where phase 4's framerate artefact came from.

STOP CONDITION: if any negative control leaves chance (|AUC - 0.5| >= 0.06,
phase 5's gate), report the confound and do NOT publish a kinematic AUC.

ONE SENSITIVITY ANALYSIS, DECLARED BEFORE RUNNING AND OUTSIDE THE HOLM FAMILY
----------------------------------------------------------------------------
22.8% of walking sessions carry `eventsflag_mean < 1`: the pipeline's automated
event detection partially fell back to foot-forward/foot-back. Running has no
equivalent -- it is a walking-specific data-quality caveat, visible in the batch
log and recorded per session in the manifest.

The PRIMARY test is therefore re-run on the `eventsflag_mean == 1.0` subset.

This is NOT a fourth hypothesis test and it does NOT enter the Holm correction.
It is the same hypothesis on cleaner data, and it is declared here before any
walking AUC exists precisely so it cannot later be mistaken for a search for a
subset where the result improves. Both numbers get reported whichever way they
fall: if the clean subset agrees, event-detection quality is not driving the
result; if it disagrees, that is a finding about the walking data and is
reported as one.

Run: .venv/Scripts/python.exe scripts/phase9_walk_limb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_cohort import STRUCTURE_FEATURES  # noqa: E402
from phase1_eval import (check_no_group_leakage, ci, evaluate,  # noqa: E402
                         fmt_result, make_folds, paired_delta)
from phase5_limb import JOINTS3, JOINTS5, SAGITTAL_PLANE  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
OUT = REPO / "results" / "phase9_walk_limb.json"

BAR_AUC = 0.60
PCA_PER_CHANNEL = 3
NC_GATE = 0.06

# Walking's control set. speed_w, not speed_r -- run speed is a confounder in
# the running analysis and walk speed is the same confounder here; using the
# running column would silently control for a trial these subjects did not do
# in this cohort.
CONTROL_NUMERIC = ["age", "Height", "Weight", "speed_w"]
CONTROL_CATEGORICAL = ["Gender", "Level"]

PREREGISTERED = ["limbsag_mean", "limb9_mean", "limb15_mean"]


def holm(pvals: list[float], names: list[str]) -> list[dict]:
    """Holm-Bonferroni, declared in advance. Returns rows in the input order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return [{"name": n, "p": p, "p_holm": a, "reject": a < 0.05}
            for n, p, a in zip(names, pvals, adj)]


def fold_p(aucs: np.ndarray) -> float:
    """One-sided p that mean AUC > 0.5, from the cross-fold spread.

    Deliberately crude and deliberately NOT a fold-level Wilcoxon: phase 5C
    established that the 25 folds are 5 seeds x 5 splits of ONE dataset and are
    not independent, so any test treating them as such reports significance the
    data cannot support. This is a t-statistic on the fold mean reported ONLY to
    drive the Holm ordering; the CI is the quantity to read.
    """
    from scipy import stats
    a = np.asarray(aucs, dtype=float)
    if a.std(ddof=1) == 0:
        return 0.0 if a.mean() > 0.5 else 1.0
    t = (a.mean() - 0.5) / (a.std(ddof=1) / np.sqrt(len(a)))
    return float(stats.t.sf(t, df=len(a) - 1))


def walk_mirror_signs(mean: np.ndarray, pos: dict) -> dict:
    """Derive L/R sign conventions FROM THE WALKING DATA, not inherited.

    Flexion/extension is anatomically symmetric; ab/adduction and rotation
    mirror about the midline. phase5_limb asserts this on running. Asserting it
    again here is not redundant -- if walking disagreed, differencing would be
    combining the wrong quantity and the assertion is what stops that silently.
    """
    signs, table = {}, []
    for j in JOINTS5:
        for p in range(3):
            L = mean[:, pos[f"ang_L_{j}_p{p}"], :].mean(axis=1)
            R = mean[:, pos[f"ang_R_{j}_p{p}"], :].mean(axis=1)
            r = float(np.corrcoef(L, R)[0, 1])
            signs[(j, p)] = 1 if r >= 0 else -1
            table.append((j, p, r, signs[(j, p)]))

    print("Mirror signs, derived from WALKING corr(L, R):")
    print(f"  {'joint':<8}{'plane':>6}{'corr':>9}{'sign':>6}")
    for j, p, r, s in table:
        print(f"  {j:<8}{p:>6}{r:>9.3f}{s:>6}")
    for j in JOINTS3:
        for p in (0, 1):
            assert signs[(j, p)] == -1, (
                f"walking: {j} plane {p} expected MIRRORED (corr < 0), got "
                f"{signs[(j, p)]}. Stop and inspect before differencing.")
        assert signs[(j, SAGITTAL_PLANE)] == 1, (
            f"walking: {j} sagittal expected SHARED (corr > 0). Stop.")
    print("  assertion OK: planes 0/1 mirrored, plane 2 shared "
          "(same anatomy as running)\n")
    return signs


def main() -> int:
    xp = DERIVED / "walking_mean.npy"
    if not xp.exists():
        print("ERROR: walking_mean.npy not found. Run phase9_assemble.py.")
        return 2

    X = np.load(xp)
    index = pd.read_parquet(DERIVED / "walking_index.parquet")
    index["sub_id"] = index["sub_id"].astype(str)
    index["filename"] = index["filename"].astype(str)
    ch = (DERIVED / "walking_channels.txt").read_text(encoding="utf-8").split("\n")
    pos = {c: i for i, c in enumerate(ch)}
    assert X.shape[1] == len(ch), (X.shape, len(ch))
    assert X.shape[0] == len(index), (X.shape, len(index))

    cohort = pd.read_parquet(DERIVED / "walk_cohort.parquet")
    cohort["sub_id"] = cohort["sub_id"].astype(str)
    cohort["filename"] = cohort["filename"].astype(str)

    signs = walk_mirror_signs(X, pos)

    # --- limb differences, same construction as phase 5 --------------------
    blocks, names, sets = [], [], {}
    made5, made3, madesag = [], [], []
    for j in JOINTS5:
        for p in range(3):
            s = signs[(j, p)]
            d = (X[:, pos[f"ang_R_{j}_p{p}"], :]
                 - s * X[:, pos[f"ang_L_{j}_p{p}"], :])
            blocks.append(d)
            cols = [f"mean_d_{j}_p{p}_t{t:03d}" for t in range(X.shape[2])]
            names.extend(cols)
            made5.extend(cols)
            if j in JOINTS3:
                made3.extend(cols)
                if p == SAGITTAL_PLANE:
                    madesag.extend(cols)
    sets["limb15_mean"], sets["limb9_mean"], sets["limbsag_mean"] = made5, made3, madesag
    wide = pd.DataFrame(np.concatenate(blocks, axis=1), columns=names)

    df = (index.join(wide)
          .merge(cohort, on=["sub_id", "filename"], how="inner")
          .reset_index(drop=True))

    struct = pd.read_parquet(DERIVED / "session_structure.parquet")
    struct["sub_id"] = struct["sub_id"].astype(str)
    struct["filename"] = struct["filename"].astype(str)
    before = len(df)
    df = df.merge(struct, on=["sub_id", "filename"], how="left")
    struct_cov = int(df[STRUCTURE_FEATURES[0]].notna().sum())
    print(f"structure features cover {struct_cov}/{before} walking sessions")

    df["label"] = (df["InjSide"].astype(str).str.strip() == "Right").astype(int)
    y, groups = df["label"].to_numpy(), df["sub_id"].to_numpy()
    print(f"\ncohort: {len(df)} walking sessions / {df['sub_id'].nunique()} subjects")
    print(f"  right-injured rate {y.mean():.3f} (chance)")
    print(f"  sampling rate: {df['hz'].value_counts().to_dict()}")
    print(f"  running comparison: 818 sessions / 675 subjects, chance 0.517\n")

    folds = make_folds(y, groups)
    check_no_group_leakage(folds, groups)

    # PCA cannot ask for more components than the smallest training fold has
    # samples. At this cohort's size the cap never binds -- it is here so a
    # small cohort fails loudly at the top rather than crashing an hour into
    # the run, and so that if it ever DOES bind the report says so.
    min_train = min(len(f["train"]) for f in folds)
    print(f"smallest training fold: {min_train} sessions")

    def cap(n_components: int, n_features: int) -> int:
        c = min(n_components, n_features, min_train - 1)
        if c != n_components:
            print(f"    NOTE: PCA capped {n_components} -> {c} "
                  f"(min train fold {min_train}, {n_features} features)")
        return c

    results: dict = {}

    def run(name, num, cat, pca=None):
        best = None
        for kind in ("logit", "hgb"):
            r = evaluate(f"{name} ({kind})", kind, df, num, cat, folds, groups,
                         pca_components=pca)
            best = r if best is None or r["auc_mean"] > best["auc_mean"] else best
        lo, hi = ci(best["auc"])
        results[name] = {"name": name, "auc_mean": float(best["auc_mean"]),
                         "ci_lo": lo, "ci_hi": hi,
                         "auc": [float(v) for v in best["auc"]]}
        print("  " + fmt_result(best))
        return results[name]

    # --- FIVE negative controls -------------------------------------------
    print("=== negative controls (constant within session -> must be ~0.5) ===")
    nc = {}
    nc["provenance"] = run("NEG provenance", ["year", "yrs_missing", "lvl_missing"], [])
    nc["demographics"] = run("NEG demographics", CONTROL_NUMERIC, CONTROL_CATEGORICAL)
    if struct_cov > 0.8 * before:
        nc["structure"] = run("NEG structure", STRUCTURE_FEATURES, [])
    else:
        print(f"  NEG structure SKIPPED -- only {struct_cov}/{before} covered")
    nc["dominant_leg"] = run("NEG DominantLeg", [], ["DominantLeg"])
    nc["sampling_rate"] = run("NEG sampling rate (walk-specific)", ["hz"], [])

    worst_name, worst = max(((k, abs(v["auc_mean"] - 0.5)) for k, v in nc.items()),
                            key=lambda t: t[1])
    gate_ok = worst < NC_GATE
    print(f"\n  gate: max |AUC - 0.5| = {worst:.3f} ({worst_name}) -> "
          f"{'PASS' if gate_ok else 'FAIL'}")

    payload = {"n_sessions": int(len(df)),
               "n_subjects": int(df["sub_id"].nunique()),
               "chance": float(y.mean()), "bar_auc": BAR_AUC,
               "design": "paired: subjects in both gait modes, OA excluded",
               "oa_excluded": True,
               "hz_mix": {str(k): int(v) for k, v in df["hz"].value_counts().items()},
               "negative_controls": nc,
               "nc_gate_pass": bool(gate_ok), "nc_worst": worst,
               "nc_worst_name": worst_name}

    if not gate_ok:
        print("\n=== STOP CONDITION FIRED ===")
        print(f"  '{worst_name}' is {worst:.3f} from chance, above the {NC_GATE} gate.")
        print("  The walking subset carries a confound. Per the pre-registered")
        print("  stop condition, NO kinematic AUC is reported.")
        payload["stopped"] = True
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {OUT.relative_to(REPO)}")
        return 0

    # --- pre-registered hypothesis tests ----------------------------------
    print(f"\n=== pre-registered tests (PCA = {PCA_PER_CHANNEL}/channel) ===")
    nch = {"limbsag_mean": 3, "limb9_mean": 9, "limb15_mean": 15}
    tested = []
    for name in PREREGISTERED:
        r = run(f"{name} [matched]", sets[name], [],
                cap(PCA_PER_CHANNEL * nch[name], len(sets[name])))
        r["p"] = fold_p(np.array(r["auc"]))
        tested.append(r)

    corrected = holm([r["p"] for r in tested], PREREGISTERED)
    print("\n=== Holm-Bonferroni across the 3 pre-registered tests ===")
    print(f"  {'test':<18}{'AUC':>8}{'CI':>20}{'p':>10}{'p_holm':>10}  bar")
    for r, c in zip(tested, corrected):
        clears = r["auc_mean"] >= BAR_AUC and r["ci_lo"] > 0.5
        interval = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        print(f"  {c['name']:<18}{r['auc_mean']:>8.3f}{interval:>20}"
              f"{c['p']:>10.4f}{c['p_holm']:>10.4f}  "
              f"{'PASS' if clears else 'fail'}")

    primary = tested[0]
    payload["tests"] = tested
    payload["holm"] = corrected
    payload["primary_auc"] = primary["auc_mean"]
    payload["primary_clears_bar"] = bool(primary["auc_mean"] >= BAR_AUC
                                         and primary["ci_lo"] > 0.5)

    print("\n=== reading, per the pre-declared bands ===")
    a = primary["auc_mean"]
    clears = a >= BAR_AUC and primary["ci_lo"] > 0.5   # BOTH halves of the bar
    if a >= 0.65 and clears:
        verdict = ("ABOVE 0.65 -- confound alert, not success. Inspect residual "
                   "composition before treating this as a stronger signal.")
    elif clears:
        verdict = ("near 0.61 with controls at chance -- the ceiling is a "
                   "property of GAIT, not of running specifically.")
    elif primary["ci_lo"] <= 0.5:
        verdict = ("at chance -- the running signal is RUNNING-SPECIFIC rather "
                   "than a generic asymmetry artefact. More interesting than it looks.")
    else:
        verdict = ("above chance but below the 0.600 bar -- weaker in walking "
                   "than in running, and it does not clear pre-registration.")
    print(f"  primary (limbsag_mean) = {a:.3f}  ->  {verdict}")
    payload["verdict"] = verdict

    # --- sensitivity: clean event detection only (NOT in the Holm family) ---
    clean = df["eventsflag_mean"] >= 1.0
    print(f"\n=== sensitivity: eventsflag_mean == 1.0 "
          f"({int(clean.sum())}/{len(df)} sessions, "
          f"{100 * (~clean).mean():.1f}% excluded) ===")
    sub = df[clean].reset_index(drop=True)
    ys, gs = sub["label"].to_numpy(), sub["sub_id"].to_numpy()
    if len(np.unique(ys)) < 2 or sub["sub_id"].nunique() < 50:
        print("  SKIPPED -- clean subset too small to split safely")
        payload["sensitivity"] = {"skipped": True, "n": int(clean.sum())}
    else:
        sfolds = make_folds(ys, gs)
        check_no_group_leakage(sfolds, gs)
        name = PREREGISTERED[0]
        best = None
        for kind in ("logit", "hgb"):
            r = evaluate(f"{name} [clean] ({kind})", kind, sub, sets[name], [],
                         sfolds, gs, pca_components=PCA_PER_CHANNEL * nch[name])
            best = r if best is None or r["auc_mean"] > best["auc_mean"] else best
        lo, hi = ci(best["auc"])
        print("  " + fmt_result(best))
        print(f"  primary on all sessions: {a:.3f}  |  clean only: "
              f"{best['auc_mean']:.3f}  (delta {best['auc_mean'] - a:+.3f})")
        payload["sensitivity"] = {
            "n_sessions": int(clean.sum()),
            "n_subjects": int(sub["sub_id"].nunique()),
            "frac_excluded": float((~clean).mean()),
            "auc_mean": float(best["auc_mean"]), "ci_lo": lo, "ci_hi": hi,
            "delta_vs_primary": float(best["auc_mean"] - a)}

    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
