"""Phase 5B -- is 0.610 a limit of the SIGNAL or of the REPRESENTATION?

Phase 4B established that the video path is not the bottleneck: measured
monocular error costs ~0.03 AUC and halving it buys nothing. The ceiling is the
mocap signal itself. Before spending effort on new feature representations, this
asks three cheap questions that decide whether any headroom exists.

D1 (PRIMARY, PRE-REGISTERED)  severity dose-response.
    If the limb signal is physiological it MUST be stronger in runners whose
    injury affects them more. Direction is declared before fitting: more severe
    -> more asymmetry. A monotone gradient says 0.610 is real physiology worth
    pushing on. A flat result says it is not.

D2  the dv_r ceiling probe.
    39 bilateral CLINICAL metrics -- pronation, hip adduction, pelvic drop, step
    width, medial heel whip. Mostly frontal/transverse, so NOT recoverable from a
    side-on camera. This is a DIAGNOSTIC, NEVER A DELIVERABLE. It separates two
    very different conclusions: if these beat 0.610 the limitation is
    specifically the video-recoverable channels; if they do not, the signal
    genuinely is not there and no representation rescues it.

D3  would more data help? Learning curve in training subjects.

Run: .venv/Scripts/python.exe scripts/phase5b_ceiling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_cohort import (CONTROL_CLEAN_CATEGORICAL,  # noqa: E402
                           CONTROL_CLEAN_NUMERIC, PROVENANCE_FEATURES,
                           STRUCTURE_FEATURES)
from phase1_eval import (check_no_group_leakage, ci, evaluate,  # noqa: E402
                         fmt_result, make_folds, make_model, paired_delta)
from phase4_real_delta import load_limb_cohort  # noqa: E402
from phase5_limb import JOINTS3, SAGITTAL_PLANE, mirror_signs  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
OUT = REPO / "results" / "phase5b_ceiling.json"

PCA_PER_CHANNEL = 3

# Declared BEFORE fitting. Ordering is the standard "how much did it disrupt
# training" ladder; 'No injury' is excluded rather than ranked (see report).
SEVERITY_RANK = {
    "Continuing to train in pain": 1,
    "Training volume/intensity affected": 2,
    "2 workouts missed in a row": 3,
}
D1_DIRECTION = "positive"     # more severe -> more asymmetry


def derive_signs(L: np.ndarray, R: np.ndarray, names: list[str],
                 label: str) -> np.ndarray:
    """Sign convention per channel, from population corr(L, R).

    Same method as phase5_limb.mirror_signs, but that function hardcodes the
    `ang_*` channel names and asserts angle-specific anatomy, so it cannot be
    reused for velocities or for the clinical dv_r metrics. Deriving this wrong
    is exactly the bug that made `asym9` the worst family in phase 1D, so the
    signs are derived and printed rather than assumed.
    """
    signs = np.ones(len(names))
    print(f"  {label}: sign convention from corr(L, R)")
    flipped = []
    for i, nm in enumerate(names):
        a, b = L[:, i], R[:, i]
        m = np.isfinite(a) & np.isfinite(b)
        r = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 30 else 0.0
        signs[i] = 1.0 if r >= 0 else -1.0
        if r < 0:
            flipped.append(f"{nm} ({r:+.2f})")
    print(f"    {len(names) - len(flipped)} shared (D = R - L), "
          f"{len(flipped)} mirrored (D = R + L)")
    if flipped:
        print("    mirrored: " + ", ".join(flipped[:8])
              + (" ..." if len(flipped) > 8 else ""))
    return signs


def negative_controls(meta: pd.DataFrame, y: np.ndarray, folds: list[dict],
                      groups: np.ndarray, results: dict) -> float:
    """Re-asserted, not inherited. These are constant within a session and so
    CANNOT identify which limb is injured; a departure means the setup leaks."""
    print("=== negative controls (must be ~0.5) ===")
    base = meta.copy()
    base["label"] = y
    worst = 0.0
    for name, num, cat in (("NEG demographics", CONTROL_CLEAN_NUMERIC,
                            CONTROL_CLEAN_CATEGORICAL),
                           ("NEG provenance", PROVENANCE_FEATURES, []),
                           ("NEG structure", STRUCTURE_FEATURES, [])):
        r = evaluate(name, "logit", base, num, cat, folds, groups)
        results[name] = r
        worst = max(worst, abs(r["auc_mean"] - 0.5))
        print("  " + fmt_result(r))
    print(f"  gate: max |AUC-0.5| = {worst:.3f} -> "
          f"{'PASS' if worst < 0.08 else 'FAIL'}\n")
    assert worst < 0.08, "negative control is not at chance -- setup leaks"
    return worst


def main() -> int:
    mean, pos, signs, meta, y, groups = load_limb_cohort()
    folds = make_folds(y, groups)
    check_no_group_leakage(folds, groups)
    print(f"{len(folds)} folds, groups=sub_id\n")

    results: dict = {}
    out: dict = {"n_sessions": int(len(meta)),
                 "n_subjects": int(meta["sub_id"].nunique()),
                 "chance": float(y.mean())}
    out["neg_control_max_dev"] = negative_controls(meta, y, folds, groups,
                                                   results)

    _ = mirror_signs({"mean": mean}, pos)      # asserts angle anatomy

    # ---------- the phase 5 reference model, rebuilt here -------------------
    def sagittal_frame() -> pd.DataFrame:
        cols, blocks = [], []
        for j in JOINTS3:
            s = signs[(j, SAGITTAL_PLANE)]
            L = mean[:, pos[f"ang_L_{j}_p{SAGITTAL_PLANE}"], :]
            R = mean[:, pos[f"ang_R_{j}_p{SAGITTAL_PLANE}"], :]
            blocks.append(R - s * L)
            cols += [f"d_{j}_t{t:03d}" for t in range(101)]
        return pd.DataFrame(np.concatenate(blocks, axis=1), columns=cols)

    sag = sagittal_frame()
    ref_frame = sag.copy()
    ref_frame["label"] = y
    ref = evaluate("limbsag_mean (phase 5 reference)", "logit", ref_frame,
                   list(sag.columns), [], folds, groups,
                   pca_components=PCA_PER_CHANNEL * 3)
    results[ref["name"]] = ref
    print("=== reference ===")
    print("  " + fmt_result(ref) + "\n")

    # =======================================================================
    # D1 -- severity dose-response.  PRIMARY, direction pre-registered.
    # =======================================================================
    print("=== D1 (PRIMARY): severity dose-response ===")
    print(f"  direction declared before fitting: {D1_DIRECTION} "
          "(more severe -> more asymmetry)\n")

    defn = meta["InjDefn"].astype(str)
    rank = defn.map(SEVERITY_RANK)
    has = rank.notna().to_numpy()
    excluded = defn[~has].value_counts()
    print("  n per stratum:")
    for k, v in sorted(SEVERITY_RANK.items(), key=lambda kv: kv[1]):
        print(f"    {v}  {k:<38} {int((defn == k).sum())}")
    print("  excluded (not on the severity ladder):")
    for k, v in excluded.items():
        print(f"       {k if k else '(blank)':<38} {v}")

    # Asymmetry MAGNITUDE per session: RMS of the sign-corrected limb difference
    # over the 3 sagittal channels. A scalar, so this uses every session and is
    # far better powered than a per-stratum classifier.
    asym_mag = np.sqrt((sag.to_numpy() ** 2).mean(axis=1))
    r_all = rank.to_numpy(dtype=float)

    rho, p = stats.spearmanr(asym_mag[has], r_all[has])
    # Bootstrap by SUBJECT -- sessions cluster within subject.
    rng = np.random.default_rng(5150)
    subs = np.unique(groups[has])
    idx_by_sub = {s: np.flatnonzero((groups == s) & has) for s in subs}
    boot = []
    for _ in range(2000):
        pick = rng.choice(subs, len(subs), replace=True)
        sel = np.concatenate([idx_by_sub[s] for s in pick])
        boot.append(stats.spearmanr(asym_mag[sel], r_all[sel]).statistic)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  asymmetry magnitude vs severity rank (n={int(has.sum())}):")
    print(f"    Spearman rho = {rho:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}")
    print(f"    -> {'CONSISTENT with' if lo > 0 else 'NOT consistent with'} "
          "the pre-registered direction")

    # Confound check: severity may track speed/age/mass, which also move
    # asymmetry. Report the partial correlation, not just the raw one.
    cov = meta.loc[has, ["speed_r", "age", "Height", "Weight"]].astype(float)
    cov = cov.fillna(cov.median())
    A = np.column_stack([np.ones(len(cov)), cov.to_numpy()])

    def resid(v):
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]

    rho_p, p_p = stats.spearmanr(resid(asym_mag[has]), resid(r_all[has]))
    print(f"    partial (speed_r, age, Height, Weight removed): "
          f"rho = {rho_p:+.4f}  p={p_p:.3f}")
    for c in cov.columns:
        cr = stats.spearmanr(asym_mag[has], cov[c]).statistic
        print(f"      corr(asymmetry, {c:<8}) = {cr:+.3f}")

    strata = {}
    print("\n  within-stratum AUC (folds regenerated per stratum -- these are "
          "NOT paired across strata):")
    for name, r_ in sorted(SEVERITY_RANK.items(), key=lambda kv: kv[1]):
        m = (defn == name).to_numpy()
        ys, gs = y[m], groups[m]
        if len(np.unique(ys)) < 2 or m.sum() < 60:
            print(f"    {name:<38} n={m.sum()} -- too small, skipped")
            continue
        fs = make_folds(ys, gs)
        check_no_group_leakage(fs, gs)
        fr = sag[m].reset_index(drop=True)
        fr2 = fr.copy()
        fr2["label"] = ys
        rr = evaluate(f"limbsag | {name}", "logit", fr2, list(fr.columns), [],
                      fs, gs, pca_components=PCA_PER_CHANNEL * 3)
        strata[name] = {"rank": r_, "n": int(m.sum()),
                        "n_subjects": int(len(np.unique(gs))),
                        "auc_mean": rr["auc_mean"], "auc_ci": rr["auc_ci"]}
        print(f"    rank {r_}  {name:<38} n={m.sum():>4}  "
              f"AUC {rr['auc_mean']:.3f} [{rr['auc_ci'][0]:.3f}, "
              f"{rr['auc_ci'][1]:.3f}]")

    out["D1"] = {"direction_declared": D1_DIRECTION,
                 "n_on_ladder": int(has.sum()),
                 "excluded": {str(k): int(v) for k, v in excluded.items()},
                 "spearman_rho": float(rho), "spearman_ci": [float(lo), float(hi)],
                 "spearman_p": float(p),
                 "partial_rho": float(rho_p), "partial_p": float(p_p),
                 "consistent_with_direction": bool(lo > 0),
                 "strata": strata}

    # =======================================================================
    # D2 -- dv_r clinical ceiling probe.  DIAGNOSTIC, NEVER A DELIVERABLE.
    # =======================================================================
    print("\n=== D2: dv_r clinical ceiling probe ===")
    print("  NOT video-recoverable -- these are mostly frontal/transverse.")
    print("  This bounds ambition; it is not a candidate feature set.\n")

    dv = pd.read_parquet(DERIVED / "dvr_features.parquet")
    dv["sub_id"] = dv["sub_id"].astype(str)
    dv["filename"] = dv["filename"].astype(str)
    key = meta[["sub_id", "filename"]].merge(dv, on=["sub_id", "filename"],
                                             how="left")
    assert len(key) == len(meta), "dv_r merge changed row count"

    base_names = sorted({c[5:] for c in dv.columns if c.startswith("left_")}
                        & {c[6:] for c in dv.columns if c.startswith("right_")})
    Lm = key[[f"left_{b}" for b in base_names]].replace(0, np.nan).to_numpy(float)
    Rm = key[[f"right_{b}" for b in base_names]].replace(0, np.nan).to_numpy(float)
    both = (np.isfinite(Lm) & np.isfinite(Rm)).mean(axis=0)
    live = both > 0.5
    names = [b for b, k in zip(base_names, live) if k]
    print(f"  {len(names)} of {len(base_names)} metrics have both limbs "
          f"populated in >50% of sessions\n")

    sgn = derive_signs(Lm[:, live], Rm[:, live], names, "dv_r")
    D = Rm[:, live] - sgn * Lm[:, live]
    dvf = pd.DataFrame(D, columns=[f"dvr_d_{n}" for n in names])
    dvf["label"] = y
    r_dv = evaluate("dv_r limb difference (39 clinical)", "logit", dvf,
                    [c for c in dvf.columns if c != "label"], [], folds, groups)
    results[r_dv["name"]] = r_dv
    print("\n  " + fmt_result(r_dv))

    comb = pd.concat([sag, dvf.drop(columns="label")], axis=1)
    comb["label"] = y
    r_cb = evaluate("dv_r + limbsag combined", "logit", comb,
                    [c for c in comb.columns if c != "label"], [], folds, groups,
                    pca_components=PCA_PER_CHANNEL * 3 + len(names))
    results[r_cb["name"]] = r_cb
    print("  " + fmt_result(r_cb))

    d_dv = paired_delta(r_dv, ref)
    d_cb = paired_delta(r_cb, ref)
    print(f"\n  dv_r      vs limbsag: dAUC {d_dv['delta_mean']:+.3f} "
          f"[{d_dv['delta_ci'][0]:+.3f}, {d_dv['delta_ci'][1]:+.3f}]  "
          f"{'EXCLUDES ZERO' if d_dv['excludes_zero'] else 'includes zero'}")
    print(f"  combined  vs limbsag: dAUC {d_cb['delta_mean']:+.3f} "
          f"[{d_cb['delta_ci'][0]:+.3f}, {d_cb['delta_ci'][1]:+.3f}]  "
          f"{'EXCLUDES ZERO' if d_cb['excludes_zero'] else 'includes zero'}")
    out["D2"] = {"n_metrics": len(names), "metrics": names,
                 "dvr_auc": r_dv["auc_mean"], "dvr_ci": r_dv["auc_ci"],
                 "combined_auc": r_cb["auc_mean"], "combined_ci": r_cb["auc_ci"],
                 "delta_dvr_vs_sag": d_dv["delta_mean"],
                 "delta_dvr_ci": list(d_dv["delta_ci"]),
                 "delta_dvr_excludes_zero": d_dv["excludes_zero"],
                 "delta_combined_vs_sag": d_cb["delta_mean"],
                 "delta_combined_ci": list(d_cb["delta_ci"]),
                 "delta_combined_excludes_zero": d_cb["excludes_zero"]}

    # =======================================================================
    # D3 -- learning curve. Would more subjects help?
    # =======================================================================
    print("\n=== D3: learning curve (would more data help?) ===")
    X = sag
    curve = []
    rngc = np.random.default_rng(909)
    for frac in (0.25, 0.50, 0.75, 1.00):
        aucs = []
        for f in folds:
            tr, te = f["train"], f["test"]
            tr_subs = np.unique(groups[tr])
            k = max(2, int(round(frac * len(tr_subs))))
            keep_subs = rngc.choice(tr_subs, k, replace=False)
            sub_tr = tr[np.isin(groups[tr], keep_subs)]
            if len(np.unique(y[sub_tr])) < 2:
                continue
            model, ng = make_model("logit", list(X.columns), [], "explicit",
                                   PCA_PER_CHANNEL * 3)
            kw = {"clf__groups": groups[sub_tr]} if ng else {}
            model.fit(X.iloc[sub_tr], y[sub_tr], **kw)
            aucs.append(roc_auc_score(
                y[te], model.predict_proba(X.iloc[te])[:, 1]))
        a = np.array(aucs)
        lo_, hi_ = ci(a)
        curve.append({"frac": frac, "n_train_subjects": int(round(
            frac * len(np.unique(groups)) * 0.8)),
            "auc_mean": float(a.mean()), "ci": [lo_, hi_]})
        print(f"  {int(frac*100):>3}% of training subjects "
              f"(~{curve[-1]['n_train_subjects']:>3}): "
              f"AUC {a.mean():.3f} [{lo_:.3f}, {hi_:.3f}]")
    slope = curve[-1]["auc_mean"] - curve[0]["auc_mean"]
    # The total 25->100 gain is the wrong thing to threshold on: what matters is
    # whether the curve is still climbing at the right-hand end. A large total
    # gain that is all in the first doubling means the curve has saturated.
    last = curve[-1]["auc_mean"] - curve[-2]["auc_mean"]
    print(f"  total 25% -> 100% gain: {slope:+.3f}")
    print(f"  FINAL increment (75% -> 100%): {last:+.3f} -> "
          + ("still climbing; more subjects may help"
             if last > 0.01 else
             "SATURATED: more subjects would not fix this"))
    out["D3"] = {"curve": curve, "gain_25_to_100": float(slope),
                 "final_increment": float(last),
                 "saturated": bool(last <= 0.01)}

    out["results"] = {n: {"auc_mean": r["auc_mean"], "auc_ci": r["auc_ci"],
                          "auc_per_fold": r["auc"].tolist()}
                      for n, r in results.items()}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
