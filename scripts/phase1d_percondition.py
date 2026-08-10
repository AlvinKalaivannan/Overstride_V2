"""Phase 1D -- per-condition screening, with variability and asymmetry.

Phase 1C found no kinematic signal against a decontaminated control. The most
likely cause is the LABEL: "injured" pools ITBS (lateral hip/knee), Achilles
(ankle), plantar fasciitis (foot), calf strain and PFPS into one class, plus 437
sessions whose diagnosis is `pain` or `other`. Different pathologies at different
joints have different -- sometimes opposing -- kinematic signatures, so pooling
them dilutes any real effect toward zero.

This runs each condition against the uninjured pool separately. Sessions with a
DIFFERENT diagnosis are excluded, not used as negatives: a runner with another
injury is not a healthy control.

Cross-validation is unchanged from phase 1: StratifiedGroupKFold(5) x 5 seeds,
groups=sub_id, folds generated once per condition and reused by every feature set
so deltas stay paired, no-leakage asserted on every fold, all preprocessing
inside the fold.

Run: .venv/Scripts/python.exe scripts/phase1d_percondition.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_cohort import (  # noqa: E402
    CONTROL_CLEAN_CATEGORICAL,
    CONTROL_CLEAN_NUMERIC,
    STRUCTURE_FEATURES,
    build_cohort,
)
from phase1_eval import (  # noqa: E402
    check_no_group_leakage,
    ci,
    evaluate,
    fmt_result,
    make_folds,
    paired_delta,
)

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
OUT = REPO / "results" / "phase1d_percondition.json"

PCA_COMPONENTS = 30
SAGITTAL_PLANE = 2          # measured, see scripts/phase1c_verify_sagittal.py
JOINTS3 = ["ankle", "knee", "hip"]

CONDITIONS = ["patellofemoral pain syndrome", "itb syndrome",
              "achilles tendonitis", "plantar fasciitis", "calf muscle strain"]

# PRE-REGISTERED, written before any per-condition model was fitted. Each entry
# names the channels the literature says should carry the signal. Reported as
# hits/misses afterwards: an AUC built from unrelated channels is a weaker claim
# than one built from these.
HYPOTHESES = {
    "itb syndrome": ["avg_hip_p1", "avg_knee_p0"],
    "patellofemoral pain syndrome": ["avg_hip_p1", "avg_knee_p2"],
    "achilles tendonitis": ["avg_ankle_p2", "avg_ankle_p1"],
    "plantar fasciitis": ["avg_ankle_p2", "avg_ankle_p1"],
    "calf muscle strain": ["avg_ankle_p2"],
}


def load() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    mean = np.load(DERIVED / "waveforms_mean.npy")
    idx = pd.read_parquet(DERIVED / "waveforms_index.parquet")
    ch = (DERIVED / "waveform_channels.txt").read_text(encoding="utf-8").split("\n")
    sd = np.load(DERIVED / "waveforms_sd.npy")
    asym = np.load(DERIVED / "waveforms_asym.npy")
    vidx = pd.read_parquet(DERIVED / "waveforms_var_index.parquet")
    sd_ch = (DERIVED / "waveform_sd_channels.txt").read_text(encoding="utf-8").split("\n")
    as_ch = (DERIVED / "waveform_asym_channels.txt").read_text(encoding="utf-8").split("\n")

    for f in (idx, vidx):
        f["sub_id"] = f["sub_id"].astype(str)
        f["filename"] = f["filename"].astype(str)

    # SD/asym cover 1721 of 1745 sessions (low-stride sessions dropped). Restrict
    # everything to the intersection so every feature set is scored on the same
    # rows and the paired comparisons remain valid.
    keep_key = set(vidx["sub_id"] + "|" + vidx["filename"])
    mask = (idx["sub_id"] + "|" + idx["filename"]).isin(keep_key).to_numpy()
    mean, idx = mean[mask], idx[mask].reset_index(drop=True)
    order = {k: i for i, k in enumerate(vidx["sub_id"] + "|" + vidx["filename"])}
    sel = np.array([order[k] for k in (idx["sub_id"] + "|" + idx["filename"])])
    sd, asym = sd[sel], asym[sel]

    pos = {c: i for i, c in enumerate(ch)}
    blocks, names = [], []

    def add(arr: np.ndarray, channels: list[str]) -> list[str]:
        made = []
        for i, c in enumerate(channels):
            blocks.append(arr[:, i, :])
            n = [f"{c}_t{t:03d}" for t in range(arr.shape[2])]
            names.extend(n)
            made.extend(n)
        return made

    sets: dict[str, list[str]] = {}
    sets["sd54"] = add(sd, sd_ch)
    sets["asym9"] = add(asym, as_ch)

    avg_blocks, avg_names = [], []
    wave9, wave3 = [], []
    for joint in JOINTS3:
        for p in range(3):
            blk = (mean[:, pos[f"ang_L_{joint}_p{p}"], :]
                   + mean[:, pos[f"ang_R_{joint}_p{p}"], :]) / 2.0
            n = [f"avg_{joint}_p{p}_t{t:03d}" for t in range(mean.shape[2])]
            avg_blocks.append(blk)
            avg_names.extend(n)
            wave9.extend(n)
            if p == SAGITTAL_PLANE:
                wave3.extend(n)
    sets["wave9"], sets["wave3"] = wave9, wave3

    wide = pd.DataFrame(np.concatenate(blocks + avg_blocks, axis=1),
                        columns=names + avg_names)

    cohort, _oa, _n = build_cohort()
    struct = pd.read_parquet(DERIVED / "session_structure.parquet")
    dvr = pd.read_parquet(DERIVED / "dvr_features.parquet")
    for f in (cohort, struct, dvr):
        f["sub_id"] = f["sub_id"].astype(str)
        f["filename"] = f["filename"].astype(str)

    df = (idx.join(wide)
          .merge(cohort, on=["sub_id", "filename"], how="inner")
          .merge(struct, on=["sub_id", "filename"], how="inner")
          .merge(dvr, on=["sub_id", "filename"], how="inner")
          .reset_index(drop=True))
    dv_cols = [c for c in dvr.columns if c not in ("sub_id", "filename")]
    sets["dv_r live"] = [c for c in dv_cols if (df[c] == 0).mean() < 1.0]
    print(f"loaded {len(df)} sessions, {df['sub_id'].nunique()} subjects")
    return df, sets


def main() -> int:
    df, sets = load()
    uninjured = df[df["label"] == 0]
    print(f"uninjured pool: {len(uninjured)} sessions / "
          f"{uninjured['sub_id'].nunique()} subjects\n")
    print("PRE-REGISTERED hypotheses (fixed before fitting):")
    for k, v in HYPOTHESES.items():
        print(f"  {k:<32} {v}")

    specs = [
        ("CONTROL_CLEAN", CONTROL_CLEAN_NUMERIC, CONTROL_CLEAN_CATEGORICAL, None),
        ("CONTROL_CLEAN + structure",
         CONTROL_CLEAN_NUMERIC + STRUCTURE_FEATURES, CONTROL_CLEAN_CATEGORICAL, None),
        ("provenance-only", ["year", "yrs_missing", "lvl_missing"], [], None),
        ("dv_r live", sets["dv_r live"], [], None),
        ("wave9", sets["wave9"], [], PCA_COMPONENTS),
        ("wave3 (sagittal)", sets["wave3"], [], PCA_COMPONENTS),
        ("sd54 (variability)", sets["sd54"], [], PCA_COMPONENTS),
        ("asym9 (asymmetry)", sets["asym9"], [], PCA_COMPONENTS),
        ("wave9 + asym9", sets["wave9"] + sets["asym9"], [], PCA_COMPONENTS),
    ]
    KINEMATIC = {"dv_r live", "wave9", "wave3 (sagittal)", "sd54 (variability)",
                 "asym9 (asymmetry)", "wave9 + asym9"}

    payload: dict = {"conditions": {}, "hypotheses": HYPOTHESES,
                     "pca_components": PCA_COMPONENTS}
    all_tests = []

    for cond in CONDITIONS:
        pos = df[df["_injury"] == cond]
        sub = pd.concat([pos, uninjured]).reset_index(drop=True)
        y = sub["label"].to_numpy()
        groups = sub["sub_id"].to_numpy()
        folds = make_folds(y, groups)
        check_no_group_leakage(folds, groups)
        print(f"\n=== {cond} ===")
        print(f"  positives {len(pos)} sessions / {pos['sub_id'].nunique()} subj"
              f"   negatives {len(uninjured)} / {uninjured['sub_id'].nunique()}"
              f"   positive rate {y.mean():.3f}   {len(folds)} folds")

        local = {}
        for name, num, cat, pca in specs:
            best = None
            for kind in ["logit", "hgb"]:
                r = evaluate(f"{name} ({kind}) [{cond}]", kind, sub, num, cat,
                             folds, groups, pca_components=pca)
                best = r if best is None or r["auc_mean"] > best["auc_mean"] else best
            local[name] = best
            print("  " + fmt_result(best))

        cond_tests = []
        for name in specs:
            n = name[0]
            if n not in KINEMATIC:
                continue
            for base in ["CONTROL_CLEAN", "CONTROL_CLEAN + structure"]:
                d = paired_delta(local[n], local[base])
                diff = local[n]["auc"] - local[base]["auc"]
                try:
                    p = float(wilcoxon(diff).pvalue)
                except ValueError:          # all-zero differences
                    p = 1.0
                rec = {"condition": cond, "features": n, "baseline": base,
                       "delta": d["delta_mean"], "ci": d["delta_ci"],
                       "wins": d["wins"], "n": d["n"], "p_raw": p}
                cond_tests.append(rec)
                all_tests.append(rec)

        payload["conditions"][cond] = {
            "n_pos_sessions": int(len(pos)),
            "n_pos_subjects": int(pos["sub_id"].nunique()),
            "positive_rate": float(y.mean()),
            "results": {n: {"auc_mean": r["auc_mean"], "auc_ci": r["auc_ci"],
                            "ap_mean": r["ap_mean"],
                            "auc_per_fold": r["auc"].tolist()}
                        for n, r in local.items()},
            "tests": cond_tests,
        }
        print("  --- vs CONTROL_CLEAN ---")
        for t in cond_tests:
            if t["baseline"] == "CONTROL_CLEAN":
                lo, hi = t["ci"]
                print(f"    {t['features']:<22} dAUC {t['delta']:+.3f} "
                      f"[{lo:+.3f}, {hi:+.3f}]  p={t['p_raw']:.4f}  "
                      f"{t['wins']}/{t['n']} folds")

    # --- Benjamini-Hochberg across the whole pre-registered set -------------
    ps = np.array([t["p_raw"] for t in all_tests])
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, ps[i] * m / k)
        adj[i] = prev
    for t, a in zip(all_tests, adj):
        t["p_bh"] = float(a)
        t["significant_bh"] = bool(a < 0.05 and t["delta"] > 0)

    print(f"\n=== Benjamini-Hochberg over all {m} pre-registered tests ===")
    surv = [t for t in all_tests if t["significant_bh"]]
    print(f"  survive BH at 0.05 with a POSITIVE delta: {len(surv)}")
    for t in sorted(all_tests, key=lambda x: x["p_raw"])[:10]:
        flag = "**" if t["significant_bh"] else "  "
        print(f"  {flag} {t['condition'][:26]:<26} {t['features']:<22} "
              f"vs {t['baseline'][:18]:<18} dAUC {t['delta']:+.3f} "
              f"p={t['p_raw']:.4f} p_bh={t['p_bh']:.3f}")
    print("\n  NOTE: CV folds are not independent (subjects recur across seeds),")
    print("  so these p-values are anti-conservative. The CI-excludes-zero bar")
    print("  and the fold-win count remain the primary evidence.")

    payload["all_tests"] = all_tests
    payload["n_tests"] = m
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
