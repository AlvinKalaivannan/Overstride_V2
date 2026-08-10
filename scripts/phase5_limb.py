"""Phase 5 -- can kinematics identify WHICH limb is injured, within a session?

Every failure so far has been a between-subject comparison confounded by
collection wave. This changes the unit of comparison: injured limb vs uninjured
limb of the SAME person in the SAME trial. Same demographics, same speed, same
collection year, same marker set, same file -- the provenance channel that scored
0.863 in phase 1D cannot operate here, because both observations come from one
file.

Uninjured subjects are deliberately EXCLUDED. They have no injured side, so
including them would make the side-alignment itself encode the label. That is the
leakage trap flagged in the phase 1 spec, avoided by construction.

PRE-REGISTERED BAR (fixed before fitting): mean AUC >= 0.60 AND the cross-fold
CI excludes 0.5.

BUILT-IN FALSIFICATION: demographics, provenance and file structure are constant
within a session, so they CANNOT predict which limb is injured. They must score
about 0.5. If they do not, the setup leaks and every number here is void.

Run: .venv/Scripts/python.exe scripts/phase5_limb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_cohort import (  # noqa: E402
    CONTROL_CLEAN_CATEGORICAL,
    CONTROL_CLEAN_NUMERIC,
    STRUCTURE_FEATURES,
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
OUT = REPO / "results" / "phase5_limb.json"

BAR_AUC = 0.60          # pre-registered
PCA_FLAT = 30           # comparability with phases 1C/1D
PCA_PER_CHANNEL = 3     # matched capacity for the degradation ladder

JOINTS3 = ["ankle", "knee", "hip"]
JOINTS5 = ["ankle", "knee", "hip", "foot", "pelvis"]
SAGITTAL_PLANE = 2
STATS = ["mean", "sd", "q25", "q50", "q75"]


def mirror_signs(arrays: dict[str, np.ndarray], pos: dict[str, int]) -> dict:
    """Derive, per joint/plane, whether L and R share a sign convention.

    Flexion/extension is anatomically symmetric; ab/adduction and rotation are
    defined relative to the midline and therefore mirror. Derived from the data
    (population corr(L, R)) and then asserted against that expectation -- if a
    channel disagrees, stop rather than silently difference the wrong quantity.
    """
    mean = arrays["mean"]
    signs, table = {}, []
    for j in JOINTS5:
        for p in range(3):
            L = mean[:, pos[f"ang_L_{j}_p{p}"], :].mean(axis=1)
            R = mean[:, pos[f"ang_R_{j}_p{p}"], :].mean(axis=1)
            r = float(np.corrcoef(L, R)[0, 1])
            s = 1 if r >= 0 else -1
            signs[(j, p)] = s
            table.append((j, p, r, s))

    print("Mirror signs, derived from population corr(L, R):")
    print(f"  {'joint':<8}{'plane':>6}{'corr':>9}{'sign':>6}   D = R - s*L")
    for j, p, r, s in table:
        print(f"  {j:<8}{p:>6}{r:>9.3f}{s:>6}   {'R - L' if s == 1 else 'R + L'}")

    # The joints the feature sets actually use must match anatomy.
    for j in JOINTS3:
        for p in (0, 1):
            assert signs[(j, p)] == -1, (
                f"{j} plane {p} was expected to be MIRRORED between limbs "
                f"(corr < 0) but is not. Stop and inspect before differencing.")
        assert signs[(j, SAGITTAL_PLANE)] == 1, (
            f"{j} sagittal plane was expected to share a sign convention "
            f"(corr > 0) but does not. Stop and inspect.")
    print("  assertion OK: planes 0/1 mirrored, plane 2 shared, for ankle/knee/hip\n")
    return signs


def main() -> int:
    idx = pd.read_parquet(DERIVED / "waveforms_var_index.parquet")
    idx["sub_id"] = idx["sub_id"].astype(str)
    idx["filename"] = idx["filename"].astype(str)
    ch = (DERIVED / "waveform_channels.txt").read_text(encoding="utf-8").split("\n")
    pos = {c: i for i, c in enumerate(ch)}

    # waveforms_mean.npy covers all 1745; the rest cover the 1721 that survived
    # the MIN_STEPS filter. Align mean to the filtered index.
    full_idx = pd.read_parquet(DERIVED / "waveforms_index.parquet")
    full_idx["sub_id"] = full_idx["sub_id"].astype(str)
    full_idx["filename"] = full_idx["filename"].astype(str)
    order = {k: i for i, k in enumerate(full_idx["sub_id"] + "|" + full_idx["filename"])}
    sel = np.array([order[k] for k in (idx["sub_id"] + "|" + idx["filename"])])

    arrays = {"mean": np.load(DERIVED / "waveforms_mean.npy")[sel],
              "sd": np.load(DERIVED / "waveforms_sd.npy")}
    for q in ("q25", "q50", "q75"):
        arrays[q] = np.load(DERIVED / f"waveforms_{q}.npy")
    for k, a in arrays.items():
        assert a.shape[0] == len(idx), (k, a.shape, len(idx))

    signs = mirror_signs(arrays, pos)

    # --- sign-corrected limb differences ----------------------------------
    blocks, names, sets = [], [], {}
    for stat in STATS:
        arr = arrays[stat]
        made5, made3, made_sag = [], [], []
        for j in JOINTS5:
            for p in range(3):
                s = signs[(j, p)]
                d = (arr[:, pos[f"ang_R_{j}_p{p}"], :]
                     - s * arr[:, pos[f"ang_L_{j}_p{p}"], :])
                blocks.append(d)
                cols = [f"{stat}_d_{j}_p{p}_t{t:03d}" for t in range(arr.shape[2])]
                names.extend(cols)
                made5.extend(cols)
                if j in JOINTS3:
                    made3.extend(cols)
                    if p == SAGITTAL_PLANE:
                        made_sag.extend(cols)
        sets[f"limb15_{stat}"] = made5
        sets[f"limb9_{stat}"] = made3
        sets[f"limbsag_{stat}"] = made_sag

    wide = pd.DataFrame(np.concatenate(blocks, axis=1), columns=names)
    sets["limb9_all"] = sum((sets[f"limb9_{s}"] for s in STATS), [])
    sets["limb9_dist"] = sum((sets[f"limb9_{s}"] for s in ("q25", "q50", "q75")), [])

    # --- cohort: unilateral injuries only ---------------------------------
    cohort, _oa, _n = build_cohort()
    struct = pd.read_parquet(DERIVED / "session_structure.parquet")
    for f in (cohort, struct):
        f["sub_id"] = f["sub_id"].astype(str)
        f["filename"] = f["filename"].astype(str)
    df = (idx.join(wide)
          .merge(cohort, on=["sub_id", "filename"], how="inner")
          .merge(struct, on=["sub_id", "filename"], how="inner")
          .reset_index(drop=True))

    n_all = len(df)
    bilateral = int(df["InjSide"].isin(["Bilateral", "Bi-lateral"]).sum())
    df = df[(df["label"] == 1) & df["InjSide"].isin(["Right", "Left"])].reset_index(drop=True)
    df["label"] = (df["InjSide"] == "Right").astype(int)   # relabel: which side
    y, groups = df["label"].to_numpy(), df["sub_id"].to_numpy()
    print(f"cohort: {len(df)} unilateral sessions / {df['sub_id'].nunique()} subjects"
          f"  (from {n_all}; {bilateral} bilateral excluded)")
    print(f"  Right-injured rate {y.mean():.3f} (chance)\n")

    folds = make_folds(y, groups)
    check_no_group_leakage(folds, groups)

    results: dict = {}

    def run(name, num, cat, pca=None):
        best = None
        for kind in ["logit", "hgb"]:
            r = evaluate(f"{name} ({kind})", kind, df, num, cat, folds, groups,
                         pca_components=pca)
            best = r if best is None or r["auc_mean"] > best["auc_mean"] else best
        results[name] = best
        print("  " + fmt_result(best))
        return best

    # --- negative controls: these MUST land near 0.5 -----------------------
    print("=== negative controls (constant within session -> must be ~0.5) ===")
    nc = {
        "NEG demographics": run("NEG demographics", CONTROL_CLEAN_NUMERIC,
                                CONTROL_CLEAN_CATEGORICAL),
        "NEG provenance": run("NEG provenance", ["year", "yrs_missing",
                                                 "lvl_missing"], []),
        "NEG structure": run("NEG structure", STRUCTURE_FEATURES, []),
    }
    print("\n=== substantive baseline ===")
    run("DominantLeg", [], ["DominantLeg"])
    run("DominantLeg + demographics", CONTROL_CLEAN_NUMERIC,
        CONTROL_CLEAN_CATEGORICAL + ["DominantLeg"])

    worst = max(abs(r["auc_mean"] - 0.5) for r in nc.values())
    gate_ok = worst < 0.06
    print(f"\n  negative-control gate: max |AUC - 0.5| = {worst:.3f} -> "
          f"{'PASS' if gate_ok else 'FAIL -- setup leaks, numbers are void'}")

    # --- kinematics --------------------------------------------------------
    print("\n=== kinematic limb differences (flat PCA=30) ===")
    for name in ["limbsag_mean", "limb9_mean", "limb15_mean", "limb9_sd",
                 "limb9_dist", "limb9_all"]:
        run(name, sets[name], [], PCA_FLAT)

    # --- degradation ladder, capacity matched per channel ------------------
    print(f"\n=== degradation ladder (PCA = {PCA_PER_CHANNEL} components/channel) ===")
    ladder = {}
    for name, nch in (("limb15_mean", 15), ("limb9_mean", 9), ("limbsag_mean", 3)):
        r = run(f"{name} [matched]", sets[name], [], PCA_PER_CHANNEL * nch)
        ladder[name] = r

    print("\n=== paired deltas ===")
    deltas = []
    best_kin = max((results[k] for k in results if k.startswith("limb")),
                   key=lambda r: r["auc_mean"])
    for a, b in [(best_kin["name"], "DominantLeg + demographics"),
                 ("limb9_mean", "NEG demographics"),
                 ("limb9_all", "limb9_mean"),
                 ("limbsag_mean [matched]", "limb15_mean [matched]"),
                 ("limb9_mean [matched]", "limb15_mean [matched]")]:
        ra = results.get(a) or next(v for v in results.values() if v["name"].startswith(a))
        rb = results.get(b) or next(v for v in results.values() if v["name"].startswith(b))
        d = paired_delta(ra, rb)
        deltas.append(d)
        print("  " + fmt_delta(d))

    # --- verdict against the pre-registered bar ----------------------------
    lo, hi = best_kin["auc_ci"]
    clears = gate_ok and best_kin["auc_mean"] >= BAR_AUC and lo > 0.5
    print(f"\n=== VERDICT (bar fixed before fitting: AUC >= {BAR_AUC} and CI excludes 0.5) ===")
    print(f"  best kinematic: {best_kin['name']}  AUC {best_kin['auc_mean']:.3f} "
          f"[{lo:.3f}, {hi:.3f}]")
    if not gate_ok:
        print("  VOID -- negative-control gate failed.")
    elif clears:
        print("  CLEARS the bar. Kinematics encode injury within-subject;")
        print("  the between-subject failures are a confounding/normalisation problem.")
    elif lo > 0.5:
        print(f"  RELIABLE BUT SMALL -- CI excludes 0.5 but AUC < {BAR_AUC}.")
        print("  A signal exists; it is too weak to screen on.")
    else:
        print("  AT CHANCE. The kill criterion is confirmed under the most")
        print("  favourable design available: same person, same trial, all")
        print("  confounds removed, balanced classes, dominance at chance.")

    payload = {
        "n_sessions": int(len(df)), "n_subjects": int(df["sub_id"].nunique()),
        "chance": float(y.mean()), "bar_auc": BAR_AUC,
        "negative_control_gate": {"max_abs_dev": float(worst), "pass": bool(gate_ok)},
        "clears_bar": bool(clears),
        "mirror_signs": {f"{j}_p{p}": int(s) for (j, p), s in signs.items()},
        "results": {n: {"auc_mean": r["auc_mean"], "auc_ci": r["auc_ci"],
                        "ap_mean": r["ap_mean"], "n_features": r["n_features"],
                        "auc_per_fold": r["auc"].tolist()}
                    for n, r in results.items()},
        "deltas": deltas,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
