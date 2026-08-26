"""Phase 8 -- a POSITIVE control: what is the smallest limb asymmetry this
design could have found?

Every control in this project so far is negative. Provenance, demographics,
structure and dominant leg all sit at chance, which establishes that the
within-subject setup does not leak. Nothing establishes the converse: that the
method has the POWER to find an asymmetry that is really there.

Without that, two very different statements are not separated by the evidence:

    "we found nothing"            (the physiology is not in these channels)
    "this method finds nothing"   (the design is underpowered)

Five probes agreeing on a 0.610 ceiling is consistent with both. So: inject a
limb asymmetry of KNOWN magnitude into the real waveforms, on the real folds,
through the exact model that produced 0.610, and sweep the magnitude.

What that buys is better than a pass/fail. A bare null says nothing was found.
A null with a calibrated floor says *nothing was found, and here is the smallest
asymmetry this design could have found* -- and, by reading the sweep backwards,
*the 0.610 that was found is worth about this many degrees.*

TWO ARMS, and the first is the one that matters:

  SYNTHETIC   Each SUBJECT is assigned a random synthetic injured side, and the
              asymmetry is injected on that side. The real injury signal is now
              uncorrelated with the synthetic label, so it acts as realistic
              background noise. AUC(0) must land at 0.5 -- that is the sanity
              check that the injection, not a bug, is doing the work.

  ADDITIVE    Injected on the REAL injured side, on top of the real signal.
              AUC(0) is the real 0.610. Shows what an extra known asymmetry buys
              on top of whatever is genuinely there.

TWO SHAPES, because a floor is only defined relative to what is injected:

  offset      constant delta across all 101 stance points. The easy case, and
              therefore an optimistic bound on the floor.
  bump        Gaussian, peak at 40% of stance, sigma 12%. Localized, closer to
              how a real compensation presents, and harder for a 3-component
              PCA to see. The pessimistic bound.

Reported per shape in BOTH peak degrees and RMS degrees, since a bump with peak
delta carries roughly a third of the energy of an offset with the same delta and
comparing them on peak alone would flatter the offset.

NOT A DATASET FABRICATION. No rows are created, no values imputed. This is the
same controlled perturbation of real measured waveforms that phases 2 and 4 use
to build the degradation ladder, and every number it produces is labelled
synthetic and is a property of the DESIGN, never of the cohort.

Run: .venv/Scripts/python.exe scripts/phase8_positive_control.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_cohort import build_cohort  # noqa: E402
from phase1_eval import (check_no_group_leakage, ci, evaluate,  # noqa: E402
                         make_folds)
from phase5_limb import JOINTS3, SAGITTAL_PLANE, STATS, mirror_signs  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
OUT = REPO / "results" / "phase8_positive_control.json"

# The exact configuration that produced the headline 0.610 in phase 5:
# limbsag_mean, capacity-matched at 3 PCA components per channel, 3 channels.
PCA_PER_CHANNEL = 3
N_SAG_CHANNELS = 3
PCA_COMPONENTS = PCA_PER_CHANNEL * N_SAG_CHANNELS

# One family, fixed in advance. Phase 5 reported the better of logit/hgb per
# feature set; taking a max over families at every rung of a sweep would let
# model selection bend the floor, so this pins the family that produced 0.610.
KIND = "logit"

OBSERVED_AUC = 0.610          # phase 5, limbsag_mean [matched]
BAR_AUC = 0.60                # phase 5's pre-registered bar

# Declared before running. Sweep is in PEAK degrees of injected asymmetry.
DELTA_GRID = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.50, 4.00]


def stance_profile(shape: str, n: int = 101) -> np.ndarray:
    """Unit-peak injection profile over normalized stance."""
    if shape == "offset":
        return np.ones(n)
    if shape == "bump":
        t = np.linspace(0.0, 1.0, n)
        return np.exp(-0.5 * ((t - 0.40) / 0.12) ** 2)
    raise ValueError(shape)


def sag_difference(mean: np.ndarray, pos: dict, signs: dict,
                   side_is_right: np.ndarray | None = None,
                   delta: float = 0.0,
                   profile: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Sagittal limb-difference block, optionally after injecting an asymmetry.

    The injection is applied to the RAW limb waveform BEFORE differencing, which
    is the only place it can go: differencing first and adding afterwards would
    inject into a quantity the camera never observes, and would silently skip
    the sign convention.
    """
    arr = mean
    if delta and profile is not None:
        arr = mean.copy()
        for j in JOINTS3:
            ri = pos[f"ang_R_{j}_p{SAGITTAL_PLANE}"]
            li = pos[f"ang_L_{j}_p{SAGITTAL_PLANE}"]
            add = delta * profile[None, :]
            r_rows = np.flatnonzero(side_is_right)
            l_rows = np.flatnonzero(~side_is_right)
            arr[r_rows, ri, :] += add
            arr[l_rows, li, :] += add

    blocks, names = [], []
    for j in JOINTS3:
        s = signs[(j, SAGITTAL_PLANE)]
        d = (arr[:, pos[f"ang_R_{j}_p{SAGITTAL_PLANE}"], :]
             - s * arr[:, pos[f"ang_L_{j}_p{SAGITTAL_PLANE}"], :])
        blocks.append(d)
        names.extend([f"mean_d_{j}_p{SAGITTAL_PLANE}_t{t:03d}"
                      for t in range(arr.shape[2])])
    return np.concatenate(blocks, axis=1), names


def synthetic_sides(groups: np.ndarray, seed: int) -> np.ndarray:
    """One random injured side per SUBJECT, not per session.

    Assigning per session would let two sessions of the same person carry
    opposite synthetic sides, which the grouped CV would then be protecting
    against a label that no longer has a subject-level meaning.
    """
    rng = np.random.default_rng(seed)
    subs = np.unique(groups)
    pick = {s: bool(b) for s, b in zip(subs, rng.integers(0, 2, len(subs)))}
    return np.array([pick[g] for g in groups])


def crossing(deltas: list[float], aucs: list[float], target: float) -> float:
    """Smallest delta at which the sweep reaches `target`, linearly interpolated.

    Reported as nan rather than extrapolated if the sweep never gets there.
    """
    for i in range(1, len(deltas)):
        a, b = aucs[i - 1], aucs[i]
        if (a < target <= b) and b > a:
            f = (target - a) / (b - a)
            return float(deltas[i - 1] + f * (deltas[i] - deltas[i - 1]))
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3,
                    help="synthetic-side draws to average over")
    ap.add_argument("--shapes", type=str, default="offset,bump")
    args = ap.parse_args()
    shapes = args.shapes.split(",")

    # --- rebuild phase 5's cohort exactly ---------------------------------
    idx = pd.read_parquet(DERIVED / "waveforms_var_index.parquet")
    idx["sub_id"] = idx["sub_id"].astype(str)
    idx["filename"] = idx["filename"].astype(str)
    ch = (DERIVED / "waveform_channels.txt").read_text(encoding="utf-8").split("\n")
    pos = {c: i for i, c in enumerate(ch)}

    full_idx = pd.read_parquet(DERIVED / "waveforms_index.parquet")
    full_idx["sub_id"] = full_idx["sub_id"].astype(str)
    full_idx["filename"] = full_idx["filename"].astype(str)
    order = {k: i for i, k in enumerate(full_idx["sub_id"] + "|" + full_idx["filename"])}
    sel = np.array([order[k] for k in (idx["sub_id"] + "|" + idx["filename"])])

    mean_all = np.load(DERIVED / "waveforms_mean.npy")[sel]
    arrays = {"mean": mean_all, "sd": np.load(DERIVED / "waveforms_sd.npy")}
    for q in ("q25", "q50", "q75"):
        arrays[q] = np.load(DERIVED / f"waveforms_{q}.npy")
    signs = mirror_signs(arrays, pos)

    cohort, _oa, _n = build_cohort()
    struct = pd.read_parquet(DERIVED / "session_structure.parquet")
    for f in (cohort, struct):
        f["sub_id"] = f["sub_id"].astype(str)
        f["filename"] = f["filename"].astype(str)

    base = (idx.assign(_row=np.arange(len(idx)))
            .merge(cohort, on=["sub_id", "filename"], how="inner")
            .merge(struct, on=["sub_id", "filename"], how="inner")
            .reset_index(drop=True))
    base = base[(base["label"] == 1)
                & base["InjSide"].isin(["Right", "Left"])].reset_index(drop=True)

    rows = base["_row"].to_numpy()
    mean = mean_all[rows]
    real_right = (base["InjSide"] == "Right").to_numpy()
    groups = base["sub_id"].to_numpy()
    print(f"cohort: {len(base)} unilateral sessions / "
          f"{base['sub_id'].nunique()} subjects")
    print(f"  real right-injured rate {real_right.mean():.3f}\n")

    # --- the scale the sweep lives on -------------------------------------
    d0, names = sag_difference(mean, pos, signs)
    per_ch_sd = [float(np.nanstd(d0[:, i * 101:(i + 1) * 101].mean(axis=1)))
                 for i in range(N_SAG_CHANNELS)]
    print("between-subject SD of the sagittal limb difference (deg):")
    for j, s in zip(JOINTS3, per_ch_sd):
        print(f"  {j:<7}{s:6.2f}")
    print(f"  mean {np.mean(per_ch_sd):.2f} deg -- the noise the injection "
          f"competes against\n")

    results = []

    def score(tag: str, block: np.ndarray, y: np.ndarray, folds) -> dict:
        frame = pd.DataFrame(block, columns=names)
        frame["label"] = y
        r = evaluate(tag, KIND, frame, names, [], folds, groups,
                     pca_components=PCA_COMPONENTS)
        lo, hi = ci(r["auc"])
        return {"auc_mean": float(r["auc_mean"]), "ci_lo": lo, "ci_hi": hi,
                "auc": [float(v) for v in r["auc"]]}

    for shape in shapes:
        prof = stance_profile(shape)
        rms = float(np.sqrt(np.mean(prof ** 2)))
        print(f"=== shape '{shape}' -- RMS = {rms:.3f} x peak ===")

        # ---- ARM 1: synthetic side, background = real data ----------------
        print(f"  SYNTHETIC arm ({args.reps} draws averaged)")
        syn_curve = []
        for delta in DELTA_GRID:
            fold_stacks, means = [], []
            for rep in range(args.reps):
                side = synthetic_sides(groups, seed=1000 + rep)
                y = side.astype(int)
                folds = make_folds(y, groups)
                check_no_group_leakage(folds, groups)
                blk, _ = sag_difference(mean, pos, signs, side, delta, prof)
                r = score(f"{shape}/syn/{delta}/r{rep}", blk, y, folds)
                fold_stacks.append(r["auc"])
                means.append(r["auc_mean"])
            per_fold = np.mean(np.array(fold_stacks), axis=0)
            lo, hi = ci(per_fold)
            row = {"shape": shape, "arm": "synthetic", "delta_peak_deg": delta,
                   "delta_rms_deg": delta * rms, "auc_mean": float(np.mean(means)),
                   "ci_lo": lo, "ci_hi": hi, "reps": args.reps}
            results.append(row)
            syn_curve.append(row["auc_mean"])
            flag = "" if row["ci_lo"] <= 0.5 else "  <- CI excludes 0.5"
            print(f"    peak {delta:5.2f} deg (rms {delta*rms:5.2f})  "
                  f"AUC {row['auc_mean']:.3f}  [{lo:.3f}, {hi:.3f}]{flag}")

        # ---- ARM 2: real side, injection on top of real signal ------------
        print(f"  ADDITIVE arm (real injured side)")
        y_real = real_right.astype(int)
        folds_real = make_folds(y_real, groups)
        check_no_group_leakage(folds_real, groups)
        for delta in DELTA_GRID:
            blk, _ = sag_difference(mean, pos, signs, real_right, delta, prof)
            r = score(f"{shape}/add/{delta}", blk, y_real, folds_real)
            row = {"shape": shape, "arm": "additive", "delta_peak_deg": delta,
                   "delta_rms_deg": delta * rms, "auc_mean": r["auc_mean"],
                   "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "reps": 1}
            results.append(row)
            print(f"    peak {delta:5.2f} deg (rms {delta*rms:5.2f})  "
                  f"AUC {row['auc_mean']:.3f}  "
                  f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")

        # ---- the two numbers this phase exists to produce -----------------
        syn = [r for r in results if r["shape"] == shape and r["arm"] == "synthetic"]
        dd = [r["delta_peak_deg"] for r in syn]
        aa = [r["auc_mean"] for r in syn]
        d_bar = crossing(dd, aa, BAR_AUC)
        d_obs = crossing(dd, aa, OBSERVED_AUC)
        sig = next((r["delta_peak_deg"] for r in syn if r["ci_lo"] > 0.5), float("nan"))
        print(f"\n  DETECTION FLOOR ({shape})")
        print(f"    smallest swept delta whose CI excludes 0.5 : {sig:.2f} deg peak")
        print(f"    delta reaching the 0.600 bar               : {d_bar:.2f} deg peak")
        print(f"    delta reproducing the observed 0.610       : {d_obs:.2f} deg peak")
        print(f"    => the real signal is worth about {d_obs:.2f} deg of "
              f"'{shape}' asymmetry\n")
        results.append({"shape": shape, "summary": True,
                        "floor_ci_excludes_half_deg": sig,
                        "delta_at_bar_deg": d_bar,
                        "delta_at_observed_deg": d_obs,
                        "rms_per_peak": rms})

    payload = {"n_sessions": int(len(base)),
               "n_subjects": int(base["sub_id"].nunique()),
               "config": "limbsag_mean [matched]",
               "pca_components": PCA_COMPONENTS, "model": KIND,
               "observed_auc": OBSERVED_AUC, "bar_auc": BAR_AUC,
               "delta_grid_peak_deg": DELTA_GRID,
               "limb_difference_sd_deg": dict(zip(JOINTS3, per_ch_sd)),
               "rows": results}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
