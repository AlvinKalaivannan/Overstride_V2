"""Phase 1C Step C -- do the waveforms carry injury signal, or collection wave?

Phase 1B showed dv_r recovers collection year at +0.171 over baseline, and file
metadata classifies injury at 0.757. So a waveform model that beats the control
proves nothing on its own -- it has to be shown not to be riding the wave.

Three things are measured here, in order of importance:

  Q1  Can the WAVEFORMS recover collection year? (the laundering channel)
  Q2  Do they beat CONTROL_CLEAN and CONTROL_CLEAN+structure, inside the
      protocol-homogeneous 28-channel stratum?
  Q3  Do they generalise to an unseen collection wave (leave-one-wave-out)?

Feature sets, which are also the phase 2 degradation ladder:
  wave54 - every channel the pipeline emits
  wave9  - 3 joints x 3 planes, side-averaged  (the CLAUDE.md convention)
  wave3  - 3 joints, flexion/extension only    (recoverable from side-on video)

Run: .venv/Scripts/python.exe scripts/phase1c_waveforms.py
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
    make_folds_leave_one_wave_out,
    paired_delta,
)
from phase1b_wave_check import wave_recoverability  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
OUT = REPO / "results" / "phase1c_waveforms.json"

PCA_COMPONENTS = 30
JOINTS3 = ["ankle", "knee", "hip"]

# MEASURED by scripts/phase1c_verify_sagittal.py, not taken from the schema doc.
# docs/ferber-schema.md S3 gives the Cardan order as flex/ext -> ab/adduction ->
# rotation, implying index 0. The data disagrees: index 2 tracks the stored
# sagittal dv_r peaks at |r| >= 0.99, index 0 at |r| <= 0.17. Using plane 0 here
# would have made the whole degradation ladder measure the wrong channels.
SAGITTAL_PLANE = 2


def build_frame() -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    X = np.load(DERIVED / "waveforms_mean.npy")
    index = pd.read_parquet(DERIVED / "waveforms_index.parquet")
    channels = (DERIVED / "waveform_channels.txt").read_text(
        encoding="utf-8").split("\n")
    assert X.shape[1] == len(channels), (X.shape, len(channels))

    # Only sessions whose re-run reproduced the archive's own dv_r are usable.
    keep = index["resolved"] == 1
    print(f"waveforms: {X.shape}, {int(keep.sum())}/{len(index)} resolved")
    X, index = X[keep.to_numpy()], index[keep].reset_index(drop=True)

    cols, data = [], []
    for ci, ch in enumerate(channels):
        for t in range(X.shape[2]):
            cols.append(f"{ch}_t{t:03d}")
        data.append(X[:, ci, :])
    wide = pd.DataFrame(np.concatenate(data, axis=1), columns=cols)

    # Side-averaged sets. Averaging L/R is deliberate: it matches the CLAUDE.md
    # 9-channel convention and avoids encoding a side, which would interact with
    # InjSide (a phase 5 concern and a leakage risk here).
    # Side-averaged sets, built as whole blocks (assigning 909 columns one at a
    # time fragments the frame badly).
    pos = {c: i for i, c in enumerate(channels)}
    sets: dict[str, list[str]] = {"wave54": cols}
    extra, extra_names = [], []
    for name, planes in (("wave9", [0, 1, 2]), ("wave3", [SAGITTAL_PLANE])):
        made = []
        for joint in JOINTS3:
            for p in planes:
                block = (X[:, pos[f"ang_L_{joint}_p{p}"], :]
                         + X[:, pos[f"ang_R_{joint}_p{p}"], :]) / 2.0
                extra.append(block)
                names = [f"avg_{joint}_p{p}_t{t:03d}" for t in range(X.shape[2])]
                extra_names += names
                made += names
        sets[name] = made
    # wave3's columns are a subset of wave9's, so only add each block once.
    seen, keep_idx, keep_names = set(), [], []
    for i, n in enumerate(extra_names):
        if n not in seen:
            seen.add(n)
            keep_idx.append(i)
            keep_names.append(n)
    avg = pd.DataFrame(np.concatenate(extra, axis=1)[:, keep_idx],
                       columns=keep_names)
    wide = pd.concat([wide, avg], axis=1)

    cohort, _oa, _notes = build_cohort()
    struct = pd.read_parquet(DERIVED / "session_structure.parquet")
    dvr = pd.read_parquet(DERIVED / "dvr_features.parquet")
    # The manifest round-trips through CSV, so sub_id comes back as int64 while
    # the cohort keeps it as str. Normalise before joining or the merge fails.
    for frame in (index, cohort, struct, dvr):
        frame["sub_id"] = frame["sub_id"].astype(str)
        frame["filename"] = frame["filename"].astype(str)
    meta = (index.join(wide)
            .merge(cohort, on=["sub_id", "filename"], how="inner")
            .merge(struct, on=["sub_id", "filename"], how="inner")
            .merge(dvr, on=["sub_id", "filename"], how="inner")
            .reset_index(drop=True))
    dv_cols = [c for c in dvr.columns if c not in ("sub_id", "filename")]
    dv_live = [c for c in dv_cols if (meta[c] == 0).mean() < 1.0]
    print(f"joined {len(meta)} sessions, {meta['sub_id'].nunique()} subjects")
    return meta, sets, dv_live


def suite(df: pd.DataFrame, sets: dict[str, list[str]], dv_live: list[str],
          tag: str, folds: list[dict], results: dict) -> list[dict]:
    y = df["label"].to_numpy()
    groups = df["sub_id"].to_numpy()
    check_no_group_leakage(folds, groups)
    print(f"\n=== {tag}: {len(df)} sessions, {df['sub_id'].nunique()} subjects, "
          f"positive rate {y.mean():.3f}, {len(folds)} folds ===")

    specs = [
        ("CONTROL_CLEAN", CONTROL_CLEAN_NUMERIC, CONTROL_CLEAN_CATEGORICAL, None),
        ("CONTROL_CLEAN + structure",
         CONTROL_CLEAN_NUMERIC + STRUCTURE_FEATURES,
         CONTROL_CLEAN_CATEGORICAL, None),
        ("provenance-only", ["year", "yrs_missing", "lvl_missing"], [], None),
        ("dv_r live", dv_live, [], None),
        ("wave54", sets["wave54"], [], PCA_COMPONENTS),
        ("wave9", sets["wave9"], [], PCA_COMPONENTS),
        ("wave3 (sagittal)", sets["wave3"], [], PCA_COMPONENTS),
        ("wave9 + CONTROL_CLEAN", sets["wave9"] + CONTROL_CLEAN_NUMERIC,
         CONTROL_CLEAN_CATEGORICAL, PCA_COMPONENTS),
    ]
    local = {}
    for name, num, cat, pca in specs:
        for kind in ["logit", "hgb"]:
            r = evaluate(f"{name} ({kind}) [{tag}]", kind, df, num, cat,
                         folds, groups, pca_components=pca)
            local[name] = max(local.get(name, r), r, key=lambda x: x["auc_mean"])
            results[r["name"]] = r
            print("  " + fmt_result(r))

    print(f"\n  --- paired deltas [{tag}] ---")
    deltas = []
    for a, b in [("wave9", "CONTROL_CLEAN"),
                 ("wave9", "CONTROL_CLEAN + structure"),
                 ("wave54", "CONTROL_CLEAN"),
                 ("wave3 (sagittal)", "CONTROL_CLEAN"),
                 ("wave9", "dv_r live"),
                 ("wave9", "wave3 (sagittal)"),
                 ("wave9 + CONTROL_CLEAN", "CONTROL_CLEAN")]:
        d = paired_delta(local[a], local[b])
        deltas.append(d)
        print("  " + fmt_delta(d))
    return deltas


def main() -> int:
    df, sets, dv_live = build_frame()
    payload: dict = {"pca_components": PCA_COMPONENTS,
                     "n_channels": len(sets["wave54"]) // 101}
    results: dict = {}

    # --- Q1: do the waveforms encode collection wave? ----------------------
    print("\n=== Q1: can these features recover the collection year? ===")
    year = df["year"].to_numpy()
    groups = df["sub_id"].to_numpy()
    rec = [
        wave_recoverability(df[sets["wave9"]], year, groups, "waveforms (wave9)"),
        wave_recoverability(df[sets["wave3"]], year, groups, "waveforms (wave3)"),
        wave_recoverability(df[dv_live], year, groups, "dv_r live"),
        wave_recoverability(df[STRUCTURE_FEATURES], year, groups, "structure"),
    ]
    payload["wave_recoverability"] = rec

    # --- Q2: primary cohort, protocol-homogeneous --------------------------
    s28 = df[df["n_marker_channels"] == 28].reset_index(drop=True)
    payload["deltas_28ch"] = suite(
        s28, sets, dv_live, "28ch",
        make_folds(s28["label"].to_numpy(), s28["sub_id"].to_numpy()), results)

    payload["deltas_full"] = suite(
        df, sets, dv_live, "full",
        make_folds(df["label"].to_numpy(), df["sub_id"].to_numpy()), results)

    # --- Q3: generalisation to an unseen wave ------------------------------
    print("\n=== Q3: leave-one-wave-out (full cohort) ===")
    lowo = make_folds_leave_one_wave_out(df["label"].to_numpy(), groups, year)
    print(f"  {len(lowo)} usable waves")
    if lowo:
        payload["deltas_lowo"] = suite(df, sets, dv_live, "LOWO", lowo, results)

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
