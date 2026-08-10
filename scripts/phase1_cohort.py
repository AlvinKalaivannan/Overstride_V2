"""Phase 1 — shared cohort construction.

Both 1A (demographics control) and 1B (dv_r kinematics) import this. Building the
cohort twice is the easiest way to make the two models silently incomparable, so
it is built once, here.

Rules honoured (CLAUDE.md, docs/phase1-spec.md):
  - Labels come from phase 0's injury_status: a recorded diagnosis outranks a
    blank InjDefn. Not reimplemented.
  - Sentinels masked to NaN in all four affected columns before anything else.
  - Osteoarthritis excluded from the primary cohort, returned separately.
  - `unknown` status dropped -- it is absent information, not a class.
  - Collection year is carried ONLY for the provenance baseline and for
    diagnostics. It is never a feature of a demographics or kinematic model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_inventory import (  # noqa: E402
    PLAUSIBLE_RANGE,
    is_null_series,
    load_meta,
    to_num,
)

REPO = Path(__file__).resolve().parents[1]

# The demographics-only control, exactly as CLAUDE.md names it.
CONTROL_NUMERIC = ["age", "Height", "Weight", "speed_r", "YrsRunning"]
CONTROL_CATEGORICAL = ["Gender", "Level"]
CONTROL_FEATURES = CONTROL_NUMERIC + CONTROL_CATEGORICAL

# The decontaminated control. `YrsRunning` and `Level` are dropped because their
# MISSINGNESS carries the label -- a blank `Level` marks an uninjured session with
# 95% precision, and phase 1A found `Level_missing` to be the control's largest
# coefficient by ~10x (results/phase01.md). What remains is measured physiology.
# Defined once, here, so every phase scores against the same yardstick.
CONTROL_CLEAN_NUMERIC = ["age", "Height", "Weight", "speed_r"]
CONTROL_CLEAN_CATEGORICAL = ["Gender"]
CONTROL_CLEAN = CONTROL_CLEAN_NUMERIC + CONTROL_CLEAN_CATEGORICAL

# File metadata. Not a property of the runner -- but it classifies injury at
# AUC 0.757 because marker configuration tracks the collection wave. Any
# kinematic model has to beat CONTROL_CLEAN + these, not demographics alone.
STRUCTURE_FEATURES = ["n_marker_channels", "n_joints_landmarks",
                      "n_neutral_markers", "n_frames", "hz_r", "size_mb"]

# Provenance baseline: paperwork only, no physiology. Used to bound how much of
# any score is a data-collection artifact rather than a finding.
PROVENANCE_FEATURES = ["yrs_missing", "lvl_missing", "year"]


def data_root() -> Path:
    load_dotenv(REPO / ".env")
    if "DATA_ROOT" not in os.environ:
        raise SystemExit("DATA_ROOT is not set. Add it to .env; never hardcode it.")
    root = Path(os.environ["DATA_ROOT"])
    if not root.exists():
        raise SystemExit(f"DATA_ROOT does not exist: {root}")
    return root


def mask_sentinels(frame: pd.DataFrame) -> pd.DataFrame:
    """999 (and 255, and 1564, and 0) are missing-data markers, not measurements.

    Applied before imputation so the sentinel becomes NaN and is imputed like any
    other absence. Left in, `YrsRunning = 999` is 51 sessions of a runner with 999
    years of experience -- and no null check catches it.
    """
    out = frame.copy()
    for col, (lo, hi) in PLAUSIBLE_RANGE.items():
        if col not in out.columns:
            continue
        v = to_num(out[col])
        out[col] = v.where(v.between(lo, hi))
    return out


def build_cohort(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (cohort, osteoarthritis, provenance_notes).

    `cohort` is the primary analysis set: labelled, OA-excluded, sentinel-masked.
    """
    root = root or data_root()
    run, _walk, _union = load_meta(root)

    notes: dict[str, object] = {"n_run_sessions": len(run),
                                "n_run_subjects": int(run["sub_id"].nunique())}

    # Drop `unknown`: absence of information, not a third class.
    labelled = run[run["_status"].isin(["injured", "uninjured"])].copy()
    notes["n_unknown_sessions"] = int((run["_status"] == "unknown").sum())
    notes["n_unknown_subjects"] = int(
        run.loc[run["_status"] == "unknown", "sub_id"].nunique())

    # Osteoarthritis: excluded from primary, reported separately (CLAUDE.md).
    is_oa = labelled["_injury"] == "osteoarthritis"
    oa = labelled[is_oa].copy()
    cohort = labelled[~is_oa].copy()
    notes["n_oa_sessions"] = int(len(oa))
    notes["n_oa_subjects"] = int(oa["sub_id"].nunique())

    cohort["label"] = (cohort["_status"] == "injured").astype(int)
    oa["label"] = (oa["_status"] == "injured").astype(int)

    for frame in (cohort, oa):
        year = pd.to_datetime(frame["datestring"], errors="coerce",
                              format="mixed").dt.year
        frame["year"] = year
        yrs = to_num(frame["YrsRunning"])
        lo, hi = PLAUSIBLE_RANGE["YrsRunning"]
        # Computed BEFORE masking, so it captures sentinel-as-missing too.
        frame["yrs_missing"] = (yrs.isna() | ~yrs.between(lo, hi)).astype(int)
        frame["lvl_missing"] = is_null_series(frame["Level"]).astype(int)

    cohort = mask_sentinels(cohort)
    oa = mask_sentinels(oa)

    # Categoricals: normalise the four spellings of absent to one explicit token.
    for frame in (cohort, oa):
        for col in CONTROL_CATEGORICAL:
            frame[col] = frame[col].where(~is_null_series(frame[col]), "missing")

    notes["n_sessions"] = int(len(cohort))
    notes["n_subjects"] = int(cohort["sub_id"].nunique())
    notes["n_injured_sessions"] = int(cohort["label"].sum())
    notes["n_uninjured_sessions"] = int((cohort["label"] == 0).sum())
    notes["n_injured_subjects"] = int(
        cohort.loc[cohort["label"] == 1, "sub_id"].nunique())
    notes["n_uninjured_subjects"] = int(
        cohort.loc[cohort["label"] == 0, "sub_id"].nunique())
    notes["positive_rate"] = float(cohort["label"].mean())

    # Subjects whose label changes between their own sessions. Kept, per spec --
    # dropping them would delete genuine recovery signal and select on outcome.
    per_sub = cohort.groupby("sub_id")["label"].nunique()
    mixed = per_sub[per_sub > 1].index
    notes["n_mixed_label_subjects"] = int(len(mixed))
    notes["n_mixed_label_sessions"] = int(cohort["sub_id"].isin(mixed).sum())

    # Non-diagnoses among the injured. A limitation, not a filter.
    inj = cohort[cohort["label"] == 1]
    junk = inj["_injury"].isin(["pain", "other", "fill in specifics below", ""])
    notes["n_nondiagnosis_injured"] = int(junk.sum())
    notes["frac_nondiagnosis_injured"] = float(junk.mean())

    notes["n_multisession_subjects"] = int(
        (cohort.groupby("sub_id").size() > 1).sum())

    return cohort.reset_index(drop=True), oa.reset_index(drop=True), notes


def feature_availability(cohort: pd.DataFrame) -> pd.DataFrame:
    """Post-masking unusable rate per control feature. Reported, never used to filter."""
    rows = []
    for col in CONTROL_NUMERIC:
        bad = cohort[col].isna()
        rows.append({"feature": col, "unusable": int(bad.sum()),
                     "pct": float(bad.mean())})
    for col in CONTROL_CATEGORICAL:
        bad = cohort[col] == "missing"
        rows.append({"feature": col, "unusable": int(bad.sum()),
                     "pct": float(bad.mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cohort, oa, notes = build_cohort()
    print("=== cohort notes ===")
    for k, v in notes.items():
        print(f"  {k:<28} {v}")
    print("\n=== feature availability (after sentinel masking) ===")
    print(feature_availability(cohort).to_string(index=False))
    print(f"\nOA held out separately: {len(oa)} sessions, "
          f"{oa['sub_id'].nunique()} subjects, "
          f"injured rate {oa['label'].mean():.3f}")
