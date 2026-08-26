"""Phase 9 (A1) step 1 -- build the paired walking cohort and its session list.

PRE-ANALYSIS DECISIONS, recorded before any waveform exists:

  1. OSTEOARTHRITIS IS EXCLUDED. Decided by the operator on 2026-08-26, before
     this file was written and before any walking curve was generated. This
     inherits the CLAUDE.md rule (degenerative, mean age 56, confounded with the
     demographic control) rather than making a new one. It is not a discovery
     and it is not negotiable after the fact -- 419 of 2,088 walking sessions
     are OA against 14 of 1,832 running, so an OA decision taken AFTER seeing a
     result would be the single most effective way to manufacture one.

  2. PAIRED DESIGN. Restricted to subjects present in BOTH gait modes, so gait
     mode is the only thing that changes within a subject. Comparing the walking
     cohort to the running cohort instead would confound gait mode with
     population, which is exactly the error phase 1 spent itself learning to
     avoid.

  3. The labelling rule is IMPORTED, never reimplemented. `injury_status` and
     `clean_injury` come from phase 0 and `mask_sentinels` from phase 1, so
     "a recorded diagnosis outranks a blank severity" means the same thing here
     as it does for running. A copied labelling rule drifts; a shared one cannot.

Reads the two metadata CSVs and writes a MATLAB session list. No session files
are opened here and no waveform is generated.

Run: .venv/Scripts/python.exe scripts/phase9_walk_cohort.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_inventory import clean_injury, injury_status, to_num  # noqa: E402
from phase1_cohort import mask_sentinels  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
LIST_OUT = DERIVED / "walk_list.txt"
COHORT_OUT = DERIVED / "walk_cohort.parquet"
NOTES_OUT = REPO / "results" / "phase9_cohort.json"

# CLAUDE.md's control set, with the walking speed column substituted. Run speed
# is a confounder there and walk speed is the same confounder here; using
# `speed_r` on a walking cohort would silently control for the wrong trial.
CONTROL_NUMERIC_WALK = ["age", "Height", "Weight", "speed_w"]
CONTROL_CATEGORICAL = ["Gender", "Level"]

OA_LABEL = "osteoarthritis"


def build_walk_cohort(root: Path) -> tuple[pd.DataFrame, dict]:
    """The walking analogue of phase1_cohort.build_cohort.

    load_meta computes `_status` / `_injury` for the run and union frames only,
    so those columns are derived here -- with the same functions, not with a
    reimplementation of the same rule.
    """
    walk = pd.read_csv(root / "walk_data_meta.csv", dtype=str,
                       keep_default_na=False)
    run = pd.read_csv(root / "run_data_meta.csv", dtype=str,
                      keep_default_na=False)

    walk["_status"] = walk.apply(injury_status, axis=1)
    walk["_injury"] = walk["SpecInjury"].map(clean_injury)

    notes: dict[str, object] = {
        "oa_excluded": True,
        "oa_decision": "operator, 2026-08-26, before any walking curve existed",
        "n_walk_sessions": int(len(walk)),
        "n_walk_subjects": int(walk["sub_id"].nunique()),
        "n_run_sessions": int(len(run)),
        "n_run_subjects": int(run["sub_id"].nunique()),
    }

    labelled = walk[walk["_status"].isin(["injured", "uninjured"])].copy()
    notes["n_unknown_sessions"] = int((walk["_status"] == "unknown").sum())

    is_oa = labelled["_injury"] == OA_LABEL
    notes["n_oa_sessions_excluded"] = int(is_oa.sum())
    notes["n_oa_subjects_excluded"] = int(labelled.loc[is_oa, "sub_id"].nunique())
    cohort = labelled[~is_oa].copy()

    cohort["label"] = (cohort["_status"] == "injured").astype(int)

    year = pd.to_datetime(cohort["datestring"], errors="coerce",
                          format="mixed").dt.year
    cohort["year"] = year
    yrs = to_num(cohort["YrsRunning"])
    cohort["yrs_missing"] = (yrs.isna() | ~yrs.between(0, 80)).astype(int)
    cohort["lvl_missing"] = cohort["Level"].astype(str).str.strip().isin(
        ["", "nan", "NaN", "None"]).astype(int)

    cohort = mask_sentinels(cohort)
    # speed_w is not in PLAUSIBLE_RANGE (that table is keyed to the run column),
    # so it is coerced explicitly rather than left as a string.
    cohort["speed_w"] = to_num(cohort["speed_w"])
    for col in CONTROL_CATEGORICAL:
        s = cohort[col].astype(str).str.strip()
        cohort[col] = s.where(~s.isin(["", "nan", "NaN", "None"]), "missing")

    # --- the paired restriction ------------------------------------------
    both = set(walk["sub_id"].astype(str)) & set(run["sub_id"].astype(str))
    notes["n_subjects_both_modes"] = int(len(both))

    uni = cohort[(cohort["label"] == 1)
                 & cohort["InjSide"].astype(str).str.strip().isin(
                     ["Right", "Left"])].copy()
    notes["n_unilateral_injured"] = int(len(uni))

    paired = uni[uni["sub_id"].astype(str).isin(both)].copy()
    notes["n_paired_sessions"] = int(len(paired))
    notes["n_paired_subjects"] = int(paired["sub_id"].nunique())
    notes["right_injured_rate"] = float(
        (paired["InjSide"].astype(str).str.strip() == "Right").mean())

    return paired.reset_index(drop=True), notes


def main() -> int:
    load_dotenv()
    root = Path(os.environ.get("DATA_ROOT", "data/ric"))
    if not (root / "walk_data_meta.csv").exists():
        print(f"ERROR: no walk_data_meta.csv under {root}. "
              "Set DATA_ROOT in .env.")
        return 2

    paired, notes = build_walk_cohort(root)

    print("=== pre-analysis decisions ===")
    print("  OA excluded: YES (operator, before any walking curve existed)")
    print("  design: paired -- subjects present in BOTH gait modes only\n")

    print("=== cohort, narrowed step by step ===")
    print(f"  all walking sessions            {notes['n_walk_sessions']:>5}"
          f" / {notes['n_walk_subjects']:>4} subjects")
    print(f"  unknown status dropped          {notes['n_unknown_sessions']:>5}")
    print(f"  OA excluded                     {notes['n_oa_sessions_excluded']:>5}"
          f" sessions / {notes['n_oa_subjects_excluded']} subjects")
    print(f"  unilateral + injured            {notes['n_unilateral_injured']:>5}")
    print(f"  and present in the run cohort   {notes['n_paired_sessions']:>5}"
          f" / {notes['n_paired_subjects']:>4} subjects")
    print(f"\n  subjects in both gait modes:    {notes['n_subjects_both_modes']}")
    print(f"  right-injured rate (chance):    {notes['right_injured_rate']:.3f}")
    print(f"\n  for comparison, the running limb cohort was 818 / 675.")

    # --- the MATLAB session list ------------------------------------------
    paths = [str((root / "ric_data" / str(r.sub_id) / str(r.filename)).resolve())
             for r in paired.itertuples()]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print(f"\n  WARNING: {len(missing)} session files not found, e.g.:")
        for p in missing[:3]:
            print(f"    {p}")
        paths = [p for p in paths if Path(p).exists()]

    LIST_OUT.write_text("\n".join(paths) + "\n", encoding="utf-8")
    paired.to_parquet(COHORT_OUT, index=False)
    notes["n_list_paths"] = len(paths)
    notes["n_missing_files"] = len(missing)
    NOTES_OUT.write_text(json.dumps(notes, indent=2, default=str),
                         encoding="utf-8")

    print(f"\nwrote {LIST_OUT.relative_to(REPO)}  ({len(paths)} sessions)")
    print(f"wrote {COHORT_OUT.relative_to(REPO)}")
    print(f"wrote {NOTES_OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
