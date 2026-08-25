"""Is Ferber's walking data a usable second cohort? A feasibility check.

The validation plan (artifact a38ebc7d) proposes running the within-subject limb
task on Ferber's walking trials, describing it as "same 1,798 subjects, same
injury labels, a second gait mode -- the closest thing to a free replication".

This measures whether that premise holds. It does not: osteoarthritis
overwhelmingly walked rather than ran, so the walking cohort is a different
POPULATION as well as a different gait mode, and CLAUDE.md excludes OA from
primary analysis.

Reads only the two metadata CSVs. No session files, no waveforms.

Run: .venv/Scripts/python.exe scripts/walk_cohort_check.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(os.environ.get("DATA_ROOT", "data/ric"))
OA = r"osteoarthritis|knee oa|hip oa|^oa$"


def spec(d: pd.DataFrame) -> pd.Series:
    return d["SpecInjury"].astype(str).str.strip().str.lower()


def main() -> int:
    w = pd.read_csv(ROOT / "walk_data_meta.csv", low_memory=False)
    r = pd.read_csv(ROOT / "run_data_meta.csv", low_memory=False)

    print("=== cohort scale ===")
    for n, d in (("walk", w), ("run", r)):
        print(f"  {n:5} {len(d):>5} sessions / {d.sub_id.nunique():>5} subjects")
    both = set(w.sub_id.astype(str)) & set(r.sub_id.astype(str))
    print(f"  subjects in BOTH gait modes: {len(both)}")

    print("\n=== osteoarthritis: the premise-breaking difference ===")
    for n, d in (("walk", w), ("run", r)):
        m = spec(d).str.contains(OA, regex=True, na=False)
        print(f"  {n:5} {int(m.sum()):>5} OA sessions  ({100 * m.mean():4.1f}% of cohort)")
    for n, d in (("walk", w), ("run", r)):
        a = pd.to_numeric(d.get("age"), errors="coerce")
        a = a[(a >= 10) & (a <= 100)]
        print(f"  {n:5} median age {a.median():.0f}")

    print("\n=== usable A1 cohort, narrowed step by step ===")
    uni = w[w.InjSide.astype(str).isin(["Right", "Left"])]
    no_oa = uni[~spec(uni).str.contains(OA, regex=True, na=False)]
    paired = no_oa[no_oa.sub_id.astype(str).isin(both)]
    for label, d in (("all walking sessions", w),
                     ("unilateral (InjSide L/R)", uni),
                     ("  excluding OA (CLAUDE.md)", no_oa),
                     ("  and also in the run cohort", paired)):
        print(f"  {label:<30} {len(d):>5} / {d.sub_id.nunique():>4} subjects")
    print(f"\n  running analysis, for comparison:   818 /  675 subjects")
    print("  (walking shrinks further after the waveform + MIN_STEPS filter)")

    print("\n=== verdict ===")
    share = 100 * spec(uni).str.contains(OA, regex=True, na=False).mean()
    print(f"  OA is {share:.1f}% of unilateral walking sessions vs "
          f"{100 * spec(r[r.InjSide.astype(str).isin(['Right','Left'])]).str.contains(OA, regex=True, na=False).mean():.1f}% of running.")
    print("  => not a clean gait-mode comparison; decide the OA exclusion BEFORE running.")
    print(f"  => but {len(paired)} sessions / {paired.sub_id.nunique()} subjects survive, "
          "and restricting to")
    print("     subjects present in both modes gives a PAIRED design where gait mode")
    print("     is the only within-subject variable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
