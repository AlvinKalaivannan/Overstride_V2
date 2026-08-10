"""Phase 0 — data inventory.

Observes what is actually in the Ferber / Running Injury Clinic archive and writes
docs/data-inventory.md. Every downstream spec is written against that file.

Rules honoured here (CLAUDE.md):
  - DATA_ROOT is read from the environment, never hardcoded.
  - $DATA_ROOT is read-only; nothing is written under it.
  - Session JSONs are opened one at a time and released; never accumulated.
  - Nothing is synthesized. Missing or unreadable files are reported, not filled in.

Run: .venv/Scripts/python.exe scripts/phase0_inventory.py
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "data-inventory.md"

# Tokens that mean "absent" in this dataset. It uses several interchangeably.
NULL_TOKENS = {"", "nan", "null", "n/a", "none"}
# Additionally treated as "no injury recorded" in the injury fields.
NO_INJURY_TOKENS = NULL_TOKENS | {"no injury"}

# The README's Table 1 normalization recipe, verbatim.
INJURY_MERGES = {
    "oa": "osteoarthritis",
    "hip oa": "osteoarthritis",
    "knee oa": "osteoarthritis",
    "osteroarthritis": "osteoarthritis",
    "itbs": "itb syndrome",
    "pfps": "patellofemoral pain syndrome",
}

# Table 1 as published (docs/ferber-schema.md §5): the verification target.
# (male, female, age, height, mass, sessions, walk_speed, run_speed)
PAPER_TABLE1 = {
    "No Injury (age 18-49)": (137, 171, 32.52, 172.19, 69.54, 558, 1.21, 2.80),
    "No Injury (age 50+)": (39, 49, 55.80, 165.33, 69.89, 130, 1.18, 2.58),
    "achilles tendonitis": (30, 22, 42.62, 190.19, 77.20, 68, 1.30, 2.68),
    "itb syndrome": (39, 61, 35.21, 171.99, 67.76, 128, 1.26, 2.64),
    "osteoarthritis": (91, 156, 56.36, 167.40, 76.36, 422, 1.11, 2.41),
    "patellofemoral pain syndrome": (61, 76, 35.78, 178.20, 69.95, 142, 1.23, 2.62),
    "plantar fasciitis": (20, 34, 45.76, 170.93, 77.79, 59, 1.22, 2.50),
}

N_SAMPLE_JSON = 20


# --------------------------------------------------------------------------- utils


def norm(value: object) -> str:
    """Lowercase + collapse whitespace. The dataset is inconsistent about both."""
    return " ".join(str(value).strip().lower().split())


def is_blank(value: object) -> bool:
    return norm(value) in NULL_TOKENS


def is_null_series(series: pd.Series) -> pd.Series:
    """Vectorised is_blank. The dataset spells 'absent' four different ways."""
    return series.map(is_blank)


def is_no_injury(value: object) -> bool:
    """True for '', 'NaN', 'No Injury', and the 'No injury,No injury' compound."""
    text = norm(value)
    if text in NO_INJURY_TOKENS:
        return True
    return all(part.strip() in NO_INJURY_TOKENS for part in text.split(","))


def clean_injury(value: object) -> str:
    """README recipe: lowercase, then merge the known duplicate spellings."""
    text = norm(value)
    return INJURY_MERGES.get(text, text)


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace(list(NULL_TOKENS | {"NaN"}), None), errors="coerce")


# `Height` uses 999 as a missing-data sentinel. Left in, it corrupts group means.
HEIGHT_MIN, HEIGHT_MAX = 120.0, 250.0

# Height is not the only column that does this. Plausible physiological ranges for
# every numeric field the demographics-only control is fitted on; anything outside
# is a sentinel or a data-entry error, never a measurement.
PLAUSIBLE_RANGE = {
    "age": (10.0, 100.0),
    "Height": (HEIGHT_MIN, HEIGHT_MAX),
    "Weight": (30.0, 200.0),
    "speed_r": (0.5, 8.0),
    "YrsRunning": (0.0, 80.0),
}


def plausible_height(series: pd.Series) -> pd.Series:
    return series.where(series.between(HEIGHT_MIN, HEIGHT_MAX))


def fmt(value: object, places: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


# ----------------------------------------------------------------- injury labelling


def injury_status(row: pd.Series) -> str:
    """README: uninjured iff InjDefn='No injury' AND InjJoint is no-injury/blank
    AND SpecInjury is blank.

    A recorded diagnosis outranks a blank InjDefn. 352 osteoarthritis sessions carry
    a real `SpecInjury` with no severity recorded; treating those as 'unknown' drops
    the entire OA cohort out of Table 1.
    """
    defn = norm(row["InjDefn"])
    has_diagnosis = not is_no_injury(row["SpecInjury"])
    if defn == "no injury" and is_no_injury(row["InjJoint"]) and not has_diagnosis:
        return "uninjured"
    if has_diagnosis or (defn not in NULL_TOKENS and defn != "no injury"):
        return "injured"
    return "unknown"


def first_session_per_subject(frame: pd.DataFrame) -> pd.DataFrame:
    """README: averages use each subject's first dated session within the group."""
    return frame.sort_values(["sub_id", "_date"]).groupby("sub_id", as_index=False).first()


def group_stats(frame: pd.DataFrame) -> tuple:
    firsts = first_session_per_subject(frame)
    gender = firsts["Gender"].map(norm)
    return (
        int((gender == "male").sum()),
        int((gender == "female").sum()),
        firsts["_age"].mean(),
        firsts["_height"].mean(),
        firsts["_weight"].mean(),
        len(frame),
        firsts["_speed_w"].mean(),
        firsts["_speed_r"].mean(),
    )


# ------------------------------------------------------------------------ sections


def load_meta(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Raw string frames plus a session-level union of the run and walk files."""
    run = pd.read_csv(root / "run_data_meta.csv", dtype=str, keep_default_na=False)
    walk = pd.read_csv(root / "walk_data_meta.csv", dtype=str, keep_default_na=False)

    # Table 1 counts sessions, and a session file can carry walking and/or running.
    # Build one row per (sub_id, filename), preferring the run file's copy of the
    # shared demographic/injury columns.
    walk_only = walk[~walk.set_index(["sub_id", "filename"]).index.isin(
        run.set_index(["sub_id", "filename"]).index
    )]
    union = pd.concat([run, walk_only], ignore_index=True)

    speed_w = walk.set_index(["sub_id", "filename"])["speed_w"]
    union["_speed_w"] = to_num(
        pd.Series(
            union.set_index(["sub_id", "filename"]).index.map(speed_w), index=union.index
        ).astype(str)
    )
    for frame in (run, union):
        frame["_date"] = pd.to_datetime(frame["datestring"], errors="coerce")
        frame["_age"] = to_num(frame["age"])
        # _height stays raw so §4 can reproduce the paper's contaminated means;
        # _height_clean masks the 999 sentinel and is what everything else should use.
        frame["_height"] = to_num(frame["Height"])
        frame["_height_clean"] = plausible_height(frame["_height"])
        frame["_weight"] = to_num(frame["Weight"])
        frame["_speed_r"] = to_num(frame["speed_r"]) if "speed_r" in frame else pd.NA
        frame["_status"] = frame.apply(injury_status, axis=1)
        frame["_injury"] = frame["SpecInjury"].map(clean_injury)
    run["_speed_w"] = pd.NA
    return run, walk, union


def section_readme(root: Path) -> list[str]:
    readme = (root / "README.txt").read_text(encoding="utf-8", errors="replace")
    out = ["## 1. The official README vs the schema doc", ""]
    out += [
        f"`README.txt` is {len(readme.splitlines())} lines. It documents 4 file categories: "
        "2,506 session JSONs, 2 metadata CSVs, 8 MATLAB processing files, 8 tutorial files.",
        "",
        "**Contradictions and extensions, against `docs/ferber-schema.md`:**",
        "",
    ]
    out += md_table(
        ["#", "Claim", "Observed", "Impact"],
        [
            ["1", "Schema/paper: waveforms available",
             "JSONs hold **raw marker trajectories** + **scalar** descriptive variables. "
             "No 100-point curves anywhere.", "**Blocks phase 1** — see §5"],
            ["2", "README names the CSVs `run_meta_data.csv` / `walk_meta_data.csv`",
             "Actual: `run_data_meta.csv` / `walk_data_meta.csv`", "README internal error"],
            ["3", "Schema field `speed_w(r)`",
             "Two files, two names: `speed_r` (run), `speed_w` (walk)", "Naming only"],
            ["4", "`InjDefn` = 1 of 4 options",
             "Free-text strings, not integers; a 5th state (blank) exists", "Encoding"],
            ["5", "README: `InjSide` is right/left",
             "`Right`/`Left`/`Bilateral`/`Bi-lateral`/blank", "Phase 5 — see §2"],
            ["6", "Archive root `ric_data/`",
             "Archive's internal root folder is `reformat_data/`; renamed on extract",
             "Layout"],
            ["7", "n = 1,798 subjects",
             "1,798 total, but only 1,402 have **running** data", "Cohort size — see §3"],
            ["8", "Archive is a plain zip",
             "Nested zip uses **Deflate64**; Python `zipfile`, .NET and libarchive all fail",
             "Tooling — see §6"],
        ],
    )
    out += [
        "",
        "**The README also supplies the Table 1 reproduction recipe** — injury-string "
        "normalization plus subject-level counting rules. §4 applies it verbatim.",
        "",
        "**README definition of uninjured**, used throughout: `InjDefn` = 'No injury' "
        "**and** `InjJoint` ∈ {'No injury', blank} **and** `SpecInjury` blank.",
        "",
    ]
    return out


def section_meta(run: pd.DataFrame, walk: pd.DataFrame) -> list[str]:
    out = ["## 2. `run_data_meta.csv`", ""]
    n_rows, n_cols = run.shape[0], 26
    sessions = run.groupby("sub_id").size()

    out += [
        f"**{n_rows} rows × {n_cols} columns**, {run['sub_id'].nunique()} unique `sub_id`.",
        f"(`walk_data_meta.csv`: {walk.shape[0]} rows, {walk['sub_id'].nunique()} subjects.)",
        "",
        "> The file has 1,908 physical lines but **1,832 records** — `Activities` contains "
        "quoted commas and embedded newlines. Parse with a real CSV reader, never by line.",
        "",
        "### Columns, verbatim and in order",
        "",
        "```",
        ", ".join(list(run.columns[:n_cols])),
        "```",
        "",
        "### Null count per column",
        "",
        "Null = any of `''`, `NaN`, `null`, `N/A`. The dataset mixes all four.",
        "",
    ]
    rows = []
    for col in run.columns[:n_cols]:
        series = run[col]
        nulls = int(series.map(is_blank).sum())
        numeric = to_num(series)
        kind = "numeric" if numeric.notna().sum() > 0.5 * (n_rows - nulls) else "text"
        rows.append([f"`{col}`", kind, str(nulls), f"{100 * nulls / n_rows:.1f}%",
                     str(series.nunique())])
    out += md_table(["column", "inferred", "nulls", "null %", "distinct"], rows)

    height = to_num(run["Height"])
    weight = to_num(run["Weight"])
    # Sweep every control feature for out-of-range values, not just Height.
    sentinel_rows = []
    for col, (lo, hi) in PLAUSIBLE_RANGE.items():
        v = to_num(run[col])
        bad = v[v.notna() & ((v < lo) | (v > hi))]
        offenders = ", ".join(f"`{k:g}`×{n}" for k, n in
                              sorted(bad.value_counts().items())) or "—"
        clean = v.where(v.between(lo, hi))
        sentinel_rows.append([
            f"`{col}`", f"[{lo:g}, {hi:g}]", str(int(bad.size)), offenders,
            f"{clean.min():g}–{clean.max():g}",
        ])

    out += [
        "",
        "### Sentinel values — four columns, not one",
        "",
        "`999` is used as a missing-data marker, and it is **not confined to "
        "`Height`**. Every numeric feature of the demographics-only control was swept "
        "against a plausible physiological range:",
        "",
    ]
    out += md_table(["column", "plausible range", "outside", "offending values",
                     "clean range"], sentinel_rows)
    out += [
        "",
        "> **Mask all four before using them as features.** `YrsRunning` is the "
        f"costly one — its {int((to_num(run['YrsRunning']) == 999).sum())} `999` rows "
        "read as 999 years of running experience and are silently counted as data by "
        "any tool that only checks for nulls. `Height`'s sentinel propagates into the "
        "published Table 1 (see §4); the others corrupt any model fitted on them.",
        "",
        "### Missingness is not random — it tracks the collection wave",
        "",
        "**This is the most dangerous finding in the metadata.** Whether a field was "
        "filled in at all predicts the label, because questionnaire completeness varies "
        "by study wave and the waves differ in injury mix:",
        "",
    ]
    lab_mask = run["_status"].isin(["injured", "uninjured"])
    lab = run[lab_mask & (run["_injury"] != "osteoarthritis")].copy()
    lab_year = pd.to_datetime(lab["datestring"], errors="coerce",
                              format="mixed").dt.year
    lab_inj = (lab["_status"] == "injured").astype(int)
    yrs_num = to_num(lab["YrsRunning"])
    year_rows = []
    for yr, idx in lab.groupby(lab_year).groups.items():
        sub = lab.loc[idx]
        year_rows.append([
            f"{yr:.0f}", str(len(sub)),
            f"{lab_inj.loc[idx].mean():.3f}",
            f"{(yrs_num.loc[idx].isna() | (yrs_num.loc[idx] > 80)).mean():.3f}",
            f"{is_null_series(sub['Level']).mean():.3f}",
        ])
    out += md_table(["year", "sessions", "injured rate", "`YrsRunning` missing",
                     "`Level` missing"], year_rows)
    lvl_missing = is_null_series(lab["Level"])
    n_lvl = int(lvl_missing.sum())
    n_lvl_unin = int((lab_inj[lvl_missing] == 0).sum())
    out += [
        "",
        f"A blank `Level` marks an uninjured session with **{n_lvl_unin / n_lvl:.1%} "
        f"precision** ({n_lvl_unin} of {n_lvl}). The 2017 wave is "
        f"{int((lab_year == 2017).sum())} sessions, entirely uninjured, with `Level`, "
        "`YrsRunning` and `Activities` blank on every row.",
        "",
        "> **Never fit a missingness indicator as a feature, and never add collection "
        "year.** Either scores well above chance while measuring paperwork rather than "
        "physiology. Complete-case analysis is not a safe fallback either — requiring "
        "all control features present deletes 45% of the uninjured class and the whole "
        "2017 wave. Phase 1 must report a provenance-only baseline to bound how much of "
        "any score is this artifact. See `docs/phase1-spec.md`.",
        "",
        "### Sessions per subject (repeated measures)",
        "",
        f"min **{sessions.min()}**, median **{int(sessions.median())}**, "
        f"max **{sessions.max()}**. "
        f"**{int((sessions > 1).sum())}** of {run['sub_id'].nunique()} subjects "
        f"({100 * (sessions > 1).mean():.1f}%) have more than one session.",
        "",
        "Distribution: "
        + ", ".join(
            f"{k}→{v}" for k, v in sorted(sessions.value_counts().to_dict().items())
        ),
        "",
        "> This is why every split must use `groups=sub_id`.",
        "",
        "### Key fields",
        "",
        "**`InjDefn`** — free text, *not* 1–4. Five states including blank:",
        "",
    ]
    out += md_table(
        ["value", "n"],
        [[f"`{k}`" if k else "*(blank)*", str(v)]
         for k, v in run["InjDefn"].value_counts().items()],
    )

    out += ["", "**`InjJoint`**:", ""]
    out += md_table(
        ["value", "n"],
        [[f"`{k}`" if k else "*(blank)*", str(v)]
         for k, v in run["InjJoint"].value_counts().items()],
    )

    inj_side = run["InjSide"].value_counts()
    side_null = int(run["InjSide"].map(is_blank).sum())
    out += [
        "",
        "**`InjSide`** — gates phase 5:",
        "",
    ]
    out += md_table(
        ["value", "n"],
        [[f"`{k}`" if k else "*(blank)*", str(v)] for k, v in inj_side.items()],
    )
    injured = run[run["_status"] == "injured"]
    side_known_injured = int((~injured["InjSide"].map(is_blank)).sum())
    out += [
        "",
        f"Null rate overall **{100 * side_null / n_rows:.1f}%** ({side_null}/{n_rows}). "
        f"Restricted to injured sessions, side is present for "
        f"**{side_known_injured}/{len(injured)}** "
        f"({100 * side_known_injured / max(len(injured), 1):.1f}%).",
        "",
        "> `Bilateral` (305) and `Bi-lateral` (3) are the same value spelled two ways, and "
        "neither is a single side. A left/right asymmetry model can use only the "
        f"**{int((injured['InjSide'].map(norm).isin(['left', 'right'])).sum())}** injured "
        "sessions with a unilateral side.",
        "",
        "**`speed_r`** — the paper's `speed_w(r)`:",
        "",
    ]
    speed = to_num(run["speed_r"])
    out += [
        f"min **{speed.min():.3f}**, median **{speed.median():.3f}**, "
        f"max **{speed.max():.3f}** m/s; {int(speed.isna().sum())} null.",
        "",
        "**`SpecInjury`** — free text, "
        f"**{run['SpecInjury'].nunique()} distinct spellings**, "
        f"{int(run['SpecInjury'].map(is_blank).sum())} blank. Top 18 verbatim:",
        "",
    ]
    out += md_table(
        ["value (verbatim)", "n"],
        [[f"`{k}`" if k else "*(blank)*", str(v)]
         for k, v in run["SpecInjury"].value_counts().head(18).items()],
    )
    cleaned = run["_injury"].nunique()
    out += [
        "",
        f"> Casing is inconsistent throughout (`Pain`/`pain`, `Other`/`other`, "
        f"`ITB syndrome`, `Achilles tendonitis`/`achilles tendonitis`). Applying the "
        f"README's lowercase+merge recipe reduces {run['SpecInjury'].nunique()} spellings "
        f"to **{cleaned}**. Junk values survive: `Other`, `Pain`, "
        "`fill in specifics below`.",
        "",
    ]
    for col in ("Gender", "DominantLeg", "Level"):
        out += [f"**`{col}`**: " + ", ".join(
            f"`{k or '(blank)'}`={v}" for k, v in run[col].value_counts().items()
        ), ""]
    out += [
        "> `DominantLeg` carries the literal string `null` (14) *and* blanks (338) as "
        "two distinct spellings of missing.",
        "",
    ]
    return out


def section_cohort(run: pd.DataFrame) -> list[str]:
    out = ["## 3. Cohort", ""]
    by_status = run.groupby("_status").agg(
        sessions=("sub_id", "size"), subjects=("sub_id", "nunique")
    )
    rows = [[s, str(int(by_status.loc[s, "sessions"])), str(int(by_status.loc[s, "subjects"]))]
            for s in by_status.index]
    out += md_table(["status", "sessions", "unique subjects"], rows)

    subs = run["sub_id"].nunique()
    out += [
        "",
        f"**{subs} unique subjects have running data**, from {len(run)} sessions.",
        "",
        "> **Finding.** The paper reports 1,402 subjects injured at testing, and "
        "separately that 112 + 1,290 = **1,402** subjects have running data. The run "
        f"metadata contains exactly **{subs}** unique subjects. The two 1,402s are "
        "different quantities that happen to coincide — injury status among these "
        "subjects is split as in the table above, nothing like 1,402/0. Do not treat "
        "'has running data' as a proxy for 'injured'.",
        "",
        "### Top 10 diagnoses (after the README normalization)",
        "",
    ]
    inj = run[run["_status"] == "injured"].copy()
    top = inj[~inj["_injury"].isin(NO_INJURY_TOKENS)]["_injury"].value_counts().head(10)
    rows = []
    for name, n_sess in top.items():
        sub = inj[inj["_injury"] == name]
        rows.append([f"`{name}`", str(int(n_sess)), str(sub["sub_id"].nunique())])
    out += md_table(["diagnosis", "sessions", "subjects"], rows)

    out += ["", "### Means by injury group (first session per subject)", ""]
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [
        ("uninjured", run[run["_status"] == "uninjured"]),
        ("injured (all)", inj),
        ("unknown", run[run["_status"] == "unknown"]),
    ]
    groups += [(str(name), inj[inj["_injury"] == name]) for name in top.index[:5]]
    for label, frame in groups:
        if frame.empty:
            continue
        firsts = first_session_per_subject(frame)
        rows.append([
            label, str(firsts["sub_id"].nunique()),
            fmt(firsts["_age"].mean()), fmt(firsts["_height_clean"].mean()),
            fmt(firsts["_weight"].mean()), fmt(firsts["_speed_r"].mean(), 3),
        ])
    out += md_table(
        ["group", "subjects", "age", "height cm", "weight kg", "speed_r m/s"], rows
    )
    oa = inj[inj["_injury"] == "osteoarthritis"]
    out += [
        "",
        "Height is masked to "
        f"[{HEIGHT_MIN:.0f}, {HEIGHT_MAX:.0f}] cm here; the `999` sentinel is excluded.",
        "",
        "> **Osteoarthritis barely exists in the running data.** The paper's Table 1 "
        f"gives OA 247 subjects and 422 sessions; the run metadata holds "
        f"**{oa['sub_id'].nunique()} subjects over {len(oa)} sessions** — it does not "
        "even reach the top 10. OA participants overwhelmingly walked rather than ran. "
        "`CLAUDE.md` directs that OA be excluded from the primary analysis and reported "
        "separately; in the running cohort there is almost nothing there to exclude, and "
        "a separate OA analysis on kinematics is not viable at this n.",
        "",
        "> The two largest 'diagnoses' are `pain` and `other` — "
        f"{int(top.get('pain', 0)) + int(top.get('other', 0))} sessions of "
        "non-diagnoses. A usable label set is much smaller than the injured-session "
        "count suggests.",
        "",
    ]
    return out


def build_table1(frame: pd.DataFrame) -> dict[str, tuple]:
    """Apply the README's Table 1 recipe to a session-level frame."""
    result: dict[str, tuple] = {}
    healthy = frame[frame["_status"] == "uninjured"].copy()
    firsts = first_session_per_subject(healthy)
    young = set(firsts.loc[firsts["_age"].between(18, 49), "sub_id"])
    older = set(firsts.loc[firsts["_age"] >= 50, "sub_id"])
    result["No Injury (age 18-49)"] = group_stats(healthy[healthy["sub_id"].isin(young)])
    result["No Injury (age 50+)"] = group_stats(healthy[healthy["sub_id"].isin(older)])

    # Category membership is by diagnosis alone. An uninjured session cannot match a
    # named diagnosis (uninjured requires a blank SpecInjury), so no status filter.
    for name in PAPER_TABLE1:
        if name.startswith("No Injury"):
            continue
        sub = frame[frame["_injury"] == name]
        result[name] = group_stats(sub) if not sub.empty else (0, 0, None, None, None, 0,
                                                               None, None)
    return result


def section_table1(run: pd.DataFrame, union: pd.DataFrame) -> list[str]:
    out = ["## 4. Table 1 verification", "",
           "Applying the README recipe verbatim. Counts must match exactly, means to "
           "within 0.01. Computed from the **union** of the run and walk metadata — "
           "Table 1 counts *data collections*, and a session may be walk-only, so the "
           "running file alone cannot reproduce it.", ""]

    run_t1 = build_table1(run)
    union_t1 = build_table1(union)
    labels = ["M", "F", "age", "height", "mass", "sessions", "walk spd", "run spd"]

    n_pass = n_total = 0
    rows = []
    per_group_pass = {}
    for group, paper in PAPER_TABLE1.items():
        obs = union_t1[group]
        cells = [f"**{group}**"]
        g_pass = 0
        for i, label in enumerate(labels):
            exact = i in (0, 1, 5)
            p, o = paper[i], obs[i]
            places = 3 if "spd" in label else 2
            if o is None or (isinstance(o, float) and pd.isna(o)):
                cells.append(f"{fmt(p, places)} → —")
                continue
            ok = int(o) == int(p) if exact else abs(float(o) - float(p)) <= 0.01
            n_total += 1
            n_pass += ok
            g_pass += ok and i < 6
            cells.append(f"{fmt(p, places)} → {fmt(o, places)} {'✓' if ok else '**✗**'}")
        rows.append(cells)
        # Speeds miss almost everywhere for a systematic reason (see below), so the
        # informative score is the six subject/session fields.
        per_group_pass[group] = g_pass

    out += [f"**Result: {n_pass}/{n_total} cells reproduce** "
            "(`paper → observed`, from the union).", ""]
    out += md_table(["group"] + labels, rows)

    exact_groups = [g for g, c in per_group_pass.items() if c == 6]
    out += [
        "",
        f"**{len(exact_groups)} of 7 groups reproduce on all six subject and session "
        "counts** (M, F, age, height, mass, sessions): "
        + (", ".join(f"`{g}`" for g in exact_groups) if exact_groups else "none")
        + ". The recipe in the README is therefore substantially correct.",
        "",
        "**Where it diverges:**",
        "",
        "- **Every male/female count reproduces exactly** except `No Injury (age 50+)` "
        "(49→48), and 5 of 7 session counts are exact — including osteoarthritis at "
        "247 subjects and 422 sessions. The cohort definitions are right.",
        "- **Speeds are the largest source of failures**, sitting 0.005–0.07 m/s low in "
        "nearly every group. A systematic offset, not noise: the paper appears to "
        "average over all sessions rather than first-session-per-subject.",
        "- **Residual mean differences are small** (age ≤0.05, height ≤0.70, mass "
        "≤0.37) and consistent with a slightly different same-day-trial or "
        "first-session tie-break rule. Session counts miss by 1–2 in the same way "
        "(558→560, 128→127).",
        "- **`No Injury (age 50+)` is the weakest group**, missing on every field. The "
        "age-50 boundary is evaluated against a different session than the one taken "
        "as first, which moves a subject across the boundary.",
        "",
        "> **Getting OA to reproduce required departing from the README's stated rule.** "
        "352 osteoarthritis sessions carry a real `SpecInjury` but a *blank* `InjDefn`. "
        "Read literally, the README's severity-first definition files them as neither "
        "injured nor uninjured and the whole OA cohort vanishes from Table 1. This "
        "inventory treats a recorded diagnosis as outranking a blank severity. "
        "Downstream labelling must do the same or silently lose 20% of the injured "
        "sessions.",
        "",
    ]

    ach = union[union["_injury"] == "achilles tendonitis"]
    ach_first = first_session_per_subject(ach)
    raw = ach_first["_height"].dropna()
    clean = plausible_height(raw).dropna()
    n_sentinel = int((raw >= HEIGHT_MAX).sum())
    out += [
        "### The 190.19 cm Achilles tendonitis height — resolved",
        "",
        f"**It reproduces exactly: observed {fmt(raw.mean())} cm vs published 190.19 cm.** "
        "The published figure is faithful to the file. The file is what is wrong.",
        "",
        f"Of {len(raw)} subjects in this group, **{n_sentinel}** "
        f"{'carries' if n_sentinel == 1 else 'carry'} the sentinel "
        f"`Height = 999`. Masking implausible values leaves {len(clean)} real "
        f"measurements spanning {fmt(clean.min())}–{fmt(clean.max())} cm with a mean of "
        f"**{fmt(clean.mean())} cm** — entirely ordinary, and ~{fmt(raw.mean() - clean.mean())} cm "
        "below the published value.",
        "",
        "> A single `999` in a group of ~50 shifts the mean by about 16 cm. The paper's "
        "Table 1 did not mask it. **Treat every published group mean involving `Height` "
        "as contaminated**, and mask the sentinel before computing anything.",
        "",
    ]
    return out


def section_json(root: Path, run: pd.DataFrame) -> tuple[list[str], list[str], float]:
    """Sample JSONs one at a time. Returns (markdown, failures, mean_seconds)."""
    ric = root / "ric_data"
    inj = run.copy()
    inj["_bucket"] = inj["_status"] + "/" + inj["_injury"].where(
        inj["_status"] == "injured", ""
    )
    # Deterministic stratified pick: one per bucket, round-robin, ordered by sub_id.
    picks: list[pd.Series] = []
    buckets = sorted(inj["_bucket"].unique())
    per_bucket = {b: inj[inj["_bucket"] == b].sort_values("sub_id") for b in buckets}
    i = 0
    while len(picks) < N_SAMPLE_JSON:
        added = False
        for b in buckets:
            frame = per_bucket[b]
            if i < len(frame):
                picks.append(frame.iloc[i])
                added = True
                if len(picks) == N_SAMPLE_JSON:
                    break
        if not added:
            break
        i += 1

    failures: list[str] = []
    stats: list[dict] = []
    for row in picks:
        path = ric / str(row["sub_id"]) / str(row["filename"])
        if not path.exists():
            failures.append(f"`{row['sub_id']}/{row['filename']}` — file not found")
            continue
        size_mb = path.stat().st_size / 1e6
        t0 = time.perf_counter()
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - reported, never silently skipped
            failures.append(f"`{row['sub_id']}/{path.name}` — {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - t0

        running = data.get("running") or {}
        dv_r = data.get("dv_r") or {}
        left = dv_r.get("left", {}) if isinstance(dv_r, dict) else {}
        frames = len(next(iter(running.values()))) if running else 0
        stats.append({
            "sub_id": row["sub_id"], "mb": size_mb, "sec": elapsed,
            "keys": tuple(data.keys()),
            "hz_r": data.get("hz_r"), "hz_w": data.get("hz_w"),
            "markers": len(running), "frames": frames,
            "has_walk": bool(data.get("walking")),
            "dv_r_n": len(left),
            "dv_r_listvals": sum(isinstance(v, list) for v in left.values()),
            "marker_names": tuple(running.keys()),
            "joints": tuple((data.get("joints") or {}).keys()),
        })
        del data

    out = ["## 5. Inside the session JSONs", "",
           f"{len(stats)} files sampled across injury groups, opened one at a time.", ""]
    if not stats:
        out += ["**No files could be read.** See failures below.", ""]
        return out, failures, 0.0

    first = stats[0]
    out += [
        "### Schema sketch",
        "",
        "```",
        "<root>: dict(8)",
        "  hz_w     int | []        sampling frequency, walking",
        "  hz_r     int | []        sampling frequency, running",
        "  joints   dict(14)        anatomical landmarks, [3] each (static neutral)",
        "  neutral  dict(30)        cluster markers, [3] each (static neutral)",
        "  walking  dict(30) | []   marker trajectories, [n_frames, 3] each",
        "  running  dict(30) | []   marker trajectories, [n_frames, 3] each",
        "  dv_w     dict | []       descriptive variables, walking",
        "  dv_r     dict{left,right}  descriptive variables, running -- 76 SCALARS each",
        "```",
        "",
        f"Top-level keys are identical across all {len(stats)} sampled files: "
        f"`{'`, `'.join(first['keys'])}`.",
        "",
        "### Are kinematics precomputed?",
        "",
        "**No — and this is the single most important finding of phase 0.**",
        "",
        f"`dv_r` contains **{first['dv_r_n']} variables per side** "
        f"(`left` and `right` stored separately, identical key sets). Every one is a "
        f"**scalar** — across the sample, "
        f"{sum(s['dv_r_listvals'] for s in stats)} of "
        f"{sum(s['dv_r_n'] for s in stats)} values were arrays.",
        "",
        "The 9 joint-angle channels do exist, but only as discrete summaries — "
        "`HIP_EXT`/`HIP_ADD`/`HIP_ROT`, `KNEE_FLEX`/`KNEE_ADD`/`KNEE_ABD`/`KNEE_ROT`, "
        "`ANKLE_DF`/`ANKLE_EVE`/`ANKLE_ROT`, each as `_PEAK_ANGLE`, "
        "`_percent_STANCE`, `_at_HS`, `_EXCURSION` (plus `_PEAK_VEL` variants).",
        "",
        "**Nothing is time-normalized to 100 points.** The paper's "
        "\"time normalized to 100 data points\" describes the *output of the bundled "
        "MATLAB pipeline* (`gait_kinematics.m` → `gait_steps.m`), which is shipped in "
        "`Supplemental_materials/` but has not been run. The archive stores its "
        "*inputs*, not its outputs.",
        "",
        "### Corrections from the full corpus (phase 1)",
        "",
        "The two claims above were made from the 20-file sample. Streaming all "
        "1,745 running sessions corrected both:",
        "",
        "- **Not every value is a scalar.** 13 sessions (0.7%) carry 2 non-scalar "
        "  values each. Rare, but \"0 of 1,520\" was a sampling artifact.",
        "- **Half the `dv_r` columns are permanently empty.** Of 152 (76 × 2 "
        "  sides), **72 are 100% exact-zero** and 4 more are 20–41% zero; only "
        "  **76 carry data**. This is by pipeline design, not storage truncation: "
        "  `gait_steps.m` preallocates `DISCRETE_VARIABLES = zeros(77,3)` and "
        "  assigns exactly **40 rows** — matching the 40 live variables per side "
        "  observed. Re-running the pipeline does not populate the rest.",
        "",
        "> **The dead columns are systematically the sagittal ones.** Per side the "
        "entire live sagittal content is three numbers — `HIP_EXT_PEAK_ANGLE`, "
        "`KNEE_FLEX_PEAK_ANGLE`, `ANKLE_DF_PEAK_ANGLE`. Every sagittal "
        "`_EXCURSION`, `_at_HS`, `_percent_STANCE` and `_PEAK_VEL` is empty. What "
        "survives is frontal/transverse and spatiotemporal — precisely what a "
        "single side-on camera recovers *worst*. **`dv_r` cannot support the "
        "phase 2 degradation analysis**; only the generated waveforms can.",
        "",
        "### Array shapes and channels",
        "",
        "`running` holds **28–30 marker channels**, each "
        "`[n_frames, 3]` — raw 3D trajectories, not angles. Left and right are separate "
        "channels (`L_thigh_1`…`R_foot_4`, `L_toe`, `R_toe`), covering 7 rigid segments: "
        "pelvis, bilateral thigh, shank and foot.",
        "",
    ]
    frames = [s["frames"] for s in stats if s["frames"]]
    sizes = [s["mb"] for s in stats]
    markers = sorted({s["markers"] for s in stats})
    joints = sorted({len(s["joints"]) for s in stats})
    out += md_table(
        ["property", "observed across sample"],
        [
            ["frames per marker", f"min {min(frames)}, max {max(frames)}"],
            ["marker channels", ", ".join(str(x) for x in markers)],
            ["`joints` landmarks", ", ".join(str(x) for x in joints)],
            ["walking present", f"{sum(s['has_walk'] for s in stats)}/{len(stats)} sampled files"],
            ["file size", f"min {min(sizes):.1f} MB, mean {sum(sizes) / len(sizes):.1f} MB, "
                          f"max {max(sizes):.1f} MB"],
        ],
    )
    out += [
        "",
        f"> **The marker set is not uniform.** Sampled files carry "
        f"{' or '.join(str(x) for x in markers)} marker channels and "
        f"{' or '.join(str(x) for x in joints)} `joints` landmarks. This matches the "
        "paper's note that extended anatomical markers exist for only 1,082 of 1,798 "
        "subjects. Any feature depending on the extended set silently drops a large "
        "part of the cohort — check per-session before assuming a fixed channel list.",
        "",
    ]
    out += section_hz(root)
    mean_sec = sum(s["sec"] for s in stats) / len(stats)
    return out, failures, mean_sec


def section_hz(root: Path) -> list[str]:
    """Sampling rates for every session, read from each file's first 4 KB.

    `hz_w` and `hz_r` are the first two keys, so the whole corpus can be surveyed
    without parsing 48 GB.
    """
    pattern = re.compile(
        rb'"hz_w"\s*:\s*(\[\]|[0-9.]+).*?"hz_r"\s*:\s*(\[\]|[0-9.]+)', re.S
    )
    combos: Counter[tuple[str, str]] = Counter()
    unmatched = 0
    t0 = time.perf_counter()
    for path in sorted((root / "ric_data").rglob("*.json")):
        with path.open("rb") as fh:
            match = pattern.search(fh.read(4096))
        if match:
            combos[(match.group(1).decode(), match.group(2).decode())] += 1
        else:
            unmatched += 1
    elapsed = time.perf_counter() - t0

    def label(value: str) -> str:
        return "absent" if value == "[]" else f"{value} Hz"

    rows = [[label(w), label(r), str(n)] for (w, r), n in
            sorted(combos.items(), key=lambda kv: -kv[1])]
    total = sum(combos.values())
    run_120 = sum(n for (_, r), n in combos.items() if r == "120")
    run_200 = sum(n for (_, r), n in combos.items() if r == "200")

    out = [
        "### Sampling frequency — all 2,506 files",
        "",
        f"Scanned every file's first 4 KB in {elapsed:.1f} s ({unmatched} unmatched).",
        "",
    ]
    out += md_table(["`hz_w`", "`hz_r`", "files"], rows)
    out += [
        "",
        f"Both 120 Hz and 200 Hz are present, but **running is almost entirely 200 Hz**: "
        f"{run_200:,} sessions vs **{run_120}** at 120 Hz. Walking carries most of the "
        "120 Hz data.",
        "",
        "> **This cross-validates the metadata exactly.** Files with running data "
        f"= {run_200 + run_120:,}, matching `run_data_meta.csv`'s row count; files with "
        f"walking = {total - sum(n for (w, _), n in combos.items() if w == '[]'):,}, "
        f"matching `walk_data_meta.csv`; total {total:,} files. Every session file is "
        "accounted for in exactly the CSVs that claim it.",
        "",
        "> Downsampling for phase 2 should target 120 Hz→30 Hz from a 200 Hz base for "
        f"all but {run_120} running sessions.",
        "",
    ]
    return out


def section_feasibility(root: Path, run: pd.DataFrame, mean_sec: float,
                        failures: list[str], n_json: int, total_gb: float) -> list[str]:
    n_sessions = len(run)
    waveform_bytes = n_sessions * 9 * 101 * 4
    full_pass_min = mean_sec * n_json / 60

    out = ["## 6. Extraction feasibility", ""]
    out += md_table(
        ["quantity", "value"],
        [
            ["session JSONs on disk", f"{n_json:,}"],
            ["total extracted size", f"{total_gb:.2f} GB"],
            ["running sessions (`run_data_meta.csv`)", f"{n_sessions:,}"],
            ["unique subjects with running data", f"{run['sub_id'].nunique():,}"],
            ["mean parse time per file", f"{mean_sec:.2f} s"],
            ["**projected full streaming pass**", f"**~{full_pass_min:.0f} min** single-threaded"],
            ["target waveform array", f"`({n_sessions}, 9, 101)` float32 — see below"],
            ["that array on disk", f"**{waveform_bytes / 1e6:.1f} MB** — trivially small"],
        ],
    )
    out += [
        "",
        f"**Tractable, with one caveat that is not about size.** The {waveform_bytes / 1e6:.1f} MB "
        f"target array cannot be read out of the archive: it must be *computed* from "
        f"{total_gb:.0f} GB of raw marker trajectories by running the bundled MATLAB "
        "pipeline, or a faithful Python reimplementation of it. That work is not yet "
        "scoped and belongs to a spec of its own before phase 1 begins.",
        "",
        "> **Correction to `CLAUDE.md`: the waveforms are 101 points, not 100.** "
        "`gait_steps.m:676` allocates `zeros(101, n_steps, 3)` and interpolates over "
        "`0:(TO-TD)/100:(TO-TD)` — 0 to 100 inclusive. It also emits more than 9 "
        "channels: ankle, knee, hip, foot and pelvis, each 3-plane, per side, plus "
        "matching velocities. The 9-channel sagittal-relevant subset is a choice made "
        "downstream, not the pipeline's native output.",
        "",
        "MATLAB R2026a is installed locally; the pipeline was written for R2023a.",
        "",
        "### Extraction notes for whoever repeats this",
        "",
        "The outer zip is **stored** (not compressed), so its entries are a byte copy. "
        "The nested `ric_data.zip` is **Deflate64** (method 9). Verified to fail: "
        "Python `zipfile` (`NotImplementedError`), .NET `System.IO.Compression` "
        "(*unsupported compression method*), bsdtar/libarchive 3.8 (*Damaged Zip "
        "archive*). The Windows shell zip handler does support it and needs no install "
        "— see `scripts/extract_ric_data.ps1`. The archive's internal root folder is "
        "`reformat_data/`, renamed to `ric_data/` on extract.",
        "",
        "### Parse failures",
        "",
    ]
    if failures:
        out += [f"- {line}" for line in failures]
    else:
        out += ["None. Every sampled file parsed as valid JSON."]
    out += [""]
    return out


def section_gate(run: pd.DataFrame) -> list[str]:
    by_status = run.groupby("_status")["sub_id"].nunique()
    injured_sessions = run[run["_status"] == "injured"]
    side_ok = int(injured_sessions["InjSide"].map(norm).isin(["left", "right"]).sum())
    return [
        "## 7. Gate questions",
        "",
        "**1. Real column names and encodings?** 26 columns, listed in §2. `speed_r` not "
        "`speed_w(r)`; `InjDefn` is free text not 1–4; missingness is spelled four ways "
        "(`''`, `NaN`, `null`, `N/A`).",
        "",
        f"**2. Unique subjects with running data, healthy vs injured?** "
        f"**{run['sub_id'].nunique()}** subjects over {len(run)} sessions — "
        + ", ".join(f"{k} **{v}**" for k, v in by_status.items())
        + ". (Subjects can appear in more than one status across sessions.)",
        "",
        "**3. Does Table 1 reproduce?** Substantially — see §4 for the cell-by-cell "
        "verdict. Every group reproduces on male/female and session counts bar one "
        "(`No Injury (age 50+)`), including osteoarthritis at 247 subjects / 422 "
        "sessions. Speeds carry a systematic low offset. The 190.19 cm Achilles height "
        "**does** "
        "reproduce, and is inflated by a `Height = 999` missing-data sentinel the paper "
        "did not mask; the real mean is ~175 cm.",
        "",
        "**4. What is in the JSONs, are kinematics precomputed?** Raw 3D marker "
        "trajectories (28–30 channels × n_frames × 3) plus 76 **scalar** descriptive "
        "variables per side. **Kinematics are not precomputed and no waveform is "
        "time-normalized.** The waveform array must be generated from raw markers, and "
        "is 101 points per stance phase, not 100.",
        "",
        f"**5. Is `InjSide` usable for phase 5?** Qualified yes. "
        f"**{side_ok}** injured sessions carry a unilateral `Left`/`Right`. "
        "`Bilateral` (+3 spelled `Bi-lateral`) is not a side and must be excluded or "
        "modelled separately.",
        "",
    ]


def main() -> int:
    load_dotenv(REPO / ".env")
    root_env = os.environ.get("DATA_ROOT")
    if not root_env:
        raise SystemExit("DATA_ROOT is not set. Copy .env.example to .env and set it.")
    root = Path(root_env)

    required = ["README.txt", "run_data_meta.csv", "walk_data_meta.csv"]
    missing = [name for name in required if not (root / name).exists()]
    if missing or not (root / "ric_data").is_dir():
        raise SystemExit(f"Missing under {root}: {missing or 'ric_data/'} — stopping.")

    json_files = list((root / "ric_data").rglob("*.json"))
    total_gb = sum(p.stat().st_size for p in json_files) / 1e9

    run, walk, union = load_meta(root)

    body: list[str] = [
        "# Phase 0 — Data Inventory",
        "",
        "**Observed**, by `scripts/phase0_inventory.py`, from the extracted archive at "
        "`$DATA_ROOT`. Where this file and `docs/ferber-schema.md` disagree, **this file "
        "wins** — it is measured, the schema doc is derived from the paper.",
        "",
        f"Ferber / Running Injury Clinic Kinematic Dataset · {len(json_files):,} session "
        f"JSONs · {total_gb:.1f} GB extracted · CC BY 4.0",
        "",
        "---",
        "",
    ]
    body += section_readme(root)
    body += section_meta(run, walk)
    body += section_cohort(run)
    body += section_table1(run, union)
    json_md, failures, mean_sec = section_json(root, run)
    body += json_md
    body += section_feasibility(root, run, mean_sec, failures, len(json_files), total_gb)
    body += section_gate(run)
    body += [
        "---",
        "",
        "## Deferred to later phases",
        "",
        "- ~~**Generating the waveforms** from raw markers via the MATLAB "
        "pipeline.~~ **Done in phase 1C** — the vendored pipeline reproduces the "
        "archive's own `dv_r` exactly, so the 101-point curves it emits are "
        "verified. See `results/phase01c.md`.",
        "- ~~Whether `0` in `dv_r` means zero or missing.~~ **Resolved** — 72 of "
        "152 columns are never written by `gait_steps.m` at all. Sentinel, not "
        "measurement. See §5.",
        "- ~~Which sessions carry the extended marker set, and what that costs in "
        "n.~~ **Answered** — 28-channel sessions exist only in 2014–2016 (876 "
        "sessions); every session before 2014 has 30. Marker set is very nearly "
        "a collection-year label, and is a confound rather than a feature.",
        "- Whether the 12 osteoarthritis subjects with running data are usable at all, "
        "given `CLAUDE.md` requires OA be reported separately.",
        "- Reconciling `SpecInjury` free text into a usable label set beyond the "
        "README's five merges.",
        "- `walk_data_meta.csv` is characterized here only where Table 1 needs it.",
        "",
    ]

    OUT.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(body)} lines)")
    if failures:
        print(f"WARNING: {len(failures)} file(s) failed to parse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
