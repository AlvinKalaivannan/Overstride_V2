# Phase 0 — Data Inventory

**Observed**, by `scripts/phase0_inventory.py`, from the extracted archive at `$DATA_ROOT`. Where this file and `docs/ferber-schema.md` disagree, **this file wins** — it is measured, the schema doc is derived from the paper.

Ferber / Running Injury Clinic Kinematic Dataset · 2,506 session JSONs · 48.3 GB extracted · CC BY 4.0

---

## 1. The official README vs the schema doc

`README.txt` is 148 lines. It documents 4 file categories: 2,506 session JSONs, 2 metadata CSVs, 8 MATLAB processing files, 8 tutorial files.

**Contradictions and extensions, against `docs/ferber-schema.md`:**

| # | Claim | Observed | Impact |
|---|---|---|---|
| 1 | Schema/paper: waveforms available | JSONs hold **raw marker trajectories** + **scalar** descriptive variables. No 100-point curves anywhere. | **Blocks phase 1** — see §5 |
| 2 | README names the CSVs `run_meta_data.csv` / `walk_meta_data.csv` | Actual: `run_data_meta.csv` / `walk_data_meta.csv` | README internal error |
| 3 | Schema field `speed_w(r)` | Two files, two names: `speed_r` (run), `speed_w` (walk) | Naming only |
| 4 | `InjDefn` = 1 of 4 options | Free-text strings, not integers; a 5th state (blank) exists | Encoding |
| 5 | README: `InjSide` is right/left | `Right`/`Left`/`Bilateral`/`Bi-lateral`/blank | Phase 5 — see §2 |
| 6 | Archive root `ric_data/` | Archive's internal root folder is `reformat_data/`; renamed on extract | Layout |
| 7 | n = 1,798 subjects | 1,798 total, but only 1,402 have **running** data | Cohort size — see §3 |
| 8 | Archive is a plain zip | Nested zip uses **Deflate64**; Python `zipfile`, .NET and libarchive all fail | Tooling — see §6 |

**The README also supplies the Table 1 reproduction recipe** — injury-string normalization plus subject-level counting rules. §4 applies it verbatim.

**README definition of uninjured**, used throughout: `InjDefn` = 'No injury' **and** `InjJoint` ∈ {'No injury', blank} **and** `SpecInjury` blank.

## 2. `run_data_meta.csv`

**1832 rows × 26 columns**, 1402 unique `sub_id`.
(`walk_data_meta.csv`: 2088 rows, 1686 subjects.)

> The file has 1,908 physical lines but **1,832 records** — `Activities` contains quoted commas and embedded newlines. Parse with a real CSV reader, never by line.

### Columns, verbatim and in order

```
sub_id, datestring, filename, speed_r, age, Height, Weight, Gender, DominantLeg, InjDefn, InjJoint, InjSide, SpecInjury, InjDuration, InjJoint2, InjSide2, SpecInjury2, Activities, Level, YrsRunning, RaceDistance, RaceTimeHrs, RaceTimeMins, RaceTimeSecs, YrPR, NumRaces
```

### Null count per column

Null = any of `''`, `NaN`, `null`, `N/A`. The dataset mixes all four.

| column | inferred | nulls | null % | distinct |
|---|---|---|---|---|
| `sub_id` | numeric | 0 | 0.0% | 1402 |
| `datestring` | text | 0 | 0.0% | 1476 |
| `filename` | text | 0 | 0.0% | 1832 |
| `speed_r` | numeric | 0 | 0.0% | 1831 |
| `age` | numeric | 0 | 0.0% | 56 |
| `Height` | numeric | 3 | 0.2% | 296 |
| `Weight` | numeric | 2 | 0.1% | 478 |
| `Gender` | text | 0 | 0.0% | 3 |
| `DominantLeg` | text | 352 | 19.2% | 5 |
| `InjDefn` | text | 80 | 4.4% | 5 |
| `InjJoint` | text | 234 | 12.8% | 12 |
| `InjSide` | text | 486 | 26.5% | 5 |
| `SpecInjury` | text | 589 | 32.2% | 98 |
| `InjDuration` | numeric | 1427 | 77.9% | 57 |
| `InjJoint2` | text | 996 | 54.4% | 11 |
| `InjSide2` | text | 1010 | 55.1% | 4 |
| `SpecInjury2` | text | 1512 | 82.5% | 57 |
| `Activities` | text | 317 | 17.3% | 802 |
| `Level` | text | 269 | 14.7% | 3 |
| `YrsRunning` | numeric | 517 | 28.2% | 55 |
| `RaceDistance` | text | 338 | 18.4% | 10 |
| `RaceTimeHrs` | numeric | 853 | 46.6% | 20 |
| `RaceTimeMins` | numeric | 807 | 44.1% | 72 |
| `RaceTimeSecs` | text | 908 | 49.6% | 49 |
| `YrPR` | numeric | 1407 | 76.8% | 23 |
| `NumRaces` | numeric | 1328 | 72.5% | 19 |

### Sentinel values — four columns, not one

`999` is used as a missing-data marker, and it is **not confined to `Height`**. Every numeric feature of the demographics-only control was swept against a plausible physiological range:

| column | plausible range | outside | offending values | clean range |
|---|---|---|---|---|
| `age` | [10, 100] | 1 | `255`×1 | 18–73 |
| `Height` | [120, 250] | 5 | `0`×3, `999`×2 | 120–196.5 |
| `Weight` | [30, 200] | 3 | `0`×2, `1564`×1 | 42.5–176 |
| `speed_r` | [0.5, 8] | 0 | — | 1.17205–4.877 |
| `YrsRunning` | [0, 80] | 51 | `999`×51 | 0–54 |

> **Mask all four before using them as features.** `YrsRunning` is the costly one — its 51 `999` rows read as 999 years of running experience and are silently counted as data by any tool that only checks for nulls. `Height`'s sentinel propagates into the published Table 1 (see §4); the others corrupt any model fitted on them.

### Missingness is not random — it tracks the collection wave

**This is the most dangerous finding in the metadata.** Whether a field was filled in at all indicates the label, because questionnaire completeness varies by study wave and the waves differ in injury mix:

| year | sessions | injured rate | `YrsRunning` missing | `Level` missing |
|---|---|---|---|---|
| 2009 | 34 | 0.353 | 0.794 | 0.000 |
| 2010 | 88 | 0.898 | 0.159 | 0.000 |
| 2011 | 181 | 0.934 | 0.370 | 0.000 |
| 2012 | 176 | 0.659 | 0.233 | 0.040 |
| 2013 | 178 | 0.522 | 0.112 | 0.000 |
| 2014 | 586 | 0.761 | 0.169 | 0.009 |
| 2015 | 314 | 0.490 | 0.427 | 0.350 |
| 2016 | 137 | 0.730 | 0.307 | 0.190 |
| 2017 | 51 | 0.000 | 1.000 | 1.000 |

A blank `Level` marks an uninjured session with **95.0% precision** (189 of 199). The 2017 wave is 51 sessions, entirely uninjured, with `Level`, `YrsRunning` and `Activities` blank on every row.

> **Never fit a missingness indicator as a feature, and never add collection year.** Either scores well above chance while measuring paperwork rather than physiology. Complete-case analysis is not a safe fallback either — requiring all control features present deletes 45% of the uninjured class and the whole 2017 wave. Phase 1 must report a provenance-only baseline to bound how much of any score is this artifact. See `docs/phase1-spec.md`.

### Sessions per subject (repeated measures)

min **1**, median **1**, max **10**. **209** of 1402 subjects (14.9%) have more than one session.

Distribution: 1→1193, 2→102, 3→51, 4→19, 5→26, 6→6, 7→2, 8→2, 10→1

> This is why every split must use `groups=sub_id`.

### Key fields

**`InjDefn`** — free text, *not* 1–4. Five states including blank:

| value | n |
|---|---|
| `No injury` | 659 |
| `Training volume/intensity affected` | 499 |
| `Continuing to train in pain` | 320 |
| `2 workouts missed in a row` | 274 |
| *(blank)* | 80 |

**`InjJoint`**:

| value | n |
|---|---|
| `Knee` | 348 |
| *(blank)* | 234 |
| `No Injury` | 228 |
| `No injury,No injury` | 189 |
| `Lower Leg` | 185 |
| `Thigh` | 181 |
| `Foot` | 141 |
| `Hip/Pelvis` | 136 |
| `Ankle` | 107 |
| `Lumbar Spine` | 41 |
| `Sacroiliac Joint` | 22 |
| `Other` | 20 |

**`InjSide`** — gates phase 5:

| value | n |
|---|---|
| `Right` | 632 |
| *(blank)* | 486 |
| `Left` | 406 |
| `Bilateral` | 305 |
| `Bi-lateral` | 3 |

Null rate overall **26.5%** (486/1832). Restricted to injured sessions, side is present for **1149/1183** (97.1%).

> `Bilateral` (305) and `Bi-lateral` (3) are the same value spelled two ways, and neither is a single side. A left/right asymmetry model can use only the **842** injured sessions with a unilateral side.

**`speed_r`** — the paper's `speed_w(r)`:

min **1.172**, median **2.721**, max **4.877** m/s; 0 null.

**`SpecInjury`** — free text, **98 distinct spellings**, 589 blank. Top 18 verbatim:

| value (verbatim) | n |
|---|---|
| *(blank)* | 550 |
| `Other` | 134 |
| `ITB syndrome` | 126 |
| `Pain` | 119 |
| `pain` | 113 |
| `patellofemoral pain syndrome` | 86 |
| `No injury` | 70 |
| `other` | 50 |
| `Patellofemoral pain syndrome` | 42 |
| `Achilles tendonitis` | 41 |
| `N/A` | 39 |
| `Calf muscle strain` | 37 |
| `Shin splints` | 33 |
| `Plantar fasciitis` | 28 |
| `plantar fasciitis` | 22 |
| `achilles tendonitis` | 21 |
| `Hamstring muscle strain` | 19 |
| `calf muscle strain` | 18 |

> Casing is inconsistent throughout (`Pain`/`pain`, `Other`/`other`, `ITB syndrome`, `Achilles tendonitis`/`achilles tendonitis`). Applying the README's lowercase+merge recipe reduces 98 spellings to **66**. Junk values survive: `Other`, `Pain`, `fill in specifics below`.

**`Gender`**: `Female`=926, `Male`=905, `Unknown`=1

**`DominantLeg`**: `Right`=1131, `Left`=348, `(blank)`=338, `null`=14, `Ambidextrous`=1

**`Level`**: `Recreational`=1042, `Competitive`=521, `(blank)`=269

> `DominantLeg` carries the literal string `null` (14) *and* blanks (338) as two distinct spellings of missing.

## 3. Cohort

| status | sessions | unique subjects |
|---|---|---|
| injured | 1183 | 988 |
| uninjured | 576 | 363 |
| unknown | 73 | 71 |

**1402 unique subjects have running data**, from 1832 sessions.

> **Finding.** The paper reports 1,402 subjects injured at testing, and separately that 112 + 1,290 = **1,402** subjects have running data. The run metadata contains exactly **1402** unique subjects. The two 1,402s are different quantities that happen to coincide — injury status among these subjects is split as in the table above, nothing like 1,402/0. Do not treat 'has running data' as a proxy for 'injured'.

### Top 10 diagnoses (after the README normalization)

| diagnosis | sessions | subjects |
|---|---|---|
| `pain` | 232 | 212 |
| `other` | 184 | 140 |
| `patellofemoral pain syndrome` | 134 | 129 |
| `itb syndrome` | 126 | 99 |
| `achilles tendonitis` | 62 | 48 |
| `calf muscle strain` | 55 | 42 |
| `plantar fasciitis` | 50 | 45 |
| `shin splints` | 36 | 19 |
| `hamstring muscle strain` | 31 | 28 |
| `medial tibial stress syndrome` | 30 | 30 |

### Means by injury group (first session per subject)

| group | subjects | age | height cm | weight kg | speed_r m/s |
|---|---|---|---|---|---|
| uninjured | 363 | 36.28 | 171.82 | 69.33 | 2.747 |
| injured (all) | 988 | 39.90 | 172.09 | 72.60 | 2.564 |
| unknown | 71 | 53.85 | 172.94 | 72.60 | 2.620 |
| pain | 212 | 41.83 | 171.45 | 70.67 | 2.486 |
| other | 140 | 41.14 | 172.27 | 71.44 | 2.617 |
| patellofemoral pain syndrome | 129 | 35.30 | 172.17 | 69.53 | 2.596 |
| itb syndrome | 99 | 35.09 | 171.97 | 67.70 | 2.615 |
| achilles tendonitis | 48 | 41.52 | 175.31 | 76.95 | 2.682 |

Height is masked to [120, 250] cm here; the `999` sentinel is excluded.

> **Osteoarthritis barely exists in the running data.** The paper's Table 1 gives OA 247 subjects and 422 sessions; the run metadata holds **12 subjects over 14 sessions** — it does not even reach the top 10. OA participants overwhelmingly walked rather than ran. `CLAUDE.md` directs that OA be excluded from the primary analysis and reported separately; in the running cohort there is almost nothing there to exclude, and a separate OA analysis on kinematics is not viable at this n.

> The two largest 'diagnoses' are `pain` and `other` — 416 sessions of non-diagnoses. A usable label set is much smaller than the injured-session count suggests.

## 4. Table 1 verification

Applying the README recipe verbatim. Counts must match exactly, means to within 0.01. Computed from the **union** of the run and walk metadata — Table 1 counts *data collections*, and a session may be walk-only, so the running file alone cannot reproduce it.

**Result: 36/56 cells reproduce** (`paper → observed`, from the union).

| group | M | F | age | height | mass | sessions | walk spd | run spd |
|---|---|---|---|---|---|---|---|---|
| **No Injury (age 18-49)** | 137 → 137 match | 171 → 171 match | 32.52 → 32.52 match | 172.19 → 172.20 match | 69.54 → 69.53 match | 558 → 560 **differs** | 1.210 → 1.207 match | 2.800 → 2.795 match |
| **No Injury (age 50+)** | 39 → 39 match | 49 → 48 **differs** | 55.80 → 55.85 **differs** | 165.33 → 165.45 **differs** | 69.89 → 70.04 **differs** | 130 → 128 **differs** | 1.180 → 1.151 **differs** | 2.580 → 2.518 **differs** |
| **achilles tendonitis** | 30 → 30 match | 22 → 22 match | 42.62 → 42.62 match | 190.19 → 190.19 match | 77.20 → 77.20 match | 68 → 68 match | 1.300 → 1.284 **differs** | 2.680 → 2.682 match |
| **itb syndrome** | 39 → 39 match | 61 → 61 match | 35.21 → 35.21 match | 171.99 → 171.99 match | 67.76 → 67.76 match | 128 → 127 **differs** | 1.260 → 1.256 match | 2.640 → 2.615 **differs** |
| **osteoarthritis** | 91 → 91 match | 156 → 156 match | 56.36 → 56.32 **differs** | 167.40 → 166.70 **differs** | 76.36 → 75.99 **differs** | 422 → 422 match | 1.110 → 1.105 match | 2.410 → 2.227 **differs** |
| **patellofemoral pain syndrome** | 61 → 61 match | 76 → 76 match | 35.78 → 35.74 **differs** | 178.20 → 178.19 match | 69.95 → 69.95 match | 142 → 142 match | 1.230 → 1.213 **differs** | 2.620 → 2.596 **differs** |
| **plantar fasciitis** | 20 → 20 match | 34 → 34 match | 45.76 → 45.76 match | 170.93 → 170.93 match | 77.79 → 77.79 match | 59 → 59 match | 1.220 → 1.174 **differs** | 2.500 → 2.432 **differs** |

**2 of 7 groups reproduce on all six subject and session counts** (M, F, age, height, mass, sessions): `achilles tendonitis`, `plantar fasciitis`. The recipe in the README is therefore substantially correct.

**Where it diverges:**

- **Every male/female count reproduces exactly** except `No Injury (age 50+)` (49→48), and 5 of 7 session counts are exact — including osteoarthritis at 247 subjects and 422 sessions. The cohort definitions are right.
- **Speeds are the largest source of failures**, sitting 0.005–0.07 m/s low in nearly every group. A systematic offset, not noise: the paper appears to average over all sessions rather than first-session-per-subject.
- **Residual mean differences are small** (age ≤0.05, height ≤0.70, mass ≤0.37) and consistent with a slightly different same-day-trial or first-session tie-break rule. Session counts miss by 1–2 in the same way (558→560, 128→127).
- **`No Injury (age 50+)` is the weakest group**, missing on every field. The age-50 boundary is evaluated against a different session than the one taken as first, which moves a subject across the boundary.

> **Getting OA to reproduce required departing from the README's stated rule.** 352 osteoarthritis sessions carry a real `SpecInjury` but a *blank* `InjDefn`. Read literally, the README's severity-first definition files them as neither injured nor uninjured and the whole OA cohort vanishes from Table 1. This inventory treats a recorded diagnosis as outranking a blank severity. Downstream labelling must do the same or silently lose 20% of the injured sessions.

### The 190.19 cm Achilles tendonitis height — resolved

**It reproduces exactly: observed 190.19 cm vs published 190.19 cm.** The published figure is faithful to the file. The file is what is wrong.

Of 52 subjects in this group, **1** carries the sentinel `Height = 999`. Masking implausible values leaves 51 real measurements spanning 155.50–189.00 cm with a mean of **174.33 cm** — entirely ordinary, and ~15.86 cm below the published value.

> A single `999` in a group of ~50 shifts the mean by about 16 cm. The paper's Table 1 did not mask it. **Treat every published group mean involving `Height` as contaminated**, and mask the sentinel before computing anything.

## 5. Inside the session JSONs

20 files sampled across injury groups, opened one at a time.

### Schema sketch

```
<root>: dict(8)
  hz_w     int | []        sampling frequency, walking
  hz_r     int | []        sampling frequency, running
  joints   dict(14)        anatomical landmarks, [3] each (static neutral)
  neutral  dict(30)        cluster markers, [3] each (static neutral)
  walking  dict(30) | []   marker trajectories, [n_frames, 3] each
  running  dict(30) | []   marker trajectories, [n_frames, 3] each
  dv_w     dict | []       descriptive variables, walking
  dv_r     dict{left,right}  descriptive variables, running -- 76 SCALARS each
```

Top-level keys are identical across all 20 sampled files: `hz_w`, `hz_r`, `walking`, `running`, `joints`, `neutral`, `dv_w`, `dv_r`.

### Are kinematics precomputed?

**No — and this is the single most important finding of phase 0.**

`dv_r` contains **76 variables per side** (`left` and `right` stored separately, identical key sets). Every one is a **scalar** — across the sample, 0 of 1520 values were arrays.

The 9 joint-angle channels do exist, but only as discrete summaries — `HIP_EXT`/`HIP_ADD`/`HIP_ROT`, `KNEE_FLEX`/`KNEE_ADD`/`KNEE_ABD`/`KNEE_ROT`, `ANKLE_DF`/`ANKLE_EVE`/`ANKLE_ROT`, each as `_PEAK_ANGLE`, `_percent_STANCE`, `_at_HS`, `_EXCURSION` (plus `_PEAK_VEL` variants).

**Nothing is time-normalized to 100 points.** The paper's "time normalized to 100 data points" describes the *output of the bundled MATLAB pipeline* (`gait_kinematics.m` → `gait_steps.m`), which is shipped in `Supplemental_materials/` but has not been run. The archive stores its *inputs*, not its outputs.

### Corrections from the full corpus (phase 1)

The two claims above were made from the 20-file sample. Streaming all 1,745 running sessions corrected both:

- **Not every value is a scalar.** 13 sessions (0.7%) carry 2 non-scalar   values each. Rare, but "0 of 1,520" was a sampling artifact.
- **Half the `dv_r` columns are permanently empty.** Of 152 (76 × 2   sides), **72 are 100% exact-zero** and 4 more are 20–41% zero; only   **76 carry data**. This is by pipeline design, not storage truncation:   `gait_steps.m` preallocates `DISCRETE_VARIABLES = zeros(77,3)` and   assigns exactly **40 rows** — matching the 40 live variables per side   observed. Re-running the pipeline does not populate the rest.

> **The dead columns are systematically the sagittal ones.** Per side the entire live sagittal content is three numbers — `HIP_EXT_PEAK_ANGLE`, `KNEE_FLEX_PEAK_ANGLE`, `ANKLE_DF_PEAK_ANGLE`. Every sagittal `_EXCURSION`, `_at_HS`, `_percent_STANCE` and `_PEAK_VEL` is empty. What survives is frontal/transverse and spatiotemporal — precisely what a single side-on camera recovers *worst*. **`dv_r` cannot support the phase 2 degradation analysis**; only the generated waveforms can.

### Array shapes and channels

`running` holds **28–30 marker channels**, each `[n_frames, 3]` — raw 3D trajectories, not angles. Left and right are separate channels (`L_thigh_1`…`R_foot_4`, `L_toe`, `R_toe`), covering 7 rigid segments: pelvis, bilateral thigh, shank and foot.

| property | observed across sample |
|---|---|
| frames per marker | min 1764, max 12000 |
| marker channels | 28, 30 |
| `joints` landmarks | 10, 14 |
| walking present | 11/20 sampled files |
| file size | min 3.2 MB, mean 16.6 MB, max 42.9 MB |

> **The marker set is not uniform.** Sampled files carry 28 or 30 marker channels and 10 or 14 `joints` landmarks. This matches the paper's note that extended anatomical markers exist for only 1,082 of 1,798 subjects. Any feature depending on the extended set silently drops a large part of the cohort — check per-session before assuming a fixed channel list.

### Sampling frequency — all 2,506 files

Scanned every file's first 4 KB in 0.3 s (0 unmatched).

| `hz_w` | `hz_r` | files |
|---|---|---|
| 200 Hz | 200 Hz | 1243 |
| 200 Hz | absent | 562 |
| absent | 200 Hz | 415 |
| 120 Hz | 200 Hz | 164 |
| 120 Hz | absent | 112 |
| 120 Hz | 120 Hz | 7 |
| absent | 120 Hz | 3 |

Both 120 Hz and 200 Hz are present, but **running is almost entirely 200 Hz**: 1,822 sessions vs **10** at 120 Hz. Walking carries most of the 120 Hz data.

> **This cross-validates the metadata exactly.** Files with running data = 1,832, matching `run_data_meta.csv`'s row count; files with walking = 2,088, matching `walk_data_meta.csv`; total 2,506 files. Every session file is accounted for in exactly the CSVs that claim it.

> Downsampling for phase 2 should target 120 Hz→30 Hz from a 200 Hz base for all but 10 running sessions.

## 6. Extraction feasibility

| quantity | value |
|---|---|
| session JSONs on disk | 2,506 |
| total extracted size | 48.25 GB |
| running sessions (`run_data_meta.csv`) | 1,832 |
| unique subjects with running data | 1,402 |
| mean parse time per file | 0.28 s |
| **projected full streaming pass** | **~12 min** single-threaded |
| target waveform array | `(1832, 9, 101)` float32 — see below |
| that array on disk | **6.7 MB** — trivially small |

**Tractable, with one caveat that is not about size.** The 6.7 MB target array cannot be read out of the archive: it must be *computed* from 48 GB of raw marker trajectories by running the bundled MATLAB pipeline, or a faithful Python reimplementation of it. That work is not yet scoped and belongs to a spec of its own before phase 1 begins.

> **Correction to `CLAUDE.md`: the waveforms are 101 points, not 100.** `gait_steps.m:676` allocates `zeros(101, n_steps, 3)` and interpolates over `0:(TO-TD)/100:(TO-TD)` — 0 to 100 inclusive. It also emits more than 9 channels: ankle, knee, hip, foot and pelvis, each 3-plane, per side, plus matching velocities. The 9-channel sagittal-relevant subset is a choice made downstream, not the pipeline's native output.

MATLAB R2026a is installed locally; the pipeline was written for R2023a.

### Extraction notes for whoever repeats this

The outer zip is **stored** (not compressed), so its entries are a byte copy. The nested `ric_data.zip` is **Deflate64** (method 9). Verified to fail: Python `zipfile` (`NotImplementedError`), .NET `System.IO.Compression` (*unsupported compression method*), bsdtar/libarchive 3.8 (*Damaged Zip archive*). The Windows shell zip handler does support it and needs no install — see `scripts/extract_ric_data.ps1`. The archive's internal root folder is `reformat_data/`, renamed to `ric_data/` on extract.

### Parse failures

None. Every sampled file parsed as valid JSON.

## 7. Gate questions

**1. Real column names and encodings?** 26 columns, listed in §2. `speed_r` not `speed_w(r)`; `InjDefn` is free text not 1–4; missingness is spelled four ways (`''`, `NaN`, `null`, `N/A`).

**2. Unique subjects with running data, healthy vs injured?** **1402** subjects over 1832 sessions — injured **988**, uninjured **363**, unknown **71**. (Subjects can appear in more than one status across sessions.)

**3. Does Table 1 reproduce?** Substantially — see §4 for the cell-by-cell verdict. Every group reproduces on male/female and session counts bar one (`No Injury (age 50+)`), including osteoarthritis at 247 subjects / 422 sessions. Speeds carry a systematic low offset. The 190.19 cm Achilles height **does** reproduce, and is inflated by a `Height = 999` missing-data sentinel the paper did not mask; the real mean is ~175 cm.

**4. What is in the JSONs, are kinematics precomputed?** Raw 3D marker trajectories (28–30 channels × n_frames × 3) plus 76 **scalar** descriptive variables per side. **Kinematics are not precomputed and no waveform is time-normalized.** The waveform array must be generated from raw markers, and is 101 points per stance phase, not 100.

**5. Is `InjSide` usable for phase 5?** Qualified yes. **842** injured sessions carry a unilateral `Left`/`Right`. `Bilateral` (+3 spelled `Bi-lateral`) is not a side and must be excluded or modelled separately.

---

## 8. Pose and 3D datasets (measurement track, phases 12–16)

Sections 1–7 describe the Ferber archive. The measurement track uses a different
set of datasets, and `CLAUDE.md` requires each one's licence, units and
coordinate conventions recorded **here**, before its first number is reported.

That rule was written in phase 12, after phases 3–11 had already reported
numbers from the two datasets below, so **these two sections are retrospective**.
Every dataset added from here on gets its section before first use.

Everything below was measured on 2026-10-13 from the files on disk. Nothing is
copied from the papers.

### 8.1 AthleticsPose — role: **tune**

| | |
|---|---|
| Source | Suzuki, Tanaka, Yeung & Fujii (2025) — arXiv 2507.12905 |
| Licence | **CC BY-NC-SA 4.0**, SPDX `CC-BY-NC-SA-4.0` — `data/athleticspose/data/LICENSE-DATASET` |
| Redistribution | Non-commercial, share-alike. **Source video is not released** — keypoints and derived metrics only |
| On disk | `data/athleticspose/` — 2.3 GB, extracted |
| Capture rate | **100 Hz** (`Timebase.Frequency`, all 14 QTM exports; cameras are Qualisys Miqus Video) |
| Subjects | **24**, `S00`–`S23` |
| Cameras | **8**, indices 0–7 |
| Trials / clips | **500** trials; **3,998** per-camera clips (two of 4,000 absent) |
| Grouping key | **subject** — see the warning below |
| Role | **tune**. The development set; the frozen phase 7 baseline lives here |

**Events, subjects and clips.** Eight events. The phase 3/7/12 action set is
`running` + `sprint`, which is 15 distinct subjects.

| event | subjects | clips | subject ids |
|---|---|---|---|
| discus | 2 | 144 | S00 S02 |
| hurdle | 4 | 688 | S01 S17 S18 S20 |
| javelin | 1 | 112 | S03 |
| racewalk | 2 | 312 | S05 S10 |
| running | 6 | 1,272 | S07 S09 S11 S13 S16 S19 |
| sd | 9 | 912 | S04 S06 S08 S12 S14 S15 S21 S22 S23 |
| shotput | 3 | 144 | S00 S02 S03 |
| sprint | 9 | 414 | S04 S06 S08 S12 S14 S15 S21 S22 S23 |

> **Subjects are not nested within event, and grouping by event leaks.** S00 and
> S02 appear in both `discus` and `shotput`, S03 in both `javelin` and
> `shotput`, and the `sd` and `sprint` cohorts are the *same nine subjects*.
> Group by **subject**. Grouping by trial also leaks, because 8 cameras share one
> trial.

**Directory layout and array shapes.**

```
data/athleticspose/data/AthleticsPoseDataset/
  camera_params/              14 .json    QTM export, 8 cameras each
  det_markers2d_by_cam_coco/  3,998 .npy  (T, 17, 3) float32
  det_markers2d_by_cam_ft/    3,998 .npy  (T, 17, 3) float32
  gt_bboxes2d_by_cam/         3,998 .npy  (T, 4)     float64
  gt_markers2d_by_cam/        3,998 .npy  (T, 17, 3) float64
  gt_markers3d_by_cam/        3,998 .npz  keys: markers_h36m (T,17,3), p2mm (T,)
  raw_markers_in_world/         500 .npy  (T, 84, 3) float64
```

Naming is `<modality>/<event>/<Snn>/<YYYYMMDD>_<trial>_<cam>.{npy,npz}`.
`raw_markers_in_world` has no camera index — it is one file per trial.

**Units, and the `p2mm` trap.** `markers_h36m` is in **PIXELS**, not
millimetres. Each clip carries a per-frame `p2mm` factor and the released
evaluator *divides* by it, uniformly on all three axes
(`linghtning_module.py`: `y_sample / p2mm_sample[:, None, None]`). Reading the
raw values as mm understates MPJPE about 3x and makes side-on views look
near-frontal. **This cost two wrong claims across two reports** — see
`results/phase03b.md`. Joint *angles* are unaffected, being scale-invariant.

`raw_markers_in_world` is **float64 world millimetres**, zero NaN, with **z
vertical** (one running trial spans z = 1.6 to 1793.9 mm, floor to head). It
needs no `p2mm` and is the only metric 3D in the dataset.

**The 84-marker mocap set is fully labelled** in the vendored upstream code:
`data/athleticspose/repo/athleticspose/statics/joints.py:3-88` names all 84, and
`joint_index_h36m_to_mocap` (same file, lines 177-195) gives the 84 → H36M-17
mapping that `mocap_to_h36m()` (`athleticspose/utils.py:53-71`) applies.

Three facts about that mapping, measured rather than assumed:

- **`RKNE`/`LKNE` and `RANK`/`LANK` are already joint centres** — exactly the
  midpoints of their medial/lateral pairs, to 0.000 mm. So knee and ankle carry
  no convention ambiguity, and shank length is identical in both conventions.
- **The hip does not match.** H36M defines the hip as `mean(RASI, RPSI)`, an
  iliac-surface midpoint. The lab's own hip joint centre (`RHIP`/`LHIP`) sits
  **81.07 mm** away, and the offset is rigid within the pelvis (max deviation
  equals mean deviation across frames).
- **That offset inflates the femur by +18.2%.** Anatomical femur measures
  413.4 mm (R) / 420.7 mm (L) on one trial — inside the 390–460 mm anatomical
  range — while the same trial in H36M convention gives 488.7 / 497.3 mm.
  **This explains the 479 mm femur recorded as unexplained in
  `results/phase03b.md`.** It is a definitional offset, not an error. Any
  comparison of lifter output to mocap lengths must use the H36M convention on
  both sides, or it charges the method for a convention mismatch.

Across the 15 running + sprint subjects, anatomical femur spans **344–421 mm**,
tibia **367–423 mm**, ASIS width **278–310 mm**.

**Trial and clip lengths — there is no calibration window.**

| | trials | frames (min / median / max) | seconds at 100 Hz |
|---|---|---|---|
| `running` raw trials | 159 | 40 / 73 / 174 | 0.40 / 0.73 / 1.74 |
| `sprint` raw trials | 52 | 23 / 69 / 89 | 0.23 / 0.69 / 0.89 |
| `running` per-camera clips | 1,272 | 25 / 68 / 174 | 0.25 / 0.68 / 1.74 |
| `sprint` per-camera clips | 414 | 10 / 48 / 89 | 0.10 / 0.48 / 0.89 |

> **No trial reaches 2 s.** `CLAUDE.md`'s phase 12 describes the scan as "2–3 s
> standing side-on + the first strides". No such footage exists in this dataset —
> these are short athletic efforts, not calibration passes. Phase 12 therefore
> defines a scan as **one whole trial** and records the deviation. A standing
> calibration pass cannot be validated on this dataset at all.

**Open discrepancy — capture rate.** This dataset is **100 Hz**, but
`scripts/video_kinematics.py:61` sets `MODEL_FPS = 120.0` and resamples input
video to it, commented as "the rate the lifter was trained at". The lifter was
trained on this dataset. `docs/AGENT_CONTEXT.md` lists the 120 figure as inferred
from stride cadence and "not documented anywhere" — it *is* documented, here, as
100 Hz. Phases 3 and 7 are unaffected: neither `phase3_infer.py` nor
`phase7_inference_fix.py` resamples at all, so **the 3.4° / 3.43° / 3.18°
figures do not depend on this**. Only the video tool's path does.

### 8.2 AthletePose3D — role: **tune** (demoted from held-out, 2026-10-13)

| | |
|---|---|
| Source | AthletePose3D — independent lab, 4 calibrated cameras |
| Licence | **Non-commercial research only** |
| On disk | `data/athletepose3d/` — 4.7 GB, **partial** |
| Role | **tune**, demoted from held-out test |

**Why it was demoted.** `CLAUDE.md` says a held-out dataset is not opened until
the phase that tests on it, and that "looking at it early turns it into a tune
set; if that happens, say so in the report and demote it." Phase 11 built a
22,153-clip index over it and ran two measurements on it. By the rule's own
logic it has been opened, so it is recorded here as tune. Phase 15's held-out
claim rests on MoVi, the dual-system set and Bath — none of which has been
opened or downloaded.

**What is on disk, and what is missing.**

```
pose_3d.zip               1.46 GB   complete; read in place, never extracted
data.zip*.part            3.26 GB   TRUNCATED — the raw video. 17 attempts, zero progress
cam_param.json              24 KB   complete
(pose_2d.zip)                       refused outright from the first attempt
(model_params/)                      deliberately purged — AP3D weights would be train-on-test
```

The video never arrived, which is why **§7.5 of `results/phase11.md` is still
open: the video tool has never run on real footage.** The blocker is Google
Drive per-file share state, not size — `pose_3d.zip` downloaded without trouble.

**Retry outcome, 2026-10-19 (one attempt, not a loop).** The September failure
was read as a transient Drive share quota -- its error said "it may take up to
24 hours". 35 days later a single fresh attempt
(`.venv/Scripts/python.exe scripts/phase11_fetch_ap3d.py`, logs
`data/athletepose3d/fetch5.log` / `.err`) still failed, and the error is now a
different one: "Cannot retrieve the public link of the file. You may need to
change the permission to 'Anyone with the link', or have had many accesses."
`results/phase11_fetch.json` now records `quota_limited: false` for both
`data.zip` and `pose_2d.zip`. The `.part` file is unchanged at 3,499,098,112
bytes. **Conclusion: this is not a quota that expires.** Either the share's
permissions changed or the per-file access state is persistent; from outside the
share the two are indistinguishable. Treat AP3D raw video as unobtainable by
this route, and do not schedule further retries -- the next move, if the video
is wanted, is to ask the authors directly. `pose_3d.zip` remains complete and
usable, so AP3D is still the phase 11 lifting-path test set; only the video,
and therefore only §7.5, is blocked.

**Units trap.** Clip records carry `data_input` and `data_label`, and
`data_label` is **image-normalised perspective 3D** — usable for angles, not for
positions or lengths. The metric alternative is `joint_3d_camera` in the
per-frame records of `train.pkl` / `valid.pkl` inside the same zip; phase 11
measured thigh 503 mm and shank 337 mm from it. No code reads it yet —
`scripts/phase11_clip_index.py` loads only `joint_3d_image`.

**Running cohort** (`results/phase11.md`, index at
`data/derived/ap3d_clip_index.json`): 1,872 clips over **3 subjects** at 120 fps
— S2 222, S3 693, S4 957. Content-matched at worst per-coordinate error
5.53e-08 against a 1e-05 tolerance; 21,529 of 22,153 clips matched.

### 8.3 Datasets named but not present

`CLAUDE.md`'s dataset table lists eight. Checked 2026-10-13:

| dataset | role | on disk |
|---|---|---|
| AthleticsPose | tune | yes, complete |
| AthletePose3D | tune (demoted) | partial — 3D yes, video no |
| AMASS | train | no |
| BEDLAM | train | no |
| AddBiomechanics | train / reference | no |
| BML MoVi | held-out test | no |
| Dual-system gait & fitness | held-out test | no |
| Bath synchronised set | candidate held-out | no; licence and content **now verified — see 8.4** |

Licences for the six absent sets are to be recorded here **before** download,
per the standing rule. **Done for the three held-out candidates in 8.4** (2026-10-19); AMASS, BEDLAM and AddBiomechanics are still outstanding and are train-side, so they block nothing until a retrain is proposed.

---

### 8.4 The three candidate held-out sets — the acquisition spike, 2026-10-19

`CLAUDE.md` names MoVi, the dual-system set and Bath as the held-out tests, and
nothing in the repository recorded a URL, a DOI, an accession id or a download
attempt for any of them: the only record of their existence was that table. This
section is the bounded spike that fixes that, run before any download, per the
standing rule that a licence is recorded here first.

**None of these datasets was opened.** Establishing that a dataset exists, what
its licence says, and whether it contains running is not browsing its contents;
nothing was downloaded, listed, plotted or measured, so all three keep their
held-out status.

| | **BML MoVi** | **Dual-system gait & fitness** | **Bath BioCV** |
|---|---|---|---|
| Exists? | **yes** | **yes** | **yes** |
| Accession | Scholars Portal Dataverse, `doi:10.5683/SP2/JRHDRN` | Zenodo, `doi:10.5281/zenodo.19720803`, v3, 2026-01-26 | Bath Research Data Archive, `doi:10.15125/BATH-01258` |
| Paper | PLoS ONE 2021, `10.1371/journal.pone.0253157`; arXiv 2003.01888 | **none linked** — no related identifier on the record | Sci. Data 2024, `10.1038/s41597-024-04077-3` |
| Licence | non-commercial scientific research only; no copying, sharing, distributing or re-selling; **also prohibits training methods for commercial use** | **CC BY 4.0** | **All Rights Reserved**, access on request, record marked *restricted* |
| Running? | **jogging yes; "running" is _running in place_** | **yes** — one of seven movements | **yes** — CMJ, walking, running |
| Raw video? | **yes** — 2× hand-held iPhone 7 and 2× FLIR Grasshopper. *"all 800×600, 30 fps" was wrong and is corrected in §8.5*: PG is 800×600 at 30.000 fps, CP is 1920×1080 at 30.009 fps | **unconfirmed** — `03_markerless_data.zip`, 486 MB, contents not stated | **yes** — 9 machine-vision cameras, HD, 200 fps |
| Reference 3D | Qualisys, 15 cameras, 120 Hz; also released as SMPL meshes | OptiTrack, 7 cameras, 120 Hz | 15-camera optical; `raw.c3d`, `markers.c3d`, `led.c3d` per trial — **marker tracks, not joint angles** |
| Size | 9 h mocap + 17 h video + 6.6 h IMU | 1.1 GB in four archives | ~15 GB **per participant** × 15 ≈ 225 GB |
| Publishable demo footage? | **no** — no distribution | **yes**, with attribution | **no** — "not to be shared outwith the dataset itself" |

**Three corrections to `CLAUDE.md`'s table follow from this.**

1. **The dual-system set is CC BY 4.0, not CC BY-NC-ND 4.0.** The table's "ND:
   summary metrics only" restriction does not exist. Attribution is the only
   condition, and derivatives — including published demo frames — are permitted.
   This is the *only* one of the three whose video could appear in a public demo.
   Re-read the record at download time before relying on it.
2. **Bath is no longer "unverified".** It contains running, and its licence is
   the most restrictive of the three: All Rights Reserved, access by request,
   reuse limited to biomechanics and machine-vision research, and no sharing
   outside the dataset. It is usable as a held-out *test*, and it can never
   supply a frame for the demo.
3. **MoVi's "21 actions incl. jogging" needs the qualifier.** Movement (2) is
   Jogging and movement (3) is **Running in place** (arXiv 2003.01888, §2.1).
   Running in place is not running gait — there is no flight-phase translation —
   so MoVi's usable running content is jogging across a 15-camera lab volume,
   which is a small number of strides per trial. How many is unknown until the
   phase that opens it.

**What this changes about phase 15.** It is no longer blocked on whether its
datasets exist; all three do, two have raw video confirmed and the third is one
filename listing away from an answer. The ordering it implies:

- **MoVi first.** Its video is 30 fps from a *hand-held iPhone* (1920×1080, not the 800×600 assumed here — see §8.5), which is
  the deployment camera almost exactly, and it is synchronised to 120 Hz
  Qualisys mocap. That combination is what §7.5 of `results/phase11.md` has been
  waiting for since phase 11: real consumer footage **with** ground truth.
  Phase 16A's consented clip has no ground truth and cannot settle the same
  question. The cost is that jogging is not sprinting and the camera is shaky,
  both of which are the point rather than a defect.
- **The dual-system set second**, and first if its video is confirmed: it is
  1.1 GB against MoVi's tens of hours, it is CC BY, and its 30 Hz smartphone
  pairing matches the deployment case equally well. The open question is
  answerable at the start of phase 15 by listing the archive's filenames, which
  is a file listing and not a reading of contents.
- **Bath last.** 200 fps HD from 9 calibrated cameras is the highest-quality
  reference of the three and the furthest from a phone, 225 GB will not fit the
  laptop as a whole (a per-participant archive will), and `c3d` marker tracks
  need the processing step that phases 1C and 10 already built for Fukuchi.

**Not resolved, and left to phase 15 deliberately:** whether any camera in MoVi
or the dual-system set is side-on to the direction of travel. Neither source
states camera azimuth relative to the subject, and MoVi's floor plan is a figure
in the paper rather than a table. Side-on is the geometry this project
prescribes, and phase 11 measured error falling 3.50° → 1.77° as a view
approaches it, so a frontal-only dataset would be a weak test. Answering it
requires looking at the data, which is what phase 15 is for.

### 8.5 BML MoVi — role: **held-out test**, opened 2026-11-02 for phase 15

**This section was written and committed BEFORE any MoVi data byte was
downloaded**, per the standing rule that a licence is recorded here first. The
phase 15 pipeline was frozen earlier still, in `64fb0bc`, so the ordering
*freeze → licence → download* is checkable in `git log` against the file
timestamps rather than asserted. The six verification checks below are filled in
after download and before the first number is reported.

#### Accession

| | |
|---|---|
| Repository | Scholars Portal Dataverse (borealisdata.ca), BioMotionLab Dataverse, York University |
| DOI | `10.5683/SP2/JRHDRN` |
| Paper to cite | Ghorbani, Mahmood, Troje et al., *MoVi: A large multi-purpose human motion and video dataset*, PLoS ONE 16(6) 2021, `10.1371/journal.pone.0253157` (also arXiv 2003.01888). The licence states that citing the dataset URL instead of the publication is **not** compliant |
| Files in the record | **1,057**, 290.2 GB total |
| `fileAccessRequest` | **false** — and 0 of 1,057 files are restricted |

**The README's access procedure is stale.** `README.pdf` (dated 2020-03-29,
downloaded here as documentation only) says access is *"limited"*, requires an
institutional-email Dataverse account, and must be *"requested"* and granted by
the lab (§4, *Requesting Access*). That is no longer how the record behaves:
`fileAccessRequest` is `false`, no file is marked restricted, and HTTP range
probes of 1 KB against datafiles 85637, 92073 and 193807 each returned **206
with 1,024 bytes** without any credential. Access is open. The licence below
still binds regardless, because it binds on *use*, not on the access mechanism.

#### Licence — non-commercial research, and no distribution

Verbatim from the record's Terms tab (`termsOfUse`, retrieved 2026-11-02; the
README does not contain the licence, it only points at this tab):

> By downloading and/or using the Data, you acknowledge that you have read these
> terms and conditions, understand them, and agree to be bound by them.

The grant is *"non-exclusive, non-transferable, free of charge"*, for
*"non-commercial scientific research, non-commercial education, or
non-commercial artistic projects"*, and it does permit *"modify, adapt,
translate or create derivative works based upon the Data"*. Three clauses bind
this project:

1. **No distribution.** *"The Data and the License herein granted shall not be
   copied, shared, distributed, re-sold, offered for re-sale, transferred or
   sub-licensed in whole or in part, except that you may make one copy for
   archive purposes only"*, and *"The Data may not be reproduced, modified
   and/or made available in any form to any third party without prior written
   permission."* **So no MoVi frame, crop, overlay, render or figure-of-pixels
   is published, committed, or shown in the demo — ever.** What leaves this
   machine is summary measurement statistics and plots of *angles*, which are
   not the Data.
2. **No commercial use, including training.** *"This license also prohibits the
   use of the Data to train methods/algorithms/neural networks/etc. for
   commercial use of any kind."* Nothing is trained on MoVi in this project at
   all — phase 15 is inference with a frozen checkpoint, and the only permitted
   training sets for this track are BEDLAM and AMASS. This also reinforces
   `CLAUDE.md`'s existing no-commercial-framing rule.
3. **Citation is a licence condition**, and it must be the paper.

Human-subjects note, recorded because it is the reason the no-distribution rule
is taken literally: the licence states participants were students and staff at
Queen's University who *"gave their informed, written, consent for the
scientific analysis and publication of their video Data"*. That consent is the
lab's to exercise, not this project's.

**AMASS overlap.** Part of MoVi is included in AMASS and was processed with
AMASS's methods; using that part *"falls under the AMASS license agreement"* and
additionally requires citing Mahmood et al. 2019. `MOVI_AMASS_MOSH.zip` and
`F_AMASS.tar` are that part. Phase 15 defaults to the Visual3D export instead,
so this clause applies only if the MoSh fallback is taken.

#### The slice taken, and why only this much

290 GB is not downloadable here and most of it is irrelevant: 21 movements per
round, of which exactly one is running gait. The slice is **≈ 1.20 GB**:

| file | datafile id | MB | why |
|---|---|---|---|
| `S_V3D.tar` | 92073 | 590.8 | Visual3D-processed mocap for round S — the reference 3D |
| `S2_PG1_Subject_45_L.avi` | 85637 | 94.5 | Point Grey, calibrated and mocap-synchronised: these carry the MAE |
| `S2_PG1_Subject_74_L.avi` | 85559 | 93.7 | " |
| `S2_PG1_Subject_80_L.avi` | 85654 | 97.6 | " |
| `S2_CP1_Subject_22.mp4` | 86001 | 167.9 | hand-held iPhone 7 — the §7.5 question |
| `S2_CP2_Subject_22.mp4` | 86078 | 153.0 | " |

Fetched with `GET https://borealisdata.ca/api/access/datafile/<id>` into
`data/movi/raw/` (gitignored, verified with `git check-ignore -v`). Fallback, if
the Visual3D export turns out not to carry the joints needed:
`MOVI_AMASS_MOSH.zip` (id 193807, 954.6 MB), which is shipped ground truth and
so permitted, but is a different convention and pulls in the AMASS clause above.

#### Naming, read out of the README

`<Round>_<CameraType>_<Subject>`, with Round in `{F, S1, S2, I1, I2}` and
CameraType in `{CP1, CP2, PG1, PG2}`. **CP = Camera Phone** (hand-held
iPhone 7), **PG = Point Grey** (FLIR machine vision). Subjects 1–90. The `_L`
suffix on the PG `.avi` files is **not explained anywhere in the README** — it
is recorded here as unexplained and is not guessed at; if it turns out to encode
something that matters (a camera side, a compression variant) that goes in this
section before any number.

#### Two facts established from metadata alone, both with consequences

1. **`Camera Parameters.tar` (id 92085, 0.1 MB) contains intrinsics and
   extrinsics for PG1 and PG2 only — zero CP entries.** The hand-held phone
   cameras ship **no calibration**. This fires the clause already frozen into
   `results/phase15.md`: CP clips carry an MAE only if their synchronisation to
   mocap can be verified from the dataset's own documentation, and **if it
   cannot, they are reported as traces with no error figure.** §7.5 then closes
   on the weaker claim rather than on a fabricated alignment.

   **And it cannot be verified, for a second and independent reason found
   2026-11-02:** `flags30` is indexed to the **PG** recording. CP is a different
   recording of the same session — 1920×1080 at 30.009 fps with 4,369
   frames against PG's 800×600 at 30.000 fps with 3,441 — so CP frame
   *k* is not the same instant as PG frame *k*. Applying the PG window to CP
   compared two unrelated signals and produced r = −0.077 hip, +0.022 knee.
   Recovering the offset by cross-correlation would be *estimating* the
   synchronisation, which the pre-registration says disqualifies the MAE. So
   **`scripts/phase15_movi.py` refuses to compute an MAE for an uncalibrated
   camera at all**, structurally, and `tests/test_phase15_refusal.py` pins it.
2. **Movement (2) is Jogging; movement (3) is _Running in place_**
   (arXiv 2003.01888 §2.1). Running in place is not running gait — no
   flight-phase translation, and a stationary subject makes the sagittal camera
   geometry a different problem. It is excluded, with its n reported.

#### The six checks — filled in after download, before the first number

Pre-registered in `results/phase15.md` and reproduced here so the answers live
with the dataset:

| # | check | answer (round F) |
|---|---|---|
| 1 | mocap units | **millimetres** — femur, tibia and hip width all inside their pre-registered human ranges (350–550, 300–500, 120–350 mm) for every subject checked. **The per-subject lengths are not reported**, see the note below |
| 2 | vertical axis and handedness | **vertical is axis 2**, agreed two independent ways: both PG camera centres sit at 1.35–1.38 m on that axis, which has to be a mounting height, and head-minus-pelvis points along it. Not mirrored — jogging knee flexion is `[-73.5, -8.7]°`, one-signed and inside the tool's own plausible band |
| 3 | marker set → H36M-17 | **stated in `MOVI_TO_H36M`** (`scripts/phase15_checks.py`). Hips are `mean(LFWT, LBWT)` / `mean(RFWT, RBWT)` = `mean(ASIS, PSIS)`, which is H36M's own definition; knees `LKJC`/`RKJC`, ankles `LAJC`/`RAJC`, thorax `THORAX`, spine the pelvis–thorax midpoint. Verified against the projected reference joint by joint: every H36M joint matches the COCO slot it should, 11.8 px mean |
| 4 | video frame rate | **read from the file, and the two camera families differ.** PG: **800×600 at 30.000 fps**, 3,441 frames for Subject_1. CP: **1920×1080 at 30.009 fps**, **4,369 frames** — 928 more than PG. `CLAUDE.md`'s "all 800×600 30 fps" was wrong and is corrected |
| 5 | synchronisation | **shipped by the dataset, not estimated.** `flags30` and `flags120` are the same movement boundaries in video and mocap frames, ratio **3.9981** (max deviation from 4: 0.0112). Video frame *k* ↔ mocap frame `f120[0] + 4k`, so the reference 3D is never resampled |
| 6 | subject identity | **matches** — the `.mat` subject field equals the filename's subject for every subject checked |

#### What is deliberately NOT reported from MoVi, and why

The licence forbids reproducing the Data *"in any form to any third party"*, and
**this repository is public**. So three classes of value are computed, tested
and then **not emitted**:

1. **Per-participant attributes** — `sex`, `height`, `mass` and `age` are in the
   Visual3D `subject` struct and are **not read at all** by
   `scripts/phase15_checks.py`. No check needs them.
2. **Per-subject segment lengths in millimetres.** Check 1 exists to answer
   "millimetres or metres", and a pass against a pre-registered human range
   answers that exactly as well as publishing the femur length of three
   identifiable people. The values are computed and tested; only the verdict is
   written out.
3. **Camera intrinsics and extrinsics**, which are verbatim content of
   `Camera Parameters.tar`. What is reported is the *inference* — that both PG
   rigs sit at a mounting height, which is what identifies the vertical axis —
   plus `view_ratio`, the travel azimuth, and the subject distance rounded to
   the metre. Not the coordinates.

What *is* reported is measurement of the method: angle MAE, correlations,
`view_ratio`, frame counts, exclusions, and the H36M-hip-to-joint-centre offset,
which is a property of a *convention* rather than of a person and is the finding
that replicates phase 12's 81 mm.

**Two traps, both recorded because both would have produced a wrong number
silently.**

1. **`markerGaps == 1` marks a missing marker, and Visual3D writes `[0, 0, 0]`
   for it.** A sentinel, not a measurement — the same family as Ferber's `999`
   and the `dv_r` zeros. Read literally it placed the knee joint centre at the
   lab origin and drove jogging knee flexion to **+148.7°**. Gapped markers are
   set to NaN and the frames they touch are **excluded, never interpolated**.
   The exclusion is **per quantity**: `sagittal_angles` reads seven joints, so a
   gap in an elbow or wrist must not cost a knee frame. Scoped globally it would
   discard 21.2% of subject 1's jogging window; scoped correctly it is 3.9% left
   and 4.5% right, and subject 10 loses nothing at all.
2. **`LHJC`/`RHJC` are the _hand_ joint centres, not the hip.**
   `markerDescription` reads "Left Hand Joint", and their height matches the
   elbow. The hip joint centres are **`LHIP`/`RHIP`**. Using the wrong one put
   the H36M-hip offset at 325 mm; the right one gives **104/110 mm** (subject 1)
   and **90/92 mm** (subject 10), which independently reproduces the **81 mm**
   phase 12 measured on AthleticsPose for the same definitional offset.

**`motions_list` order is randomised per subject.** Jogging is index 18 for
subject 1 and **16** for subject 10; `running_in_spot` moves from 7 to 19. The
index must be read per subject — a hardcoded one silently measures a different
movement.

**The side-on question, which `CLAUDE.md` says phase 15 must answer first.**
Both metrics agree on both subjects: **PG1 is side-on** (`view_ratio` 0.720 and
0.705 against AthleticsPose's own median of 0.751; travel azimuth 75.7° and
75.0°, IQR 62–84, where 90° is side-on), and **PG2 is near-frontal** (0.468 and
0.465; azimuth 11.0° and 10.5°). So the held-out set carries the geometry this
project prescribes *and* a frontal contrast camera.

Any check that fails and cannot be resolved from the dataset's own documentation
**stops the MAE for the affected clips** rather than being worked around.

#### Which subjects are opened, and which are still a clean test set

**Subjects 1, 10 and 11 are spent as a test set.** They have been measured,
their per-clip errors inspected, their gap patterns read, their check verdicts
acted on, and one of them excluded on those grounds. Any fix evaluated on them
from here is being evaluated on data it was shaped by. They remain the phase 15
*result*; they are no longer an independent test.

| subject | status |
|---|---|
| `Subject_1`, `Subject_10`, `Subject_11` | **opened and examined in detail** — phase 15 |
| round S `Subject_22`, `_45`, `_74`, `_80` | **structure inspected only** (marker names, field layout) while establishing that round S is unusable. No round-S angle was computed, and their **round F data has never been touched** |
| every other round F subject | **never opened** |

**A fresh set, costed from Dataverse metadata only — nothing downloaded.**
Establishing file ids and sizes is the same kind of lookup §8.4 used, which
`CLAUDE.md` says is not "opening" a held-out set.

The mocap lives in one 1,814.6 MB tar, `F_Subjects_1_45.tar` (id 128299), whose
members sit in **lexicographic** order — `Subject_1`, `Subject_10`, `Subject_11`,
`Subject_12`, `Subject_13`, `Subject_14`, `Subject_15`, … — so a ranged GET of a
prefix reaches the early ones cheaply and the rest of the archive is never
transferred. Measured member sizes: 32.6, 43.9 and 52.4 MB for the first three;
the archive mean is 40.3 MB.

**`Subject_14` has no round F video at all** — no `F_PG1`, `F_PG2`, `F_CP1` or
`F_CP2` entry exists for it, consistent with the README's 89-of-90 count for
PG1. It is skipped.

So the next three usable subjects in archive order are **`Subject_12`,
`Subject_13`, `Subject_15`**:

| item | size |
|---|---|
| mocap: one ranged prefix reaching the 7th member | **~290 MB** of 1,814.6 MB |
| `F_PG1_Subject_12_L.avi` (id 85270) | 247 MB |
| `F_PG1_Subject_13_L.avi` (id 85234) | 260 MB |
| `F_PG1_Subject_15_L.avi` (id 85255) | 234 MB |
| **total, side-on camera only** | **~1,030 MB** |
| adding `F_PG2` 12/13/15 (ids 85338, 85336, 85342) | +1,036 MB |
| **total, both cameras** | **~2,066 MB** |

At the 6.5–8.7 MB/s this record serves, PG1-only is roughly **2–3 minutes** of
transfer. Inference is the real cost: at 540–1,060 ms/frame and ~200–450 jogging
frames per subject, three subjects is **15–40 min** of CPU per arm.

**Recommendation, not yet acted on:** take **PG1 only** for the fresh set.
PG2 doubles the bytes and the CPU to answer a *different* question (whether the
viewpoint curve replicates), and that question deserves its own deliberate run
rather than riding along. **Nothing has been downloaded.**

#### Role

`CLAUDE.md` assigns MoVi **held-out test**, and this is the phase that opens it.
The acquisition spike in §8.4 established existence and licence from outside the
data, which is not opening it; downloading and measuring is, and that is phase
15's job. Nothing is tuned on these numbers — `results/phase15.md`
§"What would make this phase wrong" governs.

---

## Deferred to later phases

- ~~**Generating the waveforms** from raw markers via the MATLAB pipeline.~~ **Done in phase 1C** — the vendored pipeline reproduces the archive's own `dv_r` exactly, so the 101-point curves it emits are verified. See `results/phase01c.md`.
- ~~Whether `0` in `dv_r` means zero or missing.~~ **Resolved** — 72 of 152 columns are never written by `gait_steps.m` at all. Sentinel, not measurement. See §5.
- ~~Which sessions carry the extended marker set, and what that costs in n.~~ **Answered** — 28-channel sessions exist only in 2014–2016 (876 sessions); every session before 2014 has 30. Marker set is very nearly a collection-year label, and is a confound rather than a feature.
- Whether the 12 osteoarthritis subjects with running data are usable at all, given `CLAUDE.md` requires OA be reported separately.
- Reconciling `SpecInjury` free text into a usable label set beyond the README's five merges.
- `walk_data_meta.csv` is characterized here only where Table 1 needs it.

