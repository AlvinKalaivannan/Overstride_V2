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

**This is the most dangerous finding in the metadata.** Whether a field was filled in at all predicts the label, because questionnaire completeness varies by study wave and the waves differ in injury mix:

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
| **No Injury (age 18-49)** | 137 → 137 ✓ | 171 → 171 ✓ | 32.52 → 32.52 ✓ | 172.19 → 172.20 ✓ | 69.54 → 69.53 ✓ | 558 → 560 **✗** | 1.210 → 1.207 ✓ | 2.800 → 2.795 ✓ |
| **No Injury (age 50+)** | 39 → 39 ✓ | 49 → 48 **✗** | 55.80 → 55.85 **✗** | 165.33 → 165.45 **✗** | 69.89 → 70.04 **✗** | 130 → 128 **✗** | 1.180 → 1.151 **✗** | 2.580 → 2.518 **✗** |
| **achilles tendonitis** | 30 → 30 ✓ | 22 → 22 ✓ | 42.62 → 42.62 ✓ | 190.19 → 190.19 ✓ | 77.20 → 77.20 ✓ | 68 → 68 ✓ | 1.300 → 1.284 **✗** | 2.680 → 2.682 ✓ |
| **itb syndrome** | 39 → 39 ✓ | 61 → 61 ✓ | 35.21 → 35.21 ✓ | 171.99 → 171.99 ✓ | 67.76 → 67.76 ✓ | 128 → 127 **✗** | 1.260 → 1.256 ✓ | 2.640 → 2.615 **✗** |
| **osteoarthritis** | 91 → 91 ✓ | 156 → 156 ✓ | 56.36 → 56.32 **✗** | 167.40 → 166.70 **✗** | 76.36 → 75.99 **✗** | 422 → 422 ✓ | 1.110 → 1.105 ✓ | 2.410 → 2.227 **✗** |
| **patellofemoral pain syndrome** | 61 → 61 ✓ | 76 → 76 ✓ | 35.78 → 35.74 **✗** | 178.20 → 178.19 ✓ | 69.95 → 69.95 ✓ | 142 → 142 ✓ | 1.230 → 1.213 **✗** | 2.620 → 2.596 **✗** |
| **plantar fasciitis** | 20 → 20 ✓ | 34 → 34 ✓ | 45.76 → 45.76 ✓ | 170.93 → 170.93 ✓ | 77.79 → 77.79 ✓ | 59 → 59 ✓ | 1.220 → 1.174 **✗** | 2.500 → 2.432 **✗** |

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

## Deferred to later phases

- ~~**Generating the waveforms** from raw markers via the MATLAB pipeline.~~ **Done in phase 1C** — the vendored pipeline reproduces the archive's own `dv_r` exactly, so the 101-point curves it emits are verified. See `results/phase01c.md`.
- ~~Whether `0` in `dv_r` means zero or missing.~~ **Resolved** — 72 of 152 columns are never written by `gait_steps.m` at all. Sentinel, not measurement. See §5.
- ~~Which sessions carry the extended marker set, and what that costs in n.~~ **Answered** — 28-channel sessions exist only in 2014–2016 (876 sessions); every session before 2014 has 30. Marker set is very nearly a collection-year label, and is a confound rather than a feature.
- Whether the 12 osteoarthritis subjects with running data are usable at all, given `CLAUDE.md` requires OA be reported separately.
- Reconciling `SpecInjury` free text into a usable label set beyond the README's five merges.
- `walk_data_meta.csv` is characterized here only where Table 1 needs it.

