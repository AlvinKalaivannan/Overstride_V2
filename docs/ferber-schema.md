# Ferber / Running Injury Clinic Kinematic Dataset — Schema

**Status: paper-derived. These are assertions to verify against, never values to
reproduce or fill in with.** Where this document and `docs/data-inventory.md`
disagree, **the inventory wins** — it is observed, this is derived.

Source: Ferber, R. et al. *A large-scale biomechanical dataset of walking and running
gait.* Scientific Data **11**:1232 (2024). https://doi.org/10.1038/s41597-024-04011-7
Data: Figshare+, CC BY 4.0. Local copy of the paper: `docs/s41597-024-04011-7.pdf`.

---

## 1. Provenance and scale

Collected at the University of Calgary Running Injury Clinic, 2009–2017, as research
studies and clinical practice. Ethics: CHREB E–21705, E–22194, E–24339.

| Quantity | Paper's figure |
|---|---|
| Unique subjects | 1,798 |
| Session JSON files | 2,506 |
| Metadata CSVs | 2 (`run_data_meta.csv`, `walk_data_meta.csv`) |
| Pain-free at testing | n = 396 subjects |
| Injured at testing | n = 1,402 subjects |
| Walking data only | 396 subjects |
| Running data only | 112 subjects |
| Both walking and running | 1,290 subjects |
| Previously unpublished | n = 1,197 (67%) |
| Previously published | n = 601 (33%) |

> **Trap, confirmed by phase 0.** The healthy/injured split (396 / 1,402) is
> numerically identical to the walk-only / has-running split (396 / 112 + 1,290).
> The run metadata does contain exactly 1,402 subjects — but they are split
> 988 injured / 363 uninjured / 71 unknown, not 1,402 / 0. The two figures are
> different quantities that coincide. Never treat "has running data" as "injured".

## 2. Collection protocol

- Vicon MX3 or Bonita, **3 or 8 cameras**, treadmill (Bertec).
- **Sampling: 120 Hz or 200 Hz.** Both are present in the archive.
- Walking at self-selected 0.4–1.9 m/s; running at self-selected 1.1–4.9 m/s.
  20–60 s of consecutive strides each.
- Core marker set (all 1,798 subjects): medial/lateral malleoli, medial/lateral
  femoral condyles, greater trochanters, bilaterally.
- Extended anatomical markers for **1,082 of 1,798** subjects only: 1st/5th metatarsal
  heads, distal shoe, tibial tuberosity, ASIS, iliac crests.
- Tracking: rigid clusters of 3–4 markers over **seven segments** — sacrum, bilateral
  thigh, bilateral shank, both shoes. ISB-compliant.
- Anatomical markers were removed after a 1 s static neutral trial; only clusters
  remained during motion.
- All participants wore the same shoe model (Nike Pegasus).

## 3. Processing pipeline (bundled MATLAB, tested on R2023a)

Segment kinematics from cluster motion via singular-value decomposition and a joint
coordinate system. 3D marker data filtered at **10 Hz, 4th-order Butterworth**. Hip,
knee and ankle angles as **Cardan angles**, distal relative to proximal, order:
flexion/extension (M-L axis) → abduction/adduction (A-P axis) → internal/external
rotation (vertical axis).

Foot strike and toe off detected by a PCA approach (`pca_td.m`, `pca_to.m`). Data are
partitioned to the **stance phase** and **time-normalized to 100 points**.

| File | Role |
|---|---|
| `gait_kinematics.m` | raw markers → joint angles and velocities |
| `gait_steps.m` | angles/velocities → normalized time-series curves + discrete variables |
| `gaitClass.m` | discriminant: is this trial walking or running |
| `pca_td.m` / `pca_to.m` | touchdown / toe-off event detection |
| `processing_code_example.m` | wrapper demonstrating the full pipeline |

> **This is the critical point for phase 1.** The 100-point normalized waveforms are
> an *output of this pipeline*, not a stored field. See §6.

## 4. File layout and fields

### `$DATA_ROOT/ric_data/<sub_id>/<timestamp>.json`

| Key | Contents |
|---|---|
| `hz_w` | sampling frequency, walking |
| `hz_r` | sampling frequency, running |
| `joints` | 3D coords of anatomical markers at neutral stance (not true joint centres) |
| `neutral` | 3D coords of cluster markers at neutral stance |
| `walking` | 3D cluster marker trajectories, walking |
| `running` | 3D cluster marker trajectories, running |
| `dv_w` | descriptive variables computed from walking |
| `dv_r` | descriptive variables computed from running |

A session contains walking and/or running; either may be empty.

### `run_data_meta.csv` / `walk_data_meta.csv`

`sub_id`, `datestring`, `filename`, `speed_w(r)`, `age`, `Height` (cm), `Weight` (kg),
`Gender`, `DominantLeg`, `InjDefn`, `InjJoint`, `InjSide`, `SpecInjury`, `InjDuration`
(days), `InjJoint2`, `InjSide2`, `SpecInjury2`, `Activities`, `Level`, `YrsRunning`,
`RaceDistance`, `RaceTimeHrs`, `RaceTimeMins`, `RaceTimeSecs`, `YrPR`, `NumRaces`.

`InjDefn` is injury severity, selected from one of four options: *No Injury*;
*Continuing to train in pain*; *training volume/intensity affected*; *2 workouts missed
in a row*. `InjJoint` / `InjSide` are `No Injury` or empty when not applicable.

> **Uninjured, per the dataset README:** `InjDefn` = 'No injury' **and** `InjJoint` ∈
> {'No injury', empty/NULL/N/A} **and** `SpecInjury` empty. `InjJoint` may contain
> the value `'No Injury, no injury'`, originally used to mean "no *secondary* injury".

## 5. Table 1 as published — the verification target

Demographics for non-injured participants (split by age) and the top 5 injuries.
Male/Female are **subject** counts; No. Sessions is a **session** count.

| Injury Status | Male | Female | Age (yrs) | Height (cm) | Body Mass (kg) | No. Sessions | Walk Speed (m/s) | Run Speed (m/s) |
|---|---|---|---|---|---|---|---|---|
| No Injury (age 18–49) | 137 | 171 | 32.52 | 172.19 | 69.54 | 558 | 1.21 | 2.80 |
| No Injury (age 50+) | 39 | 49 | 55.80 | 165.33 | 69.89 | 130 | 1.18 | 2.58 |
| Achilles tendonitis | 30 | 22 | 42.62 | **190.19** | 77.20 | 68 | 1.30 | 2.68 |
| Iliotibial band syndrome | 39 | 61 | 35.21 | 171.99 | 67.76 | 128 | 1.26 | 2.64 |
| Osteoarthritis | 91 | 156 | 56.36 | 167.40 | 76.36 | 422 | 1.11 | 2.41 |
| Patellofemoral pain | 61 | 76 | 35.78 | 178.20 | 69.95 | 142 | 1.23 | 2.62 |
| Plantar fasciitis | 20 | 34 | 45.76 | 170.93 | 77.79 | 59 | 1.22 | 2.50 |

> **Resolved by phase 0 — it is not a transcription error.** The 190.19 cm reproduces
> exactly from the raw file. It is inflated by a `Height = 999` missing-data sentinel
> that the paper did not mask; masking it gives ~174.33 cm. See
> `docs/data-inventory.md` §4. **Mask `Height` outside [120, 250] cm before computing
> anything**, and treat every published group mean involving `Height` as contaminated.

### Reproduction recipe (from the dataset README)

To match Table 1, the README instructs:

1. Lowercase all injury strings.
2. Merge `oa`, `hip oa`, `knee oa`, `osteroarthritis` → `osteoarthritis`.
3. Merge `itb syndrome`, `itbs` → `itb syndrome`.
4. Merge `patellofemoral pain syndrome`, `pfps` → `patellofemoral pain syndrome`.
5. A subject retested with the **same** injury counts **once** in that category; a
   subject retested with a **different** injury counts once in **each** category.
   Multiple same-day trials count once.
6. Averages (anthropometrics and speeds) use only each subject's **first dated
   session** within that injury category.

## 6. Known traps

1. **The waveforms are not stored.** The JSONs contain raw marker trajectories plus
   76 *scalar* descriptive variables per side. Any waveform array must be computed by
   running the bundled MATLAB pipeline (or a faithful reimplementation) over the raw
   markers. Budget for this before planning phase 1. Note the pipeline emits **101**
   points per stance phase, not 100 (`gait_steps.m:676`).
2. **A blank `InjDefn` does not mean uninjured.** 352 osteoarthritis sessions carry a
   real `SpecInjury` with no severity recorded. Applying the README's severity-first
   rule literally drops the entire OA cohort. A recorded diagnosis outranks a blank
   severity.
2. **Rows are sessions, not subjects.** 2,506 files, 1,798 subjects. All splits group
   by `sub_id`.
3. **Two metadata files.** `walk_data_meta.csv` is unused by this project; a session
   JSON can appear in both CSVs.
4. **Secondary injury fields** (`InjJoint2`, `InjSide2`, `SpecInjury2`) are
   inconsistently populated and can carry a side with no injury.
5. **Osteoarthritis is confounded** — mean age 56.36, slowest run speed 2.41 m/s.
   Excluded from the primary analysis per `CLAUDE.md`. In practice it is also nearly
   absent from the running data: 247 subjects overall, but only **12** ran.
6. **Run speed is a confounder**, not a feature.
7. **Extended markers exist for only 1,082 of 1,798 subjects** — any feature relying
   on metatarsal, tibial tuberosity, ASIS or iliac crest markers silently drops 40%
   of the cohort. Confirmed observationally: sampled files carry 28 *or* 30 marker
   channels and 10 *or* 14 `joints` landmarks.
8. **The archive is Deflate64-compressed.** Python's `zipfile`, .NET
   `System.IO.Compression` and libarchive all fail on it.
9. **`999` is a missing-data sentinel in four columns, not just `Height`.**
   `age` (`255`×1), `Height` (`0`×3, `999`×2), `Weight` (`0`×2, `1564`×1) and
   `YrsRunning` (`999`×51). All four are demographics-control features. Mask to
   plausible ranges — [10, 100], [120, 250] cm, [30, 200] kg, [0, 80] yrs.
10. **Metadata missingness leaks the label.** Questionnaire completeness tracks
   the collection wave and the waves differ in injury mix — a blank `Level`
   marks an uninjured session with 95% precision, and a missingness flag alone
   scores AUC ≈ 0.66. Never fit missingness indicators or collection year.
   `docs/data-inventory.md` §2.
