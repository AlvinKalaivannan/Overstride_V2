# Overstride

Injury-risk screening for runners from monocular video.

**The claim being tested:** how much injury-classification performance is lost
when lower-limb kinematics are recovered from a single camera instead of a
motion capture lab. The deliverable is that degradation curve, measured. A
negative result is a valid result.

**This is screening, not prediction.** Every label in the dataset describes
injury status *at the time of testing*. The word "predict" must not appear in
any output, README, docstring, or plot title. Use "detect", "classify", or
"screen".

---

> ## PROJECT STATUS: COMPLETE — the answer is a negative result
>
> **The degradation curve is measured: monocular video costs ~0.03 AUC against
> marker mocap.** It is that small because there was very little to lose. The
> only task carrying kinematic injury signal — identifying *which limb* is
> injured — tops out at **AUC 0.610** on perfect mocap and **0.583** with real
> video error, against a chance rate of 0.517.
>
> Screening failed outright (0 of 60 pre-registered tests). Within-subject limb
> identification cleared its bar but fails every deployment requirement, and
> phases 5B–5D established the ceiling is the **signal**, not the model or the
> camera.
>
> **See `README.md` for the synthesis.** The rules below still bind any further
> work on this repository; the phase table records what happened, not a plan.

---

## Non-negotiable rules

Violating any of these invalidates the project's results. If a task appears to
require breaking one, stop and raise it rather than proceeding.

### Evaluation protocol

- **All splits are grouped by `sub_id`.** Rows are sessions, not subjects; some
  subjects have multiple sessions. Use `StratifiedGroupKFold` with
  `groups=sub_id`. Never `train_test_split` on rows.
- **No subject appears in both train and test in any fold, ever.**
- **Every kinematic model reports alongside a demographics-only control**
  fitted on `age`, `Gender`, `Height`, `Weight`, `speed_r`, `YrsRunning`,
  `Level`. A kinematic result is only reportable as a delta against that
  control. If the kinematic model does not clearly beat it, say so plainly.
  **Mask sentinels first — `999` is a missing-data marker and it is in four
  columns, not one:** `age` outside [10, 100], `Height` outside [120, 250] cm,
  `Weight` outside [30, 200] kg, `YrsRunning` outside [0, 80] yrs. 51 sessions
  claim 999 years of running experience. See `docs/data-inventory.md` §2.
- **Missingness leaks the label — never fit an indicator for it, and never fit
  collection year.** Questionnaire completeness tracks the study wave, and the
  waves differ in injury mix: a blank `Level` marks an uninjured session with
  95% precision, and the 2017 wave is 51 sessions that are 100% uninjured with a
  100% blank questionnaire. A missingness flag alone scores AUC ≈ 0.66 while
  measuring paperwork, not physiology. Complete-case analysis is not a safe
  fallback either — it deletes 45% of the uninjured class. **Every phase reports
  a provenance-only baseline** (missingness flags + collection year, no
  demographics, no kinematics) to bound how much of any score is this artifact.
- **Report AUC with cross-fold confidence intervals**, never a single number.
- **All preprocessing is fit inside the fold, on train only.** Scalers,
  imputers, PCA, feature selection. No exceptions.
- **Osteoarthritis is excluded from the primary analysis** and reported
  separately. It is degenerative, mean age 56, slowest run speed — confounded
  with the demographic control. In practice the exclusion costs almost nothing:
  of 247 OA subjects, only **12** ran (14 sessions). A separate OA kinematic
  analysis is not viable at that n — say so rather than fitting one.
- **A recorded diagnosis outranks a blank severity when labelling.** The
  dataset README's uninjured rule is severity-first, and applied literally it
  files 352 sessions that carry a real `SpecInjury` with a blank `InjDefn` as
  neither injured nor uninjured — deleting the entire OA cohort and ~20% of
  injured sessions. Label as phase 0 did (`scripts/phase0_inventory.py`,
  `injury_status`), not as the README reads.
- **Run speed is a confounder, not a feature.** Healthy young runners average
  2.80 m/s, osteoarthritis 2.41. Control for it explicitly and run a
  speed-matched subsample as a secondary check.

### Data integrity

- **Never generate, synthesize, impute, or mock dataset rows.** If a file is
  missing or unreadable, stop and say so. Summary statistics in
  `docs/ferber-schema.md` are assertions to verify against, never values to
  reproduce or fill in with.
- **Never commit data.** See `.gitignore`. Data lives at `$DATA_ROOT`, which is
  currently *inside* the working tree at `data/ric/` and covered by
  `.gitignore`. Verify with `git check-ignore -v` before any commit — the rule
  is enforced by that one line, not by the data being elsewhere.
- **Never load all session files into memory at once.** 2,506 JSON files of
  marker trajectories will exhaust RAM. Stream, or sample and report.

---

## Data

Ferber / Running Injury Clinic Kinematic Dataset (n=1,798).
Full schema, column dictionary, cohort structure, and known traps:
**`docs/ferber-schema.md`** — read it before touching the data.

Location is configured, never hardcoded:

```
DATA_ROOT=/absolute/path/to/ric/     # set in .env, read via os.environ
```

Expected layout under `$DATA_ROOT`:

```
README.txt
run_data_meta.csv          # labels + covariates  (primary)
walk_data_meta.csv         # unused
Supplemental_materials/    # MATLAB processing code + tutorials
ric_data/<sub_id>/*.json   # raw markers + scalar summaries. No waveforms.
```

`docs/data-inventory.md` describes what is *actually* in these files. It exists
— phase 0 is complete (`results/phase00.md`). **Read it before the schema doc**,
and where the two disagree, **the inventory wins** — it is observed, the schema
doc is derived from the paper.

---

## Environments

Two environments, deliberately separated. The 21 GB archive never leaves the
laptop; the only thing that crosses is a feature file of roughly 10 MB.

| Phase | Environment | Notes |
|---|---|---|
| 0–2 | Laptop, CPU | No GPU available and none needed |
| 3 | ~~Kaggle Notebooks (P100/T4)~~ → **Laptop, CPU** | See deviation below |
| 4–6 | Laptop, CPU | |

Do not propose solutions requiring a local GPU or requiring the Ferber archive
to be uploaded to cloud storage.

> **Deviation, recorded.** Phase 3 was planned for Kaggle because CUDA was
> assumed necessary. Two Kaggle runs were given neither GPU nor Internet — the
> account is not phone-verified and Kaggle withholds both silently. **Phase 3 ran
> locally on CPU instead**, in ~90 min. GPU was only ever a speed convenience:
> the quantity measured (angular error between estimated and ground-truth 3D) is
> identical on CPU, and the rule above forbids solutions *requiring* a local GPU,
> which CPU inference does not. The Ferber archive was never involved.
> `docs/phase3-setup.md` and `notebooks/phase3_kaggle.*` are the superseded route.

---

## Phases

Each phase has a gate. Do not begin a phase before its predecessor's gate is
recorded in `results/`.

| Phase | Work | Gate | Outcome |
|---|---|---|---|
| **0** | Inventory: parse, join, characterize, verify against Table 1 | `docs/data-inventory.md` exists and Table 1 reproduces | ✅ |
| **1** | Injury classifier on all 9 mocap waveforms | Beats the demographics-only control | ❌ **kill criterion fired** (0/60 tests) |
| **2** | Restrict to 3 sagittal waveforms, downsample, inject keypoint noise | Degradation measured — **the headline result** | ✅ (σ labels corrected in phase 4) |
| **3** | Video → 3D kinematics via released checkpoints | Joint-angle MAE within range of published figures | ✅ 3.4° fine-tuned |
| **3B** | Viewpoint geometry; the pixel/`p2mm` unit error | — | ✅ occlusion penalty ~0 |
| **4** | Video-derived features through the phase 2 model | Real ΔAUC, not simulated | ✅ −0.027 to −0.033 |
| **4B** | Operating point, calibration, repeatability | — | ❌ not deployable |
| **5** | Personalization: `InjSide` asymmetry, within-session stride distributions | Beats the population model | ✅ 0.610 |
| **5B–5D** | Ceiling diagnostics, further feature families, abstention | — | ❌ ceiling is the signal |
| **6** | ~~Demo shell~~ → **Methods demo + synthesis** | — | ✅ `README.md` |

> **Phase 6 was redefined.** "Demo shell" assumed something worth demonstrating
> to a user. Phases 4B and 5D showed a per-user limb verdict is unsupportable —
> PPV 0.574 against a 0.517 base rate, calibration slope 0.716, and the model
> disagrees with itself on a third of repeat scans. **Building one would
> misrepresent the evidence.** Phase 6 became the honest presentation of the
> measurement instead. See `results/phase04b.md` and `results/phase05d.md`.

**Phase 1 has a kill criterion.** Build the demographics-only control *first*,
before any kinematic model. If the kinematic model cannot beat it, the gait
data carries no injury signal beyond "older, heavier, slower people are
injured," and no downstream video pipeline can rescue that. Report it and stop.

> **It fired, and the project continued deliberately.** Phases 1C/1D reported the
> failure plainly: 0 of 60 pre-registered tests survived correction, and a
> provenance-only baseline scored 0.775–0.863 while measuring paperwork. Work
> continued onto the **within-subject** limb task, which is a different question
> against a stronger control set (provenance, demographics, structure and limb
> dominance all verified at chance). That pivot is a real departure from this
> rule and is recorded in `results/phase02.md`. The video pipeline was then built
> to measure degradation of *that* signal, not to rescue screening.

Phases 0–2 require no camera, no GPU, and no pose estimation.

---

## Reporting

After every phase, write `results/phaseNN.md` containing:

- The numbers, with cross-fold CIs
- The exact split used, and the `groups=` key
- n per group after filtering, and what was excluded and why
- The demographics-only control's score alongside every kinematic score
- Anything that surprised you or looks wrong

If the split cannot be stated precisely, something went wrong — investigate
before writing results.

---

## Conventions

- Python 3.11+, `uv` for dependency management
- scikit-learn for phases 1–2. Do not reach for deep learning on ~1,400
  subjects with tabular-ish features; it will overfit and it will not be
  defensible.
- **Waveforms are derived, not stored.** The archive holds raw marker
  trajectories and *scalar* summaries only; the curves are an output of the
  bundled MATLAB pipeline (`gait_kinematics.m` → `gait_steps.m`), which has not
  been run. Generating them is an unscoped prerequisite for phase 1.
- **101** stance-phase points, not 100 (`gait_steps.m:676` allocates
  `zeros(101, n_steps, 3)`). Generated in phase 1C; the pipeline reproduces the
  archive's own `dv_r` exactly, so the curves are verified, not assumed.
- **The pipeline emits 54 channels**, and `data/derived/waveforms_mean.npy` is
  `(n_sessions, 54, 101)` float32 in this fixed order:
  - `0..29` angles — L/R × {ankle, knee, hip, foot, pelvis} × 3 planes
  - `30..53` velocities — L/R × {ankle, knee, hip, pelvis} × 3 planes
  - **Plane 2 is flexion/extension — not plane 0.** `docs/ferber-schema.md` §3
    gives the Cardan order as flex/ext → ab/adduction → rotation, which implies
    index 0; the emitted array disagrees. Measured against the live sagittal
    `dv_r` peaks: plane 2 gives |r| ≥ 0.99, plane 0 gives |r| ≤ 0.17. The sign
    convention is also inverted relative to `dv_r`. Re-check with
    `scripts/phase1c_verify_sagittal.py` before trusting any sagittal result.
- The modelling subsets are derived from that array, side-averaged:
  **`wave9`** = 3 joints × 3 planes (the "9 channels" this file used to name),
  **`wave3`** = hip/knee/ankle flexion-extension — the video-recoverable subset,
  and the phase 2 degradation ladder is `wave54 → wave9 → wave3`.
- Per-step curves are kept in `data/derived/waveforms_steps/` — phase 5 needs
  within-session stride distributions, and regenerating means re-running hours
  of MATLAB.
- Sagittal subset is channels for hip/knee/ankle flexion-extension only
- **AthleticsPose `markers_h36m` is in PIXELS, not millimetres.** Each `.npz`
  carries a per-frame `p2mm` factor and the released evaluator *divides* by it,
  uniformly across all three axes
  (`linghtning_module.py`: `y_sample / p2mm_sample[:, None, None]`). Reading the
  raw values as mm understates MPJPE ~3x and makes side-on views look
  near-frontal. **This cost two wrong claims across two reports** before it was
  caught — see `results/phase03b.md`. Joint *angles* are unaffected: they are
  scale-invariant, and every angle MAE re-ran bit-identical after the fix.
- Plots go to `figures/`, tables to `results/`, never inline in notebooks only
- Prefer scripts over notebooks for anything reproducible

---

## Vocabulary

| Term | Meaning here |
|---|---|
| Session | One data collection. A row in `run_data_meta.csv`. |
| Subject | One person, `sub_id`. May have several sessions. |
| Waveform | 100-point time-normalized stance-phase joint angle trace |
| Sagittal subset | The 3 flexion/extension channels recoverable from monocular side-view video |
| Demographic control | The no-kinematics baseline model. The thing every result is measured against. |
| Degradation | ΔAUC between full-mocap and video-constrained feature sets |

---

## Out of scope

Do not propose, scaffold, or build these:

- Injury *prediction* over time. The data does not support it.
- Multi-person tracking or re-identification across camera cuts.
- Any commercial framing. AthleticsPose (CC BY-NC-SA 4.0) and AthletePose3D
  (non-commercial research only) prohibit it.
- Fusing the kinematic score and any training-load score into a single number.
- A mobile app, a user account system, or a database.
- Deep learning on the Ferber tabular/waveform data.
