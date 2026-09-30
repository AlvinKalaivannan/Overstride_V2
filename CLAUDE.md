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
> work on this repository; phases 0–11 record what happened, not a plan.

---

> ## NEW TRACK OPEN: measurement and demo (phases 12–16)
>
> The injury question is closed and stays closed. The part of Overstride that
> works is the **measurement**: side-on video → sagittal hip/knee/ankle angles at
> 3.4° MAE. Phases 12–16 make that measurement more accurate, test it on datasets
> it has never seen, and ship it as a demo that runs on real footage.
>
> The method is borrowed from VSCS (the vehicle scanning project): **scan** the
> subject once to build a body model, **decompose** it into rigid segments,
> **track** each segment on its own, and report a **per-segment confidence**.
> In VSCS the per-component output is collision risk. Here it is *measurement
> reliability* — how much to trust each segment's angle in each frame. It is
> never an injury score. See [Measurement and demo track](#measurement-and-demo-track-phases-1216).

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

### Measurement track (phases 12–16)

- **No injury output, in any form.** Not a score, not a flag, not a colour, not
  a "consult a professional" nudge driven by the kinematics. Phases 4B and 5D
  settled this. Per-segment confidence describes the *measurement*, and its
  name, UI and docstrings must say so. The word "risk" does not appear in any
  output, UI, docstring or plot of this track.
- **Dataset roles are fixed before a phase starts and never change mid-phase.**
  Every pose dataset is either *train*, *tune*, or *held-out test* (see the
  dataset table under Data). A held-out dataset is not opened — not browsed, not
  plotted, not used to pick a threshold — until the phase that tests on it.
  Looking at it early turns it into a tune set; if that happens, say so in the
  report and demote it.
- **Splits inside every dataset are grouped by subject**, same rule as `sub_id`
  above. A subject's clips never appear on both sides of a split.
- **Every accuracy claim is a delta against the frozen baseline**: the phase 7
  pipeline (`scripts/video_kinematics.py`, 3.43° sagittal MAE on AthleticsPose).
  Report per dataset, per joint, with subject-level bootstrap CIs. A pooled
  number alone is not reportable. If a change does not beat the baseline, say
  so plainly and keep the baseline.
- **Near and far limb are always reported separately.** A method that helps the
  near limb and hides a far-limb regression inside an average is a failed method.
- **Ground truth is what the dataset shipped.** Never fit, smooth or re-derive a
  dataset's reference 3D before scoring against it. Unit traps like the
  AthleticsPose `p2mm` one are checked and written into `docs/data-inventory.md`
  for every new dataset before its first number is reported.
- **Licenses are recorded per dataset before download** in
  `docs/data-inventory.md`. Raw video from a research dataset is never
  redistributed, including inside the public demo, unless its license permits
  it. No-derivatives licenses (ND) mean nothing derived from that dataset is
  published beyond summary metrics.

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

### Pose and 3D datasets (phases 12–16)

The Ferber archive plays no part in the measurement track. These datasets do.
Each one lives under `data/<name>/`, covered by `.gitignore`, with its license,
units and coordinate conventions written into `docs/data-inventory.md` before
use. Roles are assigned here and frozen.

| Dataset | What it gives | Running? | License (verify before download) | Role |
|---|---|---|---|---|
| **AthleticsPose** (in use) | Video + 3D GT, athletics | yes | CC BY-NC-SA 4.0 | **tune** — the development set; phase 7 baseline lives here |
| **AMASS** | Mocap as SMPL body models, many labs incl. CMU running | yes | MPI non-commercial research | **train** — body-shape and pose priors for the scan and the segment fitter |
| **BEDLAM** | Synthetic video with exact SMPL-X GT; varied bodies, clothing, cameras | some | MPI non-commercial research | **train** — lifter robustness to body shape and camera; targets the phase 11 transfer gap |
| **AddBiomechanics** | 273 subjects, 70+ h of OpenSim-processed mocap from 15 datasets / 12 labs; no video | yes | CC BY 4.0 | **train / reference** — joint-angle ranges and segment-length ratios as plausibility bounds; normal-range reference for the demo |
| **AthletePose3D** (in use) | Video + 3D GT, independent lab, 4 calibrated cameras | yes | non-commercial research only | **held-out test** — already measured once in phase 11; re-used only as the transfer check |
| **BML MoVi** (York University) | 90 actors, synchronized mocap + 4-view video + IMU, 21 actions incl. jogging | yes | non-commercial research; no training for commercial use | **held-out test** |
| **Dual-system gait & fitness set** (Sci. Data 2026) | 21 subjects; OptiTrack 120 Hz + two smartphones at 30 Hz; includes running | yes | CC BY-NC-ND 4.0 | **held-out test** — closest match to the deployment camera (phone, 30 fps). ND: summary metrics only. Confirm raw video is released, not only its YOLOv8 2D keypoints |
| **Bath synchronised video / mocap / force plate set** (Sci. Data 2024) | Synchronized video, mocap and force plates for markerless validation | **unverified** | **unverified** | candidate held-out test — confirm movements and license first |

Not used, and why: **SportsPose** (activities not confirmed to include running);
**OpenCap Monocular** (a *comparator*, not data — validated on walking, squats
and sit-to-stand at 4.8° MAE, not running; cite it, don't train on it).

**Why this mix.** The train sets teach the pipeline what bodies and running
look like. The tune set is where decisions get made. The held-out sets are
where the claim gets tested, and none of them has been looked at yet except
AthletePose3D.

---

## Environments

Two environments, deliberately separated. The 21 GB archive never leaves the
laptop; the only thing that crosses is a feature file of roughly 10 MB.

| Phase | Environment | Notes |
|---|---|---|
| 0–2 | Laptop, CPU | No GPU available and none needed |
| 3 | ~~Kaggle Notebooks (P100/T4)~~ → **Laptop, CPU** | See deviation below |
| 4–6 | Laptop, CPU | |
| 12–15 | Laptop, CPU for inference and scoring; a cloud GPU notebook only for training on the *train* sets | Only pose datasets whose license permits cloud processing go to the notebook. The Ferber archive never does |
| 16 | Browser (the demo) + laptop | The demo must run on a mid-range laptop with no GPU setup |

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
| **0** | Inventory: parse, join, characterize, verify against Table 1 | `docs/data-inventory.md` exists and Table 1 reproduces | Met |
| **1** | Injury classifier on all 9 mocap waveforms | Beats the demographics-only control | Not met — **kill criterion fired** (0/60 tests) |
| **2** | Restrict to 3 sagittal waveforms, downsample, inject keypoint noise | Degradation measured — **the headline result** | Met (σ labels corrected in phase 4) |
| **3** | Video → 3D kinematics via released checkpoints | Joint-angle MAE within range of published figures | Met — 3.4° fine-tuned |
| **3B** | Viewpoint geometry; the pixel/`p2mm` unit error | — | Met — occlusion penalty ~0 |
| **4** | Video-derived features through the phase 2 model | Real ΔAUC, not simulated | Met — −0.027 to −0.033 |
| **4B** | Operating point, calibration, repeatability | — | Not met — not deployable |
| **5** | Personalization: `InjSide` asymmetry, within-session stride distributions | Beats the population model | Met — 0.610 |
| **5B–5D** | Ceiling diagnostics, further feature families, abstention | — | Not met — ceiling is the signal |
| **6** | ~~Demo shell~~ → **Methods demo + synthesis** | — | Met — `README.md` |
| **7** | Harden the inference path; `video_kinematics.py` | — | Met — −0.25° recovered; first tests in the repo |
| **8** | Positive control: inject a known asymmetry, sweep its magnitude | Method detects an asymmetry known to be present | Met — resolves 0.25°; 0.610 ≈ 0.17° RMS |
| **9** | A1 — the same limb task on **walking**, paired dual-mode subjects | Clears phase 5's bar with controls at chance | Not met — **0.547**, does not replicate |
| **10** | C — a second lab's markers (Fukuchi) through the same MATLAB pipeline | Recovers sane kinematics outside the archive | Met — speed 1.71%; hip \|r\| 0.993; ankle weakest |
| **11** | B1 — AthletePose3D: an independent lab through the lifting path | Lifter transfers; viewpoint penalty measured | Mixed — lifter degrades **1.84×**; viewpoint **3.50°→1.77°** measured; §7.5 still open |
| **12** | **Scan** — per-subject body model from a short calibration pass | Segment lengths within the pre-registered error of mocap GT, stable across a subject's clips | Planned |
| **13** | **Decompose & track** — segment-constrained fitting, per-segment state | Beats the 3.43° baseline on the tune set; far limb improves or holds; no transfer regression | Planned |
| **14** | **Per-segment confidence** — calibrated measurement reliability + viewpoint-based expected error | Stated confidence matches realized error | Planned |
| **15** | **Held-out validation** on datasets never touched | Per-dataset MAE with CIs, reported whether it wins or loses | Planned |
| **16** | **Streaming demo** — near-real-time, browser, runs on real footage | Latency and accuracy budgets met on held-out footage; closes §7.5 | Planned |

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

## Measurement and demo track (phases 12–16)

**Goal:** a side-on running clip goes in; per-segment sagittal kinematics come
out, each with a confidence and an expected error, fast enough to watch live.
The claim to earn is *"more accurate than phase 7, and it holds on footage from
labs it has never seen."*

**Where the ideas come from.** VSCS scans a vehicle, isolates it into
mechanical components, tracks each component and scores each one on its own.
Mapped onto a runner:

| VSCS | Overstride |
|---|---|
| Scan the vehicle | Calibration pass: 2–3 s standing side-on + the first strides |
| Isolate components | Rigid segments: trunk, pelvis, L/R thigh, shank, foot |
| Track each component | Segment-constrained fitting; one state per segment |
| Per-component collision risk | Per-segment **measurement confidence** (not injury risk) |
| Near-real-time stream | Frame-in, state-out pipeline with a latency budget |
| Demo on existing footage | Demo runs on held-out dataset clips and phone clips |

**Pre-registration.** Before each phase starts, its bars (the numbers in
*Gate* below marked "pre-register") are written into `results/phaseNN.md`
with the date, and are not edited once the first result exists. The proposed
values here are starting points, not commitments.

### Phase 12 — Scan

- Fit a per-subject body model from a calibration window: segment lengths for
  every segment in the table above, both sides, plus a left/right length
  symmetry check. Start with direct 2D/3D segment measurement; an SMPL shape fit
  using AMASS priors is the upgrade path if it measurably helps.
- **Gate:** on the tune set, estimated segment lengths within **≤ 5%** of
  mocap-derived lengths (pre-register), and a subject's scans agree with each
  other to **≤ 3% CV** (pre-register). Report per segment.
- **Kill:** if lengths from one clip are no more stable than per-frame lengths,
  the scan adds nothing — record it and go to phase 13 without it.

### Phase 13 — Decompose and track

- Constrain every frame to the scanned segment lengths; the near limb's
  lengths constrain the far limb. One filtered state per segment
  (Kalman or one-euro, chosen on the tune set). AddBiomechanics joint ranges
  are plausibility bounds, never a smoothing target.
- If the lifter is retrained, BEDLAM and AMASS are the only training data.
- **Gate:** sagittal MAE beats the **3.43°** baseline on AthleticsPose with a
  subject-bootstrap CI excluding zero; far-limb MAE improves or stays within
  its CI; AthletePose3D does not get worse than its phase 11 figure.
- **Ablations are mandatory:** scan only, filter only, both. Each gets its own
  row. A gain that only shows up combined is reported as such.

### Phase 14 — Per-segment confidence

- Each segment, each frame: a confidence built from keypoint visibility,
  deviation from scanned length, and an estimated camera azimuth (from apparent
  hip/shoulder width against the scan). Expected error per clip comes from the
  **measured** phase 11 viewpoint curve (3.50° → 1.77°), not a new guess.
- **Gate:** confidence ranks frames by realized error (Spearman ρ ≥ **0.3**,
  pre-register), and stated expected error vs realized error has a calibration
  slope in **[0.8, 1.2]** (pre-register). Same structure as phase 4B's
  calibration check, applied to angles instead of labels.

### Phase 15 — Held-out validation

- Run the frozen phase 13–14 pipeline, unchanged, on MoVi jogging, the
  dual-system running trials and (if verified) the Bath set. Tuning anything
  after seeing these numbers invalidates them.
- **Report:** per dataset, per joint, near/far limb, MAE with CIs, against the
  phase 7 baseline on the same clips, plus how often the phase 14 confidence
  flagged the frames that turned out worst. OpenCap Monocular's 4.8° (walking,
  squat, sit-to-stand) is cited as context, not as a head-to-head.
- **There is no pass/fail gate.** A loss on held-out data is a result and goes
  in the README exactly as prominently as a win.

### Phase 16 — Streaming demo

- Refactor `video_kinematics.py` into a stream: frame in → scan state →
  per-segment state out, with a stated latency budget per stage. The browser
  demo gets the scan step, per-segment confidence on the overlay, and the
  expected-error readout.
- The demo keeps what already made it honest: it states its error budget,
  refuses to write output when it can't find the subject, and has no injury
  output (see Measurement track rules).
- **Gate:** ≥ **20 fps** end to end on a mid-range laptop CPU (pre-register);
  demo angles on held-out clips match the phase 15 offline numbers within their
  CI. **Running it on real phone footage closes §7.5** from phase 11.
- Demo clips come from datasets whose license allows showing them, or from
  footage filmed for this project with the runner's consent.

### Optional research follow-up

- **Heterogeneous positive control** — phase 8's own suggested extension:
  draw a per-subject asymmetry profile instead of one shared offset. Separate
  from 12–16; it touches the Ferber data and follows the original rules above.

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

For phases 12–16, the report replaces the classifier items above with:

- MAE per dataset, per joint, near and far limb, with subject-bootstrap CIs
- The phase 7 baseline on the same clips, and the delta
- The dataset role (train / tune / held-out) and the subject-grouped split
- Which pre-registered bars were set, when, and whether they were met
- Latency per pipeline stage, for any phase that touches the demo

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
| Scan | The per-subject calibration pass that produces the body model (phase 12) |
| Segment | One rigid body part in the model: trunk, pelvis, thigh, shank or foot, per side |
| Confidence | Per-segment, per-frame reliability of the *measurement*. Never an injury quantity |
| Baseline (measurement) | The frozen phase 7 pipeline, 3.43° sagittal MAE on AthleticsPose |
| Held-out | A dataset not opened until the phase that tests on it |

---

## Out of scope

Do not propose, scaffold, or build these:

- Injury *prediction* over time. The data does not support it.
- Multi-person tracking or re-identification across camera cuts.
- Any commercial framing. AthleticsPose (CC BY-NC-SA 4.0) and AthletePose3D
  (non-commercial research only) prohibit it, and so do AMASS, BEDLAM, MoVi and
  the dual-system set.
- Fusing the kinematic score and any training-load score into a single number.
- A mobile app, a user account system, or a database. The browser demo is a
  single page that processes video locally; it stays that way.
- Deep learning on the Ferber tabular/waveform data.
- Using the scan, the segment model or per-segment confidence to reintroduce an
  injury output of any kind.
- Tuning anything on a held-out dataset.
