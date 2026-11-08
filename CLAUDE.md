# Overstride

Injury screening for runners from monocular video.

Learning mode: DEEP  -- see LEARNING.md

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
> **3.51° MAE as the tool actually ships** (3.43° in the configuration phase 3
> ran — see the two baselines below). Phases 12–16 make that measurement more
> accurate, test it on datasets
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
- **Every accuracy claim is a delta against the frozen baseline, and the
  baseline is two numbers, not one.** Phase 7 measured six configurations on the
  same 592 AthleticsPose clips (`results/phase7_inference_full.json`); the one
  the tool runs is not the one that was being quoted:
  - **3.43°** — *historical*. Fine-tuned detections, zero padding, consecutive
    windows: exactly what phase 3 ran. Quote it only as that.
  - **3.51° near limb / 3.62° far limb** — **deployable, and the one phases
    13–16 are measured against.** Generic COCO detections, edge padding,
    overlapped windows — what `scripts/video_kinematics.py:241,244` actually
    loads and runs. Subject-level, 3.45° [3.27, 3.69] near and 3.56°
    [3.13, 3.90] far (`results/phase7_baseline_ci.json`). **The far limb is the
    larger of the two on this path**, and gates are read on the *paired*
    per-subject difference, not on overlapping marginal intervals. With three
    held-out subjects the resolution floor is about **0.2°**; below that, a real
    improvement cannot be shown here.
  - **3.18°** — *not deployable*. It needs AthleticsPose's fine-tuned
    detections, which no real video can produce. A gate passed by switching to
    them has measured nothing, so **every gate names its detector.**

  They differ only by detector provenance (+0.45°) and the two inference fixes
  (−0.25°). Report per dataset, per joint, with subject-level bootstrap CIs. A
  pooled number alone is not reportable. If a change does not beat the baseline,
  say so plainly and keep the baseline.
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
| **AthletePose3D** (in use) | Video + 3D GT, independent lab, 4 calibrated cameras | yes | non-commercial research only | **tune** — demoted from held-out in phase 12: phase 11 indexed and measured it twice, so by the rule above it has been opened. Raw video never downloaded |
| **BML MoVi** (York University) | 90 actors, synchronized mocap (Qualisys, 120 Hz) + 4-view video + IMU. 21 actions: **jogging yes; movement 3 is _running in place_**, which is not running gait | **jogging only** | non-commercial research; no redistribution; no training for commercial use. `doi:10.5683/SP2/JRHDRN` | **held-out test** — **do first.** 2 hand-held iPhone 7 + 2 FLIR, **and they are NOT the same format** — measured 2026-11-02 from the files: FLIR/Point Grey is **800×600 at 30.000 fps**, the iPhones are **1920×1080 at 30.009 fps** with ~27% more frames per recording. Only the PG cameras are calibrated and only they can be aligned to mocap from the shipped flags, so **only PG carries an angle error**: real consumer footage *with* ground truth, which is what §7.5 needs. No frame of it may be published |
| **Dual-system gait & fitness set** (Zenodo `10.5281/zenodo.19720803`, v3) | 21 subjects; OptiTrack 7-cam 120 Hz + two smartphones at 30 Hz; includes running | yes | **CC BY 4.0** — verified 2026-10-19, *not* NC-ND as previously recorded | **held-out test** — closest match to the deployment camera, and the **only one whose video may appear in the demo** (attribution only). 1.1 GB. Still open: whether `03_markerless_data.zip` holds video or only 2D keypoints — answerable by a filename listing at the start of phase 15 |
| **Bath BioCV** (Sci. Data 2024, `doi:10.15125/BATH-01258`) | 15 subjects; 9-camera HD video at **200 fps**, 15-camera optical mocap, force plates, photogrammetry. CMJ, walking, **running** | **yes** | **All Rights Reserved**; access on request, record marked restricted; biomechanics / machine-vision research only; not to be shared outside the dataset | **held-out test** — verified 2026-10-19. Highest-quality reference and furthest from a phone. ~15 GB **per participant**, ≈225 GB total, so take participants singly. Ships `c3d` marker tracks, not joint angles. No published footage |

Not used, and why: **SportsPose** (activities not confirmed to include running);
**OpenCap Monocular** (a *comparator*, not data — validated on walking, squats
and sit-to-stand at 4.8° MAE, not running; cite it, don't train on it).

**Why this mix.** The train sets teach the pipeline what bodies and running
look like. The tune sets are where decisions get made. The held-out sets are
where the claim gets tested.

> **Status, 2026-11-03. MoVi has been opened; the other two have not.** Phase 15
> tested on MoVi round F, PG1 — and the sealed test spent five further subjects
> (`Subject_16/17/18/19/2`) that had never been looked at. **Those subjects are
> now spent and may not be reused as a held-out test.** Round F has more
> subjects with PG1 video, so a *new* sealed set is still possible inside MoVi,
> but it must be designated and pre-registered before download, exactly as
> `268f28b` did. **The dual-system set and Bath BioCV remain unopened and keep
> their role.** Nothing has been tuned on any of them.

The 2026-10-19 acquisition spike established that all three exist, what each
licence actually says, and that two of the three release raw video — see
`docs/data-inventory.md` §8.4, which carries the accessions, the blockers and
the order to take them in. Establishing existence and licence is **not** opening
a held-out set; browsing or plotting its contents would be, and none of that
happened, so all three keep their role. What the spike did not settle, because it
cannot be settled from outside the data, is whether any camera in MoVi or the
dual-system set is **side-on** to the direction of travel — the geometry this
project prescribes, and the one phase 11 measured error falling 3.50° → 1.77°
towards. Phase 15 answers that first, and reports it even if the answer weakens
the test.

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
| **12** | **Scan** — per-subject body model from a short calibration pass | Segment lengths within the pre-registered error of mocap GT, stable across a subject's clips | Mixed — stability met (CV 1.7%), accuracy **not met** (0/8 ratios at 3%, errors 4.5–9.2%). The scan is precise but biased: limbs compressed 4.3%, pelvis widened 3.5%. Kill did not fire — keep it as a stabiliser, not anthropometry |
| **13** | **Decompose & track** — arm 1 filter only (run), arm 2 scan as a *consistency* constraint (deferred, geometric argument), arm 3 both (not run) | Beats **3.51° near / 3.62° far**, same detector both sides, paired per subject, near and far judged separately | **Not met.** Filtering is harmful at every setting, monotonically: +0.12° at the tool's own σ, +4.61° at 6 Hz, all CIs excluding zero. The input-side half of the phase is **closed** — the checkpoint is the binding constraint |
| **14** | **Per-segment confidence** — calibrated measurement reliability + viewpoint-based expected error | Confidence ranks frames by realized error (ρ ≥ 0.30, every subject) and the stated error is the right size (slope in [0.8, 1.2]) | Mixed — **2 of 4 segments rank their error** (both shanks, ρ 0.34); **1 of 4 is also calibrated**; and the permutation control shows ~4/5 of the skill is **between clips**, not between frames. It is a viewpoint meter. Camera azimuth *is* recoverable from video alone, r = 0.974 |
| **15** | **Held-out validation** — BML MoVi opened, PG1 (calibrated machine-vision), jogging | Per-dataset MAE with CIs, reported whether it wins or loses. **No pass/fail gate** | Met as a *report*, lost as a *result* — **9.610° near / 10.604° far** (n=2) against the 3.51°/3.62° deployable baseline, ~2.7×. One subject excluded by a check pre-registered before download. **CP (iPhone) refused: no calibration, sync unverifiable from the shipped flags — §7.5 stayed open here** |
| **15B** | Why "perfect 2D" underperformed detected 2D on the hip — three pre-registered hypotheses | Each hypothesis has its outcome rule fixed before running | **All three REJECTED.** Detector hip offset is **zero** over 76,813 frames (phase 07's "21–23 px" is a magnitude, not a vector); recipe-matched reference made bias ~9° *worse* on all six rows; posture R² 0.108–0.315, 1 of 4. **Detector cost spans −2.65° to +5.69° and changes sign** |
| **15C** | Settle the lifter's thorax definition — on AthleticsPose only, then apply to MoVi once | Candidates and decision rule fixed before running | Met, with a transfer failure. The released thorax **is `CLAV`, to 5.04 mm** on AthleticsPose — but on MoVi the same checkpoint is nearer Visual3D `THORAX` on **7 of 7** subjects (0.3–5.7° vs 9.0–16.2°). **The lifter does not put its thorax on the same anatomical point in two datasets.** Knee is immune by construction: 24 of 24 rows bit-identical |
| **15D** | Report hip and knee separately; segment angles against **gravity** instead of the trunk | G1 thigh MAE ≤ 4.118° on AthleticsPose; G2 thigh \|bias\| < hip \|bias\| on ≥ 3 of 4 subjects | **G1 FAIL** (6.694° [6.290, 7.390]), **G2 PASS 4/4**. Per joint, MoVi vs AthleticsPose: knee 8.163° vs 4.071° (2.00×), hip 11.057° vs 2.834° (3.90×) — **the ordering inverts between labs**. The hip decomposes exactly: `hip bias = thigh bias + trunk-lean bias`, residual ~1e-15 |
| **15 sealed** | The one run on subjects **never opened** — 5 MoVi subjects, pipeline pinned by content hash, pre-registration pushed before any sealed byte was downloaded | **No gate.** Report as it comes out | Met. **All five passed every check, none excluded** (n=5, the largest MoVi n). Knee **8.410° near** [6.275, 10.975] / 8.685° far; thigh inclination **5.401° near** [4.136, 6.837] / 7.372° far; combined hip+knee **8.078° near**. **Not worse than the spent subjects** — 3 of 4 primary figures lower, every CI containing its comparison, so *indistinguishable*, not better. **The quantity that failed G1 is the one that transfers:** gravity-referenced thigh is 5.401° on held-out MoVi against 6.694° on the tune set, while knee degrades 2.07× and hip 2.73×. Error is a **per-subject offset** (MAE vs \|bias\| r = 0.983 knee / 0.917 thigh; shape r +0.77 to +0.92). **First resolved far-limb penalty** in the project — thigh +1.971° [+0.977, +3.187], hip +1.510° [+0.611, +2.531], 5/5 same sign; knee none. Perfect 2D **still not an upper bound**: worse on the trunk for 5/5 |
| **16** | **Streaming demo** → split into **16A offline** (one consented clip) and **16C streaming**, deferred | ≥ 20 fps end to end; demo angles match the phase 15 offline numbers | **≥ 20 fps NOT MET and not attemptable** — 207 ms/frame is the floor with these weights, 5.3× over. 16A proceeds; 16C waits for phase 15 |

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

> **Re-cut 2026-10-19, before any of it was built.** The original plan rested on
> "the near limb's lengths constrain the far limb". Phase 12 undermined exactly
> that half: the scanned lengths are *precise* (across-trial CV 1.7%) and
> *biased* (ratio error 4.5–9.2%, 0 of 8 ratios at the 3% bar), so propagating a
> near-limb length to the far limb propagates the bias. The scan survives as a
> **consistency constraint**, not as anthropometry, and the far-limb-from-near-
> limb idea is dropped unless arm 2 shows the constraint is worth anything at
> all. Two further facts, both measured rather than assumed, set the order:
>
> * the largest **quantified** term left is the 2D detector, +0.43° near and
>   +0.42° far, both excluding zero (`results/phase7_baseline_ci.json`) — but see
>   the bound below: most of the detector-shaped gap is a marker convention, and
>   the realistic prize is 0.33°;
> * with three held-out subjects the **resolution floor is about 0.2°** — below
>   that nothing can be shown here, however real it is. Any arm whose expected
>   gain is smaller than the floor is not worth building yet, and must say so
>   instead of being run and reported as "no effect".

Three arms, cheapest first, each stopping early if it clears the gate. Every arm
is a separate row; a gain that only appears in combination is reported as such.

1. **Filter only** — one filtered state per segment (Kalman or one-euro, chosen
   on the tune set), no scan, no new data. The cheap half of the mandatory
   ablation. AddBiomechanics joint ranges are plausibility bounds, never a
   smoothing target.
2. **Scan as a consistency constraint** — hold a subject's proportions fixed
   across frames (14.5% frame-to-frame length wander down to 1.7%), with **no
   claim that the lengths are true**. Reported in those words.
3. **Both**, only if either arm moves the number.

**Prerequisite, not part of this phase:** the detector replacement — and it is
now **bounded**, which changes what it is for. `results/phase07.md` §5, all 592
clips:

- a **perfect** 2D detector with matched weights gives **1.47° near / 1.63° far**,
  so the whole detector-shaped gap is 2.04°;
- but **most of that gap is a convention, not a detector.** Both hips sit at
  21–23 px from projected mocap for *both* released detectors, including the one
  fine-tuned on this dataset's own 2D, because H36M's hip is the mean of the ASI
  and PSI markers — a point inside the pelvis that no image-based detector marks.
  Phase 12 measured the same offset in 3D at 81 mm;
- the **realistic** accuracy prize is the measured **0.33°** between a generic
  COCO detector and a fine-tuned one — above the 0.2° floor, worth having,
  modest. And it comes from **distal-joint accuracy specifically**: the two
  detectors have the *same* mean 2D error, 12.28 px against 12.18 px, and differ
  only in where it sits (knees and ankles 3.2–4.1 px against 5.4–6.4 px);
- **a swap is not a drop-in.** Handing the shipped checkpoint cleaner-than-trained
  2D is monotonically worse — 3.51° → 4.33° → 4.68° → 5.16° as the input moves
  toward truth — so a new detector needs a matched lifter or it degrades the
  tool. Phase 7's `CKPT_FOR` lesson, measured in the other direction.

**So the detector is replaced for SPEED** — 2,790–3,280 ms/frame against phase
16's 50 ms budget — with a bounded accuracy benefit that arrives only with
matched weights. Judge candidates on **knee and ankle pixel error**, never on
mean pixel error or COCO AP.

- If the lifter is retrained, BEDLAM and AMASS are the only training data.
- **Gate:** sagittal MAE beats the **3.51° near / 3.62° far** deployable
  baseline on AthleticsPose — the same detector on both sides of the comparison,
  never the fine-tuned detections — read as a **paired per-subject difference**
  with a subject-bootstrap CI excluding zero, because at n=3 every marginal
  interval overlaps every other. Near and far limb each get their own verdict;
  AthletePose3D does not get worse than its phase 11 figure.

### Phase 14 — Per-segment confidence

- Each segment, each frame: a confidence built from keypoint visibility,
  deviation from scanned length, and an estimated camera azimuth (from apparent
  hip/shoulder width against the scan). Expected error per clip comes from the
  **measured** phase 11 viewpoint curve (3.50° → 1.77°), not a new guess.
- **Gate:** confidence ranks frames by realized error (Spearman ρ ≥ **0.3**,
  pre-register), and stated expected error vs realized error has a calibration
  slope in **[0.8, 1.2]** (pre-register). Same structure as phase 4B's
  calibration check, applied to angles instead of labels.

> **Outcome, 2026-10-19 — `results/phase14.md`.** Partially met, and the useful
> part is not the part that was planned.
>
> * **Only four segments have a sagittal error to rank.** The body frame is built
>   from the pelvis and the trunk, so those two cannot have one; foot has no toe
>   keypoint. Thigh and shank, near and far, is the whole measurable set.
> * **Both shanks clear ρ 0.30 for every subject; both thighs do not.** One
>   segment of four is also correctly scaled. Kill did not fire.
> * **It is a viewpoint meter.** Camera azimuth carries essentially all the
>   signal; keypoint score, segment-length deviation, left/right asymmetry and
>   window blend weight are within noise of zero, and the detector's own
>   confidence is *anti*-correlated with thigh error.
> * **~4/5 of the ranking is between clips**, per the pre-registered
>   within-clip permutation control. So phase 16 gets a **clip-level** expected
>   error it can defend, per-segment confidence on the **shanks** with a caveat,
>   and **no** per-segment claim on the thighs.
> * **Reusable, and it closes an open question:** this phase asked for "an
>   estimated camera azimuth (from apparent hip/shoulder width against the
>   scan)" without knowing one was obtainable. It is — apparent 2D hip
>   separation over apparent trunk length, over a population ratio, at
>   **Pearson 0.974** against the ground-truth depth metric, with no depth, no
>   millimetres and no ground truth. The near limb is also identifiable from the
>   estimate alone, 592 of 592 clips.

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

> ## The 20 fps gate is NOT MET, and it is not being attempted (2026-10-19)
>
> **The pre-registered bar stands as written and is recorded as failed.** It is
> not being lowered, re-defined or quietly dropped; what changes is the order of
> work, which is an operator decision taken on the measurements below
> (`results/phase07.md` §§5–6, `results/phase16_detector_speed.py` output).
>
> **What was taken, because it is free.** `scripts/video_kinematics.py` now
> detects at the frame's **native resolution** instead of letting torchvision
> upscale every frame to `min_size=800`, and runs on a **padded person crop**
> that tracks the previous frame, re-detecting full-frame whenever a limb leaves
> the window. **1,055 ms/frame → 207 ms, a 5.1× speed-up with the same weights**
> — so no retraining, and no shift in the keypoint distribution the lifter was
> trained against.
>
> **Why that is not enough, and why a faster detector is not the answer yet.**
> 207 ms against the ~39 ms the detector may have inside a 50 ms frame is still
> 5.3× over, and a two-stage ResNet50 Keypoint R-CNN will not close that on a
> laptop CPU. A different architecture would — but §5 measured what that costs:
> the lifter has **learned its detector's error distribution**, so handing the
> shipped checkpoint cleaner-than-trained 2D is monotonically *worse*
> (3.51° → 5.16°). A new detector therefore needs a **matched lifter**, a matched
> lifter needs that detector's 2D on data with 3D ground truth, and
> **AthleticsPose cannot provide it — it releases no video.** BEDLAM is the only
> train-side set that could.
>
> **So the swap cannot be measured today, and this project does not ship what it
> cannot measure.** Phase 15 brings MoVi — 30 fps hand-held iPhone video
> synchronised to 120 Hz Qualisys — which is both the §7.5 answer and the first
> test bed on which a candidate detector could be judged. The architecture
> question is deferred to after it, deliberately and on the record.
>
> **16A proceeds now:** one consented side-on clip, offline, through the fixed
> tool, publishing the traces and the error budget. That closes §7.5's substance
> at ~5 fps without claiming a frame rate it does not have. **16C** — streaming
> and the browser port — waits.
>
> **Judge a candidate detector on knee and ankle pixel error**, never on mean
> pixel error or COCO AP: §5 found the two released detectors have the *same*
> mean 2D error, 12.28 px against 12.18 px, and differ by 0.33° in angle.

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

- **No Co-Authored-By or other attribution trailers in commit messages.**
  Every commit here is authored by the repository owner alone. Enforced by
  `.claude/settings.json`'s `attribution` block; this line is the statement
  of intent the setting implements.
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
| Baseline (measurement) | The frozen phase 7 pipeline on AthleticsPose. **3.51°** as the tool ships (generic COCO + edge padding + overlapped windows) — the figure phases 13–16 are measured against. **3.43°** is the historical phase 3 configuration; **3.18°** is not deployable |
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
