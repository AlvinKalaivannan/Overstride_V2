# Overstride — cold-start context

**Read this before touching anything.** It exists so someone with no memory of
how this was built can be productive without reconstructing it from 14 reports
and 42 scripts. `README.md` is the *finding*, written for a reader; this is
*working context*, written for whoever resumes.

Current HEAD: `038014e` (phase 7). Phases 0–7 all have recorded gates.

---

## 1. Orientation

Overstride asked one question: **how much injury-classification performance is
lost when lower-limb kinematics come from a single camera instead of a motion
capture lab?**

The answer is **about 0.03 AUC**, and it is that small because there was very
little signal to lose. The only task in this dataset carrying kinematic injury
information — identifying *which limb* is injured in a runner already known to be
injured — reaches **0.610** on perfect mocap and **0.583** with real measured
video error, against a chance rate of **0.517**.

**The research is finished. Do not try to raise the injury signal** — that was
attacked five independent ways in phases 5B–5D and the ceiling is the signal, not
the model, the features, or the camera. See §9.

**The video tool is partial.** It runs decode → detect → lift → angles, but has
never been run on a real running clip, and its detector has never been validated.

---

## 2. Hard rules

From `CLAUDE.md`. Violating any of these invalidates results. If a task seems to
require breaking one, stop and say so rather than proceeding.

- **All splits are grouped by `sub_id`.** `StratifiedGroupKFold`, seeds
  `[11, 23, 37, 53, 71]` → 25 folds. Rows are sessions; some subjects have
  several. Never `train_test_split` on rows. `check_no_group_leakage` must be
  asserted, not just called.
- **Every kinematic score reports a demographics-only control and a
  provenance-only baseline beside it.** Not inherited from an earlier phase —
  re-asserted in the script, with `assert worst < 0.08`.
- **Never commit data.** `data/` is gitignored. Verify with `git check-ignore -v`
  before any commit.
- **Never load all session files at once.** 2,506 JSONs of marker trajectories
  will exhaust RAM. Stream, or sample and report.
- **Report AUC with cross-fold CIs**, never a bare number.
- **All preprocessing inside the fold**, via `Pipeline`. Scalers, imputers, PCA,
  feature selection — no exceptions.
- **No per-runner injury verdict**, and no output that could be read as one. See
  §11.
- **The word "predict" must not appear** in any output, README, docstring or plot
  title. Use detect / classify / screen. (Code identifiers like sklearn's
  `predict_proba` and the lifting call are the accepted exception; prose is not.)

---

## 3. Repository map

```
scripts/   42 files   phase0…phase7, video_kinematics, MATLAB vendoring
tests/      4 files   19 tests, ~5 s, pytest
results/   31 files   14 phaseNN.md reports + their JSON artifacts
docs/       6 files   schema, inventory, this file
figures/    4 files   the published plots
notebooks/  2 files   superseded Kaggle route, kept as record
```

**Shared machinery — reuse, do not rewrite:**

| need | where |
|---|---|
| cohort building, sentinel masking, control feature groups | `scripts/phase1_cohort.py` |
| folds, CV, CIs, paired deltas | `scripts/phase1_eval.py` — `make_folds`, `check_no_group_leakage`, `evaluate`, `ci`, `paired_delta` |
| the 818-session limb cohort, feature building, error bank | `scripts/phase4_real_delta.py` — `load_limb_cohort`, `build_features`, `load_bank` |
| sagittal angles from 3D keypoints | `scripts/phase3_angles.py` — `sagittal_angles` |
| checkpoint loading, 2D→3D lift | `scripts/phase3_infer.py`, and the hardened `lift()` in `scripts/phase7_inference_fix.py` |
| out-of-fold scores, calibration, repeatability | `scripts/phase4b_operating_point.py` |
| mirror-sign derivation | `scripts/phase5_limb.py` — `mirror_signs` (angles only; write your own for other channel families) |
| every published number, from source | `scripts/phase6_numbers.py` |

---

## 4. Data and assets

**Nothing below is committed.** All gitignored; all must be present locally.

| path | size | what |
|---|---|---|
| `data/ric/` | 47 GB | Ferber / Running Injury Clinic archive. Path set by `DATA_ROOT` in `.env`. |
| `data/derived/` | 1.4 GB | Generated waveforms, indices, error bank |
| `data/derived/waveforms_steps/` | 1.3 GB | 1,745 per-stride `.mat` files — phase 5C needs these |
| `data/athleticspose/data/` | 788 MB | AthleticsPose keypoints + GT 3D |
| `data/athleticspose/checkpoints/` | 471 MB | Five MotionAGFormer checkpoints |

**Waveforms are derived, not stored.** The archive holds raw marker trajectories
and scalar summaries only. The 101-point stance curves come from a vendored
MATLAB pipeline (`scripts/vendor_matlab_code.py`, `scripts/matlab/*.m`) that
patches a *copy* of the archive's own code. **This is the single biggest barrier
to reproducing from scratch** — it needs MATLAB, and the vendoring script asserts
the source archive is unmodified.

**Licensing:** AthleticsPose is CC BY-NC-SA 4.0, AthletePose3D is
research-use-only. **Commercial use is prohibited**, and no commercial framing of
this work is permitted.

---

## 5. What functions today

**The research pipeline reproduces.** Phases 3, 4, 5B and 5C were re-run during
development and came back bit-identical — including a refactor of
`phase4_real_delta.py` verified across 41 models × 25 folds at max per-fold AUC
difference `0.00e+00`.

**The video tool is partial.** `scripts/video_kinematics.py` decodes video,
detects COCO-17 keypoints, lifts to 3D and emits sagittal hip/knee traces plus a
CSV, figure and JSON. It has been exercised on synthetic video (decode, detect,
refusal path) and on AthleticsPose keypoints (lift, angles). **It has never run
on a real running clip** — no such footage is available here.

It refuses rather than emits: if a person is found in under half the frames it
aborts with exit code 2 and writes nothing. That behaviour exists because the
first test produced warnings *and* a CSV of garbage angles, which looks exactly
like a real result.

---

## 6. The two data flows

**Research** — this is what produced every published number:

```
data/ric/ric_data/<sub_id>/*.json          raw markers + scalar summaries
  → MATLAB (gait_kinematics.m → gait_steps.m, vendored + patched)
  → data/derived/waveforms_mean.npy        (n_sessions, 54, 101) float32
  → phase1_cohort.build_cohort()           labels, sentinel masking, controls
  → phase1_eval.make_folds()               25 subject-grouped folds
  → logistic regression, PCA in-fold       results/*.json + phaseNN.md
```

The 54 channels are `0..29` angles (L/R × ankle, knee, hip, foot, pelvis × 3
planes) then `30..53` velocities (L/R × ankle, knee, hip, pelvis × 3 planes).

**Video** — `scripts/video_kinematics.py`:

```
clip.mp4
  → cv2 decode
  → torchvision Keypoint R-CNN            COCO-17 pixel coords + confidence
  → 10 Hz low-pass on the 2D tracks
  → resample to ~120 fps                  the rate the lifter was trained at
  → MotionAGFormer lift                   edge padding, 50% overlapped windows,
                                          SCALE-FREE
  → phase3_angles.sagittal_angles         hip + knee; ankle is NaN by design
  → resample back to source frame rate
  → CSV + figure + JSON with error budget
```

---

## 7. Headline numbers

Regenerate all of these with `scripts/phase6_numbers.py`, which reads
`results/*.json`. Do not retype them from here.

| | |
|---|---|
| cohort (limb task) | 818 unilateral sessions / 675 subjects, chance 0.517 |
| screening | **0 of 60** pre-registered tests survived correction |
| provenance-only baseline | 0.775 pooled, 0.863 on patellofemoral pain — measuring paperwork |
| limb task, clean mocap | **0.610** [0.532, 0.686] |
| limb task, real video error | **0.583** [0.515, 0.631] |
| ΔAUC from video | **−0.027 to −0.033**, no delta CI excludes zero |
| monocular angle error | **3.4°** fine-tuned (MPJPE 51.7 mm), 18.1° generic |
| far-limb occlusion penalty | **+0.07°** extrapolated to fully lateral — effectively zero |
| operating point | PPV 0.574 vs 0.517 base rate; sens 0.130 at 90% spec |
| calibration slope | 0.716 (1.0 is perfect); Brier 1.5% better than a constant |
| repeat-scan agreement | **0.667** deployed, 0.744 on mocap |
| multi-session averaging | −0.003 — does **not** help; error is subject-specific |

---

## 8. Traps that already cost real errors

Each of these produced a wrong published claim before being caught. Assume the
next one is still hiding.

**`markers_h36m` is in PIXELS, not millimetres.** Each AthleticsPose `.npz`
carries a per-frame `p2mm` factor and the released evaluator **divides** by it,
uniformly across all three axes. Reading raw values as mm understates MPJPE ~3×
and makes side-on views look near-frontal. Cost two wrong claims across two
reports. Angles are unaffected — they are scale-invariant.

**Sagittal is plane 2, not plane 0.** `docs/ferber-schema.md` implies index 0
from the Cardan order; the emitted array disagrees. Measured against live `dv_r`
peaks, plane 2 gives |r| ≥ 0.99 and plane 0 gives |r| ≤ 0.17. The sign convention
is also inverted relative to `dv_r`. Verify with
`scripts/phase1c_verify_sagittal.py` before trusting any sagittal result.

**Missingness leaks the label.** Questionnaire completeness tracks the study wave,
and waves differ in injury mix. A blank `Level` marks an uninjured session with
95% precision. Never fit a missingness indicator or collection year as a feature;
always report the provenance-only baseline to bound the artifact.

**`999` is a missing-data marker in four columns, not one** — `age`, `Height`,
`Weight`, `YrsRunning`. 51 sessions claim 999 years of running experience.
`phase1_cohort.mask_sentinels()` handles it.

**L/R mirror signs.** Planes 0 and 1 (ab/adduction, rotation) are defined
relative to the midline and **mirror** between limbs; plane 2 is shared. So the
limb difference is `R + L` for planes 0/1 and `R − L` for plane 2. Getting this
wrong made `asym9` the worst-performing family in phase 1D. Derive signs from
population `corr(L, R)` and assert them — never assume.

**The 25 folds are 5 seeds × 5 splits of one dataset and are NOT independent.** A
fold-level Wilcoxon returned q = 0.000 for ITB syndrome while that condition's
own CI ran [0.410, 0.986]. Both cannot be right. **Resample subjects** —
`phase5c_features.subject_bootstrap_ci` does this correctly.

**Smart App Control blocks native binaries on this machine.** It killed pandas
3.0.5's `tzconversion.pyd` (hence the `pandas<3` pin) and mediapipe's shared
library. Note the failure mode: `import mediapipe` *succeeds* because its C
bindings load lazily, then `WinError 4551` fires on first use. **Verify a new
native dependency actually works before building on it**, not just that it
imports.

---

## 9. Gap register

Ordered by payoff per unit effort.

### Worth doing

**1. Checkpoint / detector mismatch — ~30 min, ML evaluation hygiene**
There are three checkpoints matched to three input types — `ath-det-ft`,
`ath-det-coco`, `ath-gt` — all identical architecture (11,721,795 params). Phase
3 paired `det-ft` weights with `det_ft` inputs correctly. **Phase 7's
detector-gap experiment fed `det_coco` inputs into the `det-ft` checkpoint**, so
its **+1.67° penalty conflates "worse detector" with "wrong checkpoint for that
detector" and is likely an overestimate.** `scripts/video_kinematics.py` inherits
the same mismatch and is probably running less accurately than it needs to.
*Fix:* one-line change in `load_model`, re-run
`scripts/phase7_inference_fix.py --clips 200`. *Payoff:* a corrected error budget
and a directly better tool.

**2. End-to-end on a real clip — minutes, needs only footage**
The pipeline has never seen real running video. *Payoff:* confirms it works in
situ at all. This is the cheapest unresolved item and blocks nothing else.

**3. Integration test — hours, test engineering**
The 19 tests cover invariants; none runs a phase end to end. Reproducibility is
this project's main claim and rests entirely on re-running scripts by hand.
*Payoff:* guards the claim.

**4. Frame-rate assumption — hours, ablation (harness exists)**
The ~120 fps figure is inferred from stride cadence (0.0221 strides/frame at an
assumed 2.5–3.0 strides/s → 113–136 fps), not documented anywhere. The
resampling step in the tool is untested for effect. *Payoff:* removes an
assumption from the tool's core path.

**5. 2D smoothing choice — hours, signal processing**
The 10 Hz low-pass was inherited from the Ferber pipeline, never tuned or
measured for this use. *Payoff:* small, but the ablation harness already exists.

### Larger, and genuinely structural

**6. Gait event / stride segmentation — 1–2 days plus validation, signal processing**
The tool emits continuous per-frame traces. The research models consume 101-point
stance-normalised curves. **Without this there is no path from a video to the
analysis that took seven phases to build** — the two halves of the project are
disjoint. This is the highest-value larger item.

**7. Detector validation on real video — days, pose benchmarking / data capture**
Keypoint R-CNN's error on real footage is the largest unquantified term in the
tool's budget, and **it cannot be closed with this project's data**: AthleticsPose
does not release its source videos, so no clip with ground truth exists in reach.
Needs either a labelled public dataset with video, or a marker-lab-plus-camera
capture. *Payoff:* turns "unquantified" into a number.

**8. Camera motion / panning — medium, tracking**
The tool refuses on crowded frames but has no handling for a panning camera.
Robustness only. Note `CLAUDE.md` forbids multi-person tracking.

### Explicitly not worth doing

**Ankle dorsiflexion.** H36M-17 has no toe keypoint, so the lifter cannot produce
it. Recovering it means a different pose model or retraining — large effort. But
**phase 4 measured that the ankle contributes nothing** to the limb task:
`wave2` (hip+knee) at 0.615 is not worse than `wave3` (hip+knee+ankle) at 0.610.
Low payoff here.

**Raising the injury signal. Do not attempt.** Phases 5B–5D attacked this five
independent ways: a different measurement modality (39 clinical frontal/transverse
metrics) landed on 0.610 to three decimals; the learning curve is saturated
(+0.001 from the last quarter of the data); there is no severity dose-response
(ρ = +0.044, CI spanning zero); four further feature families failed (velocities
scored 0.499); and neither abstention nor repeat scans rescue deployment. The
ceiling is the signal.

---

## 10. Commands

```bash
# environment
uv sync
cp .env.example .env          # set DATA_ROOT to the Ferber archive

# tests — fast, run these first
.venv/Scripts/python.exe -m pytest tests/ -q

# regenerate every published number from results/*.json
.venv/Scripts/python.exe scripts/phase6_numbers.py

# the research pipeline
.venv/Scripts/python.exe scripts/phase0_inventory.py
.venv/Scripts/python.exe scripts/phase5_limb.py            # the limb task
.venv/Scripts/python.exe scripts/phase3_infer.py           # ~90 min CPU
.venv/Scripts/python.exe scripts/phase4_real_delta.py      # the degradation curve
.venv/Scripts/python.exe scripts/phase4b_operating_point.py
.venv/Scripts/python.exe scripts/phase5b_ceiling.py
.venv/Scripts/python.exe scripts/phase5d_selective.py

# inference-path ablation (gap 1 lives here)
.venv/Scripts/python.exe scripts/phase7_inference_fix.py --clips 200
.venv/Scripts/python.exe scripts/phase7_inference_fix.py --clips 0 --only 0,3

# the video tool
.venv/Scripts/python.exe scripts/video_kinematics.py --video CLIP.mp4 --out DIR

# figures
.venv/Scripts/python.exe scripts/phase4_figure.py
.venv/Scripts/python.exe scripts/phase6_ceiling_figure.py
```

Runtimes are CPU. Phase 3 is ~90 min; phase 4 and 5C are ~25 min each; the
phase 7 ablation is ~40 min at 200 clips.

---

## 11. What not to do

**Do not build a per-runner injury readout**, or any output that could be read as
one. This is not caution — it is a measured result. At the realistic operating
point the model reaches PPV 0.574 against a 0.517 base rate, sensitivity collapses
to 0.130 at 90% specificity, the calibration slope is 0.716, and **it disagrees
with itself on a third of repeat scans of the same runner**. Averaging repeat
scans does not help, which means the error is subject-specific rather than noise:
it is consistently wrong about the same people. See `results/phase04b.md` and
`results/phase05d.md`.

Also out of scope, per `CLAUDE.md`: injury forecasting over time, multi-person
tracking or re-identification, any commercial framing, fusing a kinematic score
with a training-load score, a mobile app, user accounts, a database, and deep
learning on the Ferber tabular/waveform data.

---

## 12. Where to read next

| you want | read |
|---|---|
| the finding, for a reader | `README.md` |
| the project's rules, in full | `CLAUDE.md` |
| what is actually in the data | `docs/data-inventory.md` (beats `docs/ferber-schema.md` where they disagree) |
| why screening failed | `results/phase01c.md`, `results/phase01d.md` |
| the degradation curve | `results/phase02.md`, `results/phase04.md` |
| monocular error and viewpoint | `results/phase03.md`, `results/phase03b.md` |
| why there is no product | `results/phase04b.md`, `results/phase05d.md` |
| why the ceiling is the signal | `results/phase05b.md`, `results/phase05c.md` |
| the video tool and inference fixes | `results/phase07.md` |
