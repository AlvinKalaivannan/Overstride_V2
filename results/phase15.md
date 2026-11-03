# Phase 15 — Held-out validation on BML MoVi

> **PRE-REGISTRATION, written 2026-10-20, before a single byte of MoVi has been
> downloaded.** The pipeline frozen below is frozen *now*, in this commit, so the
> freeze is checkable in `git log` against the download timestamp rather than
> asserted afterwards. Results are appended under "Results"; nothing above that
> heading changes.

**Question.** The measurement was built and tuned entirely on AthleticsPose.
Does it hold on a lab it has never seen, filmed with a camera it has never seen?

`CLAUDE.md`: *"There is no pass/fail gate. A loss on held-out data is a result
and goes in the README exactly as prominently as a win."* That stands.

---

## The pipeline, frozen

Nothing below may be changed after MoVi is opened. Every number in Results comes
from exactly this:

| stage | implementation | setting |
|---|---|---|
| decode | `video_kinematics.decode` | native resolution |
| 2D detection | torchvision `keypointrcnn_resnet50_fpn`, DEFAULT weights | `min_size` = the frame's own shortest side; padded person crop, `CROP_PAD = 0.35`, full-frame re-detection when a limb leaves the window |
| smoothing | **none** | `LOWPASS_HZ = 0.0` — phase 13 measured it harmful |
| joint order | `video_kinematics.coco17_to_h36m17` | COCO-17 → H36M-17 |
| rate matching | `video_kinematics.resample` | to `MODEL_FPS = 100.0` |
| lifting | `phase7_inference_fix.lift` | `motionagformer-b-ath-det-coco-v1.ckpt`, `pad="edge"`, `windows="overlap"` |
| angles | `phase3_angles.sagittal_angles` | hip and knee; ankle is NaN, H36M-17 has no toe |
| confidence | `phase14_confidence` | clip-level expected error from `azimuth_proxy` + the published viewpoint curve; per-segment confidence for the **shanks only** |

**Reference point:** the deployable AthleticsPose figure, **3.509° near /
3.618° far** (`results/phase7_baseline_ci.json`). MoVi cannot be compared "on the
same clips", so both numbers are reported side by side and the difference is
called what it is — a change of dataset, camera, athlete population and movement.

## What is used, and what is not

- **PG (Point Grey machine-vision) clips carry the MAE.** The paper states the
  FLIR/Point Grey cameras are synchronised to the mocap, aligned with every
  fourth mocap frame at 30 fps. Synchronisation is a precondition for an angle
  error, so these are the measured clips.
- **CP (hand-held iPhone 7) clips answer §7.5** — "the video tool has never run
  on real footage" — because that is what real footage looks like: 800×600,
  30 fps, shaky. If their synchronisation to mocap can be verified from the
  dataset's own documentation, they carry an MAE too; **if it cannot, they are
  reported as traces with no error figure**, and §7.5 closes on the weaker claim
  rather than on a fabricated alignment.
- **Jogging only.** MoVi's movement 2 is jogging and movement 3 is *running in
  place*, which is not running gait: no flight-phase translation, and a
  stationary subject makes the sagittal camera geometry a different problem.
  Running in place is excluded, and the exclusion is reported with its n.
- **No MoVi frame, crop or rendering is published.** The licence is
  non-commercial research with no redistribution, and it also forbids training
  for commercial use. Summary metrics and plots of *angles* only.

## Checks that must pass before the first number is reported

Per `CLAUDE.md`: *"Unit traps like the AthleticsPose `p2mm` one are checked and
written into `docs/data-inventory.md` for every new dataset before its first
number is reported."* For MoVi:

1. **Mocap units** — mm or m, verified against a segment length that has to be
   human (femur 350–550 mm).
2. **Coordinate convention** — which axis is vertical, and the handedness.
   `sagittal_angles` builds its own frame, so it is robust to rotation but not to
   a mirrored axis, which would flip flexion sign.
3. **Marker set → H36M-17.** MoVi ships Vicon markers and MoSh/AMASS SMPL fits.
   The mapping used is stated explicitly, and a left/right sanity check is run
   (the near limb must be the one nearer the camera).
4. **Video frame rate**, read from the file, not assumed to be 30.
5. **Synchronisation offset** between video and mocap, from the dataset's own
   documentation. A constant offset of a few frames would corrupt every angle
   error; it is checked, stated, and if it cannot be established, the MAE is not
   reported.
6. **Subject identity** across modalities — the video file's subject id must be
   the mocap file's subject id.

All six go into `docs/data-inventory.md` §8.5 before Results is written.

## What is reported

- MAE per joint (hip, knee), **near and far limb separately**, with
  **subject-bootstrap CIs**, and the AthleticsPose figure beside each.
- n per group after filtering, and what was excluded and why.
- The **frame rate penalty, isolated if possible**: MoVi video is 30 fps against
  AthleticsPose's 100 Hz mocap-locked detections, so some of any degradation is
  temporal resolution rather than lab transfer. Where the mocap allows it, the
  same clips are scored with the mocap downsampled to 30 Hz, to separate the two.
- **How often phase 14's confidence flagged the frames that turned out worst** —
  Spearman of stated against realized error on MoVi, using the AthleticsPose-fitted
  model unchanged. Phase 14 found the confidence is largely a viewpoint meter, so
  this is a test of whether that transfers.
- OpenCap Monocular's 4.8° (walking, squats, sit-to-stand) as **context, not a
  head-to-head**, per `CLAUDE.md`.

## What would make this phase wrong

- **Tuning anything after seeing a MoVi number.** If a defect is found that
  forces a pipeline change — as four were found in the deployable path this week —
  the change is made, the old numbers are **voided and kept on disk**, and the
  phase re-runs from the frozen definition above with the deviation recorded.
  That is what happened to phase 14 attempt 1 and it is the only honest route.
- **Reading the degradation as lab transfer when it is frame rate**, or the
  reverse. Hence the 30 Hz control above.
- **n = 3 again.** However many MoVi subjects are used, the AthleticsPose side of
  every comparison rests on S11, S13 and S16.

---

## Deviations from the frozen definition, recorded before any result

The section above was frozen in `64fb0bc` and nothing in it has been edited.
Three things happened between that freeze and the first number, and the rule
this phase set for itself is the one being followed: *"If a defect is found that
forces a pipeline change … the change is made, the old numbers are **voided and
kept on disk**, and the phase re-runs from the frozen definition above with the
deviation recorded."* No reportable number was produced before any of these, so
there is nothing to void.

### D1 — Round F, not round S (`fe247a8`)

**Check 3 of the six failed for round S, irrecoverably.** `S_V3D.tar` ships 12
physical markers — head, wrists and ankles — and **no hip, knee or pelvis**,
which are exactly the joints this phase scores. It also ships no `motions_list`
and no frame flags, so it cannot be cut into movements at all.

Round F passes all six and ships more: 87 markers including `LFWT`/`LBWT` and
`RFWT`/`RBWT`, which *are* the ASIS and PSIS markers, so **H36M's own hip
definition — `mean(ASI, PSI)` — is exactly computable** rather than
approximated. This is a change of *which clips*, not of the pipeline, and it was
forced by a pre-registered check rather than chosen.

The 1.26 GB of round-S video downloaded first is the wrong round and is kept
only as the record of how this was found.

### D2 — The confidence channel was 14× out of scale (`2db742f`)

**A real defect in the deployable tool, not a property of MoVi.** `lift` feeds
channel 2 of the 2D array to the model as a per-keypoint weight. The checkpoint
was trained on AthleticsPose's released detections, whose channel 2 is a
*normalised* confidence — mean 0.8676, sd 0.1164 over 60 clips and 131,121
values. torchvision's `keypoints_scores` are *raw* heatmap scores, [3.9, 18.4]
on MoVi, and `detect_coco17` wrote them straight through.

| | MAE | r vs reference |
|---|---|---|
| raw scores | **30.07°** | +0.02 hip, +0.17 knee |
| fixed | **11.45°** | +0.84 hip, +0.91 knee |

With raw scores the estimated knee flexion reached **+66°**, which
`video_kinematics.PLAUSIBLE` itself calls impossible. The fix is a constant and
the choice is *a priori*: the trained channel is nearly constant anyway (13% CV),
phase 14 measured the detector's own confidence as **anti**-correlated with
thigh error, and anything data-dependent would have to be chosen on ground truth
*with video* — which only MoVi provides, and choosing on MoVi would invalidate
this phase. **Zero parameters are fitted to held-out data.**

**Why no earlier phase could catch it:** every number before this one came from
released `.npy` files that already carry normalised confidences, and
AthleticsPose ships no video. The detector half of this path had never once been
scored against ground truth. Second defect of this family after the COCO/H36M
joint order; both lived in the only code path with no ground truth to check it
against.

### D3 — Time alignment is exact, and better than the freeze assumed

The frozen text listed check 5 as *"synchronisation offset … if it cannot be
established, the MAE is not reported."* It can be, and not as an estimate: MoVi
ships `flags30` and `flags120`, the same movement boundaries expressed in video
and in mocap frames, at a measured ratio of **3.9981** (max deviation from 4 is
0.0112). So video frame *k* scores against mocap frame `f120[0] + 4k` and
**the reference 3D is never resampled**, as `CLAUDE.md` requires.

That also satisfies the pre-registered 30 Hz control *by construction*: the
comparison already happens on a 30 Hz grid against untouched 120 Hz mocap. What
it does **not** isolate is the frame-rate penalty on the AthleticsPose side,
which would need the tune set re-scored at 30 fps — a separate run, named here
so it is not mistaken for done.

### Two corrections to earlier claims of mine

- **`motions_list` order differs per subject.** Jogging is index 18 for subject
  1 and **16** for subject 10; `running_in_spot` moves from 7 to 19. An earlier
  commit message stated the index as if it were a dataset constant. The code
  always read it per subject; the claim about it was wrong.
- **`LHJC`/`RHJC` are the _hand_ joint centres**, not the hip — `markerDescription`
  reads "Left Hand Joint" and their height matches the elbow. The hip joint
  centres are `LHIP`/`RHIP`. With the right marker, the offset from H36M's
  `mean(ASI,PSI)` hip is **104/110 mm** on subject 1 and **90/92 mm** on subject
  10, independently reproducing the **81 mm** phase 12 measured on AthleticsPose.

## The six checks, answered

Full detail in `results/phase15_checks.json` and `docs/data-inventory.md` §8.5.

| # | check | round F |
|---|---|---|
| 1 | mocap units | **PASS** — millimetres. Femur, tibia and hip width all fall inside their pre-registered human ranges (350–550 mm, 300–500 mm). The per-subject lengths are deliberately **not** reported: see `docs/data-inventory.md` §8.5 |
| 2 | axis and handedness | **PASS** — vertical is axis 2, agreed independently by both camera centres sitting at 1.35–1.38 m and by head-minus-pelvis; jogging knee flexion `[-73.5, -8.7]°`, one-signed and anatomical |
| 3 | marker set → H36M-17 | **PASS** for round F, **FAIL** for round S (see D1) |
| 4 | video frame rate | **PASS** — read from the file: 30.000 fps, 800×600 |
| 5 | synchronisation | **PASS** — shipped by the dataset, ratio 3.9981 (see D3) |
| 6 | subject identity | **PASS** — the `.mat` subject field matches the filename for every subject |

**Gap sentinel, found and handled.** `markerGaps == 1` marks a *missing* marker
and Visual3D writes `[0, 0, 0]` for it. Read literally that put the knee joint
centre at the lab origin and drove knee flexion to +148.7°. Gapped markers are
NaN and the frames they touch are **excluded, never interpolated**. The
exclusion is **per quantity**: `sagittal_angles` reads seven joints, so an elbow
gap must not cost a knee frame. Globally it would have been 21.2% of subject 1's
window; correctly scoped it is 3.9% left / 4.5% right, and subject 10 loses
nothing.

## The side-on question, answered

`CLAUDE.md` says phase 15 must settle this first. Both metrics agree, on both
subjects measured:

| camera | `view_ratio` (1 = side-on) | travel azimuth (90° = side-on) | verdict |
|---|---|---|---|
| **PG1** | 0.720 / 0.705 | 75.7° / 75.0° [62–84] | **side-on** |
| **PG2** | 0.468 / 0.465 | 11.0° / 10.5° [4–23] | near-frontal |

**PG1 at 0.71 is essentially AthleticsPose's own median of 0.751**, so the
held-out set is viewed from the geometry this project prescribes — and PG2 gives
the frontal contrast that makes the phase 11 viewpoint curve (3.50° → 1.77°)
measurable on a lab the pipeline has never seen.

## Results

Produced by `scripts/phase15_movi.py` (no arguments) from the frozen pipeline,
**with** the D2 confidence fix, on MoVi **round F**, **jogging only**. Numbers in
`results/phase15_movi.json`; the perfect-2D arm in
`scripts/diagnostics/diag_bisect.py`.

### D4 — Subject_11 is excluded, and the exclusion is PRESCRIBED

The frozen pre-registration says, verbatim:

> Any check that fails and cannot be resolved from the dataset's own
> documentation **stops the MAE for the affected clips** rather than being
> papered over.

**Subject_11 fails check 1.** Its median femur or tibia length falls outside the
pre-registered human range (350–550 mm / 300–500 mm), bars written into
`scripts/phase15_checks.py` before MoVi was downloaded. The per-subject lengths
are not reported here, per the licence note in `docs/data-inventory.md` §8.5;
the verdict is.

**And the reason is visible in the gap sentinel.** Its jogging window loses
**42.0% of frames** — 739 of 1,761 — against 3.9%/4.5% for Subject_1 and
**0%** for Subject_10. The gapped markers are the worst possible set:

| marker | gapped frames | what it is |
|---|---|---|
| `LFWT` | **346** | **Left ASIS** — half of H36M's left hip |
| `CLAV` | **244** | clavicle |
| `THORAX` | **244** | **the trunk reference the body frame is built on** |
| `RFWT` | **208** | **Right ASIS** — half of H36M's right hip |
| `LWJC` | 82 | left wrist centre |
| `LKJC` | 21 | left knee centre |

`sagittal_angles` builds its frame from **pelvis→thorax** and **hip↔hip**.
`LFWT`, `RFWT` and `THORAX` are exactly those inputs, so on this subject the
frame itself is missing for a large fraction of the window — which is also the
most plausible cause of the out-of-range segment length.

**This is an application of a frozen rule, not a choice.** The exclusion is
enforced in code (`phase15_movi.preregistered_checks`), runs **before** any
decoding so it cannot be informed by the error, and is pinned by
`tests/test_phase15_refusal.py`. **The n=3 figure is kept below rather than
deleted.**

### D5 — a third sentinel bug, in the perfect-2D arm, found by cross-checking

**Two code paths computed the same quantity and disagreed, which is the only
reason this was caught.** `scripts/diagnostics/diag_bisect.py` and
`phase15_movi.py --perfect-2d` both lift a projected reference 2D. On
**Subject_10 they agreed exactly** (7.039°); on **Subject_1 they differed by
2°** (5.983° against 7.999°).

The cause: `project_reference` returns **NaN** where a reference marker was
gapped, and `phase15_movi` called `np.nan_to_num(kp, nan=0.0)` — turning 45
of Subject_1's 206 frames into a **skeleton at the image origin**. The scoring
mask excluded those frames, but **the lifter still saw them**, and `lift` blends
50%-overlap windows, so each zero frame corrupted its neighbours' estimates as
well. Subject_10 has **no** gapped frames, so it was unaffected — which is
exactly what localised the fault.

Fixed: incomplete frames are **dropped before lifting**, never zero-filled. That
breaks temporal contiguity, which the lifter also dislikes, but a gap in the
time axis is far closer to its training distribution than a limb at the origin.
After the fix both paths agree to three decimals.

**This is the third sentinel of the same family in this phase** — after
MoVi's `markerGaps` zeros and the detector's all-zero undetected frames — and
the first one that was in code written *during* this phase rather than
inherited. The pattern is now unmistakable: **a missing value represented as a
valid-looking number will be consumed by something downstream that cannot tell
the difference.** The lesson is in `LEARNING.md`.

### The held-out figure — n = 2 after the prescribed exclusion

| camera | limb | hip | knee | **MAE** | 95% CI | n |
|---|---|---|---|---|---|---|
| **PG1** — side-on | **near** | 11.057 | 8.163 | **9.610°** | **[8.892, 10.328]** | **2** |
| **PG1** — side-on | **far** | 10.698 | 10.510 | **10.604°** | **[10.241, 10.966]** | **2** |
| PG2 — near-frontal | near | 9.392 | 8.959 | 9.175° | — | **1, suggestive only** |
| PG2 — near-frontal | far | 10.928 | 11.808 | 11.368° | — | **1, suggestive only** |
| **CP1** — hand-held iPhone | — | — | — | **no error figure — traces only** | — | — |

Per-subject PG1 near: Subject_1 **10.328**, Subject_10 **8.892**.
PG1 far: **10.966**, **10.241**.

**Beside it, before the prescribed exclusion (n = 3, Subject_11 included):**
**9.729° near [8.892, 10.328] / 10.556° far [10.241, 10.966]**, per-subject
near 10.328 / 8.892 / 9.967 and far 10.966 / 10.241 / 10.462. Excluding
Subject_11 moves the near figure by **−0.119°** and the far figure by
**+0.048°** — both far inside the ~0.2° resolution floor, so **the
exclusion does not change any conclusion.** It is done because the rule says so,
not because it mattered.

> ### A two-subject bootstrap CI is barely meaningful, and here it is literally
> ### just the range
>
> A bootstrap over two values can only ever resample those two values, so its
> 2.5th and 97.5th percentiles **are the minimum and the maximum**. The interval
> `[8.892, 10.328]` carries **no information beyond the two subject means
> themselves** — it is not an interval estimate, and it must not be read as
> one. The n=3 interval is degenerate the same way: also exactly `[min, max]`.
>
> Treat every CI in this phase as **the observed spread of a handful of
> subjects**, not as uncertainty about a population. The honest statement is
> "two subjects, 9.6° and one of them at 8.9° and the other at 10.3°",
> and `CLAUDE.md` already warns that at three held-out subjects every marginal
> interval overlaps every other.

### The perfect-2D arm, same exclusion, n = 2

The reference 3D projected through the same camera — a perfect 2D,
H36M-ordered by construction — through the identical lifter:

| camera | limb | hip | knee | MAE | 95% CI (= range) | Subject_1 | Subject_10 |
|---|---|---|---|---|---|---|---|
| **PG1** | near | 8.366 | **4.656** | **6.511°** | [5.983, 7.039] | 5.983 | 7.039 |
| **PG1** | far | 9.017 | **5.013** | **7.015°** | [6.576, 7.454] | 6.576 | 7.454 |
| PG2 | near | 7.087 | 7.005 | 7.046° | — | 7.046 | — |
| PG2 | far | 6.863 | 7.725 | 7.294° | — | 7.294 | — |

Note the split by joint: with a perfect 2D the **knee reaches 4.656°** while
the **hip stays at 8.366°**. The knee is close to the 5.16° AthleticsPose
perfect-2D figure; the hip is nearly double it, and that is where this phase's
error lives.

For scale, `results/phase07.md` §5 independently measured perfect 2D into
**this same checkpoint** at **5.16°** on AthleticsPose. So the lifter lands
within ~1.4° of its own AthleticsPose behaviour on a lab it has never seen,
which is a much smaller transfer penalty than the headline suggests.

### How the error splits — and the split does NOT generalise

| | Subject_1 | Subject_10 | n=2 mean |
|---|---|---|---|
| detected 2D, near | 10.328 | 8.892 | 9.610 |
| perfect 2D, near | 5.983 | 7.039 | 6.511 |
| **difference = the 2D detector's cost** | **4.345** | **1.853** | **3.099** |

**The detector's cost differs 2.3× between two subjects.** An earlier draft of
this report said "about 4.3° of the 10.3° is the 2D detector" on the strength
of Subject_1 alone; **that does not hold on Subject_10, where it is 1.9°.**
At n=2 the only defensible statement is: **roughly a third of the error is the
detector and roughly two thirds is already present with a perfect 2D, and both
shares vary substantially by subject.**

### The hip is biased, the bias is NOT a constant

With a **perfect** 2D — no detector in the loop at all:

| | near hip bias | near hip r | near knee bias | near knee r |
|---|---|---|---|---|
| Subject_1 | **−7.235°** | +0.885 | +2.100° | +0.956 |
| Subject_10 | **−3.347°** | +0.587 | +0.892° | +0.972 |

Hip error is dominated by a **one-signed negative offset** on both subjects,
which is the signature of a systematic effect rather than noise — but it is
**−7.2° on one subject and −3.3° on the other**, so it is *not* a single
fixed convention offset. A pure marker-convention difference should be nearly
the same for both, because both use the same marker set and the same
`MOVI_TO_H36M` mapping. **A subject-varying bias points at something that
interacts with the individual's posture** — most likely trunk lean, since hip
flexion here is measured against the pelvis→thorax axis, and MoVi's `THORAX`
virtual marker need not sit where H36M's thorax does. **Not resolved. Do not
"correct" for it with a constant.**

Knee behaves differently and better: small positive bias, **r +0.956 and
+0.972**, so the knee trace is genuinely tracked.

### This is a loss, and it is reported as one

**`CLAUDE.md`: *"There is no pass/fail gate. A loss on held-out data is a result
and goes in the README exactly as prominently as a win."*** So, plainly:

**The measurement is about 2.7× worse on a lab it has never seen.**
**9.610° near against the deployable AthleticsPose baseline of 3.5093°, and
10.604° far against 3.6177°.** No interval here comes close to the baseline.

**The baseline is not a comparison on matched data and must never be quoted as
one.** It is **a different dataset, a different camera, a different population
and a different movement**: AthleticsPose is 592 clips of *sprinting and
running* by *athletes*, scored on *released 2D detections* with no video
involved, on subjects S11/S13/S16. MoVi is *jogging* by *students and staff* in
a lab volume, from *real video decoded by this tool*. "2.7×" measures the
change in all four at once, and **nothing here attributes it to any one of
them.** The same AthleticsPose figure at subject level is 3.4528°
[3.266, 3.695] near and 3.5597° [3.130, 3.903] far
(`results/phase7_baseline_ci.json`).

### What the structure of the error says

- **The measurement is tracking, not guessing.** Knee correlation against the
  reference is **+0.924 / +0.930 / +0.937** across the three subjects as
  detected — high and stable, and it survives the exclusion. Hip is weaker
  and more variable, **+0.841 / +0.665 / +0.709**, and **hip carries the larger
  error at every camera and limb.**
- **Detection was 100%** — 890 of 890 PG frames, 0 full-frame re-detections
  needed on the tracked crop. The error is not missing detections.
- **The far limb is worse than the near limb** on PG1 (10.604 vs 9.610),
  matching the direction on AthleticsPose (3.6177 vs 3.5093). `CLAUDE.md`
  requires them reported separately and they are.
- **The lifter is sensitive to the VALUE of the confidence channel, not only to
  its scale — recorded rather than exploited.** The perfect-2D arm scores
  4.493° near on Subject_1 with channel 2 set to 1.0 and **5.983°** with
  `TRAIN_CONF_MEAN` (0.8676, the tune-set mean) — ~1.5° from a constant,
  large next to the 0.2° floor. **The a priori choice stands**: 0.8676 is the
  mean of the distribution the checkpoint was trained on, measured on the *tune*
  set, and picking 1.0 *because it scores better on MoVi* would be tuning on a
  held-out set and would invalidate this phase. Choosing properly needs a tune
  set that ships video, which this project does not have. `TRAIN_CONF_SD` is
  defined and **unused**.

### n, exclusions, and what was left out

| subject | camera | jogging idx | video frames | detected | scored near / far | excluded |
|---|---|---|---|---|---|---|
| Subject_1 | PG1 | 18 | 206 | 206 | 198 / 197 | gap sentinel, ~4% |
| Subject_1 | PG2 | 18 | 206 | 206 | 197 / 198 | gap sentinel, ~4% |
| Subject_10 | PG1 | 16 | 243 | 243 | 243 / 243 | none |
| ~~Subject_11~~ | ~~PG1~~ | 15 | 441 | 441 | — | **EXCLUDED — fails check 1; see D4** |

- **`running_in_spot` is excluded**, per the frozen definition: it is running
  without flight-phase translation, which is not running gait. It sits at
  `motions_list` index 7, 19 and 7 for the three subjects.
- **The jogging index differs on every subject — 18, 16, 15.** `motions_list`
  order is randomised per subject, so it is read per subject; a hardcoded index
  would have measured three different movements.
- **Subjects 1, 10 and 11 were chosen by archive order** in
  `F_Subjects_1_45.tar`, for streaming cost and nothing else. No measurement
  informed the choice, because none existed when it was made.

### CP1 — traces, with no error figure

**The frozen pre-registration fires here, as written.** It said CP clips carry
an MAE *"if their synchronisation to mocap can be verified from the dataset's own
documentation ... if it cannot, they are reported as traces with no error
figure"*. It cannot be verified, for two independent reasons:

1. **`Camera Parameters.tar` covers PG1 and PG2 only — zero CP entries.** The
   hand-held phone cameras ship no calibration at all.
2. **`flags30` is indexed to the PG recording.** CP is a *different recording* of
   the same session: **1920×1080 at 30.009 fps with 4,369 frames** for
   Subject_1, against PG's **800×600 at 30.000 fps with 3,441** — 928
   frames more. So **CP frame *k* is not the same instant as PG frame *k***, and
   the shipped flags carry no CP correspondence.

An earlier run did apply the PG window to CP and produced a figure. **That
figure is void and is not reported here.** Its own diagnostics say why:
r = **−0.077** hip and **+0.022** knee — no relationship at all, the same
signature the D2 confidence bug produced, and for the same reason, comparing two
unrelated signals. The forensic account is in `docs/context-dump-2026-11-02.md`
§6.5.

Recovering the offset by cross-correlating CP against PG or against the mocap
would be **estimating** the synchronisation, which is precisely what the frozen
text says disqualifies the MAE. So it is not done. Instead
**`scripts/phase15_movi.py` now refuses to compute an MAE for an uncalibrated
camera at all** — structurally, before any scoring code is reached — and
`tests/test_phase15_refusal.py` pins that so the void figure cannot return.

**Consequence for §7.5 of `results/phase11.md`** — *"the tool has never
run on real footage"*. It still has not, in the sense that matters:
**phase 15 has not measured the tool on consumer phone video.** PG1 and PG2 are
calibrated machine-vision cameras. MoVi has the phone footage §7.5 needs, and
this phase could not score it. §7.5 stays **open**.

## Subject designation, fixed 2026-11-02 before any further download

The result above rests on **two** subjects, and they disagree by **2.3×** on
the detector's share and **2×** on the perfect-2D hip bias. More subjects are
needed to diagnose that, but spending every remaining subject on diagnosis would
leave nothing to test a fix against. So the roles are fixed **now**, before the
data exists on disk:

| set | subjects | role |
|---|---|---|
| spent | `Subject_1`, `Subject_10` | the headline result above |
| spent / excluded | `Subject_11` | excluded by the frozen rule, D4 |
| **DIAGNOSTIC** | `Subject_12`, `Subject_13`, `Subject_15` | PG1 only. **Spent on arrival** — used to work out *why*, never to show that a fix works |
| **SEALED TEST** | `Subject_16`, `Subject_17`, `Subject_18`, `Subject_19`, `Subject_2` | named and costed, **not downloaded**. Opened only against a fix frozen beforehand |

`Subject_14` is skipped: no round F video of any kind.

**Nothing below this line changes the phase 15 headline.** The diagnostic rows
are labelled as diagnostic, they are not pooled into the result, and the
reported figure stays **9.610° near / 10.604° far on PG1 at n = 2**. Sizes,
file ids and the full rationale are in `docs/data-inventory.md` §8.5.

### What this phase does NOT establish

- **Nothing about phone footage** — see above.
- **Nothing attributable to lab transfer specifically.** The 2.8× bundles
  dataset, camera, population and movement, and the perfect-2D arm says most of
  the gap on the one subject tested is downstream of the 2D rather than a
  property of the new lab.
- **The AthleticsPose-side frame-rate control was not run.** MoVi's comparison is
  already on a 30 Hz grid against untouched 120 Hz mocap, so that half is
  satisfied by construction (D3), but the tune set has **not** been re-scored at
  30 fps, so the frame-rate penalty is **not** isolated.
- **The viewpoint curve is not confirmed, and may not replicate.** PG2 is
  near-frontal (`view_ratio` 0.468, azimuth 10.5°) and scored **9.175°
  near** against PG1's **10.328°** on the same subject — frontal was *not*
  worse, which is the opposite direction to phase 11's measured
  3.50° → 1.77°. **n = 1**, it is marked suggestive, and it is not
  claimed as a finding either way. Two more PG2 clips would settle it.

### What is still open

1. ~~Subject_11 needs a decision.~~ **CLOSED — excluded, see D4.** It was a
   prescribed exclusion, it is enforced in code, and it moved the figure by
   −0.119° near, well inside the resolution floor. **n is now 2 on PG1**,
   which is the real cost: see the warning about two-subject intervals above.
   A fresh, never-opened set is costed in `docs/data-inventory.md` §8.5.
2. **The hip offset is now measured (−7.2° / −7.9° with a
   perfect 2D) but not explained.** The next step is to test the trunk
   convention directly — MoVi's `THORAX` virtual marker against H36M's thorax
   — because the hip angle is measured against the trunk and a mis-placed
   trunk reference offsets it by a constant. If confirmed, report a
   convention-corrected figure *beside* the raw one, never instead of it.
3. ~~Run `--perfect-2d` on subjects 10 and 11.~~ **Done for Subject_10** — and
   it showed the split does **not** generalise (4.345° vs 1.853° detector
   cost). Subject_11 is excluded, so n=2 is the ceiling until fresh subjects are
   taken. **This is the strongest argument for the fresh set**: two subjects
   disagreeing by 2.3× on the most important decomposition in the phase is not
   a basis for a decision about the detector.
4. **OpenCap Monocular's 4.8°** (walking, squats, sit-to-stand) is cited as
   **context, not a head-to-head**, per `CLAUDE.md`. It is a different task on a
   different movement set and is not a baseline for this number.
