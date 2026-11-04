# Learning log — Overstride

`Learning mode: DEEP` (recorded in `CLAUDE.md`). Computer vision, 3D geometry,
evaluation protocol and measurement statistics — the concepts here are the point,
not the code.

Entry format is fixed by `~/.claude/CLAUDE.md`. Entries are appended when
something genuinely new is learned, got wrong, or needed a non-obvious
explanation. Trivia is not logged.

Started 2026-10-19, part-way through the measurement track (phases 12–16), so
phases 0–12 are not represented here. The two entries below are from the work
that opened this file, not a backfill.

## 2026-10-19 — Why a mean MAE is not a baseline you can gate against

- What it is: a gate like "beat 3.43° with a CI excluding zero" needs three
  things the published number did not have — the *configuration* that produced
  it, a *spread*, and the *same inputs on both sides*. Phase 7 measured six
  configurations; 3.43° was the one phase 3 ran, and the tool actually ships
  3.51°. The gate could therefore be passed by swapping the 2D detector for
  AthleticsPose's fine-tuned one (3.18°) with no method work at all.
- Where it shows up here: `scripts/video_kinematics.py:241,244` loads the
  `det-coco` checkpoint with `pad="edge", windows="overlap"` — the 3.51° row of
  `results/phase7_inference_full.json`.
- Gap I had: treating a single published mean as "the baseline", when a baseline
  is a *measurement protocol* — a configuration, an estimator, and an interval.
  `results/phase7_inference_full.json` carried `n: 592` and no spread, so no
  interval was possible from it at all.
- Go deeper: Demšar, "Statistical Comparisons of Classifiers over Multiple Data
  Sets" (JMLR 2006), §2 — why paired comparisons on the same items are the right
  test, and why the overlap of two marginal intervals is not.

## 2026-10-19 — A self-referential frame cannot measure its own axes

- What it is: `sagittal_angles` builds the body frame *from* the pelvis and the
  trunk (`up = unit(thorax - pelvis)`, `ml = unit(r_hip - l_hip)`). Expressed in
  that frame, the trunk *is* the vertical and the pelvis has no sagittal
  component, so their "error" is identically zero — not small, absent. Only the
  thigh and shank have a direction error to measure.
- Where it shows up here: `scripts/phase14_confidence.py:body_frame` and
  `SEGMENTS`; `tests/test_confidence.py::test_pelvis_and_trunk_cannot_have_a_sagittal_error`.
- Gap I had: carrying phase 12's six-segment list into phase 14 without checking
  it against the quantity being measured. The lifter's output is scale-free,
  root-relative and gravity-free, which is *why* the frame has to be built from
  the body — and that is exactly what makes two of its segments unmeasurable.
- Go deeper: Wu & Cavanagh, "ISB recommendations for standardization in the
  reporting of kinematic data" (J. Biomech. 1995) — segment coordinate systems,
  and what a joint angle is actually relative to.

## 2026-10-19 — A bug lives in the seam between two validated halves

- What it is: `video_kinematics.py` fed torchvision's **COCO-17** keypoints into
  a lifter whose input convention is **H36M-17** — a nose where a pelvis belongs.
  The detector half had a test (torchvision emits COCO-17: true) and the lifting
  half had a test (angles from AthleticsPose keypoints: true). Neither test
  crossed the join, and that is where the defect was.
- Where it shows up here: `scripts/video_kinematics.py:coco17_to_h36m17` (the
  fix) and
  `tests/test_video_pipeline.py::test_the_lifters_input_convention_is_h36m_read_from_the_released_data`.
- Gap I had: trusting a docstring's claim about a data convention. The order was
  knowable from the data in three ways — which joints are lowest in the image,
  per-joint agreement with 2D ground truth, and whether upstream reorders before
  the model. A comment is not evidence.
- Also: **a plausibility range cannot catch this.** The lifter is a learned
  prior, so it returns a human-shaped pose for almost any input, and hip ∈
  [−40°, 60°] passes happily on a scrambled skeleton. Plausible is not correct.
- Go deeper: MotionBERT's `coco2h36m` in `lib/utils/utils_data.py` is the
  canonical version of this mapping — worth diffing against the one written
  here, particularly its neck and head construction.

## 2026-10-19 — Self-normalisation can cancel the very thing you are measuring

- What it is: to estimate camera azimuth from video, phase 14's frozen protocol
  divided the **apparent 2D hip width** by the **lifter's own 3D hip width**.
  But the lifter's 3D comes from that same 2D, so the camera's compression
  appears in numerator and denominator and divides out. Measured against the
  ground-truth viewpoint metric the feature scored Spearman **−0.02**; the raw 2D
  ratio, divided instead by a *population* anthropometric constant, reaches
  Pearson **−0.973** over the same 592 clips.
- Where it shows up here: `scripts/phase14_confidence.py:features_from_video`,
  `R_POP`, and the diagnosis section of `results/phase14.md`.
- Gap I had: treating "divide by something to cancel the unknown scale" as
  automatically safe. The denominator has to be *independent* of the effect being
  measured. Here the geometry is explicit: `w_2d = s·W·|cos θ|` and
  `t_2d = s·T`, so `w_2d/t_2d = (W/T)·|cos θ|` — the trunk cancels the image
  scale `s` precisely because the trunk is **not** affected by azimuth, and the
  estimate's own width is affected, which is why it fails.
- Go deeper: any projective-geometry text on cross-ratios — the general lesson is
  that a scale-invariant quantity must be built from measurements that transform
  differently under the nuisance parameter.

## 2026-10-19 — A permutation control tells you *which level* your model works at

- What it is: phase 14's confidence scored Spearman 0.37 against realized error.
  Shuffling the stated error **within each clip** — destroying frame-level
  ordering while leaving each clip's overall level untouched — still scored 0.28.
  So the model ranks *clips*, and the per-frame information is worth only ~0.09
  of the 0.37.
- Where it shows up here: `scripts/phase14_confidence.py` (the `perm` block) and
  the control table in `results/phase14.md`.
- Gap I had: reading a single correlation as evidence of frame-level skill. A
  correlation computed over pooled frames absorbs *every* level of variation at
  once — between subjects, between clips, within a clip — and the headline number
  cannot tell you which one it came from. The fix is cheap: shuffle within the
  grouping you want to rule out and re-score.
- This is the same idea as the project's subject-grouped splits, applied to the
  *estimator* instead of the split: both exist to stop coarse-grained variation
  masquerading as fine-grained skill.
- Go deeper: Anselin's spatial-lag intuition, or more practically the
  within/between decomposition behind mixed-effects models — a pooled
  correlation is a weighted blend of a within-group and a between-group one.

## 2026-10-19 — A mean hides which error matters, and some error is definitional

- What it is: the generic COCO detector and the one fine-tuned on this dataset
  have **the same** mean 2D error over the eight joints the sagittal angles use —
  12.28 px against 12.18 px — yet differ by 0.33° in angle error. Fine-tuning did
  not reduce the error, it **moved** it: knees and ankles 3.2–4.1 px instead of
  5.4–6.4, hips slightly worse. The angles depend on the distal joints, so that
  trade is a win that the mean cannot see.
- And the hips sit at 21–23 px for *both*, because H36M's hip is the mean of the
  ASI and PSI markers — a point **inside the pelvis** — while an image-based
  detector marks a hip on the body surface. That error is **definitional**: no
  detector can remove it without predicting mocap marker conventions. Phase 12
  measured the same thing in 3D (81 mm, +18.2% femur inflation).
- Where it shows up here: `scripts/phase7_detector_headroom.py` and
  `results/phase07.md` §5.
- Gap I had: treating a single "2D keypoint error" number as the quantity a
  detector should be judged on, and treating the full gap to ground truth as
  *reducible*. Decompose first: which joints does the downstream quantity
  actually depend on, and how much of the residual is a definition rather than a
  mistake?
- Go deeper: the problem has a name — **hip joint-centre estimation**. Start
  with the regression-equation literature (Harrington et al. 2007 is the usual
  reference for predicting the hip centre from pelvic landmarks) and note that
  markerless and marker-based kinematics are therefore compared with a stated
  offset rather than expected to agree. Verify the reference before citing it.

## 2026-10-19 — A low-pass helps only where the noise beats the signal

- What it is: I pre-registered the expectation that filtering the 2D could remove
  about **1.39 px** of the 16.34 px error, by decomposing the *error* signal into
  a constant bias (13.21 px), a low-frequency part (6.45 px) and a
  high-frequency part above 10 Hz (4.49 px). Measured, filtering made the 2D
  error **worse**: 12.195 px unfiltered, 13.075 at 10 Hz, 15.591 at 6 Hz.
- Gap I had: I measured the numerator and assumed the denominator. A low-pass
  removes everything above the cutoff — error *and* signal — so it only helps
  where the error's high-frequency energy exceeds the **true trajectory's**
  high-frequency energy. A running ankle at 100 Hz has real content up there:
  foot strike is near-impulsive, and smoothing it smears genuine motion. The
  quantity that decides this is the per-band SNR, not the error spectrum alone.
- Where it shows up here: the expectation in `results/phase13.md`, the
  measurement in `scripts/phase13_filter.py`, and the correction in that report's
  Results.
- Go deeper: the Wiener filter makes this explicit — the optimal gain in each
  band is `S/(S+N)`, so it keeps a band when signal dominates and suppresses it
  when noise does. A fixed-cutoff Gaussian is that filter with the SNR assumed
  rather than measured.

## 2026-10-19 — A model learns its input pipeline, not just its task

- What it is: the lifter was trained on one detector's 2D, and it has absorbed
  that detector's error distribution. Three independent measurements, all the
  same day, all pointing the same way:
  1. perfect ground-truth 2D into the shipped checkpoint: **3.51° → 5.16°**,
     *worse*, and monotone as the input is moved toward truth;
  2. the same input into a checkpoint trained on it: **1.47°**, far better;
  3. smoothing the 2D before lifting: worse at every cutoff, monotone again.
- So "improve the input" is not automatically "improve the system". Anything
  that changes the input distribution — a better detector, a filter, a different
  resolution, int8 quantisation — is a **pairing** change, and the model has to
  come with it.
- Where it shows up here: `results/phase07.md` §§5–6, `results/phase13.md`, and
  the `CKPT_FOR` table in `scripts/phase7_inference_fix.py` that phase 7 added
  for this reason.
- Why it matters beyond this project: it is why the free speed-ups taken in
  phase 16 were chosen specifically to **not** touch the weights or the keypoint
  distribution — native resolution and a person crop change how the detector is
  *called*, not what it outputs.
- Go deeper: this is dataset shift, and specifically covariate shift between
  training and deployment inputs. Quiñonero-Candela et al., *Dataset Shift in
  Machine Learning* (2009) is the standard reference.

## 2026-11-02 — An auxiliary input channel has a scale, and the scale is part of the contract

- What it is: when a model takes a side channel alongside its main input — a
  confidence, a mask, a weight — that channel has a **distribution** in
  training, and the distribution is as much part of the interface as the tensor
  shape. Shape mismatches raise an exception. **Scale mismatches do not: they
  run, and they return numbers.** Here the lifter's third channel was trained on
  normalised confidences (mean 0.868, sd 0.116) and was being handed
  torchvision's raw heatmap scores (mean 12.4, range [3.9, 18.4]) — ~14× out.
  The pipeline did not fail; it produced an MAE of 30° with a correlation of
  **+0.02** against ground truth and an estimated knee flexion of +66°.
- Where it shows up here: `scripts/video_kinematics.py`, the `TRAIN_CONF_MEAN`
  comment and `scores_to_train_scale`, which is deliberately a **named step at
  the seam** rather than a quiet edit inside `detect_coco17`.
- Gap I had: I treated "shape is right and values look like plausible numbers"
  as evidence the array was correct. A confidence of 12.4 *is* a plausible
  number — it is just not a plausible confidence. The check I was missing is:
  compare the distribution of every input channel against the distribution the
  weights were trained on, before trusting a single output.
- Why it was invisible for so long: every earlier number in this project came
  from released `.npy` files that already carried normalised confidences, and
  AthleticsPose ships no video — so the detector half of the deployable path had
  **never been scored against ground truth**. Two defects of this exact family
  (this one and the COCO/H36M joint order) both lived in that one unverified
  path. The lesson is structural: **the code path with no ground truth is where
  your bugs are**, and the count of bugs found there is not evidence it is now
  clean.
- Go deeper: same covariate-shift family as [[a-model-learns-its-input-pipeline]],
  but the shift is in an auxiliary channel rather than the main one, which makes
  it far easier to miss.

## 2026-11-02 — Bisect a pipeline with a synthetic-perfect input

- What it is: an end-to-end error tells you the pipeline is broken but not
  *where*. To localise it, feed the back half a **perfect version of its own
  input**, built from ground truth, and see which half the error follows. Here
  MoVi ships camera intrinsics, extrinsics and radial distortion, so the
  reference 3D could be projected into the image and handed to the lifter —
  H36M-ordered by construction, no detector, no joint conversion. That gave
  4.5°, matching the 5.16° phase 7 had independently measured for perfect 2D
  into the same checkpoint, which cleared the lifter, the ground-truth mapping,
  the projection, the time alignment and the angle code in one test.
- Where it shows up here: the diagnostic chain behind commit `2db742f`, and
  `results/phase15.md` §D2.
- Gap I had: my first instinct was to suspect the thing that had *recently*
  changed (my own ground-truth mapping) rather than run the test that would
  partition the pipeline. A bisection costs one run and rules out five
  components; guessing costs one run and rules out one.
- Also learned, smaller but it bit me: **the mean of directions is not the mean
  of angles.** My side-on cross-check averaged velocity vectors and then took
  the angle, which partly cancels for a subject shuttling back and forth in a
  lab volume — it read 88.6° and 36.3° for two clips whose `view_ratio` was
  0.72 and 0.71. Computing the angle per frame and taking the median gave 75.7°
  and 75.0°, agreeing with the frozen metric. This is why circular quantities
  get circular statistics.
- Go deeper: Mardia & Jupp, *Directional Statistics* (2000), ch. 2 for the
  circular mean; for the debugging method, delta debugging / bisection as in
  Zeller, *Why Programs Fail*, ch. 13.

## 2026-11-02 — Signed bias and correlation separate a convention error from noise

- What it is: an MAE is a magnitude and throws away the sign, so it cannot tell
  "this measurement is noisy" from "this measurement is consistently shifted".
  Two cheap extras do: **Pearson r** against the reference, and the **mean
  signed error**. High r with a large MAE and a one-signed bias is a *systematic
  offset*; low r with a large MAE is an unrelated signal; high r with a small
  bias is genuine noise.
- Where it shows up here: `scripts/diagnostics/diag_bisect.py`, and
  `results/phase15.md` Results. Feeding the lifter a **perfect** 2D on MoVi gave
  hip MAE 8.29° with r = +0.885 and a signed bias of **−7.235°** — so almost all
  of the hip error is a constant shift, present with no detector in the loop at
  all. That points at a **convention** difference (which trunk reference defines
  "hip flexion") rather than at anything the detector or lifter is doing wrong.
- Gap I had: I reported MAE alone for several phases and read a large number as
  "degraded measurement". Twice this session a near-zero r showed the signal was
  *unrelated* rather than degraded (the confidence bug, then the CP1 window bug),
  and once a large bias showed it was *shifted*. Three different faults that an
  MAE renders identically.
- Also learned, and uncomfortable: **the lifter is sensitive to the VALUE of its
  confidence channel, not just its scale.** The same perfect-2D arm scores 4.49°
  with channel 2 = 1.0 and 5.98° with 0.8676, the tune-set mean — ~1.5° from a
  constant, against a 0.2° resolution floor. The tempting move is to pick 1.0
  because it scores better, and that would be **tuning on a held-out set**. The
  a priori choice stands and the sensitivity is reported instead.
- Go deeper: this is the bias–variance decomposition of an error, applied to a
  measurement rather than to a model. For the circular case see Mardia & Jupp,
  *Directional Statistics*, ch. 2 on the mean resultant direction.

## 2026-11-02 — A missing value dressed as a valid number will always be consumed

- What it is: whenever "absent" is encoded as something the type system accepts
  — `0`, `999`, `[0,0,0]`, a zeroed row — some consumer downstream will read it
  as data, because nothing distinguishes it from a real measurement. Three
  instances in one phase: MoVi's `markerGaps` writes `[0,0,0]` for a dropped
  marker (knee centre at the lab origin, +148.7° flexion); `detect_coco17`
  leaves an undetected frame all-zero; and my own perfect-2D arm called
  `np.nan_to_num(kp, nan=0.0)`, feeding the lifter 45 skeletons at the image
  origin.
- Where it shows up here: `scripts/phase15_checks.py:to_h36m`,
  `scripts/phase15_movi.py` (the `detected` mask and the perfect-2D drop), and
  `results/phase15.md` D5.
- Gap I had: I treated "excluded from scoring" as equivalent to "removed". It is
  not. The third bug excluded the bad frames from the **error computation** but
  still handed them to the **model**, and `lift` blends 50%-overlap windows, so
  each zero frame corrupted its neighbours too. **Masking an output is not the
  same as never putting the value in.** Remove it at the earliest stage that
  touches it.
- How it was caught, and this is the transferable part: **two code paths
  computed the same quantity and disagreed.** `diag_bisect.py` and
  `phase15_movi --perfect-2d` matched exactly on the subject with no gaps and
  differed by 2° on the subject with 45 gapped frames. Neither number was
  obviously wrong on its own. A redundant implementation is a cheap oracle, and
  the disagreement pattern — identical where the suspected cause is absent —
  localises the fault immediately.
- Go deeper: this is the "nullable by sentinel" anti-pattern; Hoare's
  "null references: my billion-dollar mistake" is the usual reference, and in
  arrays the right answer is a separate validity mask or a masked array
  (`numpy.ma`) rather than an in-band magic value.

## 2026-11-03 — Reference frames decide which errors cancel, and a tune-set gate can rank measurements wrongly
- What it is: two ways to report the same limb motion behave completely
  differently under estimation error. A **within-frame difference** (knee
  flexion = shank inclination − thigh inclination) is invariant to rotating the
  frame it is measured in, so any error common to both segments cancels. An
  angle measured **against an external axis** (thigh vs gravity, hip vs
  pelvis→thorax) keeps that error in full, because there is nothing to subtract
  it against.
- Where it shows up here: `scripts/phase15d_segments.py:inclination` and
  `segment_angles` (the `knee_from_segments` identity), and the sealed-test
  far-limb table in `results/phase15_sealed.md`.
- Gap I had: I expected the far-limb penalty to be one number for the method.
  It is not — it is a property of the *quantity*. On five held-out subjects the
  far limb cost **+1.97° (thigh) and +1.51° (hip)**, CIs excluding zero and the
  same sign on 5/5, but **+0.28° on the knee with the CI spanning zero**.
  Occlusion degrades the far limb's *depth*; that rotates an externally
  referenced direction systematically, and largely cancels in a difference.
  Pooling near and far, or reporting only the knee, would have hidden it.
- The second, sharper lesson: **a gate set on the tuning set can be confidently
  wrong about which measurement is good.** Phase 15D's G1 rejected
  gravity-referenced thigh inclination (6.69° vs a 4.12° bar) because the
  trunk-referenced hip scored better *on AthleticsPose*. On a lab the model had
  never seen, the thigh was the only quantity that held its value (5.40°) while
  the hip degraded 2.73×. The tune set could not distinguish "accurate" from
  "accurate because the model memorised this dataset's convention" — only
  held-out data separates those, which is the whole argument for sealing a set
  and for pre-registering before opening it.
- Go deeper: this is the measurement-theory distinction between a *relative*
  and an *absolute* observable, and it is the same reason differential GPS and
  lock-in amplifiers work — subtract two readings sharing a common-mode error.
  For the gate half, read on **covariate shift** and the difference between
  validation and test sets (Hastie, *ESL* §7.2 "Bias, Variance and Model
  Complexity", especially the distinction between tuning and test error).
