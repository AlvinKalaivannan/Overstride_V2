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
