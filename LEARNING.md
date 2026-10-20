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
