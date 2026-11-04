# Phase 15E — the 30 fps control

> **PRE-REGISTRATION, written 2026-11-03.** Committed **alone** before the
> measurement is run. Everything above the "Results" heading is fixed by this
> commit and is not edited afterwards. **Tune set only — AthleticsPose. No MoVi
> file is opened by this phase.**

## The question

**How much of the error comes from 30 fps input alone, with no change of lab?**

The sealed test measured 8.410° (knee, near) and 5.401° (thigh, near) on MoVi
against 4.071° and 6.694° on AthleticsPose. That gap is usually described as "a
different lab" — but it bundles at least five things: a different population, a
different camera, a different marker convention, a different movement (jogging
against running), and **a different frame rate**. MoVi's PG1 runs at 30.000 fps;
AthleticsPose is captured at 100 Hz.

This phase isolates the last one. It changes **nothing but temporal density** on
a single dataset, so whatever it finds is attributable to frame rate and to
nothing else.

## AthleticsPose is 100 Hz, from the dataset's own metadata

Read from the shipped Qualisys QCA export, `camera_params/*.json`,
`Timebase/Frequency`: **100.0 Hz on all 14 capture sessions**, running and
sprint included, 8 cameras each.

**This supersedes an inference.** Phase 7 estimated 113–136 fps from stride
cadence (`results/phase7_inference.json`, 0.02206 strides/frame under an
assumed 2.5–3.0 strides/s). The documented rate is 100 Hz, so the cadence
assumption was 13–36% too high: at 100 Hz the same measurement gives **2.21
strides/s**. The figure is recorded here because this phase depends on it, and
because `docs/REVIEW.md` item 4 cites the cadence inference.

**100 Hz is also exactly `video_kinematics.MODEL_FPS`.** The checkpoint was
trained at its own dataset's native rate, which has a consequence this phase
relies on and which is verified below.

## The two paths, and why the native arm is a true no-op

The deployed path is `phase15b_conventions.lift_2d`:

```
lift_2d(model, kp, fps) = resample(lift(resample(kp, n*MODEL_FPS/fps),
                                        "edge", "overlap"), n)
```

- **Native arm, fps = 100.** `n_model = round(n * 100/100) = n`, and
  `resample(x, len(x))` is **bit-identical** to `x` (verified: `np.array_equal`
  True, max |diff| 0.0 — it is `np.interp` onto the same grid). So the native
  arm reduces **exactly** to the `lift(det, "edge", "overlap")` call that
  produced the committed AthleticsPose baselines. **There is no resampling
  confound in the control.**
- **30 fps arm.** A clip of 83 native frames becomes 25 real frames, which
  resample **up** to `round(25 * 100/30) = 83` model-frames — the same count,
  the same window structure, the same edge padding. The only difference is that
  **two thirds of the temporal detail was discarded and linearly interpolated
  back.**

That is the isolation: identical model input length, identical padding,
identical scoring frames. Only the information content differs.

## Method, fixed now

**Subjects: `S11`, `S13`, `S16`** — `phase3_infer.TEST_SUBJECTS`, the
checkpoint's held-out subjects. **Action: `running`. 592 clips**, all of which
have all four required inputs present (verified: 0 missing `gt_markers3d_by_cam`,
0 missing `det_markers2d_by_cam_coco`, 0 missing `raw_markers_in_world`).

> ### Sprint is excluded, and this is a departure from the task as specified
>
> The task asked for running **and** sprint. **There is no sprint data for these
> subjects.** `sprint` and `sd` share one subject set
> (S04, S06, S08, S12, S14, S15, S21, S22, S23) and `running` another
> (S07, S09, S11, S13, S16, S19); the two do not intersect, and
> `TEST_SUBJECTS` appear in `running` only.
>
> Running sprint would therefore mean scoring subjects the checkpoint was
> **trained on**, which makes an absolute "native MAE" meaningless as a
> baseline. So sprint appears only as a **clearly labelled secondary**, below,
> and never contributes to the primary or to any baseline.

**Clip inclusion.** Native `n >= 34`, so the 30 fps arm keeps at least 10 real
frames. That admits **589 of 592**; the three excluded are 28–33 frames and are
named in the results.

**Decimation — no filtering, actual frames only.**

- **Primary, 30.0 fps:** nearest-tick index selection,
  `idx_k = round(k * 100/30)` for `k = 0 .. floor((n-1) * 30/100)`. This lands
  on a true 30.0 fps grid, matching MoVi's PG1 exactly, and takes **real
  frames** — no interpolation and no smoothing of the input. 100/30 is not an
  integer, so a tick falls up to 5 ms from the nearest frame.
- **Sensitivity, 33.3 fps:** strict every-third-frame, `idx_k = 3k`. The pure
  integer decimation, reported beside the primary because the two bracket the
  answer and neither is uniquely correct at a non-integer ratio.

**Input arms.**

- **Generic COCO detections** — `det_markers2d_by_cam_coco`, the deployable
  detector, the one the 3.51°/3.62° baseline names.
- **GT 2D** — `gt_markers2d_by_cam`. **Its third channel ships a constant 1.0,
  not the training confidence scale.** Feeding that raw is the same class of
  defect the sealed test found at 14× (`scores_to_train_scale`, 30.07° MAE), so
  the channel is **overwritten to `TRAIN_CONF_MEAN = 0.8676`**, exactly as
  `phase15b_conventions.project` does for MoVi's diagnostic arm. The raw-1.0
  variant is reported beside it, because it is a one-line difference and a
  reader should see what it costs.

**Reference and conventions — all unchanged, none re-derived.**

- Reference 3D: `phase15c_thorax.gt3d_mm`, which applies the per-frame `p2mm`
  division. Never re-fitted or smoothed before scoring.
- Gravity vertical: `phase15d_run.world_up_athleticspose` on the raw world
  markers, carried into the released camera frame by the phase 15C per-frame
  rotation (`align_per_frame`, `scale=False`) — exactly what 15D's G1 did.
- Estimate vertical: the level-camera `EST_UP = (0, −1, 0)`.
- Near limb: GT 3D is already in camera coordinates, so the limb with the
  smaller mean axis-2 (depth) hip, as phase 14 computed it.
- Angles: `phase3_angles.sagittal_angles` and
  `phase15d_segments.segment_angles`, untouched.

**Quantities.** `knee_from_segments`, gravity **thigh** inclination,
**trunk-referenced hip**, **trunk lean**. **Near and far limb separately** —
never pooled. Per clip: MAE, bias, SD, r. Per subject: the mean over its clips.
Pooled: unweighted mean of per-subject values, with a **subject-bootstrap CI,
`N_BOOT = 2000`, `SEED = 11`**. At three subjects the interval is close to the
range of three numbers and carries little more than that — stated here so it is
not over-read later.

**Scoring grid.** **Both rate arms are scored at the same native frame
indices** — the decimated subset — so the two arms differ in temporal
information and in nothing else, not even in which frames are compared. The
native arm's full-rate figure is also reported, as a check that restricting to
the subset does not itself move the number.

**This phase produces AthleticsPose near/far figures for the thigh and trunk
for the first time.** Phase 15D's G1 pooled left and right into each subject's
list, so the committed 6.694° thigh figure cannot be split. The native arm's
near/far split therefore becomes the reference for the share calculation below,
and the committed side-pooled numbers are reported beside it as a cross-check.

## What is reported

**1. The 30 fps penalty**, per quantity, per limb, per input arm:

> `penalty = MAE(30 fps) − MAE(native)`

paired **per subject**, with a subject-bootstrap CI. A penalty whose CI includes
zero is reported as unresolved, not as zero.

**2. The share of the MoVi sealed loss it would account for if it transferred
unchanged.** The sealed figures are **fixed by this commit** and will not be
recomputed:

| quantity | sealed near | sealed far |
|---|---|---|
| knee flexion | **8.410°** | **8.685°** |
| thigh inclination (gravity) | **5.401°** | **7.372°** |
| hip (trunk-referenced) | **7.746°** | **9.257°** |
| trunk lean | **5.158°** | **5.099°** |

Then, per quantity and limb:

> `total_loss = sealed_MAE − AthleticsPose_native_MAE` (this phase's native arm)
> `share = penalty / total_loss`

**The assumption in that division is stated, not hidden:** it supposes the 30 fps
penalty measured on AthleticsPose running at 100 Hz would be the same on MoVi
jogging at 30 fps. It is a decomposition of a *difference between datasets* using
a term measured on only one of them. Where `total_loss` is near zero or negative
— which the thigh may well be, since MoVi's 5.401° near is already **below**
AthleticsPose's 6.694° pooled — **the share is undefined and will be reported as
such rather than as a large or negative percentage.**

**3. Secondary, clearly labelled.** The same penalty on `sprint`, on **sprint's
own subjects, which are TRAIN subjects for this checkpoint**. Reported as a
penalty only — never as an absolute baseline, never pooled with the primary. It
is included because a penalty is a difference between two arms on identical
clips, so training-set familiarity affects both arms and largely cancels; it
answers whether the penalty depends on the movement.

## No gate

**There is no pass/fail threshold on any number in this phase.** It is a
measurement of a confound, reported as it comes out. A finding that 30 fps costs
nothing is as useful as a finding that it costs everything, and both get the same
prominence.

## What would make this phase worthless

- **Changing the decimation rule after seeing a number.** Both rules are fixed
  above; both are reported whichever way they fall.
- **Feeding GT 2D's raw 1.0 confidence channel and calling it "perfect 2D"**
  without saying so. Both variants are reported.
- **Quoting the 30 fps arm against the *fine-tuned* detections.** This phase
  uses generic COCO and GT 2D only. Every figure names its input.
- **Letting the native arm drift from the committed baseline path.** It is
  bit-identical by construction, and that is verified in the results.
- **Treating the share-of-loss figure as a causal decomposition.** It is an
  if-it-transferred arithmetic, and is labelled as one everywhere it appears.

---

## Results

*Not yet run at the time of this commit.*
