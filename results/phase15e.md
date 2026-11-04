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

**Run 2026-11-04**, after the pre-registration above was committed alone as
`45dc30a`. `results/phase15e.json`, driver `scripts/phase15e_fps.py`.
**589 clips** scored (3 of 592 excluded by the pre-registered `n >= 34` rule),
AthleticsPose running, S11/S13/S16. No gate; no MoVi file was opened.

### First: the native arm reproduces the committed baselines

The control is only worth reading if its native arm really is the baseline path.
It is, to within rounding:

| quantity | limb | 15E native | 15E native, all frames | committed | source |
|---|---|---|---|---|---|
| knee flexion | near | **4.032** | **4.073** | **4.071** | `results/phase15.md` |
| hip (trunk-ref) | near | **2.838** | **2.833** | **2.834** | `results/phase15.md` |
| thigh inclination | near | 6.532 | 6.548 | — | — |
| thigh inclination | far | 6.816 | 6.840 | — | — |
| thigh, near/far mean | — | **6.674** | 6.694 | **6.694** | `results/phase15d.md` (side-pooled) |

The hip matches to **0.004 deg**, the knee's all-frames figure to **0.002 deg**,
and the thigh's near/far mean to **0.020 deg**. As pre-registered, this follows
from `MODEL_FPS == 100 Hz` making `lift_2d(..., fps=100)` a no-op wrapper around
the committed `lift()` call.

**Restricting the score to the decimated frames moves nothing**: across all four
quantities and both limbs the subset-vs-all-frames difference is **-0.042 to
+0.005 deg**. So the common scoring grid is not a confound.

### The 30 fps penalty, paired per subject

`penalty = MAE(30 fps) - MAE(native)`, same clips, same scored frames, same
padding. **Generic COCO detections — the deployable arm:**

| quantity | limb | native | 30 fps | **penalty** | 95% CI | resolved? |
|---|---|---|---|---|---|---|
| **knee flexion** | near | 4.032 | 5.866 | **+1.835** | [+0.637, +2.540] | **yes** |
| **knee flexion** | far | 4.076 | 5.833 | **+1.757** | [+0.429, +2.548] | **yes** |
| **hip (trunk-ref)** | near | 2.838 | 3.479 | **+0.641** | [+0.201, +0.862] | **yes** |
| **hip (trunk-ref)** | far | 3.020 | 3.754 | **+0.734** | [+0.164, +1.109] | **yes** |
| thigh inclination | near | 6.532 | 6.913 | **+0.381** | [+0.104, +0.527] | **yes** |
| thigh inclination | far | 6.816 | 7.281 | **+0.465** | [+0.092, +0.726] | **yes** |
| **trunk lean** | near | 5.533 | 5.331 | **-0.202** | [-0.371, -0.002] | **yes, it IMPROVES** |
| **trunk lean** | far | 5.533 | 5.331 | **-0.202** | [-0.371, -0.002] | **yes, it IMPROVES** |

**Every penalty is resolved** — all eight intervals exclude zero. The knee pays
**+1.8 deg**, by far the largest; the hip **+0.7 deg**; the thigh **+0.4 deg**;
and **the trunk gets slightly better.**

### Dose-response: the 33.3 fps sensitivity arm

Strict every-third-frame keeps 11% more frames than the 30.0 fps grid, and
**every penalty is correspondingly smaller, in the same order**:

| quantity | limb | 33.3 fps | 30.0 fps |
|---|---|---|---|
| knee flexion | near | +1.509 | +1.835 |
| knee flexion | far | +1.597 | +1.757 |
| hip (trunk-ref) | near | +0.539 | +0.641 |
| hip (trunk-ref) | far | +0.594 | +0.734 |
| thigh inclination | near | +0.310 | +0.381 |
| thigh inclination | far | +0.462 | +0.465 |
| trunk lean | both | -0.169 | -0.202 |

All eight remain resolved, and **nothing changes sign**. A monotone
dose-response across two independently chosen decimation rules is stronger
evidence that this is frame rate than either rule alone.

### Share of the sealed loss, if the penalty transferred unchanged

`loss = sealed MAE - AthleticsPose native MAE`; `share = penalty / loss`.
Sealed figures quoted from the pre-registration, not recomputed. **COCO arm.**

| quantity | limb | AP native | sealed | loss | penalty | **share** |
|---|---|---|---|---|---|---|
| **knee flexion** | near | 4.032 | 8.410 | 4.378 | +1.835 | **41.9%** |
| **knee flexion** | far | 4.076 | 8.685 | 4.609 | +1.757 | **38.1%** |
| **hip (trunk-ref)** | near | 2.838 | 7.746 | 4.908 | +0.641 | **13.1%** |
| **hip (trunk-ref)** | far | 3.020 | 9.257 | 6.237 | +0.734 | **11.8%** |
| thigh inclination | far | 6.816 | 7.372 | 0.556 | +0.465 | **83.7%** |
| thigh inclination | near | 6.532 | 5.401 | **-1.131** | +0.381 | **undefined** |
| trunk lean | near | 5.533 | 5.158 | **-0.375** | -0.202 | **undefined** |
| trunk lean | far | 5.533 | 5.099 | **-0.434** | -0.202 | **undefined** |

**The headline: roughly 40% of the knee's cross-lab loss is frame rate alone,
and only about 12% of the hip's.** Three of eight cells are **undefined because
the loss is negative** — MoVi is *better* than AthleticsPose on the near thigh
and on trunk lean, so there is no loss for the penalty to be a share of. That
was pre-registered as the likely outcome for the thigh and is reported as
undefined rather than as a misleading percentage.

This is **if-it-transferred arithmetic, not a causal decomposition**: the
penalty is measured on AthleticsPose running at 100 Hz and divided into a
difference between two datasets that also differ in population, camera, marker
convention and movement.

### Secondary: sprint, on its own TRAIN subjects — and the knee penalty replicates

**9 subjects, 390 clips, COCO only, penalty only.** These are subjects the
checkpoint was **trained on**, so no absolute MAE from this arm is a baseline
and none is quoted as one. A penalty is a difference between two arms on
identical clips, so training familiarity affects both and largely cancels.

| quantity | limb | sprint penalty (n=9) | 95% CI | running penalty (n=3) |
|---|---|---|---|---|
| **knee flexion** | near | **+1.905** | [+1.501, +2.344] | +1.835 |
| **knee flexion** | far | **+1.829** | [+1.404, +2.207] | +1.757 |
| **hip (trunk-ref)** | near | **+0.492** | [+0.274, +0.749] | +0.641 |
| **hip (trunk-ref)** | far | **+0.635** | [+0.421, +0.867] | +0.734 |
| thigh inclination | near | +0.033 | [-0.096, +0.171] | +0.381 |
| thigh inclination | far | +0.228 | [+0.086, +0.363] | +0.465 |
| trunk lean | both | +0.042 | [-0.100, +0.162] | -0.202 |

**The knee and hip penalties replicate on a different movement and three times
the subjects, with much tighter intervals** — knee +1.905 against +1.835, hip
+0.492 against +0.641. At n=9 the knee interval is [+1.501, +2.344], so the
+1.8 deg knee penalty is the most firmly established number in this phase.

**The thigh and trunk penalties do NOT replicate.** The thigh's near-limb
penalty falls to +0.033 and becomes unresolved, and **the trunk's improvement
disappears and flips sign (+0.042, unresolved)**. So the trunk result in the
primary table should be read as running-specific or as an n=3 artifact, **not as
"decimation helps the trunk"** — and that caveat is the honest reading even
though the primary interval excluded zero.

### Input arms: perfect 2D is worse again, and the confidence channel costs real degrees

Native-rate MAE by input, near limb:

| input | knee | hip (trunk-ref) | thigh | trunk |
|---|---|---|---|---|
| **generic COCO detections** | **4.032** | **2.838** | 6.532 | 5.533 |
| GT 2D, confidence at `TRAIN_CONF_MEAN` | 4.496 | 5.350 | **5.461** | **5.129** |
| GT 2D, raw 1.0 confidence | 5.091 | 5.578 | 7.092 | 6.412 |

Two things fall out:

- **Ground-truth 2D is 2.5 deg WORSE than detections on the hip** (5.350 against
  2.838) and 0.46 deg worse on the knee, on 589 tune-set clips. The hip bias
  marches monotonically the wrong way as the input gets cleaner: **-1.243 ->
  -2.335 -> -4.252**. This is phase 7 section 5's `CKPT_FOR` lesson and the
  sealed test's trunk result, reproduced a third time and now on the tune set at
  scale: **the lifter has learned its detector's error distribution, and perfect
  2D is not an upper bound.** GT 2D does help the thigh and trunk.
- **The confidence channel is not cosmetic.** Overwriting GT 2D's shipped
  constant 1.0 with `TRAIN_CONF_MEAN = 0.8676` is worth **0.60 deg on the knee,
  0.23 deg on the hip and 1.63 deg on the thigh**. The pre-registration called
  this the same class of defect as the sealed test's 14x bug; it is the same
  defect, two orders of magnitude smaller.

### The sealed loss is a BIAS gain, not a shape loss

Worth recording beside the correction appended to `results/phase15_sealed.md`.
On AthleticsPose at native rate, COCO, near limb, the error is **not**
offset-dominated: bias **+1.388** (knee) and **-1.243** (hip), SD 4.28 and 2.74,
with **r = +0.990 to +0.991**. On sealed MoVi the same quantities carry bias
**+5.446** and **-6.054**, SD ~8.7 and ~7.3, and r **+0.77 to +0.92**.

So crossing to MoVi roughly **quadruples the bias and doubles the SD**. And
within AthleticsPose, **the 30 fps penalty also acts mainly through bias** —
knee bias +1.388 -> +2.232 while r falls only 0.991 -> 0.979. Decimation shifts
the angle more than it blurs the trace.

One oddity: **trunk lean has r = +0.764 at native rate**, far below every other
quantity's 0.99, and it falls further with GT 2D (0.671, then 0.636 at raw 1.0).
The trunk is the worst-tracked quantity on AthleticsPose at any frame rate and
with any input, which is consistent with phase 15C finding the lifter's thorax
is not a stable anatomical point.

### Deviations, recorded

- **D-E1 — sprint is not in the primary.** Pre-registered above, with the
  reason: sprint and running have disjoint subject sets and `TEST_SUBJECTS`
  appear in running only. 390 of 414 sprint clips passed `n >= 34`.
- **D-E2 — the gravity quantities run on 231 clips, not 589, and the
  pre-registration did not say so.** Thigh and trunk need the world markers
  aligned to GT to recover the vertical, and only length-matched raw/gt pairs
  qualify (phase 15C's plumbing constraint, 231 of 592). Knee and
  trunk-referenced hip need no vertical and use all **589**. This is an omission
  in the pre-registration, not a choice made after seeing numbers: the
  constraint is a property of the files and was recorded in phase 15C before
  this phase existed. Every table above can be read with the per-quantity n from
  `results/phase15e.json` (`pooled.*.n_clips`).
- **D-E3 — the secondary used COCO only.** The pre-registration said "the same
  penalty on sprint" without naming input arms; COCO is the deployable one and
  the only one the primary headline uses.
- **D-E4 — a crash in the first smoke run, fixed before the full run.**
  `args.out.relative_to(REPO)` raised `ValueError` for an output path outside
  the repository — the identical defect fixed in `phase15_checks.py` in
  `2b5b74d`, reintroduced in a brand-new file hours later. It affected only the
  success message, after the JSON was written. No number is affected.

### What surprised me

1. **The knee's 30 fps penalty is large and the hip's is small** — +1.8 deg
   against +0.7 deg, and it holds at n=9 on a different movement. Before this
   phase I would have guessed frame rate hurt both about equally.
2. **About 40% of the knee's cross-lab loss is frame rate**, measured with no
   change of lab. That makes "it is a different lab" a materially worse
   explanation of the knee gap than it looked.
3. **Three of eight share cells are undefined because MoVi is better**, on the
   near thigh and on trunk lean. The sealed test already showed the thigh
   transferring without loss; this phase shows there is literally no loss to
   decompose there.
4. **The trunk's apparent improvement at 30 fps did not replicate** on sprint at
   three times the subjects. A resolved CI at n=3 survived a dose-response check
   and still failed replication, which is a useful reminder about what a
   three-subject interval is worth.
5. **Ground-truth 2D is worse than real detections on the hip by 2.5 deg**, now
   on 589 tune-set clips. The third independent reproduction of that effect in
   this project.
