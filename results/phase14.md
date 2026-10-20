# Phase 14 — Per-segment measurement confidence

> **PRE-REGISTRATION, written 2026-10-19, before any result exists.**
> The bars below are frozen from this point. `CLAUDE.md` requires them recorded
> with the date and not edited once the first number lands. Results are appended
> under "Results" below; nothing above that heading changes.

**Question.** The tool reports sagittal kinematics at a mean error of 3.51° on
the deployable configuration, and that mean hides an enormous spread: phase 3B
measured 4.66° → 2.11° across view-angle quartiles on AthleticsPose, phase 11
measured 3.50° → 1.77° on AthletePose3D, and phase 12 found that **which camera
you film from moves the measurement more than the runner does** (across-camera
CV 2.92–5.22% against across-trial CV 1.53–1.93%). A single pooled number
therefore tells a user almost nothing about the frame in front of them.

This phase asks whether the tool can say, per segment and per frame, **how much
to trust its own measurement** — and whether that statement is honest.

It produces no injury output of any kind. Confidence describes the
*measurement*; it is never a clinical quantity, and the word "risk" does not
appear in this phase's code, output, plots or docstrings.

**Why this runs before phase 13.** It depends on nothing phase 13 produces,
needs no new data, and is useful whether or not phase 13 ever lowers the MAE. A
measurement that states its own reliability is worth more than one that is
0.1° better and silent.

---

## What is being measured

### Segments

The same decomposition as phase 12, from the H36M-17 skeleton the lifter emits:

| segment | direction | a sagittal error exists? |
|---|---|---|
| thigh, L and R | hip → knee | **yes — gated** |
| shank, L and R | knee → ankle | **yes — gated** |
| pelvis | l_hip → r_hip | no — it *is* the frame's medio-lateral axis |
| trunk | pelvis → thorax | no — it *is* the frame's vertical axis |
| **foot** | heel → toe | **no — H36M-17 has no toe keypoint** |

**Second correction to this pre-registration, also 2026-10-19 and also before
any result.** Phase 12's segment list was taken over without checking it against
the quantity being measured. Only four of its segments have a sagittal direction
error at all.

The body frame is built *from* the pelvis and the trunk: `up` is
`unit(thorax - pelvis)` re-orthogonalised and `ml` is `unit(r_hip - l_hip)`
(`phase3_angles.sagittal_angles`). Expressed in that frame, the trunk direction
is the vertical by construction and the pelvis has no sagittal component at all,
so their "error" is identically zero in every frame of both skeletons. It is not
small, it is definitionally absent — the frame cannot measure its own axes.

Measuring them for real would need a shared world frame, and there is none: the
lifter's output is scale-free and root-relative with no gravity, which is exactly
why the frame is self-referential in the first place. **So pelvis and trunk are
reported as not measurable against this ground truth, with that reason, in the
same way foot is reported as unavailable** — the precedent `sagittal_angles` set
by returning NaN ankle angles rather than faking them, and phase 12 followed.

The gated set is therefore **four segments: thigh and shank, near and far**. That
is the same information the hip and knee angles carry, which is the honest extent
of what this skeleton supports, and the demo overlay may claim confidence for
those four and nothing else.

**What this buys, stated as a check rather than a hope.** Because
`hip_flex = wrap(seg_angle(thigh) - 180)` and
`knee_flex = wrap(seg_angle(shank) - seg_angle(thigh))`, the thigh's segment
error must equal the hip angle error *exactly*, and the knee angle error must be
the difference of the shank and thigh segment errors. Both identities are
asserted in tests before any confidence is fitted. They are what make the
per-segment decomposition a decomposition rather than a relabelling: a knee
angle that looks fine because a thigh error and a shank error cancelled is a
case this phase can see and the phase 7 report could not.

### Realized error, per segment, per frame

Joint angles are *relative* between two segments, so a per-joint error cannot be
attributed to a segment. The per-segment quantity is the segment's own
direction, in the body's own frame — the `fwd`/`up` basis
`phase3_angles.sagittal_angles` already builds from pelvis→thorax and hip↔hip, so
a rotated camera does not change it:

```
realized_error(segment, frame)
  = | angle( proj(d_est), proj(d_gt) ) |        in degrees
    where d = unit(distal - proximal)
          proj = projection into the sagittal plane (normal = medio-lateral)
```

This is the quantity the confidence must rank, for the four gated segments. Two
adjacent segment errors bound the joint angle error between them.

### Confidence inputs — and the rule that constrains them

**Every input must be computable from the video alone.** No ground truth, no
`norm_scale`, no `p2mm`, nothing a phone clip cannot produce. Phase 12's
millimetre path fails this test and is therefore not available here; its
*ratio* path is.

The declared superset, fixed now — nothing may be added after a number exists:

| input | from | why |
|---|---|---|
| keypoint score, proximal and distal | the detector's own confidence channel (`det2d[:, :, 2]`, used by `lift`) | the detector knows when it is guessing |
| per-frame bone length / this clip's median bone length | the estimate itself, scale-free | phase 12: frame-to-frame length varies 14.52% against 1.68% across trials, so a frame whose limb is the wrong length is a frame the lifter lost |
| left/right length asymmetry for the frame | same | a limb that disagrees with its partner is usually the occluded one |
| `ml_compression` — the medio-lateral axis as the camera sees it, defined below | 2D detections + the phase 12 scan ratio | the measured dominant term |
| near or far limb | pelvis depth ordering, as `phase7_inference_fix.py` already does | the occluded limb is a different measurement problem |
| frame position within its lifting window, and the triangular blend weight it received | `lift` | edge-padded and single-covered frames are out of distribution |
| per-frame keypoint displacement | the estimate itself | fast frames are blurred frames |

Feature *selection* happens inside the fold, never across it.

**Correction to this pre-registration, made 2026-10-19 before any result
exists.** The first version of the table above listed `view_ratio` as the
viewpoint input, pointing at `phase11_viewpoint.py:74` and
`phase3b_viewpoint.py:67`. **Both of those read ground-truth 3D** — they take
`markers_h36m` and use its depth axis, `|z_left - z_right| / hip_width`. That
violates this phase's own rule in the first row it applies to: a phone clip has
no depth. The viewpoint input is therefore defined from the video alone:

```
apparent_ml(frame) = || hip_l_2d - hip_r_2d ||  /  || pelvis_2d - thorax_2d ||
scanned_ml         = pelvis_width / trunk_length          (phase 12 scan ratio,
                                                           scale-free)
ml_compression     = apparent_ml(frame) / scanned_ml           in [0, ~1]
view_proxy_deg     = degrees( arccos( clip(ml_compression, 0, 1) ) )
                                                     90 deg = fully side-on
```

Camera azimuth compresses the medio-lateral axis and leaves the sagittal plane
roughly intact, so dividing the apparent hip separation by a sagittal length
cancels both the unknown image scale and the subject's own size, and the phase 12
scan supplies the one ratio needed to turn what is left into an angle. This is
what `CLAUDE.md` asks for in its own words: "an estimated camera azimuth (from
apparent hip/shoulder width against the scan)".

The two ground-truth implementations keep a role, but as a **validator, not an
input**: on the tune set, where depth exists, the GT `view_ratio` is
approximately `|sin(azimuth)|` while `ml_compression` is approximately
`cos(azimuth)`, so the two must satisfy `ml_compression ~ sqrt(1 - view_ratio^2)`.
The Spearman correlation between `view_proxy_deg` and the GT
`out_of_plane_deg` is **reported, and is not a gate** — it says how much is lost
by not having depth, which is exactly the penalty a real clip pays. The
consolidation is then of the GT function alone: one implementation, imported,
not a third copy.

### Stated expected error

Two levels, deliberately separated:

1. **Per clip**, from the **already-published viewpoint curve for that dataset** —
   phase 3B's 4.66° → 2.11° for AthleticsPose, phase 11's 3.50° → 1.77° for
   AthletePose3D — evaluated at the clip's median `view_proxy_deg`. **The curve
   is not refit**; only the point it is read at is estimated from video. `CLAUDE.md` requires expected error to come from the measured
   viewpoint curve rather than a new guess.
2. **Per segment per frame**, a ridge regression on log realized error over the
   inputs above, fitted on training-fold frames only, exponentiated back to
   degrees. Confidence is a monotone function of that stated error, so the two
   bars below test the same model in two different ways: ranking, then
   magnitude.

---

## The gate

Both bars are the ones `CLAUDE.md` pre-registered for this phase. Their
estimators are pinned here, because a rank correlation can be computed four ways
and they do not agree.

### Bar 1 — confidence ranks frames by realized error

```
rho(segment, subject) = Spearman( stated_error , realized_error )
                        over all out-of-fold frames of that subject

GATE:  rho  >=  0.30   for EVERY held-out subject,  per segment
reported:  the mean, the minimum, and a subject-bootstrap interval marked
           nominal
```

**Third correction, 2026-10-19, before any result.** The gate above originally
read "mean over subjects >= 0.30 and the 95% subject-bootstrap CI lower bound
> 0". That interval cannot exist here. **AthleticsPose holds out three
subjects** — S11, S13, S16 (`phase3_infer.py:53`) — and the other 21 trained the
lifter, so its error on them is training error and is unusable as a label. A
bootstrap over three numbers resamples three numbers; its interval is an
artefact of the resampling, not evidence.

So the bar is stated per subject instead: **every** held-out subject must clear
0.30. At n=3 that is both stronger than a mean and honest about what three
subjects can support. The mean, the minimum and the nominal interval are all
reported, and the figure's error bar is the observed per-subject range rather
than a bootstrap.

The same follows for the folds: **leave-one-subject-out, 3 folds**, not 5.

Spearman is computed **within subject, then averaged**. Pooling frames across
subjects would let between-subject differences in overall error masquerade as
within-clip ranking skill, which is the same mistake a pooled clip mean makes.
Reported per segment, and **near and far limb separately, never averaged
together** — a confidence that works on the near limb and fails on the far one
is a failed confidence, and it must be visible.

### Bar 2 — stated expected error matches realized error

```
OLS:  realized_error  =  a + b * stated_error      out-of-fold frames

GATE:  b in [0.80, 1.20]       per segment, pooled over subjects
reported:  b per subject, a, R^2, and the mean absolute calibration gap in
           degrees
```

Same structure as phase 4B's calibration check, applied to angles instead of
labels. A slope below 0.8 means the confidence exaggerates how bad bad frames
are; above 1.2 means it understates it, which is the dangerous direction.

### Kill criterion

```
if  no segment clears rho >= 0.30
then the confidence carries no information: record it and ship no
     confidence number, per-segment or otherwise
```

A confidence that does not rank error is worse than no confidence, because it
invites trust it has not earned. Phase 16's overlay then shows the measurement
and the clip-level expected error only.

---

## Protocol

- **Dataset: AthleticsPose only** — role *tune* (`docs/data-inventory.md` §8.1).
  The same 592 held-out clips as the phase 7 baseline, so the realized errors
  are the baseline's own errors.
- **Detections: generic COCO** (`det_markers2d_by_cam_coco`), with the
  `det-coco` checkpoint, edge padding and overlapped windows — the **deployable**
  configuration, 3.51°. Fine-tuned detections are not used: no real video can
  produce them, so a confidence fitted on them would be fitted on an artifact.
- **Split: `GroupKFold` on `subject`, 3 folds — leave-one-subject-out**, since
  only three subjects are held out. A subject's frames never appear on both
  sides. 8 cameras share one trial, so clip-level splitting would leak; this is
  the same rule as `groups=sub_id`.
- **All fitting inside the fold**: scaler, feature selection, ridge, calibration.
  Every reported number is out-of-fold.
- **Intervals: subject bootstrap**, `subject_bootstrap_ci` from
  `scripts/phase12_scan.py` (2,000 resamples, seed 11), resampling subjects.
- **Reported alongside, not gated:** a frozen-model run on AthletePose3D (also
  *tune* since phase 12) to see whether the calibration transfers — phase 11
  measured the lifter itself degrading 1.84× on that lab, so it may not. Nothing
  is tuned after that number exists.
- **Held-out datasets are not touched.** MoVi, the dual-system set and Bath stay
  closed until phase 15.

## What would make this phase wrong

Written down now, so it is not rationalised later:

- **The confidence could be reading the detector's score and nothing else.** A
  single-feature control — detector score alone — is reported beside the full
  model. If the full model does not beat it, say so plainly and ship the simple
  one.
- **It could be ranking clips, not frames.** A per-clip permutation control is
  reported: shuffle stated error within each clip and re-score. It must collapse
  to ρ ≈ 0.
- **Three subjects is three subjects.** Whatever this phase finds, it is found
  on S11, S13 and S16. A confidence model that works on three people may not
  work on thirty, and nothing in this phase can tell the difference. The
  AthletePose3D transfer check is the only outside evidence available, and it
  has three running subjects of its own.
- **Realized error is measured against the lifter's own 3D, not the lab's.** The
  ground truth is AthleticsPose's released `markers_h36m`, used exactly as
  shipped, including its `p2mm` convention. Angles are scale-invariant, so the
  unit trap of `results/phase03b.md` does not reach them — verified again before
  the first number.

---

## Results

Run 2026-10-19. 592 held-out AthleticsPose clips, 214,500 segment-frames, 3
subjects (S11, S13, S16), leave-one-subject-out, deployable configuration
(generic COCO detections + edge padding + overlapped windows).
`results/phase14_confidence.json`, `figures/phase14_confidence.png`.

### The verdict, per segment

| segment | realized | stated | rho (min over subjects) | bar 1 (every subject >= 0.30) | slope | bar 2 ([0.8, 1.2]) |
|---|---|---|---|---|---|---|
| shank_l | 3.80° | 3.06° | 0.37 (**0.34**) | **met** | 1.47 | not met |
| shank_r | 3.05° | 3.04° | 0.34 (**0.34**) | **met** | 0.83 | **met** |
| thigh_l | 3.09° | 2.87° | 0.20 (0.09) | not met | 0.65 | not met |
| thigh_r | 3.07° | 2.87° | 0.26 (0.17) | not met | 0.57 | not met |

**The kill criterion did not fire** — two segments rank error above the bar — but
only **one segment of four clears both bars**, and it clears bar 2 at 0.83
against a floor of 0.80. The thighs fail bar 1 on individual subjects while
averaging close to it (thigh_l: 0.093, 0.186, 0.328), which is exactly why the
bar was stated per subject rather than as a mean.

The pre-registered control passes: the full model beats detector score alone on
every segment (0.37 vs 0.21, 0.34 vs 0.15, 0.20 vs −0.08, 0.26 vs −0.08). **The
detector's own confidence channel is worthless on the thighs** — negative, and
its calibration slope is negative too, meaning it is anti-correlated with error.

### What the confidence actually is: a viewpoint meter

Standardised ridge coefficients on log realized error (descriptive, whole-sample):

| input | coefficient |
|---|---|
| `view_proxy_deg` | **−0.614** |
| `ml_2d_ratio` | **−0.368** |
| `score_min` | −0.067 |
| `score_dist` | −0.049 |
| `speed` | −0.044 |
| `len_dev` | +0.036 |
| everything else | ≤ 0.016 |

The first two are the same quantity — one is `arccos` of the other, rescaled — so
**this is effectively a single-feature model, and the feature is camera
azimuth.** Nothing else earns its place: keypoint score, segment-length
deviation, left/right asymmetry, window blend weight and speed are all within
noise of zero. The sign reproduces phases 3B and 11 from video alone: the more
side-on the view, the lower the error.

**And the per-clip ranking is most of the skill.** Shuffling the stated error
*within each clip* — destroying all frame-level ordering while leaving each
clip's level intact — still scores:

| segment | confidence | shuffled within clip | what the frames add |
|---|---|---|---|
| shank_l | 0.37 | 0.28 | +0.09 |
| shank_r | 0.34 | 0.29 | +0.05 |
| thigh_l | 0.20 | 0.19 | +0.01 |
| thigh_r | 0.26 | 0.22 | +0.04 |

So roughly **four fifths of the ranking is between clips, not between frames**.
That control was pre-registered precisely to catch this, and it caught it. The
honest description of what phase 14 produced is a **clip-level reliability
estimate with a weak per-frame component**, not the per-segment per-frame
confidence the phase set out to build. `R²` of realized on stated is 0.036–0.141,
and the per-subject calibration slopes scatter widely (shank_l: 1.03, 1.32,
1.77), so the stated error's *size* does not transfer across subjects even where
its *ranking* does.

### Two findings that stand on their own

**1. Camera azimuth is recoverable from video alone.** The proxy — apparent 2D
hip separation over apparent trunk length, divided by a population
anthropometric ratio — reaches **Spearman 0.864, Pearson 0.974** against the
ground-truth depth metric over 592 clips. `CLAUDE.md` asked phase 14 for "an
estimated camera azimuth (from apparent hip/shoulder width against the scan)"
without knowing whether one was obtainable. It is, and to within 3% of variance
explained. This is reusable: it is the input phase 16's expected-error readout
needs, and it needs no ground truth, no depth and no scan in millimetres.

**2. The near limb can be identified without ground truth.** Taking the nearer
limb from the *estimate's* own depth ordering agrees with the ground-truth
ordering on **592 of 592 clips**. Near and far must be reported separately, and
on real footage there is no ground truth to label them with; this says the tool
can label them itself, with no observed failure.

### What was wrong with the first run, and why it is not in this table

The first run of the frozen protocol is kept at
`results/phase14_confidence_frozen_protocol.json` and
`figures/phase14_confidence_frozen_protocol.png`. **Its numbers are void**, for
two code defects rather than two findings:

1. **The azimuth proxy was self-cancelling.** It divided the apparent 2D hip
   width by the *estimate's own 3D* hip width. The estimate's x and y come from
   those same detections, so the camera's compression appeared above and below
   the line and divided out.
2. **The 2D was read in the wrong joint order.** `det_markers2d_by_cam_coco` is
   named after the detector that produced it, not after its joint order, and it
   ships in **H36M-17** — so COCO indices read a shoulder where a hip belonged.
   The same mistake `scripts/video_kinematics.py` was making, found the same day
   (see `results/phase07.md` §4).

Neither was catchable by the unit tests, which all pass on random arrays: a
wrong index leaves every feature varying plausibly. What caught both was the
**ungated validator** — the proxy scored Spearman −0.02 against ground truth
where it should have scored −0.86. It existed only because `CLAUDE.md` requires
expected error to come from the measured viewpoint curve, which forced the proxy
to be compared against something real. A test now asserts that correlation on
real clips (`tests/test_confidence.py::test_the_azimuth_proxy_tracks_real_camera_obliquity`),
so the seam cannot break silently again.

Fitting also changed in one way that is a method fix, not a defect: `exp()` of a
least-squares fit in log space estimates the **geometric** mean, which sits below
the arithmetic mean the calibration bar is stated against. **Duan's smearing
estimator**, fitted on the training fold like everything else, corrects it — it
is what moved stated error from 2.0° against 3.8° realized to 3.06° against
3.80°.

### What this means downstream

- **Phase 16 may show a clip-level expected error**, built from the azimuth
  proxy and the already-measured viewpoint curve. That is well supported.
- **It may not claim per-segment confidence on the thighs.** Two segments of
  four fail to rank their own error on individual subjects.
- **It may show per-segment confidence on the shanks**, with the caveat that the
  stated magnitude is only calibrated for one of the two.
- **Three subjects is three subjects.** Every number here rests on S11, S13 and
  S16. The AthletePose3D transfer check is the only outside evidence available
  and has three running subjects of its own.
- **The thighs' failure has a plausible cause worth testing, not assuming:** the
  hip is the joint where the detector and the mocap *disagree by definition* —
  24–28 px against 3–5 px at the knees and ankles (`results/phase07.md` §4) — so
  thigh direction error may be dominated by a definitional offset that no
  video-only input can see. That is a phase 13 question, not a phase 14 one.
