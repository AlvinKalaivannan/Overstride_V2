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

| segment | direction | available? |
|---|---|---|
| thigh, L and R | hip → knee | yes |
| shank, L and R | knee → ankle | yes |
| pelvis | l_hip → r_hip | yes |
| trunk | pelvis → thorax | yes |
| **foot** | heel → toe | **no — H36M-17 has no toe keypoint** |

Foot is reported as unavailable, not approximated — the precedent set by
`sagittal_angles` returning NaN ankle angles rather than faking them, and
followed by phase 12.

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

This is the quantity the confidence must rank. It is defined for every segment
above, including the ones no published joint angle covers (pelvis, trunk), and
two adjacent segment errors bound the joint angle error between them.

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

GATE:  mean over subjects of rho  >=  0.30        per segment
  and  the 95% subject-bootstrap CI lower bound  >  0
```

Spearman is computed **within subject, then averaged**. Pooling frames across
subjects would let between-subject differences in overall error masquerade as
within-clip ranking skill, which is the same mistake a pooled clip mean makes.
Reported per segment, and **near and far limb separately, never averaged
together** — a confidence that works on the near limb and fails on the far one
is a failed confidence, and it must be visible.

### Bar 2 — stated expected error matches realized error

```
OLS:  realized_error  =  a + b * stated_error      out-of-fold frames

GATE:  b in [0.80, 1.20]       per segment
reported:  a, R^2, and the mean absolute calibration gap in degrees
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
- **Split: `StratifiedGroupKFold`-equivalent grouping on `subject`**, 5 folds. A
  subject's frames never appear on both sides. 8 cameras share one trial, so
  clip-level splitting would leak; this is the same rule as `groups=sub_id`.
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
- **Realized error is measured against the lifter's own 3D, not the lab's.** The
  ground truth is AthleticsPose's released `markers_h36m`, used exactly as
  shipped, including its `p2mm` convention. Angles are scale-invariant, so the
  unit trap of `results/phase03b.md` does not reach them — verified again before
  the first number.

---

## Results

*Not yet run. Nothing above this line changes once it is.*
