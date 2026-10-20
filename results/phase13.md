# Phase 13 — Decompose and track

> **PRE-REGISTRATION, written 2026-10-19, before any result exists.**
> The bars below are frozen from this point. Results are appended under
> "Results"; nothing above that heading changes.

**Question.** Can the measurement be made more accurate without changing the
model — by filtering the input, or by holding a subject's segment lengths fixed?

**The re-cut this phase inherits.** `CLAUDE.md` originally had phase 13
constrain each frame to the scanned lengths and let the near limb's lengths
constrain the far limb. Phase 12 undermined that: the scan is *precise*
(across-trial CV 1.7%) and *biased* (ratio error 4.5–9.2%, 0 of 8 ratios at the
3% bar), so propagating a near-limb length to the far limb propagates the bias.
The scan survives as a **consistency constraint** — it holds a subject's
proportions fixed, with no claim that the lengths are true.

---

## What this phase is measured against

| | |
|---|---|
| Baseline | the **deployable** configuration: generic COCO detections, `det-coco` checkpoint, edge padding, overlapped windows |
| Baseline value | **3.509° near / 3.618° far** pooled; 3.45° [3.27, 3.69] and 3.56° [3.13, 3.90] subject-level (`results/phase7_baseline_ci.json`) |
| Data | AthleticsPose, role *tune*, the same 592 held-out clips |
| Subjects | **three** — S11, S13, S16. The other 21 trained the lifter |
| Estimator | **paired** per-subject difference, 2,000 subject bootstraps, seed 11 |
| Reported | near and far limb separately, never pooled |

**The resolution floor is about 0.2°.** The smallest paired difference whose
interval excludes zero anywhere in the phase 7 table is 0.24°, and on the near
limb nothing below 0.43° separates (`results/phase07.md` §3). An arm whose
expected gain is below that is **not worth building**, and this phase says so
rather than running it and reporting "no effect".

## The gate

```
GATE:  paired per-subject improvement vs the deployable baseline,
       95% subject-bootstrap CI excluding zero,
       NEAR and FAR limb each judged on its own.

Also required: the chosen setting is picked on the tune set and stated, and
AthletePose3D does not get worse than its phase 11 figure.
```

---

## Arm 1 — filter only

Filter the 2D detections before lifting, with a zero-phase Gaussian low-pass —
the same filter `scripts/video_kinematics.py:smooth` already applies at 10 Hz.
Cutoffs **6, 10 and 15 Hz** at the dataset's 100 Hz capture rate, all three
reported, the best stated as the choice.

**This also closes a gap nobody had noticed: the tool has been filtering all
along, and it was never validated.** The 3.509° baseline is measured on
*unfiltered* released detections, while the video tool low-passes at 10 Hz
before lifting. Whichever way this arm comes out, the tool and its own baseline
stop disagreeing afterwards.

**Pre-stated expectation — small, and possibly under the floor.** The 2D
detection error decomposes (562 clips, per-joint, over the eight joints the
sagittal angles use):

| component | RMS |
|---|---|
| total 2D error | 16.34 px |
| **constant bias** | **13.21 px** |
| low-frequency residual | 6.45 px |
| high-frequency residual, above 10 Hz | 4.49 px |

**81% of the error energy is a constant offset, which no filter touches**, and a
perfect low-pass removes only **1.39 px — 8.5% of the total**. The lifter is also
already temporal: it sees 81 frames at once, so some of that jitter is suppressed
before any filter is added. The honest prior is therefore a gain in the low
tenths of a degree, quite possibly below the floor. Running it anyway is cheap
(one pass per cutoff) and it settles the tool-versus-baseline discrepancy above.

**Noted for phase 16:** a zero-phase Gaussian is **non-causal**. It cannot run in
a stream without a lag of roughly 3σ, on top of the ~0.4 s the overlapped
lifting windows already cost. If this arm wins, the streaming version must be
re-measured with a causal filter, and that is a different number.

## Arm 2 — the scan as a consistency constraint

**Not being built yet, and the reason is geometric rather than empirical.**

Sagittal angles depend only on segment **directions**: `sagittal_angles` takes
`unit(knee - hip)` and `unit(ankle - knee)` and measures them in a frame built
from the pelvis and thorax. Rescaling a bone **along its own direction** to match
a scanned length therefore changes **no angle at all** — the angles are
identically invariant to it. A length constraint can only move an angle if it is
imposed as a *joint* fit, where enforcing every length at once pulls joint
positions sideways and so rotates neighbouring segments: an inverse-kinematics
solve, not a rescale.

Before that gets built, two measurements say what it could buy:

- **Phase 14 already tested the premise.** Per-frame deviation from the clip's own
  median segment length was one of eleven confidence inputs, and its standardised
  coefficient on log realized error was **+0.036**, against −0.614 for camera
  azimuth. A frame whose limb is the wrong length is **barely** a frame with a
  wrong angle. If length deviation does not predict error, enforcing length
  consistency has little error to remove.
- **Phase 12 measured the lengths as biased 4.5–9.2%.** An IK solve towards a
  biased target trades jitter for bias.

**So arm 2 is deferred, with its bar pre-registered for when it runs:** it needs
an IK formulation (fixed lengths, least-squares fit to the lifted joint
positions), and it is only worth the work if arm 1 shows the input-side path
clears the floor, or if a future measurement shows length deviation predicting
angle error better than +0.036.

## Arm 3 — both

Only if arm 1 and arm 2 both run. A gain that appears only in combination is
reported as such.

---

## Results

Run 2026-10-20, 592 held-out clips, `results/phase13_filter.json`. The
unfiltered row reproduces **3.509° near / 3.618° far** exactly — the third
independent reproduction of the deployable baseline, on a third code path.

### Arm 1 — not met, and in the opposite direction

| arm | σ (frames) | near | far | paired Δ near | 95% CI | |
|---|---|---|---|---|---|---|
| unfiltered | — | **3.509°** | **3.618°** | baseline | | |
| 6 Hz | 2.65 | 7.268° | 7.123° | **+4.613°** | [+1.580, +6.385] | excludes 0 |
| 10 Hz | 1.59 | 5.167° | 5.147° | **+2.069°** | [+0.590, +2.895] | excludes 0 |
| 15 Hz | 1.06 | 4.332° | 4.393° | **+1.036°** | [+0.265, +1.462] | excludes 0 |
| **the tool's own setting** | **0.48** | 3.602° | 3.708° | **+0.119°** | **[+0.019, +0.176]** | **excludes 0** |

**Filtering the 2D before lifting is harmful at every setting tested, and
monotonically in filter strength.** The gate is not met; nothing improved.

**The pre-registered expectation was wrong, and wrong in a specific way worth
recording.** It predicted a gain of up to 1.39 px of 2D error — 8.5% — from
removing the error's high-frequency component. Measured, filtering makes **the 2D
itself worse**:

| filter | mean 2D error over the 8 sagittal joints |
|---|---|
| unfiltered | **12.195 px** |
| 15 Hz | 12.322 px |
| 10 Hz | 13.075 px |
| 6 Hz | 15.591 px |

The error decomposition was correct and the inference from it was not. A low-pass
removes everything above the cutoff — error *and* signal — so it can only help
where the error's high-frequency energy exceeds the **true trajectory's**. A
running ankle at 100 Hz has real content up there: foot strike is near-impulsive.
The deciding quantity is the per-band signal-to-noise ratio, and the
pre-registration measured only the noise. (A Wiener filter states this
explicitly: optimal gain `S/(S+N)` per band. A fixed-cutoff Gaussian is that
filter with the SNR assumed instead of measured.)

The angle penalty is then larger than the 2D penalty, which is the third
confirmation of the pairing effect in `results/phase07.md` §5: the lifter was
trained on unsmoothed detections and is sensitive to any change in that input
distribution, in either direction.

### What this changes in the tool

`scripts/video_kinematics.py` **low-passed the 2D at 10 Hz on every clip it ever
processed**, justified in a comment as "the lifter was trained on temporally
coherent input". It was not — it was trained on the raw released detections. The
tool's effective setting on 30 fps footage is σ = 0.48 frames, which this phase
measures at **+0.119° near, CI [+0.019, +0.176], excluding zero**.

Small, and real. **Smoothing is now off by default** (`LOWPASS_HZ = 0.0`), kept
behind `--smooth-hz` so this measurement can be reproduced, with a test asserting
the default. That is the fourth thing this phase's harness has caught in the
deployable path rather than in the research path, after the joint order, the
capture rate and the detector's resolution default.

### A correction to the "0.2° resolution floor"

`results/phase07.md` §3 concluded from the phase 7 comparisons that nothing below
about 0.2° can be resolved at three subjects. **That is not a property of n = 3.**
The +0.119° result here has an interval excluding zero, because this comparison
changes **one knob** and holds the clips, the detections, the checkpoint and the
windows identical, so the per-subject differences are tightly paired and their
variance is small. The phase 7 comparisons swapped a detector *and* a checkpoint,
which moves far more.

The floor therefore belongs to the *comparison*, not to the dataset: a
single-knob paired change resolves to roughly 0.1°, while a whole-pipeline swap
needs 0.4° or more. Phase 13's remaining arms should be budgeted against the
former when they change one thing, and the latter when they change several.

### Arm 2 — still deferred, and now with one more reason

The geometric argument stands: rescaling a bone along its own direction cannot
change any angle, so arm 2 needs an IK solve rather than a projection. Arm 1 adds
a third piece of evidence against building it. The input-side path is not merely
unhelpful here — it is *actively* harmful in both directions, cleaner **and**
smoother, which says the lifter's error is not dominated by recoverable input
noise at all. An IK solve that pulls joints toward biased scanned lengths is a
larger change to the input distribution than a filter, aimed at the same place.

**Arm 2 and arm 3 are not run.** The honest statement is that the input-side half
of phase 13 is closed: no manipulation of the 2D or of the lifted skeleton that
leaves the checkpoint untouched has improved this measurement, and three separate
experiments now say the checkpoint is the binding constraint. What remains is a
**matched pairing** — a detector and a lifter trained together — which is phase
15's prerequisite and not available on a dataset with no video.
