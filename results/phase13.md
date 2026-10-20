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

*Not yet run. Nothing above this line changes once it is.*
