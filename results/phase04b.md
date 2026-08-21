# Phase 4B — what the score means for one runner

**Phase 6 should not be built as a per-user verdict. At the deployment operating
point the model disagrees with itself on a third of repeat scans of the same
runner, and its probabilities carry almost no information beyond the base rate.**

Everything in this project until now is AUC. A demo cannot be built on AUC: it
has to put a number or a decision in front of a person. That requires knowing
what happens at a threshold, whether the probability means anything, and whether
the same runner gets the same answer twice. None of the three had been measured.

---

## The deployment configuration

`wave2` (hip + knee), **side-on error bank**, **30 fps** — the most realistic
configuration this project can construct. Pooled out-of-fold **AUC 0.569**.
Clean mocap is carried alongside as an upper bound (**AUC 0.614**).

Scores are genuinely out-of-fold: each session is held out once per seed, so
every session has 5 scores from models that never saw that subject. They are
averaged. `check_no_group_leakage` asserted on all 25 folds.

## 1. Operating point

Chance is 0.517 (the base rate of right-side injuries).

**Deployment (side-on, 30 fps):**

| operating point | threshold | sens | spec | PPV | NPV | balanced acc |
|---|---|---|---|---|---|---|
| Youden-optimal | 0.500 | 0.657 | 0.478 | **0.574** | 0.566 | 0.568 |
| specificity ≥ 0.90 | 0.615 | **0.130** | 0.899 | 0.579 | 0.491 | 0.514 |
| sensitivity ≥ 0.90 | 0.418 | 0.901 | **0.165** | 0.536 | 0.607 | 0.533 |

**Clean mocap (upper bound):**

| operating point | threshold | sens | spec | PPV | NPV | balanced acc |
|---|---|---|---|---|---|---|
| Youden-optimal | 0.485 | 0.704 | 0.491 | 0.597 | 0.608 | 0.598 |
| specificity ≥ 0.90 | 0.644 | 0.151 | 0.901 | 0.621 | 0.498 | 0.526 |
| sensitivity ≥ 0.90 | 0.413 | 0.898 | 0.238 | 0.558 | 0.686 | 0.568 |

**At the best available operating point, PPV is 0.574 against a 0.517 base rate —
a 5.7-point lift.** Push specificity to a level that would make a positive call
meaningful and sensitivity collapses to 0.13. There is no threshold at which this
model is both useful and trustworthy.

## 2. Calibration

| | deployment | clean mocap |
|---|---|---|
| Brier score | 0.2460 | 0.2392 |
| Brier, constant base-rate model | 0.2497 | 0.2497 |
| **improvement over base rate** | **1.5%** | 4.2% |
| calibration slope (1.0 = perfect) | **0.716** | 0.827 |

Reliability by quintile of the model's output probability (deployment):

| n | model p | observed |
|---|---|---|
| 164 | 0.390 | 0.409 |
| 163 | 0.473 | 0.466 |
| 164 | 0.521 | 0.598 |
| 163 | 0.565 | 0.534 |
| 164 | 0.633 | 0.579 |

The ordering is not even monotone across the top three bins. **A calibration
slope of 0.716 means the probabilities are more extreme than the evidence
supports** — showing a user "72% likely your right side" would overstate what the
model knows. A 1.5% Brier improvement over a constant is, for practical purposes,
no information.

## 3. Repeatability — the decisive result

74 subjects have two or more sessions; 72 keep the same injured side across them.
Every pair of same-side sessions from the same subject is a test–retest.

| | deployment | clean mocap |
|---|---|---|
| session pairs | 258 | 258 |
| **binary agreement** | **0.667** | 0.744 |
| score correlation | +0.472 | +0.647 |

Floor is 0.5 (independent coin flips at this marginal rate); ceiling is 1.0, since
the true answer is identical for both sessions by construction.

**The model changes its mind about the same runner on a third of repeat scans.**
Even on perfect mocap it disagrees a quarter of the time. This is the number that
should govern phase 6: a screening tool that gives a different answer when you
walk in twice is not one you can put in front of a person, whatever its AUC.

> **Caveat, stated because it cuts against the strength of this claim.** The 258
> pairs come from 72 subjects, so they are not independent — a subject with 8
> sessions contributes 28 pairs and dominates. Agreement is therefore an estimate
> with real uncertainty, not a precise 0.667. It is not, however, plausibly near
> 1.0.

## 4. How wrong could the angle-convention transfer be?

`phase3_angles.py` flagged this and required phase 4 to handle it rather than
assume it away:

> Keypoint-derived joint angles are three-point angles between segment vectors.
> The Ferber waveforms are Cardan angles from marker-cluster segment coordinate
> systems. These are NOT the same quantity … it matters a great deal in phase 4.

It cannot be tested directly — no subject has both mocap and video. So instead of
asserting the transfer is fine, this bounds how wrong it could be. Residual
magnitude is scaled and the verdict re-measured (side-on bank, 30 fps):

| scale | realized error | AUC | 95% CI | survives |
|---|---|---|---|---|
| 0.5× | 1.01° | 0.599 | [0.519, 0.665] | ✅ |
| **1.0×** | **1.94°** | **0.583** | [0.515, 0.631] | ✅ |
| 1.5× | 2.90° | 0.567 | [0.510, 0.609] | ✅ |
| 2.0× | 3.87° | 0.554 | [0.505, 0.597] | ✅ |
| 3.0× | 5.83° | 0.534 | [0.494, 0.580] | ❌ CI touches chance |

**The conclusion is robust to a 2× error in the convention transfer and breaks at
about 3×.** Note also that even at 0.5× — half the measured error — AUC only
reaches 0.599, still under the bar. The finding does not depend on this
assumption in either direction.

## The split

Identical to phases 2, 4 and 5: `StratifiedGroupKFold(n_splits=5, shuffle=True)`,
**`groups=sub_id`**, seeds `[11, 23, 37, 53, 71]` → 25 folds.
818 unilateral-injury sessions / 675 subjects, chance 0.517. All preprocessing
inside the `Pipeline`; PCA at 3 components per channel; logistic regression fixed
in advance.

Negative controls for this cohort are reported in `results/phase04.md`
(demographics 0.486, provenance 0.449, structure 0.472; max deviation 0.051).

## What this means for phase 6

**The demo shell as originally conceived — show a runner which limb is at risk —
is not supportable by these numbers.** That is a finding, not a blocker, and it
is consistent with the rest of the project: this was always a measurement
exercise about how much is lost going from mocap to monocular video, and the
answer is that the loss is small because there was not much to lose.

Three framings that *are* supportable, in decreasing ambition:

1. **A methods demo.** Show the pipeline and the degradation curve — video in,
   kinematics out, and an honest display of how much signal survives. The
   deliverable is the measurement, which is what CLAUDE.md says the project is.
2. **A group-level tool.** AUC 0.57–0.61 is uninformative per person but can
   still rank a cohort. Nothing here supports an individual readout.
3. **A negative-result artifact.** The most defensible: publish the curve, the
   operating point, and the repeatability, as an argument about what monocular
   screening can and cannot do.

Any phase 6 that displays a per-user verdict must display the repeatability
number next to it.

## Reproduction

```
.venv/Scripts/python.exe scripts/phase4b_operating_point.py    # ~20 min CPU
```

Output: `results/phase4b_operating_point.json`.
