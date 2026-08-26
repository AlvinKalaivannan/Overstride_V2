# Phase 8 — a positive control, and the detection floor

**Question.** Every control in this project is negative. Provenance,
demographics, structure and dominant leg all sit at chance, which establishes
that the within-subject setup does not leak. Nothing established the converse:
that the method has the *power* to find an asymmetry that is genuinely present.

Without that, two very different statements were not separated by the evidence:

- *we found nothing* — the physiology is not in these channels
- *this method finds nothing* — the design is underpowered

Five probes agreeing on a 0.610 ceiling (phases 5B–5D) is consistent with both.

**Answer: the design is not underpowered, by a wide margin.** It detects a
consistent limb asymmetry of **0.25°** at AUC 0.681 and **0.50°** at AUC 0.844.
The observed 0.610 carries as much discriminative information as **≈0.17° RMS**
of consistent asymmetry. The null is a real null.

---

## Split and cohort

Identical to phase 5, by construction — the point is to calibrate *that* design.

| | |
|---|---|
| split | `StratifiedGroupKFold(n_splits=5, shuffle=True)`, seeds `[11,23,37,53,71]` → 25 folds |
| `groups=` | `sub_id` |
| leakage check | `check_no_group_leakage` asserted on every fold list, including the synthetic-label ones |
| cohort | **818 unilateral sessions / 675 subjects** |
| excluded | uninjured (no injured side — including them makes side-alignment encode the label); bilateral; OA per `CLAUDE.md` |
| chance rate | 0.517 right-injured |
| features | `limbsag_mean` — sagittal hip/knee/ankle limb differences, the video-recoverable subset |
| model | logistic + PCA at 3 components/channel (9 total) — the exact configuration that produced 0.610 |

**Model family fixed in advance.** Phase 5 reported the better of logit/hgb per
feature set. Taking a max over families at every rung of a sweep would let model
selection bend the floor, so this pins the single family that produced the
headline. The unperturbed pipeline reproduces **0.610 exactly**, which is the
check that the harness is the same one.

**No rows were created, imputed or mocked.** This is the same controlled
perturbation of real measured waveforms that phases 2 and 4 use to build the
degradation ladder. Every number below is a property of the *design*, never of
the cohort.

---

## Design

**Two arms.**

- **Synthetic** — each *subject* is assigned a random synthetic injured side and
  the asymmetry is injected there. The real injury signal becomes uncorrelated
  with the synthetic label and acts as realistic background noise. AUC(0) must
  land at 0.5.
- **Additive** — injected on the *real* injured side, on top of the real signal.
  AUC(0) is the real 0.610.

Sides are drawn per subject, not per session, so two sessions of one person can
never carry opposite synthetic sides.

**Two shapes**, because a floor is only defined relative to what is injected:
`offset` (constant across all 101 stance points — the easy case) and `bump`
(Gaussian, peak at 40% of stance, σ 12% — localized, harder for a 3-component
PCA). Reported in both peak and RMS degrees, since a bump carries 0.459× the
energy of an offset at equal peak.

**The scale it competes against.** Between-subject SD of the sagittal limb
difference: ankle 2.22°, knee 3.61°, hip 2.73° — **mean 2.86°**.

---

## Results

### Synthetic arm — the floor (3 draws averaged, CI across 25 folds)

| peak δ | RMS δ | AUC (offset) | 95% CI | AUC (bump) | 95% CI |
|---|---|---|---|---|---|
| 0.00 | 0.00 | **0.480** | [0.431, 0.523] | **0.480** | [0.431, 0.523] |
| 0.10 | 0.10 / 0.05 | 0.539 | [0.478, 0.593] | 0.490 | [0.437, 0.527] |
| 0.25 | 0.25 / 0.11 | 0.681 | [0.634, 0.751] | 0.564 | [0.507, 0.620] |
| 0.50 | 0.50 / 0.23 | 0.844 | [0.813, 0.894] | 0.685 | [0.644, 0.739] |
| 0.75 | 0.75 / 0.34 | 0.932 | [0.910, 0.963] | 0.790 | [0.760, 0.834] |
| 1.00 | 1.00 / 0.46 | 0.972 | [0.956, 0.988] | 0.873 | [0.847, 0.906] |
| 1.50 | 1.50 / 0.69 | 0.995 | [0.988, 0.999] | 0.960 | [0.940, 0.980] |
| 2.50 | 2.50 / 1.15 | 1.000 | [0.999, 1.000] | 0.996 | [0.991, 1.000] |
| 4.00 | 4.00 / 1.84 | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |

**δ=0 lands at 0.480, CI [0.431, 0.523] — includes 0.5.** That is the sanity
check: the injection, not a bug, is doing the work.

### Additive arm — on top of the real signal

| peak δ | AUC (offset) | 95% CI | AUC (bump) | 95% CI |
|---|---|---|---|---|
| 0.00 | **0.610** | [0.532, 0.685] | **0.610** | [0.532, 0.685] |
| 0.25 | 0.717 | [0.642, 0.792] | 0.641 | [0.568, 0.719] |
| 0.50 | 0.858 | [0.796, 0.905] | 0.715 | [0.644, 0.787] |
| 1.00 | 0.979 | [0.959, 0.993] | 0.874 | [0.814, 0.919] |
| 2.50 | 1.000 | [1.000, 1.000] | 0.997 | [0.992, 1.000] |

### The floor, and what 0.610 is worth in degrees

| | offset (peak / RMS) | bump (peak / RMS) |
|---|---|---|
| smallest swept δ whose CI excludes 0.5 | 0.25 / 0.25 | 0.25 / 0.11 |
| δ reaching the 0.600 bar | 0.164 / **0.164** | 0.325 / **0.149** |
| δ reproducing the observed 0.610 | 0.175 / **0.175** | 0.345 / **0.158** |

**The two shapes agree to within 10% once expressed in RMS degrees** — 0.175 vs
0.158 — despite differing by a factor of two on peak. That agreement is the
reason the floor is worth quoting: it is a property of the design, not an
artefact of the injection shape.

---

## Reading

**The ambiguity is resolved.** The method detects a consistent sagittal
asymmetry of 0.25° at AUC 0.681 — an asymmetry far below anything a clinician
would call one, and well inside the noise of most marker labs. A design with that
much power did not miss a real effect. **The 0.610 ceiling reflects the absence
of a systematic injured-limb asymmetry in these channels, not an underpowered
test.** Phases 5B–5D's conclusion survives its own strongest challenge.

**0.610 expressed in degrees is ≈0.17° RMS.** Against a between-subject SD of
2.86°, that is roughly 0.06 SD. Stated plainly: whatever real limb signal exists
is equivalent, in discriminative terms, to a consistent asymmetry two orders of
magnitude smaller than the between-subject spread.

**A negative result with a calibrated floor is a much stronger object than a
bare null.** The claim is no longer "we looked and found little." It is: *we
looked with an instrument that resolves 0.25°, and found the equivalent of
0.17°.*

---

## The limit of this claim — stated because it is easy to over-read

**The injection is consistent across subjects.** Every session receives the same
profile in the same direction. That is the easiest possible signal and it is
almost certainly *not* how real injury asymmetry presents — different runners
compensate differently, and a heterogeneous asymmetry of much larger magnitude
could produce the same AUC as a small consistent one.

So the correct statement is **"0.17° RMS of *consistent* asymmetry"**, not "the
real asymmetry is 0.17°." The floor bounds the design's power against consistent
effects only.

This also means the tempting comparison — *the signal (0.17°) is smaller than the
lifting error (3.4°)* — does not carry as stated. The 3.4° is largely random
error, and random error and consistent offset do not trade off one-for-one. The
two numbers are not on the same axis and should not be put on one.

**The natural extension** is a heterogeneous injection: draw a per-subject
profile from a distribution and sweep its scale. That would calibrate power
against the kind of asymmetry actually expected, and it is the obvious follow-up
if a reviewer presses on the consistency assumption.

---

## What surprised me

**How steep the curve is.** 0.5° of consistent offset takes the model from
chance to 0.844, and 1.0° to 0.972. Going in, a floor somewhere near the 2.86°
between-subject SD seemed plausible; the actual floor is ~17× smaller. Averaging
101 stance points across three channels and 818 sessions buys far more
sensitivity to a consistent shift than intuition suggests.

**The bump/offset agreement in RMS.** Not designed for — the two shapes were
included to bound the floor from either side, and they turned out to bracket it
so tightly that the floor is effectively shape-independent. That was luck, but it
is a genuine internal check and it makes the number quotable.

**The additive arm rises more slowly than the synthetic arm from its own
baseline.** At 0.5° offset the synthetic arm gains +0.364 from 0.480 while the
additive arm gains +0.248 from 0.610 — consistent with AUC compressing as it
approaches 1, and with the real signal and the injected one not being aligned.
Nothing here looks wrong, but it is the reason the two arms should not be
differenced directly.

---

## Files

- `scripts/phase8_positive_control.py`
- `results/phase8_positive_control.json`
