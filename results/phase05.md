# Phase 5 — within-subject injured-limb identification

**The pre-registered bar is cleared. Kinematics do carry injury information —
but the result does not rescue the screening claim, and the distinction matters.**

Best kinematic model: **AUC 0.610 [0.532, 0.686]** identifying *which limb* is
injured, within the same person and the same trial. The bar, fixed before any
model was fitted, was mean AUC ≥ 0.60 with the cross-fold CI excluding 0.5.

The decisive supporting fact is not the AUC. It is that **every confound that has
dominated this project sits at chance here**, exactly as the design requires.

---

## The negative-control gate — the strongest validation in the project

Demographics, provenance and file structure are constant within a session, so
they *cannot* indicate which limb is injured. They were run as a hard gate: any
meaningful departure from 0.5 would mean the setup leaks and every number is
void.

| control | AUC (phase 5) | same model, between-subject |
|---|---|---|
| provenance-only | **0.449** | 0.775 pooled / **0.863** PFPS |
| demographics (`CONTROL_CLEAN`) | **0.486** | 0.656 pooled / 0.764 PFPS |
| file structure | **0.520** | 0.748 |
| `DominantLeg` | **0.500** | — |
| `DominantLeg` + demographics | 0.471 | — |

**Max |AUC − 0.5| = 0.051 → PASS.** The provenance signal that scored 0.863 in
phase 1D is annihilated. This is what "eliminated by construction" looks like
when it is actually verified rather than asserted, and no earlier phase had a
control this decisive available.

## The split

`StratifiedGroupKFold(n_splits=5, shuffle=True)`, **`groups=sub_id`**, seeds
`[11, 23, 37, 53, 71]` → 25 folds, generated once and reused by every feature set
so deltas are paired. `check_no_group_leakage` asserted and passed on every fold.
All preprocessing inside the `Pipeline`.

## n after filtering

| | sessions | subjects |
|---|---|---|
| with waveforms + ≥8 strides | 1,721 | — |
| **unilateral injured (analysed)** | **818** | **675** |
| bilateral, excluded (not a side) | 300 | — |

**Uninjured subjects are excluded by design.** They have no injured side, so
including them would make the side-alignment itself encode the label — the
leakage trap flagged back in the phase 1 spec, here avoided by construction
rather than mitigated. Chance is **0.517** (423 Right / 395 Left).

## The mirroring correction

Derived from population `corr(L, R)` and asserted against anatomy before use:

| joint | plane 0 | plane 1 | plane 2 (sagittal) |
|---|---|---|---|
| ankle | −0.513 (mirrored) | −0.479 (mirrored) | +0.804 (shared) |
| knee | −0.582 (mirrored) | −0.463 (mirrored) | +0.757 (shared) |
| hip | −0.375 (mirrored) | −0.392 (mirrored) | +0.889 (shared) |

So the limb difference is `R + L` for planes 0/1 and `R − L` for plane 2.
Flexion/extension is anatomically symmetric; ab/adduction and rotation are
defined relative to the midline and mirror.

> **This confirms a bug in phase 1D.** Its `asym9 = |L − R|` measured a *sum of
> magnitudes* on 6 of 9 channels, not asymmetry. That is consistent with `asym9`
> being the worst-performing family in all five conditions there. Corrected here.

Foot plane 1 and all pelvis planes are *not* mirrored — noted, and outside the
asserted ankle/knee/hip set.

---

## Results

| feature set | AUC | 95% CI | CI excludes 0.5 |
|---|---|---|---|
| **limbsag_mean [matched]** | **0.610** | [0.532, 0.686] | ✅ |
| limb9_mean [matched] | 0.603 | [0.522, 0.699] | ✅ |
| limb15_mean [matched] | 0.602 | [0.504, 0.677] | ✅ |
| limbsag_mean | 0.602 | [0.536, 0.677] | ✅ |
| limb15_mean | 0.601 | [0.501, 0.694] | ✅ |
| limb9_dist (stride quartiles) | 0.599 | [0.510, 0.677] | ✅ |
| limb9_mean | 0.598 | [0.515, 0.674] | ✅ |
| limb9_all (mean+SD+quartiles) | 0.593 | [0.500, 0.679] | ✗ |
| **limb9_sd (variability)** | **0.505** | [0.440, 0.589] | ✗ |

**The evidence is the consistency, not the maximum.** Reporting only the best of
nine sets would inflate the result. Of the eight mean- or quantile-based sets,
**seven have CIs excluding 0.5** and all cluster in 0.593–0.610. They are not
independent — they share channels — but a fluke would not produce a tight cluster
alongside four negative controls sitting at 0.45–0.52.

**Stride-to-stride variability is at chance (0.505).** This contradicts the hint
from phase 1D, where SD looked like the best kinematic family in 3 of 5
conditions. That apparent advantage was between-subject and has not survived a
design where the confound is removed. Adding quartiles or SD to the mean curves
gains nothing (`limb9_all` −0.005 vs `limb9_mean`). **The signal is in the mean
curve, not the distribution around it** — which retires the second phase 5
component as a dead end rather than a promising lead.

### Degradation ladder — capacity matched (3 PCA components per channel)

| rung | channels | AUC | ΔAUC vs full mocap |
|---|---|---|---|
| limb15_mean (full mocap) | 15 | 0.602 | — |
| limb9_mean | 9 | 0.603 | +0.001 [−0.064, +0.044] |
| **limbsag_mean (video-recoverable)** | **3** | **0.610** | **+0.008 [−0.063, +0.112]** |

**The degradation from full mocap to the sagittal subset is indistinguishable
from zero.** Restricting to the three flexion/extension channels a side-on camera
can see loses nothing measurable. With capacity matched per channel, the phase 1C
artifact where narrower sets looked better is gone — the rungs are genuinely flat.

---

## What this does and does not establish

**Falsified:** `CLAUDE.md`'s premise that failure would mean "the gait data
carries no injury signal beyond older, heavier, slower". It does carry signal.
Four independent confounds at chance while kinematics reach 0.61 is not a
demographic artifact.

**Not established: screening.** This task *presupposes you already know the
person is injured* and asks only which side. It is localisation, not detection.
Nothing here shows that an injured runner can be distinguished from a healthy one
— every phase that tested that failed, and this result does not overturn them.

**The reconciliation** is that between-subject kinematic variation is far larger
than the injury effect. Comparing a runner's limbs against each other removes
that variance; comparing runners against each other does not, and the effect
drowns. That is a statement about the measurement design, not about physiology.

**The effect is modest.** AUC 0.610 with CI reaching 0.532 is a real but small
effect on a balanced two-class problem. It supports "kinematics contain injury
information"; it does not support a product.

## Regroup on the kill criterion

Against the framework fixed before the result:

- The bar was cleared → **kinematics encode injury; the between-subject failures
  are a confounding and normalisation problem, not absence of signal.**
- But `CLAUDE.md`'s kill criterion is about *screening*, and screening remains
  unsupported across phases 1C, 1D and every cohort tested.

**The honest position: the kill criterion stands for screening, and its stated
rationale is now known to be wrong.** Both halves need reporting — dropping
either would misrepresent the evidence.

**The available path** is to reframe the deliverable onto the within-subject
task, where the degradation curve is measurable, confound-immune, and already
shows sagittal ≈ full mocap. That changes the claim from "screen runners for
injury from video" to "localise an existing injury from video" — narrower,
defensible, and actually supported.

**The obstacle that path must clear**, stated plainly: this task needs *bilateral*
kinematics, and a single side-on camera occludes the far limb. The flat ladder
above says nothing about that — it varies which channels are used, not how well
each limb is observed. **The real video degradation for this task is far-limb
recovery error, and it is unmeasured.** That is phase 3's question, and it should
be answered before the reframing is committed to.

## Anything that surprised us, or looks wrong

1. **Provenance at 0.449 after 0.863 in phase 1D.** The size of that collapse is
   the clearest evidence the confound was structural, not physiological.
2. **Variability carries nothing (0.505)** despite phase 1D suggesting it was the
   most promising family. A between-subject lead that vanished under a clean
   design — a useful reminder about which phase's evidence to trust.
3. **The sagittal-only set is the best, not the worst.** Three channels match
   fifteen. Either the frontal/transverse channels add noise at this sample size,
   or the injury signature is genuinely sagittal.
4. **Foot plane 1 and all pelvis planes are not mirrored**, unlike ankle/knee/hip.
   Handled by deriving signs from data rather than assuming.

## Reproduction

```
.venv/Scripts/python.exe scripts/phase5_features.py   # stride quartiles, ~12 min
.venv/Scripts/python.exe scripts/phase5_limb.py       # ~7 min
```

Seeds fixed. Per-fold AUCs in `results/phase5_limb.json`.

## Verification

| # | item | status |
|---|---|---|
| 1 | negative controls ≈0.5 (hard gate) | ✅ max dev 0.051, PASS |
| 2 | mirror signs derived and asserted vs anatomy | ✅ |
| 3 | cohort reconciles, bilateral excluded | ✅ 818 / 675, 300 excluded |
| 4 | `check_no_group_leakage` every fold | ✅ |
| 5 | phase 1D `asym9` bug quantified | ✅ 6 of 9 channels were summing |
| 6 | no `data/` paths tracked | ✅ |
