# Phase 5B — is 0.610 a limit of the signal or of the representation?

**Of the signal.** The pre-registered dose-response test fails, a completely
different measurement modality lands on the same number to three decimals, and
the learning curve has saturated. Three independent lines all say the same thing:
there is no representation of this data that rescues the injured-limb task.

| diagnostic | result | verdict |
|---|---|---|
| **D1 severity dose-response** (primary) | ρ = **+0.044** [−0.038, +0.130], p = 0.228 | ❌ **fails** — direction not supported |
| **D2 `dv_r` clinical probe** | **0.610** vs reference **0.610**, ΔAUC +0.000 [−0.067, +0.071] | ❌ no headroom from a different modality |
| **D3 learning curve** | 0.585 → 0.599 → 0.608 → **0.610** | ❌ saturated; final increment +0.001 |

Reference: `limbsag_mean` AUC **0.610** [0.532, 0.686], the phase 5 winner,
reproduced here exactly.

---

## The split

818 unilateral-injury sessions / 675 subjects, chance 0.517.
`StratifiedGroupKFold(n_splits=5, shuffle=True)`, **`groups=sub_id`**, seeds
`[11, 23, 37, 53, 71]` → 25 folds, shared with phases 4 and 5 so every delta is
paired. `check_no_group_leakage` asserted on every fold.

**Negative controls, re-asserted in-script** (`assert worst < 0.08`), not
inherited: demographics 0.486, provenance 0.449, structure 0.472.
**Max |AUC − 0.5| = 0.051 → PASS.**

---

## D1 — severity dose-response (PRIMARY, pre-registered)

**The direction was declared before fitting: more severe → more asymmetry.** If
the limb signal is physiological it must be stronger in runners whose injury
affects them more.

`InjDefn` gives an ordered training-disruption ladder:

| rank | stratum | n |
|---|---|---|
| 1 | Continuing to train in pain | 228 |
| 2 | Training volume/intensity affected | 349 |
| 3 | 2 workouts missed in a row | 185 |

Excluded: 51 sessions reading "No injury" despite carrying a real `SpecInjury`
(the diagnosis-outranks-blank-severity rule in CLAUDE.md), and 5 blank. 762
remain.

### Primary test — asymmetry magnitude vs severity rank

A scalar per session (RMS of the sign-corrected sagittal limb difference)
regressed on severity rank across all 762. Not a classifier, so it uses every
session and is far better powered than per-stratum AUC. CI from a 2,000-draw
bootstrap **resampled by subject**, since sessions cluster within subject.

| | ρ | 95% CI | p |
|---|---|---|---|
| raw | **+0.0437** | [−0.0377, +0.1296] | 0.228 |
| partial (speed_r, age, Height, Weight removed) | +0.0487 | — | 0.179 |

**The CI spans zero.** Confound check — asymmetry magnitude against each
covariate: speed_r +0.087, age −0.033, Height −0.074, Weight −0.039. Nothing
large enough to be masking an effect, and the partial correlation is essentially
the raw one.

### Secondary — AUC within each stratum

Folds regenerated per stratum, so these are **not paired across strata**.

| rank | stratum | n | AUC | 95% CI |
|---|---|---|---|---|
| 1 | Continuing to train in pain | 228 | **0.622** | [0.495, 0.743] |
| 2 | Training volume/intensity affected | 349 | 0.619 | [0.515, 0.730] |
| 3 | 2 workouts missed in a row | 185 | **0.521** | [0.383, 0.671] |

The ordering is **anti-monotone** — best in the mildest group, at chance in the
most severe. **This should not be over-read**: the CIs are wide and heavily
overlapping, so the ordering is not itself significant. What matters is that the
well-powered primary test is flat and the secondary does not rescue it.

> **A limitation of this test, stated because it softens the inference.**
> `InjDefn` measures *training disruption*, not *mechanical severity*. A runner
> continuing to train in pain may have a chronic mechanical problem with clear
> asymmetry; one who missed two workouts may have an acute strain that has
> largely resolved by testing. So a flat dose-response is evidence against a
> physiological signal, but it is not a knockout on its own. It is the agreement
> with D2 and D3 that makes the picture consistent.

---

## D2 — the `dv_r` clinical ceiling probe

**This is a diagnostic and never a deliverable.** These 39 metrics are mostly
frontal/transverse — pronation onset/offset, hip adduction, pelvic drop, step
width, medial heel whip — precisely what a single side-on camera recovers
*worst*. Phase 0 already established `dv_r` cannot support the degradation
analysis. It is used here only to separate two very different conclusions.

39 of 76 paired `left_`/`right_` metrics have both limbs populated in >50% of the
cohort (most at 100%). Signs derived from population `corr(L, R)` rather than
assumed — **38 shared, 1 mirrored** (`PELVIS_DROP_EXCURSION`, r = −0.73).
Deriving this wrong is the bug that made `asym9` the worst family in phase 1D.

| feature set | AUC | 95% CI | Δ vs reference | Δ CI | excludes 0 |
|---|---|---|---|---|---|
| `limbsag_mean` (reference) | 0.610 | [0.532, 0.686] | — | — | — |
| **`dv_r` limb difference (39 clinical)** | **0.610** | [0.543, 0.664] | **+0.000** | [−0.067, +0.071] | no |
| `dv_r` + `limbsag` combined | 0.614 | [0.540, 0.693] | +0.005 | [−0.069, +0.052] | no |

**Two entirely different measurement modalities — 303 sagittal waveform points
through PCA, and 39 clinical frontal/transverse scalars — land on 0.610.** And
combining them adds nothing (+0.005, CI spanning zero).

That convergence is the strongest single result in this phase. If the ceiling
were an artefact of the sagittal waveform representation, a clinical scalar set
built from different planes and different physics would not reproduce it to three
decimals. **The limitation is the signal.**

---

## D3 — would more data help?

`limbsag_mean` retrained on subsampled training subjects, same 25 folds:

| training subjects | AUC | 95% CI | increment |
|---|---|---|---|
| 25% (~135) | 0.585 | [0.514, 0.663] | — |
| 50% (~270) | 0.599 | [0.510, 0.674] | +0.014 |
| 75% (~405) | 0.608 | [0.518, 0.693] | +0.009 |
| **100% (~540)** | **0.610** | [0.532, 0.685] | **+0.001** |

**Saturated.** The total 25→100% gain is +0.025, but it is nearly all in the
first doubling — the last quarter of the data bought **+0.001**. Extrapolating
the shape, doubling the cohort would be worth on the order of +0.005.

> The script originally thresholded on the total gain and printed "more data may
> help". That reads the wrong number: what matters is whether the curve is still
> climbing at the right-hand end. Fixed to report the final increment.

**n = 675 subjects is not the constraint.** Collecting more sessions of the same
kind would not change the conclusion.

---

## What this means

The three diagnostics were chosen to be independent, and they agree:

1. The signal does not scale with how badly the runner is affected.
2. A different measurement modality hits the identical ceiling.
3. More data of the same kind is saturating.

**Together they say the ~0.61 ceiling is a property of what unilateral running
injury does to gait symmetry in this cohort, not of how it has been
represented.** Phase 5C's plan pre-registered that a flat D1 would scope Stage 2
down to the cheap items — that is what happens, and the reason is recorded there.

## What would still be worth trying, and what would not

**Not worth trying:** more channels, better models, more sessions of the same
kind, or any further work on the video path (phase 4B already showed it costs
only ~0.03).

**Possibly worth trying, in order:**

1. **Within-session stride distributions** — the one information axis neither D2
   nor D3 covers, since both are session-level aggregates. Run in phase 5C.
2. **Per-condition models** — a pooled cohort mixing ITBS, PFPS, calf strain and
   158 sessions labelled only "pain" may be diluting a condition-specific effect.
   Underpowered at n ≈ 100; hypothesis-generating only. Run in phase 5C.
3. **Accepting the ceiling and engineering around it** — selective
   classification, multi-session averaging. Deliberately deferred so it does not
   compete with the above.

## Reproduction

```
.venv/Scripts/python.exe scripts/phase5b_ceiling.py    # ~15 min CPU
```

Output: `results/phase5b_ceiling.json`, including per-fold AUCs for every model.
