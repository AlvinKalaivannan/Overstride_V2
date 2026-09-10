# Phase 9 (A1) — the limb signal does not survive a change of gait mode

**Question.** Phase 5 found the project's only signal-carrying task: identify
*which* limb is injured, within a session, from sagittal limb differences. It
tops out at 0.610 on running. Is that a property of **running**, or of **gait**?

**Answer: it does not reproduce in walking.** The primary test lands at
**0.547 [0.465, 0.627]** against a chance rate of 0.526, and fails the
pre-registered bar. All five negative controls sit at chance — more tightly than
in running — so this is a real null, not a broken harness.

**The 0.610 ceiling is not a general property of gait.** It did not reproduce in
a second gait mode, on largely the same people, with cleaner controls and a
comparable n.

---

## Pre-registration

Committed at `756ed4c` **before any walking curve existed**, and the guards and
sensitivity analysis at `00dd9b6` **before any walking AUC existed**. Nothing in
either commit altered the declared tests, the bar, or the stop condition.

- **OA excluded** — operator decision, 2026-08-26, recorded before generation.
  Inherits the `CLAUDE.md` rule rather than making a new one. This matters: OA is
  20.1% of walking sessions against 0.8% of running, so an OA call taken *after*
  seeing a result would have been the most effective way to manufacture one.
- **Paired design** — only subjects present in both gait modes, so gait mode is
  the only thing changing within a subject.
- **3 hypothesis tests**, Holm–Bonferroni.
- **Bar, inherited from phase 5 unchanged**: mean AUC ≥ 0.60 **and** the
  cross-fold CI excludes 0.5.
- **Stop condition**: any negative control ≥ 0.06 from chance → report the
  confound, publish no kinematic AUC.

---

## Split and cohort

| | |
|---|---|
| split | `StratifiedGroupKFold(n_splits=5, shuffle=True)`, seeds `[11,23,37,53,71]` → 25 folds |
| `groups=` | `sub_id` |
| leakage check | `check_no_group_leakage` asserted on every fold list |
| smallest training fold | 545 sessions (the PCA cap never binds) |
| features | `limbsag_mean` — sagittal hip/knee/ankle limb differences |
| model | best of logistic / HGB, PCA at 3 components per channel |

**Narrowing, with reasons:**

```
all walking sessions                      2088 / 1686 subjects
  unknown injury status dropped            −73   (absence of information, not a third class)
  osteoarthritis excluded                 −419 sessions / 246 subjects  (CLAUDE.md)
  unilateral + injured                     783
  and present in the run cohort            686 /  614 subjects  (the paired restriction)
  reproduce stored dv_w exactly            684   (−2 pipeline failures)
  MIN_STEPS >= 5                           683   (−1 session with a single step)
                                          ----
  final                                    683 /  611 subjects,  chance 0.526

running comparison (phase 5)               818 /  675 subjects,  chance 0.517
```

**1,290 subjects appear in both gait modes**; 611 survive into the final cohort.

### These curves are earned, not assumed

The walking waveforms did not exist — generating them was A1's real prerequisite
and the validation plan's step 1. The bundled RIC pipeline supports both gait
modes symmetrically (`processing_code_example.m:78-82`), so the walking batch
mirrors the running one field for field: `out.walking` / `out.hz_w` / `dv_w` in
place of `out.running` / `out.hz_r` / `dv_r`.

Each session is processed under **both** gait labels and kept only if the
regenerated `DISCRETE_VARIABLES` reproduce the archive's own stored `dv_w`:

```
684 / 686 sessions reproduce stored dv_w EXACTLY   (99.7%)
  2 genuine pipeline failures, both recorded and dropped:
      "Error calculating vertical oscillation"
      "Automated event detection unable to pull adequate number of strides"
label chosen by reproduction: walk 684, unresolved 2
```

That is the same standard that verified the running curves (1,745/1,745), and it
is what makes these curves data rather than an assumption.

**Sampling rate:** 594 @ 200 Hz, 89 @ 120 Hz. Running was almost entirely 200 Hz
(1,822 vs 10), so rate is a channel that exists here and not there — hence the
fifth control below.

---

## Negative controls — five, not four

All are constant within a session, so none of them *can* indicate which limb is
injured. If any scores above chance, the limb differences carry an artefact and
every kinematic number is void. They are re-asserted on the walking subset rather
than inherited: a control that cleared on running has said nothing about walking.

| control | AUC | 95% CI |
|---|---|---|
| provenance (`year`, `yrs_missing`, `lvl_missing`) | 0.480 | [0.417, 0.526] |
| demographics (`age`, `Height`, `Weight`, `speed_w`, `Gender`, `Level`) | 0.474 | [0.390, 0.555] |
| file structure | 0.478 | [0.427, 0.540] |
| dominant leg | 0.487 | [0.401, 0.536] |
| **sampling rate — walk-specific** | **0.486** | [0.450, 0.522] |

```
gate: max |AUC - 0.5| = 0.026 (demographics)  ->  PASS
```

**Every interval includes 0.5, which is the whole requirement.** Running's gate
figure was 0.051, but the two should not be compared as a measure of relative
cohort cleanliness — see the correction under *What surprised me*.

The fifth control was not in the validation plan. It was added because walking
mixes 120 and 200 Hz and decimation is exactly where phase 4's framerate artefact
came from. It cleared.

**`speed_w`, not `speed_r`.** Run speed is a confounder in the running analysis
and walk speed is the same confounder here; using the running column would have
silently controlled for a trial these subjects did not perform in this cohort.

---

## The pre-registered tests

| test | AUC | 95% CI | p | p (Holm) | bar |
|---|---|---|---|---|---|
| **`limbsag_mean` (primary)** | **0.547** | **[0.465, 0.627]** | 8.3e−05 | 8.3e−05 | **fail** |
| `limb9_mean` | 0.553 | [0.473, 0.641] | 3.2e−06 | 8.0e−06 | fail |
| `limb15_mean` | 0.561 | [0.463, 0.652] | 2.7e−06 | 8.0e−06 | fail |

**None clears. Every CI includes 0.5.**

### Sensitivity analysis — declared in advance, and it passed

22.7% of walking sessions carry `eventsflag_mean < 1`: automated event detection
partially fell back to foot-forward/foot-back. Running has no equivalent. The
primary was re-run on the clean subset — declared in `00dd9b6` before any result
existed, explicitly **outside** the Holm family, because it is the same hypothesis
on cleaner data rather than a fourth test.

| | n | AUC | 95% CI |
|---|---|---|---|
| all sessions | 683 | 0.547 | [0.465, 0.627] |
| `eventsflag_mean == 1.0` | 528 | 0.542 | [0.443, 0.633] |
| | | **Δ −0.004** | |

**Event-detection quality is not driving the result.** Had this not been declared
up front, running it afterwards would have been indistinguishable from hunting
for a subset where the number moved.

---

## Read the CIs, not the p-values

**The Holm-adjusted p-values are 8e−05 and smaller while every CI includes 0.5
and every test fails the bar. The CIs are right and the p-values are not
evidence.**

This is the same pathology phase 5C caught and documented: a fold-level test
there returned q = 0.000 for ITB syndrome while that condition's own cross-fold
CI ran [0.410, 0.986].

The cause is structural. The 25 folds are 5 seeds × 5 splits of **one** dataset —
each subject appears in the test set five times, once per seed. They are not
independent draws, so the standard error of the fold mean is understated and any
t-test built on it is anti-conservative. `fold_p`'s own docstring says it exists
only to order the Holm family.

The p-values appear here because the pre-registration named Holm–Bonferroni as
the correction method and changing that after the fact would be worse. **The bar
is the criterion**: mean ≥ 0.60 *and* CI excludes 0.5. All three tests fail it.

**Recommendation for any future phase:** drop `fold_p` entirely and use the
subject-level bootstrap introduced in `scripts/phase5c_features.py`
(`subject_bootstrap_ci`), which resamples subjects rather than folds and does not
have this defect.

---

## What this does and does not establish

**Established.** The limb signal does not replicate in walking at the
pre-registered bar. With five controls at chance and a comparable n, that is a
clean negative result, and it means **the 0.610 ceiling is not a general property
of gait** — it did not survive a change of gait mode in largely the same people.

**Not established: that walking and running differ from each other.** The script's
pre-declared verdict band prints *"the running signal is RUNNING-SPECIFIC"*, and
that is stronger than the data supports. Walking's [0.465, 0.627] and running's
[0.532, 0.686] **overlap substantially**. No test compared them, and the intervals
would not support one if it had been run. The band was written before the result
and is left in the JSON as declared; this section is the correction to it.

**Not a replication, and it must not be described as one.** Same clinic, same
collection waves, same paperwork confound, largely the same people. A1 is a
**gait-mode generalisation test**. The one-cohort limitation in `docs/REVIEW.md`
§7.6 survives phase 9 untouched.

### The stronger test that was not run

**611 subjects appear in both gait modes.** Running could be re-scored restricted
to exactly those subjects and differenced against walking fold-by-fold — a truly
paired comparison, stronger than either number alone.

It was not pre-registered, so it is not in this phase's results and no number from
it appears above. If it is wanted it is a separate, separately-declared test.
Given that phase 8 established the design resolves a 0.25° consistent asymmetry,
its marginal value is modest.

---

## What surprised me

**The cohort's known weaknesses did not bite.** Going in, the plausible
explanation for a weaker walking result was a messier cohort: a different
population, a mixed 120/200 Hz sampling rate, 22.7% partial event detection. All
three were real concerns. None of them showed up — the controls are at chance and
the sensitivity analysis moved the primary by 0.004. The comfortable explanation
is not available, which leaves the uncomfortable one.

> **Correction to an earlier draft of this section.** It read *"the controls are
> tighter than running's — worst deviation 0.026 against 0.051"* and concluded the
> walking cohort was the cleaner of the two. **That comparison does not hold and
> the conclusion should not be drawn from it.** Running's worst deviation was
> provenance at **0.449 — 0.051 *below* chance**, and leakage pushes a control
> *above* chance, so that number was estimation noise rather than contamination.
> The CI widths are also comparable (running 0.100–0.186, walking 0.072–0.165),
> and the comparison took the maximum of four noisy estimates against the maximum
> of five, which is the kind of comparison that manufactures differences.
>
> What the controls do establish is narrower and entirely sufficient: **every
> control CI in both cohorts includes 0.5, so neither design leaks and the walking
> null is not a leakage artefact.** No claim about relative cohort quality is
> supported, and none is needed.

Worth noting without chasing: all five walking controls sit slightly *below*
chance (0.474–0.487). Given intervals 0.072–0.165 wide, and controls that share
folds, subjects and overlapping features, that is noise rather than a pattern.

**The sampling-rate control was the tightest of the five** (0.486, [0.450, 0.522])
despite being the one added specifically out of suspicion. The 101-point stance
normalisation absorbs the 120/200 Hz difference as intended.

**The direction of the ladder inverted.** In running, the *narrowest* feature set
did best (`limbsag` 0.610 > `limb9` 0.603 > `limb15` 0.602). In walking the order
reverses (`limbsag` 0.547 < `limb9` 0.553 < `limb15` 0.561). All three CIs overlap
almost entirely, so this is not a finding — but it is the pattern one expects when
sets are fitting noise rather than a shared underlying signal, since added capacity
then stops being penalised.

---

## Files

- `scripts/phase9_walk_cohort.py` — pre-registered cohort and session list
- `scripts/matlab/batch_waveforms_walk.m` — walking waveform generation (resumable)
- `scripts/matlab/run_walk_batch.m` — detached entry point
- `scripts/phase9_assemble.py` — curve files → `(683, 54, 101)` array
- `scripts/phase9_walk_limb.py` — the analysis
- `results/phase9_cohort.json`, `results/phase9_walk_limb.json`
