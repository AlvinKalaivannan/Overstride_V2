# Phase 1 — Injury classifier vs. the demographics-only control

**Status: draft for approval. Not yet executed.**

Written against `docs/data-inventory.md` (observed) and `CLAUDE.md`. Phase 0's
gate is recorded in `results/phase00.md`. Every number below is measured, not
assumed.

---

## Context

`CLAUDE.md` defines phase 1 as *"injury classifier on all 9 mocap waveforms"*,
gated on *"beats the demographics-only control"*, with a kill criterion: build
the control **first**, and if kinematics cannot beat it, report that and stop.

Phase 0 found the waveforms are not in the archive — they must be computed from
48 GB of raw markers by the bundled MATLAB pipeline, which is unscoped work.
That blocks the *kinematic* half of phase 1. It does not block the control, and
it does not block a cheaper read on whether kinematic signal exists at all.

| | Work | Blocked by waveform generation? |
|---|---|---|
| **1A** | Demographics-only control + provenance baseline | No |
| **1B** | Discrete kinematics from stored `dv_r` scalars | No |
| ~~1C~~ | Waveform classifier | **Yes** — separate spec, written after 1B |

**1B is not phase 1 as `CLAUDE.md` specifies it.** `dv_r` holds discrete
summaries — peak angles, excursions, angle at heel strike, percent-stance at
peak, peak velocities — for the same hip/knee/ankle channels, not the 101-point
curves. It is a de-risking check bought with one streaming pass we need to make
anyway, and it must be reported as such, never as the headline result.

### What each 1B outcome licenses

A waveform contains strictly more information than its own discrete summaries,
so the two directions are **not symmetric**:

- **`dv_r` beats the control** → strong green light. The curves carry a superset
  of this information.
- **`dv_r` does not beat the control** → informative, **not** a kill. Curve shape
  and timing morphology can carry signal that scalar summaries discard. It shifts
  the prior substantially toward "no signal", and that shift should be stated
  plainly, but it does not by itself end the project.

Only the waveform result (1C) can trigger `CLAUDE.md`'s kill criterion.

---

## The finding that reshapes this phase

**Missingness in the metadata indicates the label, and it is an artifact of data
collection, not physiology.** Measured on the phase 1 cohort:

| year | sessions | injured rate | `YrsRunning` missing | `Level` missing |
|---|---|---|---|---|
| 2009 | 34 | 0.353 | 0.794 | 0.000 |
| 2010 | 88 | 0.898 | 0.159 | 0.000 |
| 2011 | 181 | 0.934 | 0.370 | 0.000 |
| 2012 | 176 | 0.659 | 0.233 | 0.040 |
| 2013 | 178 | 0.522 | 0.112 | 0.000 |
| 2014 | 586 | 0.761 | 0.169 | 0.009 |
| 2015 | 314 | 0.490 | 0.427 | 0.350 |
| 2016 | 137 | 0.730 | 0.307 | 0.190 |
| **2017** | **51** | **0.000** | **1.000** | **1.000** |

- A blank `Level` marks an uninjured session with **95.0% precision** (189/199).
- `lvl_missing` alone: **AUC 0.660**. `yrs_missing` alone: **0.624**. Collection
  year alone: **0.615**. None of these touch biomechanics.
- The 2017 wave is 51 sessions / 17 subjects, entirely uninjured, with `Level`,
  `YrsRunning` and `Activities` blank on every row — a healthy-cohort study with
  no questionnaire.

**Consequences, binding on everything below:**

1. **No missingness indicators as features. No collection year as a feature.**
2. **Median imputation is not automatically safe.** Collapsing missing values to
   a single point creates a detectable spike a tree model can split on,
   re-introducing the same leak by the back door. This must be measured, not
   assumed away — see the provenance baseline.
3. **Complete-case analysis is not a safe fallback.** Requiring all control
   features present keeps 1,244 of 1,745 sessions but deletes **45% of the
   uninjured class** (576 → 316) and the entire 2017 wave, moving the injured
   rate from 0.670 to 0.746. It is a sensitivity check, never the primary.
4. **A provenance-only baseline is mandatory**, additional to what `CLAUDE.md`
   requires. If the demographics control does not clearly beat it, the control
   is measuring paperwork and every downstream delta is measured against noise.

---

## Prerequisites

`scikit-learn` is **not yet installed** — phase 0 did not need it. `uv add
scikit-learn` is the first execution step. Confirm at that point that
`StratifiedGroupKFold` accepts `shuffle` / `random_state` in the installed
version; the repeated-CV scheme depends on it. If not, permute group ordering
per repeat instead.

## Hard constraints

Violating any of these invalidates the result.

1. **`StratifiedGroupKFold` with `groups=sub_id`.** Never `train_test_split` on
   rows. No subject in both train and test in any fold, ever.
2. **All preprocessing inside the fold.** Enforce structurally with an sklearn
   `Pipeline` — imputer, scaler, encoder and estimator as steps, so `fit` cannot
   see test data. Never `fit_transform` before `cross_validate`.
3. **Mask sentinels to NaN before imputation**, in all four columns:
   `age` ∉ [10, 100] (1 row, `255`), `Height` ∉ [120, 250] (5 rows, `0`×3
   `999`×2), `Weight` ∉ [30, 200] (3 rows, `0`×2 `1564`×1), `YrsRunning` ∉
   [0, 80] (51 rows, all `999`).
4. **No missingness indicator features. No collection year.** See above.
5. **Do not align features to `InjSide`.** Uninjured subjects have no injured
   side, so any injured-side-relative feature encodes the label. That is label
   leakage and it would manufacture a result. Side-alignment is phase 5.
6. **Osteoarthritis excluded** from the primary analysis, reported separately.
   Measured cost: 14 sessions / 12 subjects. Near-zero, as expected.
7. **Label as phase 0 did** — a recorded diagnosis outranks a blank `InjDefn`.
   Import `injury_status` from `scripts/phase0_inventory.py`; do not reimplement.
8. **No deep learning.** scikit-learn only.
9. **Never load all session JSONs at once.** 1B streams one file at a time.
10. **Nothing synthesized.** Missing or unreadable files are reported.
11. **`$DATA_ROOT` from the environment, read-only.**

---

## Step 0 — cohort construction (shared by 1A and 1B)

Build once, in one module both steps import. This is the thing most likely to
quietly differ between the two models and invalidate the comparison. All figures
below are measured.

| stage | sessions | subjects |
|---|---|---|
| `run_data_meta.csv` | 1,832 | 1,402 |
| drop `unknown` status | 1,759 | 1,335 |
| exclude osteoarthritis | **1,745** | **1,323** |

- **Labels:** binary. Injured 1,169 sessions, uninjured 576. Positive rate
  **0.670**. Not severe enough to resample — report ROC-AUC primary, PR-AUC
  secondary.
- **`unknown` (73 sessions / 71 subjects) is dropped.** It is not a class, it is
  absence of information. State the drop in the report.
- **`pain` and `other` are kept.** They are non-diagnoses but carry a recorded
  severity, and dropping them would select on the outcome. This is **36.6% of
  injured sessions (428/1,169)** — a materially larger caveat than it sounds, and
  it belongs in the report's limitations, not a footnote.
- **Unit of analysis: the session**, all sessions, grouped by `sub_id`. 202
  subjects have >1 session (max 10).

**Subjects whose label changes between their own sessions: 16, accounting for 50
sessions.** Measured, not assumed. These are people who were injured at one
visit and healthy at another — e.g. `100004` injured with a calf strain in 2011,
uninjured in 2014. Grouping by `sub_id` puts them on both sides of their own
label within a fold's training set.

**Policy: keep them, report the count.** At 16 subjects (1.2%) the alternative —
dropping them — would delete genuine recovery signal and select on the outcome.
Re-state the number in `results/phase01.md`.

---

## Step 1A — the demographics-only control, and its own control

Features exactly as `CLAUDE.md` names them: `age`, `Gender`, `Height`, `Weight`,
`speed_r`, `YrsRunning`, `Level`.

Measured unusable rates within the 1,745-session cohort, **after** sentinel
masking:

| feature | unusable | note |
|---|---|---|
| `age` | 1 (0.1%) | |
| `Height` | 8 (0.5%) | |
| `Weight` | 5 (0.3%) | |
| `speed_r` | 0 (0.0%) | |
| `YrsRunning` | **495 (28.4%)** | 444 blank + 51 sentinel |
| `Gender` | 0 (0.0%) | |
| `Level` | **199 (11.4%)** | |

- Numerics: median imputation, fold-internal.
- `Gender` / `Level`: one-hot with an explicit `"missing"` level. **Note the
  tension** — an explicit missing level is exactly the leak of constraint 4 for
  `Level`, since blank-`Level` is 95% uninjured. Mitigate by fitting the control
  **both** ways (explicit level vs. mode-imputed) and reporting both. If they
  differ materially, the gap is the size of the leak.
- **`speed_r` is a control feature, deliberately.** `CLAUDE.md` calls run speed a
  confounder for *kinematic* models; here it is part of the boring explanation
  the control represents. Flag this in the report — it reads as a rule violation
  to anyone skimming.

**Models:** penalized logistic regression as primary (7 features,
interpretable); `HistGradientBoostingClassifier` as a secondary check for
non-linearity. **The control is the stronger of the two** — a weak control
flatters everything measured against it.

**Report coefficient signs and magnitudes.** If the control works we should be
able to say in one sentence why, most likely "older, heavier, slower."

### The provenance-only baseline (new, mandatory)

Fit on nothing but `lvl_missing`, `yrs_missing` and collection year — no
demographics, no kinematics. Same folds, same seeds.

Expected ≈ 0.66 from the single-feature numbers above. **Interpretation gate:**

- Control ≫ provenance baseline → the control is measuring demography. Proceed.
- Control ≈ provenance baseline → **stop and report.** The control is detecting
  which study a session came from, and no kinematic delta measured against it
  means anything. This would be a phase 1 finding in its own right.

---

## Step 1B — discrete kinematics from `dv_r`

**Extraction.** One streaming pass over the 1,745 cohort sessions, one file open
at a time, released before the next. Write a single feature table (~10 MB) to
`data/derived/` — gitignored, outside `$DATA_ROOT`.

- 76 scalars × 2 sides = 152 raw features, prefixed `left_` / `right_`.
- Add side-agnostic derived features: per-variable `mean` and `abs_diff` across
  sides. `abs_diff` captures asymmetry **without** referencing `InjSide`, so it
  respects constraint 5.
- **Resolve the `0`-as-missing question first.** Phase 0 flagged fields that are
  exactly `0` where the value is implausible (`ANKLE_DF_at_HS`,
  `KNEE_FLEX_EXCURSION`). Report the exact-zero rate per field and decide: true
  zero, or sentinel. A field that is 40% exact-zero is a sentinel, and treating
  it as a measurement injects a fake cluster. **Then re-run the missingness-vs-
  label check on those fields** — the metadata leak may have a kinematic twin.
- **Record `n_marker_channels` and `n_joints_landmarks` per session while
  streaming** (28 vs 30, and 10 vs 14 — phase 0 §5). Cross-tabulate against
  collection year and label before modelling. The extended marker set exists for
  only 1,082 of 1,798 subjects; if its presence tracks the collection wave, then
  *which markers exist* is another provenance leak, and it would contaminate 1C
  as well. This check is cheap now and expensive to retrofit.
- Verify the row count reconciles and every `sub_id` joins to the metadata.
  Report any file that fails to parse; never skip silently.

**Models:** same two families as 1A, same folds, same seeds. ~150–450 features
against 1,745 sessions, so regularization is doing real work — report the chosen
penalty strength and how it was selected (nested CV, or a fixed defensible
default; never tuned on the test fold).

**Speed control**, per `CLAUDE.md`:
1. kinematics alone,
2. kinematics + demographics — the honest "does it add anything" model,
3. a speed-matched subsample as a secondary check.

---

## Comparison protocol

Specified in advance, so results are not decided by post-hoc judgment.

- **Identical folds across all models.** Same `StratifiedGroupKFold` object, same
  `random_state`. Comparisons are paired by construction.
- **Repeat the CV** with ≥5 seeds (5 folds × 5 repeats = 25 estimates). Five
  folds alone is too thin to put a CI on.
- **Report the paired per-fold difference**, not two independent means. The
  headline is Δ AUC (kinematic − control) with its CI across folds.
- **"Clearly beats" = the paired ΔAUC CI excludes zero** and the effect is large
  enough to matter. Threshold stated before running, not after.
- **Every model reports against two baselines**: the demographics control and the
  provenance baseline.
- If it does not beat the control, **say so plainly in the first paragraph**.
  `CLAUDE.md`: *"A negative result is a valid result."*

---

## Deliverable

`results/phase01.md`, per `CLAUDE.md` §Reporting:

- The numbers, with cross-fold CIs
- The exact split and the `groups=` key
- n per group after filtering, what was excluded and why
- The control's score alongside every kinematic score
- The provenance baseline alongside both
- Anything surprising or that looks wrong

Plus the 1B verdict framed against "what each outcome licenses", and a go/no-go
on waveform generation.

Code as scripts, not notebooks. Figures to `figures/`, tables to `results/`.

---

## Gate

Phase 1 (this spec) is complete when:

1. `results/phase01.md` exists with the control's AUC and CI.
2. The provenance-only baseline is reported and the control clearly beats it —
   or the failure to do so is reported as the phase's finding.
3. The `dv_r` feature table reconciles to the 1,745-session cohort.
4. The marker-set-vs-collection-wave check is reported.
5. The paired ΔAUC and CI are reported for every kinematic model.
6. A go/no-go on waveform generation is stated with reasoning.

The `CLAUDE.md` phase 1 gate — *"beats the demographics-only control"* — is
**not** settled by this spec. It is settled by 1C, on waveforms.

---

## Out of scope

Waveform generation and the MATLAB pipeline (own spec, after 1B). Video, pose
estimation, phase 2 degradation. `InjSide` asymmetry modelling (phase 5).
Multi-class injury-type classification. Walking data. Any use of the word
"predict".
