# Phase 1C — waveforms generated, benchmark repaired, signal measured

**The waveforms now exist and are verified. They do not beat a decontaminated
demographics control anywhere the comparison is clean.**

Three results, in order of what they change:

1. **Generating the curves worked.** 1,745/1,745 sessions processed, zero
   failures, and every one reproduces the archive's own `dv_r` exactly. The
   `(1745, 54, 101)` array is trustworthy in a way `dv_r` never was.
2. **Waveforms are markedly less wave-contaminated than `dv_r`.** They recover
   collection year at **+0.091** over baseline against `dv_r`'s **+0.182** and
   file metadata's **+0.299**. Deriving kinematics from raw markers escapes most
   of the provenance channel.
3. **But the injury signal is not there.** Best case is the sagittal set on the
   full cohort: ΔAUC **+0.041**, CI **[−0.006, +0.087]**, 23/25 folds positive —
   suggestive, and short of the bar set in advance. In the protocol-homogeneous
   stratum nothing exceeds 0.572. Under leave-one-wave-out nothing generalizes.

---

## The split

`StratifiedGroupKFold(n_splits=5, shuffle=True)`, **`groups=sub_id`**, 5 seeds
`[11, 23, 37, 53, 71]` → **25 folds**, generated once and reused by every model
so all comparisons are paired. `check_no_group_leakage` passes on every fold in
every cohort. Leave-one-wave-out uses 8 folds (2017 skipped — 51 sessions, zero
positives, AUC undefined); subjects spanning two waves are dropped from training
so the grouping rule still holds.

All preprocessing — median imputation, scaling, **PCA(30)** — is inside an
sklearn `Pipeline`, fit on the training fold only.

## n per group

| stage | sessions | subjects |
|---|---|---|
| `run_data_meta.csv` | 1,832 | 1,402 |
| drop `unknown`, exclude OA | 1,745 | 1,323 |
| waveforms generated and resolved | **1,745** | **1,323** |
| 28-channel stratum (primary) | **876** | **619** |

Positive rate 0.670 full, **0.782** in the 28-channel stratum (~191 uninjured
sessions — the cost of protocol homogeneity, flagged before running).

---

## Step B — waveform generation

Adapted from the bundled `processing_code_example.m`. **The archive is never
modified**; `scripts/vendor_matlab_code.py` makes a patched copy and asserts the
source is unchanged.

**Three R2023a → R2026a breakages**, each fixed and documented:

1. `import classreg.learning.classif.CompactClassificationDiscriminant` — a
   parse-time error; that internal namespace is gone.
2. Removing the import is **not** sufficient: `gaitClass.mat` holds a
   `ClassificationDiscriminant` that R2026a cannot instantiate at all. It loads
   as `uint32`.
3. `nanmin` / `nanmean` / `nanmedian` were removed. Restored as shims rather
   than editing numerical code.

**(2) forced a behavioural change, so it was verified rather than assumed.** The
classifier's only job is walk-vs-run. The batch runs each session as `run`,
compares `DISCRETE_VARIABLES` against the stored `dv_r`, and retries as `walk`
if they disagree — recovering the lost classifier's decision from the archive's
own output.

| | |
|---|---|
| sessions processed | **1,745 / 1,745**, 0 errors |
| reproduce stored `dv_r` | **1,745 / 1,745** |
| resolved as `run` / `walk` | 1,743 / **2** |
| throughput | 3.5 s/session, ~1.7 h total |
| outputs | `waveforms_mean.npy` (1745, 54, 101) 38 MB; per-step curves 1.0 GB |

The 2 `walk` sessions matter: they are filed under running in
`run_data_meta.csv` but only reproduce as walks (40/40 live variables vs 4/40).
The pipeline's own classifier disagreed with the metadata, and without this
check they would have been silently mis-processed.

### The sagittal plane is index 2, not index 0

`docs/ferber-schema.md` §3 gives the Cardan order as flexion/extension →
ab/adduction → rotation, implying plane 0. **The emitted array disagrees.**
Measured against the three live sagittal `dv_r` peaks:

| variable | r \| plane 0 | r \| plane 1 | **r \| plane 2** |
|---|---|---|---|
| `L_KNEE_FLEX_PEAK_ANGLE` | 0.037 | −0.262 | **−0.991** |
| `L_HIP_EXT_PEAK_ANGLE` | −0.115 | −0.245 | **−0.999** |
| `L_ANKLE_DF_PEAK_ANGLE` | −0.058 | 0.017 | **−0.997** |

All six checks (both sides) agree. The sign convention is also inverted relative
to `dv_r`, which does not affect a classifier. **Had this not been checked, the
entire degradation ladder would have measured the wrong channels** —
`scripts/phase1c_verify_sagittal.py`.

**Data-quality note:** all 30 angle channels are physiologically plausible (max
|178°|, zero sessions above |200°|). Extremes are confined to *velocity*
channels — ankle ab/adduction velocity reaches 63,700 °/s, an artifact of
differentiating rotation angles. `wave9` and `wave3` are angle-only and
unaffected; `wave54` includes those velocities and should be read with that in
mind.

---

## Step A/C — results

### Q1: can these features recover the collection year?

Multiclass, grouped, against always guessing the modal year (0.336).

| features | year accuracy | lift |
|---|---|---|
| session structure (file metadata) | 0.635 | **+0.299** |
| `dv_r` live (80 scalars) | 0.517 | **+0.182** |
| **waveforms `wave9`** | 0.427 | **+0.091** |
| **waveforms `wave3` (sagittal)** | 0.406 | **+0.070** |

**This is the phase's one clearly positive finding.** Recomputing kinematics from
raw markers halves the provenance leak relative to the stored summaries. The
curves are a substantially cleaner substrate than anything used before.

### Q2: injury classification

| model | 28-ch (primary) | full | LOWO |
|---|---|---|---|
| provenance-only | 0.542 | **0.775** | 0.630 |
| CONTROL_CLEAN + structure | 0.571 | **0.748** | 0.631 |
| `dv_r` live | 0.557 | 0.687 | 0.630 |
| **wave54** | **0.572** | 0.697 | 0.590 |
| **wave3 (sagittal)** | 0.552 | **0.697** | **0.614** |
| **wave9** | 0.531 | 0.658 | 0.590 |
| **CONTROL_CLEAN** | **0.568** | 0.656 | 0.608 |

Paired deltas against `CONTROL_CLEAN`:

| comparison | cohort | ΔAUC | CI | folds |
|---|---|---|---|---|
| wave3 − control | full | **+0.041** | [−0.006, +0.087] | 23/25 |
| wave54 − control | full | +0.040 | [−0.012, +0.092] | 22/25 |
| wave9 − control | full | +0.002 | [−0.050, +0.069] | 12/25 |
| wave3 − control | 28-ch | −0.016 | [−0.146, +0.126] | 10/25 |
| wave3 − control | LOWO | +0.006 | [−0.076, +0.080] | 5/8 |
| **wave9 − (control + structure)** | full | **−0.090** | **[−0.141, −0.037]** | **0/25** |

Every kinematic comparison against the control includes zero. The only CI that
excludes zero does so in the wrong direction: waveforms lose decisively to
demographics-plus-file-metadata.

### The degradation ladder runs backwards

`wave3` (3 sagittal channels — what a single side-on camera recovers) matches or
beats `wave9` (9 channels) in all three cohorts: full 0.697 vs 0.658 (wave9 −
wave3 = −0.039, only 2/25 folds positive), 28-ch 0.552 vs 0.531, LOWO 0.614 vs
0.590.

**Do not read this as "video-constrained kinematics beat mocap."** PCA is fixed
at 30 components regardless of input width, so `wave3` spends its budget on 3
channels while `wave9` spreads the same 30 across 9 — `wave3` retains more
sagittal detail per component. The comparison is confounded by the component
budget and needs a matched-capacity design before any degradation number is
quotable. That is a phase 2 requirement, and it is now a known trap rather than
a surprise.

---

## What this means

**The kinematic mechanism is now available and verified.** That was the blocker,
and it is cleared: the curves exist, reproduce the archive exactly, are far less
provenance-contaminated than the stored scalars, and the sagittal channels are
correctly identified. Phase 2 can be built.

**But this dataset does not show mocap kinematics beating demographics.** Not on
the full cohort (+0.041, CI crossing zero), not in the protocol-homogeneous
stratum (−0.016), and not across unseen collection waves (+0.006). The signal
that made phase 1A look promising was provenance, and removing provenance
removes the signal.

**`CLAUDE.md`'s kill criterion is now genuinely in play.** It says: if the
kinematic model cannot beat the demographics control, report it and stop. The
waveforms — the real thing, not a proxy — do not beat it. The honest reading is
that the criterion is met for *screening from stance-phase joint-angle means*.

Two caveats that stop this being final, both stated in advance:

- **The 28-channel stratum is underpowered.** ~191 uninjured sessions gives CI
  widths near ±0.11; it cannot resolve a +0.04 effect. A null there is ambiguous.
- **Session-mean curves discard within-session variability.** Per-step curves
  were written (1.0 GB) precisely because stride-to-stride distributions are a
  different signal, and phase 5 targets them. That has not been tested.

## Anything that surprised us, or looks wrong

1. **The sagittal plane is index 2, contradicting the schema doc.** Caught only
   because the check was pre-registered. This is the single most dangerous thing
   found in the phase.
2. **Waveforms leak far less provenance than the stored summaries** (+0.091 vs
   +0.182). Not obvious in advance — the same markers produced both.
3. **Two sessions filed as running are walks** by the pipeline's own classifier.
4. **Half the `dv_r` columns are dead by design** — `gait_steps.m` preallocates
   `zeros(77,3)` and assigns exactly 40 rows, matching the 40 live variables
   observed per side. Re-running does not populate them.
5. **`wave9` underperforms both `wave54` and `wave3`** — the middle rung of the
   ladder is the weakest, which is a PCA-budget artifact, not physiology.
6. **HGB loses to logistic regression on every waveform set**, reversing the
   pattern in phase 1A where the tree exploited the confound best. Consistent
   with weak signal that regularization handles better than a tree chasing noise.

## Reproduction

```
.venv/Scripts/python.exe scripts/vendor_matlab_code.py
matlab -batch "addpath('scripts/matlab'); batch_waveforms(<code>,<list>,<out>,<manifest>)"
.venv/Scripts/python.exe scripts/phase1c_assemble.py
.venv/Scripts/python.exe scripts/phase1c_verify_sagittal.py   # gate: must PASS
.venv/Scripts/python.exe scripts/phase1c_benchmark.py
.venv/Scripts/python.exe scripts/phase1c_waveforms.py
```

Seeds fixed throughout. Per-fold AUCs in `results/phase1c_benchmark.json` and
`results/phase1c_waveforms.json`.

## Gate status

| # | plan item | status |
|---|---|---|
| 1 | `CONTROL_CLEAN` defined once, imported | ✅ `phase1_cohort.py` |
| 2 | computed `DISCRETE_VARIABLES` match stored `dv_r` | ✅ **1,745/1,745** |
| 3 | `waveforms_mean.npy` is `(n, 54, 101)`, reconciles | ✅ 1,745, 0 skipped |
| 4 | no group leakage, incl. LOWO | ✅ asserted every fold |
| 5 | sagittal subset verified against `dv_r` | ✅ **plane 2**, |r| ≥ 0.99 |
| 6 | no `data/` paths tracked by git | ✅ |
