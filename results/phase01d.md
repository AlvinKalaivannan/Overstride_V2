# Phase 1D — per-condition labels, variability, asymmetry

**Answer to "is there a way to build the injury signal": on this evidence, no.**

**0 of 60 pre-registered tests produced a positive delta surviving
Benjamini–Hochberg.** Not one kinematic feature set beat a decontaminated
demographics control in any of the five conditions. The single positive delta
anywhere was stride-to-stride variability for ITB syndrome (+0.029, 17/25 folds,
CI [−0.128, +0.160]) — well short of the bar set in advance.

---

## A correction to the reasoning that motivated this phase

The plan argued per-condition comparisons would be better conditioned because
each diagnosis spans 7–8 collection years with uninjured sessions in the same
years. **That check was wrong, and the error is instructive.**

Overlap is not balance. Both classes appearing in the same years says nothing
about whether their *ratio* is stable, and it is not:

| condition | positive rate by year | overall |
|---|---|---|
| ITB syndrome | 0.00 → **0.74** | 0.18 |
| PFPS | 0.00 → **0.76** | 0.19 |

Knowing the collection year is therefore highly informative about the class.
Measured consequence — provenance-only AUC, fitted on year and two
missingness flags, nothing physiological:

| task | provenance-only AUC |
|---|---|
| pooled injured vs uninjured (phase 1C) | 0.775 |
| **PFPS vs uninjured** | **0.863** |
| **ITB syndrome vs uninjured** | **0.831** |
| Achilles tendonitis | 0.760 |
| calf muscle strain | 0.747 |
| plantar fasciitis | 0.707 |

**Per-condition labelling made the confound worse, not weaker.** The premise of
this phase was mistaken and the data says so plainly.

---

## The split

Unchanged from phase 1, as committed: `StratifiedGroupKFold(n_splits=5,
shuffle=True)`, **`groups=sub_id`**, seeds `[11, 23, 37, 53, 71]` → **25 folds**,
generated once per condition and reused by every feature set so all deltas are
paired fold-by-fold. `check_no_group_leakage` asserted and passed on every fold
of every condition. All preprocessing — impute, scale, PCA(30) — inside an
sklearn `Pipeline`, fit on the training fold only; logistic `C` by nested inner
`StratifiedGroupKFold(3)`.

## n per condition

Positives = that diagnosis only. Negatives = the uninjured pool. **Sessions with
a different diagnosis are excluded, not used as negatives** — a runner with
another injury is not a healthy control.

| condition | sessions | subjects | positive rate |
|---|---|---|---|
| patellofemoral pain syndrome | 129 | 124 | 0.185 |
| itb syndrome | 124 | 99 | 0.179 |
| achilles tendonitis | 61 | 47 | 0.097 |
| calf muscle strain | 54 | 41 | 0.087 |
| plantar fasciitis | 50 | 45 | 0.081 |
| **uninjured pool** | **570** | **359** | — |

Counts are marginally below the pre-run figures (PFPS 134→129, Achilles 62→61,
uninjured 576→570) because **24 sessions were dropped for fewer than 8 retained
strides** — an SD over <8 strides is noise, not a variability estimate. All 24
are named in the `phase1d_features.py` output. Median retained strides is 28.

---

## Results

Best model per feature set (logit or HGB, whichever scored higher).

| features | PFPS | ITBS | Achilles | plantar f. | calf |
|---|---|---|---|---|---|
| provenance-only | **0.863** | **0.831** | **0.760** | 0.707 | 0.747 |
| CONTROL_CLEAN + structure | 0.840 | 0.780 | 0.730 | **0.788** | **0.822** |
| **CONTROL_CLEAN** | **0.764** | **0.607** | **0.662** | **0.777** | **0.723** |
| dv_r live | 0.753 | 0.560 | 0.597 | 0.681 | 0.526 |
| wave3 (sagittal) | 0.746 | 0.562 | 0.629 | 0.648 | 0.552 |
| sd54 (variability) | 0.746 | 0.636 | 0.639 | 0.627 | 0.649 |
| wave9 | 0.645 | 0.525 | 0.526 | 0.573 | 0.524 |
| asym9 (asymmetry) | 0.614 | 0.517 | 0.619 | 0.594 | 0.580 |
| wave9 + asym9 | 0.639 | 0.545 | 0.616 | 0.572 | 0.569 |

Paired ΔAUC vs `CONTROL_CLEAN` (folds won of 25):

| features | PFPS | ITBS | Achilles | plantar f. | calf |
|---|---|---|---|---|---|
| dv_r live | −0.012 (11) | −0.047 (8) | −0.065 (11) | −0.096 (2) | −0.197 (3) |
| wave3 (sagittal) | −0.019 (10) | −0.046 (8) | −0.034 (10) | −0.128 (2) | −0.171 (2) |
| **sd54 (variability)** | −0.019 (12) | **+0.029 (17)** | −0.023 (8) | −0.150 (0) | −0.074 (10) |
| wave9 | −0.119 (2) | −0.083 (3) | −0.137 (6) | −0.204 (0) | −0.199 (2) |
| asym9 (asymmetry) | −0.151 (1) | −0.090 (4) | −0.043 (10) | −0.183 (1) | −0.143 (5) |

**Benjamini–Hochberg over all 60 tests: 0 survive at 0.05 with a positive
delta.** Every test that reaches significance does so in the negative direction —
kinematics losing to the control.

> Caveat on the p-values, stated rather than buried: CV folds are not
> independent (subjects recur across the 5 seeds), so Wilcoxon p-values here are
> anti-conservative. The CI-excludes-zero bar and the fold-win counts are the
> primary evidence, and they agree with the BH verdict.

## Pre-registered hypotheses

Fixed before any per-condition model was fitted:

| condition | expected channels |
|---|---|
| itb syndrome | hip ab/adduction, knee plane 0 |
| patellofemoral pain syndrome | hip ab/adduction, knee flexion |
| achilles tendonitis | ankle flexion, ankle eversion |
| plantar fasciitis | ankle flexion, ankle eversion |
| calf muscle strain | ankle flexion |

**Not assessable.** The hypothesis check was conditional on a condition first
showing signal above its control, so that the question "does the signal sit
where the literature says it should" is meaningful. No condition cleared that
gate, so there is no signal whose location can be interrogated. Reported as a
miss rather than quietly dropped.

---

## What this means

**The two adequately powered conditions are both clearly null.** PFPS (124
subjects) and ITBS (99) were pre-identified as the only conditions able to carry
a conclusion, and neither shows any kinematic feature beating demographics. That
is the substantive result.

**The three small conditions are ambiguous, as stated in advance.** Achilles
(47 subjects), plantar fasciitis (45) and calf strain (41) have CIs spanning
0.25–0.35 AUC. Their nulls are uninformative, not evidence of absence.

**Demographics alone got stronger, not weaker.** `CONTROL_CLEAN` rises from 0.656
pooled to 0.764 (PFPS) and 0.777 (plantar fasciitis), because specific conditions
carry distinct demographic profiles. The bar the kinematics had to clear went up.

**Variability was the best of the new levers, and it was not enough.** `sd54` is
the only feature family that ever edged the control (ITBS, +0.029, 17/25 folds)
and the best kinematic set in 3 of 5 conditions. It is genuinely different
information from the means — worth noting for phase 5, which targets
within-session distributions directly — but it does not rescue phase 1.

**Asymmetry was the weakest.** `asym9` lost in every condition, badly in most.
The physiological motivation (unilateral injury → greater L/R asymmetry) is not
detectable here at session-mean resolution.

### Bearing on the project

`CLAUDE.md`'s kill criterion states: if the kinematic model cannot beat the
demographics control, the gait data carries no injury signal beyond the boring
explanation, and no downstream video pipeline can rescue it — report it and stop.

That criterion has now been tested against: the pooled label (phase 1C),
verified mocap waveforms rather than a proxy, three feature families, five
individual conditions, and two label definitions. **It fails every time.**

The degradation curve remains buildable — the waveforms exist and are verified —
but with no mocap-side signal to degrade, ΔAUC between full-mocap and
video-constrained features would measure the decay of noise.

## Anything that surprised us, or looks wrong

1. **The confound is stronger per-condition than pooled** (0.863 vs 0.775). The
   opposite of this phase's premise.
2. **Overlap ≠ balance.** The feasibility check that justified this phase tested
   the wrong property. Worth remembering as a general trap: for confound
   screening, check the class *ratio* per stratum, not class presence.
3. **Variability beats the mean curves in 3 of 5 conditions** despite being
   derived from the same per-step data. The means may be the wrong summary even
   though neither is sufficient here.
4. **PR-AUC stays near base rate everywhere** (e.g. PFPS 0.44 at a 0.185 rate for
   the control; kinematics 0.29–0.43). ROC-AUC flatters these models.

## Reproduction

```
.venv/Scripts/python.exe scripts/phase1d_features.py        # SD + asymmetry
.venv/Scripts/python.exe scripts/phase1d_percondition.py    # ~17 min
```

Seeds fixed. Per-fold AUCs and every one of the 60 tests in
`results/phase1d_percondition.json`.

## Verification

| # | item | status |
|---|---|---|
| 1 | per-condition cohorts, other-diagnosis excluded | ✅ counts above, 24 low-stride drops named |
| 2 | `check_no_group_leakage` on every fold | ✅ all conditions |
| 3 | SD/asym arrays reconcile to the index | ✅ 1,721 of 1,745, drops named |
| 4 | asymmetry never references `InjSide` | ✅ grep: docstring only, no code use |
| 5 | sagittal = plane 2 | ✅ `phase1c_verify_sagittal.py` PASS |
| 6 | no `data/` paths tracked | ✅ |
