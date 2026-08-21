# Phase 5C — three attempts to raise the injured-limb signal

**None of them worked.** The best result in the project is now 0.618, up +0.008
on the phase 5 reference, with a delta CI spanning [−0.110, +0.089]. Velocity
asymmetry is at chance and actively harmful. No injury condition survives its own
confidence interval. And the one candidate that nominally improved AUC does not
improve repeatability, which was the criterion that mattered.

| attempt | AUC | 95% CI | Δ vs reference | Δ CI | excludes 0 |
|---|---|---|---|---|---|
| `limbsag_mean` (phase 5 reference) | 0.610 | [0.532, 0.686] | — | — | — |
| **R1** `limbvel` (24 velocity channels) | **0.499** | [0.431, 0.579] | **−0.111** | [−0.214, −0.028] | **YES (worse)** |
| **R1** `limbsag` + `limbvel` | 0.556 | [0.472, 0.615] | −0.054 | [−0.116, +0.005] | no |
| **R3a** stride distributions | 0.617 | [0.513, 0.688] | +0.007 | [−0.118, +0.084] | no |
| **R3a** `limbsag` + stride distributions | **0.618** | [0.522, 0.686] | +0.008 | [−0.110, +0.089] | no |

Decision rule is the project's established one: the paired delta CI across the
shared 25 folds excluding zero. **All four sets tried are reported**, not the
best of them.

---

## Scope was set by phase 5B, not by ambition

`results/phase05b.md` pre-registered that a flat severity dose-response would
scope this stage "down to the cheap items only". It came out flat (ρ = +0.044,
CI spanning zero) and anti-monotone by stratum, and the `dv_r` probe landed on
0.610 — identical to the sagittal model from a completely different measurement
modality.

**So the expensive stride-pair augmentation (R3b in the plan) was not run.** What
was run is everything cheap, plus the one genuinely different information axis
neither 5B probe covers: within-session stride *distributions*. Both 5B probes
were session-level aggregates; this is not.

## The split

Identical to phases 4, 5 and 5B: 818 unilateral-injury sessions / 675 subjects,
chance 0.517, `StratifiedGroupKFold(n_splits=5, shuffle=True)`,
**`groups=sub_id`**, seeds `[11, 23, 37, 53, 71]` → 25 folds shared across every
row so all deltas are paired. `check_no_group_leakage` asserted.

**Negative controls re-asserted in-script**: demographics 0.486, provenance
0.449, structure 0.472, max |AUC − 0.5| = 0.051 → PASS.

---

## R1 — velocity asymmetry (0.499, and it makes things worse)

`waveforms_mean.npy` carries 24 velocity channels — `vel_{L,R}_{ankle, knee,
hip, pelvis}_p{0,1,2}` — that `scripts/phase5_limb.py` never read; it touches
only `ang_*`. Angular velocity is where an antalgic gait often shows first, so
this was the cheapest plausible win in the project.

Signs derived from population `corr(L, R)`, not assumed: **4 shared, 8 mirrored**
(ankle/knee/hip/pelvis planes 0 and 1), which matches the angle convention and
anatomy — so the sign derivation is behaving.

**AUC 0.499. Exactly chance.** And the delta is one of the only ones in this
project whose CI excludes zero — in the *wrong* direction (−0.111
[−0.214, −0.028]). Adding velocities to the sagittal set drags it from 0.610 down
to 0.556.

Velocity asymmetry carries no limb information here and dilutes what the angles
have. Reported because it was tried, not because it is interesting.

## R3a — within-session stride distributions (0.617, the numerical best)

Streamed from `data/derived/waveforms_steps/*.mat` one file at a time
(CLAUDE.md forbids loading all 1,745 at once). **L and R stride counts differ** —
median 29 each but equal in only 25% of sessions — so the limbs are compared as
distributions over strides rather than differenced pairwise.

Per channel (9 = 3 joints × 3 planes) and per stance block (5): differences in
location (mean, median), in spread (SD, IQR), and the Wasserstein distance
between the two stride sets. 225 features. Location and spread terms are signed
so they can indicate *which* limb; the Wasserstein term is unsigned magnitude.

**0.617 alone, 0.618 combined with the sagittal means — the highest numbers this
project has produced.** Both deltas include zero. This is a nominal improvement,
not a demonstrated one, and phase 5B's diagnostics say the honest reading is that
it is noise around the same ceiling.

## R2 — per condition (0 of 5 survive)

Pooling ITBS, PFPS, calf strain and 158 sessions labelled only "pain" may dilute
a condition-specific effect. Conditions with n ≥ 35, excluding the uninformative
`pain` (158) and `other` (138) labels. Case-normalised, since the raw labels
duplicate across capitalisation.

| condition | n | subjects | AUC | fold CI | subject-bootstrap CI | survives |
|---|---|---|---|---|---|---|
| ITB syndrome | 100 | 78 | 0.658 | [0.410, 0.986] | [0.472, 0.774] | no |
| patellofemoral pain | 81 | 76 | 0.644 | [0.346, 0.828] | **[0.509, 0.772]** | no |
| Achilles tendonitis | 40 | 29 | 0.575 | [0.188, 0.900] | [0.389, 0.809] | no |
| plantar fasciitis | 35 | 30 | 0.472 | [0.100, 0.833] | [0.212, 0.684] | no |
| calf muscle strain | 43 | 31 | 0.330 | [0.062, 0.667] | [0.120, 0.489] | no |

**Patellofemoral pain is the only case where either interval excludes chance**,
and only one of the two — its subject-bootstrap CI starts at 0.509 while its fold
CI spans 0.5 comfortably. Requiring both is the conservative rule and it is the
right one at this n. **This is hypothesis-generating and nothing more.**

The calf-strain figure of 0.330 is not evidence of an inverted effect; at n = 43
with a bootstrap CI reaching 0.489 it is noise.

> ### A statistical error found and corrected
>
> The plan called for BH-corrected p-values across these families, and a Wilcoxon
> over the 25 fold AUCs was implemented first. It returned **q = 0.000 for ITB
> syndrome while that same condition's cross-fold CI ran [0.410, 0.986]**. A
> p-value and an interval cannot both be right. The test was wrong: the 25 folds
> are 5 seeds × 5 splits of *one* dataset, so they are heavily correlated, and
> treating them as 25 independent observations inflates significance enormously.
>
> Replaced with a **bootstrap that resamples subjects**, the actual independent
> unit. Those intervals are tighter than the fold intervals and agree with them.
> Had the original been published, this phase would have reported three
> "significant" conditions where there are none.

## The check that actually mattered

An AUC gain that does not improve **repeatability** does not address the question
that motivated this work — phase 4B found the model changes its mind about the
same runner on a quarter to a third of repeat scans.

258 same-subject session pairs from 72 subjects, both scored here on the same
feature definitions so the comparison is like-for-like:

| | binary agreement | score r | calibration slope | Brier |
|---|---|---|---|---|
| `limbsag_mean` (reference) | 0.725 | +0.529 | 0.739 | 0.2408 |
| `limbsag` + stride distributions | **0.736** | +0.533 | 0.715 | 0.2428 |

**+0.011 agreement, and the calibration slope is slightly worse.** The best
feature set found in this phase does not make the model meaningfully more
consistent about the same runner.

(Phase 4B quoted 0.744 for its clean-mocap reference; that was `wave2` — hip and
knee only. The 0.725 here is `limbsag`, which adds the ankle. Different feature
set, so the two are not directly comparable; the within-table comparison is.)

---

## Answer to the question this phase was asked

**"What can be done to address the limb being at risk?"** — On this dataset,
nothing that has been tried moves it. Across phases 5, 5B and 5C the task has
been attacked with:

- sagittal / 9-channel / 15-channel angle differences (0.598–0.610)
- stride-quartile and SD summaries (0.593–0.603)
- 24 velocity channels (0.499)
- 39 clinical frontal/transverse metrics from a different modality (0.610)
- within-session stride distributions (0.617)
- five injury conditions separately (0 survive)
- 4× more training data via the learning curve (saturated at +0.002)

Everything lands in a narrow band around 0.61, and the two probes designed to
detect headroom — a different modality, and more data — both say there is none.

**The honest conclusion is that unilateral running injury, in this cohort, does
not produce a limb asymmetry in stance-phase kinematics large enough to identify
the injured side reliably in an individual.** That is a real finding about the
physiology and the data, not a modelling failure.

## What remains

The only avenue not yet taken is the one deliberately deferred: **accept the
ceiling and engineer around it** — selective classification (answer only the most
confident N% of scans, and measure coverage against accuracy), and multi-session
averaging for runners who are scanned more than once. Neither raises the signal;
both change what is done with it. `results/phase04b.md` sets out the three
framings for phase 6 that these numbers can support.

## Reproduction

```
.venv/Scripts/python.exe scripts/phase5b_ceiling.py     # ~15 min CPU
.venv/Scripts/python.exe scripts/phase5c_features.py    # ~25 min CPU
```

Outputs: `results/phase5b_ceiling.json`, `results/phase5c_features.json`,
including per-fold AUCs for every model scored.
