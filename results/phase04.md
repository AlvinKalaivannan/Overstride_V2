# Phase 4 — the limb signal under *measured* monocular error

**Real monocular error costs about 0.03 AUC, which is small. It is also enough
to put the honest video-recoverable feature set below the pre-registered bar.
The verdict is two-sided and both sides matter.**

| feature set | realized error | AUC | spread | 95% CI | CI > 0.5 | mean ≥ 0.60 | Δ vs clean |
|---|---|---|---|---|---|---|---|
| **wave2 clean mocap** | 0° | **0.615** | — | [0.533, 0.693] | ✅ | ✅ | — |
| **wave2 + real error, mocap** | 3.31° | **0.588** | 0.020 | [0.547, 0.622] | ✅ | ❌ | −0.027 |
| **wave2 + real error, 60 fps** | 2.96° | **0.587** | 0.022 | [0.544, 0.626] | ✅ | ❌ | −0.028 |
| **wave2 + real error, 30 fps** | 2.87° | **0.582** | 0.026 | [0.540, 0.632] | ✅ | ❌ | −0.033 |
| wave3 clean mocap *(reference)* | 0° | 0.610 | — | [0.532, 0.686] | ✅ | ✅ | — |
| wave3_hk + real error, mocap | 2.20° | 0.606 | 0.011 | [0.546, 0.667] | ✅ | ✅ | −0.003 |
| wave3_hk + real error, 60 fps | 2.01° | 0.604 | 0.012 | [0.545, 0.668] | ✅ | ✅ | −0.006 |
| wave3_hk + real error, 30 fps | 2.09° | 0.602 | 0.017 | [0.540, 0.663] | ✅ | ✅ | −0.008 |

**`wave2` is the result. `wave3_hk` is the optimistic control** and is reported
only so the gap between them is visible — see "why wave3_hk is not the answer".

**No ΔAUC excludes zero.** Every delta CI spans 0, so the degradation is real in
direction and consistent across conditions, but it is not individually
significant against fold noise. Saying "measured monocular error costs 0.03 AUC"
is defensible; saying "it significantly degrades the signal" is not.

![degradation curve](../figures/phase4_degradation.png)

---

## What this phase is, and what it is not

`CLAUDE.md` phase 4 reads *"video-derived features through the phase 2 model"*.
Taken literally that is impossible with these datasets: the Ferber archive is
marker-based mocap with **no video**, AthleticsPose has video-derived poses and
**no injury labels**, and **no subject appears in both**. There is no join.

What was done instead, and why it is a real advance on phase 2: phase 2 degraded
the Ferber waveforms with *assumed* Gaussian noise. Phase 3 *measured* what
monocular pose actually does. Phase 4 injects those measured residual curves —
real magnitude, real temporal structure, real systematic bias — and reports the
resulting ΔAUC.

**The noise is video-derived and empirical rather than invented. This is not an
end-to-end video pipeline, and nothing here should be read as one.** No frame of
video was decoded, no person was detected, and no Ferber subject was filmed.

## The error bank

`scripts/phase4_noise_bank.py` → `data/derived/phase4_error_bank.npz` (not
committed; regenerable).

592 held-out AthleticsPose clips (subjects S11 / S13 / S16 only) →
**1,184 residual curves**, shape `(1184, 2, 101)`, channels `[hip, knee]`,
each tagged `near` or `far`.

| channel | MAE | bias | within-curve SD |
|---|---|---|---|
| hip | 2.85° | +0.11° | 3.08° |
| knee | 3.79° | +0.73° | 4.03° |
| near limb (n=592) | 3.38° | — | — |
| far limb (n=592) | 3.26° | — | — |

Overall mean \|error\| **3.32°**. Only the sport-fine-tuned checkpoint was run:
it is the operating point worth propagating, and running all three would triple
the cost for no decision value.

**Residual *curves*, not a σ.** A scalar σ discards the two things the real error
actually has — temporal structure (the residual drifts across stance rather than
jittering independently) and a systematic offset. Both matter here, because the
feature is a left–right difference: error common to both limbs cancels, error
specific to one limb does not.

## The split

Identical to phases 2 and 5, deliberately, so every comparison is paired:
`StratifiedGroupKFold(n_splits=5, shuffle=True)`, **`groups=sub_id`**, seeds
`[11, 23, 37, 53, 71]` → **25 folds**, generated once and reused by every row in
the table above. `check_no_group_leakage` asserted and passed on every fold. All
preprocessing inside the `Pipeline`; PCA capacity matched at 3 components per
channel. Model fixed in advance to logistic regression.

## n after filtering

**818 unilateral-injury sessions / 675 subjects.** Injured sessions with
`InjSide ∈ {Left, Right}` only; 300 bilateral sessions excluded (no side to
identify); uninjured subjects excluded by design, since they have no injured side
and including them would make side-alignment itself encode the label. Chance =
**0.517**.

Far/near limb assignment is a fixed random draw independent of injured side
(seed 20260810), inherited from phase 2 where its independence was asserted at
max deviation 0.010.

## Each realization is averaged fold-by-fold

Three seeded realizations per condition. Because all three are scored on the same
25 folds, they are averaged **per fold**, giving 25 realization-averaged fold
scores; the CI and the paired delta both come from those. So the AUC, the CI and
the ΔAUC in any one row describe the same quantity and reconcile
(0.588 − 0.615 = −0.027). `auc_spread` is kept as the across-realization range so
sampling variability stays visible.

> An earlier version of this script reported a mean AUC beside a *single*
> realization's CI and delta, which did not reconcile. Fixed before publication.

---

## Three things that were not expected

### 1. The missing ankle costs nothing — the missing *toe keypoint* is not the problem

H36M-17 has no toe keypoint, so monocular pose cannot deliver ankle
dorsiflexion, and `wave3` (hip/knee/ankle) is unavailable. That looked like a
structural loss. It is not:

| clean mocap | AUC | CI |
|---|---|---|
| wave3 (hip + knee + ankle) | 0.610 | [0.532, 0.686] |
| **wave2 (hip + knee only)** | **0.615** | [0.533, 0.693] |

Dropping the ankle entirely is **not worse** — slightly better, well within
noise. The ankle channel carries no independent limb-asymmetry information here.
The ~0.03 that phase 4 loses is the injected error, not the missing joint.

### 2. Why `wave3_hk` is not the answer, quantified

`wave3_hk` keeps the ankle **clean** while perturbing hip and knee — it credits
the model with a channel a camera cannot deliver. Its realized error is
**2.0–2.2°** against wave2's 2.9–3.3°, and that gap is arithmetic: one third of
its channels have exactly zero error, dragging the mean down. Its higher AUC is
purchased by a clean channel diluting the perturbed ones.

That is why it clears the bar and wave2 does not, and why **wave2 is reported as
the result**. `wave3_hk` is in the table as the size of the optimism, not as a
finding.

### 3. Framerate does almost nothing at this error level — after a bug was fixed

Realized error and AUC both fall slightly as framerate drops:

| fps | realized error | wave2 AUC |
|---|---|---|
| mocap (101 pts) | 3.31° | 0.588 |
| 60 fps (18 pts) | 2.96° | 0.587 |
| 30 fps (9 pts) | 2.87° | 0.582 |

Monotone, tiny, and inside both the realization spread and every CI. **At 3.3° of
error, framerate is not the binding constraint** — consistent with phase 2's
finding that 30 fps alone costs nothing, and with phase 2's interaction only
biting at much higher error.

> **A defect found and fixed mid-phase.** The first version added the residual to
> the 101-point curve and decimated *afterwards*, which let decimation low-pass
> away error a real 30 fps camera would actually deliver. It produced the
> suspicious result that **AUC rose as framerate fell** (0.585 → 0.596). Rewritten
> to run in acquisition order — camera samples k frames, the estimator errs *at
> those frames*, then the curve is reconstructed — the artefact disappears and the
> trend is monotone downward. The suspicious number was the bug.

---

## A correction to how phase 2 and phase 3 were read

Placing phase 4 on the phase 2 surface required putting both on the same x-axis.
Phase 2 is parameterised by a Gaussian **σ**; phase 3 and 4 report **mean
absolute error**. Those are not the same number, and the analytic conversion
`E|N(0,σ)| = σ√(2/π)` is *also* wrong here, because phase 2 does not add noise
and stop — it decimates, adds noise, **low-pass filters at 10 Hz**, then
re-interpolates.

Re-running phase 2's own `degrade()` and measuring `mean |degraded − clean|`:

| phase 2 condition | nominal σ | **realized error** |
|---|---|---|
| 101 pts, σ 2 | 2° | 0.64° |
| 101 pts, σ 8 | 8° | 1.69° |
| 101 pts, σ 15 | 15° | **3.08°** |
| 30 fps, σ 8 | 8° | 4.95° |
| 30 fps, σ 15 | 15° | 9.18° |
| 30 fps, no noise | — | 0.24° |

(The 0.24° decimation-only figure matches phase 2's own verification table, which
measured 0.263° at k=9 — so the calibration is sound.)

**At full resolution the 10 Hz filter removes roughly 80% of the injected
noise.** With fs ≈ 342 Hz at 101 points, a 10 Hz filter has many samples to
average over; at 9 points it has almost none. That is physically real — a genuine
pipeline does filter, and filtering at a high sample rate genuinely does suppress
pose noise better. It is also a large part of the framerate × error *interaction*
that phase 2 identified, which is worth knowing as a mechanism.

But it means two earlier statements were read too generously:

1. **`results/phase02.md` says the signal "absorbs 15° of symmetric error" at
   full resolution.** The error actually reaching the classifier in that
   condition was **3.08°**, not 15°. Phase 2's *conditions* are correctly
   described; its σ labels overstate the residual error by ~4×.
2. **`results/phase03.md` concluded the fine-tuned 3.4° sits "below every tested
   noise level".** On realized error it does not — 3.4° is at or above most of
   phase 2's full-resolution conditions, and comparable to 30 fps σ 8 (4.95°,
   where AUC fell to 0.529). That conclusion was too optimistic.

**This is exactly why phase 4 was run rather than interpolated, and phase 4's
direct measurement supersedes both readings.** The measured answer — 0.582–0.588
— is better than the pessimistic interpolation off phase 2's 30 fps curve
(0.565 at 2.87°) and worse than phase 3's optimistic placement implied.
Interpolation was not a substitute for measurement in either direction.

Realized errors are cached in `results/phase2_realized_error.json`.

---

## What still bounds this result

Carried from phase 3, unresolved here:

1. **3.32° is a lower bound.** AthleticsPose does not release the original
   videos, so this is 2D→3D lifting only — video decode and person detection
   error are excluded. Per-clip denormalisation uses a scale derived from
   ground-truth 3D, which a deployed system does not have. And the fine-tuned
   checkpoint was trained on this capture rig, these cameras, this sport:
   held-out *subjects* is not held-out *domain*.
2. **The side-on occlusion penalty is still unmeasured, and phase 4 does not
   close it.** In AthleticsPose the hips separate in depth by a median of 44 mm
   (max 64 mm) against ~200–250 mm for a true lateral view — these are
   near-frontal captures. The near/far residuals in the bank therefore carry
   almost no occlusion asymmetry (3.38° near vs 3.26° far). **Phase 2 found that
   occlusion asymmetry combined with low framerate is the lethal combination**,
   and that cell remains untested with real error. A dataset with genuinely
   lateral views is required.
3. **The residuals are drawn independently per limb.** Real bilateral error from
   one camera would share a common component that partly cancels in the R − L
   difference. Independent draws are the conservative choice, but they are a
   choice.

## Environment deviation, recorded

pandas 3.0.5 stopped importing on this machine — Windows Smart App Control
(`VerifiedAndReputablePolicyState = 1`) began blocking
`pandas/_libs/tslibs/tzconversion.cp311-win_amd64.pyd`. Pinned to
`pandas>=2.2,<3`. Verified immaterial to results: **every per-realization AUC
under 2.3.3 was bit-identical to the run under 3.0.5.** No result in this
repository depends on the pandas version.

## Gate status

| # | item | status |
|---|---|---|
| 1 | measured (not assumed) error injected | ✅ 1,184 empirical residual curves, 3.32° |
| 2 | real ΔAUC reported | ✅ −0.027 to −0.033 for wave2; no delta CI excludes zero |
| 3 | honest video-recoverable set separated from optimistic one | ✅ wave2 vs wave3_hk, gap quantified |
| 4 | verdict against the pre-registered bar | ⚠️ **split** — CI leg passes, mean ≥ 0.60 leg fails |
| 5 | occlusion penalty under real error | ❌ **not measured** — no side-on views exist in the source data |

**Item 4 is the headline and item 5 is the reason not to round it up.** The video
path is not closed — the signal clearly survives contact with real monocular
error, at all three framerates, with every CI excluding chance. But at 0.582–0.588
the honest feature set sits below the bar this project fixed in advance, and the
one condition phase 2 identified as fatal has still never been tested with real
error.

## Reproduction

```
.venv/Scripts/python.exe scripts/phase4_noise_bank.py    # ~25 min CPU
.venv/Scripts/python.exe scripts/phase4_real_delta.py    # ~10 min CPU
.venv/Scripts/python.exe scripts/phase4_figure.py
```

Per-fold AUCs for all 20 scored models in `results/phase4_real_delta.json`.
