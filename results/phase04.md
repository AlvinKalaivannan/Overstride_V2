# Phase 4 — the limb signal under *measured* monocular error

**Real monocular error costs about 0.03 AUC, which is small. The honest
video-recoverable feature set still lands below the pre-registered bar, in every
condition — including the favourable side-on geometry and including at 30 fps.
Error is not what is limiting it.**

**`wave2` — hip + knee, the honest video-recoverable set:**

| bank | fps | realized error | AUC | spread | 95% CI | CI > 0.5 | mean ≥ 0.60 | Δ vs clean |
|---|---|---|---|---|---|---|---|---|
| **clean mocap** | — | 0° | **0.615** | — | [0.533, 0.693] | ✅ | ✅ | — |
| all views | mocap | 3.29° | 0.586 | 0.032 | [0.535, 0.636] | ✅ | ❌ | −0.029 |
| all views | 60 fps | 2.94° | 0.588 | 0.033 | [0.538, 0.635] | ✅ | ❌ | −0.028 |
| all views | 30 fps | 2.86° | 0.587 | 0.023 | [0.535, 0.631] | ✅ | ❌ | −0.029 |
| **side-on only** | mocap | 2.24° | 0.587 | 0.018 | [0.517, 0.642] | ✅ | ❌ | −0.028 |
| **side-on only** | 60 fps | 2.04° | 0.588 | 0.014 | [0.518, 0.640] | ✅ | ❌ | −0.027 |
| **side-on only** | **30 fps** | **1.94°** | **0.583** | 0.023 | [0.515, 0.631] | ✅ | ❌ | −0.033 |

**`wave3_hk` — the optimistic control (clean ankle a camera cannot deliver):**

| bank | fps | realized error | AUC | 95% CI | mean ≥ 0.60 | Δ vs clean |
|---|---|---|---|---|---|---|
| clean mocap | — | 0° | 0.610 | [0.532, 0.686] | ✅ | — |
| all views | mocap / 60 / 30 | 2.19 / 2.00 / 2.08° | 0.608 / 0.610 / 0.607 | — | ✅ | −0.002 / +0.000 / −0.002 |
| side-on only | mocap / 60 / 30 | 1.49 / 1.40 / 1.47° | 0.599 / 0.599 / 0.595 | — | ❌ | −0.011 / −0.010 / −0.014 |

The last row matters: **under the realistic side-on bank, even the optimistic
feature set falls below the bar.** `wave3_hk` only cleared 0.60 when it was fed
the less accurate all-views error, because a clean ankle channel diluting noisier
hip/knee channels is worth more when those channels are noisier.

**No ΔAUC excludes zero.** Every delta CI spans 0, so the degradation is real in
direction and consistent across conditions, but it is not individually
significant against fold noise. Saying "measured monocular error costs ~0.03 AUC"
is defensible; saying "it significantly degrades the signal" is not.

**The striking thing is how flat this is.** `wave2` lands at 0.583–0.588 in all
six conditions — across a 1.7× range of realized error and an 11× range of
temporal resolution. Reducing error from 3.29° to 1.94° by restricting to side-on
views buys nothing. **Error is not the binding constraint. The signal is simply
weak**, and phase 4B measures what that means in practice.

## Negative controls, re-asserted rather than inherited

CLAUDE.md requires a demographics-only control and a provenance-only baseline
alongside every kinematic score. On this within-subject task all three are
constant within a session and so *must* sit at chance; a departure would mean the
setup leaks and every number above is void. Asserted in-script (`assert worst <
0.08`), not merely printed:

| control | AUC | 95% CI |
|---|---|---|
| demographics (`CONTROL_CLEAN`) | 0.486 | [0.386, 0.572] |
| provenance-only | 0.449 | [0.403, 0.503] |
| file structure | 0.472 | [0.415, 0.537] |

**Max \|AUC − 0.5\| = 0.051 → PASS.** Run speed is inside the demographics
control and constant within a session, so it cannot drive limb choice; that is a
stronger guarantee than the speed-matched subsample CLAUDE.md asks for on
between-subject analyses.

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

**Drawn as clip pairs, not independently.** Each source clip contributes exactly
one near row and one far row. They are drawn together, so the common-mode error a
single camera puts on both limbs survives into the left–right difference —
because that is exactly what cancels. Independent draws destroy it and overstate
the damage. One clip is drawn per Ferber session and shared across both limbs and
both joints, so the hip/knee error correlation survives too.

**Two banks.** `all views` uses all 592 paired clips (3.32° mean error).
`side-on only` restricts to the 296 clips with view ratio ≥ 0.85 — the geometry
Overstride actually prescribes, and, per phase 3B, the geometry where the pose
estimator is most accurate (2.26° mean error).

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

Realized error falls slightly as framerate drops (reconstruction between sparse
samples smooths the injected residual), while AUC barely moves:

| fps | realized error | wave2 AUC (all views) | wave2 AUC (side-on) |
|---|---|---|---|
| mocap (101 pts) | 3.29° / 2.24° | 0.586 | 0.587 |
| 60 fps (18 pts) | 2.94° / 2.04° | 0.588 | 0.588 |
| 30 fps (9 pts) | 2.86° / 1.94° | 0.587 | 0.583 |

Differences are inside the realization spread and every CI. **At 2–3° of error,
framerate is not the binding constraint** — consistent with phase 2's
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

1. **3.32° is a lower bound.** AthleticsPose does not release the original
   videos, so this is 2D→3D lifting only — video decode and person detection
   error are excluded. Per-clip denormalisation uses a scale derived from
   ground-truth 3D, which a deployed system does not have. And the fine-tuned
   checkpoint was trained on this capture rig, these cameras, this sport:
   held-out *subjects* is not held-out *domain*.

2. > **~~The side-on occlusion penalty is still unmeasured.~~ WITHDRAWN.** The
   > original version of this report called this the project's largest open gap,
   > on phase 3's claim that AthleticsPose contains only near-frontal views.
   > **That claim was a unit error** — 44 *pixels* compared against a 200–250
   > *mm* reference. Measured unit-free, half the clips are near-lateral and the
   > far-limb penalty does not grow with view angle (`results/phase03b.md`). The
   > `side-on only` bank above is the direct test, and it changes nothing.

3. **The angle-convention transfer is an assumption, now bounded rather than
   assumed away.** `phase3_angles.py` flagged that keypoint three-point angles
   and Ferber Cardan angles are not the same quantity, and that phase 4 must
   handle it. It cannot be tested directly — no subject has both — so phase 4B
   bounds it instead: the verdict holds unless the magnitude transfer is off by
   more than ~2×, and only reaches chance at ~3×. See `results/phase04b.md`.

4. **Nothing reaches a full 90° lateral view** (max view ratio ~0.97), so the
   side-on bank is near-lateral rather than perfectly lateral.

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
| 3 | honest set separated from optimistic one | ✅ wave2 vs wave3_hk, gap quantified |
| 4 | negative controls re-asserted | ✅ max \|AUC−0.5\| = 0.051, asserted in-script |
| 5 | occlusion penalty under real error | ✅ **resolved** — side-on bank run; phase 3B shows the penalty is ~0 and flat in view angle |
| 6 | verdict against the pre-registered bar | ❌ **fails the mean ≥ 0.60 leg in every condition** |

**Item 6 is the headline, and item 5 no longer softens it.** The signal survives
contact with real monocular error — every CI excludes chance, at every framerate,
under both banks. But `wave2` sits at **0.583–0.588** against a bar of 0.60, and
the occlusion escape hatch is gone: the geometry is favourable, the error is
small, and the score still does not reach the bar.

**Error is not what is limiting this.** Halving the realized error changes
nothing. What limits it is the strength of the underlying limb-asymmetry signal,
which phase 5 measured at 0.610 on *perfect mocap*. Phase 4B measures what a
score in this range means for an actual user, and that is the result phase 6
has to be designed around.

## Reproduction

```
.venv/Scripts/python.exe scripts/phase4_noise_bank.py    # ~25 min CPU
.venv/Scripts/python.exe scripts/phase4_real_delta.py    # ~10 min CPU
.venv/Scripts/python.exe scripts/phase4_figure.py
```

Per-fold AUCs for all 20 scored models in `results/phase4_real_delta.json`.
