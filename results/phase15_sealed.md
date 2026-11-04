# Phase 15 — the sealed test

> **PRE-REGISTRATION, written 2026-11-03.** Committed **alone**, and **pushed**,
> before a single sealed byte is downloaded. Everything below is fixed by this
> commit: the pipeline, the subjects, which quantity is primary, and the figures
> the sealed result will be compared against. Results are appended under
> "Results"; nothing above that heading is edited afterwards.

## What this is

Every MoVi number reported so far comes from subjects that are **spent** —
measured, inspected, decomposed, and in two cases used to diagnose defects in
the pipeline itself. Phases 15, 15B, 15C and 15D each found something that
changed how the measurement is computed or understood, and each was run on the
same handful of people.

**This is the one test on subjects that have never been opened.** It is run
once, with the pipeline frozen by this commit, and reported as it comes out.

## The frozen pipeline

**No code changes after this commit**, with one exception stated below. The
state being frozen is `HEAD = 10da6e0`, and the files that matter, pinned by
content:

| file | last modified in | blob (first 12) |
|---|---|---|
| `scripts/video_kinematics.py` | `27a9eb7` | `31524f7de685` |
| `scripts/phase15_movi.py` | `7a985d2` | `6a9d319f1cc5` |
| `scripts/phase15d_segments.py` | `10da6e0` | `254b2d1f1f85` |
| `scripts/phase3_angles.py` | `2716163` | `e1c6a6937125` |
| `scripts/phase15_checks.py` | `7a985d2` | `75617f64a781` |
| `scripts/phase7_inference_fix.py` | `86f0fea` | `10a879fec3c6` |

Checkpoint: `motionagformer-b-ath-det-coco-v1.ckpt`, 141,862,608 bytes,
sha256 `fd052ed1cb66347c1276df7455f5440b…`.

Pipeline settings, all already frozen: generic COCO detections at native
resolution on a padded person crop, `LOWPASS_HZ = 0.0`, COCO→H36M reordering,
`scores_to_train_scale` onto `TRAIN_CONF_MEAN = 0.8676`, resample to
`MODEL_FPS = 100.0`, lift with `pad="edge"` and `windows="overlap"`, angles from
`phase3_angles.sagittal_angles` and `phase15d_segments.segment_angles` with the
level-camera vertical `(0, −1, 0)`.

**The one exception.** A bug that makes a run **crash** or produce **impossible
values** (outside `video_kinematics.PLAUSIBLE`, or a reference marker read as a
sentinel) may be fixed. **Any such fix is reported with the before and after
numbers, never silently.** Four defects of exactly that kind were found in
phases 15–15D, so this clause is not hypothetical; what it forbids is changing
anything *because of how the sealed numbers look*.

## Subjects

**`Subject_16`, `Subject_17`, `Subject_18`, `Subject_19`, `Subject_2`** — the
sealed set as designated in `f549bfd`, before any of it was downloaded.
**PG1 only** (the side-on camera; `view_ratio` 0.67–0.72 on the diagnostic set).

Every subject must pass the **re-registered check 1** (`FEMUR_MM = (420, 700)`,
derived in `7a985d2` from the check's purpose rather than from any subject).
**Any failure is excluded before its video is decoded**, by
`phase15_movi.preregistered_checks`, per the frozen rule. Exclusions are
reported with the failing check named and the gap counts.

`Subject_14` is absent from this list because it has no round F video of any
kind.

## PRIMARY — the cross-lab claim

**Knee flexion and thigh inclination against gravity, near and far limb,
detected 2D only.**

These two are primary because phases 15C and 15D established that they are the
quantities that mean the same thing in two different labs:

- the **knee** is invariant to the thorax definition, structurally and
  empirically (24 of 24 rows bit-identical in 15C);
- **thigh inclination against gravity** does not depend on where the model puts
  its thorax, which 15C showed moves 87–144 mm between datasets.

Reported **per subject**: MAE, bias, SD, r. Then **pooled** with a
subject-bootstrap CI, **N_BOOT 2000, SEED 11**.

**Detected 2D only.** The projected-reference arm is diagnostic and is never
pooled into this.

## SECONDARY — the originally pre-registered quantities

Because phase 15's own pre-registration named them, and dropping them now
because a later phase found them less transferable would be choosing what to
report after seeing why it looks bad:

- the **frozen trunk-referenced hip** angle,
- the **combined hip + knee mean** — the form the phase 15 headline takes,
- **trunk lean**,
- and **per subject, the identity**

  > **hip bias = thigh bias + trunk-lean bias**

  which held to ~10⁻¹⁵ on all twelve 15D rows and is asserted in
  `tests/test_segment_angles.py`. Showing it per sealed subject is a check that
  the sealed run is internally consistent, independent of whether the numbers
  are good.

## DIAGNOSTIC — never pooled into the primary

The **`--perfect-2d`** arm: the reference 3D projected through PG1's own
calibration and lifted. Labelled diagnostic throughout. It exists to separate
the detector's contribution from the lifter's, and it is **not** part of the
cross-lab claim because a deployable clip cannot produce it.

## The comparison, with every figure fixed now

So that it is visible whether the sealed set agrees with what the method was
developed on, the sealed figures are reported against the **diagnostic set**
(`Subject_1`, `Subject_10`, `Subject_12`, `Subject_13`), detected 2D, PG1,
pooled with the same bootstrap. **These values are computed and written down in
this commit and will not be recomputed:**

| quantity | limb | MAE | 95% CI | bias |
|---|---|---|---|---|
| **knee** | near | **9.269°** | [7.556, 11.326] | +8.143 |
| **knee** | far | **9.725°** | [7.938, 11.889] | +8.157 |
| **thigh inclination** | near | **6.282°** | [5.270, 7.128] | −5.473 |
| **thigh inclination** | far | **6.487°** | [5.387, 7.844] | −5.601 |
| trunk lean | near | 7.174° | [5.634, 8.095] | −5.352 |
| trunk lean | far | 7.169° | [5.632, 8.086] | −5.349 |
| hip (trunk-referenced) | near | 12.270° | [9.732, 13.926] | −10.825 |
| hip (trunk-referenced) | far | 11.874° | [9.002, 14.423] | −10.951 |
| combined hip+knee mean | near | 10.769° | [9.330, 12.570] | — |
| combined hip+knee mean | far | 10.799° | [9.964, 11.787] | — |

Per-subject diagnostic-set MAE, detected, near limb, for the record:

| subject | thigh | knee | hip (trunk-ref) |
|---|---|---|---|
| Subject_1 | 6.508 | 6.950 | 13.706 |
| Subject_10 | 4.857 | 9.376 | 8.408 |
| Subject_12 | 7.374 | 12.278 | 14.145 |
| Subject_13 | 6.390 | 8.472 | 12.819 |

Also quoted for context, not as a head-to-head: the AthleticsPose **tune-set**
figures — knee 4.071° near, trunk-referenced hip 2.834° near, thigh inclination
6.694° — and OpenCap Monocular's 4.8° on walking, squats and sit-to-stand,
which is a different task on a different movement set.

## No gates

**There is no pass/fail threshold on any sealed number.** `CLAUDE.md`: *"There
is no pass/fail gate. A loss on held-out data is a result and goes in the README
exactly as prominently as a win."*

The numbers are reported **as they come out**, and the README is updated with
them whether they are better or worse than the diagnostic set. **Nothing is
tuned, re-run with different settings, or re-scored against a different
reference because of how they look.**

## What would make this test worthless

- **Running it more than once.** It is run once. If the frozen pipeline crashes,
  the fix and its before/after are reported, and that is the only re-run.
- **Changing the reference, the thorax definition, or the camera after seeing a
  number.** The reference is `MOVI_TO_H36M` as frozen; the thorax stays
  Visual3D's `THORAX` per the 15C decision; the camera is PG1.
- **Quietly promoting the diagnostic arm** if the detected numbers disappoint.
  The primary claim is detected 2D, because that is what a real clip gives.
- **Reporting the primary and burying the secondary**, or the reverse. Both are
  named above and both appear.

---

## Results

**Run once, 2026-11-03, after the pre-registration above was pushed as
`268f28b`.** `results/phase15_sealed.json`, driver
`scripts/phase15_sealed_run.py` (which imports the six frozen blobs and edits
none of them). No re-runs: the pipeline did not crash and produced no impossible
value, so the one-exception clause was never invoked.

**Blob verification, after the run.** `git hash-object` on the six pinned files
returns exactly the blobs this pre-registration fixed —
`video_kinematics.py 31524f7de685`, `phase15_movi.py 6a9d319f1cc5`,
`phase15d_segments.py 254b2d1f1f85`, `phase3_angles.py e1c6a6937125`,
`phase15_checks.py 75617f64a781`, `phase7_inference_fix.py 10a879fec3c6`. Not
one byte of the measured pipeline moved between the push and the result.

### Subjects: five in, none excluded

All five pass pre-registered checks 1 (units), 2 (convention), 5 (sync) and 6
(identity) — `results/phase15_sealed_checks.json`. **n = 5**, the largest MoVi
n in the project; the diagnostic comparison is n = 4 and the phase 15 headline
was n = 2.

| subject | c1 | c2 | c5 | c6 | PG1 `view_ratio` | travel azimuth | near limb | jogging frames (120 Hz) | video frames | detected | gaps excl. L / R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Subject_16 | PASS | PASS | PASS | PASS | 0.610 | 74.4° | left | 905 | 227 | 227 | 6.5% / 6.5% |
| Subject_17 | PASS | PASS | PASS | PASS | 0.698 | 78.9° | left | 881 | 221 | 221 | 8.0% / 6.4% |
| Subject_18 | PASS | PASS | PASS | PASS | 0.555 | 73.6° | left | 1113 | 279 | 279 | 0% / 0% |
| Subject_19 | PASS | PASS | PASS | PASS | 0.661 | 72.9° | left | 949 | 238 | 238 | 10.3% / 4.1% |
| Subject_2 | PASS | PASS | PASS | PASS | 0.537 | 77.4° | left | 625 | 157 | 157 | 0% / 0% |

The **re-registered femur bar `(420, 700)` mm passed all five.** The superseded
`(350, 550)` bar — an anatomical range applied to a `mean(ASI,PSI)→KJC` length
that is longer by construction — rejected two diagnostic subjects for being
tall. Re-deriving it from the check's *purpose* (unit detection) rather than
from the subjects is what made it hold on data it had never seen.

**The person detector found the subject in 227/227, 221/221, 279/279, 238/238
and 157/157 frames, with zero full-frame re-detections.** The crop tracker
never lost a limb. Split: grouped by subject, one subject per video, no subject
on both sides of anything — there is nothing to split, because nothing is
fitted here.

### PRIMARY — detected 2D, the cross-lab claim

Knee flexion and gravity-referenced thigh inclination. Per-subject MAE pooled
unweighted; CI by resampling **subjects**, `N_BOOT = 2000`, `SEED = 11`.
Diagnostic-set column is **quoted** from the pre-registration above, not
recomputed.

| quantity | limb | MAE | 95% CI | bias | SD of subject MAEs | diagnostic set (n=4) | Δ |
|---|---|---|---|---|---|---|---|
| **knee flexion** | **near** | **8.410°** | [6.275, 10.975] | +5.446 | 2.938 | 9.269° | **−0.859** |
| **knee flexion** | **far** | **8.685°** | [6.552, 10.679] | +6.030 | 2.567 | 9.725° | **−1.040** |
| **thigh inclination (gravity)** | **near** | **5.401°** | [4.136, 6.837] | −3.716 | 1.736 | 6.282° | **−0.881** |
| **thigh inclination (gravity)** | **far** | **7.372°** | [5.456, 9.258] | −5.996 | 2.506 | 6.487° | **+0.885** |

**The sealed set did not come out worse.** Three of the four primary figures are
*lower* than the subjects the method was developed on; the fourth, the far
thigh, is 0.885° higher. Every one of those four CIs contains the diagnostic
figure it is being compared against, and these are **different people, so no
paired test exists** — the honest reading is that the sealed numbers are
**indistinguishable from the spent ones**, not that they are better. What the
test rules out is the thing it was built to catch: that the MoVi figures were an
artifact of four subjects that had been looked at repeatedly.

Per subject, detected, **near** limb:

| subject | n | knee MAE | knee bias | knee r | thigh MAE | thigh bias | thigh r | hip MAE | trunk MAE |
|---|---|---|---|---|---|---|---|---|---|
| Subject_16 | 212 | 9.105 | +6.730 | +0.922 | 5.046 | −3.871 | +0.872 | 7.550 | 5.430 |
| Subject_17 | 203 | 5.130 | +0.562 | +0.918 | 3.274 | −0.052 | +0.908 | 5.156 | 5.079 |
| Subject_18 | 279 | 6.879 | +1.949 | +0.924 | 5.811 | −2.958 | +0.700 | 7.940 | 4.397 |
| Subject_19 | 214 | 7.973 | +5.881 | +0.894 | 4.841 | −4.279 | +0.832 | 9.601 | 6.265 |
| Subject_2 | 157 | 12.964 | +12.106 | +0.766 | 8.032 | −7.419 | +0.540 | 8.485 | 4.618 |

Per subject, detected, **far** limb:

| subject | n | knee MAE | knee bias | knee r | thigh MAE | thigh bias | thigh r | hip MAE | trunk MAE |
|---|---|---|---|---|---|---|---|---|---|
| Subject_16 | 212 | 9.092 | +7.165 | +0.898 | 9.339 | −9.001 | +0.790 | 10.970 | 5.430 |
| Subject_17 | 207 | 5.005 | +2.164 | +0.913 | 3.659 | −1.215 | +0.820 | 6.441 | 4.991 |
| Subject_18 | 279 | 7.583 | +3.342 | +0.883 | 6.871 | −4.323 | +0.671 | 9.050 | 4.397 |
| Subject_19 | 228 | 11.838 | +9.527 | +0.826 | 6.965 | −6.742 | +0.771 | 9.705 | 6.059 |
| Subject_2 | 157 | 9.906 | +7.951 | +0.754 | 10.026 | −8.697 | +0.409 | 10.118 | 4.618 |

### The error is almost entirely a per-subject offset

> **OVERSTATED — see "Correction, 2026-11-03" at the end of this file.** The
> heading and the paragraph below are kept as written. The correlation is real,
> but it describes *between-subject differences* in MAE, not the magnitude of
> the error: the within-subject SD is **4.5–9.7°**, comparable to the bias, and
> even oracle per-subject offset removal cuts MAE by only 17% (knee), 11%
> (thigh), 24% (hip).

Across the five subjects, near limb, **MAE tracks `|bias|` at Pearson r =
0.983 for the knee and 0.917 for the thigh.** Subject_17 is nearly unbiased
(knee +0.562°, thigh −0.052°) and scores 5.13° / 3.27°; Subject_2 carries a
+12.1° knee offset and scores 12.96°. Meanwhile the **waveform shape is tracked
well on every subject** — knee r +0.766 to +0.924, thigh r +0.540 to +0.908.

So this is not a method that fails to see the gait cycle. It sees it and places
it at the wrong absolute angle, by an amount that is a property of the person,
not of the camera: PG1's tilt is one fixed 3.699° (pitch −3.598°, roll +0.859°)
and the **level-camera error is −1.42° to −1.93° mean on all five subjects**
(SD 1.54–1.94), which is far too uniform to explain a bias range of +0.56° to
+12.11°.

### Near and far are not the same, and for two quantities the difference is resolved

Paired per-subject far-minus-near difference, subject-bootstrap CI:

| quantity | far − near | 95% CI | per-subject signs | resolved? |
|---|---|---|---|---|
| **thigh inclination** | **+1.971°** | **[+0.977, +3.187]** | +4.29, +0.38, +1.06, +2.12, +1.99 — **5/5 positive** | **yes, far is worse** |
| **hip (trunk-referenced)** | **+1.510°** | **[+0.611, +2.531]** | +3.42, +1.29, +1.11, +0.10, +1.63 — **5/5 positive** | **yes, far is worse** |
| knee flexion | +0.275° | [−1.719, +2.435] | −0.01, −0.12, +0.70, +3.87, −3.06 | no |
| shank inclination | −0.510° | [−1.404, +0.500] | −1.56, −0.97, +0.44, +1.01, −1.46 | no |

**This is the first far-limb penalty this project has measured with an interval
excluding zero**, and it splits exactly along the structural line phase 15C
drew. The **knee is a difference of two angles inside one frame**, so a depth
error on the occluded limb mostly adds noise and cancels — no resolved penalty.
**Thigh inclination and trunk-referenced hip are measured against an external
axis** (gravity, pelvis→thorax), so a far-limb depth error rotates the apparent
segment direction systematically — and both show ~1.5–2.0°, same sign on every
subject.

This does **not** overturn the README's "far-limb occlusion penalty is ~0": that
claim is about the sagittal joint angles on AthleticsPose and AthletePose3D, and
the knee — the closest quantity — still shows no resolved penalty here. What is
new is that **quantities referenced to an external axis do carry one**, and the
project's `CLAUDE.md` rule never to pool near and far is what made it visible.

### SECONDARY — the originally pre-registered quantities

| quantity | limb | MAE | 95% CI | bias | diagnostic set (n=4) | Δ |
|---|---|---|---|---|---|---|
| hip (trunk-referenced) | near | **7.746°** | [6.301, 8.931] | −6.054 | 12.270° | **−4.524** |
| hip (trunk-referenced) | far | **9.257°** | [7.829, 10.376] | −8.282 | 11.874° | **−2.617** |
| trunk lean | near | 5.158° | [4.622, 5.769] | −2.339 | 7.174° | −2.016 |
| trunk lean | far | 5.099° | [4.604, 5.646] | −2.286 | 7.169° | −2.070 |
| combined hip+knee mean | near | **8.078°** | [6.417, 9.674] | — | 10.769° | **−2.691** |
| combined hip+knee mean | far | **8.971°** | [7.278, 10.323] | — | 10.799° | **−1.828** |

**The trunk-referenced hip is the one quantity whose intervals do not overlap:**
7.746° [6.301, 8.931] on the sealed set against 12.270° [9.732, 13.926] on the
diagnostic set. **The quantity phase 15C identified as the *least* transferable
is the one that improved most on people it had never seen.**

That is not a contradiction of 15C, it is a consequence of it. 15C showed the
lifter's thorax lands on a different anatomical point in the two datasets, and
phase 15 already noted the resulting hip offset was **not** one fixed
convention: −7.2° against −3.3° on two subjects. The sealed set extends that
range to **−2.92°, −5.23°, −6.85°, −7.05°, −8.23°** — a smooth per-subject
spread, not a constant. A four-subject pool simply happened to be drawn from the
bad end of it. **The hip error is a subject-dependent trunk-posture term, and
four subjects were not enough to see its spread.**

The combined hip+knee mean has no bias entry because it is a mean of two MAEs,
not a signed quantity. The driver printed `+nan` there; the JSON stores no bias
for it. Cosmetic, no number affected.

### The pre-registered identity holds

`hip bias = thigh bias + trunk-lean bias`, detected arm, near limb:

| subject | hip bias | thigh bias + trunk bias | residual |
|---|---|---|---|
| Subject_16 | −5.2282 | −5.2282 | +3.33e−15 |
| Subject_17 | −2.9190 | −2.9190 | −2.22e−15 |
| Subject_18 | −6.8543 | −6.8543 | +1.78e−15 |
| Subject_19 | −7.0450 | −7.0450 | +2.22e−15 |
| Subject_2 | −8.2253 | −8.2253 | +5.22e−15 |

Floating-point exact on all five, as it was on all twelve 15D rows. The sealed
run is internally consistent — which it would be whether the numbers were good
or bad, and that is the point of checking it.

### Against the tune set: the gravity-referenced thigh transfers, the joint angles do not

AthleticsPose figures as quoted in the pre-registration. **The thigh baseline is
both sides pooled** — phase 15D's G1 pooled left and right into each subject's
list, so it cannot be split near/far from the stored output, and the sealed near
and far are each compared against that pooled number and labelled as such.

| quantity | AthleticsPose (tune) | sealed near | ratio | sealed far | ratio |
|---|---|---|---|---|---|
| knee flexion | 4.071° [3.836, 4.459] | 8.410° | **2.07×** | 8.685° | **2.13×** |
| hip (trunk-ref) | 2.834° [2.695, 2.930] | 7.746° | **2.73×** | 9.257° | **3.27×** |
| **thigh inclination** | 6.694° [6.290, 7.390] *(pooled)* | **5.401°** | **0.81×** | 7.372° | 1.10× |
| combined hip+knee | 3.452° *(mean of the two above)* | 8.078° | **2.34×** | 8.971° | 2.60× |

**Gravity-referenced thigh inclination is the only quantity that survives the
change of lab, and on the near limb it is better on held-out MoVi than on the
tune set** — 5.401° against 6.694°, with overlapping intervals, so call it *no
loss* rather than a gain.

This inverts phase 15D's verdict in a way worth stating plainly. **G1 failed:**
thigh inclination scored 6.694° against a 4.118° bar on AthleticsPose, worse
there than the trunk-referenced hip's 3.697°. **The quantity that failed the
tune-set gate is the quantity that transfers.** The reason is the one G2 gave
when it passed 4 of 4: gravity is an external reference the model does not have
to infer, while the pelvis→thorax axis is one it re-invents per dataset. A gate
set on the tune set measured the wrong thing, and only held-out data could show
that.

OpenCap Monocular's 4.8° (walking, squats, sit-to-stand) remains cited as
context, not a head-to-head: different movement, different task.

### DIAGNOSTIC — projected reference 2D, never pooled into the primary

| quantity | limb | MAE | 95% CI | bias | detected MAE | detector share |
|---|---|---|---|---|---|---|
| knee flexion | near | 6.159° | [4.729, 8.000] | −0.950 | 8.410° | +2.251 |
| knee flexion | far | 5.055° | [4.267, 6.011] | +0.908 | 8.685° | +3.630 |
| thigh inclination | near | 4.067° | [3.093, 4.831] | −3.271 | 5.401° | +1.334 |
| thigh inclination | far | 5.749° | [4.210, 7.075] | −5.167 | 7.372° | +1.623 |
| hip (trunk-ref) | near | 8.879° | [7.249, 9.924] | −0.781 | 7.746° | **−1.133** |
| hip (trunk-ref) | far | 9.200° | [7.897, 10.315] | −2.677 | 9.257° | −0.057 |
| trunk lean | near | 8.808° | [7.864, 9.594] | +2.490 | 5.158° | **−3.650** |
| trunk lean | far | 8.808° | [7.864, 9.594] | +2.490 | 5.099° | **−3.709** |

**Perfect 2D is not an upper bound on performance, and the sealed set makes that
unmistakable.** On the trunk, feeding the lifter the reference 3D projected
through PG1's own calibration is **worse on all five subjects** (−2.51° to
−4.58°); on the near hip it is worse on four of five. Only the knee and thigh
improve.

Per-subject detector share (detected MAE − perfect-2D MAE), near limb:

| subject | knee | thigh | hip | trunk |
|---|---|---|---|---|
| Subject_16 | +4.047 | +0.632 | −1.899 | −4.583 |
| Subject_17 | −1.786 | +1.084 | −4.017 | −3.522 |
| Subject_18 | −2.464 | +0.976 | −2.273 | −4.574 |
| Subject_19 | +2.542 | +1.032 | −0.205 | −3.064 |
| Subject_2 | +8.916 | +2.946 | +2.730 | −2.508 |

**The knee's detector share spans 11.4° and changes sign** — −2.46° on
Subject_18, +8.92° on Subject_2. This replicates phase 15B at n=5 (it measured
−2.65° to +5.69°, changing sign) and phase 7 §5's `CKPT_FOR` lesson: the lifter
has learned its detector's error distribution, so cleaner-than-trained 2D is not
automatically better. **The thigh is the exception: its detector share is
positive on all five subjects, +0.63° to +2.95°** — the only quantity here for
which "a better detector would help" is supported rather than assumed.

The trunk's bias also *flips sign* between arms, −2.339° detected against
+2.490° perfect. A quantity whose error changes direction when the input is
improved is not measuring what its name says.

### Deviations, recorded

- **D-S1 — `--out` is inert in a frozen file, found during the run.**
  `scripts/phase15_checks.py` parses `--out` but writes to the module-level
  `OUT` regardless, so running it on the sealed subjects overwrote the canonical
  diagnostic-set `results/phase15_checks.json`. **The file is one of the six
  blobs this pre-registration pinned, so it was not edited.** The file it wrote
  was copied to `results/phase15_sealed_checks.json` and the canonical one
  restored with `git checkout --`; verified afterwards to hold
  Subject_1/10/11/12/13/14/15 against the sealed file's 16/17/18/19/2. **Before
  and after: no number changed** — this is a write-path bug, not a measurement
  bug, and the pre-registration's one-exception clause (crash or impossible
  value) does not cover it, so it stays unfixed until after this test is
  committed. It is the same shape as the `f_probe` default-argument bug fixed
  in `7a985d2` (recorded in that script's own comment, not as a numbered
  deviation): an argument that looks wired up and is ignored at the point of
  use, failing by writing to the wrong place rather than by raising.
- **D-S2 — combined hip+knee prints `+nan` for bias.** It is a mean of two
  MAEs and has no signed form. Cosmetic; the JSON stores no bias for it.
- **D-S3 — the AthleticsPose thigh baseline cannot be split near/far.** Phase
  15D's G1 pooled left and right sides into each subject's list. The sealed near
  and far figures are each compared against the pooled 6.694° and labelled.
- **D-S4 — `MOCAP_PER_VIDEO` truncation.** Per-subject `n` in the primary
  tables (157–279) is below the jogging window's 120 Hz frame count because the
  reference is sampled at the video's 30 fps and then masked per quantity for
  marker gaps, exactly as the frozen path does.

### What surprised me

1. **The sealed set is not worse.** Going in, the expectation from phases 15–15D
   was that four repeatedly-inspected subjects would flatter the method. They did
   not; on the hip they were markedly *harsher* than five fresh ones.
2. **The quantity that failed the tune-set gate is the quantity that transfers.**
   G1's bar was set on AthleticsPose and rejected gravity-referenced thigh
   inclination at 6.694°. On held-out data it is the only quantity within range
   of its tune-set figure. A tune-set gate can be confidently wrong about which
   measurement is good.
3. **A resolved far-limb penalty, after the project repeatedly measured ~0.**
   It appears only in quantities referenced to an external axis, and the
   never-pool-near-and-far rule is the only reason it was visible at all.
4. **The hip's error is a subject-dependent posture term, not a convention
   offset** — five more subjects turned a two-point spread into a smooth
   −2.92° to −8.23° range.
5. **Nothing failed a check.** After phases 15–15D found four separate defects
   and excluded a subject, five consecutive clean subjects was not what I
   expected, and it is largely because the femur bar was re-derived from purpose
   rather than from the subjects it had to pass.

### No MoVi frame, crop, overlay or render is published

The MoVi licence forbids distributing the data in any form to any third party.
This report carries angles and summary statistics only; no participant
anthropometry, no camera parameters, and no imagery. Same rule as
`docs/data-inventory.md` §8.5.

---

## Correction, 2026-11-03

**What was claimed.** The section "The error is almost entirely a per-subject
offset" above, and the matching README paragraph, said the sealed error is a
per-subject offset rather than noise, on the strength of MAE tracking `|bias|`
at Pearson **r = 0.983** (knee) and **0.917** (thigh) across the five subjects.

**What is measured.** The correlation is correct and is not withdrawn. The
inference drawn from it was wrong. **r describes the *between-subject
differences* in MAE, not the magnitude of the error within a subject.** The
quantity that decides whether an offset can be removed is the per-subject
**SD**, and it is large:

| quantity | per-subject SD, near limb, detected |
|---|---|
| knee flexion | 7.58, 8.41, 8.71, 9.47, 9.73 deg |
| thigh inclination | 4.45, 4.48, 5.15, 6.63, 9.31 deg |
| hip (trunk-referenced) | 5.47, 6.27, 6.97, 7.57, 9.29 deg |

**4.5-9.7 deg, comparable to or larger than the bias itself.** So how much could
a per-subject offset correction actually remove? Under a folded-normal model --
which reproduces the five known per-subject MAEs to within **1.4 deg worst case**
(Subject_18's hip, 9.34 modelled against 7.94 observed), so read these as
estimates, not measurements:

| quantity | MAE now | subtract a **population** constant | subtract each subject's **own** bias (oracle) |
|---|---|---|---|
| hip (trunk-referenced) | 7.75 deg | 5.90 deg (**-24%**) | 5.68 deg (**-27%**) |
| knee flexion | 8.41 deg | 7.72 deg (-8%) | 7.00 deg (**-17%**) |
| thigh inclination | 5.40 deg | 5.19 deg (-4%) | 4.79 deg (**-11%**) |

**Even a perfect oracle leaves 4.8-7.0 deg**, against the 3.51 deg deployable
baseline. The only rigorous part, holding for any distribution, is
`MAE >= |bias|` by Jensen; the rest is the normal approximation above.

**Why the high shape correlation is not evidence against this.** Knee flexion
swings roughly 100 deg over a stride, so a shape correlation of r = 0.9 still
permits a residual SD near 10 deg: `sd_residual ~ sd_gt * sqrt(1 - r^2)`. **r is
high because the signal is large, not because the error is small.**

**The corrected wording**, which is what the README now carries:

> Bias explains the *between-subject differences* in MAE -- r = 0.983 (knee),
> 0.917 (thigh) -- but the within-subject SD is 4.5-9.7 deg, so most of the
> error is not a removable offset. Oracle per-subject offset removal cuts MAE by
> only 17% (knee), 11% (thigh) and 24% (hip).

Nothing above the Results heading is edited, and no measured number changes --
this corrects an inference, not a figure.

---

## Scoping, 2026-11-03: the calibration-pass idea, and why MoVi cannot test it

**The idea.** The subject stands still side-on for a few seconds; the tool reads
its own estimate of that pose, compares it against an assumed population
standing angle, and subtracts the difference as a per-subject offset. It uses no
ground truth at run time, which is what would make it deployable.

**MoVi round F cannot test it. Read-only finding, no model run and no new
subject opened** -- `scripts/phase15_standing_scope.py`,
`scripts/phase15_standing_viewpoint.py`, over the twelve already-opened
subjects.

**1. There is no standing pose, by name.** `motions_list` carries 20 motions
common to all twelve subjects plus one subject-specific `*_rm` motion. Nothing
matches stand / neutral / T-pose / A-pose / static / calib / rest / idle.

**2. There is plenty of stationary material.** Eight of the 20 are upper-body
tasks performed from standing. "Legs still" below means lower-limb marker median
speed under 100 mm/s at 120 Hz -- the four best are present in all twelve
subjects:

| motion | median continuous still | min across subjects | median limb speed |
|---|---|---|---|
| `hand_clapping` | 3.37 s | 1.54 s | 14.4 mm/s |
| `crossarms` | 3.17 s | 1.29 s | 13.2 mm/s |
| `scratching_head` | 3.10 s | 0.30 s | 13.9 mm/s |
| `checking_watch` | 2.92 s | 1.08 s | 18.4 mm/s |

14-18 mm/s is postural sway. **Indices are randomised per subject**, the same
trap `jogging_window` exists to avoid. Separately, **11.5-28.7% of all frames
belong to no motion window** -- the rest periods between motions -- giving a
longest single still run of 3.2-8.9 s and a longest-still-anywhere of
4.35-22.07 s. Every frame of it is already in the PG1 video on disk, because PG1
is one continuous recording per subject and `flags30` cuts it.

**3. And all of it is at the wrong viewpoint. This is the disqualifying fact.**
`view_ratio` is phase 3B's, unchanged (0 = frontal, 1 = side-on):

| window | median `view_ratio` | range |
|---|---|---|
| **jogging** | **0.675** | [0.536, 0.720] |
| the four still motions | **0.057** | [0.022, 0.109] |
| inter-motion rest | **0.027** | [0.012, 0.090] |

MoVi subjects stand facing the cameras and then run across the volume, so every
stationary frame is essentially frontal. Two consequences:

- Phase 11 measured sagittal error falling **3.50 deg frontal -> 1.77 deg
  side-on**, and phase 14 found camera azimuth carries essentially *all* the
  per-frame confidence signal. An offset read at `view_ratio` 0.03 would be
  dominated by viewpoint error rather than by the per-subject constant it is
  meant to capture.
- **The near limb differs between the standing pose and jogging on 6 of 12
  subjects**, so on half of them the calibration would be read off the other
  leg.

**This is a property of MoVi's protocol, not a verdict on the method.** Testing
the idea needs a subject standing *side-on*: phase 16A's consented clip, or
possibly the dual-system set, which is unopened.

### Where the population standing value could come from, without touching MoVi

| option | verdict |
|---|---|
| **AthleticsPose static frames** | **Measurably unavailable.** Eight actions, none static. The slowest is `sd` (912 clips) at 5.45 mm/frame; **0% of frames below 1 mm/frame**, longest static run **4 frames** (~0.03 s at its ~120 fps inferred in phase 7). `shotput` 2 frames, `discus` 1, `javelin` 0. Closed. |
| **AddBiomechanics** (CC BY 4.0, 273 subjects, not downloaded) | **Best source.** OpenSim pipelines *require* a per-subject static calibration trial, so neutral-pose joint angles are intrinsic to the processing rather than something to re-derive, and it is the most permissive licence in the dataset table. Its existing role -- "train / reference: joint-angle ranges and segment-length ratios as plausibility bounds" -- already covers this use. **Ships no video**, so it supplies the reference value only, never a check on what the tool reads. |
| **BEDLAM** (not downloaded) | The only option offering standing *with* rendered video, so it could check both the reference value and what the tool reads at a chosen viewpoint. Synthetic, and brings an SMPL-X convention mapping. |
| **AMASS** (not downloaded) | Weaker. Poses are relative to SMPL's rest pose, not anatomical standing, so it adds exactly the convention-mapping error class phase 15C measured at 87-144 mm of thorax displacement. |
| **Literature** | Adequate as a sanity bound, poor as a correction: published "neutral" differs by marker set and by the definition of neutral -- the same convention problem. |
| **Ferber archive** | Excluded by rule: it plays no part in the measurement track. |
| **Footage filmed for this project** | The only source of a *side-on* standing pose. No ground truth, so it cannot supply the population value -- but it can test whether the offset is stable across repeat scans. |

**Judgement, given the correction above.** The calibration pass is worth
building for the **hip**, where a population constant recovers an estimated
-24%, and is close to pointless for the **thigh** at -4%. It is not a route to
the 3.51 deg baseline, and a pre-registration should say so before it is built.
