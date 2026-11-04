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

*Not yet run. No sealed-set byte has been downloaded at the time of this commit.*
