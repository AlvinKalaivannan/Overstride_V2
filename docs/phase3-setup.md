# Phase 3 — setup on Kaggle

**Status: setup ready, not yet run.** Phase 3 needs CUDA, which this laptop does
not have. `CLAUDE.md` puts it on Kaggle Notebooks (P100/T4) and forbids the
Ferber archive leaving the laptop — nothing here needs it.

---

## What phase 3 has to answer

`CLAUDE.md`'s gate is *"joint-angle MAE within range of published figures"*. That
is necessary but not sufficient for this project, because **phase 2 already told
us which number decides the outcome**: far-limb angular error.

Phase 2 found the injured-limb signal survives 30 fps and survives 8° of
far-limb error, degrading only in combination (AUC 0.610 → 0.565). It did not
sweep beyond 8°. Published monocular knee-flexion MAE against marker-based mocap
is **14.1–25.8°** for generic models on clinical gait — outside that range.

So phase 3 must report error **split by near vs far limb**, not as a single
average, and must say whether the error is **systematic across strides** or
random. Phase 2 injects noise after stride-averaging, which models systematic
error; if the real error is mostly random it averages out over ~28 strides and
phase 2 is pessimistic.

The phase 2 sweep has been extended to σ_far ∈ {12, 16, 20}° and symmetric
8°/15° so phase 3's measurement can be read off the surface wherever it lands.

## Why AthleticsPose is the primary dataset

| | AthleticsPose | AthletePose3D |
|---|---|---|
| access | public, GitHub release, direct download | licence agreement, manual download |
| licence | CC BY-NC-SA 4.0 | non-commercial research only |
| checkpoints | **3 released** (AthleticsPose / Human3.6M / AthletePose3D) | fine-tuned params, less clearly packaged |
| content | 23 athletes, real athletics on a track | 12 sports, 1.3 M frames, running at 120 fps |
| reported angle error | — | correlations only (lower limb r = 0.81), **no MAE in degrees** |
| tooling | `uv`, matches this repo | own pipeline |

AthleticsPose is the primary: public, packaged, and it ships a **generic
(Human3.6M) checkpoint alongside sport-fine-tuned ones**. That contrast is the
point — it brackets the realistic operating point instead of assuming one.
AthletePose3D is optional secondary breadth; it needs the licence agreement
accepted manually before download.

Both are non-commercial. `CLAUDE.md` already lists commercial framing as out of
scope, so no new constraint.

## Steps

1. **New Kaggle notebook.** Settings → Accelerator **GPU T4 ×2** (or P100);
   Settings → Internet **ON** (needed for the repo, data and checkpoints).
2. **Upload `scripts/phase3_angles.py`** as a Kaggle dataset named
   `overstride-scripts`. It has no Ferber dependency and imports nothing from
   this repo.
3. **Run `notebooks/phase3_kaggle.py` cell by cell.** Cells are marked
   `# %% CELL n`.
4. **Cells 3–4 inspect before computing.** The AthleticsPose array layout has not
   been verified from here. Fill cell 4 against what cell 3 actually prints —
   do not write the pipeline against a guessed schema. This is the same
   discipline that surfaced the phase 0 traps.
5. **Download `phase3_angle_errors.json`** (a few KB) and run
   `scripts/phase3_report.py` locally.

## Watch for

- **Kaggle GPU quota is ~30 h/week.** Inference over the dataset should be well
  inside it; a full fine-tune would not be, and is not needed — the point is
  *released checkpoints*.
- **Near/far limb assignment needs camera extrinsics.** If they are not in the
  release, the near/far split cannot be computed and **must not be guessed** — a
  wrong assignment would invert the headline result. Report it as unavailable
  and fall back to reporting symmetric error only.
- **Ankle dorsiflexion is not recoverable** from the Human3.6M 17-joint skeleton:
  there is no toe keypoint, so there is no foot vector. `sagittal_angles`
  returns NaN for ankle rather than fabricating it. Hip and knee are available.
  Phase 2's `wave3` used hip/knee/ankle, so the video-side subset is narrower
  than the mocap-side one — a real limitation for phase 4, not a bug.

## The convention gap that phase 4 must handle

Keypoint-derived joint angles are **three-point angles between segment vectors**.
The Ferber waveforms are **Cardan angles from marker-cluster segment coordinate
systems** (`gait_kinematics.m`). These are not the same quantity.

Within phase 3 it does not matter — estimate and ground truth use the identical
convention, so the MAE is valid. **In phase 4 it matters a great deal**, because
video-derived angles would be fed to a model trained on Cardan angles. Options
are to recompute Ferber-side angles in the keypoint convention from the stored
markers, or to fit the phase 4 model directly in the keypoint convention. Either
way it is real work and must not be assumed away.

## Gate

Phase 3 is complete when `results/phase03.md` records:

1. Sagittal hip and knee MAE in degrees, per checkpoint, against the published
   band (14.1–25.8° generic clinical gait; AthletePose3D lower-limb r = 0.81).
2. The **near/far limb split**, or an explicit statement that extrinsics were
   unavailable.
3. The **systematic vs random** decomposition, and the effective σ to use in
   phase 2's surface.
4. Where that σ places us on the extended phase 2 sweep — i.e. whether the signal
   survives at the measured error level.

Item 4 is the one that decides whether phase 4 is worth starting.
