# Phase 3 — setup on Kaggle

> ## ⚠️ SUPERSEDED — this route was abandoned. Phase 3 ran locally on CPU.
>
> Two Kaggle runs were given neither GPU nor Internet: the account is not
> phone-verified and Kaggle withholds both silently (`torch 2.10.0+cpu`,
> `Could not resolve host: github.com`). **Phase 3 was run on this laptop's CPU
> instead**, in ~90 min, and completed. GPU was only ever a speed convenience —
> the quantity measured is identical either way.
>
> **Results: `results/phase03.md`.** Runner: `scripts/phase3_infer.py`.
> `notebooks/phase3_kaggle.ipynb` and `notebooks/phase3_kaggle.py` are the
> superseded Kaggle route, kept as the record.
>
> **Two claims below are also outdated**, and are corrected elsewhere:
> - *"phase 3 needs CUDA"* — it does not.
> - *"the signal survives 8° of far-limb error"* — that reads phase 2's **nominal
>   σ**, not the error that actually reached the classifier. Phase 2 low-pass
>   filters after injecting noise, so σ 8 delivers a realized 1.69° at full
>   resolution. See the correction in `results/phase04.md`.
>
> The body is kept unchanged as a record of what was planned.

**Status: ~~setup ready, not yet run~~ — abandoned, see above.** Phase 3 needs
CUDA, which this laptop does not have. `CLAUDE.md` puts it on Kaggle Notebooks
(P100/T4) and forbids the Ferber archive leaving the laptop — nothing here needs
it.

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

## What the repo actually provides — checked, not assumed

- **Model:** MotionAGFormer (`base` and `small` variants).
- **Three evaluation configs**, which is the whole reason to use this dataset:
  `default` (trained on AthleticsPose), `h36m_pretrained` (generic baseline),
  `ap3d_pretrained` (AthletePose3D, needs `model=small`). Generic vs
  sport-fine-tuned brackets the operating point instead of assuming one.
- **Running and sprinting are both present**: `configs/data/running.yaml` and
  `configs/data/sd_sprint.yaml`.
- **Joint convention is H36M-17**, order `PELVIS, R_HIP, R_KNEE, R_ANKLE, L_HIP,
  L_KNEE, L_ANKLE, SPINE, THORAX, NECK, HEAD, L_SHOULDER, …` — verified against
  `athleticspose/statics/joints.py`, and it matches the map in
  `scripts/phase3_angles.py` exactly.
- **Estimates** are `.npy` `(T, 17, 3)` mirroring the input tree; **ground
  truth** is `.npz` under key `markers_h36m`, same shape.

### Two limits that bound what phase 3 can claim

1. **Original videos are not released** (anonymisation). What ships is 2D marker
   detections plus 3D ground truth, so this measures the **2D → 3D lifting**
   stage. Using `marker_type=det_ft` means real detector error *is* included, but
   raw video decoding and person detection are not. The result is a **lower
   bound** on a full in-the-wild pipeline.
2. **The estimator denormalises each clip using a scale derived from ground-truth
   3D.** A deployed system has no such scale. The reported errors are therefore
   **optimistic**, and phase 4 must not assume that scale is available.

Neither is a reason to skip phase 3 — a lower bound is exactly what decides
whether the path is worth pursuing. Both belong in `results/phase03.md`.

## Steps

1. **New Kaggle notebook.** Settings → Accelerator **GPU T4 ×2** (or P100);
   Settings → Internet **ON** (needed for the repo, data and checkpoints).
2. **Upload `scripts/phase3_angles.py`** as a Kaggle dataset named
   `overstride-scripts`. It is the only file that needs to cross, has no Ferber
   dependency, and imports nothing from this repo.
3. **Import `notebooks/phase3_kaggle.ipynb`** (File → Import Notebook). It is
   generated from `notebooks/phase3_kaggle.py` by
   `scripts/make_phase3_notebook.py` — edit the `.py`, regenerate, so changes
   stay reviewable in diffs.
4. **Cell 3 inspects before anything computes.** The layout above is expected,
   not verified from outside Kaggle. Confirm it rather than trusting it — writing
   a pipeline against a guessed schema is how phase 0's traps were made.
5. **Download `phase3_angle_errors.json`** (a few KB) — the only thing that
   leaves Kaggle.

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
  (`joints.py` also defines 84- and 64-marker mocap sets which *do* include toe;
  if the release exposes those for the ground truth, ankle becomes recoverable on
  the GT side but still not from an H36M-17 estimate.)

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
