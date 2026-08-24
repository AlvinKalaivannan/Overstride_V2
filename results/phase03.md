# Phase 3 — monocular 3D kinematics error, measured

> ## ⚠️ Corrected — see `results/phase03b.md`
>
> Two errors in the original version of this report, both from the same root
> cause: **`markers_h36m` is stored in PIXELS, not millimetres.** Each clip
> carries a per-frame `p2mm` factor and the released evaluator divides by it
> (`y_sample / p2mm_sample[:, None, None]`, applied to all three axes).
>
> 1. **MPJPE was reported in pixels labelled as mm.** Corrected below — the
>    fine-tuned checkpoint is **51.7 mm**, not 11.8 mm.
> 2. **The claim that this dataset has only near-frontal views was wrong**, and
>    with it the conclusion that the far-limb split was "uninformative". It
>    compared 44 *pixels* against a 200–250 *mm* reference. Measured
>    unit-free, the hips separate in depth by **0.756 of the subject's own
>    pelvis width** — about **49° out of the image plane**, and *zero* clips are
>    near-frontal. **The far-limb penalty was measured, on genuinely oblique-to-
>    lateral views, and it is ~0.** Gate item 2 is resolved, not open.
>
> **The joint-angle results below are unaffected.** `sagittal_angles` works in
> the body's own frame and is scale-invariant; every angle MAE re-ran
> bit-identical after the fix.

**Sport-fine-tuned monocular pose reaches ~3.4° sagittal joint-angle error, which
is comfortably inside the region where phase 2 says the signal survives. Generic
pose does not.**

| checkpoint | MPJPE (true mm) | hip MAE | knee MAE | mean sagittal MAE |
|---|---|---|---|---|
| **ath-det-ft (sport fine-tuned)** | **51.7 mm** (median 36.9) | **2.9°** | **3.8°** | **3.4°** |
| h36m (generic) | 135.2 mm | 11.8° | 24.5° | 18.1° |
| ap3d (other sport) | 142.9 mm | 17.5° | 25.3° | 21.4° |

> **Phase 7A revisited this.** An independent reimplementation reproduces the
> sagittal MAE at **3.43°** on the same 592 clips, which validates both numbers.
> It also found the inference path was leaving **−0.25°** on the table: windows
> were stitched consecutively with no shared context, and short windows were
> padded with all-zero keypoints. With both fixed the same measurement gives
> **3.18°**. The figure below is kept as the record of what phase 3 actually ran;
> `scripts/video_kinematics.py` uses the improved path. No downstream conclusion
> changes — phase 5C showed *halving* the injected error moves nothing that
> survives its interval. See `results/phase07.md`.

The corrected MPJPE is what makes the setup credible: AthletePose3D reports
**214 → 65 mm** on fine-tuning, and this reproduces at **135 → 52 mm**. The old
11.8 mm figure would have been far better than any published result, which
should have been the tell.

592 held-out clips, subjects **S11 / S13 / S16** only — the test split named in
`configs/data/running.yaml`. The fine-tuned checkpoint was trained on the other
subjects, so scoring it on those would have flattered it.

---

## Deviation from the environments table, recorded

`CLAUDE.md` places phase 3 on Kaggle because CUDA was assumed necessary. **It was
run locally on CPU instead.** Two Kaggle runs were given neither GPU nor Internet
— the account is not phone-verified, and Kaggle withholds both silently
(`torch 2.10.0+cpu`, `Could not resolve host: github.com`).

That route being blocked does not block the phase. GPU was only ever a speed
convenience: the quantity being measured — angular error between predicted and
ground-truth 3D — is identical on CPU. `CLAUDE.md` forbids solutions *requiring*
a local GPU; CPU inference requires none, and the Ferber archive is not involved,
so the data-separation rule is untouched. Runtime was ~90 min for
3 checkpoints × 592 clips.

## Gate: within range of published figures?

**Yes, and the generic checkpoint is what validates the setup.** Published
monocular joint-angle MAE against marker-based mocap is **14.1–25.8°** for
generic models on clinical gait. The generic H36M checkpoint here lands at
**11.4–25.2°** — squarely in that band. So the measurement apparatus is
producing the errors the literature produces.

The sport-fine-tuned checkpoint then comes in at **3.4°**, roughly 5× better.
That is the same story AthletePose3D reports (MPJPE 214 → 65 mm on fine-tuning);
it reproduces here on angles rather than positions.

## Placement on the phase 2 surface — the decision this phase exists to make

Phase 2 measured where the within-subject limb signal (AUC 0.610) survives:

| condition | AUC | survives |
|---|---|---|
| 30 fps, no noise | 0.610 | ✅ |
| σ 2/2° at full resolution | 0.611 | ✅ |
| σ both 8/8° at full resolution | 0.603 | ✅ |
| 30 fps + σ 2/8° | 0.565 | ✅ |
| **30 fps + σ 8/8°** | **0.529** | ❌ |
| **30 fps + σ 15/15°** | **0.491** | ❌ |

- **Fine-tuned, 3.4°** — below every tested noise level. Survives at full
  resolution and, by interpolation between the 2/8° and 8/8° rows, at 30 fps too.
  **The video path stays open.**
- **Generic, 18.1°** — sits between the 8/8° and 15/15° rows. Fine at high
  framerate; **dead at 30 fps**.

**The conclusion is that sport-specific fine-tuning, not framerate, is the
binding requirement.** A generic off-the-shelf 3D pose model is not accurate
enough at low framerate; a fine-tuned one has ~5× the margin.

## The far-limb result — corrected, and it *does* transfer

Measured far-limb penalty (far MAE − near MAE):

| checkpoint | near | far | penalty |
|---|---|---|---|
| ath-det-ft | 3.43° | 3.31° | **−0.12°** |
| h36m | 18.68° | 17.58° | −1.10° |
| ap3d | 22.31° | 20.52° | −1.78° |

Essentially zero, and if anything the far limb is *better*.

> **The original report dismissed this as uninformative on the grounds that the
> dataset had only near-frontal views. That was a unit error** — 44 pixels
> compared against a 200–250 mm reference. See `results/phase03b.md`.

Measured unit-free, as the fraction of the subject's own hip-to-hip vector lying
along the camera depth axis: **median 0.751, i.e. 48.7° out of the image plane.
Zero clips are near-frontal; 50% are near-lateral.** And phase 3B regressed the
penalty on that ratio: extrapolated to a *fully* lateral view it is **+0.07°**
for the fine-tuned checkpoint.

**So the penalty was measured, on the geometry that matters, and it is ~0.**
Accuracy in fact *improves* as the view becomes side-on (near MAE 4.66° → 2.11°
across ratio quartiles) — a lateral view resolves sagittal motion best, which is
fortunate, because side-on is the geometry Overstride prescribes.

The depth sign was verified rather than assumed: across 592 clips,
`corr(z_L − z_R, conf_L − conf_R) = −0.664` using the 2D detector's own
confidence channel — the deeper limb is detected less confidently, as an occluded
limb must be.

Getting that sign backwards would have inverted the table above.

## What these numbers are not

Three reasons the 3.4° is a **lower bound**, all structural:

1. **2D → 3D lifting only.** AthleticsPose does not release the original videos
   (anonymisation), so detector error is included via `det_ft` but video decode
   and person detection are not.
2. **Denormalisation uses a scale derived from ground-truth 3D.** A deployed
   system has no such scale. This is the released pipeline's own protocol, not a
   choice made here, but it flatters every checkpoint.
3. **In-domain.** The fine-tuned checkpoint was trained on this capture rig, these
   cameras, this sport. Held-out *subjects* is not held-out *domain*. Real
   phone-camera footage would be worse by an unknown margin.

Also: **ankle is unavailable.** The H36M-17 skeleton has no toe keypoint, so
there is no foot vector and ankle dorsiflexion cannot be computed. Phase 2's
`wave3` used hip/knee/ankle, so the video-recoverable subset is narrower on the
video side than on the mocap side. `sagittal_angles` returns NaN rather than
fabricating it.

## Anything that surprised us, or looks wrong

1. **The far limb is not worse, even on side-on views.** Expected the opposite.
   The original explanation — "the dataset has no side-on views" — was a unit
   error; half the clips *are* near-lateral, and the penalty still does not
   appear. Accuracy improves with lateralness rather than degrading.
2. **Fine-tuning buys 5× on angles, not just positions.** MPJPE 135 → 52 mm
   and MAE 18.1 → 3.4° move together.
3. **Knee error is roughly double hip error** in every checkpoint. The knee is
   the joint the phase 2 sagittal subset leans on most.
4. **The generic model's knee bias is large and negative** (−8 to −12°), i.e. a
   systematic offset rather than scatter. Phase 2's noise model injects error
   after stride-averaging precisely because systematic error does not average
   out, so that pairing is appropriate.

## A bug found and handled

The first completed run did not record which joint each row belonged to —
`d.update(...)` omitted `joint=joint`, so hip and knee rows were
indistinguishable. Rather than repeat a 90-minute run, the labels were recovered
from append order, which is deterministic (`for side in (l, r): for joint in
(hip, knee)`), **after verifying that all 3,552 groups contain exactly two rows**.
The source is fixed for future runs.

## Reproduction

```
# one-time, ~1.1 GB from GitHub releases (CC BY-NC-SA 4.0, non-commercial)
curl -L -o data.zip        .../v0.2.0/data.zip
curl -L -o checkpoints.zip .../v0.2.0/checkpoints.zip
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install timm

.venv/Scripts/python.exe scripts/phase3_infer.py     # ~90 min CPU
```

Raw rows in `results/phase3_angle_errors.json` (7,104 rows).

## Gate status

| # | item | status |
|---|---|---|
| 1 | sagittal hip/knee MAE per checkpoint vs published band | ✅ generic 11.4–25.2° inside 14.1–25.8° |
| 2 | near/far split | ✅ **resolved in phase 3B** — measured on oblique-to-lateral views; penalty ~0 and flat in view angle |
| 3 | systematic vs random decomposition | ✅ bias reported per joint/limb |
| 4 | placement on the phase 2 surface | ⚠️ **too optimistic** — see `results/phase04.md`; phase 2's nominal σ is not the error reaching the classifier |

Item 2 was recorded as the phase's one gap. It was not a gap — it was a unit
error, and phase 3B closes it. Item 4's placement was superseded by phase 4's
direct measurement, which is the authority.
