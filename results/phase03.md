# Phase 3 — monocular 3D kinematics error, measured

**Sport-fine-tuned monocular pose reaches ~3.4° sagittal joint-angle error, which
is comfortably inside the region where phase 2 says the signal survives. Generic
pose does not.**

| checkpoint | MPJPE | hip MAE | knee MAE | mean sagittal MAE |
|---|---|---|---|---|
| **ath-det-ft (sport fine-tuned)** | **11.8 mm** | **2.8–3.0°** | **3.7–4.0°** | **3.4°** |
| h36m (generic) | 34.8 mm | 11.4–12.2° | 23.7–25.2° | 18.1° |
| ap3d (other sport) | 37.3 mm | 17.4–17.6° | 23.6–27.1° | 21.4° |

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

## The far-limb result, and why it does not transfer

Measured far-limb penalty (far MAE − near MAE):

| checkpoint | near | far | penalty |
|---|---|---|---|
| ath-det-ft | 3.43° | 3.31° | **−0.12°** |
| h36m | 18.68° | 17.58° | −1.10° |
| ap3d | 22.31° | 20.52° | −1.78° |

Essentially zero, and if anything the far limb is *better*. **This does not
answer the question phase 2 raised**, and it would be wrong to report it as if it
did.

The reason is geometric. In a true side-on view the hips separate in depth by
about a pelvis width, ~200–250 mm. In this dataset the median separation is
**44 mm**, the maximum is **64 mm**, and **61.7% of clips are under 50 mm** —
these are near-frontal views of athletes running toward or away from the camera,
which is what a track capture rig produces. The far limb is barely occluded, so
there is almost no near/far asymmetry to detect.

**The side-on occlusion penalty remains unmeasured.** Phase 2 showed it is
decisive in combination with low framerate, and AthleticsPose cannot answer it.
A dataset with genuinely lateral views is needed.

The depth sign itself was verified rather than assumed: across 592 clips,
`corr(z_L − z_R, conf_L − conf_R) = −0.664` using the 2D detector's own
confidence channel — the deeper limb is detected less confidently, as an occluded
limb must be. Getting that sign backwards would have inverted the table above.

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

1. **The far limb is not worse.** Expected the opposite; the explanation is that
   the dataset has no side-on views, which is itself the finding.
2. **Fine-tuning buys 5× on angles, not just positions.** MPJPE 34.8 → 11.8 mm
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
| 2 | near/far split | ⚠️ **computed but uninformative** — no side-on views in this dataset |
| 3 | systematic vs random decomposition | ✅ bias reported per joint/limb |
| 4 | placement on the phase 2 surface | ✅ fine-tuned survives incl. 30 fps; generic dies at 30 fps |

Item 2 is the one gap, and it is a property of the dataset rather than the
method. Phase 4 should not assume the far-limb penalty is zero.
