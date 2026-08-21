# Phase 3B — viewpoint geometry, and the far-limb penalty that isn't there

**The largest open gap in this project was not a gap. It was a unit error.**
AthleticsPose is not near-frontal: half the held-out clips are near-lateral, and
the far-limb penalty does not grow with how side-on the view is. Extrapolated to
a fully lateral view it is **+0.07°**.

---

## The error

`results/phase03.md` reported the near/far split as "computed but uninformative",
and `results/phase04.md` carried that forward as the project's biggest unresolved
question. The stated reason:

> In a true side-on view the hips separate in depth by about a pelvis width,
> ~200–250 mm. In this dataset the median separation is **44 mm** … these are
> near-frontal views of athletes running toward or away from the camera.

**`markers_h36m` is stored in pixels, not millimetres.** Each `.npz` carries a
per-frame `p2mm` factor, and the released evaluator divides by it — uniformly
across all three axes:

```python
# athleticspose/plmodules/linghtning_module.py
y_sample = y_sample / p2mm_sample[:, None, None]
```

So 44 pixels was being compared against a 200–250 mm reference. Converting
properly, on the 592 held-out clips:

| segment | measured | anatomical |
|---|---|---|
| femur | 479 mm | 390–460 |
| inter-hip-joint-centre | 207 mm | 170–200 (H36M hips are joint *centres*, not iliac crests) |
| hip depth separation | **145 mm** | — (reported as "44 mm") |

## The unit-free measurement, which sidesteps the question entirely

The quantity that actually matters is a ratio of two lengths in the same units,
so it is immune to the pixel/mm confusion:

```
view ratio = |z_L − z_R| / ‖p_L − p_R‖
```

— the fraction of the subject's **own** hip-to-hip vector lying along the camera
depth axis. 0 = perfectly frontal, 1 = perfectly lateral, `arcsin` of it is the
angle of the pelvis out of the image plane.

| | value |
|---|---|
| median view ratio | **0.751** |
| median out-of-plane angle | **48.7°** |
| p10 / p90 | 0.416 / 0.953 |

| view class | ratio | clips | share |
|---|---|---|---|
| near-frontal | [0.00, 0.30) | **0** | **0.0%** |
| oblique | [0.30, 0.60) | 268 | 45.3% |
| strongly oblique | [0.60, 0.85) | 28 | 4.7% |
| **near-lateral** | [0.85, 1.01) | **296** | **50.0%** |

**Zero clips are near-frontal. Half are near-lateral.** The original
characterisation was the opposite of the truth.

## Does the far-limb penalty grow as the view becomes side-on?

Per-clip penalty (far MAE − near MAE) regressed on that clip's view ratio,
n = 592 clips, slope CI from a 2,000-draw clip bootstrap:

| checkpoint | near | far | penalty | corr | slope / unit ratio | **penalty at ratio 1.0** |
|---|---|---|---|---|---|---|
| **ath-det-ft (fine-tuned)** | 3.43° | 3.31° | −0.12° | +0.161 | +0.64 [+0.26, +0.97] | **+0.07°** |
| h36m (generic) | 18.68° | 17.58° | −1.10° | +0.196 | +4.60 [+2.79, +6.50] | +0.26° |
| ap3d (other sport) | 22.31° | 20.52° | −1.78° | +0.176 | +3.79 [+2.20, +5.35] | −0.66° |

The slope excludes zero, so there *is* a real trend — but it runs from a slightly
**negative** penalty at oblique views to approximately **zero** at lateral ones.
It never becomes a penalty. By view-ratio quartile, fine-tuned:

| ratio bin | n | near | far | penalty |
|---|---|---|---|---|
| [0.35, 0.48) | 148 | 4.66° | 4.45° | −0.21° |
| [0.48, 0.75) | 148 | 4.51° | 4.13° | −0.39° |
| [0.75, 0.93) | 148 | 2.43° | 2.57° | +0.14° |
| [0.93, 0.97) | 148 | 2.11° | 2.08° | −0.03° |

## The finding that matters more than the correction

**Accuracy improves sharply as the view becomes side-on** — fine-tuned near-limb
MAE falls 4.66° → 2.11° from the most oblique quartile to the most lateral. This
is geometrically sensible: in a frontal view the leg swings along the camera's
depth axis, which is the one direction a monocular lifter cannot see. A lateral
view puts sagittal motion in the image plane, where it is most observable.

**Side-on is the best case for sagittal kinematics, not the worst.** Overstride
prescribes a side-on camera, so the geometry the product wants is the geometry
the model handles best. That is a genuinely favourable result and it was hidden
behind a unit error.

## What is still not measured

1. **Nothing reaches a full 90°.** Max view ratio is ~0.97 (76° out of plane).
   The extrapolation to 1.0 is short and the trend across the observed range is
   flat, but it is still an extrapolation.
2. **Occlusion by the *torso* is not the same as hip depth separation.** The
   ratio measures pelvis orientation; a genuinely occluded far limb also depends
   on limb crossing during swing. This measures the former.
3. **These are still track-rig captures**, multi-camera and well lit. Real
   phone footage is a different domain, unmeasured here and in phase 3.

## Reproduction

```
.venv/Scripts/python.exe scripts/phase3b_viewpoint.py    # ~2 min, no GPU, no model
```

Reads `results/phase3_angle_errors.json` and the AthleticsPose ground-truth
`.npz` geometry. Output: `results/phase3b_viewpoint.json`.

## Consequences recorded elsewhere

- `results/phase03.md` — MPJPE corrected to true mm (51.7 / 135.2 / 142.9);
  gate item 2 changed from ⚠️ open to ✅ resolved.
- `results/phase04.md` — the "side-on occlusion penalty is unmeasured" caveat is
  withdrawn, and a **lateral-only error bank** is now run as its own condition.
- **All joint-angle results are unaffected.** `sagittal_angles` works in the
  body's own frame and is scale-invariant; every angle MAE re-ran bit-identical
  after the correction, which is itself the check that the fix was inert where it
  should be.
