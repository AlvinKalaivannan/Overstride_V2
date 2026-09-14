# Phase 11 (B1) — an independent lab's data through the lifting path

**Question.** `docs/REVIEW.md` §7.6 records that every accuracy number in this
project rests on one dataset. AthletePose3D (Yeung et al., CVSports at CVPR 2025)
is the first independent one: a different lab, different athletes, 120 fps, four
calibrated cameras.

**Answer: the lifter degrades by 1.84× on data it did not train on.** 1.25° on
AthleticsPose against **2.29°** on AthletePose3D, across three subjects each, with
no overlap between the two groups.

---

## What this measures, and what it does not

**It does not validate the headline 3.4°.** That figure is a **detected-2D**
number. AthletePose3D's shipped clips carry **ground-truth 2D** — `data_input`'s x
and y are bit-identical to `data_label`'s, with the third channel a constant 1.0.
So what is measured here is the **lifting stage alone**, under perfect 2D input.

The detector's contribution was measured separately in phase 7 (**+0.45°** at full
n). Validating the two together needs raw video, which is addressed at the end of
this report.

Reported as a decomposition, not a replication: the error budget has never before
been split on data this project did not also train on.

---

## Data and provenance

| | |
|---|---|
| source | AthletePose3D, Yeung et al., CVSports at CVPR 2025 |
| licence | non-commercial scientific research only |
| retrieved | 2026-09-11, Google Drive folder `10YnMJAluiscnLkrdiluIeehNetdry5Ft` |
| `pose_3d.zip` | 1460 MB, Drive mtime **2025-07-11** |

**The erratum check passed, and it mattered.** The repository warns: *"11/07/2025 —
Erratum: A preprocessing mistake occurred in one camera angle of the running
motions (3D)… please re-download."* B1 is entirely about running and rests on that
3D ground truth. Drive preserves per-file modified times, so this was checked
rather than assumed: `pose_3d.zip` dates to 2025-07-11, an exact match to the
erratum read DD/MM. It is the corrected release.

**`model_params/` is deleted on download.** It holds AthletePose3D's own trained
weights. Phase 10's B0 check established this project's lifting weights were
trained on AthleticsPose, which is exactly what makes AthletePose3D an independent
test set; an AP3D-trained model on AP3D data would be train-on-test and would
destroy that property. `scripts/phase11_fetch_ap3d.py` purges the directory.

### Selecting the running clips

The shipped clips carry only `{data_input, data_label}` — no action label — so the
running subset could not be selected at all. Reconstructing the authors'
segmentation fails: grouping frames by video and cutting non-overlapping 81-frame
windows yields 3,143 test clips against the 3,067 shipped, and videos are not
even contiguous in `valid.pkl` (1,010 runs over 826 distinct videos).

Matched by **content** instead. The normalisation was recovered exactly —
`x/w*2-1`, `y/w*2-h/w`, verified at max |diff| 0.000000 — and each clip's first
frame matched against the per-frame records by KD-tree:

```
matched 21,529 of 22,153 clips | unmatched 624
worst ACCEPTED per-coordinate error 5.53e-08  (tolerance 1e-05)

RUNNING cohort: 1,872 clips / 3 subjects / 120 fps
  S2 222 | S3 693 | S4 957
```

**Both splits are indexed, deliberately.** AP3D's train/valid split partitions
*their* models' training and says nothing about this project. Every AP3D frame is
held-out data here, and restricting to their "valid" split would buy no
independence while costing two of the three running subjects — valid carries S2
alone.

**§7.4 is answered as a side effect.** Running is **120 fps from metadata**, not
inferred from cadence.

---

## Why the angles are comparable across two representations

`data_label` is an image-normalised perspective representation: its bone lengths
vary at **CV 6–12%**, which a rigid bone cannot do. That looked disqualifying, and
work stopped until it was resolved — this is the same class of error as the `p2mm`
pixel/mm confusion in phase 3B.

The quantity that matters for angles is the **within-frame bone-length ratio**, and
AthleticsPose's own ground truth is no better:

| | ratio CV |
|---|---|
| AthletePose3D `data_label` | 2.4 – 5.2 % |
| AthleticsPose `markers_h36m` | 2.4 – 6.5 % |

One AthleticsPose file shows thigh and shank CV *identical* at 27.24% — the
signature of the per-frame `p2mm` uniform scale, which cancels in angles. The two
representations are on equal footing, so the cross-dataset **angle** comparison is
fair. Positional quantities are not, and use `joint_3d_camera` (metric: thigh
503 mm, shank 337 mm) instead.

Separately verified: `normalize_kpts(pixels) == normalize_kpts(image-normalised)`
to **2.2e-16**, so AP3D's pre-normalisation passes through `lift()` harmlessly.

---

## Matched conditions

Both datasets go through the same checkpoint, the same `lift()` and the same
`sagittal_angles()`. The only thing differing is the data.

The **`ath-gt`** checkpoint is used — the one trained for ground-truth 2D input.
Pairing GT 2D with the `det-ft` weights is exactly the checkpoint/detector mismatch
phase 7 caught, which manufactured 1.67° of imaginary error.

Phase 3 never ran a GT-2D condition, so the AthleticsPose comparator did not exist
and is produced here.

**Ankle is excluded.** H36M-17 has no toe keypoint, so ankle dorsiflexion is not
recoverable; `sagittal_angles` already returns NaN for it.

---

## Result

Reported **per subject and averaged across subjects, never pooled** — three running
subjects is exactly the case where a clip-count-weighted figure overstates
precision.

| dataset | subject | clips | hip | knee | mean |
|---|---|---|---|---|---|
| AthleticsPose | S11 | 184 | 0.93 | 1.35 | 1.14 |
| AthleticsPose | S13 | 104 | 1.39 | 1.21 | 1.30 |
| AthleticsPose | S16 | 304 | 1.28 | 1.32 | 1.30 |
| | | | | | **1.25** (sd 0.08) |
| **AthletePose3D** | S2 | 222 | 1.66 | 2.38 | 2.02 |
| **AthletePose3D** | S3 | 693 | 2.01 | 2.53 | 2.27 |
| **AthletePose3D** | S4 | 957 | 2.41 | 2.76 | 2.59 |
| | | | | | **2.29** (sd 0.23) |

```
ratio 1.84x
pre-declared: <=1.5 transfers | <=3.0 degrades | >3.0 dataset-specific
VERDICT: DEGRADES on unseen data
```

**The separation is clean.** Every AthletePose3D subject (2.02, 2.27, 2.59) exceeds
every AthleticsPose subject (1.14, 1.30, 1.30). The worst AthleticsPose subject sits
at 1.30° against the best AthletePose3D subject at 2.02° — **no overlap**, so this
is not one outlier dragging a mean.

**The knee degrades more than the hip**: 1.98× against 1.69×. The knee was already
the worse joint on home data and it loses more on unseen data.

### How to read 1.84×

**It is a real degradation and should not be waved away.** A lifter that gives 1.25°
at home and 2.29° elsewhere is measurably dataset-sensitive, and that is a direct
qualification of §7.6.

**It is also still small in absolute terms.** 2.29° on an independent lab's data,
with perfect 2D, sits below the 3.4° the project reports for its own detected-2D
path. Nothing here suggests the lifting stage is the binding constraint.

**The comparison is conservative against this project's interest.** AthleticsPose's
held-out subjects are held out from *training* but share the lab, rig, marker
protocol and capture conditions of the training data. AthletePose3D shares none of
those. Some of the 1.84× is the gap between "held-out subject" and "held-out
everything", which is the gap that actually matters for deployment.

---

## Files

- `scripts/phase11_fetch_ap3d.py` — download, erratum check, `model_params` purge
- `scripts/phase11_clip_index.py` — clip → action mapping by content
- `scripts/phase11_lifter_transfer.py` — the measurement
- `results/phase11_lifter_transfer.json`

---

# Part 2 — the viewpoint penalty, measured instead of extrapolated

## What this replaces

`docs/REVIEW.md` §4 records the far-limb occlusion penalty as **+0.07°
extrapolated**. Phase 3B could not measure it: AthleticsPose's held-out clips
cluster tightly around one geometry — median view ratio 0.751, **no near-frontal
clips at all** — so the penalty had to be read off a regression extended past the
data.

AthletePose3D supplies what was missing: **four calibrated running cameras**,
spread 43.8° to 169.5° apart.

```
rm_camera_1  xyz [-2725, -3827, 1371]   azimuth -135.7 deg
rm_camera_2  xyz [  217,  4101, 1303]   azimuth  +98.6
rm_camera_3  xyz [ 3098,  2847, 1352]   azimuth  +54.9
rm_camera_4  xyz [ 2948, -4321, 1370]   azimuth  -62.8
```

The metric is **phase 3B's, unchanged** — `view ratio = |z_L − z_R| / ‖p_L − p_R‖`,
the fraction of the subject's own hip-to-hip vector lying along the camera depth
axis. 0 is perfectly frontal, 1 perfectly lateral, and `arcsin` of it is the pelvis
angle out of the image plane. Being a ratio of two lengths in the same units, it is
immune to the representation questions that make `data_label` unusable for
positional work. Using the identical metric is the point: the result has to be
comparable with phase 3B's, not merely similar in spirit.

## Coverage — the actual contribution

```
AthletePose3D : median 0.662   range [0.483, 0.850]   = 29-58 deg out of plane
AthleticsPose : median 0.751   (phase 3B, no near-frontal clips)
```

AP3D reaches down to **0.483**, well onto the frontal side of anything
AthleticsPose contains. That extension is what converts an extrapolation into a
measurement.

## Result

| view ratio | clips | MAE |
|---|---|---|
| [0.48, 0.61) | 468 | **3.50°** |
| [0.61, 0.66) | 468 | 2.77° |
| [0.66, 0.71) | 468 | 2.28° |
| [0.71, 0.85) | 468 | **1.77°** |

```
regression : MAE = -10.325 x view_ratio + 9.641
correlation: r = -0.512
```

**Monotone across all four quartiles, by a factor of two.** Accuracy improves as
the view becomes side-on.

### An independent cross-check the original could not do

The view ratio is derived from the **pose**; the camera azimuth from the published
**extrinsics**. They are independent routes to the same geometry, so agreement
validates both — and disagreement would have exposed a clip mis-assigned to a
camera by the phase 11 index.

| camera | clips | view ratio | MAE |
|---|---|---|---|
| `rm_camera_4` | 624 | 0.588 (most frontal) | **4.02°** |
| `rm_camera_1` | 624 | 0.665 | 2.43° |
| `rm_camera_3` | 624 | 0.731 (most lateral) | **1.85°** |

The per-camera ordering is monotone and matches the quartile trend exactly. Camera
identity, view ratio and error all agree.

**Only three of the four cameras appear.** `rm_camera_2`'s clips are not in the
matched index — they fall among the 624 unmatched clips, or its windows were not
shipped. Recorded rather than worked around.

## What this settles

**§4's claim is confirmed and strengthened.** It stated the far-limb occlusion
penalty is "~0 and does not grow with viewing angle". On an independent dataset
spanning a much wider range of geometries, it does not grow — **it shrinks**, and
substantially.

**It also reproduces phase 3 on independent data.** Phase 3 reported accuracy
improving 4.66° → 2.11° across view-angle quartiles on AthleticsPose. AthletePose3D
gives 3.50° → 1.77° — same direction, similar magnitude, different lab, different
athletes, different cameras. A viewpoint effect that survives that change of
dataset is a property of monocular lifting, not of one rig.

**And it is good news for the deployment geometry**, which prescribes a side-on
camera. The view this application asks for is the view the method handles best.

### The honest limit

**Fully lateral is still an extrapolation.** AP3D's range tops out at ratio 0.850;
ratio 1.0 lies outside it, so the fitted −3.49° for "median → fully lateral" is a
projection, not a measurement. What is measured is the trend across
[0.483, 0.850], which is a far wider span than phase 3B had, and the direction and
approximate slope are now established on data the project did not train on.

**Ground-truth 2D throughout**, so this is the *lifter's* viewpoint sensitivity. A
real detector's own sensitivity to viewing angle is not measured here and would
plausibly be larger, since a near-frontal view occludes the limbs it must find.

## Files

- `scripts/phase11_viewpoint.py`
- `results/phase11_viewpoint.json`
