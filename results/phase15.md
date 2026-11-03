# Phase 15 — Held-out validation on BML MoVi

> **PRE-REGISTRATION, written 2026-10-20, before a single byte of MoVi has been
> downloaded.** The pipeline frozen below is frozen *now*, in this commit, so the
> freeze is checkable in `git log` against the download timestamp rather than
> asserted afterwards. Results are appended under "Results"; nothing above that
> heading changes.

**Question.** The measurement was built and tuned entirely on AthleticsPose.
Does it hold on a lab it has never seen, filmed with a camera it has never seen?

`CLAUDE.md`: *"There is no pass/fail gate. A loss on held-out data is a result
and goes in the README exactly as prominently as a win."* That stands.

---

## The pipeline, frozen

Nothing below may be changed after MoVi is opened. Every number in Results comes
from exactly this:

| stage | implementation | setting |
|---|---|---|
| decode | `video_kinematics.decode` | native resolution |
| 2D detection | torchvision `keypointrcnn_resnet50_fpn`, DEFAULT weights | `min_size` = the frame's own shortest side; padded person crop, `CROP_PAD = 0.35`, full-frame re-detection when a limb leaves the window |
| smoothing | **none** | `LOWPASS_HZ = 0.0` — phase 13 measured it harmful |
| joint order | `video_kinematics.coco17_to_h36m17` | COCO-17 → H36M-17 |
| rate matching | `video_kinematics.resample` | to `MODEL_FPS = 100.0` |
| lifting | `phase7_inference_fix.lift` | `motionagformer-b-ath-det-coco-v1.ckpt`, `pad="edge"`, `windows="overlap"` |
| angles | `phase3_angles.sagittal_angles` | hip and knee; ankle is NaN, H36M-17 has no toe |
| confidence | `phase14_confidence` | clip-level expected error from `azimuth_proxy` + the published viewpoint curve; per-segment confidence for the **shanks only** |

**Reference point:** the deployable AthleticsPose figure, **3.509° near /
3.618° far** (`results/phase7_baseline_ci.json`). MoVi cannot be compared "on the
same clips", so both numbers are reported side by side and the difference is
called what it is — a change of dataset, camera, athlete population and movement.

## What is used, and what is not

- **PG (Point Grey machine-vision) clips carry the MAE.** The paper states the
  FLIR/Point Grey cameras are synchronised to the mocap, aligned with every
  fourth mocap frame at 30 fps. Synchronisation is a precondition for an angle
  error, so these are the measured clips.
- **CP (hand-held iPhone 7) clips answer §7.5** — "the video tool has never run
  on real footage" — because that is what real footage looks like: 800×600,
  30 fps, shaky. If their synchronisation to mocap can be verified from the
  dataset's own documentation, they carry an MAE too; **if it cannot, they are
  reported as traces with no error figure**, and §7.5 closes on the weaker claim
  rather than on a fabricated alignment.
- **Jogging only.** MoVi's movement 2 is jogging and movement 3 is *running in
  place*, which is not running gait: no flight-phase translation, and a
  stationary subject makes the sagittal camera geometry a different problem.
  Running in place is excluded, and the exclusion is reported with its n.
- **No MoVi frame, crop or rendering is published.** The licence is
  non-commercial research with no redistribution, and it also forbids training
  for commercial use. Summary metrics and plots of *angles* only.

## Checks that must pass before the first number is reported

Per `CLAUDE.md`: *"Unit traps like the AthleticsPose `p2mm` one are checked and
written into `docs/data-inventory.md` for every new dataset before its first
number is reported."* For MoVi:

1. **Mocap units** — mm or m, verified against a segment length that has to be
   human (femur 350–550 mm).
2. **Coordinate convention** — which axis is vertical, and the handedness.
   `sagittal_angles` builds its own frame, so it is robust to rotation but not to
   a mirrored axis, which would flip flexion sign.
3. **Marker set → H36M-17.** MoVi ships Vicon markers and MoSh/AMASS SMPL fits.
   The mapping used is stated explicitly, and a left/right sanity check is run
   (the near limb must be the one nearer the camera).
4. **Video frame rate**, read from the file, not assumed to be 30.
5. **Synchronisation offset** between video and mocap, from the dataset's own
   documentation. A constant offset of a few frames would corrupt every angle
   error; it is checked, stated, and if it cannot be established, the MAE is not
   reported.
6. **Subject identity** across modalities — the video file's subject id must be
   the mocap file's subject id.

All six go into `docs/data-inventory.md` §8.5 before Results is written.

## What is reported

- MAE per joint (hip, knee), **near and far limb separately**, with
  **subject-bootstrap CIs**, and the AthleticsPose figure beside each.
- n per group after filtering, and what was excluded and why.
- The **frame rate penalty, isolated if possible**: MoVi video is 30 fps against
  AthleticsPose's 100 Hz mocap-locked detections, so some of any degradation is
  temporal resolution rather than lab transfer. Where the mocap allows it, the
  same clips are scored with the mocap downsampled to 30 Hz, to separate the two.
- **How often phase 14's confidence flagged the frames that turned out worst** —
  Spearman of stated against realized error on MoVi, using the AthleticsPose-fitted
  model unchanged. Phase 14 found the confidence is largely a viewpoint meter, so
  this is a test of whether that transfers.
- OpenCap Monocular's 4.8° (walking, squats, sit-to-stand) as **context, not a
  head-to-head**, per `CLAUDE.md`.

## What would make this phase wrong

- **Tuning anything after seeing a MoVi number.** If a defect is found that
  forces a pipeline change — as four were found in the deployable path this week —
  the change is made, the old numbers are **voided and kept on disk**, and the
  phase re-runs from the frozen definition above with the deviation recorded.
  That is what happened to phase 14 attempt 1 and it is the only honest route.
- **Reading the degradation as lab transfer when it is frame rate**, or the
  reverse. Hence the 30 Hz control above.
- **n = 3 again.** However many MoVi subjects are used, the AthleticsPose side of
  every comparison rests on S11, S13 and S16.

---

## Deviations from the frozen definition, recorded before any result

The section above was frozen in `64fb0bc` and nothing in it has been edited.
Three things happened between that freeze and the first number, and the rule
this phase set for itself is the one being followed: *"If a defect is found that
forces a pipeline change … the change is made, the old numbers are **voided and
kept on disk**, and the phase re-runs from the frozen definition above with the
deviation recorded."* No reportable number was produced before any of these, so
there is nothing to void.

### D1 — Round F, not round S (`fe247a8`)

**Check 3 of the six failed for round S, irrecoverably.** `S_V3D.tar` ships 12
physical markers — head, wrists and ankles — and **no hip, knee or pelvis**,
which are exactly the joints this phase scores. It also ships no `motions_list`
and no frame flags, so it cannot be cut into movements at all.

Round F passes all six and ships more: 87 markers including `LFWT`/`LBWT` and
`RFWT`/`RBWT`, which *are* the ASIS and PSIS markers, so **H36M's own hip
definition — `mean(ASI, PSI)` — is exactly computable** rather than
approximated. This is a change of *which clips*, not of the pipeline, and it was
forced by a pre-registered check rather than chosen.

The 1.26 GB of round-S video downloaded first is the wrong round and is kept
only as the record of how this was found.

### D2 — The confidence channel was 14× out of scale (`2db742f`)

**A real defect in the deployable tool, not a property of MoVi.** `lift` feeds
channel 2 of the 2D array to the model as a per-keypoint weight. The checkpoint
was trained on AthleticsPose's released detections, whose channel 2 is a
*normalised* confidence — mean 0.8676, sd 0.1164 over 60 clips and 131,121
values. torchvision's `keypoints_scores` are *raw* heatmap scores, [3.9, 18.4]
on MoVi, and `detect_coco17` wrote them straight through.

| | MAE | r vs reference |
|---|---|---|
| raw scores | **30.07°** | +0.02 hip, +0.17 knee |
| fixed | **11.45°** | +0.84 hip, +0.91 knee |

With raw scores the estimated knee flexion reached **+66°**, which
`video_kinematics.PLAUSIBLE` itself calls impossible. The fix is a constant and
the choice is *a priori*: the trained channel is nearly constant anyway (13% CV),
phase 14 measured the detector's own confidence as **anti**-correlated with
thigh error, and anything data-dependent would have to be chosen on ground truth
*with video* — which only MoVi provides, and choosing on MoVi would invalidate
this phase. **Zero parameters are fitted to held-out data.**

**Why no earlier phase could catch it:** every number before this one came from
released `.npy` files that already carry normalised confidences, and
AthleticsPose ships no video. The detector half of this path had never once been
scored against ground truth. Second defect of this family after the COCO/H36M
joint order; both lived in the only code path with no ground truth to check it
against.

### D3 — Time alignment is exact, and better than the freeze assumed

The frozen text listed check 5 as *"synchronisation offset … if it cannot be
established, the MAE is not reported."* It can be, and not as an estimate: MoVi
ships `flags30` and `flags120`, the same movement boundaries expressed in video
and in mocap frames, at a measured ratio of **3.9981** (max deviation from 4 is
0.0112). So video frame *k* scores against mocap frame `f120[0] + 4k` and
**the reference 3D is never resampled**, as `CLAUDE.md` requires.

That also satisfies the pre-registered 30 Hz control *by construction*: the
comparison already happens on a 30 Hz grid against untouched 120 Hz mocap. What
it does **not** isolate is the frame-rate penalty on the AthleticsPose side,
which would need the tune set re-scored at 30 fps — a separate run, named here
so it is not mistaken for done.

### Two corrections to earlier claims of mine

- **`motions_list` order differs per subject.** Jogging is index 18 for subject
  1 and **16** for subject 10; `running_in_spot` moves from 7 to 19. An earlier
  commit message stated the index as if it were a dataset constant. The code
  always read it per subject; the claim about it was wrong.
- **`LHJC`/`RHJC` are the _hand_ joint centres**, not the hip — `markerDescription`
  reads "Left Hand Joint" and their height matches the elbow. The hip joint
  centres are `LHIP`/`RHIP`. With the right marker, the offset from H36M's
  `mean(ASI,PSI)` hip is **104/110 mm** on subject 1 and **90/92 mm** on subject
  10, independently reproducing the **81 mm** phase 12 measured on AthleticsPose.

## The six checks, answered

Full detail in `results/phase15_checks.json` and `docs/data-inventory.md` §8.5.

| # | check | round F |
|---|---|---|
| 1 | mocap units | **PASS** — millimetres. Femur, tibia and hip width all fall inside their pre-registered human ranges (350–550 mm, 300–500 mm). The per-subject lengths are deliberately **not** reported: see `docs/data-inventory.md` §8.5 |
| 2 | axis and handedness | **PASS** — vertical is axis 2, agreed independently by both camera centres sitting at 1.35–1.38 m and by head-minus-pelvis; jogging knee flexion `[-73.5, -8.7]°`, one-signed and anatomical |
| 3 | marker set → H36M-17 | **PASS** for round F, **FAIL** for round S (see D1) |
| 4 | video frame rate | **PASS** — read from the file: 30.000 fps, 800×600 |
| 5 | synchronisation | **PASS** — shipped by the dataset, ratio 3.9981 (see D3) |
| 6 | subject identity | **PASS** — the `.mat` subject field matches the filename for every subject |

**Gap sentinel, found and handled.** `markerGaps == 1` marks a *missing* marker
and Visual3D writes `[0, 0, 0]` for it. Read literally that put the knee joint
centre at the lab origin and drove knee flexion to +148.7°. Gapped markers are
NaN and the frames they touch are **excluded, never interpolated**. The
exclusion is **per quantity**: `sagittal_angles` reads seven joints, so an elbow
gap must not cost a knee frame. Globally it would have been 21.2% of subject 1's
window; correctly scoped it is 3.9% left / 4.5% right, and subject 10 loses
nothing.

## The side-on question, answered

`CLAUDE.md` says phase 15 must settle this first. Both metrics agree, on both
subjects measured:

| camera | `view_ratio` (1 = side-on) | travel azimuth (90° = side-on) | verdict |
|---|---|---|---|
| **PG1** | 0.720 / 0.705 | 75.7° / 75.0° [62–84] | **side-on** |
| **PG2** | 0.468 / 0.465 | 11.0° / 10.5° [4–23] | near-frontal |

**PG1 at 0.71 is essentially AthleticsPose's own median of 0.751**, so the
held-out set is viewed from the geometry this project prescribes — and PG2 gives
the frontal contrast that makes the phase 11 viewpoint curve (3.50° → 1.77°)
measurable on a lab the pipeline has never seen.

## Results

*The full run is in progress at the time of this commit. Numbers are appended
below with their CIs; the smoke figures quoted in D2 are from one clip and 150
frames and are not the result.*
