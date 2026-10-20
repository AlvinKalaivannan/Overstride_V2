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

## Results

*Not yet run. MoVi has not been downloaded at the time of this commit.*
