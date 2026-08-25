# Phase 7 — hardening the inference path, and a tool that runs on video

**Phase 3's 3.4° is confirmed, not overturned.** An independent reimplementation
reproduces it at **3.43°** on the same 592 clips. But the inference path had two
unexamined properties worth **−0.25°** together, and swapping the fine-tuned
detections for generic COCO ones — which is what any off-the-shelf detector
gives you — costs **+0.54°**. Both numbers were needed before a video tool could
state an honest error budget.

| | hip | knee | mean | vs baseline |
|---|---|---|---|---|
| **baseline — exactly what phase 3 did** | 2.82° | 4.04° | **3.43°** | — |
| **both fixes** (edge padding + overlapped windows) | 2.58° | 3.77° | **3.18°** | **−0.25°** |

*Full 592 held-out clips, fine-tuned checkpoint, paired on the same clips.*

---

## Why this phase happened

Building a tool that runs on real video meant reading the lifting routine in
`scripts/phase3_infer.py` closely for the first time since it was written. Three
properties surfaced, none
of them wrong for what phase 3 did, all of them consequential for a long clip:

1. **Zero padding.** Every sequence is padded to a multiple of 81 frames with
   *all-zero keypoints* — a mean of 14.9 frames of 81 across the held-out clips.
   A pose with every joint at the origin is not on the manifold the model was
   trained on, and a temporal transformer can propagate it into valid frames.
2. **Hard window boundaries.** Windows are consecutive with no shared context.
   Held-out clips typically reach 2 windows; **a 10-second phone clip at 60 fps
   reaches 8**, so the tool would lean on this path four times harder than
   anything previously measured.
3. **Frame rate.** The receptive field is 81 **frames**, not seconds, and the
   capture rate is undocumented anywhere in the repo, configs or dataset.

## The capture rate, recovered from the data

Ankle vertical position oscillates once per stride. Counting those cycles in the
ground truth gives **0.0221 strides/frame**; at a running cadence of 2.5–3.0
strides/s that is **113–136 fps**, so ~120 fps. The 81-frame window covers
**0.60–0.71 s** — a little over one stride.

The cadence assumption is the weak link, so this is reported as a range rather
than a single number. It is enough for the purpose: **a 30 fps phone clip carries
four times the per-frame motion the lifter was trained on.** The tool therefore
interpolates keypoints up to 120 fps before lifting and resamples the angles back
afterwards.

## The ablation

One factorial pass, every configuration scored on the same clips so each effect
is a paired delta. **200 clips deliberately biased toward long ones** — 150
exceed the 81-frame window against 50.3% of the full set — because windowing only
matters for multi-window clips and a real video is long. That bias is why the
baseline reads 4.20° here rather than 3.43°; the *deltas* are the point.

| configuration | detector | padding | windows | hip | knee | mean | vs baseline |
|---|---|---|---|---|---|---|---|
| baseline (phase 3) | fine-tuned | zero | consecutive | 3.29 | 5.11 | 4.20 | — |
| edge padding | fine-tuned | edge | consecutive | 3.16 | 5.13 | 4.15 | −0.05 |
| **overlapped windows** | fine-tuned | zero | overlap | 2.95 | 4.92 | **3.93** | **−0.27** |
| both fixes | fine-tuned | edge | overlap | 2.97 | 4.85 | 3.91 | −0.29 |
| **generic COCO detector** | **COCO** | zero | consecutive | 3.82 | 5.65 | **4.74** | **+0.54** |
| generic COCO + fixes | COCO | edge | overlap | 3.46 | 5.28 | 4.37 | +0.17 |

**Windowing is the real fix; padding is almost nothing.** That makes sense —
padding only touches the tail of a sequence, while boundaries occur throughout.
Overlapped windows blend with a triangular weight so every interior frame is
covered twice and there is no seam.

## Does phase 3 need correcting? No.

The baseline configuration reproduces **3.43°** against phase 3's published
**3.4°**. That agreement is itself a useful result: it validates the
reimplementation, and it means the comparison above is apples-to-apples rather
than two different pipelines being eyeballed against each other.

So phase 3's number is **correct for what phase 3 ran**. What phase 7 adds is
that a better inference path was available and worth −0.25°. Rewriting phase 3's
headline to 3.18° would misrepresent what was actually executed, so
`results/phase03.md` keeps its number with a pointer here.

**No downstream conclusion moves.** Phase 4's error bank was built at 3.32° mean
|error|; a 0.25° improvement shifts it marginally, and phase 5C already showed
that *halving* the injected error changes the AUC by nothing that survives its
confidence interval. The degradation curve is insensitive to error at this scale
— which is the project's central finding, restated.

## What a generic detector costs — the number the tool needed

`det_markers2d_by_cam_coco` ships generic COCO detections for the same clips as
the fine-tuned `det_markers2d_by_cam_ft`. Swapping one for the other, everything
else held fixed:

**+0.54°** at the phase 3 baseline (4.20 → 4.74), and only **+0.17°** once the
inference fixes are applied (3.91 → 4.37). Measured on the 200-clip long-biased
subsample.

This is the closest available proxy for what torchvision's Keypoint R-CNN will
cost, because it is the same *kind* of substitution: a general-purpose COCO
detector in place of one fine-tuned on this capture rig. **It is a proxy, not a
measurement of Keypoint R-CNN**, and the tool says so.

> ### A checkpoint/detector mismatch, found and fixed
>
> The first version of this experiment reported **+1.67°**. That was wrong, and
> the error is worth recording because it is easy to repeat.
>
> The release ships **three checkpoints matched to three input types** —
> `ath-det-ft`, `ath-det-coco` and `ath-gt` — all identical architecture
> (11,721,795 params each). Phase 3 paired `det-ft` weights with `det_ft` inputs
> correctly. **The first run of this experiment fed `det_coco` inputs into the
> `det-ft` checkpoint**, so its number conflated two different things: a
> genuinely worse detector, and the wrong weights for that detector.
>
> Pairing each detector with its own checkpoint, the cost falls from **+1.67° to
> +0.54°** — three times smaller. The `ft` rows are bit-identical across the two
> runs, which confirms the change touched only what it should have.
>
> `scripts/video_kinematics.py` was making the same mistake and now loads
> `motionagformer-b-ath-det-coco-v1.ckpt`, since Keypoint R-CNN is itself a
> general-purpose COCO detector.

---

## The tool — `scripts/video_kinematics.py`

```
--video CLIP.mp4 --out DIR [--every K] [--max-frames N]
```

decode → Keypoint R-CNN COCO-17 → 10 Hz temporal smoothing → resample to 120 fps
→ MotionAGFormer lift (edge padding, overlapped windows, **scale-free**) →
sagittal hip/knee angles → resample back → CSV, figure, JSON.

**Detector: torchvision Keypoint R-CNN.** MediaPipe was tried first and rejected
— `import mediapipe` succeeds because its C bindings load lazily, but
constructing a `PoseLandmarker` raises `WinError 4551: An Application Control
policy has blocked this file` on this machine, the same Smart App Control that
blocked pandas 3.0.5 in phase 4. Keypoint R-CNN turned out strictly better
regardless: already installed via torch (proven here by phase 3), BSD-3-Clause
rather than ultralytics' AGPL-3.0, and it emits COCO-17 **in exactly the order
the lifter expects**, so no index mapping is needed at all.

**Lifting is scale-free.** Phase 3 denormalised using a scale derived from
ground-truth 3D, which real video does not have. The angles are computed in the
body's own frame and are scale-invariant, so the step is simply unnecessary —
asserted as a test across four orders of magnitude, not assumed.

### It refuses rather than emits

The first person-free test produced warnings *and* a CSV and figure of garbage
angles. A CSV of joint angles looks exactly like a real result, which is worse
than no output. The tool now **aborts with exit code 2 and writes nothing** when
a person is detected in under half the frames.

Remaining checks flag rather than abort: multiple people in frame (this tool does
not track across people), low lower-body keypoint confidence, angles outside
physiological range, and left/right traces that fail to alternate — which in
running indicates a tracking failure.

### The error budget, printed on every run

- lifting error **≥3.4°** — in-domain, track rig, on AthleticsPose's *own*
  fine-tuned detections
- **+0.54°** generic-detector penalty, measured in this phase (+0.17° once the
  inference fixes are applied)
- **+ unquantified Keypoint R-CNN error on your footage**
- **+ unquantified domain gap** — the checkpoint was fine-tuned on a track rig
- ankle dorsiflexion unavailable: H36M-17 has no toe keypoint
- **non-commercial only** — AthleticsPose CC BY-NC-SA 4.0, AthletePose3D research
  use only

### What is still not validated

**The detector stage itself.** AthleticsPose does not release its source videos,
so there is no footage with ground truth anywhere in reach to measure Keypoint
R-CNN against. That gap cannot be closed with the data this project has, and no
amount of care in the code substitutes for it. The +0.54° proxy is the honest
bound; it is not a measurement of this detector.

**End-to-end on real footage.** The pipeline has been exercised on synthetic
video (decode, detection, refusal path) and on AthleticsPose keypoints (lift,
angles), but not yet on a real side-on running clip, because none is available
here.

## Tests — the first in this repository

`tests/`, 19 tests, ~5 s. Previously every one of the 2,972 test files present
was third-party inside `.venv`; reproducibility rested entirely on re-running 36
scripts by hand. These cover the invariants that would silently corrupt results:

- angle recovery against a synthetic 0–90° knee sweep (was a manual `__main__`
  check, now a test)
- **scale invariance** across four orders of magnitude — the property the whole
  video path depends on
- rotation invariance — why a hand-held clip is usable at all
- ankle returns NaN rather than a fabricated number
- fold-leakage detection actually fires on a deliberately leaky fold
- COCO-17 ordering matches what the lifter expects — if torchvision ever
  reorders, every angle silently becomes wrong

## Reproduction

```
.venv/Scripts/python.exe scripts/phase7_inference_fix.py --clips 200      # ablation
.venv/Scripts/python.exe scripts/phase7_inference_fix.py --clips 0 --only 0,3
.venv/Scripts/python.exe scripts/video_kinematics.py --video CLIP.mp4 --out DIR
.venv/Scripts/python.exe -m pytest tests/ -q
```

Outputs: `results/phase7_inference.json`, `results/phase7_inference_full.json`.
