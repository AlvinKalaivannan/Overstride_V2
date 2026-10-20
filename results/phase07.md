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

---

# Addendum — the ablation at full n (A2)

The table above ran on a **200-clip subsample deliberately weighted toward long
clips**, which was the right choice for exercising the multi-window path but left
an open caveat: `docs/REVIEW.md` §7.3 recorded the detector cost as "a proxy on a
biased subsample." This re-runs the identical factorial pass on **every held-out
clip**, closing it.

```
.venv/Scripts/python.exe scripts/phase7_inference_fix.py --clips 0
-> results/phase7_inference_full.json
```

**592 clips**, 298 of which exceed the 81-frame window (50.3%, against 75% in the
subsample — that skew is exactly what was being corrected for).

| configuration | detector | padding | windows | hip | knee | mean | vs baseline |
|---|---|---|---|---|---|---|---|
| baseline (phase 3) | fine-tuned | zero | consecutive | 2.82 | 4.04 | **3.43** | — |
| edge padding | fine-tuned | edge | consecutive | 2.68 | 3.93 | 3.30 | −0.13 |
| overlapped windows | fine-tuned | zero | overlap | 2.62 | 3.93 | 3.27 | −0.15 |
| **both fixes** | fine-tuned | edge | overlap | 2.58 | 3.77 | **3.18** | **−0.25** |
| **generic COCO detector** | **COCO** | zero | consecutive | 3.19 | 4.56 | **3.88** | **+0.45** |
| generic COCO + fixes | COCO | edge | overlap | 2.85 | 4.17 | 3.51 | +0.08 |

## What changed, and what did not

| quantity | 200-clip subsample | **592 clips (full)** |
|---|---|---|
| baseline MAE | 4.20° | **3.43°** |
| effect of the inference fixes | −0.29° | **−0.25°** |
| generic-detector cost | +0.54° | **+0.45°** |

**The absolute error moved a lot; the deltas barely moved.** The subsample
overstated baseline MAE by 0.77° (22%) because long clips are harder, exactly as
the original report anticipated. Both deltas held to within 0.09°, so the
subsample's conclusions were sound — they are now measured rather than inferred.

**Precisely which caveat this closes.** The fix effect had *already* been run at
full n on two configurations (`--clips 0 --only 0,3`); only the **detector cost**
was subsample-bound, which is what `docs/REVIEW.md` §7.3 recorded. This pass adds
the missing four configurations. The two that overlap reproduce **bit-identically**
— baseline 3.428984°, both-fixes 3.176139°, difference 0.00e+00 — which is a
determinism check on the whole ablation, not just a repeat.

**The published figure is the full-n one from here on.** The generic-detector
penalty in the error budget is **+0.45°**, and `scripts/video_kinematics.py`
prefers `phase7_inference_full.json` when it exists so the tool cannot quote the
subsample. The 200-clip file is kept as the record of what was run.

## The baseline reproduces phase 3 exactly

Full-n baseline: **3.43°** against phase 3's published **3.4°**. The subsample
was the outlier, not phase 3 — which is worth stating, because the 4.20° in the
table above could otherwise be misread as phase 3 being optimistic. It was not.

## Capture rate

Recovered from ankle-vertical zero crossings on 33 clips: **113–136 fps** for a
running cadence of 2.5–3.0 strides/s, bracketing 120. Worth noting alongside the
validation plan's rev 3, which reports AthletePose3D as natively 120 fps — that
would turn this inference into a direct check, on a dataset where the rate is
documented rather than recovered. §7.4 is not closed by this addendum; it is only
made consistent.

**n = 33 is small** for the fps estimate and the cadence assumption is still the
weak link. The claim remains a range, not a number.

---

## Update, 2026-10-19 — the baseline with intervals, and the far limb

**Nothing above this line changed.** The ablation was re-run over the same 592
clips to emit per-clip rows, and every mean reproduced **bit-identically** to six
decimal places, `capture_fps` included. The re-run adds two things the original
could not supply, both of which the phase 13 gate asks for by name.

### 1. The far limb, which was never measured

The harness scored only the limb nearer the camera. The measurement track
forbids pooling the two, and the phase 13 gate asks the far limb to "improve or
stay within its CI" — against a figure that did not exist. It does now:

| configuration | detector | near mean | far mean |
|---|---|---|---|
| baseline (phase 3) | fine-tuned | 3.4290° | **3.3066°** |
| edge padding | fine-tuned | 3.3029° | **3.0166°** |
| overlapped windows | fine-tuned | 3.2741° | **3.1917°** |
| both fixes | fine-tuned | 3.1761° | **2.9259°** |
| generic COCO | COCO | 3.8765° | **3.9518°** |
| **generic COCO + fixes (deployable)** | COCO | **3.5093°** | **3.6177°** |

**The sign of the near/far gap flips with the detector.** On fine-tuned
detections the far limb is *better* than the near limb, by 0.12° at baseline and
0.25° with both fixes. On generic COCO detections it is *worse*, by 0.08°
without the fixes and 0.11° with them. Phase 3B measured the occlusion penalty
as approximately zero and phase 11 measured error *falling* as the view becomes
side-on; both of those used the fine-tuned path, and this says the far limb's
apparent immunity does not survive a generic detector. That is a detector
property, not a geometry property: the far limb is where a weaker 2D detector's
keypoints are least certain.

It also means the deployable configuration's honest headline is **two numbers,
3.51° near and 3.62° far**, and the far one is the larger.

### 2. Intervals, and what they say about what is measurable

`scripts/phase7_baseline_ci.py` re-aggregates the rows: per-subject means first,
then a percentile interval over 2,000 **subject** bootstraps
(`subject_bootstrap_ci`, seed 11). Clips are not independent — 8 cameras share a
trial — so resampling clips would understate the interval badly.

AthleticsPose holds out **three** subjects (S11, S13, S16), so these intervals
rest on three numbers and are reported as such. The subject-level mean also
differs slightly from the published pooled clip mean, because subjects contribute
unequal numbers of clips; both are in `results/phase7_baseline_ci.json`.

Deployable configuration, subject-level:

| limb | hip | knee | mean | 95% CI | pooled |
|---|---|---|---|---|---|
| near | 2.83° | 4.07° | 3.45° | [3.27, 3.69] | 3.51° |
| far | 3.02° | 4.10° | 3.56° | [3.13, 3.90] | 3.62° |

**The marginal intervals overlap for every configuration**, which is the expected
result at n=3 and the reason the gate must be read on a *paired* comparison:
every configuration is scored on the same clips, so the difference can be taken
per subject before the interval is formed.

Paired against the deployable configuration, negative meaning better:

| comparison | limb | delta | 95% CI | |
|---|---|---|---|---|
| the two inference fixes, on the COCO path (COCO vs COCO + fixes) | near | **+0.43°** | [+0.19, +0.74] | excludes 0 |
| | far | **+0.42°** | [+0.23, +0.80] | excludes 0 |
| fine-tuned detections + both fixes | near | −0.36° | [−0.70, +0.04] | spans 0 |
| | far | −0.69° | [−0.80, −0.52] | excludes 0 |
| fine-tuned detections, phase 3 config | near | −0.09° | [−0.24, +0.09] | spans 0 |
| | far | −0.24° | [−0.53, −0.04] | excludes 0 |

### 3. What this fixes about the phase 13 gate

Three consequences, and the third is the one that matters most for planning.

1. **The gate now has the interval it demanded.** It can be stated against
   3.5093° near / 3.6177° far, paired, per limb.
2. **The far limb is the harder half on the deployable path**, so a method
   reported as a single pooled improvement is hiding the half that needs it.
3. **There is a resolution floor, and it is about 0.2°.** The smallest paired
   difference whose interval excludes zero anywhere in this table is 0.24°
   (far limb); on the **near** limb nothing below 0.43° separates. A phase 13
   change worth 0.1° is therefore unmeasurable here no matter how real it is —
   not because the effect is absent, but because three subjects cannot resolve
   it. Any phase 13 arm should be budgeted against that floor before it is
   built, and the detector replacement (+0.43° near, +0.42° far, both excluding
   zero) is the one candidate already known to sit above it.

### 4. A joint-ordering bug in the tool, found 2026-10-19

While looking for a way to bound what a better 2D detector could buy, the
released 2D files turned out not to be in the convention the tool assumed.

`detect_coco17`'s docstring claimed torchvision's COCO-17 order "is already
exactly what the lifter expects". **It is not.** The released
`det_markers2d_by_cam_{coco,ft}` and `gt_markers2d_by_cam` arrays — the only
2D the checkpoint has ever been fed, and the source of every figure in this
report — are in **H36M-17** order. Three independent checks, all on the files
themselves rather than on documentation:

| check | result |
|---|---|
| lowest joints in the image (max mean *y*) | **3 and 6** — the H36M ankles. In COCO those indices are an ear and a shoulder |
| highest joints (min mean *y*) | **10, 9, 8** — H36M head, neck, thorax |
| per-joint distance, detections vs 2D ground truth | **3–28 px for all 17 joints**. A permuted order would put hundreds of pixels on most of them |
| upstream code | `datasets/dynamic_dataset.py.__getitem__` loads the array and feeds it to the model **with no reordering**, so the file order *is* the input convention |

That last table row also incidentally confirms the hip definition phase 12
found: the four largest 2D errors are joints 0, 1, 4 and 7 — pelvis, both hips
and spine, 24–28 px — while knees and ankles sit at 3–5 px. The hips are
*defined* differently, they are not badly detected.

**What was affected, and what was not.**

- **No published number moves.** Phases 3, 3B, 7, 11 and 12 all read the released
  H36M-ordered files directly. 3.4290°, 3.5093°, 3.1761°, the viewpoint curves
  and the phase 12 scan are untouched.
- **The video tool's output was invalid.** `scripts/video_kinematics.py` passed
  torchvision's COCO-ordered keypoints straight into the lifter, so the model
  received a nose where it expected a pelvis. Anyone running it on their own
  footage got angles computed from a scrambled skeleton. Fixed by
  `coco17_to_h36m17`, with pelvis, spine, thorax and neck constructed the
  standard way from mid-hip, mid-shoulder and the nose.
- **§7.5 is the reason this was survivable.** The tool has never been run on
  real footage, so nothing was ever reported from it. The one claim that must be
  withdrawn is implicit: this report's "the pipeline has been exercised on
  synthetic video ... and on AthleticsPose keypoints" is exactly the shape of
  the hole. Each half was exercised; **the seam between them was not**, and the
  bug lived in the seam.

**Why the guards did not catch it.** The tool checks hip ∈ [−40°, 60°] and knee
∈ [−170°, 20°] and refuses to write output when it cannot find a subject. A
scrambled skeleton still produces angles inside a plausible range, because the
lifter is a learned prior that returns a human-shaped pose for almost any input.
Plausibility is not correctness, and a range check cannot tell the difference.
The test that should have caught it asserted the wrong half of the contract —
that *torchvision* emits COCO-17 — which was true and irrelevant. It has been
replaced by one that reads the lifter's input convention **out of the released
data**, the same principle as the `MODEL_FPS` test: never pin a second constant
beside the first, read it from the thing it describes.

### 5. What a better 2D detector is actually worth, 2026-10-19

The approved plan called the detector replacement "the single highest-value
change" on the strength of a +0.43° paired delta, a 56–66× speed gap and an
unquantified error term. The accuracy half of that argument had never been
bounded: if the lifter's own error with a *perfect* detector were already 3.4°,
replacing the detector could buy 0.1° and no more.

AthleticsPose ships `gt_markers2d_by_cam` — mocap markers projected into each
camera, in the same H36M-17 convention as the detections — and
`motionagformer-b-ath-gt-v1.ckpt`, trained on that input. So the bound is
measurable with data already on disk and no video.
`scripts/phase7_detector_headroom.py`, all 592 held-out clips:

| input 2D | checkpoint | 2D error (px) | near | far | vs deployable |
|---|---|---|---|---|---|
| COCO detections | det-coco | 12.28 | **3.509°** | **3.618°** | — |
| blend, half the 2D error | det-coco | 6.14 | 4.332° | 4.264° | **+0.823°** |
| blend, quarter | det-coco | 3.07 | 4.680° | 4.628° | **+1.171°** |
| ground-truth 2D | det-coco | 0.00 | 5.163° | 5.107° | **+1.653°** |
| **ground-truth 2D** | **gt (matched)** | 0.00 | **1.468°** | **1.626°** | **−2.041°** |

The deployable row reproduces 3.509° / 3.618° exactly, which is the script
agreeing with `results/phase7_baseline_ci.json` on an independent code path.

**Finding 1 — the checkpoint has learned its detector's error distribution, and
cleaning the input without changing the weights is monotonically worse.**
3.509° → 4.332° → 4.680° → 5.163° as the 2D is moved toward truth with the
`det-coco` weights fixed. A *perfect* detector handed to the shipped checkpoint
costs **+1.65°**. Phase 7 already documented pairing input and weights wrongly
as a mistake; this is the same lesson measured in the other direction, and it
means **a detector swap is not a drop-in change.** It requires a matched lifter,
or it makes the tool worse.

**Finding 2 — the 2.04° gap is mostly not a detector problem, and mean pixel
error is the wrong statistic.** The fine-tuned detector is worth 0.33° of angle
error (3.509° → 3.176°, phase 7 §1) while having **the same mean 2D error**:

| joint | COCO det | fine-tuned det |
|---|---|---|
| pelvis | 20.5 px | 22.7 px |
| r_hip / l_hip | 21.2 / 21.2 | **22.8 / 23.0** |
| r_knee / l_knee | 5.4 / 5.8 | **3.6 / 4.1** |
| r_ankle / l_ankle | 6.4 / 6.4 | **3.2 / 3.4** |
| thorax | 11.3 | 14.9 |
| **mean over the eight** | **12.28** | **12.18** |

Two detectors, the same mean, 0.33° apart. Fine-tuning did not reduce the error,
it **moved** it: tighter on the distal joints that set knee and ankle angles,
looser on the hips. And the hips sit at 21–23 px for *both* — including a
detector fine-tuned on this very mocap's 2D. That residual is **definitional**:
H36M's hip is the mean of the ASI and PSI markers, a point inside the pelvis,
while any image-based detector marks a hip on the body's surface. Phase 12
measured the same offset in 3D — 81 mm, +18.2% femur inflation — and §4 above
found it again as the four largest per-joint 2D errors.

**So the plan's step 2 needs restating, not abandoning.**

- The accuracy prize from a better detector is **not 2.04°**. That bound is
  contaminated by a convention no detector can satisfy without predicting mocap
  marker positions.
- The realistic prize is the measured **0.33°** between a generic COCO detector
  and one fine-tuned on this dataset — above the 0.2° resolution floor, and
  worth having, but modest.
- Most of it is available by being accurate **at the knees and ankles
  specifically**. A candidate detector should be judged on distal-joint pixel
  error, not on mean pixel error or COCO AP.
- **The speed argument is untouched** and is now the main reason to replace the
  detector: 2,790–3,280 ms/frame against phase 16's 50 ms budget. Accuracy is a
  secondary, bounded benefit that arrives only with a matched lifter.

**One hypothesis this falsified.** The blend was built to give a sensitivity
curve — 2D pixel error against angle error — so that a candidate detector could
be placed on it from its 2D error alone, without the video AthleticsPose does not
release. It cannot: the blend curve measures out-of-distribution degradation of a
fixed checkpoint, which runs the *opposite* way to detector quality. Placing a
candidate detector still requires either a matched checkpoint or a dataset with
video and 3D ground truth, which is phase 15's job.

### 6. The 20 fps gate is unreachable with these weights, 2026-10-19

`scripts/phase16_detector_speed.py` holds `keypointrcnn_resnet50_fpn` and its
weights fixed and varies only **how it is called**, so everything it finds is
free: no retraining, no distribution shift, no new dependency. Synthetic frames,
16 threads, 6 frames per configuration.

| configuration | ms/frame | fps | over the detector's 39 ms share |
|---|---|---|---|
| **current path, 640×480** (`min_size=800`) | **1,055** | 0.95 | **27×** |
| current path, 1280×720 | 1,252 | 0.80 | 32× |
| native resolution (`min_size=480`) | 464 | 2.15 | 12× |
| half resolution (`min_size=320`) | 332 | 3.01 | 8.6× |
| tight person crop, 192×256 | 248 | 4.03 | 6.4× |
| crop + batch of 4 | 238 | 4.20 | 6.1× |
| **crop, 16 threads** | **207** | **4.84** | **5.3×** |

**A 5.1× speed-up is sitting in the call path.** The single largest item is that
torchvision resizes every frame so its shortest side is `min_size=800` before
the backbone sees it: a 640×480 frame is silently upscaled to 1067×800, 2.8× the
pixels, and the tool never overrode it. Running at native resolution is 2.3× on
its own; a tight person crop — the standard top-down pose pipeline, box found
once and tracked — takes it to 5.1×.

**And it is still 5.3× short.** 207 ms against the ~39 ms the detector may have
if the 11.2 ms lifter is to fit inside a 50 ms frame. A two-stage ResNet50
Keypoint R-CNN does not reach 39 ms on a laptop CPU; ONNX Runtime and int8
quantisation might find another 2–3×, which still leaves it short, and int8 would
perturb the keypoints that §5 showed the lifter is sensitive to.

**So phase 16's pre-registered ≥ 20 fps gate cannot be met without changing the
architecture, and §5 showed changing the architecture costs up to +1.65° unless
the lifter is retrained to match it.** Those two facts together are a real
decision, not an implementation detail, and the decision is recorded in
`CLAUDE.md` rather than taken quietly here.

**A measurement discrepancy, recorded rather than reconciled away.** The figures
this track has been planning against — 2,790 ms at 640×480 and 3,280 ms at
1280×720 — were timed ad hoc in an earlier session. This script measures 1,055
and 1,252 ms for the same two cases. The ratio between the two resolutions
agrees (1.19 against 1.18), so the *shape* reproduces and only the level differs;
the likely cause is background load on the laptop at the time, since both runs
used 16 threads on the same machine. **The 1,055 ms figure supersedes the
2,790 ms one**, because it comes from a script in the repository that anyone can
re-run, and because the relative comparisons that matter here are internal to a
single run. The conclusion is unchanged either way: 27× or 56× over, the gate
needs a different architecture.
