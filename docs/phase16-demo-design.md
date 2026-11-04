# Phase 16 — demo design

> **Design document, 2026-11-04. Step 1 of phase 16: no app code is written
> yet.** Every number here is measured and carries its source file and the
> conditions it was measured under. Where a condition was never measured, this
> document says so instead of interpolating.

---

## 1. What it is

**A local, offline analyzer.** A side-on running clip goes in. Out come
**per-frame and stride-averaged knee flexion and gravity-referenced thigh
inclination, for both legs**, drawn on a skeleton overlay, next to a stated
error budget.

It runs **nothing but the frozen pipeline the sealed test measured**:

| stage | component | pinned as |
|---|---|---|
| decode | OpenCV / imageio | — |
| detect 2D | `keypointrcnn_resnet50_fpn`, generic COCO weights, native resolution, padded person crop | `video_kinematics.detect_coco17` |
| reorder | COCO-17 → H36M-17 | `video_kinematics.coco17_to_h36m17` |
| confidence | overwrite to `TRAIN_CONF_MEAN = 0.8676` | `video_kinematics.scores_to_train_scale` |
| lift | MotionAGFormer-B `motionagformer-b-ath-det-coco-v1.ckpt`, 81-frame windows, `pad="edge"`, `windows="overlap"` | `phase7_inference_fix.lift` |
| angles | knee + gravity-referenced thigh | `phase15d_segments.segment_angles` |

**No component swaps.** The error budget in §3 is the measured behaviour of
*this* combination. Phase 7 §5 showed the lifter has learned its detector's
error distribution, so handing it cleaner-than-trained 2D is monotonically
**worse** — 3.51° → 4.33° → 4.68° → 5.16° as the input moves toward truth — and
phase 15F reproduced the same effect a third time, with ground-truth 2D scoring
**2.5° worse than real detections** on the hip over 589 clips. **A different
detector or lifter does not make the budget uncertain; it makes it void.** If
the detector is ever replaced, the lifter must be retrained to match it and
every number in §3 must be re-measured.

### Real-time is out of scope, and the gate is recorded as failed

**Phase 16's pre-registered `>= 20 fps` gate is NOT MET.** It is not being
lowered, redefined or quietly dropped.

| stage | measured | source |
|---|---|---|
| detector, as the tool now calls it (crop, native resolution, 16 threads) | **207 ms/frame** (4.84 fps) | `results/phase16_detector_speed.json`, `results/phase07.md` §6 |
| detector, before the call-path fix (`min_size=800`, 640×480) | 1,055 ms/frame (0.95 fps) | same |
| lifter | **11.2 ms/frame** | same |
| **total** | **~218 ms/frame, ~4.6 fps** | sum |
| budget for 20 fps | 50 ms/frame | pre-registered |
| **shortfall** | **4.4× over** | — |

The 5.1× already recovered came entirely from *how the detector is called*, with
the same weights — no retraining and no shift in the keypoint distribution the
lifter was trained against. The remaining 4.4× needs a different architecture,
and §5 of phase 7 measured what that costs without a matched lifter. **So the
demo is an offline analyzer, by decision and on the record.**

---

## 2. What it shows, and what it does not

### PRIMARY — shown as measurements

**Knee flexion** and **gravity-referenced thigh inclination**, both legs,
near and far limb labelled separately.

These two are primary because phases 15C and 15D established that they are the
quantities that **mean the same thing in two different labs**:

- the **knee** is a difference of two angles inside one frame, so it is
  invariant to rotating that frame — structurally immune to the thorax
  definition, and **24 of 24 rows were bit-identical across thorax definitions**
  in phase 15C;
- **thigh inclination against gravity** needs no body-frame convention at all,
  and it is the only quantity in the project with **no measurable cross-lab
  loss**: −0.345°, CI **[−1.885, +1.219]** spanning zero at n = 11
  (`results/phase15f.md`), confirming the sealed test's near-limb finding on
  nearly four times the subjects.

**Near and far limb are always shown separately, never averaged.** The sealed
test measured a far-limb penalty with an interval excluding zero and the same
sign on all five subjects — **thigh +1.971° [+0.977, +3.187]** — so an average
would hide a real asymmetry.

### EXPERIMENTAL — shown, labelled, with its bias stated

**Trunk lean.** Shown in a separately headed, visually de-emphasised panel
marked **EXPERIMENTAL**, carrying this text:

> Trunk lean carries a measured **−2.34° bias** on side-on lab footage
> (`results/phase15_sealed.md`, 5 never-opened subjects) and is **+1.92°
> [+0.583, +3.204] worse** across labs than on the tune set
> (`results/phase15f.md`, n = 11). It is also the worst-tracked quantity in the
> pipeline at any frame rate: **r = +0.764** against **+0.99** for the knee and
> thigh (`results/phase15e.md`). Read it as a direction, not a number.

Two further facts that keep it out of the primary: phase 15C found the lifter's
thorax lands **87–144 mm** from `CLAV` on MoVi while sitting within **5.04 mm**
of it on AthleticsPose — and trunk lean *is* that axis; and phase 15E's finding
that trunk lean *improves* at lower frame rates **failed to replicate on sprint
at three times the subjects**, so even its rate behaviour is unsettled.

### NOT SHOWN as a measurement — trunk-referenced hip

**The hip flexion angle is computed internally but is not displayed as a
measurement**, because its reference axis does not transfer across labs: phase
15C showed the lifter does not put its own thorax on the same anatomical point
in two datasets, and phase 15D showed the consequence — the hip degrades
**3.90×** cross-lab against the knee's **2.00×**, and the ordering of which
joint is harder **inverts between labs**.

Phase 15F sharpened this into a one-line reason the UI can carry:

> An H36M-trained lifter that has never seen running, this lab or this camera
> **beats** this pipeline on the hip by **1.69–1.84°** on held-out-lab footage
> (paired, n = 11, both CIs excluding zero) while being 5× worse on the tune
> set. **So the hip error is a learned convention, not a measurement limit** —
> and a number whose error is a convention is not a measurement.

A collapsed "diagnostics" section may show the hip trace **labelled "internal,
not a measurement"**, because it is needed to explain the identity
`hip bias = thigh bias + trunk-lean bias` if anyone asks. It is never in the
primary readout, never in an export, and never in a summary table.

### No injury output of any kind

Per `CLAUDE.md`'s measurement-track rules, which phases 4B and 5D settled:

- **no injury score, flag, colour, gauge, or "consult a professional" nudge**
  driven by the kinematics;
- the word **"risk"** does not appear in any output, UI string, docstring, plot
  title, filename or export of this demo;
- the word **"predict"** does not appear either — the vocabulary is *measure*,
  *detect*, *classify*, *screen*;
- **no asymmetry verdict.** Left-right differences are *displayed as two
  numbers*, never reduced to a single "asymmetry index", because phase 5's limb
  task topped out at AUC 0.610 against a 0.517 base rate and phase 4B found a
  per-user verdict unsupportable (PPV 0.574, calibration slope 0.716, disagreeing
  with itself on a third of repeat scans);
- **no normative comparison.** AddBiomechanics ranges are plausibility bounds
  inside the code, never an on-screen "normal / abnormal" band.

A CI check greps the UI strings, templates and exports for the banned
vocabulary; see milestone M6.

---

## 3. The error budget

**Every row is measured. Each names its source file and its conditions.** The
app shows the row matching the clip's own frame rate and says plainly when a
condition was not measured.

### Row A — tune set, in-domain, 100 Hz capture

`results/phase15.md`, `results/phase15d.md`, reproduced by `results/phase15e.md`'s
native arm to within 0.002–0.02°. AthleticsPose, S11/S13/S16, running, generic
COCO detections, 589 clips (231 for the gravity-referenced thigh).

| quantity | limb | MAE | 95% CI |
|---|---|---|---|
| knee flexion | near | **4.03–4.07°** | [3.836, 4.459] |
| knee flexion | far | 4.08° | — |
| thigh inclination | near | **6.53°** | [6.291, 6.760] |
| thigh inclination | far | 6.82° | [5.834, 8.142] |
| *(trunk lean, experimental)* | both | 5.53° | — |

**Conditions:** a calibrated 8-camera lab, athletes, 100 Hz, the dataset this
checkpoint was fine-tuned on. **This is the floor, not what a user should
expect.**

### Row B — held-out lab footage at 30 fps. The honest headline.

`results/phase15_sealed.md`, the sealed test: **5 subjects never opened**,
pipeline pinned by content hash, pre-registration pushed before a single byte
was downloaded, run once. MoVi PG1, 800×600, **30.000 fps**, side-on
(`view_ratio` 0.54–0.70), jogging.

| quantity | limb | MAE | 95% CI | bias |
|---|---|---|---|---|
| **knee flexion** | near | **8.41°** | [6.275, 10.975] | +5.45° |
| **knee flexion** | far | **8.69°** | [6.552, 10.679] | +6.03° |
| **thigh inclination** | near | **5.40°** | [4.136, 6.837] | −3.72° |
| **thigh inclination** | far | **7.37°** | [5.456, 9.258] | −6.00° |
| *(trunk lean, experimental)* | near | 5.16° | [4.622, 5.769] | −2.34° |

Corroborated at **n = 11** on all spent MoVi subjects (`results/phase15f.md`,
diagnostic): knee near 8.68°, thigh near 5.69°, trunk near 5.93°.

**Two things the user must be told alongside this row.** The error is largely a
**per-subject offset** — one sealed subject carried a +0.56° knee bias and
another +12.11° — but it is **not** a removable offset: the within-subject SD is
**4.5–9.7°**, and even an oracle per-subject correction would cut MAE by only
17% (knee) and 11% (thigh), leaving 4.8–7.0°
(`results/phase15_sealed.md`, Correction 2026-11-03). And the **shape is tracked
well** — knee r **+0.77 to +0.92** — so the trace's *form* is more trustworthy
than its absolute level.

### Row C — frame-rate penalties, measured on the tune set

`results/phase15e.md` (33.3 and 30.0 fps) and `results/phase15f.md` (50.0 fps).
AthleticsPose, generic COCO, **paired per subject**, native 100 Hz reference.
**All resolved** — every interval excludes zero.

| clip fps | knee near | knee far | thigh near | thigh far | *(trunk)* |
|---|---|---|---|---|---|
| ≥ 100 Hz | 0 (reference) | 0 | 0 | 0 | 0 |
| **50.0 fps** | **+0.60°** | +0.87° | **+0.06°** | +0.29° | −0.39° |
| 33.3 fps | +1.51° | +1.60° | +0.31° | +0.46° | −0.17° |
| **30.0 fps** | **+1.84°** | +1.76° | **+0.38°** | +0.47° | −0.20° |

**The curve steepens below 50 fps rather than flattening**: 67% of the knee's
30 fps penalty and 84% of the near thigh's is incurred between 50 and 30 fps.
That is the whole reason §4 warns below 50.

### What the app displays

For a clip at frame rate *f*, the app shows **Row B as the expected error** —
because Row B is the only row measured on footage the pipeline had never seen —
and then the **Row C adjustment for the nearest measured rate at or below *f***,
labelled with the rate actually used.

### Conditions that were NOT measured, and are stated as such

The app must print these verbatim rather than interpolate:

| condition | status |
|---|---|
| **60 fps, anywhere** | **UNMEASURED.** 100/60 is not an integer, so no clean 60 fps decimation exists on the tune set and none was run. The nearest measured rate is **50 fps**, and that is the row shown for a 60 fps clip, labelled "50 fps (nearest measured)". |
| **any rate above 30 fps on held-out lab footage** | **UNMEASURED.** MoVi PG1 is 30.000 fps; there is no higher-rate held-out footage. Row C's penalties are measured on the **tune set only** and their transfer to another lab is an assumption, not a measurement. |
| **phone / consumer footage, at any rate** | **UNMEASURED, and this is the project's largest open gap** (`docs/REVIEW.md` §7.5). MoVi's iPhone cameras ship no calibration and their sync cannot be verified, so the pipeline **refuses** to score them. The demo's own clips are the first phone footage this tool will ever process, and **no error figure applies to them.** The UI says so. |
| **running faster than jogging, outdoors, on a treadmill, in low light** | UNMEASURED. |

**Subtracting Row C from Row B to produce a "60 fps expected error" is
forbidden** — Row C is tune-set-measured and Row B is a different lab; phase
15E's share-of-loss arithmetic was explicitly labelled "if it transferred
unchanged" and three of its eight cells came out undefined.

---

## 4. Input checks, before any processing

Run in order. Each either passes, warns (processing continues, the warning is
carried into the report) or refuses (nothing is processed and no numbers are
emitted). **Refusing is the existing behaviour of the tool** — `video_kinematics`
already refuses to write output when it cannot find the subject — and this
extends it.

| # | check | refuse | warn | source |
|---|---|---|---|---|
| 1 | **frame rate** | **< 25 fps** | **< 50 fps** | Row C: the penalty curve steepens below 50 fps, and below 25 the behaviour is off the measured curve entirely |
| 2 | **resolution** | shortest side < 360 px | shortest side < 480 px | the detector runs at native resolution; MoVi PG1 is 800×600 and is the only resolution with a measured error |
| 3 | **clip length** | < 3 s | < 8 s | the lifter's window is 81 frames; at 30 fps a 3 s clip is 90 frames, barely one window, and short clips are mostly edge padding (phase 15F measured a 52%-padding handicap costing a checkpoint 20°+) |
| 4 | **person found** | no person in > 20% of frames, or more than one person in any frame | any frame without a person | the sealed test detected 100% of frames on 11 of 11 subjects with zero re-detections, so a miss rate is an input problem |
| 5 | **viewpoint** | `view_ratio` < 0.35 | `view_ratio` < 0.55 | below |

### The viewpoint check, in detail

**`view_ratio` is estimated from the detected 2D alone** — apparent 2D hip
separation over apparent trunk length, against a population ratio — which phase
14 measured at **Pearson 0.974 / Spearman 0.864** against the ground-truth depth
metric, with no depth, no millimetres and no ground truth required. The near
limb is also identifiable from the estimate alone, **592 of 592 clips**.

The thresholds come from measured viewpoints, not taste:

| reference | `view_ratio` | source |
|---|---|---|
| AthleticsPose, side-on lab | median **0.751** | phase 3B |
| MoVi PG1, side-on, where Row B was measured | **0.675** [0.536, 0.720] | `results/phase15_standing_viewpoint.json` |
| MoVi PG2, near-frontal | 0.44–0.61 | `results/phase15_checks.json` |
| MoVi standing, frontal | **0.057** | `results/phase15_standing_viewpoint.json` |

And the cost of getting it wrong is measured: phase 11 found sagittal error
falling **3.50° → 1.77°** monotonically as the view becomes side-on, on an
independent lab's four calibrated cameras. **Row B was measured at 0.54–0.70, so
a clip below 0.55 is outside the conditions the budget describes** — hence warn
— and below 0.35 it is nearer frontal than side-on, where the measured error is
twice as large — hence refuse.

### Viewpoint quality, not degrees

Phase 14's confidence is reported as **"viewpoint quality"** on a coarse
ordinal scale (**good / fair / poor**), derived from the estimated `view_ratio`,
**never as a number of degrees**. The reasons are measured:

- it is **a viewpoint meter**: camera azimuth carries essentially all the
  signal, while keypoint score, segment-length deviation, left/right asymmetry
  and window blend weight are all within noise of zero — and the detector's own
  confidence is *anti*-correlated with thigh error;
- **~4/5 of its ranking skill is between clips, not between frames**, per
  phase 14's pre-registered within-clip permutation control, so it ranks
  **clips**, which is exactly how the demo uses it;
- it is **not calibrated**: only 1 of 4 segments had a calibration slope in
  [0.8, 1.2];
- and the segments it *can* rank are **both shanks** (ρ 0.34, every subject) —
  **not the thighs** (ρ 0.09–0.17, bar not met). **The demo's primary quantities
  are knee and thigh, so per-segment confidence is not shown at all.** Phase 14's
  own closing guidance: *"It may not claim per-segment confidence on the
  thighs."*

So: **one clip-level viewpoint-quality badge, ordinal, no degrees, no
per-segment confidence.**

---

## 5. Architecture

### Recommendation: a CLI that writes a self-contained HTML report. No server.

The brief offered FastAPI plus a single-page UI, or a simpler option with
justification. **The simpler option is right here, and the justification is the
measured processing time.**

```
overstride-analyze  clip.mp4  --out report.html
        |
        +-- 1. probe      fps, resolution, duration, frame count        (< 1 s)
        +-- 2. checks     §4 checks 1-3; refuse early, before decoding
        +-- 3. decode     frames to memory, one pass                    (~2 s)
        +-- 4. detect     video_kinematics.detect_coco17, crop-tracked  (207 ms/frame)
        +-- 5. checks     §4 checks 4-5 on the detected 2D; refuse here
        +-- 6. prepare    coco17_to_h36m17 -> scores_to_train_scale
        +-- 7. lift       resample -> 81-frame overlap windows -> 3D    (11.2 ms/frame)
        +-- 8. angles     phase15d_segments.segment_angles, EST_UP
        +-- 9. strides    heel-strike detection -> stride averages
        +-- 10. render     overlay frames + traces + budget -> ONE .html
```

**Why no server.** Processing a 10 s clip takes **minutes, not milliseconds**
(below), so there is nothing to be interactive about. A server would add a port,
a CORS policy, an upload path, multipart handling, a process lifecycle and a
"is it still running?" problem — all to deliver a document that could simply be
written to disk. It would also create the one thing this design most wants to
make structurally impossible: **a network interface.** A CLI with no sockets
cannot leak a frame, which matters because the regression test in §6 runs on
MoVi footage that may never leave `data/`.

**What the report is.** One self-contained `.html` file: inline CSS, inline
`<svg>` traces, and overlay frames as inline base64 PNGs. No CDN, no fonts
fetched, no JavaScript required to read it. It opens by double-clicking and can
be archived as a single file. Charts are drawn server-side-style into SVG by the
CLI, so there is no chart library to pin.

**When a server would become justified:** drag-and-drop for non-technical users,
a live progress bar, or batch queueing. None is needed for step 1, and adding it
later does not change stages 1–9 — the renderer is the only thing that moves.

### Hardware

Runs on this laptop's CPU. **No GPU anywhere**, per `CLAUDE.md`'s environment
rule. Dependencies are already installed and already used by the measured
pipeline: `torch` (CPU), `torchvision`, `numpy`, `scipy`, `opencv-python`. The
checkpoint is 141,862,608 bytes and loads in a few seconds. 16 threads, which is
the configuration the 207 ms/frame figure was measured at.

### Processing time for a 10 s, 60 fps clip — 600 frames

From the measured per-stage costs in §1:

| stage | arithmetic | time |
|---|---|---|
| probe + checks | — | < 1 s |
| decode 600 frames | — | ~2 s |
| **detect** | 600 × 207 ms | **124 s** |
| resample to the lifter's 100 Hz | 600 × 100/60 = 1,000 model-frames | < 1 s |
| **lift** | 1,000 × 11.2 ms | **11 s** |
| angles + strides | vectorised | < 1 s |
| render overlay (600 PNGs) | — | ~5 s |
| **total** | | **~145 s, about 2½ minutes** |

That is **~14× slower than real time**, and ~97% of it is the 2D detector.
Model load adds a few seconds once. A 15 s clip at 60 fps — the upper end of the
filming instructions — is **~3½ minutes**.

**Two speed-ups available without touching the measured pipeline**, because both
change only *how* the same weights are called: batching detector crops
(measured at 238 ms/frame in a batch of 4, so no help as measured) and
**processing every frame but rendering an overlay only every 3rd frame**, which
cuts the render but not the 124 s detector cost. **There is no honest way to make
this fast without replacing the detector, and that voids the budget.** The
design accepts 2½ minutes.

---

## 6. Correctness test

**The demo's processing path must reproduce a committed phase 15 number exactly,
on a real clip, locally.**

`tests/test_demo_regression.py` — marked `@pytest.mark.local`, **skipped
automatically when `data/movi/` is absent**, so the suite still passes on a
clean checkout and in CI.

```
Given   MoVi PG1 Subject_17, the jogging window, the frozen detector and
        checkpoint  (Subject_17 is SPENT -- sealed test, 2026-11-03)
When    the demo's processing path runs end to end
Then    knee flexion, near limb, MAE == 5.130 deg   (atol 1e-3)
  and   thigh inclination, near limb, MAE == 3.274 deg
  and   the near limb is identified as LEFT
  and   n frames == 203  and  detected frames == 221/221
```

Those four values are committed in `results/phase15_sealed.json` and quoted in
`results/phase15_sealed.md`. **`Subject_17` is chosen deliberately**: it is the
best-behaved sealed subject (knee bias +0.562°, thigh bias −0.052°), so a
regression shows up as a clear break rather than hiding inside a subject that is
already poor. A second case on `Subject_2` — the worst, knee bias +12.106° —
guards the other end.

**The test asserts the demo path, not the research script.** If the demo
refactors detection, resampling or windowing and the number moves, the test
fails and the demo is wrong — which is the entire point of pinning the pipeline
in §1. Phase 15F already demonstrated this works: a freshly written harness
reproduced phase 15D's committed `Subject_1` values (knee **6.950**, hip
**13.706**) exactly.

### MoVi containment — absolute

| rule | enforcement |
|---|---|
| MoVi video, frames, crops, overlays and renders **never leave `data/`** | the CLI writes only to the `--out` path; the test writes to `tmp_path` |
| **no MoVi byte ever enters the repository** | `data/` is in `.gitignore`; verified with `git check-ignore -v` before any commit |
| **no MoVi frame in any screenshot, README image, UI example or published artifact** | every screenshot is of the user's own filmed clip (§7) |
| **no per-subject MoVi measurement in a committed file** | the standard already applied in `8c407af`: the two scoping JSONs were cut from 267 KB to 9.8 KB with zero subject ids |
| the regression test prints **no MoVi pixel data and no per-subject anthropometry** | it asserts angles only |

MoVi's licence forbids distributing the Data in any form to any third party;
citing the dataset URL in place of the paper is itself non-compliant, so the
attribution in §8 cites the paper.

---

## 7. Demo footage — filmed by the user

**No dataset frame is ever shown.** Not MoVi, not AthleticsPose, not
AthletePose3D. Every screenshot, example and demo clip is footage filmed for
this project with the runner's consent.

### Filming instructions

**Camera**
1. **Side-on.** Point the camera at a right angle to the direction of travel.
   This is the single most important instruction: error roughly **halves**
   between a frontal and a side-on view (3.50° → 1.77°, phase 11), and the
   analyzer **refuses** a clip whose estimated `view_ratio` is below 0.35.
2. **Camera at hip height.** Roughly the runner's greater trochanter. The
   pipeline assumes a level camera; only the *in-plane* component of tilt
   affects a sagittal angle, and the lab camera Row B was measured on had a
   total tilt of **3.699°** contributing −1.4° to −1.9° of error. Shooting from
   chest or ground height adds tilt that is not in the budget.
3. **4–6 m away**, perpendicular distance. The lab camera was **4.4–4.6 m** from
   the subject. Closer adds perspective distortion; further shrinks the runner
   below the detector's useful size.
4. **Steady.** Tripod, or braced against something. Hand-held is acceptable but
   hand-held pans are not: the analyzer's crop tracker expects the runner to
   cross the frame, not the frame to follow the runner.

**Settings**
5. **60 fps if the phone offers it**, 50 fps at minimum. Below 50 the penalty
   curve steepens sharply (§3 Row C) and the analyzer warns; below 25 it
   refuses. **60 fps itself is unmeasured** — the app will show the 50 fps row —
   but more frames is never worse.
6. **Shortest side ≥ 720 px.** Do not use a "slow motion" mode that drops
   resolution.
7. Lock exposure and focus if the phone allows it.

**Scene**
8. **Plain, uncluttered background**, and **only the runner in frame.** The
   analyzer refuses a clip with more than one person in any frame.
9. Even lighting, no strong backlight, no shadow crossing the runner.
10. The runner wears **fitted clothing**; loose shorts over the thigh is the
    worst case for a thigh-inclination measurement.

**The run**
11. **10–15 s of steady running** at a constant, comfortable pace — not an
    acceleration, not a sprint start. Row B was measured on **jogging**.
12. The runner should cross the full frame, ideally more than once, giving
    several complete strides per pass.
13. Film **both directions** (left side to camera, then right side), so each leg
    gets a turn as the near limb — the far limb carries a measured **+1.97°**
    penalty on the thigh.

### What the report will say about this footage

> **This clip is consumer phone footage, and no measured error figure applies to
> it.** Every number in the error budget was measured on calibrated lab cameras
> (`results/phase15_sealed.md`, `results/phase15e.md`). The pipeline has never
> been scored against ground truth on phone video, because no dataset this
> project can access provides phone video with verifiable synchronised ground
> truth (`docs/REVIEW.md` §7.5). **Treat the budget as the best available
> estimate and not as this clip's accuracy.**

---

## 8. Licence — and an unresolved blocker for *publishing* the demo

### What is established

| artifact | licence | verified |
|---|---|---|
| **AthleticsPose Dataset** (`AthleticsPoseDataset.zip`) | **CC BY-NC-SA 4.0**, SPDX `CC-BY-NC-SA-4.0` | `data/athleticspose/data/LICENSE-DATASET`, full text in `LICENSES/CC-BY-NC-SA-4.0.txt` |
| AthleticsPose **code** | **none granted** | the repo's own README TODO list: `- [ ] Add license for codes`, unchecked |
| **the checkpoint** `motionagformer-b-ath-det-coco-v1.ckpt` | **not covered by any licence file on disk** | see below |

### The problem, stated plainly

**The dataset licence's own scope clause does not reach the checkpoint.**
Verbatim from `LICENSE-DATASET`:

> *"Scope: This license applies to all dataset contents
> (images/videos/annotations/metadata) in AthleticsPoseDataset.zip. Code in this
> repository may be licensed differently; see the repository's LICENSE for
> code."*

The checkpoint did not come from `AthleticsPoseDataset.zip` — it came from
`checkpoints.zip`, a separate release artifact (`Makefile`, `make checkpoints`).
There is **no `LICENSE` for code** in the repository, and no licence inside
`checkpoints.zip`. So the weights sit in a gap: arguably an *adaptation* of
CC BY-NC-SA material, for which ShareAlike and NonCommercial would carry
through; arguably unlicensed, in which case no distribution right was granted at
all.

**I cannot confirm the checkpoint's licence permits a public demo, and this
design does not assume it.**

### What this design does about it

1. **The demo is local and offline, and redistributes nothing.** The user already
   has the checkpoint under whatever terms they obtained it; the CLI loads it
   from `data/athleticspose/checkpoints/`. **No weights are served, bundled or
   uploaded**, so the ambiguity is not triggered. This is the main reason §5
   recommends a CLI over a hosted app.
2. **Non-commercial regardless.** `CLAUDE.md` already puts any commercial
   framing out of scope, and the NC term would bind even on the generous
   reading.
3. **Publishing the demo *with weights* is BLOCKED pending clarification.**
   Shipping a downloadable bundle, a hosted page, a container image or a
   release asset containing the checkpoint requires an explicit grant. **Action,
   not an assumption:** open an issue on `SZucchini/AthleticsPose` asking which
   licence covers `checkpoints.zip`. Until it is answered, the demo is
   distributed as **code plus instructions to obtain the checkpoint from
   upstream**, which is how this project already obtains it.
4. **Source code of the demo** is this project's own and carries this
   repository's licence.

### Attribution the UI must carry

Visible in the report footer and in `--version`, verbatim:

> **Pose model:** MotionAGFormer-B fine-tuned on the AthleticsPose Dataset.
> This work uses AthleticsPose Dataset (CC BY-NC-SA 4.0) by Suzuki et al.
> (https://arxiv.org/abs/2507.12905).
> Architecture: MotionAGFormer, Mehraban et al. (https://arxiv.org/abs/2310.16288).
> 2D keypoints: Keypoint R-CNN (torchvision), COCO-pretrained.
> **Non-commercial research use only.** Not a medical device. Provided "AS IS",
> without warranty.

The citation is the **paper**, not the dataset download URL — MoVi's licence
makes the URL-instead-of-paper substitution non-compliant, and the same courtesy
applies here. The BibTeX entry (`suzuki2025athleticsposeauthenticsportsmotion`)
goes in the demo's README.

**Validation scope:** the AthleticsPose licence carries no warranty and the
authors have not validated this pipeline. The UI says the error budget is this
project's own measurement, not the dataset authors' claim.

---

## 9. Milestones

Each has a test that must pass before the next begins. Estimates assume working
alone, and assume the measured pipeline is reused rather than reimplemented.

| # | milestone | deliverable | test | estimate |
|---|---|---|---|---|
| **M0** | **Design approved** | this document | approved by the operator; sections 2, 3, 5, 9 reviewed | done |
| **M1** | **Processing path extracted** | `overstride/analyze.py` — stages 1, 3–8 of §5, importing the frozen functions, no new maths | **the §6 regression test**: `Subject_17` reproduces knee 5.130°, thigh 3.274°, near limb LEFT, 221/221 detected, `atol 1e-3`; plus `Subject_2` at the other extreme | **1 day** |
| **M2** | **Input checks** | `overstride/checks.py` — the five checks in §4 with their measured thresholds, returning pass/warn/refuse | unit tests per check with synthetic inputs at each threshold boundary; a frontal-view fixture (`view_ratio` ≈ 0.06, from the measured standing geometry) must **refuse**; a 30 fps clip must **warn** and still process | **0.5 day** |
| **M3** | **Stride segmentation** | heel-strike detection → per-stride averages and a stride count | **this is new code and the only unmeasured component.** Test: on a sealed MoVi clip the detected stride count matches a manually counted ground truth within ±1, on 3 clips; and per-stride knee peaks have a CV below the across-subject spread. **If it fails, the demo ships per-frame traces only** — stride averaging is a convenience, not part of the budget | **1 day** |
| **M4** | **Error-budget renderer** | §3's three rows as data, not prose: a table keyed by quantity × limb × frame rate, with source file and conditions per cell | a test asserts every displayed number is present in the source JSON it cites (`phase15_sealed.json`, `phase15e.json`, `phase15f.json`) — **no number may be hardcoded without a matching source**; and that an unmeasured condition renders the word "UNMEASURED" | **0.5 day** |
| **M5** | **HTML report** | one self-contained `.html`: overlay strip, per-frame traces, stride table, budget, viewpoint-quality badge, attribution footer | the output contains no `http://` or `https://` resource reference except the attribution citations; it opens with JavaScript disabled; the file is under 25 MB for a 15 s clip | **1.5 days** |
| **M6** | **Vocabulary and output guard** | CI check over UI strings, templates, exports and docstrings | greps for **"risk"**, **"predict"**, "injury", "diagnos", "abnormal", "asymmetry index", "normal range" and **fails the build** on a hit; asserts the hip appears nowhere outside a collapsed block labelled "internal, not a measurement" | **0.5 day** |
| **M7** | **Film and run on own footage** | 2–3 consented clips per §7, processed end to end | the analyzer's checks pass on at least one clip; `view_ratio` ≥ 0.55; processing time recorded against §5's ~145 s estimate; **this closes `docs/REVIEW.md` §7.5's substance** — the first real phone footage through this tool | **0.5 day** + filming |
| **M8** | **Write-up** | `results/phase16a.md`: what was measured, the 20 fps gate recorded as **NOT MET** with its ms/frame, the licence position from §8, and the §7.5 status change | the report's numbers match `results/phase16a.json`; the gate is recorded as failed, not redefined | **0.5 day** |

**Total: ~6.5 working days**, plus filming. **M3 carries the only real uncertainty** —
it is the one component with no prior measurement — and it has an explicit
fallback that does not block the demo.

**Ordering constraint:** M1 must pass its regression test before anything is
rendered. A demo that shows numbers the sealed test cannot reproduce is worse
than no demo, which is the lesson phase 6 was redefined around.
