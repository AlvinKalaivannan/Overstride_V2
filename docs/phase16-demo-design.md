# Phase 16 — demo design

> **Design document, 2026-11-04. Step 1 of phase 16: no app code is written
> yet. AMENDED 2026-11-04 after design review** — five amendments, marked
> `[A1]`-`[A5]` where they land, plus a licence action item.
>
> Every number here is measured and carries its source file and the conditions
> it was measured under. Where a condition was never measured, this document
> says so instead of interpolating. **Superseded text is struck or replaced in
> place and the reason is stated**, so the amendments are auditable.

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
| **resize** `[A1]` | **short side → 600 px**, `INTER_AREA` down / `INTER_LINEAR` up | `overstride/preprocess.py` |
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

### `[A1]` A fixed resize to short side 600 px, before detection

**Every clip is resized so its short side is 600 px before the detector runs.**
This is a **condition-matching step, not a speed optimisation**, and the reason
is structural: `detect_coco17` sets `min_size = min(h, w)`, so **the input's
short side *is* the detector's working resolution**. The sealed test — every
number in Row B of §3 — was measured on MoVi PG1 at **800×600**, i.e. short side
**600**. A clip at 1920×1080 runs the detector at `min_size=1080`, which changes
the pixel scale of every keypoint, and the lifter is measurably sensitive to
exactly that (phase 7 §5; phase 15F reproduced it a third time).

So resizing to short side 600 puts a clip of any resolution into **the condition
the error budget describes**. Not resizing would make the budget inapplicable.

It is also faster, which is a consequence and not the justification.

### Measured per-frame cost, on REAL frames

**The 207 ms/frame figure quoted in the first draft of this document is
superseded for runtime estimation.** `results/phase16_detector_speed.json` timed
**synthetic** frames, and says so in its own note: a synthetic frame contains no
person, so the crop tracker never engages and the keypoint head sees no
detections. Good for comparing call paths; wrong for estimating runtime.

`scripts/phase16a_timing.py` → `results/phase16a_timing.json`, real decoded MoVi
frames, deployed path, 16 threads, 24 timed frames per configuration after 4
warm-up:

| configuration | resolution | detector `min_size` | **ms/frame** | fps |
|---|---|---|---|---|
| **PG1 native — exactly the sealed condition** | 800×600 | 600 | **426.5** | 2.34 |
| CP1 native, unresized | 1920×1080 | 1080 | **1,452.0** | 0.69 |
| **CP1 resized to short side 600** `[A1]` | 1067×600 | 600 | **680.4** | 1.47 |

1080p unresized costs **3.4× the PG1 frame** (4.32× the pixels); the resize
recovers **2.13×** of that. A 16:9 clip at short side 600 is 1067×600, so it
still carries 1.33× the pixels of 4:3 800×600 — **the detector's working
resolution is matched exactly, the total frame area is not**, and that residual
is why 680 ms ≠ 426 ms.

> **A measurement discrepancy, recorded rather than reconciled away.** The
> design review cited 892–1,056 ms/frame at 800×600 and 3,243 ms at 1920×1080.
> **Those figures are not recorded anywhere in this repository**, and the
> measurement above gives 426.5 and 1,452 — roughly **2.2× lower in absolute
> terms** while the *ratio* between the two resolutions agrees closely (cited
> 3.33×, measured 3.4×). Two measurements of the same resolution effect at
> different absolute levels, most likely thread count or machine load. This
> document uses the scripted, committed figures and names the others; phase 7 §6
> recorded the identical pattern when it superseded an ad-hoc 2,790 ms figure
> with a measured 1,055 ms.

### Real-time is out of scope, and the gate is recorded as failed

**Phase 16's pre-registered `>= 20 fps` gate is NOT MET.** It is not being
lowered, redefined or quietly dropped.

| stage | measured | source |
|---|---|---|
| **detector, real frames at short side 600** `[A1]` | **426–680 ms/frame** (1.5–2.3 fps) | `results/phase16a_timing.json` |
| detector, synthetic frames, same call path | 207 ms/frame (4.84 fps) | `results/phase16_detector_speed.json` — **synthetic, superseded for runtime** |
| detector, before the call-path fix (`min_size=800`) | 1,055 ms/frame | same |
| lifter | **11.2 ms/frame** | same |
| **total, real frames** | **~437–692 ms/frame, ~1.4–2.3 fps** | sum |
| budget for 20 fps | 50 ms/frame | pre-registered |
| **shortfall** | **8.7–13.8× over** | — |

The 5.1× already recovered came entirely from *how the detector is called*, with
the same weights — no retraining and no shift in the keypoint distribution the
lifter was trained against. The remaining **8.7–13.8×** needs a different
architecture,
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

**Legs are labelled LEFT and RIGHT, never averaged.** The detector
distinguishes left from right unambiguously, so both legs always get their own
trace and their own numbers. An average would hide a real asymmetry: the sealed
test measured a far-limb thigh penalty of **+1.971° [+0.977, +3.187]**, same
sign on all five subjects.

> ### `[M1]` The demo does not INFER which leg is near the camera — the user declares it
>
> **Found while building M1, and it changes what the report can say.** The
> sealed test used the **camera** to decide which limb was nearer. The demo has
> no calibration — that is the point of it — so it must infer the near limb
> from the lifted 3D's hip depth. **Measured on all 11 spent MoVi PG1 subjects
> (`scripts/phase16a_nearlimb.py`, `results/phase16a_nearlimb.json`): that
> inference is correct on 9 of 11.**
>
> Worse, **no available signal separates the hits from the misses**, so the demo
> cannot even flag when it is unsure:
>
> | separator | agreeing subjects | disagreeing subjects | separates? |
> |---|---|---|---|
> | estimated hip depth separation / width | min **0.0354** | max **0.0549** | **no — overlapping, and inverted** |
> | per-frame sign consistency | min 0.476 | max 0.688 | **no — inverted** |
> | ground-truth `view_ratio` | 0.554 agrees | 0.665 fails | **no** |
>
> Phase 14's *"the near limb is identifiable from the estimate alone, 592 of 592
> clips"* was measured on **AthleticsPose** and does not transfer. Phase 14's
> azimuth proxy cannot rescue it either: it recovers `|cos θ|`, a **magnitude**,
> so it gives how side-on the view is and **not which side faces the camera**.
>
> **So the demo does not infer it. `--near-side left|right` is REQUIRED**, and
> it is the only thing that sets near and far. `AnalysisResult.near_limb`
> raises; the depth inference survives as `near_limb_from_depth`, documented as
> 9/11 and used for nothing. With the side declared, **each leg gets its own
> budget column** — §3.
>
> #### The direction-of-travel suggestion
>
> A geometric rule can *suggest* the near side. In the OpenCV camera frame
> (x right, y down, z into the scene, right-handed), world up is
> `u = (0, −1, 0)`, and the anatomical identity `anterior × left = superior`
> gives `left = superior × anterior`. For a runner travelling **rightward**,
> `anterior = (+1, 0, 0)`:
>
> `left = (0,−1,0) × (1,0,0) = (0, 0, +1)` — **+z, away from the camera.**
>
> | travel across the image | side facing the camera |
> |---|---|
> | rightward | **RIGHT** |
> | leftward | **LEFT** |
>
> **Measured on all 11 spent MoVi PG1 subjects against the calibrated near
> limb** (`scripts/phase16a_travelrule.py`,
> `results/phase16a_travelrule.json`):
>
> | | result |
> |---|---|
> | **hit rate, abstentions counted as misses** | **6/11** |
> | **accuracy when it was confident** | **6 of 6** |
> | abstained (travel too small to read) | 5 of 11 |
> | if forced to answer on every clip | 8/11 |
>
> **The rule is never wrong when it speaks, and the confidence floor is clean.**
> Confident clips carry `|dx|` of **0.0102–0.0310** hip widths per frame;
> abstained clips **0.0012–0.0095**. Those ranges **do not overlap**, so the
> 0.01 floor separates them exactly — unlike the near-limb separators above,
> every one of which overlapped. Forced to answer on all eleven it would score
> 8/11, so the abstention is doing real work rather than hiding failures.
>
> **Why 6/11 is a floor set by the test footage.** MoVi's jogging windows
> carry almost no net horizontal travel — subjects jog *within* a lab volume
> rather than crossing the camera's view — so five clips simply do not say which
> way the runner is going, and one can even reverse direction inside the window,
> in which case *there is no single near limb for the clip*. The geometry is
> exact and is verified against synthetic clips in
> `tests/test_demo_regression.py`. The demo's filming instructions (§7) ask for
> precisely the condition MoVi lacks: the runner crossing the full frame,
> ideally more than once.
>
> **So the suggestion is a WARNING ONLY, never a default**, per the design review's rule for a
> result below 11/11. It appears in the report, it warns when it disagrees
> with `--near-side`, and **it never overrides the flag.**
>
> #### Mirroring, which cannot be detected
>
> The rule inverts under mirroring, and mirroring breaks more than the rule: the
> 2D detector assigns left and right from learned appearance, so **in a mirrored
> frame it labels the anatomical left limb as the right one.** The anatomical
> labels themselves are wrong, not just the suggestion.
>
> **There is no reliable image-only test for it** — a mirrored runner looks
> exactly like an unmirrored runner going the other way. So the demo states the
> assumption, warns, and the filming instructions say **use the rear camera**.
> Nothing here pretends to detect it.

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

**`[A4]` The hip trace is NOT in the report at all.** The first draft allowed a
collapsed "diagnostics" block; that is removed. There is no hip trace, no hip
number and no hip axis anywhere in `report.html` — not collapsed, not greyed,
not behind a toggle, because a collapsed block is still a shipped number and the
reason the hip is excluded is that **its error is a learned convention**, which
no amount of labelling fixes.

It remains computable for internal work: `--debug` writes a **separate**
`report-debug.html` (or `.json`) carrying the hip trace, clearly marked
*internal, not a measurement*. **That file is never produced by a normal run,
never embedded in `report.html`, and never published.** The M6 guard asserts
`report.html` contains no hip trace.

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

### Row A `[A3]` — BEST CASE: in-domain lab data at 100 Hz

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

**Conditions:** a calibrated 8-camera lab, athletes, 100 Hz, and **the dataset
this checkpoint was fine-tuned on** — so this is in-domain performance measured
on held-out subjects of the *training* dataset.

> **`[A3]` This row is the best case and nothing more.** It is the ceiling the
> method reaches under ideal, in-domain conditions. **A user filming their own
> footage should expect Row B**, not Row A. Row A exists in this document so the
> gap between in-domain and cross-lab is visible, and for no other purpose. It
> is never shown as the expected error for a user clip.

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

### `[A2]` What the app displays — and the rule that B and C are never combined

**Row B is the expected error on new footage. Full stop.** It is displayed with
the label:

> **Expected error: measured on held-out lab footage at 30 fps.**

> ### `[M1]` Each leg gets its own column, from `--near-side`
>
> `--near-side left|right` is required, so the demo knows which leg the camera
> was on and quotes **Row B's near column for the near leg and the far column
> for the far leg**:
>
> | quantity | near leg | far leg |
> |---|---|---|
> | **knee flexion** | **8.41°** [6.275, 10.975] | **8.69°** [6.552, 10.679] |
> | **thigh inclination** | **5.40°** [4.136, 6.837] | **7.37°** [5.456, 9.258] |
>
> **This is what the flag buys.** Without it the demo would have to quote the
> far column for both legs — the worse of the two — and the thigh's stated error
> would rise from 5.40° to 7.37°, a 36% increase, on a leg that does not deserve
> it. One piece of information the user trivially has, and which no amount of
> computation recovers from the video, removes that penalty.
>
> `budget_limb(side)` returns `"near"` for the declared near side and `"far"`
> for the other, and an M4 test asserts both. **If the user declares the wrong
> side, the two columns are swapped and the error is understated on one leg by
> the measured gap** — hence the direction-of-travel warning in §2.

**Rows B and C are never added, subtracted, scaled or combined in any way.**
The first draft of this document said the app would show "Row B plus the Row C
adjustment for the clip's frame rate". **That was wrong and is removed.** Row C's
penalties are measured on the **tune set**, in-domain, at 100 Hz native; Row B is
a **different lab**. Combining them would manufacture a number for a condition
nobody measured — precisely the move phase 15E labelled "if it transferred
unchanged" and whose arithmetic produced **three undefined cells out of eight**
when the cross-lab loss came out negative.

So:

| the clip is | what the app shows |
|---|---|
| at or near 30 fps | **Row B**, labelled "measured on held-out lab footage at 30 fps" |
| **above 30 fps** | **Row B, plus the words "your clip is faster than the footage this was measured on; the effect of that is UNMEASURED on new footage."** Row B is *not* adjusted downward. |
| below 25 fps | refused (§4); no number is emitted |

**Row C is shown separately**, in its own panel headed *"Why we ask for 60 fps:
tune-set context"*, with one sentence making its status explicit:

> These penalties are measured **in-domain on the tune set**, not on new
> footage. They are the reason the filming instructions ask for 50 fps or
> better — **67% of the knee's 30 fps penalty is incurred between 50 and 30
> fps** — and they are **not** a correction you can apply to the expected error
> above.

An M4 test enforces this: no rendered value may be an arithmetic combination of
a Row B and a Row C figure.

### Conditions that were NOT measured, and are stated as such

The app must print these verbatim rather than interpolate:

| condition | status |
|---|---|
| **60 fps, anywhere** | **UNMEASURED.** 100/60 is not an integer, so no clean 60 fps decimation exists on the tune set and none was run. The nearest measured rate on the **tune set** is 50 fps; **no 60 fps figure is shown for a user clip.** |
| **`[A2]` ANY rate above 30 fps on new footage** | **UNMEASURED, and this is now the explicit rule.** MoVi PG1 is 30.000 fps and there is no higher-rate held-out footage in existence for this pipeline. A 60 fps user clip gets **Row B unchanged** plus a statement that the benefit of the higher rate is unmeasured on new footage. **Row C must not be used to adjust it.** |
| **phone / consumer footage, at any rate** | **UNMEASURED, and this is the project's largest open gap** (`docs/REVIEW.md` §7.5). MoVi's iPhone cameras ship no calibration and their sync cannot be verified, so the pipeline **refuses** to score them. The demo's own clips are the first phone footage this tool will ever process, and **no error figure applies to them.** The UI says so. |
| **running faster than jogging, outdoors, on a treadmill, in low light** | UNMEASURED. |

**`[A2]` Subtracting Row C from Row B to produce a "60 fps expected error" is
forbidden**, and so is adding, scaling or interpolating between them — Row C is
tune-set-measured and Row B is a different lab. Phase 15E's share-of-loss
arithmetic was explicitly labelled "if it transferred unchanged" and three of
its eight cells came out undefined. **The M4 test checks this mechanically.**

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
| 5 | **viewpoint** `[A5]` `[M2]` | estimated `view_ratio` **< 0.49** | estimated `view_ratio` **< 0.60** | below — the bars are in ESTIMATE units, shifted by the measured +0.139 estimator bias |

### The viewpoint check, in detail

**`view_ratio` is estimated from the detected 2D alone** — apparent 2D hip
separation over apparent trunk length, against a population ratio — which phase
14 measured at **Pearson 0.974 / Spearman 0.864** against the ground-truth depth
metric, with no depth, no millimetres and no ground truth required. The near
limb is also identifiable from the estimate alone, **592 of 592 clips**.

**`[A5]` The thresholds are derived from the measured evidence, and the binding
constraint is that no clip inside the measured range may be refused or even
warned.**

| reference | `view_ratio` | source |
|---|---|---|
| AthleticsPose, side-on lab | median **0.751** | phase 3B |
| **MoVi PG1 jogging — where Row B was measured, all 12 subjects** | median 0.675, **full range [0.536, 0.720]** | `results/phase15_standing_viewpoint.json`, `results/phase15_checks.json`, `results/phase15_sealed_checks.json` |
| — sealed subjects individually | 0.537, 0.555, 0.610, 0.661, 0.698 | `results/phase15_sealed_checks.json` |
| MoVi PG2, near-frontal | 0.436–0.607 | `results/phase15_checks.json` |
| MoVi standing, frontal | **0.057** | `results/phase15_standing_viewpoint.json` |

**Derivation:**

1. **PASS at `view_ratio` ≥ 0.50.** Every clip Row B was measured on lies in
   **[0.536, 0.720]**, and the lowest single sealed subject is **0.537**. A
   threshold at 0.55 — as the first draft had it — **would have warned on two of
   the five sealed subjects whose numbers *are* the budget**, which is
   incoherent. 0.50 sits below the measured minimum with a small margin and
   **cannot flag a clip inside the measured range.**
2. **WARN at 0.35 ≤ `view_ratio` < 0.50.** Below any clip the budget was
   measured on. Phase 11 measured sagittal error falling **3.50° → 1.77°**
   monotonically as the view becomes side-on across an independent lab's four
   calibrated cameras, so error is expected to rise through this band — but the
   *amount* is not measured for this pipeline, so it warns rather than refuses
   and the warning says the clip is outside the measured conditions.
3. **REFUSE below 0.35.** This is nearer frontal than side-on. Phase 11's curve
   puts the error at roughly **double** the side-on value at the frontal end,
   and phase 15 found **the viewpoint curve did not replicate on the one frontal
   clip it measured** — so there is not even a usable model of the error there.
   MoVi's frontal standing geometry sits at **0.057**, far below this line, so
   the refuse band is where a genuinely mis-aimed camera lands.

> ### `[M2]` The bars had to be restated in ESTIMATE units
>
> The derivation above is in **ground-truth** `view_ratio`, measured from
> calibrated 3D. **The demo has no calibration** — it computes
> `overstride.checks.estimate_view_ratio` from the detected 2D via phase 14's
> azimuth proxy. Phase 14 validated that proxy at Pearson 0.974, but
> **correlation is not scale, and an absolute threshold needs the scale.**
>
> Measured against calibrated ground truth on all 11 spent MoVi PG1 subjects
> (`scripts/phase16a_viewratio.py`, `results/phase16a_viewratio.json`):
>
> | | |
> |---|---|
> | **binding check: Row B's own footage passing its own check** | **11/11** |
> | **est − gt** | **+0.139** mean, sd 0.048, range [+0.025, +0.187] |
> | Pearson(gt, est) | **0.879** (phase 14 reported 0.974 on AthleticsPose) |
> | estimate range | [0.633, 0.887] against ground truth [0.537, 0.719] |
>
> **The estimate is systematically optimistic by about 0.14**, so the binding
> check passes — but for the wrong reason. A GT-unit bar applied to that
> estimate **under-refuses**: a clip whose *true* `view_ratio` is 0.35 estimates
> near 0.49 and would only warn. So the bars are restated in the units the demo
> actually computes:
>
> | | ground-truth units (derived above) | **estimate units (used)** |
> |---|---|---|
> | PASS at or above | 0.50 | **0.60** |
> | REFUSE below | 0.35 | **0.49** |
>
> `REFUSE = 0.35 + 0.139`, the GT bar plus the measured bias. `PASS = 0.60`
> sits below the **lowest measured estimate (0.633)** with margin, so the
> binding constraint still holds — now checked in the right units. The
> estimator's noise (sd 0.048) makes the effective GT cut fuzzy by roughly
> ±0.05, and the report says so rather than implying a sharp bar.

**Range check, stated as a requirement:** the lowest estimate Row B's own
footage produced is **0.633**, above the **0.60** pass bar, so **all 11
measured clips pass without warning.** An M2 test asserts this against the
committed aggregate.

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
        +-- 3b. RESIZE    short side -> 600 px  [A1]  condition-matching (~1 s)
        +-- 4. detect     video_kinematics.detect_coco17, crop-tracked  (680 ms/frame)
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

### `[A1]` Processing time, re-estimated from the REAL measured timings

At the measured **680 ms/frame** for a 16:9 clip resized to short side 600
(`results/phase16a_timing.json`), plus the 11.2 ms/frame lifter on model-frames
after resampling to 100 Hz:

| clip | frames | model-frames | detect | lift | decode+render | **total** | × real time |
|---|---|---|---|---|---|---|---|
| 10 s @ 60 fps | 600 | 1,000 | 408 s | 11 s | 8 s | **427 s ≈ 7.1 min** | **43×** |
| 15 s @ 60 fps | 900 | 1,500 | 612 s | 17 s | 8 s | **637 s ≈ 10.6 min** | 42× |
| 10 s @ 30 fps | 300 | 1,000 | 204 s | 11 s | 8 s | **223 s ≈ 3.7 min** | 22× |
| **8 s @ 60 fps** (the default analysis window, below) | 480 | 800 | 326 s | 9 s | 8 s | **343 s ≈ 5.7 min** | 43× |

**This is 3× worse than the first draft's estimate**, which used the synthetic
207 ms/frame. **~95% of it is the 2D detector.** It strengthens rather than
weakens §5's argument: at 6–11 minutes per clip there is categorically nothing
to be interactive about, and a server would be adding machinery to a batch job.

### `[A1]` Consequence: analyse a window, not the whole clip

Because the cost is linear in frames and a 15 s clip is now **10.6 minutes**,
the analyzer **selects an analysis window rather than processing the whole
clip**, defaulting to the **longest contiguous stretch in which the runner is
fully in frame, capped at 8 s**, overridable with `--window START END`.

This does not touch the error budget: the window has **the same frame rate and
the same resolution** as the clip, so it is the same measured condition — only
shorter. The user still films 10–15 s (§7), because the extra footage is what
makes a clean 8 s window available. Filming instructions are unchanged.

**No further speed-up is taken.** Batching crops measured no better
(238 ms/frame in a batch of 4), and frame skipping is not available — the lifter
consumes a contiguous sequence at a known rate, so dropping frames changes the
rate it sees, which is exactly the penalty Row C measures. **There is no honest
way to make this fast without replacing the detector, and that voids the
budget.** The design accepts ~6 minutes for an 8 s window.

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
   analyzer **refuses** a clip whose estimated `view_ratio` is below 0.49
   (about 0.35 in calibrated terms).
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

**Before anything else**
0. **Use the REAR camera, never the front/selfie camera.** Front cameras
   usually mirror, and `[M1]` **mirroring cannot be detected from the image**.
   A mirrored clip does not merely flip the near-side suggestion — the 2D
   detector labels the anatomical left limb as the right one, so **both legs'
   numbers are attributed to the wrong leg** and nothing in the report would
   reveal it.

**Settings**
5. **60 fps if the phone offers it**, 50 fps at minimum. Below 50 the penalty
   curve steepens sharply — **67% of the knee's 30 fps penalty is incurred
   between 50 and 30 fps** (§3 Row C, tune-set context) — and the analyzer
   warns; below 25 it refuses. **`[A2]` The benefit of 60 fps is UNMEASURED on
   new footage**, so the report shows **Row B unchanged** and says the effect of
   the higher rate is unmeasured. It does **not** show a 50 fps or 60 fps
   figure for your clip. More frames should not be worse, but on this pipeline
   that is an expectation, not a measurement.
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
13. Film **both directions** (left side to camera, then right side). The far
    limb carries a measured **+1.97°** thigh penalty, and `[M1]` the analyzer
    **cannot tell which leg is which**, so filming both ways means neither leg
    is always the far one across your clips. Compare a leg against *itself* in
    the two passes, not against the other leg in one pass.

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
   release asset containing the checkpoint requires an explicit grant.

> ### ACTION ITEM — open before any publishing step
>
> **Open a GitHub issue on `SZucchini/AthleticsPose` asking which licence covers
> the released checkpoints** (`checkpoints.zip`, obtained via `make
> checkpoints`), noting that `LICENSE-DATASET` scopes itself to
> `AthleticsPoseDataset.zip` and that the README's `- [ ] Add license for codes`
> is unchecked. Ask specifically whether the fine-tuned weights may be
> redistributed for non-commercial research use, and whether CC BY-NC-SA 4.0's
> ShareAlike term is intended to carry through to them.
>
> **Until that is answered:**
> - **the demo stays local** — a CLI the user runs on their own machine, loading
>   the checkpoint they obtained themselves. **No weights are served, bundled,
>   containerised or uploaded.**
> - **only reports of the user's own filmed footage are published.** No dataset
>   frame, no dataset-derived overlay, and no report generated from MoVi,
>   AthleticsPose or AthletePose3D footage leaves this machine.
> - the demo is distributed as **code plus instructions to obtain the checkpoint
>   from upstream**, which is how this project already obtains it.
>
> Record the answer, with a date and a link, in this section when it arrives.
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
| **M1** | **Processing path extracted — DONE** | `overstride/analyze.py` + `overstride/preprocess.py` — stages 1, 3, **3b**, 4–8 of §5, importing the frozen functions, no new maths | **the §6 regression test**: `Subject_17` reproduces knee 5.130°, thigh 3.274° on the **left leg**, 221/221 detected, `atol 1e-3`; plus `Subject_2` at the other extreme. *(The first draft said "near limb LEFT"; see the M1 results below — that assertion is wrong to make of the demo, which cannot determine the near limb.)* **`[A1]` plus two resize tests:** (a) the resize is applied and the detector's `min_size` is 600 for inputs at 800×600, 1280×720 and 1920×1080; (b) a 1080p clip completes within the §5 re-estimate — asserted as a per-frame ceiling of **900 ms/frame**, 1.3× the measured 680, so the test fails on a regression but not on machine noise | **1 day** |
| **M2** | **Input checks — DONE** | `overstride/checks.py` — the five checks in §4 with their measured thresholds, returning pass/warn/refuse | unit tests per check with synthetic inputs at each threshold boundary; a frontal-view fixture (GT `view_ratio` ≈ 0.06, about 0.20 estimated) must **refuse**; a 30 fps clip must **warn** and still process; **`[M2]` and the bars must be in ESTIMATE units**, with a test that Row B's own footage (estimated [0.633, 0.887]) all passes | **0.5 day** |
| **M3** | **Stride segmentation — DONE, and its count bar was NOT met** | `overstride/strides.py` — gait cycles bounded at **peak swing-phase knee flexion**, not heel strike `[M3]`; per-stride averages and a cycle count | **this is new code and the only unmeasured component.** Test: on a sealed MoVi clip the detected stride count matches a manually counted ground truth within ±1, on 3 clips; and per-stride knee peaks have a CV below the across-subject spread. **If it fails, the demo ships per-frame traces only** — stride averaging is a convenience, not part of the budget | **1 day** |
| **M4** | **Error-budget renderer — DONE** | §3's three rows as data, not prose: a table keyed by quantity × limb × frame rate, with source file and conditions per cell. **`[M1]` the renderer must select the FAR column for both legs** | a test asserts every displayed number is present in the source JSON it cites (`phase15_sealed.json`, `phase15e.json`, `phase15f.json`) — **no number may be hardcoded without a matching source**; and that an unmeasured condition renders the word "UNMEASURED". **`[A2]` plus the no-combination test:** for a 60 fps clip the rendered expected error must equal a Row B value **exactly**, and must not equal any Row B ± Row C arithmetic combination; Row A must be labelled "best case, in-domain" `[A3]`; and the word UNMEASURED must appear for the above-30-fps case | **0.5 day** |
| **M5** | **HTML report** | one self-contained `.html`: overlay strip, per-frame traces, stride table, budget, viewpoint-quality badge, attribution footer | the output contains no `http://` or `https://` resource reference except the attribution citations; it opens with JavaScript disabled; the file is under 25 MB for a 15 s clip | **1.5 days** |
| **M6** | **Vocabulary and output guard** | CI check over UI strings, templates, exports and docstrings | greps for **"risk"**, **"predict"**, "injury", "diagnos", "abnormal", "asymmetry index", "normal range" and **fails the build** on a hit. **`[A4]` asserts `report.html` contains NO hip trace at all** — no hip series, no hip axis label, no hip summary cell — and that a hip trace appears only in a `--debug` file that a normal run does not produce | **0.5 day** |
| **M7** | **Film and run on own footage** | 2–3 consented clips per §7, processed end to end | **`[A5]`** the analyzer's checks pass on at least one clip with **`view_ratio` ≥ 0.50** (the §4 pass threshold, which the full measured MoVi range [0.536, 0.720] clears); the resize to short side 600 is confirmed applied; processing time recorded against §5's re-estimated **~5.7 min for an 8 s window**; **this closes `docs/REVIEW.md` §7.5's substance** — the first real phone footage through this tool | **0.5 day** + filming |
| **M8** | **Write-up** | `results/phase16a.md`: what was measured, the 20 fps gate recorded as **NOT MET** with its ms/frame, the licence position from §8, and the §7.5 status change | the report's numbers match `results/phase16a.json`; the gate is recorded as failed, not redefined | **0.5 day** |

**Total: ~6.5 working days**, plus filming. **M3 carries the only real uncertainty** —
it is the one component with no prior measurement — and it has an explicit
fallback that does not block the demo.

**Ordering constraint:** M1 must pass its regression test before anything is
rendered. A demo that shows numbers the sealed test cannot reproduce is worse
than no demo, which is the lesson phase 6 was redefined around.

---

## M1 — complete, 2026-11-04

`overstride/analyze.py`, `overstride/preprocess.py`,
`tests/test_demo_regression.py`. **13 tests, all passing; 115 in the suite.**

| assertion | result |
|---|---|
| Subject_17 left leg, knee flexion MAE | **5.130°** == committed, `atol 1e-3` |
| Subject_17 left leg, thigh inclination MAE | **3.274°** == committed |
| Subject_2 left leg, knee flexion MAE | **12.964°** == committed |
| Subject_2 left leg, thigh inclination MAE | **8.032°** == committed |
| frame count / detection | 221/221 and 157/157, unchanged |
| resize applied, detector `min_size` | **600** for 800×600, 1280×720, 1920×1080, 640×480, 1080×1920 |
| resize is identity at 800×600 | byte-identical — so adding it changed no measured number |
| 1080p clip, after resize | **passed the 900 ms/frame ceiling** |
| aspect ratio preserved | within 0.01 |

**The regression test earned its place immediately:** it failed on first run,
on Subject_2, and the failure was the near-limb claim rather than an angle —
which is what produced the §2 and §3 findings above. A test that only ever
passes would not have found it.

**Deviation from the design, recorded:** the test asserts **left/right**, not
near/far, mapping the sealed near/far columns through the sealed result's own
`near_limb` field. The first draft of §6 specified "near limb LEFT" as an
assertion; that assertion is **wrong to make of the demo**, because the demo
cannot determine it.


---

## M3 — complete, 2026-11-08. **The count bar was NOT met.**

`overstride/strides.py`, `scripts/phase16a_strides.py`,
`results/phase16a_strides.json`, `tests/test_strides.py`. **36 tests passing;
235 in the suite.**

### `[M3]` The cycle boundary is peak swing-phase knee flexion, not heel strike

The design asked for heel-strike detection. **It is not available on this
pipeline**, and the attempt is recorded rather than deleted.

* There is **no foot or heel keypoint**. H36M-17's most distal joint is the
  ankle, the same limitation phase 14 recorded for the foot segment, so there
  is no ground-contact signal to threshold.
* The standard monocular substitute for a force plate is the coordinate-based
  method of **Zeni et al. (2008)**: heel strike is the frame of maximum
  *anterior* ankle offset from the pelvis. "Anterior" is a direction, so the
  method needs to know which way the runner travels across the image.
* **That direction is not readable on this footage.** `overstride.nearside`
  abstains on 5 of 11 MoVi clips. Reading it instead from running's own duty
  factor — stance is the shorter half-cycle, because running has a flight
  phase — was measured and **failed**:

| the rejected candidate, 11 spent MoVi PG1 subjects | result |
|---|---|
| duty-factor statistic, readable range | **4 of 11** clips readable |
| measured range of the statistic | 0.28 – 5.11 (0.35 or 0.65 reads the direction; ~0.5 cannot, and **above 1 the pairing spans more than a whole cycle**, so the method is incoherent rather than merely ambiguous) |
| pelvis excursion across the camera's x axis | **3.6 – 6.9 trunk lengths** |
| its cycle count vs the shipped method | differs by **up to 8 strides** |

**Why it fails is specific and was visible in the data:** MoVi subjects
traverse the capture volume and **turn around inside the window**, so anterior
reverses mid-clip. The design already suspected this in §2 — *"one can even
reverse direction inside the window"* — and here it is measured.

So the cycle is bounded at the **peak of knee flexion**, the sharpest,
highest-amplitude event in the running cycle (39–95° of swing on the mocap),
which needs no direction at all. **The report says so** — the stride axis reads
*"% of stride cycle (0 = peak swing-phase knee flexion)"* — and the demo never
claims a heel strike, a ground-contact time or a stance/swing split.

### The detector: a hysteresis trigger

Peak picking with a refractory period needs a period estimate, and the period
is not constant when a runner decelerates and turns: at a refractory of half a
period the spent subjects produced stride gaps of **0.33–1.97 s against a
0.70 s period**, half of them impossible. A **Schmitt trigger** needs no period
at all — one cycle per excursion above a high edge and back below a low edge,
the event being that excursion's maximum.

**A second bug the ground truth exposed:** a single NaN frame split the signal
into two islands and re-armed the trigger mid-cycle, counting it twice. Runs
shorter than 0.2 s are now interpolated across (`fill_short_gaps`), which at
30 fps cannot hide a gait event.

### Validated against the mocap, all 11 spent subjects

The segmentation runs on the **mocap 3D in the PG1 camera frame** for ground
truth, with a consistency check that needs no ground truth at all: **two legs
necessarily take the same number of steps.**

| | result |
|---|---|
| **left/right cycle counts agreeing within 1** | **11 of 11 subjects** |
| cycles per leg | 5 – 14 |
| mean stride duration | **0.75 – 0.95 s** (jogging, as expected) |
| knee swing amplitude | 39 – 95° |
| stride duration CV | 0.14 – 0.47 — **reported, never gated** |
| subjects passing the self-check | **10 of 11** |

**The one subject that failed its own check is `Subject_11`** — the subject
phase 15 had already excluded by a pre-registered check, reached here by a
completely different route.

### The count bar: **2 of 3, recorded as FAILED**

M3's test was *"the detected stride count matches ground truth within ±1, on
3 clips"*. The demo path was run end to end on three spent subjects:

| clip | ground truth | estimate | error | within ±1 | passed its own check |
|---|---|---|---|---|---|
| best-behaved sealed subject | L7 R7 | L6 R7 | −1 / 0 | **yes** | yes |
| worst sealed subject | L5 R6 | L5 R6 | 0 / 0 | **yes** | no |
| a diagnostic subject | L5 R6 | L5 R4 | 0 / **−2** | **no** | no |

**The bar stands as written and is recorded as failed.** It is not lowered, not
redefined, and not re-read as "2 of 3 is close". The design's stated
consequence therefore applies: **the demo ships per-frame traces only**, and
stride averaging is opt-in behind `--stride-average`.

What can honestly be added is very little, and it is stated rather than
implied: the clip whose count was wrong **also failed the module's own check**,
so the fallback would have caught it — but **only 1 of the 3 clips passed that
check**, so the check's screening value rests on a single clip and is not
evidence of anything. `STRIDE_COUNT_BAR_MET = False` and
`STRIDE_AVERAGING_IS_DEFAULT = False` are asserted in `tests/test_strides.py`,
and the report prints the failed bar whenever stride averaging is used.

### The second bar, which **was** met

Per-stride knee peaks vary **less within a subject (median CV 0.282) than knee
peaks vary across subjects (0.433)**. That is what makes a stride average worth
showing at all: if the two were equal, the average would be averaging noise.

### Deviation from the design, recorded

The M3 test specified *"a manually counted ground truth"*. The count is taken
instead from the **120 Hz Qualisys mocap** sampled at the video's own frame
times — a stronger reference than a hand count, and one that can be re-run —
with the ground-truth-free left/right agreement above as the independent
criterion. `scripts/phase16a_strides.py` also prints a text trace of the mocap
knee signal for each scored clip, so the cycles remain hand-countable from the
log.


---

## M4 — complete, 2026-11-08

`overstride/budget.py`, `tests/test_budget.py`. **25 tests passing; 260 in the
suite.**

### No number is written down

Every cell carries its source file *and a JSON pointer into it*, and reads its
value out of the committed result at load time. A figure therefore cannot
drift away from the measurement that produced it: change the result file and
the report changes with it.

**Two tests, because one is not enough.** Resolving the pointer proves the path
exists; it does not prove the module reads it. So a second test copies the
three result files to a temp directory, adds 1.0° to the sealed knee figure,
loads the budget from there, and asserts the rendered value moved. A hardcoded
number would survive the first test and fail the second.

| row | cells | source |
|---|---|---|
| **A** — best case, in-domain, 100 Hz `[A3]` | 6 | `results/phase15e.json` `pooled/coco_det\|native/…` |
| **B** — held-out lab footage at 30 fps | 6 | `results/phase15_sealed.json` `pooled/detected/…` |
| **C** — frame-rate penalties, tune set | 18 (3 rates × 3 quantities × 2 limbs) | `results/phase15e.json`, `results/phase15f.json` |

A missing **file** now raises `MissingSource`; a missing **path inside** an
existing file yields a missing cell that renders `UNMEASURED`. Those were the
same code path until the test caught it — a deleted result file silently
produced an *empty* budget rather than refusing.

### `[A2]` B and C are never combined — enforced by provenance

The first implementation checked this numerically: is the displayed value
within 5e-3 of any Row B ± Row C sum? **That cannot work at this density.**
6 Row B values × 18 Row C penalties × 6 multipliers is 648 candidate sums, and
one of them lands on the legitimate thigh figure **5.4008°** by coincidence —
a birthday-paradox false positive, not a combination.

So the rule is **provenance, not arithmetic**: a displayed expected error must
be **bit-identical** to exactly one Row B cell (`quoted_from`). Nothing added
to, scaled or interpolated from Row B can satisfy that, and coincidence cannot
trip it. The numeric check is kept as a secondary net, applied only after
provenance. Both are asserted over every quote the report can make, at
25/30/30.009/50/60/120 fps.

A 60 fps clip therefore gets **Row B unchanged** plus the words *"your clip is
faster than the footage this was measured on; the effect of that is
UNMEASURED on new footage, and the number above is NOT adjusted downward."*
A slower clip gets the mirror of it, also unadjusted.

### What else is asserted

* **`[M1]`** each leg gets its own column from `--near-side`: the near leg
  Row B's near column, the far leg its far column — and that the gap the flag
  buys back is the measured 36% on the thigh.
* **`[A4]`** no cell, label or source path anywhere in the budget mentions the
  hip.
* **`[A3]`** Row A is labelled *"best case: in-domain lab data at 100 Hz"* and
  every Row A cell's conditions say "fine-tuned".
* Row C's conditions say **"IN-DOMAIN on the tune set"** on every cell.
* The four unmeasured conditions print verbatim, carrying `UNMEASURED` and the
  sentence *"Row C must not be used to adjust it."*
* Row B's two measured caveats travel with it: the error is largely a
  per-subject offset but **not a removable one** (17% / 11% under an oracle
  correction), and the **shape** is tracked well (knee r +0.77 to +0.92).
