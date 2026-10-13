# Phase 12 — Scan: a per-subject body model from monocular video

> **PRE-REGISTRATION, written 2026-10-13, before any result exists.**
> The bars below are frozen from this point. `CLAUDE.md` requires them recorded
> with the date and not edited once the first number lands. Results are appended
> under "Results" below; nothing above that heading changes.

**Question.** Phase 13 wants to constrain every frame to a subject's own segment
lengths, and to let the near limb's lengths constrain the far limb. That only
works if the lengths can be recovered from video in the first place. This phase
asks whether they can: **accurately enough, and stably enough across a subject's
clips, to be worth imposing as a constraint.**

This is a measurement of the measurement. It produces no injury output of any
kind, and nothing here is a clinical quantity.

---

## What is being measured

A **scan** is a per-subject body model: one length per segment, per side,
estimated from a single clip.

Segments, and where each comes from in the H36M-17 skeleton the lifter emits:

| segment | from | available? |
|---|---|---|
| thigh | hip → knee | yes |
| shank | knee → ankle | yes |
| pelvis width | l_hip ↔ r_hip | yes |
| trunk | pelvis → thorax | yes |
| **foot** | heel → toe | **no — reported as unavailable** |

**Foot length is not estimable and will not be approximated.** H36M-17 has no toe
keypoint, so the lifter cannot produce one. The mocap *does* have `R_Toe`/`L_Toe`,
so ground truth exists and is reported, but the estimate cannot. This follows the
precedent already set by `sagittal_angles`, which returns ankle angle as NaN
"rather than faked".

---

## The gate

Both bars are tightened from the ≤ 5% in `CLAUDE.md` to **≤ 3%**, decided before
any result was computed.

### Accuracy — measured on scale-invariant ratios

The lifter is **scale-free**: `normalize_kpts` divides by `max|xy|`
(`athleticspose/utils.py:86`) and `lift(..., scale=1.0)`
(`scripts/phase7_inference_fix.py:86`) passes that through, which is why the tool
records `"scale_free": True`. Absolute millimetres can only be recovered using
`norm_scale` — derived from ground truth — and `p2mm`. **Neither exists for a
real clip**, so a gate in millimetres would test something the phase 16 demo
could never reproduce.

The gate is therefore on ratios, which a real clip can produce:

```
thigh / shank
thigh / pelvis_width
shank / pelvis_width
trunk / thigh

GATE:  |ratio_est - ratio_gt| / ratio_gt  <=  3%     per ratio, per side
```

### Stability — CV across a subject's trials

`CLAUDE.md` says "a subject's scans agree with each other to ≤ 3% CV" without
saying what varies between scans. With 8 cameras per trial and ~26 trials per
running subject, there are two different quantities. Pinned now:

```
GATE:      CV of the scan across a subject's TRIALS        <=  3%
reported:  CV across CAMERAS within one trial  (view sensitivity)
reported:  CV pooled over both
```

Across-trials is the gate because it is the deployment question: the same runner,
filmed again, should scan the same.

### Kill criterion

```
if  CV(whole-clip scan length)  >=  CV(per-frame length)
then the scan adds nothing: record it and go to phase 13 without it
```

A scan that is no more stable than reading the length off a single frame is not a
body model, it is an average. Phase 13 would then constrain against per-frame
lengths directly.

---

## Reported alongside, and explicitly not gated

- **Absolute millimetres**, AthleticsPose only, via `norm_scale` + `p2mm`. Not a
  gate, because it is not reproducible on real video. It is reported because it
  is what catches anatomical nonsense — it is how the 479 mm femur was caught.
- **Both length conventions.** H36M-convention ground truth is the reference,
  because that is what the lifter was trained to emit. Anatomical lengths from
  the medial/lateral marker pairs are reported beside it, so the definitional
  offset is quantified rather than hidden. See `docs/data-inventory.md` §8.1: the
  hip is the only affected segment, at 81.07 mm / +18.2% on the femur.
- **Near and far limb separately**, per `CLAUDE.md`. A scan that recovers the
  near thigh and fails the far one is a failed scan, and an average would hide it.
- **L/R symmetry.** One subject (S19) shows a 36 mm femur asymmetry in mocap —
  about 10%, which is not anatomically plausible and is more likely a marker or
  short-trial artifact. Flagged, not silently averaged.
- **Subject-level bootstrap CIs.** A pooled point estimate is not reportable
  under `CLAUDE.md`.

---

## Split and cohort

| | |
|---|---|
| dataset | AthleticsPose — role **tune** (`docs/data-inventory.md` §8.1) |
| actions | `running` + `sprint`, matching the baseline's own action set (`phase3_infer.py:54`) |
| subjects | **15** — running S07 S09 S11 S13 S16 S19; sprint S04 S06 S08 S12 S14 S15 S21 S22 S23 |
| clips | 1,272 running + 414 sprint = **1,686** per-camera clips over 211 trials |
| `groups=` | **`subject`** |
| leakage | asserted, not merely called |
| ground truth | `raw_markers_in_world`, `(T, 84, 3)` float64 **world millimetres**, via the vendored `mocap_to_h36m` |

**Why subject and not event or trial.** Subjects are *not* nested within event —
S00/S02 appear in discus and shotput, S03 in javelin and shotput, and the `sd`
and `sprint` cohorts are the same nine people. Grouping by event leaks. Grouping
by trial also leaks, because 8 cameras share one trial.

**Why running + sprint rather than running alone.** Running has only 6 subjects,
too few to bootstrap. Sprint adds 9 and is the same locomotion, and the pair is
what phases 3 and 7 already used.

**No fitting happens in this phase**, so there is no train/test split to make.
The grouping key still matters: every number is computed per subject and
aggregated across subjects, never pooled over clips as though clips were
independent.

---

## Deviation from `CLAUDE.md`, recorded before the fact

`CLAUDE.md` describes the scan as taking "a calibration window: 2–3 s standing
side-on + the first strides". **No such footage exists in this dataset.**
Measured trial lengths at 100 Hz:

| | trials | frames (min / median / max) | seconds |
|---|---|---|---|
| running | 159 | 40 / 73 / 174 | 0.40 / 0.73 / 1.74 |
| sprint | 52 | 23 / 69 / 89 | 0.23 / 0.69 / 0.89 |

Not one trial reaches 2 s, and none is a standing pose — these are short athletic
efforts. So **a scan here is one whole trial**, roughly one to three strides of
running, and the standing calibration pass is untested. That is a limitation of
the tune set, not a change to the method: a standing pass should only make the
scan easier, and it can be validated when a dataset with one arrives, or on
consented footage in phase 16.

The capture rate above is **100 Hz**, from all 14 QTM exports. It is not the
120 fps that `scripts/video_kinematics.py` assumes. Segment lengths are a median
over frames and so are rate-independent; this phase's numbers do not depend on
the discrepancy. It is recorded in `docs/data-inventory.md` §8.1 as an open item
against the tool.

---

## Results

Run 2026-10-13. **1,674 clips** of the 1,686 available (12 dropped: fewer than
25 frames, or no matching mocap trial), **15 subjects**, grouped by subject.
Regenerate with `scripts/phase12_scan.py` then `scripts/phase12_aggregate.py`;
per-clip rows are in `results/phase12_scan.json`, the verdict in
`results/phase12_gate.json`.

> ### Verdict: the accuracy bar fails, the stability bar passes, the kill does not fire
>
> **Accuracy 0 of 8 ratios.** Errors run 4.46–9.17%, every CI clear of the 3%
> bar. The scan does not recover a subject's proportions.
>
> **Stability 8 of 8 ratios.** Across-trial CV 1.53–1.93%, comfortably inside
> the 3% bar. Film the same runner again and the scan returns the same numbers.
>
> **The kill criterion does not fire.** A clip's median is **1.68%** CV across
> trials against **14.52%** frame-to-frame within a clip — the scan is about
> **8.6x** more stable than reading a length off one frame. So the scan is worth
> keeping; it just is not anthropometry.
>
> In one line: **the scan is precise and biased.**

### Accuracy — gate not met

Mean absolute ratio error, computed per subject then averaged across the 15,
with a subject-level bootstrap CI (2,000 resamples, seed 11).

| ratio | error % | 95% CI | bar (3%) |
|---|---|---|---|
| thigh/shank, L | 4.64 | [3.80, 5.61] | Not met |
| thigh/shank, R | 5.49 | [4.39, 6.82] | Not met |
| thigh/pelvis, L | 6.94 | [6.09, 7.93] | Not met |
| thigh/pelvis, R | 6.71 | [5.28, 8.41] | Not met |
| shank/pelvis, L | 8.00 | [6.72, 9.53] | Not met |
| shank/pelvis, R | 9.17 | [7.71, 10.73] | Not met |
| trunk/thigh, L | 5.05 | [3.79, 6.56] | Not met |
| trunk/thigh, R | 4.46 | [3.13, 6.34] | Not met |

**The tightened bar is not what failed it.** The bar was moved from the 5% in
`CLAUDE.md` to 3% before anything was computed, so it is fair to ask whether the
original would have passed. It would not: at 5% two of eight ratios pass, and it
takes a **7%** bar to get six of eight. The result is robust to that choice.

### Near and far limb

| limb | error % | 95% CI |
|---|---|---|
| near | 6.03 | [5.26, 6.90] |
| far | 6.58 | [5.79, 7.46] |

The far limb is nominally worse by 0.55 points, but the intervals overlap across
most of their range. **There is no near/far split worth claiming here** — which
is consistent with phases 3B and 11, where the far-limb occlusion penalty was
also about zero.

### Stability — gate met

| ratio | CV across trials % | CV across cameras % | bar (3%) |
|---|---|---|---|
| thigh/shank, L | 1.93 | 3.99 | Met |
| thigh/shank, R | 1.72 | 4.30 | Met |
| thigh/pelvis, L | 1.65 | 3.87 | Met |
| thigh/pelvis, R | 1.54 | 4.11 | Met |
| shank/pelvis, L | 1.53 | 4.85 | Met |
| shank/pelvis, R | 1.62 | 5.22 | Met |
| trunk/thigh, L | 1.83 | 2.92 | Met |
| trunk/thigh, R | 1.63 | 3.03 | Met |

**Across cameras is consistently worse than across trials** — 2.92–5.22% against
1.53–1.93%. Viewpoint moves the scan more than the runner's own
trial-to-trial variation does. That is a direct argument for the phase 14
viewpoint term, and it means a scan should be taken from a known view rather
than averaged over arbitrary ones.

### Why it fails: the error is shape, not scale

Ratios are scale-invariant by construction, so no per-clip scale factor can move
them. The ratio error is therefore a **shape** error, and the absolute
millimetre figures say what shape:

| segment | bias vs H36M GT % | 95% CI | sd across subjects |
|---|---|---|---|
| thigh, L | −3.66 | [−6.99, +0.69] | 7.79 |
| thigh, R | −3.02 | [−6.95, +2.39] | 9.61 |
| shank, L | −4.59 | [−8.07, +0.21] | 8.70 |
| shank, R | −5.98 | [−9.27, −1.75] | 7.78 |
| **limb segments, mean** | **−4.31** | | |
| pelvis width | **+3.48** | [−0.26, +8.34] | 8.77 |
| trunk | −0.26 | [−4.02, +4.72] | 9.08 |

**Read these intervals before the means.** Only `shank_r` has a bias CI that
excludes zero. Across 15 subjects the per-segment sd is 7.8–9.6 points, larger
than the biases themselves, so **no individual segment's bias is established
here**. What *is* established is the ratio error in the table above, because that
is an absolute error and does not average away.

So the decomposition below is an explanation consistent with the data, not a
second measured result. Stated at that strength: the mean biases point the same
way in all six segments — limbs negative, pelvis positive — and that pattern
reproduces the ordering of the accuracy table arithmetically:

- `shank/pelvis` and `thigh/pelvis` are worst (6.7–9.2%) because the errors
  compound — numerator down ~4%, denominator up ~3.5%, so the ratio is off about
  7.5%, which is what was measured.
- `thigh/shank` is best (4.6–5.5%) because both ends are compressed together and
  only the difference between them (−3.3% vs −5.3%) survives.
- Trunk sits nearest zero (−0.26%), and pooled over all clips the median pelvis
  width lands on 197.7 mm against a ground truth of 197.6 mm. That pooled
  agreement is a descriptive coincidence, not a precision claim — the
  per-subject interval is [−0.26, +8.34].

What the data supports, then, is that **the ratio errors are real and large, and
a differential distortion of proportions is the natural explanation** — limbs
foreshortened relative to pelvis width, as a lifter regressing toward a mean
pose would produce. Establishing that distortion segment by segment needs more
subjects than the 15 here.

### What this means for phase 13

Phase 13 planned to constrain every frame to the scanned lengths and to let the
near limb's lengths constrain the far limb. This result splits that plan in two:

- **Not usable as anthropometry.** Imposing lengths that are 4–9% wrong in shape
  as if they were true would push the skeleton into the wrong proportions. The
  far-limb-from-near-limb idea is the weakest part: it assumes the near limb's
  measured length is right, and it is not.
- **Usable as a consistency constraint.** The honest version of the constraint is
  *"whatever this subject's proportions are, hold them constant"*, and that is
  worth something precise: frame-to-frame length CV falls from **14.52%** to
  **1.68%**. Joint angles are scale-invariant, so a constant proportional bias
  does not move them the way a fluctuating one does.

Phase 13's mandatory ablation (scan only / filter only / both) is what separates
these, and it must now be reported with the scan labelled a stabiliser rather
than a measurement. **No claim that the scan recovers true segment lengths may
be made downstream.**

### What surprised, and what looks wrong

- **Pelvis width is the one segment that comes back near-unbiased**, while limbs
  are short by 3–6%. Both come from the same lifter and the same frames. The
  pelvis is also the segment nearly always closest to the image plane in a
  side-on view, which fits an out-of-plane foreshortening account — but this
  phase did not test that, and the per-subject interval on pelvis bias is wide,
  so it stays an observation rather than a finding.
- **Between-subject scatter is large, and it is the headline caveat** — sd of the
  per-segment bias is 7.8–9.6 points across 15 subjects, bigger than the biases
  themselves. Stability says the distortion is reproducible *within* a subject;
  this says it varies a lot *between* them. That combination is what makes a
  single population correction factor useless and a per-subject one
  unidentifiable without ground truth, which is precisely what a deployed scan
  would not have.
- **Two mocap subjects carry implausible L/R asymmetry**: S19 at 9.9% and S16 at
  7.3% on the thigh, with S19 also 5.3% on the shank. A 10% femur-length
  difference is not normal anatomy. This is in the *ground truth*, not the
  estimate, so it is a marker-placement or short-trial artifact in
  AthleticsPose. It is left in — excluding subjects after seeing the numbers
  would be exactly the kind of choice this project does not make — but any
  phase 13 result that leans on per-side lengths should report with and without
  S19.
- **Foot was reported, never estimated.** Mocap gives 200–206 mm heel-to-toe;
  H36M-17 has no toe keypoint so there is no estimate to compare, and none was
  invented.

### Honest limits

- **One dataset, one lab, 15 subjects**, 6 of them running and 9 sprinting.
- **The scan is one short trial**, 0.23–1.74 s. The standing calibration pass
  that `CLAUDE.md` describes was never tested, because no such footage exists
  here. A standing pass could plausibly do better; that is untested, not denied.
- **Detected 2D only.** Everything above used `det_markers2d_by_cam_coco` with
  the matching checkpoint, which is the deployable path. The split between
  lifting error and detection error was not measured in this phase.
- Nothing here is an injury quantity, and nothing here supports one.
