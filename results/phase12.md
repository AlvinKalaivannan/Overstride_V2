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

*Not yet run. This file was committed with the bars above and no numbers, so
that the pre-registration is verifiable in `git log` rather than asserted.*
