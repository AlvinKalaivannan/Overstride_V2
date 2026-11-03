# Phase 15B — why is the hip biased, and is "perfect 2D" even in distribution?

> **PRE-REGISTRATION, written 2026-11-02.** Committed **alone, before anything
> is run.** Three hypotheses, each with the rule that decides it written here
> first. Results are appended under "Results"; nothing above that heading is
> edited afterwards.

## What prompted this

The phase 15 diagnostic rows (`results/phase15.md`, n = 4) contain a result that
does not fit the story they were gathered to tell:

| subject | near hip r, **detected** 2D | near hip r, **perfect** 2D |
|---|---|---|
| Subject_12 | **0.349** | **0.112** |
| Subject_13 | **0.637** | **0.183** |

**The detector's 2D tracks the hip better than the projected reference does.**
That is backwards if "perfect 2D" means what the name says.

A mechanism is already on the record that would explain it. The deployed
checkpoint is `motionagformer-b-ath-det-coco-v1.ckpt` — trained on a **COCO
detector's** hip points. `results/phase07.md` §5 measured those sitting
**21–23 px** from projected mocap for *both* released detectors, because H36M's
hip is `mean(ASI, PSI)`, a point inside the pelvis that no image detector marks.
So the projected reference may be **out-of-distribution input** for this
checkpoint at the hip, and the quantity phase 15 called **"detector cost"**
(detected − perfect) may be **confounding detector error with a hip-convention
shift**.

Two further candidate conventions sit upstream of the same number, and a third
hypothesis replaces a test that n = 4 could not support.

## Scope and standing rules

- **Nothing is fitted to MoVi.** Every offset or recipe used below is taken from
  AthleticsPose (the *tune* set) or from the dataset authors' own code and
  documentation.
- **Every original figure stays.** New numbers sit **beside** them, each
  labelled with the convention it uses. The phase 15 headline —
  **9.610° near / 10.604° far on PG1 at n = 2** — **does not change.**
- **The angle code and the lifter are not touched.** `phase3_angles`,
  `phase7_inference_fix.lift` and the checkpoint are exactly as frozen.
- **The sealed test set stays sealed:** `Subject_16`, `_17`, `_18`, `_19`,
  `Subject_2` are not opened here.
- **Diagnostic subjects: `Subject_1`, `Subject_10`, `Subject_12`, `Subject_13`,
  plus `Subject_11` and `Subject_15`**, which now pass the re-registered check 1
  (`7a985d2`) and are therefore included **as diagnostic rows, labelled as
  such**. Every subject in this document is spent.

---

## H1 — INPUT convention

**Claim.** The projected reference is out-of-distribution input at the hip,
because the checkpoint was trained on COCO-detector hips rather than on
`mean(ASI, PSI)`.

**Method.** On **AthleticsPose tune data only**, measure where the COCO
detector's hips (`det_markers2d_by_cam_coco`) sit relative to the ground-truth
2D hips (`gt_markers2d_by_cam`): **mean offset and spread**, expressed **in the
pelvis frame** — one component along the pelvis (hip-to-hip) axis and one
perpendicular to it — and scaled in **torso-length units**, so the offset
transfers across camera views and subject sizes rather than being a pixel count
tied to one view. Then build a **convention-matched perfect 2D** for MoVi: the
projected reference with the hips displaced by that AthleticsPose-measured
offset. **No MoVi measurement enters the offset.**

**Outcome rule, fixed now.** H1 is **supported** if convention-matched perfect
2D, compared with the original perfect 2D:

1. raises near-hip **r** on **Subject_12 and Subject_13** to **at least the
   detected arm's r** on the same subject (≥ 0.349 and ≥ 0.637), **and**
2. moves the near-hip **bias** toward the detected arm's bias on those subjects.

Both conditions, both subjects. Otherwise **rejected**.

## H2 — REFERENCE convention

**Claim.** The MoVi reference was built with a different joint recipe from the
one AthleticsPose used, so the error includes a reference-definition term.

**Method.** The recipe is in the authors' own code,
`data/athleticspose/repo/athleticspose/utils.py:mocap_to_h36m` with
`joint_index_h36m_to_mocap` in `athleticspose/statics/joints.py`: each H36M
joint is the **mean of a named marker set**. Document it **joint by joint** —
pelvis, hips, spine, thorax, neck, head — against this project's
`MOVI_TO_H36M`, and rebuild the MoVi reference with the same recipe. Where MoVi
lacks a marker the recipe names, **the closest anatomical equivalent is chosen
from the marker descriptions and written down before any score is computed**,
never selected by which choice scores better.

**Outcome rule, fixed now.** H2 is **supported** if the recipe-matched reference
cuts the perfect-2D hip **|bias|** by **at least half** on **at least 3 of the 4
original diagnostic subjects** (`Subject_1`, `_10`, `_12`, `_13`). Otherwise
**rejected**.

## H3 — Within-subject posture

**Claim.** The hip error tracks the subject's posture, which the across-subject
test at n = 4 could not address (it gave Pearson r = −0.353 on four points,
indistinguishable from zero, with lean spanning only 5.0–6.7°).

**Method.** **Per frame, within each subject**, regress the hip error on two
mocap-derived posture variables — **pelvic tilt** and **trunk angle** — and
report the **slope, its confidence interval, and R²** per subject. This replaces
the across-subject lean test entirely; that test is not re-run.

**Outcome rule, fixed now.** H3 is **supported** if posture explains **at least
25% of hip error variance (R² ≥ 0.25)** in **at least 3 of the 4** original
diagnostic subjects. Otherwise **rejected**.

---

## What gets reported

1. A **joint-by-joint table**: `MOVI_TO_H36M` against the AthleticsPose recipe,
   with the **mean 3D distance in mm** between the two definitions of each joint.
2. Per subject, a **grid of arms × references**:
   - **arms:** detected / perfect (original input) / perfect (convention-matched
     input, H1)
   - **references:** original / recipe-matched (H2)
   - each cell: hip and knee, near and far, **MAE / bias / SD / r**.
3. **H3 regression results** per subject.
4. A **verdict per hypothesis** — supported or rejected — against the rule
   written above, not against whatever looks best.
5. **Detector cost recomputed** using whichever perfect-2D arm the verdicts
   support, **saying which arm and why**.

## What would make this phase wrong

- **Choosing a marker substitution by its score.** The H2 substitutions are
  fixed from the marker descriptions before scoring, and listed in Results
  whether or not they help.
- **Reading a supported H1 as "the detector is fine".** If the projected
  reference is out-of-distribution, that invalidates the *decomposition*, not
  the measured end-to-end error. The headline stands either way.
- **Letting any of this move the headline.** These are diagnostic rows on spent
  subjects. The number phase 15 reports is unchanged, and the sealed set is
  where a fix gets tested.

---

## Results

Run 2026-11-03, `scripts/phase15b_conventions.py` then
`scripts/phase15b_report.py`, which applies the three outcome rules
**mechanically from the JSON** rather than by eye. Numbers in
`results/phase15b.json`. Camera PG1, jogging only, six **spent** subjects.

**All three hypotheses are REJECTED against the rules fixed in `b13b48f`.**
What they ruled out is worth more than what they would have confirmed.

| hypothesis | rule | result | verdict |
|---|---|---|---|
| **H1** input convention | matched-input raises near-hip r to ≥ the detected arm's on both S12 and S13 **and** moves the bias toward it | the measured offset is **zero**, so the arm is identical to the third decimal | **REJECTED** |
| **H2** reference convention | recipe-matched reference halves perfect-2D near-hip \|bias\| on ≥ 3 of 4 | **0 of 4**; it makes the bias **~9° worse** on all six | **REJECTED** |
| **H3** within-subject posture | R² ≥ 0.25 in ≥ 3 of 4 | **1 of 4** (0.108, 0.124, 0.234, 0.315) | **REJECTED** |

---

### H1 — REJECTED, and the premise behind it was wrong

Measured on **1,094 AthleticsPose clips / 76,813 frames**, non-held-out subjects
only:

| hip | along the pelvis axis | perpendicular |
|---|---|---|
| left | **−0.0007** (sd 0.0835) | −0.0027 (sd 0.1668) |
| right | **+0.0011** (sd 0.0831) | −0.0034 (sd 0.1668) |

*torso-length units.*

**The COCO detector's hips sit on the ground-truth 2D hips, on average,
exactly.** The mean offset is under 0.1% of torso length in every direction.

So **there is no systematic 2D hip-convention offset to correct**, the
convention-matched arm differs from the original in the third decimal
(Subject_1 near-hip MAE 8.291° → 8.308°), and H1 fails its rule trivially.

**This corrects how `results/phase07.md` §5's "21–23 px" should be read.**
That figure is a mean **magnitude** — a Euclidean distance, always positive.
The mean **vector** offset is ~0 with a per-axis sd of 0.08–0.17 torso lengths,
which at AthleticsPose's subject scale is the same 10–20 px. **The detector's
hip error is large but zero-mean: it is scatter, not a convention shift.** Any
future claim that a detector "puts the hip in the wrong place" has to show a
non-zero mean, and this one does not.

**A caution recorded against myself:** 12-, 20- and 30-clip pilots of this same
measurement gave offsets of 0.02–0.04 torso lengths, which looked like a real
medial displacement and would have supported H1. At 1,094 clips it is zero.
**Those pilots were sampling noise and I read them as signal** in the working
notes before the full run; the full-n figure is the one that counts.

### H2 — REJECTED, and it fails in the opposite direction

The recipe, from the authors' own code
(`athleticspose/utils.py:mocap_to_h36m`, `statics/joints.py`), against this
project's `MOVI_TO_H36M`, with the mean 3D distance between the two definitions
of each joint:

| H36M joint | AthleticsPose recipe | `MOVI_TO_H36M` | distance |
|---|---|---|---|
| **neck** | mean(CLAV, Nasion) | CLAV | **178.5 mm** |
| **thorax** | **CLAV** | **THORAX** (Visual3D virtual) | **83.0 mm** |
| r_knee / l_knee | RKNE / LKNE *(lateral marker)* | RKJC / LKJC *(joint centre)* | 62.4 / 62.2 mm |
| r_elbow / l_elbow | RELB / LELB | REJC / LEJC | 51.2 / 52.7 mm |
| r_ankle / l_ankle | RANK / LANK | RAJC / LAJC | 35.3 / 35.6 mm |
| spine | mean(Xiphoid, T10) | midpoint(pelvis, thorax) | 25.1 mm |
| pelvis, l_hip, r_hip, head, shoulders, wrists | — | — | **0.0 mm** *(already identical)* |

**The hips were already identical** — both are `mean(ASIS, PSIS)` — so the
reference hip *position* was never the issue.

Substitutions, all fixed from the marker descriptions **before** scoring:
`STRN` for Xiphoid; the LFHD/RFHD midpoint for Nasion; `HEAD` for
mean(Vertex, EARC); the LWRA/LWRB midpoint for LWRT.

Perfect-2D near-hip **bias**, original reference → recipe-matched reference:

| subject | original ref | recipe ref | change |
|---|---|---|---|
| Subject_1 | −7.235 | **−16.540** | −9.31 |
| Subject_10 | −3.347 | **−11.211** | −7.86 |
| Subject_11 | **+0.924** | **−9.275** | −10.20 |
| Subject_12 | −3.498 | **−12.950** | −9.45 |
| Subject_13 | −5.699 | **−15.688** | −9.99 |
| Subject_15 | −1.522 | **−12.518** | −11.00 |

**0 of 4 rule subjects halved. Every subject got worse, by a strikingly
consistent −7.9 to −11.0°.**

**The real finding is that consistency.** Swapping one marker — the thorax,
from Visual3D's `THORAX` point to `CLAV`, 83 mm apart — moves the hip angle by
**about 9° on every subject**. Hip flexion here is measured against the
pelvis→thorax axis, so the thorax definition *is* the hip angle's reference
direction, and 83 mm of thorax displacement buys 9° of hip bias.

**That is larger than the entire error this phase set out to explain.** It also
means the −3 to −7° hip bias on the original reference is **not** evidence
that the original reference is wrong: matching the authors' recipe exactly makes
it three times worse. Whatever the lifter's internal thorax is, it is closer to
Visual3D's `THORAX` than to `CLAV` — which is the opposite of what training on
AthleticsPose's recipe would predict, and is unexplained.

**What this does NOT license.** It does not license picking the reference that
scores best. The original `MOVI_TO_H36M` stays the reference for phase 15
because it was fixed before any number existed, and the recipe-matched figures
are reported here as a **diagnostic row**, labelled. Choosing between them on
score is exactly the move the pre-registration forbids.

### H3 — REJECTED, narrowly, and the regressors are not independent

Per frame, within subject: near-hip error (perfect-2D arm, original reference)
on **anatomical pelvic tilt** — mean(LFWT,RFWT) → mean(LBWT,RBWT) against
horizontal — and **trunk angle**, pelvis→thorax against vertical.

| subject | R² | n | tilt–trunk correlation | meets R² ≥ 0.25? |
|---|---|---|---|---|
| Subject_1 | 0.108 | 161 | +0.549 | no |
| Subject_10 | 0.124 | 243 | +0.249 | no |
| Subject_12 | 0.234 | 416 | −0.643 | no |
| Subject_13 | **0.315** | 275 | +0.131 | **yes** |
| *Subject_11* | *0.017* | *249* | *+0.191* | *— (not in the rule set)* |
| *Subject_15* | *0.326* | *220* | *+0.810* | *— (not in the rule set)* |

**1 of 4. The rule required 3 of 4, so H3 is rejected.** Posture explains
11–32% of within-subject hip error variance — real, not nothing, and far more
than the across-subject test at n = 4 could see (that gave Pearson r = −0.353
on four points) — but **not the dominant term**.

**And the two regressors are not independent**: their correlation ranges from
−0.643 to **+0.810** across subjects, so the per-term slopes should not be read
as separate effects of tilt and of trunk angle. The R² is interpretable; the
individual coefficients largely are not. The first version of this test used
pelvis→spine as a stand-in for pelvic tilt, which was both mislabelled and
*more* collinear with trunk angle; it was replaced with the anatomical
definition before the run, and that replacement is why the collinearity is
reported at all.

### Detector cost, recomputed — and it changes sign

The verdicts support **neither** intervention, so the cost is recomputed with
the arm and reference phase 15 already used: **`perfect_original`** input,
**original** reference. Nothing about the decomposition is revised; what changes
is the confidence in it.

| subject | near detected | near perfect | **near cost** | far cost |
|---|---|---|---|---|
| Subject_1 | 10.328 | 5.983 | **+4.345** | +4.390 |
| Subject_10 | 8.892 | 7.039 | **+1.853** | +2.787 |
| Subject_11 | 9.967 | 7.210 | **+2.758** | +4.106 |
| Subject_12 | 13.211 | 7.521 | **+5.690** | +2.640 |
| Subject_13 | 10.646 | 7.783 | **+2.863** | +6.207 |
| **Subject_15** | **5.459** | **8.112** | **−2.653** | **−2.480** |
| **mean, n = 6** | | | **+2.476** | **+2.942** |

**On `Subject_15` the detected arm beats perfect 2D** — near-hip MAE 4.448° at
r = 0.869 against 9.004° at r = 0.442. The real detector outperforms an exact
2D input, by 2.65°.

**So "detector cost" is not a quantity.** Across six subjects it spans
**−2.65° to +5.69°** and **changes sign**. A mean of +2.48° over that range
describes nothing. Phase 15's statement that the split "does not generalise" was
an understatement: there is no stable split to generalise.

### What phase 15B establishes

1. **The detector does not displace the hip.** Zero mean offset over 76,813
   frames. The 21–23 px figure is scatter, and "21 px of hip convention error"
   is not a thing that exists.
2. **The hip angle is ~9° sensitive to the thorax marker alone.** 83 mm of
   thorax displacement, one marker substitution, moves hip bias by −7.9 to
   −11.0° on every subject. **The reported hip error is dominated by a
   definitional choice, not by the camera or the detector.**
3. **Matching the authors' own recipe makes it worse**, consistently and by a
   lot, which is unexplained and is the most interesting open question here.
4. **Posture is a real but minor term**, 11–32% of within-subject variance.
5. **There is no stable detector cost.** It changes sign between subjects.

**Nothing here moves the phase 15 headline**, which stays **9.610° near /
10.604° far on PG1 at n = 2**, and nothing here was fitted to MoVi. The sealed
test set — `Subject_16`, `_17`, `_18`, `_19`, `Subject_2` — **remains sealed**.

### What should happen next, and what should not

**Should not:** report a hip figure under whichever thorax definition flatters
it, or treat the −3 to −7° bias as a measurement of the camera path when a
single marker choice moves it by 9°.

**Should:** decide what the hip angle is *defined against* and fix it before any
further hip number is quoted — on the **tune** set, where AthleticsPose has both
`gt_markers3d_by_cam` and `raw_markers_in_world` and the thorax question can be
settled against the data the checkpoint was trained on, with no held-out subject
involved. Until then the knee is the joint this pipeline measures: its r is
0.86–0.97 across every subject, arm and reference in this table, and the recipe
swap moves its MAE by under 1°.
