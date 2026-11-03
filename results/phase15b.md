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

*Not yet run at the time of this commit.*
