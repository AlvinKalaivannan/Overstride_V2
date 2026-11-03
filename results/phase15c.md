# Phase 15C — what is this lifter's "thorax", anatomically?

> **PRE-REGISTRATION, written 2026-11-03.** Committed **alone, before anything
> is run.** The candidates, their cross-dataset mapping and the decision rule
> are all fixed here. Results are appended under "Results"; nothing above that
> heading is edited afterwards.

## Question

**Which anatomical point does the deployed lifter's output joint 8 ("thorax")
actually correspond to?**

It matters because `phase3_angles.sagittal_angles` builds its body frame from
**pelvis → thorax**, so the thorax definition *is* the direction hip flexion is
measured against. Phase 15B measured the size of that: swapping the MoVi
reference's thorax from Visual3D's `THORAX` point to `CLAV` — **83 mm apart** —
moved the hip bias by **−7.9° to −11.0° on every one of six subjects**
(`results/phase15b.md`). That is larger than the entire hip error phase 15 set
out to explain, and phase 15B could not say which definition is right, only that
the choice dominates.

Phase 15B also left a specific puzzle: matching the authors' own recipe
(thorax = `CLAV`) made the MoVi hip bias **three times worse**, which is the
opposite of what training on that recipe predicts.

**This is decided on AthleticsPose only** — the *tune* set, which ships both
`raw_markers_in_world` and `gt_markers3d_by_cam`, and which the deployed
checkpoint was trained on. **No MoVi subject plays any part in choosing the
definition.**

## Candidates, fixed now

Five, with the cross-dataset mapping fixed **before any scoring**:

| candidate | AthleticsPose marker(s) | MoVi marker(s) | why this pairing |
|---|---|---|---|
| **CLAV** | `CLAV` | `CLAV` | same name, same landmark — the clavicle / jugular-notch marker. This is what the authors' recipe uses for H36M thorax |
| **C7** | `C7` | `C7` | same name — the 7th cervical vertebra's spinous process, the posterior counterpart to CLAV at the same level |
| **STRN** | `Xiphoid` | `STRN` | AthleticsPose has no marker named STRN. In Plug-in-Gait, STRN sits on the **xiphoid process** at the inferior sternum, which is exactly AthleticsPose's `Xiphoid`. MoVi's own description for `STRN` is "Sternum" |
| **midpoint(CLAV, C7)** | mean(`CLAV`, `C7`) | mean(`CLAV`, `C7`) | both datasets carry both markers. An anterior/posterior midpoint sits **inside** the thorax rather than on its surface, which is where a fitted skeleton's joint would be expected to lie |
| **mean(Xiphoid, T10)** *(control)* | mean(`Xiphoid`, `T10`) | mean(`STRN`, `T10`) | the AthleticsPose recipe's **spine** joint, included as a control: if the lifter's "thorax" matched this, the released data's joint 7 and joint 8 would be confounded |

All five markers were verified present in both datasets before this commit.

## Decision rule, fixed now

The chosen definition is the candidate whose trunk axis — **pelvis → candidate**
— has the **smallest mean sagittal angular offset** from the **lifter's output
trunk axis**, measured on AthleticsPose **test-split subjects S11 / S13 / S16**,
**running and sprint** clips.

**Tie-break:** if two candidates are within **1°** of each other, the one the
authors' recipe uses (**CLAV**) is chosen.

The angular offset is measured in the **sagittal plane**, defined as the plane
perpendicular to the ground-truth hip-to-hip axis, so it needs no independent
"up" direction — which matters, because the usual "up" *is* the quantity under
test. Estimate and reference are brought into a common frame by a rigid
alignment on the **lower body only** (pelvis, hips, knees, ankles), so the trunk
is never used to align the thing being measured about the trunk.

## MoVi rule, fixed now

The chosen definition is applied to MoVi **once**, and the result is reported
**beside** the original numbers, labelled **"thorax-matched (15C)"**.

- **The phase 15 headline does not change.** It stays **9.610° near /
  10.604° far on PG1 at n = 2**.
- **The sealed test set stays sealed:** `Subject_16`, `_17`, `_18`, `_19`,
  `Subject_2` are not opened.
- **The angle code and the lifter are not touched.**
- Subjects `1`, `10`, `12`, `13` carry the re-score; `11` and `15` are included
  as **diagnostic rows, labelled**. Every subject involved is already spent.

## What gets reported

1. **Recipe verification against the released data.** Rebuild `markers_h36m`
   from `raw_markers_in_world` with the authors' own
   `athleticspose/utils.py:mocap_to_h36m`, and compare it with
   `gt_markers3d_by_cam` **per joint, in mm** — minding that
   `gt_markers3d_by_cam` is in **pixels** with a per-frame `p2mm` factor that
   the released evaluator *divides* by (`CLAUDE.md`; `results/phase03b.md`).
   **If the released thorax is not `CLAV`, say what it actually matches.**
2. **Candidate offsets.** The deployed `det-coco` checkpoint run on
   `det_markers2d_by_cam_coco` for S11/S13/S16; per candidate, the sagittal
   angular offset between the lifter's output trunk axis and pelvis → candidate:
   **mean, SD, per subject**. Plus the **3D distance** from the lifter's thorax
   to each candidate after aligning on pelvis and hips.
3. **The decision rule applied mechanically**, and the chosen definition stated.
4. **Sanity check on the tune set:** hip MAE and bias with ground-truth hip
   angles computed using **each** candidate, so the size of the definition
   effect is measured on AthleticsPose too, not only on MoVi.
5. **PART 3:** MoVi re-scored against the thorax-matched reference, detected and
   perfect 2D, hip and knee, near and far, **MAE / bias / SD / r**, beside the
   original.

## What would make this phase wrong

- **Choosing the candidate by which MoVi number it produces.** The rule is
  evaluated on AthleticsPose and nowhere else, and it is applied in code.
- **Reading a chosen definition as a correction to the headline.** It is a
  diagnostic re-score of spent subjects. If it turns out the reference was
  mis-specified, that is a finding about the *reference*, and the place to act
  on it is the sealed set.
- **Treating the control's success as good news.** If `mean(Xiphoid, T10)` wins,
  the released joints 7 and 8 are confounded and the whole trunk frame needs
  re-examining, not a one-marker swap.

---

## Results

*Not yet run at the time of this commit.*
