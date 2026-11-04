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

Run 2026-11-03. Part 2 `scripts/phase15c_thorax.py`
(`results/phase15c.json`), part 3 `scripts/phase15c_movi.py`
(`results/phase15c_movi.json`), addendum
`scripts/phase15c_thorax_on_movi.py` (`results/phase15c_thorax_on_movi.json`).

**231 AthleticsPose clips, 24,908 frames**, subjects S11/S13/S16, running and
sprint.

> **The question is answered, and the answer does not transfer.** On
> AthleticsPose the lifter's thorax is **`CLAV`**, verified three independent
> ways. Applied to MoVi it makes the hip **worse on every subject** — because
> on MoVi the same checkpoint's thorax lands somewhere else. **The hip angle, as
> this pipeline defines it, is not a transferable measurement.**

### Two defects found before any result, both from absurd numbers

The first run gave a **510 mm residual on the pelvis** and 2 m thorax
distances. Two causes, both fixed:

1. **`raw_markers_in_world` is per clip; `gt_markers3d_by_cam` is per camera and
   differently trimmed** — 59 frames against 41 on the first clip tried — and
   the dataset records no frame correspondence. Frame *i* of one was being
   compared against frame *i* of the other. **Only the 231 of 592 length-matched
   pairs are used.** That is data plumbing, not a measurement-driven selection.
2. **`p2mm` varies per frame** (0.2677–0.2744 on one clip), so dividing all
   three axes by it is a **per-frame perspective normalisation**, not a global
   unit conversion — `CLAUDE.md` says the released evaluator divides "uniformly
   across all three axes". No single rigid transform can fit such a sequence, so
   **alignment has to be per frame.**

The clue that found both: the rebuilt recipe's trunk length and the released
array's agreed to **0.15 mm** (471.86 vs 471.71) while the positional residual
was 510 mm. Geometry that exact alongside an error that large can only be a
correspondence or frame problem, never a recipe problem.

### Step 1 — the recipe verified, and the released thorax IS CLAV

`markers_h36m` rebuilt from `raw_markers_in_world` with the authors' own
`mocap_to_h36m`, compared per joint against the released
`gt_markers3d_by_cam`, per-frame rigid alignment on all 17 joints:

| joint | residual | | joint | residual |
|---|---|---|---|---|
| **pelvis** | **4.27 mm** | | r_wrist | 22.35 mm |
| **spine** | **4.38 mm** | | l_wrist | 23.59 mm |
| **thorax** | **5.04 mm** | | r_elbow | 29.49 mm |
| head | 10.77 mm | | l_ankle | 30.54 mm |
| l_hip / r_hip | 11.04 / 11.33 mm | | l_elbow | 30.77 mm |
| neck | 11.06 mm | | r_ankle | 30.93 mm |
| shoulders | 19.54 / 19.75 mm | | knees | 19.56 / 20.24 mm |

**The released thorax is `CLAV` to 5.04 mm** — and the trunk joints (pelvis,
spine, thorax) are the **three best-matched joints in the entire skeleton**.
The recipe is confirmed, joint by joint, against the data it describes.

The residual *pattern* is itself the explanation of the remainder: ankles and
elbows at ~30 mm against 4–5 mm at the body centre is exactly what a single
per-frame scalar standing in for a perspective projection produces — worst
where limbs swing away from the depth-reference plane, best near the centre.
**It is not recipe error.**

### Step 2 — the lifter's own trunk axis

The deployed `det-coco` checkpoint on `det_markers2d_by_cam_coco`; sagittal
angular offset between the lifter's output trunk axis and pelvis → candidate.
Alignment per frame on the **lower body only**, so the trunk never helps place
the trunk, and the sagittal plane is defined by the hip axis alone, so no
independent "up" is assumed — which matters, because the usual up *is* the
quantity under test.

| candidate | mean | \|mean\| | sd | S11 | S13 | S16 | thorax→cand. |
|---|---|---|---|---|---|---|---|
| **CLAV** | **−0.750°** | **0.750** | 3.482 | −1.065 | −0.127 | −0.751 | **203.4 mm** |
| mid_CLAV_C7 | +3.256° | 3.256 | 3.454 | +2.715 | +3.719 | +3.324 | 228.5 mm |
| mean_Xiphoid_T10 *(control)* | +3.598° | 3.598 | 4.131 | +4.485 | +5.719 | +3.162 | 288.5 mm |
| C7 | +6.837° | 6.837 | 3.435 | +6.097 | +7.138 | +6.968 | 258.4 mm |
| STRN | −22.270° | 22.270 | 4.212 | −20.524 | −19.978 | −22.915 | 275.3 mm |

**CLAV is within 0.75°, and it is consistent across all three subjects**
(−1.07, −0.13, −0.75). The control did not win, so the released joints 7
and 8 are not confounded.

### Step 3 — decision, applied mechanically

Ranking by \|mean\| offset: **CLAV 0.750**, mid_CLAV_C7 3.256,
mean_Xiphoid_T10 3.598, C7 6.837, STRN 22.270.
Gap first-to-second **2.505°**, above the 1° tie threshold, so **no tie-break
was needed.**

> ### CHOSEN: **CLAV** — smallest \|mean\| sagittal offset, 0.750°.

This agrees with the recipe, which was the pre-registered tie-break, and with
step 1's 5.04 mm. **Three independent lines of evidence, one answer.**

### Step 4 — the size of the definition effect, on the tune set

AthleticsPose hip error with the ground truth's thorax set to each candidate
(the estimate is unchanged throughout; only the reference moves):

| candidate | hip MAE | hip bias |
|---|---|---|
| **CLAV** | **4.118°** | **−1.799°** |
| mid_CLAV_C7 | 4.092° | +2.150° |
| mean_Xiphoid_T10 | 4.918° | +2.931° |
| C7 | 6.145° | +5.679° |
| STRN | 22.788° | −22.772° |

**CLAV gives a hip bias of −1.8° and a MAE of 4.12°**, consistent with the
frozen AthleticsPose baseline. The wrong thorax costs up to **22.8°** — so on
the tune set too, **the thorax definition dominates the hip number.**

---

## Part 3 — applied to MoVi once, and it makes the hip worse

`thorax := CLAV`, changing **only** H36M joint 8; pelvis, hips, knees and ankles
are exactly `phase15_checks.to_h36m` as before. Near limb, PG1.

| subject | arm | reference | hip MAE | hip bias | hip SD | hip r | knee MAE | knee r |
|---|---|---|---|---|---|---|---|---|
| Subject_1 | detected | original | 13.706 | −13.706 | 6.03 | +0.841 | 6.950 | +0.924 |
| Subject_1 | detected | **thorax-matched** | **24.216** | **−24.216** | 5.94 | +0.846 | **6.950** | +0.924 |
| Subject_1 | perfect | original | 8.291 | −7.235 | 5.33 | +0.885 | 3.674 | +0.956 |
| Subject_1 | perfect | **thorax-matched** | **17.747** | **−17.747** | 5.32 | +0.885 | **3.674** | +0.956 |
| Subject_10 | detected | original | 8.408 | −5.082 | 8.25 | +0.665 | 9.376 | +0.930 |
| Subject_10 | detected | **thorax-matched** | **13.832** | **−12.451** | 8.34 | +0.658 | **9.376** | +0.930 |
| Subject_10 | perfect | original | 8.440 | −3.347 | 9.30 | +0.587 | 5.638 | +0.972 |
| Subject_10 | perfect | **thorax-matched** | **13.509** | **−10.716** | 9.39 | +0.580 | **5.638** | +0.972 |
| Subject_12 | detected | original | 14.145 | −12.119 | 8.65 | +0.349 | 12.278 | +0.835 |
| Subject_12 | detected | **thorax-matched** | **21.240** | **−20.488** | 8.72 | +0.344 | **12.278** | +0.835 |
| Subject_12 | perfect | original | 9.689 | −3.498 | 11.16 | +0.112 | 5.354 | +0.893 |
| Subject_12 | perfect | **thorax-matched** | **15.175** | **−11.865** | 11.24 | +0.105 | **5.354** | +0.893 |
| Subject_13 | detected | original | 12.819 | −12.394 | 6.83 | +0.637 | 8.472 | +0.868 |
| Subject_13 | detected | **thorax-matched** | **22.651** | **−22.651** | 6.72 | +0.642 | **8.472** | +0.868 |
| Subject_13 | perfect | original | 10.834 | −5.699 | 10.56 | +0.183 | 4.732 | +0.936 |
| Subject_13 | perfect | **thorax-matched** | **18.073** | **−15.956** | 10.44 | +0.191 | **4.732** | +0.936 |
| *Subject_11* | *detected* | *original* | *10.107* | *−9.344* | *7.45* | *+0.709* | *9.828* | *+0.937* |
| *Subject_11* | *detected* | *thorax-matched* | *21.778* | *−21.778* | *7.77* | *+0.677* | *9.828* | *+0.937* |
| *Subject_11* | *perfect* | *original* | *9.019* | *+0.924* | *11.04* | *+0.502* | *5.401* | *+0.862* |
| *Subject_11* | *perfect* | *thorax-matched* | *14.460* | *−11.470* | *11.39* | *+0.455* | *5.401* | *+0.862* |
| *Subject_15* | *detected* | *original* | *4.448* | *−1.617* | *5.54* | *+0.869* | *6.470* | *+0.942* |
| *Subject_15* | *detected* | *thorax-matched* | *10.375* | *−10.312* | *5.56* | *+0.869* | *6.470* | *+0.942* |
| *Subject_15* | *perfect* | *original* | *9.004* | *−1.522* | *11.68* | *+0.442* | *7.220* | *+0.893* |
| *Subject_15* | *perfect* | *thorax-matched* | *14.702* | *−10.217* | *11.71* | *+0.442* | *7.220* | *+0.893* |

*Subject_11 and Subject_15 are diagnostic rows (italic), not in the rule set.*

**Three things read straight off this table.**

**1. The knee MAE is bit-identical in every single row.** 6.950 and 6.950,
9.376 and 9.376, 5.354 and 5.354 — to three decimals, 24 rows. **That is
structural, not luck.** `sagittal_angles` computes knee flexion as the signed
angle *from the thigh to the shank*, both projected into the same sagittal
frame; a difference of two angles measured in one frame is invariant to
rotating that frame within its plane. Changing the thorax rotates (fwd, up)
inside the plane but leaves the plane itself — set by the hip axis — alone.
**Hip flexion is measured against the frame's own down direction, so it moves;
knee flexion is a difference within the frame, so it cannot.** The thorax
definition is a hip-only quantity, by construction.

**2. The hip change equals the trunk-axis rotation, exactly.** The MoVi
THORAX→CLAV displacement is **68–100 mm** at a trunk length of **429–485 mm**,
which is up to ~11° of axis rotation; the measured hip bias shifts by
**−5.4 to −11.0°**. Subject_1: bias −7.235 → −17.747, a change of −10.5°,
against a 95.0 mm displacement on a 484.9 mm trunk. The arithmetic closes.

**3. It makes the hip worse on all six subjects, by a lot.** \|bias\| rises from
0.9–7.2° to 10.2–24.2°. **r barely moves** (±0.05), which is the signature
of a pure offset: the trace shape is untouched, only its zero point moves.

### A correction to my own reasoning from phase 15B

Phase 15B's H2 arm swapped **six joints at once** — thorax, spine, knees
(KJC→RKNE, 62 mm), ankles (AJC→RANK, 35 mm), elbows and neck — and I argued
that its rejection therefore could not be attributed to the thorax, and that a
thorax-only test might point the other way. **That was wrong.** The thorax-only
change here reproduces 15B's effect almost exactly (Subject_1 perfect-arm bias
−16.540 under the six-joint swap, −17.747 under thorax alone). **The thorax
accounts for essentially all of it**, and the knee and ankle redefinitions
contributed almost nothing to the *hip*. 15B's verdict was right for the reason
it gave.

### The addendum that resolves the contradiction

CLAV is right on AthleticsPose and wrong on MoVi, which cannot both be true of
one fixed definition — so the remaining question is **where the lifter's thorax
actually lands on MoVi input.** Measured the same way as step 2, perfect-2D arm,
per-frame lower-body alignment
(`results/phase15c_thorax_on_movi.json`):

| subject | dist to THORAX | dist to CLAV | sagittal offset vs THORAX | vs CLAV |
|---|---|---|---|---|
| Subject_1 | **79.6 mm** | 144.4 mm | **−5.70°** | −16.21° |
| Subject_10 | **75.6 mm** | 100.8 mm | **−1.77°** | −9.14° |
| Subject_11 | **77.9 mm** | 133.7 mm | **−0.56°** | −12.95° |
| Subject_12 | **81.3 mm** | 115.4 mm | **−1.26°** | −9.62° |
| Subject_13 | **77.2 mm** | 130.9 mm | **−4.60°** | −14.85° |
| Subject_14 | **52.1 mm** | 86.8 mm | **−0.82°** | −10.38° |
| Subject_15 | **78.3 mm** | 116.6 mm | **−0.29°** | −8.99° |

**Nearer Visual3D's `THORAX` on 7 of 7 subjects**, and within **0.3–5.7°** of
it in the sagittal plane against **9.0–16.2°** from CLAV.

On AthleticsPose the *same checkpoint* matched CLAV to **0.750°**.

> ### **The lifter's output thorax does not land on the same anatomical point in
> ### the two datasets.**
>
> This is not a reference error and not a marker-naming mistake. It is the
> model's **own output convention shifting between datasets** — ~8–15° of
> trunk-axis direction, which is 2–4× the entire AthleticsPose hip MAE.

### What phase 15C establishes

1. **The authors' recipe is verified joint by joint**, and the released
   `markers_h36m` thorax is `CLAV` to **5.04 mm** — the trunk joints being the
   three best-matched in the skeleton.
2. **On AthleticsPose, the lifter's thorax is `CLAV`**, by three independent
   measures: 0.750° trunk-axis offset, 5.04 mm reference match, and a hip MAE
   of 4.118° at −1.8° bias that reproduces the frozen baseline.
3. **The thorax definition is a hip-only quantity**, proved structurally by 24
   bit-identical knee rows — the knee is immune to it.
4. **The wrong thorax costs up to 22.8° on the tune set** and 5–11° on MoVi.
   Hip numbers are definition-limited before they are camera-limited.
5. **The answer does not transfer.** On MoVi the same checkpoint's thorax tracks
   Visual3D's `THORAX` on 7 of 7 subjects, 9–16° away from CLAV. **So no single
   thorax definition makes the hip comparable across both datasets**, and the
   hip angle as currently defined is not a transferable measurement.

### Decisions, and what is NOT being done

**The MoVi reference stays `MOVI_TO_H36M`, unchanged.** The pre-registration
said the chosen definition is applied **once** and reported **beside** the
original, and that is what happened. It is not adopted, for a reason the
addendum makes evidence-based rather than defaulted: **on MoVi the lifter's own
thorax is nearer `THORAX` on every subject**, so switching to CLAV would move
the reference *away* from what the model produces there.

**The phase 15 headline is unchanged: 9.610° near / 10.604° far on PG1 at
n = 2.** The sealed test set — `Subject_16`, `_17`, `_18`, `_19`, `Subject_2` —
**remains sealed.** The angle code and the lifter were not touched. Nothing was
chosen on MoVi.

**What I am not claiming:** that MoVi's hip bias is now explained. It is
*bounded* — the thorax is worth 5–11° of it, and the model's convention drift
is worth 9–16° of trunk direction — but no definition tested makes the hip
agree across both datasets, and why the lifter's thorax moves between them is
unresolved.

### What should happen next

**The hip and the knee should stop being reported as one measurement.** The knee
is convention-immune by construction and stable in every table in this phase
and in 15B (r 0.84–0.97 everywhere). The hip is dominated by a definitional
choice that the model itself does not hold fixed across datasets.

The defensible next step is to **report the knee as the measurement and the hip
with its definitional uncertainty stated**, and — if a transferable hip angle is
wanted — to define it against something the model does hold fixed, such as the
pelvis axis from the hip markers, rather than against a trunk the model places
differently in different datasets. That is a change to the angle definition,
which is frozen, so it belongs in a pre-registered phase of its own and **not**
in a retrofit of these numbers.
