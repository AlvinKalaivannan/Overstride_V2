# Phase 15D — segment angles against gravity, instead of against the trunk

> **PRE-REGISTRATION, written 2026-11-03.** Committed **alone, before anything
> is run.** Definitions, the ruled-out option, the vertical conventions and both
> gates are fixed here. Results are appended under "Results"; nothing above that
> heading is edited afterwards.

## Why

The hip angle is measured against the **pelvis → thorax** axis, and phase 15C
established that **the lifter does not place its own thorax on the same
anatomical point in two datasets**: `CLAV` to within **0.750°** on
AthleticsPose, but **87–144 mm** from CLAV on MoVi, nearer Visual3D's `THORAX`
on 7 of 7 subjects. That is 8–16° of trunk direction, two to four times the
entire AthleticsPose hip MAE.

So the trunk-referenced hip angle carries a convention term the model itself
does not hold fixed across labs, and the per-joint split (`results/phase15.md`)
shows the consequence: the hip degrades **3.90×** across labs while the knee
degrades **2.00×**, and the ordering of which joint is harder **inverts**
between datasets.

**This phase asks whether referencing to gravity instead of to the trunk gives
a quantity that transfers.**

## The option that is ruled out, and why — recorded before anything is run

**A pelvis-referenced hip angle is not available from H36M-17.** Measured
before this commit:

| reference | \|pelvis − midpoint(hips)\| | hip width | ratio |
|---|---|---|---|
| MoVi `MOVI_TO_H36M` | **0.000 mm** | 225.3 mm | **0.0%** |
| AthleticsPose released `gt_markers3d_by_cam` | **8.96 mm** | 210.5 mm | **4.3%** |

In the MoVi reference the pelvis is `mean(LFWT, LBWT, RFWT, RBWT)` and each hip
is `mean(ASIS, PSIS)` of its own side, so **the pelvis is exactly the midpoint
of the two hips** — the three pelvic points are **collinear by construction**
and span no plane. There is no sagittal direction to reference against, at all.

In the released AthleticsPose reference the pelvis is a separate `PELVIS`
marker, so it is not exactly collinear — but it sits **8.96 mm** off the hip
midpoint on a 210.5 mm hip width. A 9 mm lever arm cannot define a sagittal
direction to any useful precision; it is comparable to the per-joint residuals
in `results/phase15c.json` (hips 11.0–11.3 mm).

**So the option is ruled out on both datasets** — exactly in one case, and
effectively in the other. It is recorded here so that it is not re-proposed as
an obvious fix.

## New quantities

All three in the **sagittal plane**, defined as the plane perpendicular to the
hip-to-hip axis, measured **against vertical**:

| quantity | vector |
|---|---|
| **thigh inclination** | hip → knee |
| **shank inclination** | knee → ankle |
| **trunk lean** | pelvis → thorax |

**Sign convention:** inclination is `atan2(v·fwd, v·up)` with
`fwd = normalise(cross(ml, up))` and `up` the vertical, so **positive is
anterior** (forward of vertical) and zero is straight down the vertical.

**The knee identity, and the test.** `sagittal_angles` computes knee flexion as
`signed_angle(thigh, shank)`, which expands to
`atan2(shank·fwd, shank·up) − atan2(thigh·fwd, thigh·up)`. So

> **knee = shank inclination − thigh inclination**

— note the order: **shank minus thigh**, not thigh minus shank. That difference
is invariant to rotating `(fwd, up)` within the plane, which is why the knee is
unaffected by the thorax definition and why it must come out **identical**
whether `up` is the trunk axis or gravity. **A test asserts that identity
against the existing frozen function**, and if it fails the segment definitions
are wrong, not the knee.

## Vertical — two different sources, deliberately

**For the ESTIMATE: the level-camera assumption.** The deployable tool has no
extrinsics and never will on a phone clip, so it assumes the image vertical is
gravity. In the lifter's output frame that is **`up = (0, −1, 0)`** — verified
before this commit: `head − pelvis` has mean y of **−0.4635** (axis 1,
negative), and the 2D detections put the head at *smaller* image y, the usual
y-down image convention.

**For the REFERENCE: mocap world vertical.**
- **MoVi:** world **axis 2**, established by check 2 of phase 15 two independent
  ways (both PG rigs sit at 1.35–1.38 m on it; head−pelvis agrees).
- **AthleticsPose:** the world vertical of `raw_markers_in_world`, carried into
  the released camera frame by the per-frame rotation already built and
  validated in `scripts/phase15c_thorax.py:align_per_frame`.

**The calibrated PG extrinsics are used only to report how much the
level-camera assumption costs** — camera pitch and roll in degrees, and the
resulting error on each angle. **They are never used inside the estimate.** The
estimate must stay exactly as deployable as it is today.

## Gates, fixed now

**G1 — tune set.** On AthleticsPose **S11 / S13 / S16**, running and sprint:
**thigh-inclination MAE no worse than the current hip MAE there, 4.118°**
(`results/phase15c.json` step 4, CLAV thorax), reported with a
**subject-bootstrap CI**.

**G2 — MoVi.** On diagnostic subjects **1, 10, 12, 13** (with **11** and **15**
as labelled rows): **thigh-inclination \|bias\| smaller than the
trunk-referenced hip \|bias\|** on **at least 3 of 4** subjects.

**Trunk lean has no gate.** It is reported as its own measurement with MAE and
bias. **It is expected to carry the lifter's transfer error** — that is precisely
why it is reported rather than hidden inside the hip angle. If trunk lean is bad
and thigh inclination is good, the phase has succeeded in separating the two.

## Standing rules

- **Nothing is fitted to MoVi.**
- **The existing angle functions are not changed.** `sagittal_angles` keeps
  producing the pre-registered numbers; the segment functions are added
  **beside** it.
- **Every existing figure stays**, including the combined headline
  **9.610° near / 10.604° far on PG1 at n = 2**.
- **The sealed test set** — `Subject_16`, `_17`, `_18`, `_19`, `Subject_2` —
  **is opened only after 15D's definitions and code are frozen and committed**,
  and then only for the full frozen pipeline.

## What would make this phase wrong

- **Letting the knee identity test slide.** If `shank − thigh` does not
  reproduce `sagittal_angles`' knee to floating-point tolerance, the new
  definitions are wrong and no result from them means anything.
- **Using extrinsics in the estimate** to make the level-camera number look
  better. The whole point is that a deployable clip has none.
- **Reading a passed G2 as "the hip problem is solved".** G2 compares biases; a
  smaller bias on a different quantity is progress on transferability, not a
  correction to the hip numbers already reported.

---

## Results

Run 2026-11-03. `scripts/phase15d_segments.py` (added beside the frozen angle
code, which is unchanged), `scripts/phase15d_run.py`, `results/phase15d.json`.
G1 on **231 length-matched AthleticsPose clips**, S11/S13/S16, running and
sprint. G2 on MoVi PG1.

> ### Verdicts, against the thresholds fixed in `e19980f`
>
> | gate | rule | result | verdict |
> |---|---|---|---|
> | **G1** | thigh-inclination MAE ≤ 4.118° on AthleticsPose | **6.694°** [6.290, 7.390] | **FAIL** |
> | **G2** | thigh \|bias\| < trunk-referenced hip \|bias\| on ≥ 3 of 4 MoVi subjects | **4 of 4** | **PASS** |
>
> **Both outcomes are informative, and together they are the point.** Gravity
> referencing is **worse on the tune set** and **much better across labs**.

### The knee identity holds on real data

**Maximum \|deviation\| between `shank − thigh` and the frozen
`sagittal_angles` knee, over all 231 clips: 1.42 × 10⁻¹³ degrees.**
Floating-point noise. The new definitions are consistent with the frozen ones,
so everything below is a change of *reference direction* and nothing else.

### G1 — AthleticsPose, and the trunk reference is genuinely helping here

| quantity | MAE | bias | 95% CI |
|---|---|---|---|
| **thigh inclination** | **6.694°** | −1.154 | [6.290, 7.390] |
| shank inclination | 6.358° | +1.448 | [5.910, 7.163] |
| trunk lean | 5.538° | −0.569 | [5.133, 6.321] |
| knee (rebuilt from segments) | 5.546° | +2.602 | [5.109, 6.236] |
| **(trunk-referenced hip, same clips)** | **3.697°** | −1.723 | [3.323, 3.890] |

**G1 fails, and not marginally: 6.694° against a 4.118° bar, with the CI
nowhere near it.** On the tune set the trunk-referenced hip is **1.8× better**
than gravity-referenced thigh inclination — 3.697° against 6.694°.

**Why, and it is not a defect in the new quantity.** Where the trunk reference
is *correct* — and phase 15C verified it is, on AthleticsPose, to 0.750° —
measuring the thigh **relative to the trunk cancels common-mode rotation.** If
the whole estimated skeleton is tilted slightly against the reference, a
trunk-relative angle subtracts that tilt out; a gravity-relative angle keeps it.
The level-camera assumption introduces exactly such a tilt, and
`sagittal_angles` was in effect getting it for free.

So **the trunk reference is not merely a liability. It is a variance-reduction
device that works whenever the model's thorax is right**, which on the tune set
it is.

### G2 — MoVi, where the trunk reference is wrong and it costs dearly

Near limb, PG1. `|bias|` is what G2 is judged on; r and MAE are reported
alongside.

| subject | arm | thigh MAE | thigh bias | thigh r | trunk MAE | trunk bias | hip bias (trunk-ref) | \|thigh\| < \|hip\|? |
|---|---|---|---|---|---|---|---|---|
| Subject_1 | detected | 6.508 | **−5.722** | +0.855 | 7.987 | −7.984 | **−13.706** | **yes** |
| Subject_1 | perfect | 3.487 | −3.037 | +0.932 | 5.992 | −4.198 | −7.235 | yes |
| Subject_10 | detected | 4.857 | **−2.866** | +0.832 | 4.850 | −2.216 | **−5.082** | **yes** |
| Subject_10 | perfect | 3.387 | −2.534 | +0.920 | 6.543 | −0.813 | −3.347 | yes |
| Subject_12 | detected | 7.374 | **−7.225** | +0.756 | 8.204 | −4.894 | **−12.119** | **yes** |
| Subject_12 | perfect | 4.751 | −4.600 | +0.785 | 8.856 | +1.101 | −3.498 | no |
| Subject_13 | detected | 6.390 | **−6.079** | +0.738 | 7.654 | −6.315 | **−12.394** | **yes** |
| Subject_13 | perfect | 5.099 | −5.011 | +0.813 | 7.791 | −0.688 | −5.699 | yes |
| *Subject_11* | *detected* | *6.272* | *−5.953* | *+0.822* | *6.645* | *−3.392* | *−9.344* | *yes* |
| *Subject_11* | *perfect* | *2.533* | *−1.841* | *+0.919* | *9.275* | *+2.765* | *+0.924* | *no* |
| *Subject_15* | *detected* | *4.178* | *−0.212* | *+0.864* | *4.131* | *−1.405* | *−1.617* | *yes* |
| *Subject_15* | *perfect* | *3.445* | *−2.154* | *+0.884* | *7.135* | *+0.632* | *−1.522* | *no* |

*`Subject_11` and `Subject_15` are diagnostic rows (italic), not in the rule
set. G2 is judged on the detected arm, near limb, over subjects 1/10/12/13.*

> **G2 PASSES 4 of 4.** Thigh \|bias\| is roughly **half** the trunk-referenced
> hip \|bias\| on every rule subject: 5.72 vs 13.71, 2.87 vs 5.08, 7.23 vs
> 12.12, 6.08 vs 12.39.

**And the correlation improves far more than the bias.** Thigh r is **+0.74 to
+0.86** on every subject, detected, and **+0.79 to +0.93** with perfect 2D. The
trunk-referenced hip's r on the same subjects was **+0.11 to +0.84**, including
**0.112** and **0.183** on Subject_12 and Subject_13 — those two had almost no
hip signal at all. **Referencing to gravity restores it.** That is a bigger gain
than G2's bias rule was written to detect.

### The exact decomposition — the finding of this phase

The additivity below is **exact to floating point**, residual ~10⁻¹⁵ on all
twelve rows above:

> ## hip bias (trunk-referenced) = thigh bias (gravity) + trunk-lean bias

| subject | arm | thigh | + trunk | = sum | hip (measured) | residual |
|---|---|---|---|---|---|---|
| Subject_1 | detected | −5.722 | −7.984 | **−13.706** | −13.706 | −5.3e−15 |
| Subject_1 | perfect | −3.037 | −4.198 | **−7.235** | −7.235 | −3.6e−15 |
| Subject_10 | detected | −2.866 | −2.216 | **−5.082** | −5.082 | −1.8e−15 |
| Subject_12 | detected | −7.225 | −4.894 | **−12.119** | −12.119 | −1.8e−15 |
| Subject_13 | detected | −6.079 | −6.315 | **−12.394** | −12.394 | +1.8e−15 |
| Subject_15 | detected | −0.212 | −1.405 | **−1.617** | −1.617 | −3.6e−15 |

*(all twelve rows in `results/phase15d.json`; asserted on a synthetic skeleton
in `tests/test_segment_angles.py`)*

**So the hip error was never one quantity.** It is a **limb term** plus a
**trunk term**, and the frozen hip angle sums them without saying so. Phase 15D
separates them, and the separation says which half fails to transfer:

| | AthleticsPose | MoVi (detected, rule subjects) |
|---|---|---|
| **limb term** (thigh) | 6.694° MAE, −1.154 bias | 4.86–7.37° MAE, −2.87 to −7.23 bias |
| **trunk term** (lean) | 5.538° MAE, −0.569 bias | 4.85–8.20° MAE, −2.22 to −7.98 bias |

**The trunk term is where the transfer failure lives**, exactly as the
pre-registration predicted when it gave trunk lean no gate and said it was
expected to carry the lifter's transfer error. On AthleticsPose its bias is
−0.569°; on MoVi it reaches −7.98°. That is the same convention drift phase
15C measured as 8–16° of trunk direction, now expressed as an angular error
term that can be added and subtracted.

### What the level-camera assumption costs

The deployable tool has no extrinsics, so it assumes the image vertical is
gravity. The calibrated PG extrinsics are used **only** to price that, never
inside the estimate.

| camera | pitch | roll | total tilt |
|---|---|---|---|
| **PG1** (side-on, used here) | **−3.598°** | **+0.859°** | **3.699°** |
| PG2 (near-frontal) | −10.251° | −0.761° | 10.280° |

Per-subject additive error on **every** inclination, measured per frame as the
in-plane component:

| subject | mean | sd | min | max |
|---|---|---|---|---|
| Subject_1 | −1.599° | 1.671 | −3.711 | +0.947 |
| Subject_10 | −0.749° | 2.110 | −3.738 | +3.719 |
| Subject_12 | −1.011° | 2.019 | −3.702 | +3.699 |
| Subject_13 | −1.919° | 1.615 | −3.728 | +0.853 |
| *Subject_11* | *−1.488°* | *1.956* | *−3.724* | *+2.206* |
| *Subject_15* | *−1.531°* | *1.819* | *−3.714* | *+1.790* |

**Total tilt overstates the sagittal cost, and the decomposition shows why.**
For a **side-on** camera the pitch tilts the image vertical inside the plane
containing the optical axis and gravity — which for a side-on subject is the
**frontal** plane, where a sagittal angle does not see it. Only the component
inside the sagittal plane counts. So PG1's 3.699° of tilt costs about
**1–2° of bias**, not 3.7°.

**The spread is the subject turning, not camera noise.** The range runs to
±3.7° — the full tilt — because the sagittal plane rotates with the direction
of travel during a jogging trial, so the pitch projects into it by a varying
amount. Hence ~2° of sd on top of ~1–2° of bias.

**This is a floor on any gravity-referenced angle from an uncalibrated clip**,
and it is now a measured number rather than an assumption. It accounts for
roughly a quarter to a half of the thigh bias on MoVi — so most of that bias is
still the estimate, not the camera.

### What phase 15D establishes

1. **The segment definitions are consistent with the frozen code**, to
   1.42×10⁻¹³ degrees on 231 real clips.
2. **The frozen hip angle is exactly a sum of two terms** — thigh inclination
   plus trunk lean — and reporting it as one number conflates a limb
   measurement with a trunk measurement.
3. **The trunk term is what fails to transfer.** Its bias is −0.57° on
   AthleticsPose and up to −7.98° on MoVi.
4. **G1 fails: gravity referencing is worse on the tune set**, 6.694° against
   3.697°, because a correct trunk reference cancels common-mode rotation and
   gravity referencing gives that up.
5. **G2 passes 4 of 4: gravity referencing transfers much better**, halving the
   bias and — more importantly — raising thigh r to +0.74–+0.86 where the
   trunk-referenced hip had collapsed to +0.11 and +0.18 on two subjects.
6. **The level-camera assumption costs ~1–2° of bias and ~2° of sd** on a
   side-on camera with 3.7° of tilt, and total tilt overstates it because only
   the in-plane component matters.

### What this does NOT change

**The phase 15 headline is unchanged: 9.610° near / 10.604° far on PG1 at
n = 2**, and every figure in `results/phase15.md`, `15b` and `15c` stands.
Nothing was fitted to MoVi. The existing angle functions were not modified. The
**sealed test set** — `Subject_16`, `_17`, `_18`, `_19`, `Subject_2` — remains
**sealed**.

**G2 passing does not mean the hip problem is solved**, and the pre-registration
said so in advance. It means a gravity-referenced limb angle transfers better
than a trunk-referenced one. It is not a correction to any hip number already
reported.

### What should happen next

The two gates point in opposite directions, which is a real trade rather than a
tie: **the trunk reference wins within a lab and loses across labs.** A
defensible reading is that there are two different reportable quantities —
a within-lab angle and a cross-lab one — and that the choice should be stated
rather than settled.

**The sealed set is now the right instrument**, and 15D's definitions and code
are frozen by this commit, which was the pre-registered precondition for opening
it. The decision that should be made before it is opened is **which quantity the
sealed test reports**, written down first: the frozen trunk-referenced angles,
the gravity-referenced segment angles, or both side by side with this phase's
decomposition. **That choice is the operator's, and it must be fixed before the
data is touched.**
