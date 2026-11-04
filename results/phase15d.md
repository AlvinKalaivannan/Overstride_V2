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

*Not yet run at the time of this commit.*
