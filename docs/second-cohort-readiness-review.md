# Notes on "Second Cohort Readiness"

**Responding to:** the readiness companion to the validation plan — R1 (adapter
contract), R2 (conformance tests), R3 (site as a fifth negative control), R4
(frozen model and pre-registered falsification), plus acquisition routes and the
de-identification trap.
**Method:** same as `docs/validation-plan-review.md` — every premise testable
against this repository was tested, with commands shown.

---

## Verdict

**The readiness argument is right and I would act on it. Its central strategic
claim is not currently supported by the numbers it rests on.**

The framing — that "get a second cohort" bundles a blocked external dependency
with weeks of unblocked local work, and that the local half pays for itself
regardless — is correct and useful. R1 through R4 are each independently
justified, and two of them are things I would adopt whether or not a second
cohort ever appears.

But §2, "the finding pays for the cohort", uses the phase 4 offset as a
conversion constant. **Phase 4 did not measure that offset precisely enough to
convert with.** That does not sink the idea; it reorders when the idea becomes
usable.

---

## 1. The conversion argument does not hold yet

The claim: because monocular video costs a measured 0.027–0.033 AUC, a partner
clinic can collect with a phone rather than a motion capture lab, and "a camera
cohort landing at 0.583 is consistent with a mocap-equivalent 0.610–0.616,
because phase 4 measured exactly that gap."

What phase 4 actually reports:

| bank | fps | AUC | ΔAUC | 95% CI | excludes 0? |
|---|---|---|---|---|---|
| lateral only | 30 fps | 0.583 | −0.033 | **[−0.065, +0.001]** | **no** |
| lateral only | mocap | 0.587 | −0.028 | [−0.067, +0.015] | no |
| all views | mocap | 0.586 | −0.029 | [−0.085, +0.027] | no |

**Not one delta interval excludes zero.** Carrying the interval instead of the
point estimate:

```
camera-collected AUC (deployment config): 0.583
measured offset: -0.033  CI [-0.065, +0.001]
  mocap-equivalent using the POINT estimate : 0.615
  mocap-equivalent using the INTERVAL       : 0.582 to 0.648
  interval width: 0.066 AUC
```

For scale: the ceiling is 0.610 and the pre-registered bar is 0.600. **The
conversion interval is seven times wider than the gap between them.** A camera
cohort at 0.583 is consistent with a mocap-equivalent that clears the bar
comfortably *and* with one that sits below the current ceiling. It cannot
distinguish replication from failure, which is the one thing the conversion was
supposed to do.

### Three further problems, in increasing order of seriousness

**Transportability.** The offset was measured by injecting residuals into *these*
Ferber waveforms on *this* cohort. Nothing establishes it carries to a different
population with a different injury mix.

**Additivity.** AUC differences are not additive. The delta depends on the score
distribution and the base rate, so subtracting a delta measured at 0.615 from a
different cohort's 0.583 is not a defined operation, only an approximation whose
error is unquantified.

**The real camera penalty is larger, and currently unbounded.** Phase 4's
residuals came from AthleticsPose 2D→3D lifting with a checkpoint fine-tuned on
that rig. A phone-camera cohort adds, on top of that: the generic-detector
penalty (+0.54°, measured in phase 7), video decode and person detection (not
measured — AthleticsPose releases no source video), and domain gap (not
measured). `scripts/video_kinematics.py` prints exactly this list as its error
budget on every run. So the conversion needs an offset **larger than the one
measured, by an unknown amount.**

### What would make the argument work

The idea is sound in principle and worth preserving. It becomes usable when the
conversion constant is re-derived rather than reused:

1. **After B1**, take the error measured on *real video with ground truth* —
   decode, detection and lifting together, not lifted keypoints alone.
2. **Re-inject that error** into the Ferber waveforms through the existing phase
   4 machinery (`build_features` in `scripts/phase4_real_delta.py` already takes
   an arbitrary residual bank).
3. **Report the new delta with its interval**, and state up front how wide the
   interval must be for the conversion to discriminate. If it stays at ±0.03, the
   conversion still cannot do the job and camera collection has to be justified on
   cost grounds alone rather than as a measurable equivalence.

That sequencing also makes the readiness case *stronger*, not weaker: it gives B1
a second reason to exist beyond validating the tool.

---

## 2. What I would adopt regardless

**"The negative controls define the metadata contract."** This is the best idea in
either document. The four controls consume specific fields —

```
provenance   : yrs_missing, lvl_missing, year
demographics : age, Height, Weight, speed_r, Gender
structure    : n_marker_channels, n_joints_landmarks, n_neutral_markers, ...
dominant leg : DominantLeg
```

— and a cohort that does not ship them cannot be validated, no matter how good
its kinematics are. That turns a vague ask ("can we have your data?") into a
specific schedule in a data-use agreement. It is worth writing down even if no
partner is ever approached, because it is also the specification the adapter in
R1 has to satisfy.

**The de-identification trap.** Standard de-identification strips dates.
`year` is a date-derived field and it sits in the provenance control — the
control that exists because collection wave produced the 0.775 artefact in phase
1. So a cohort de-identified by the usual defaults arrives **unable to support the
control that matters most**, and nothing surfaces the problem until ingestion.
Asking up front for a coarsened but non-constant wave indicator is cheap; asking
afterwards means re-opening an agreement. This is the single most actionable item
in the document and it belongs in the first paperwork conversation.

**R3's reasoning about what differencing cancels.** Correct and precisely stated:
a left-minus-right difference within one session cancels any site offset applied
symmetrically to both limbs, exactly as it cancels demographics — but it does not
cancel site differences in *noise level* or in how asymmetry is measured. The
proposed control (fit site from the within-subject difference features; require
chance) is the right test and mirrors the existing negative-control pattern, so it
drops straight into `phase1_eval.evaluate` with no new machinery.

**R4's timing argument.** A frozen model applied to new data answers a question;
the same model tuned after seeing that data answers nothing, and no amount of good
intent recovers the difference. Freezing costs days now and is unavailable later.

---

## 3. Smaller notes

**R1 slightly overreaches on §7.8.** The adapter boundary "is the only thing that
actually retires" the MATLAB concern — it retires the *barrier* for future work,
and makes Fukuchi a second implementation that proves the boundary is real. But
every existing published result still came through the patched pipeline. The
adapter improves what happens next; it does not retrospectively validate what has
already been computed. The honest claim is that it retires the barrier, not the
error source.

**R2's strongest test is the one it lists last.** "Every provenance field the
negative controls consume, present and non-constant" is the check that will
actually fire, and it is the same failure the de-identification trap describes.
Worth promoting to first, since it is the one that determines whether a cohort is
analysable at all rather than merely well-formed.

**The acquisition ranking is sensible**, and the Ferber-authors route is
genuinely the cheapest email available. Worth noting it costs nothing to send
while R1–R4 proceed.

---

## 4. Sequencing

Nothing here changes the validation plan's order. It adds work that runs
alongside it:

- **R1, R2, R3 start now** — all local, all independently justified, none
  dependent on any external answer.
- **R4 before any second cohort exists**, which is the whole point of it.
- **§2's conversion argument waits for B1**, and should be re-derived there rather
  than carried forward as written.
- **The de-identification ask goes into the first conversation** with any partner,
  including the Ferber authors.

The readiness document's own test — does this still pay if the cohort never
arrives? — is the right one, and R1 through R4 each pass it. §2 does not pass it,
because its value is entirely contingent on an acquisition happening *and* on an
offset tight enough to convert with. That is the one part to hold loosely.
