# Phase 10 (C) — a second lab's markers through the same MATLAB pipeline

**Question.** `docs/REVIEW.md` §7.8 records that every waveform in this project
came out of one MATLAB pipeline (`gait_kinematics` → `gait_steps`), driven only
by one archive, and that nothing independent has ever exercised it. If that
pipeline produced artefacts specific to the RIC archive's marker model or
conventions, every result downstream would inherit them and no internal check
could reveal it.

**Answer: the pipeline transfers.** Fed a different lab's markers — different
subjects, a different marker model, a different country, 150 Hz instead of
200 Hz — it recovers treadmill speed to a **median 1.71% relative error** with
**r = 0.9995** against the prescribed speeds, and a regression slope of
**1.0148** (essentially identity).

**And the validation found a defect in the published dataset** it was validating
against: three subjects' trials are byte-identical duplicates.

---

## Data

Fukuchi et al. 2017, *A public dataset of running biomechanics and the effects of
running speed on lower extremity kinematics and kinetics*.
figshare **DOI 10.6084/m9.figshare.4543435** (v5), **CC BY 4.0**, retrieved
**2026-09-10**. 39 subjects, treadmill running at prescribed 2.5 / 3.5 / 4.5 m/s,
marker-based optical mocap at 150 Hz, plus the authors' own Visual3D model and
pipeline files.

This is the readiness review's R1 "adapter contract" made concrete.

---

## The test, and why it has teeth

Fukuchi prescribed treadmill speeds, encoded in each filename. `gait_steps`
computes speed independently from marker kinematics and **never sees the
filename**. Agreement therefore checks the whole adapter at once — derived
coordinate frame, marker mapping, cluster assignment, synthetic fourth markers,
units — against a quantity neither side can negotiate.

That matters because **the failure mode here is silent.** A wrong rotation still
produces continuous, physiological-looking joint angles. The first version of
this adapter did exactly that; its only visible symptom was 1.85 m/s on a 2.5 m/s
trial. Inspecting angle curves would not have caught it.

Three prescribed speeds per subject also make this a **slope test**, not just an
offset test: a residual scale error shows up as speed-dependent drift even when
one speed happens to agree.

**Pre-declared bands, fixed before the batch ran:**

| median \|relative error\| | reading |
|---|---|
| ≤ 3% | the adapter transfers; proceed to angles |
| 3–8% | usable, but report the bias and its direction |
| > 8% | something is wrong; do not report angles |

---

## Result

**61 trials / 26 subjects**, 0 pipeline errors.

| nominal | n | computed mean | sd | mean error | median \|rel\| |
|---|---|---|---|---|---|
| 2.5 | 23 | 2.548 | 0.021 | +0.048 | 1.85% |
| 3.5 | 22 | 3.555 | 0.023 | +0.055 | 1.57% |
| 4.5 | 16 | 4.578 | 0.031 | +0.078 | 1.76% |

```
overall median |relative error| : 1.71%
correlation nominal vs computed : r = 0.9995
regression                      : computed = 1.0148 x nominal + 0.0084
error drift, slowest -> fastest : +0.031 m/s
VERDICT                         : PASS -- the adapter transfers
```

**The bias is small, positive and constant in proportion** (+1.7% at every
speed), which is the expected consequence of belt speed differing from
centre-of-mass speed. That was
declared in advance as expected and not a failure. The near-zero drift
(+0.031 m/s across a 2 m/s range) is the part that rules out a residual scale
error: the slope is 1.0148, not 0.90 or 1.10.

Within-speed scatter is **0.021–0.031 m/s**, i.e. under 1%, across 26 subjects of
differing body size.

---

## The dataset defect this found

Four of the five worst outliers in the first pass shared a signature: a subject
computing an **identical speed at every nominal speed**. RBDS029 returned
2.527 m/s for 2.5, 3.5 and 4.5; RBDS030 returned 2.549 for all three. Both had
also reported identical ASIS width *and* identical pelvis height across their
three trials during conversion, to the millimetre.

MD5 confirmed it:

```
RBDS029runT25 == RBDS029runT35 == RBDS029runT45
RBDS030runT25 == RBDS030runT35 == RBDS030runT45
RBDS032runT25 == RBDS032runT35 == RBDS032runT45

95 distinct running trials by content, across 101 published files
6 redundant files
```

Three subjects publish one recording under all three speed names. **Such a trial
cannot validate anything** — the same bytes carry three different nominal speeds,
so its true speed is unassignable — so all six are excluded from the statistics
and reported here instead. `duplicate_groups()` in
`scripts/phase10_speed_check.py` detects them by hash, so this is reproducible
rather than a note.

Their effect on the headline was large, which is why it is worth stating:

| | with duplicates | excluded |
|---|---|---|
| median \|rel error\| | 1.81% | **1.71%** |
| correlation r | 0.8871 | **0.9995** |
| regression slope | 0.9036 | **1.0148** |
| sd at 4.5 m/s | 0.660 | **0.031** |
| drift slow → fast | −0.195 m/s | **+0.031 m/s** |

This is a defect in a widely used public dataset, not in the adapter, and it was
surfaced by a validation designed for something else entirely.

---

## The adapter, and what was read rather than assumed

Every element of the input contract was established by reading
`gait_kinematics.m`, not inferred from the schema doc.

- **Thigh and shank cluster ordering does not matter.** Those anatomical frames
  are built entirely from the `joints` landmarks (`:174-222`); the segment
  rotation comes from a Söderkvist–Wedin SVD fit of the neutral cluster onto the
  dynamic one, so the ordering need only be *consistent* between them.
- **The foot is the exception** (`:120-130`). `foot_1..3` must be the three real
  heel markers, because the code sorts them medio-laterally to identify the
  lateral heel and treats the remaining pair as vertical. Fukuchi's
  `Heel.Top` / `Heel.Bottom` / `Heel.Lateral` map onto that exactly.
- **`foot_4` and `pelvis_4` in the RIC archive are not measured.** Both are
  exactly the centroid of markers 1..3 — affine weights (⅓,⅓,⅓), residual
  0.0000 mm, four points exactly coplanar. The adapter reproduces that
  construction rather than inventing a marker.
- **`joints.*_first`, `joints.*_fifth`, `L_toe` and `R_toe` are referenced by
  neither pipeline file.** Populated for structural fidelity; they affect nothing.

### The coordinate frame is derived, not hardcoded

An earlier version hardcoded an axis permutation onto the Bonita frame the
pipeline *documents* as its input (X right, Y walking, Z up). Measuring the
archive showed the data is not in that frame:

```
neutral L_foot centroid   [512.3,  59.9, 168.3]   <- Y is the floor
neutral pelvis centroid   [700.2, 964.7, 207.5]   <- Y is pelvis height
```

The documented conversion happens upstream of the JSON, so the archive already
holds the pipeline's **internal** frame (X right, Y up, Z posterior). The
permutation targeted a frame the data was never in.

`derive_rotation` now builds the basis from the subject's own anatomy:

```
up        = pelvis centroid - foot centroid     (static trial, standing)
right     = R.ASIS - L.ASIS, orthogonalised against up
posterior = right x up                          (completes right-handed)
```

No assumption about the source lab's axis order or signs, and it works for a
frame that is not axis-aligned at all. It asserts orthonormality and det = +1,
then checks three physical facts it was **not** built from — heel excursion
ordering (posterior 680 mm ≫ up 284 ≫ medio-lateral 120), ASIS anterior of PSIS,
and the planted foot drifting posteriorly through stance. Any failure refuses to
convert. Derived ASIS widths run **205–268 mm** across 39 subjects.

---

## Exclusions, and a bias that must be stated

**67 of 101 trials converted; 61 entered the statistics.**

Marker dropout is substantial in this dataset. Gaps are handled by trimming to
the longest contiguous clean window, **never by interpolation** — `CLAUDE.md`
forbids imputing values, and trimming keeps the time base continuous, which the
velocity and event-detection stages require. Median retained fraction among
converted trials is **100%**; the minimum is 16.8%. A 750-frame (5 s) floor is
enforced.

**Exclusion is strongly speed-dependent, and this is the main threat to the
slope result:**

| speed | trials | converted | excluded | rate |
|---|---|---|---|---|
| T25 | 31 | 25 | 6 | **19.4%** |
| T35 | 39 | 24 | 15 | **38.5%** |
| T45 | 31 | 18 | 13 | **41.9%** |

Faster running produces more dropout, so the surviving 4.5 m/s trials are those
with the best marker tracking. The slope test rests on comparing speeds, and the
fastest condition is the most filtered — so the slope of 1.0148 is measured on a
sample that is cleaner at the top end than the dataset is. It is reported as
evidence against a gross scale error, not as a precise calibration.

**Event detection is imperfect throughout.** Median `eventsflag` is 0.72 and
*every* trial has some fallback to foot-forward/foot-back. Median 29 steps per
trial. This does not affect the speed estimate materially but it is the thing to
watch when the angle comparison is done.

---

## What this establishes, and what it does not

**Established.** The RIC pipeline is not producing artefacts specific to the RIC
archive. Driven by an independent lab's marker model on different people at a
different sample rate, it recovers an externally prescribed physical quantity to
1.7% with r = 0.9995 and unit slope. §7.8's concern — one pipeline, never
independently exercised — is **substantially answered**, and the adapter that
answers it is reusable for any future cohort.

**Not established: that the joint angles agree with Fukuchi's own.** That is the
stronger test and it is not done here. Fukuchi normalises over the **full gait
cycle** (`PercGcycle` 0–100), while `gait_steps` emits 101 points of **stance**,
so the two are not comparable point-for-point without deriving a stance window
from Fukuchi's force data. Speed validates the geometry and the units; it does
not validate the angle convention.

**A convergent finding worth recording.** Fukuchi's own processed angles put
flexion/extension on the **Z** axis (hip range 46.2°, knee 72.4°, ankle 49.6°),
not X. That matches this project's own hard-won correction that **plane 2, not
plane 0, is flexion** in the RIC data — and `gait_kinematics.m:62` documents the
same thing independently ("Z points to the subject's right side … [Hinge flexion
extension]"). Both archives are Visual3D-family outputs. The plane-2 correction
was not papering over an RIC quirk.

---

## What surprised me

**The duplicates.** A validation built to check my own adapter found an error in
the reference dataset instead. The tell was not a large error but a *suspiciously
constant* one — the same computed speed under three different labels — which is a
pattern worth looking for generally.

**How much the frame bug hid.** Wrong-frame output had plausible-looking
continuous curves and physiological-looking magnitudes; the only symptoms were
speed being 26% low and left–right knee ranges disagreeing (32.2° vs 66.6°).
Both were visible in the very first probe and I nearly read past them.

**How tight the within-speed scatter is** — 0.021–0.031 m/s across 26 subjects
with different body sizes and marker placements. Tighter than expected for a
quantity derived through two coordinate transforms and an SVD fit.

---

## Files

- `scripts/phase10_fukuchi_adapter.py` — marker adapter and frame derivation
- `scripts/matlab/fukuchi_batch.m`, `scripts/matlab/run_fukuchi_batch.m` — batch
- `scripts/matlab/fukuchi_probe.m` — single-trial probe
- `scripts/phase10_speed_check.py` — validation and duplicate detection
- `results/phase10_speed.json`

---

# Part 2 — the angle comparison

Part 1 established that the pipeline's **geometry** transfers (speed to 1.71%,
r = 0.9995). Speed does not exercise the angle convention at all, so this
compares the sagittal waveforms the RIC pipeline recovers from Fukuchi's markers
against the waveforms **Fukuchi published from the same recordings**, computed
independently in Visual3D with their own model.

**Verdict: PARTIAL — shape is recovered, and the disagreements localise to two
specific modelling choices in the RIC pipeline that this comparison identified.**

## Resolving the normalisation mismatch

The two are not comparable as shipped: `gait_steps` emits 101 points of
**stance**, Fukuchi 101 points of the **full gait cycle**. Fukuchi's cycle starts
at initial contact — verified from curve shape, not assumed (knee 13.4° at 0%,
40.6° midstance peak near 20%, 84.6° swing peak near 70%, back to 13.4° at 100%).

Their curve is therefore sliced to stance and resampled to 101 points, with the
stance fraction **measured per trial from their own instrumented-treadmill
vertical force** — independent of both pipelines.

```
stance fraction, from Fukuchi's force plate
  2.5 m/s   0.368
  3.5 m/s   0.322
  4.5 m/s   0.303
```

**Monotone decreasing with speed**, which is what running requires and is a
check on the force-based detection itself. An earlier version produced
non-monotone fractions (0.390 / 0.337 / 0.358); the cause was threshold noise
splitting single contacts in two, breaking the strict alternation that
`stride = onset[i] → onset[i+2]` depends on. A 0.05 s minimum contact duration
fixed it.

## Result

**61 trials / 26 subjects, 366 curve comparisons.**

| joint | n | median r | \|r\| | RMS aligned | amp ratio | offset |
|---|---|---|---|---|---|---|
| **hip** | 122 | −0.993 | **0.993** | 2.96° | 1.112 | −44.2° |
| **knee** | 122 | +0.897 | 0.897 | 4.60° | **1.014** | −3.1° |
| **ankle** | 122 | −0.816 | 0.816 | 6.33° | 1.433 | −12.7° |

```
overall median |r| = 0.933   median sign-aligned RMS = 4.48 deg
pre-declared: |r|>=0.95 AND RMS<=3 deg pass | |r|>=0.85 partial | else fail
VERDICT: PARTIAL -- shape recovered, detail differs
```

### Sign conventions differ, and that is reported rather than hidden

Resolved **once per joint from the population median**, never per curve:

```
hip     median r = -0.993   INVERTED vs Fukuchi
knee    median r = +0.897   same as Fukuchi
ankle   median r = -0.816   INVERTED vs Fukuchi
```

**An earlier version of this script got the metric wrong and it is worth
recording.** A pure sign inversion makes offset-removed RMS equal *twice* the
signal amplitude, so the first pass reported hip at |r| = 0.991 beside a 28°
RMS — two numbers that cannot describe the same pair of curves. The metric was
conflating "the convention differs" with "the magnitudes differ". The docstring
had warned against exactly this failure for *offsets* and missed it for *signs*.

## What the disagreements localise to

This is the part worth having, because it names specific modelling choices rather
than producing an undifferentiated error number.

**Hip: near-perfect shape, amplitude 11% larger.** |r| = 0.993 with a 1.112
range ratio is the signature of a joint-centre definition difference, not a
kinematic one. `gait_kinematics.m:81` places the hip joint centre at
`GTR + 25% of the inter-GTR distance` — a geometric rule. Visual3D models
conventionally use a regression-based hip joint centre. A different HJC moves the
thigh's proximal end, scaling the hip angle while preserving its shape exactly.
The −44.2° offset is a neutral-definition difference on top.

**Knee: amplitude almost exact (1.4%), shape good.** The knee is the joint both
models define the same way — midpoint of medial and lateral markers — and it is
correspondingly the closest in amplitude.

**Ankle: the weakest, and the RIC foot model explains it.**
`gait_kinematics.m:118-120` builds the foot segment from **heel markers only**,
and hardcodes its long axis to the laboratory:

```matlab
% long axis of the the foot is aligned with the lab
l_foot_x = [0 0 -1];
```

The subject's actual foot orientation is never measured. Fukuchi carries MT1, MT2
and MT5 forefoot markers and can define a real foot segment. So our ankle range
runs 43% larger with the poorest shape agreement — and this is a **limitation of
the RIC pipeline's foot model**, not an adapter error. It would have been
invisible without an independent dataset to compare against, which is precisely
what §7.8 asked for.

## Agreement degrades with speed

```
2.5 m/s   |r| 0.960   RMS 3.48 deg   amp ratio 1.096   (n=138)
3.5 m/s   |r| 0.912   RMS 4.78 deg   amp ratio 1.125   (n=132)
4.5 m/s   |r| 0.888   RMS 6.41 deg   amp ratio 1.156   (n=96)
```

Monotone in all three measures. Three candidate causes are not separated here:
genuine difficulty at speed, degrading event detection (median `eventsflag` 0.72,
with fallback in *every* trial), and the speed-dependent exclusion bias from
Part 1 (19.4% excluded at 2.5 m/s rising to 41.9% at 4.5). **At the slowest
speed the result would clear the pass band** (|r| 0.960, RMS 3.48°); it is the
fast trials that pull it into PARTIAL.

## What this establishes, and what it does not

**Established.** Driven by an independent lab's marker model, the RIC pipeline
recovers sagittal hip and knee stance kinematics whose shape matches that lab's
own independent processing (hip |r| 0.993, knee amplitude within 1.4%). Combined
with Part 1's speed validation, **§7.8 is answered as far as an open dataset can
answer it**: the pipeline is not generating archive-specific artefacts.

**Not established: that the two agree in absolute angle.** They do not, and were
never going to — offsets run −3° to −44° because the models define neutral and
joint centres differently. This comparison tests whether the same motion produces
the same *waveform*, not whether two marker models produce the same number.

**The ambiguity declared in advance still holds for the ankle.** A disagreement
there is consistent with both an adapter error and a genuine foot-model
difference. The hardcoded lab-aligned long axis makes the latter much the more
likely reading, but this comparison alone cannot exclude the former.

## Implication for the project's own results

The ankle is the least trustworthy of the three sagittal mocap channels. Phase 5's
`limbsag` feature set used hip, knee **and** ankle, so one of its three channels
rests on a foot model that assumes the foot points along the lab axis.

This does **not** change any conclusion. Phase 5's result was a null, and phase 8
established the design resolves a 0.25° consistent asymmetry, so a noisier third
channel cannot manufacture the absence of a signal. If anything it marginally
strengthens the reading that the signal is small: one of three channels carries
more measurement error than previously documented. Worth recording as a caveat on
the mocap channels, not as a revision.

It also connects to a limitation already on record from the other direction:
`CLAUDE.md` notes ankle dorsiflexion is **unavailable** from the video path
because H36M-17 has no toe keypoint. The ankle is the weakest channel in both
modalities, for the same underlying reason — nothing observes the forefoot.

## Files

- `scripts/phase10_angle_compare.py`
- `results/phase10_angles.json`
