# Notes on the Overstride Validation Plan

**Responding to:** the validation plan produced after `docs/REVIEW.md` — dataset
triage, a soft-spot closure matrix, and a sequence B0 → A1/A2 → B1 → C → D.
**Method:** every premise testable against this repository was tested. Commands
and outputs are shown so nothing needs taking on trust.

---

## Verdict

**The plan is sound and I would run it, with one correction.** Its central
move — splitting my §7 "one dataset" weakness into *does the ceiling generalize*
and *is the measured error right* — is sharper than how I wrote it, and its
conclusion that no open dataset fixes the first is a real finding rather than a
failed search.

Three things need saying:

1. **A1's premise is wrong in a way that changes its design**, though the step
   survives and can be made stronger than proposed.
2. **One matrix cell is misattributed**, and a second premise needs a check the
   plan does not name.
3. **The plan identifies a genuine gap in my own review** that I missed.

---

## 1. A1 re-scoped — Ferber walking is not a free replication

The plan describes the walking trials as "same 1,798 subjects, same injury
labels, a second gait mode… the closest thing to a free replication in this whole
plan." Measured, that does not hold.

```
.venv/Scripts/python.exe scripts/walk_cohort_check.py
```

| | walk | run |
|---|---|---|
| sessions / subjects | 2,088 / 1,686 | 1,832 / 1,402 |
| unilateral (`InjSide` L/R) | 1,281 / 1,093 | 1,038 / 869 |
| **OA-labelled sessions** | **419 (20.1%)** | **14 (0.8%)** |
| median age | 44 | 37 |

Broken down by label: `knee oa` 190 walk / 0 run, `hip oa` 87 / 0, `oa` 75 / 0,
`osteoarthritis` 67 / 14. **Essentially the entire osteoarthritis cohort walked
and did not run** — which matches `docs/data-inventory.md`, where the paper's 247
OA subjects reduce to 12 in the running metadata.

This matters because `CLAUDE.md` excludes OA from primary analysis: degenerative,
mean age 56, slowest run speed, **confounded with the demographic control**. The
seven-year median age gap between the cohorts is that composition showing through.

So A1 is not a gait-mode comparison holding everything else fixed. It is a
different population *and* a different gait mode, and reporting it as the former
would repeat the class of error this project spent phase 1 learning to avoid.

### It survives, at a workable scale

```
all walking sessions            2088 / 1686 subjects
unilateral (InjSide L/R)        1281 / 1093 subjects
  excluding OA (CLAUDE.md)       995 /  904 subjects
  and also in the run cohort     880 /  792 subjects

running analysis, for comparison:   818 /  675 subjects
```

**880 sessions / 792 subjects** against the running analysis's 818 / 675 —
comparable, before the waveform and MIN_STEPS filters shrink it further as they
did for running.

### And a better design is available than the plan proposes

**1,290 subjects appear in both gait modes.** Restricting to those makes gait
mode the only variable changing within a subject — a paired design, strictly
stronger than comparing two cohorts that differ in composition. The plan compares
cohorts; it should compare subjects to themselves. That is the same logic that
rescued the project at phase 5, applied one level up.

### What this changes about A1

- **Decide the OA exclusion before starting, not during.** The plan's step 1 asks
  whether walking regenerates through the MATLAB path; the composition question
  is separate and comes first, because it determines what is being measured.
- **Restrict to the 1,290 dual-mode subjects** and state the paired design
  explicitly.
- **Keep the plan's stop condition** — it is correct and important. If any
  negative control leaves chance on the walking subset, report the confound and
  stop.
- **Revise the cost framing.** Still days, still no acquisition, but "free
  replication" oversells it. It is a *cheap and differently-composed* second look.

---

## 2. Two smaller corrections

**The sampling rate differs, and A1's feasibility check should target it.**
Walking is mostly 120 Hz while running is almost entirely 200 Hz — 1,822 running
sessions at 200 Hz against 10 at 120 (`docs/data-inventory.md`). The 101-point
stance normalisation should absorb this, but filter cutoffs and the MIN_STEPS
threshold interact with sample rate. That is the specific thing to verify in step
1, and the plan does not name it.

**§7.3 is misattributed in the matrix.** The plan credits BioCV with closing
"detector cost on biased subsample". That caveat is closed by **A2** — the
full-592 re-run, forty minutes, no new data. BioCV *supersedes* it with an
independent measurement from a different lab, which is a different and better
thing, but the matrix should not show a weeks-long external dependency closing
something an afternoon closes. Corrected, BioCV still closes three of nine on its
own, which remains the strongest concentration in the table.

---

## 3. What the plan gets right

**The A/B split.** My §7 treated "one dataset" as a single weakness across items
7.3 through 7.6. Separating *generality of the finding* from *accuracy of the
measurement* is the better decomposition, because only one of them has an
available fix. I would adopt this framing in any future version of the brief.

**The honest conclusion.** "Nothing found here lets the negative result claim
generality" is the right answer, stated plainly rather than worked around with
adjacent datasets. Its closing section — that the one-clinic limitation survives
every step, and that a real fix is a collaboration rather than a download — is
correct and should stay exactly as written.

**Falsification conditions declared in advance.** B1's MAE bands (≤5° consistent,
6–10° softens, >10° revises the headline) mirror the pre-registration discipline
the project used for its own bars. Naming the outcome that would break the claim
*before* running is the right habit.

**The BioCV licence tension.** Well spotted, and it bites harder than the plan
says: §10 of my brief promises a reviewer can regenerate every figure from
`results/*.json`. A BioCV-derived number cannot honour that. The resolution is
probably to ship the code and a documented access path while keeping the number
outside the reproducibility guarantee — but that is a decision, and it should be
made before building on it rather than discovered at write-up.

### It found a gap in my review

The plan's rationale for GaitRec is that the Ferber null is ambiguous between
*running overuse injury is a subtle signal* and *the limb-asymmetry method is
underpowered*. **My brief did not raise that, and it should have.**

The project has strong negative controls — provenance, demographics, structure
and dominant leg all verified at chance, which establishes the design does not
leak. It has **no positive control**: nothing demonstrates the method can detect
a limb asymmetry known to be large. Without one, "we found nothing" and "this
method finds nothing" are not separated by the evidence.

That is a real hole in the argument, and it is cheaper to fix than generality.
GaitRec is a defensible probe provided it is framed as a positive control and
never as replication — the plan's own warning about that framing is exactly
right. A synthetic positive control would be cheaper still: inject a known
asymmetry of known magnitude into the Ferber waveforms and confirm the pipeline
recovers it at the expected effect size. That costs hours, not a week, and it
answers the same question without importing a different modality.

---

## 4. On sequencing

**B0 first is right** — it is the only step with an external dependency, and
everything downstream of it waits.

**A2 this week** — forty minutes, closes §7.3 outright, and both the plan and my
brief already recommend it. There is no argument for deferring.

**A1 after the OA decision**, not before. The analysis is days; the design
question is an afternoon and determines what the days measure.

**The synthetic positive control belongs alongside A2** on cost grounds — hours,
no new data, and it addresses the ambiguity that D would otherwise take a week to
probe. If it comes back showing the method recovers a known asymmetry cleanly,
D's value drops considerably.

---

## 5. The limitation that survives

The plan's closing section is correct and I would not soften it. Nothing in this
sequence supplies a second cohort of clinically diagnosed runners with per-limb
waveforms, because no such open dataset appears to exist. The finding continues
to rest on one clinic, one protocol, one population.

That boundary is not a defect to be managed. Stating it precisely is the same
discipline that produced the corrections ledger in `docs/REVIEW.md`, and it is
the reason the null is worth anything at all.
