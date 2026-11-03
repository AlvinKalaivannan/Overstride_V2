# Diagnostics

Throwaway scripts that turned out not to be throwaway. These three are what
localised the confidence-scale bug in phase 15 (`results/phase15.md` D2), and
they are kept because the next defect in the video path will be found the same
way.

| script | question it answers |
|---|---|
| `diag_bisect.py` | **Which half of the pipeline is broken?** Feeds the lifter a perfect 2D built by projecting the reference 3D, so the back half is tested on its own input. |
| `diag_2d.py` | **Is the 2D detector or the joint-order conversion at fault?** Per-joint pixel error of the detected 2D against that same projected reference. |
| `diag_align.py` | **Is it a temporal offset?** Cross-correlates the estimate against the reference over a lag range, and checks the estimate against the tool's own plausibility bands. |

They are diagnostics, not measurements: **nothing is selected on their output.**
Selecting anything on a held-out set would invalidate phase 15.

## The rule these scripts follow

MoVi's licence forbids reproducing the Data *"in any form to any third party"*
and this repository is public. So none of them prints or writes:

- camera intrinsics, extrinsics, distortion coefficients or camera positions —
  projection goes through `phase15_checks.project_reference`, which takes a
  camera *name* and returns pixels, so the calibration never passes through a
  diagnostic;
- participant sex, stature, mass, age, or segment lengths in millimetres.

What they do print is measurement of the *method*: angular error, correlation,
pixel error, and frame counts. See `docs/data-inventory.md` §8.5.

## Running them

All three take MoVi round F from `data/movi/raw/round_f/` and default to
`Subject_1` on `PG1`:

```
.venv/Scripts/python.exe scripts/diagnostics/diag_bisect.py
.venv/Scripts/python.exe scripts/diagnostics/diag_2d.py --frames 150
.venv/Scripts/python.exe scripts/diagnostics/diag_align.py --frames 150
```
