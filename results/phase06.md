# Phase 6 — no report, by design

Every other phase has a `results/phaseNN.md`. This one does not, and the absence
is deliberate rather than an omission — this file exists so a reader scanning
`results/` does not have to wonder.

## What happened

Phase 6 was originally scoped as a **demo shell**: a user-facing surface over the
model. Phases 4B and 5D established that a per-runner verdict is unsupportable —
PPV 0.574 against a 0.517 base rate, calibration slope 0.716, and the model
disagrees with itself on a third of repeat scans. **Building the demo would have
misrepresented the evidence.**

Phase 6 became the honest presentation of the measurement instead, and its output
is the repository's front door rather than a phase report:

- **[`README.md`](../README.md)** — the synthesis: the finding, the two verdicts,
  why the ceiling is the signal, why no demo was built.
- **`figures/phase4_degradation.png`** — the degradation curve
- **`figures/phase6_ceiling.png`** — every feature family tried
- **`figures/phase6_angle_recovery.png`** — recovered vs true joint angles
- **`figures/phase5d_selective.png`** — what abstention buys
- **`scripts/phase6_numbers.py`** — every number in the README, regenerated from
  `results/*.json`, so the prose cannot drift from the artifacts

The redefinition is recorded in `CLAUDE.md`'s phase table and in
`results/phase04b.md` / `results/phase05d.md`, which supply the evidence for it.

## Verifying the README

```
.venv/Scripts/python.exe scripts/phase6_numbers.py
```

Reads all 28 files in `results/`. If a number it prints disagrees with the
README, the README is wrong.
