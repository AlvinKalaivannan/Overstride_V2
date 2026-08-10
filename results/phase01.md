# Phase 1 — the demographics control, and whether kinematics beat it

**Two negative results, reported as such.**

1. **The demographics-only control cannot serve as a benchmark.** It scores AUC
   **0.700**; a model fitted on nothing but data-collection artifacts scores
   **0.775** and beats it on 25 of 25 folds.
2. **Kinematics track the collection wave.** `dv_r` features recover the
   collection year at 50.7% accuracy against a 33.6% baseline, and six columns of
   pure *file metadata* — marker count, frame count, file size — classify injury
   at **AUC 0.757**, better than the demographics control and better than the
   kinematics themselves.

`dv_r` kinematics did **not** beat the demographics control: ΔAUC **−0.013**,
CI [−0.077, +0.051], 8 of 25 folds. Per `docs/phase1-spec.md` this is *not* a
kill — discrete summaries are lossy against the 101-point waveforms — but it
shifts the prior substantially, and it arrives alongside a confound that would
make a positive waveform result hard to believe anyway.

---

## The split

`StratifiedGroupKFold(n_splits=5, shuffle=True)`, **`groups=sub_id`**, 5 seeds
`[11, 23, 37, 53, 71]` → **25 folds**. Fold assignments generated once and reused
by every model, so all comparisons are paired fold-by-fold. A no-leakage
assertion runs over every fold and passed: no `sub_id` in train and test
together. All preprocessing sits inside an sklearn `Pipeline`, fit on the
training fold only.

## n per group, and what was excluded

| stage | sessions | subjects |
|---|---|---|
| `run_data_meta.csv` | 1,832 | 1,402 |
| drop `unknown` status | 1,759 | 1,335 |
| exclude osteoarthritis | **1,745** | **1,323** |
| joined to extracted `dv_r` | **1,745** | **1,323** |

Injured 1,169 sessions / 976 subjects; uninjured 576 / 363. Positive rate
**0.670**. The extraction reconciled exactly — **1,745 of 1,745 files parsed, zero
failures** — so 1A and 1B are scored on identical rows and the pairing is valid.

- **`unknown` dropped:** 73 sessions / 71 subjects. Absence of information.
- **Osteoarthritis held out:** 14 sessions / 12 subjects — and **all 14 are
  injured**, so the holdout contains no classification task. Counted, not
  modelled.
- **Kept and reported, not filtered:** 16 subjects (50 sessions) whose label
  changes between their own visits; 428 injured sessions (**36.6%**) whose
  diagnosis is `pain` or `other`.
- Sentinels masked to NaN before imputation: `age` (1), `Height` (8), `Weight`
  (5), `YrsRunning` (495, of which 51 are literal `999`).

---

## 1A — the control, and the provenance baseline

| model | features | AUC | 95% CI | PR-AUC |
|---|---|---|---|---|
| **provenance-only (HGB)** | 3 | **0.775** | [0.730, 0.817] | 0.837 |
| year alone (HGB) | 1 | 0.720 | [0.688, 0.775] | 0.810 |
| **demographics (logit, explicit missing)** | 7 | **0.700** | [0.642, 0.753] | 0.789 |
| demographics (HGB, explicit missing) | 7 | 0.700 | [0.661, 0.747] | 0.786 |
| demographics (HGB, mode-imputed) | 7 | 0.684 | [0.629, 0.744] | 0.782 |
| demographics (logit, mode-imputed) | 7 | 0.665 | [0.601, 0.715] | 0.773 |
| missingness flags alone (HGB) | 2 | 0.660 | [0.617, 0.700] | 0.750 |
| demographics minus leaky columns | 5 | 0.656 | [0.600, 0.697] | 0.776 |
| `speed_r` alone | 1 | 0.652 | [0.607, 0.686] | 0.787 |

Paired against the provenance baseline, every demographics variant loses with a
CI excluding zero — the best is −0.075 [−0.132, −0.027], winning 0 of 25 folds.

**Collection year alone, fitted non-linearly, scores 0.720 and beats the whole
seven-feature control.** The dataset concatenates studies with injury rates from
0.00 to 0.93 (2011: 0.934; 2015: 0.490; 2017: 0.000), and a tree on `year` is a
lookup table over them. The linear model gets only 0.615 from the same column
because the relationship is not monotone in time — a linear-only protocol would
have missed this entirely and reported a clean-looking control.

**The control rides the same leak.** Its largest coefficient is `Level_missing`
at **−2.377**, about 10× `speed_r` (−0.219) or `age` (+0.098). Stripping
`YrsRunning` and `Level` drops it 0.700 → 0.656.

Neither escape route works:

| cohort | provenance (HGB) | demographics (HGB) | ΔAUC | folds won |
|---|---|---|---|---|
| full (n=1,745) | 0.775 | 0.700 | −0.075 [−0.136, −0.008] | 1 / 25 |
| no 2017 (n=1,694) | 0.753 | 0.678 | −0.075 [−0.123, −0.004] | 0 / 25 |
| 2012–2016 (n=1,391) | 0.714 | 0.666 | −0.048 [−0.132, +0.012] | 2 / 25 |

---

## 1B — do kinematics track the wave?

### Extraction

1,745 files streamed one at a time in 10.0 min (2.9 files/s), peak working set
~294 MB against ~29 GB read. **Zero parse failures.** 152 `dv_r` columns
(76 × 2 sides) written to `data/derived/dvr_features.parquet` (1.3 MB).

**Two corrections to phase 0**, both from sampling artifacts:

- Phase 0 reported "0 of 1,520 sampled values were arrays". At full scale, **13
  sessions carry 2 non-scalar values each** (0.7%). Handled as missing, never
  coerced.
- Phase 0 flagged that `0` may double as missing. Quantified: of 152 columns,
  **72 are 100% exact-zero** (pure placeholders — `ANKLE_DF_at_HS`,
  `KNEE_FLEX_EXCURSION`, all `percent_STANCE` fields), 4 are 20–41% zero, and
  **only 76 carry real data**. The usable kinematic feature set is half the
  advertised size. Masking is a no-op for the 72; it is a judgement call for
  exactly 4 columns, so the results are robust to that choice.

### Q1 — session structure is almost a year indicator

| `n_marker_channels` | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 |
|---|---|---|---|---|---|---|---|---|---|
| 28 | 0 | 0 | 0 | 0 | 0 | 561 | 204 | 111 | 0 |
| 30 | 34 | 88 | 181 | 176 | 178 | 25 | 110 | 26 | 51 |

Every session before 2014 has 30 marker channels; 28-channel sessions exist only
in 2014–2016. And the marker count tracks the label directly: **28-channel
sessions are 78.2% injured, 30-channel 55.7%.**

### Q2 — the crux: can these features recover the collection year?

Multiclass, grouped 5-fold, against always guessing the modal year (0.336).

| features | year accuracy | lift over baseline |
|---|---|---|
| session structure (6 cols) | **0.635** | **+0.299** |
| **`dv_r` kinematics (152 cols)** | **0.507** | **+0.171** |
| demographics (numeric) | 0.426 | +0.091 |

**The answer is yes.** Kinematics recover which study a session came from at
half again the base rate. The problem is *not* confined to the metadata, which
was the hypothesis this run was designed to test.

### Q3 — injury classification

| model | features | AUC | 95% CI | PR-AUC |
|---|---|---|---|---|
| provenance-only (HGB) | 3 | **0.775** | [0.730, 0.817] | 0.837 |
| **structure-only (HGB)** | **6** | **0.757** | [0.710, 0.804] | 0.833 |
| `dv_r` + demographics (logit) | 159 | 0.719 | [0.661, 0.760] | 0.791 |
| structure-only (logit) | 6 | 0.706 | [0.637, 0.756] | 0.777 |
| demographics (logit / HGB) | 7 | 0.700 | [0.642, 0.753] | 0.789 |
| **`dv_r` kinematics (logit)** | 152 | **0.688** | [0.650, 0.753] | 0.789 |
| `dv_r` kinematics (HGB) | 152 | 0.669 | [0.626, 0.710] | 0.775 |

| paired delta | ΔAUC | CI | folds won |
|---|---|---|---|
| `dv_r` − demographics | **−0.013** | [−0.077, +0.051] | 8 / 25 |
| `dv_r` − provenance | −0.087 | [−0.146, −0.017] | 0 / 25 |
| (`dv_r` + demographics) − demographics | +0.018 | [−0.023, +0.055] | 19 / 25 |

**Six columns of file metadata classify injury at 0.757** — better than 152
kinematic features (0.688) and better than the demographics control (0.700).
Nothing in `n_marker_channels`, `n_frames` or `size_mb` is a property of the
runner.

### Q4 — inside the wave-balanced cohort (2012–2016, n=1,391)

| model | AUC | 95% CI |
|---|---|---|
| provenance (HGB) | 0.714 | [0.638, 0.775] |
| `dv_r` (logit) | 0.692 | [0.621, 0.753] |
| demographics (HGB) | 0.666 | [0.603, 0.723] |

`dv_r` − demographics: **+0.026** [−0.019, +0.092], 18/25 folds — positive but
the CI crosses zero. `dv_r` − provenance: −0.022 [−0.111, +0.029], 6/25.

This is the most favourable framing available to the kinematics, and it still
does not clear the bar the spec set in advance ("CI excludes zero").

---

## What this means

**`CLAUDE.md`'s kill criterion is not formally triggered.** It concerns
kinematics failing to beat the control, and the waveforms have not been tested.
`dv_r` is discrete summaries — half of them empty — and curve shape and timing
can carry signal that scalars discard. The spec committed in advance to treating
a `dv_r` failure as informative rather than decisive, and that commitment holds
now that the result is in.

**But the benchmark itself is broken, and that is the larger problem.** The
demographics control was meant to represent "older, heavier, slower people are
injured" so kinematics could be scored against it. It cannot play that role
while a three-column artifact model beats it and a six-column file-metadata
model nearly matches it.

**And the confound reaches the kinematics.** This run was designed to test
whether the damage was confined to metadata. It is not. Marker configuration is
nearly a year label, marker count alone shifts the injured rate from 56% to 78%,
and `dv_r` recovers the collection year at +17 points over baseline. A waveform
model that beat the control could be detecting lab protocol rather than
physiology, and phase 1 as designed could not distinguish the two.

**Consequence for phases 2–4.** The headline deliverable is a degradation curve
— ΔAUC between full-mocap and video-constrained features. If the mocap-side AUC
is substantially batch artifact, the curve measures the decay of an artifact.
Degrading a contaminated signal produces a clean-looking plot and a meaningless
one.

## Anything that surprised us, or looks wrong

1. **File size and frame count predict injury at 0.757.** This is the single most
   alarming number in the phase. It is not a subtle confound.
2. **The provenance baseline beat the control outright.** The spec anticipated a
   tie as the bad case; losing 25/25 folds was not on the list.
3. **Non-linearity is the whole story for `year`** — 0.615 logit vs 0.720 HGB on
   one identical column.
4. **Half the `dv_r` feature set is empty.** 72 of 152 columns are entirely zero.
   The advertised "76 discrete variables per side" is closer to 38.
5. **`dv_r` (HGB) is *worse* than `dv_r` (logit)** — 0.669 vs 0.688 — which is
   the opposite of the pattern everywhere else here, where the tree exploits the
   confound better. Consistent with the kinematics carrying weak signal that
   regularized linear models handle better than a tree that chases noise.
6. **The OA holdout has no uninjured sessions**, so "report OA separately" has
   nothing to report as a model, exactly as phase 0 predicted at n=12.

## Reproduction

```
uv add scikit-learn pyarrow
.venv/Scripts/python.exe scripts/phase1a_control.py       # 1A, ~4 min
.venv/Scripts/python.exe scripts/phase1b_extract.py       # stream, ~10 min
.venv/Scripts/python.exe scripts/phase1b_wave_check.py    # 1B, ~5 min
```

Seeds fixed throughout (`SEEDS = [11, 23, 37, 53, 71]`, HGB `random_state=0`).
Per-fold AUCs in `results/phase1a_control.json` and `results/phase1b_wave.json`.
Cohort construction is shared via `scripts/phase1_cohort.py` so 1A and 1B cannot
drift apart.

## Gate status

| # | spec gate item | status |
|---|---|---|
| 1 | control AUC + CI reported | ✅ 0.700 [0.642, 0.753] |
| 2 | control clearly beats provenance | ❌ **loses**, 0/25 folds |
| 3 | `dv_r` table reconciles to 1,745 | ✅ 1,745/1,745, 0 failures |
| 4 | marker-set-vs-wave check | ✅ **confound confirmed** |
| 5 | paired ΔAUC per kinematic model | ✅ reported above |
| 6 | go/no-go on waveform generation | ⬜ **decision required** |

The `CLAUDE.md` phase 1 gate ("beats the demographics-only control") is
**ill-posed as written** — the control is not a valid benchmark. It is not
answerable by fixing the model; it needs the benchmark redefined.
