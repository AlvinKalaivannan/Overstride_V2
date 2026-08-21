# Phase 5D — does abstention make a 0.61 model usable?

**Partly, and only on data a camera cannot produce.** Refusing to answer does
buy real accuracy — the effect is validated against a control that stays flat —
but on the deployment configuration it does not hold up, and multi-session
averaging fails outright in a way that explains why.

| configuration | 100% coverage | 50% coverage | 30% coverage |
|---|---|---|---|
| clean mocap, `limbsag` | 0.577 [0.510, 0.652] | **0.637** [0.548, 0.722] | 0.653 [0.532, 0.745] |
| clean mocap + stride distributions | 0.581 [0.503, 0.627] | 0.640 [0.523, 0.711] | 0.674 [0.507, 0.824] |
| **deployment** (video error, 30 fps) | 0.556 [0.484, 0.616] | 0.585 [0.503, 0.668] | 0.601 [0.492, 0.710] |
| CONTROL: provenance only | 0.499 [0.440, 0.532] | 0.479 [0.419, 0.527] | 0.472 [0.399, 0.544] |

Always guessing the majority side scores **0.517**.

![selective classification](../figures/phase5d_selective.png)

---

## The method, and why it is not circular

Selection is by the model's **own confidence** `|p − 0.5|`, never by the label,
and the most-confident X% is taken **within each test fold independently**.
Coverage is a design parameter, not a fitted threshold, so nothing is tuned on
the evaluation data.

**The control is what makes the rest interpretable.** A selective-accuracy curve
that rises as coverage falls looks like a result but can be an artifact of the
selection procedure itself. So the identical sweep was run on the
**provenance-only** model, which phases 4, 5 and 5B all verified sits at chance
on this task. It must stay flat.

**It does — drift 0.052, and it drifts *downward*** (0.499 → 0.449). The
procedure does not manufacture an upward trend. Every number above is therefore
signal rather than an artifact of selecting on confidence.

## What abstention actually buys

**On clean mocap, a real but modest gain.** The curve rises monotonically and
the accuracy CI clears the majority baseline from **80% coverage down**:

| coverage | n kept / fold | accuracy | 95% CI | lift over baseline | CI beats baseline |
|---|---|---|---|---|---|
| 100% | 164 | 0.577 | [0.510, 0.652] | +0.060 | no |
| 90% | 147 | 0.585 | [0.512, 0.673] | +0.067 | no |
| **80%** | 131 | 0.598 | [0.520, 0.690] | +0.080 | ✅ |
| 70% | 114 | 0.607 | [0.535, 0.686] | +0.090 | ✅ |
| 60% | 98 | 0.626 | [0.553, 0.712] | +0.109 | ✅ |
| **50%** | 82 | **0.637** | [0.548, 0.722] | +0.119 | ✅ |
| 40% | 65 | 0.649 | [0.567, 0.772] | +0.132 | ✅ |
| 30% | 49 | 0.653 | [0.532, 0.745] | +0.136 | ✅ |
| 20% | 33 | 0.652 | [0.535, 0.760] | +0.135 | ✅ |
| 10% | 16 | 0.716 | [0.537, 0.869] | +0.199 | ✅ |

So on perfect marker data you could answer **half the scans at ~64% accuracy**
against a 52% baseline. That is a genuine gain, and it is the first result in
this project where a practical operating point clears its own interval.

**The 10% row should not be quoted on its own.** Sixteen sessions per fold
remain; the interval runs [0.537, 0.869] and the neighbouring 20% and 30% rows
sit at 0.652–0.653. The apparent jump to 0.716 is noise at the tail, which is
also why the stride-distribution curve *falls* at 10%.

## Where it breaks: the deployment configuration

`wave2` + side-on error bank + 30 fps — the realistic configuration — is the row
that matters, and abstention does not rescue it:

| coverage | accuracy | 95% CI | CI beats baseline |
|---|---|---|---|
| 100% | 0.556 | [0.484, 0.616] | no |
| 70% | 0.565 | [0.512, 0.625] | no |
| 50% | 0.585 | [0.503, 0.668] | no |
| 40% | 0.593 | [0.523, 0.681] | ✅ |
| 30% | 0.601 | [0.492, 0.710] | **no** |
| 20% | 0.616 | [0.494, 0.752] | **no** |
| 10% | 0.682 | [0.526, 0.900] | ✅ |

**Only 2 of 10 coverage levels clear the baseline, and they are not adjacent** —
40% passes, 30% and 20% fail, 10% passes. A genuine operating region is
contiguous; this pattern is what noise looks like. There is no coverage at which
the video-derived model can be said to beat guessing.

The mean accuracy does rise monotonically, so the effect is probably real and
simply too small to resolve at n = 818. But "probably real and unresolvable" is
not something to build a user-facing readout on.

## Multi-session averaging fails, and that is the most informative result here

72 subjects have two or more sessions with the same injured side (208 sessions).
If the model's error were independent noise between sessions, averaging a
runner's scores should beat a single scan.

| | accuracy |
|---|---|
| single-session, on those same sessions | 0.587 |
| after averaging each subject's sessions | **0.583** (−0.003) |

**Averaging does not help at all.** That is diagnostic. It means the residual
error is **subject-specific, not random**: the model is *consistently* wrong
about the same people, so seeing them a second time adds nothing. It also
explains phase 4B's repeatability finding from the other direction — agreement
of 0.725 is not a model that is noisy around the right answer, it is a model
that is stably right about some runners and stably wrong about others.

**This closes the "just scan them twice" escape route.** More scans of the same
runner do not improve the answer.

## The split

Identical to phases 4, 5, 5B and 5C: 818 unilateral-injury sessions / 675
subjects, chance 0.517, `StratifiedGroupKFold(n_splits=5, shuffle=True)`,
**`groups=sub_id`**, seeds `[11, 23, 37, 53, 71]` → 25 folds.
`check_no_group_leakage` asserted. Accuracy CIs are percentile intervals across
the 25 folds, the same convention used throughout.

## What this means for phase 6

Abstention is a real mechanism and it is now measured rather than assumed. But
the two configurations diverge, and the honest summary is:

1. **On mocap-quality data**, a selective system answering ~50% of scans at ~64%
   accuracy is defensible. Overstride cannot produce mocap-quality data.
2. **On video-derived data**, no coverage level reliably beats guessing.
3. **Repeat scans do not help**, because the error is subject-specific.

**A per-user limb verdict remains unsupportable**, and phase 4B's conclusion
stands unchanged. What phase 5D adds is that this is not for want of the obvious
engineering fixes — abstention and repeat measurement were the two levers
available, they have both now been measured, and neither closes the gap.

The framings in `results/phase04b.md` are unaffected: a methods demo of the
degradation curve remains the defensible deliverable, and it is what CLAUDE.md
describes as the point of the project.

## Reproduction

```
.venv/Scripts/python.exe scripts/phase5d_selective.py   # ~25 min CPU
.venv/Scripts/python.exe scripts/phase5d_figure.py
```

Output: `results/phase5d_selective.json`, `figures/phase5d_selective.png`.
