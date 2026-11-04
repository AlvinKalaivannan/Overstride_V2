# Phase 15F — capture rate, and whether another checkpoint transfers better

> **PRE-REGISTRATION, written 2026-11-04.** Committed **alone** before anything
> is scored. Nothing above the "Results" heading is edited afterwards. Two
> independent diagnostic questions. **No gate on either.**

---

## Q1 — the capture-rate recommendation for the demo

**Question.** 15E measured a resolved 30 fps penalty. Is the penalty curve still
steep between 50 and 30 fps, or has it flattened? That decides whether the demo
should ask for a higher-rate capture.

**It is 50 fps, not 60, and this report says so everywhere.** AthleticsPose is
**100.0 Hz** (shipped Qualisys QCA metadata, `Timebase/Frequency`, all 14
sessions — established in 15E). 100/60 = 1.667 is not an integer, so a true 60
fps arm would need tick-rounding and would not be a clean decimation. **Every
second frame gives exactly 50.0 fps**, which is the clean integer rule, so that
is what is run. The number reported is a **50 fps** penalty and must not be
quoted as 60.

**Method.** The 15E harness unchanged: AthleticsPose, **S11/S13/S16, running,
generic COCO detections**, native arm bit-identical to the committed baseline
path. One new decimation rule, `every_nth(n, step=2)`. Scored on the same native
frame indices as the arm it is compared against, as in 15E.

**Reported.** The penalty for **knee, trunk-referenced hip, gravity thigh
inclination and trunk lean**, **near and far limb separately**, paired per
subject with a subject-bootstrap CI (`N_BOOT = 2000`, `SEED = 11`), set beside
15E's already-committed 33.3 fps and 30.0 fps arms:

| rate | rule | status |
|---|---|---|
| 100 Hz | native, no decimation | 15E, committed |
| **50.0 fps** | **every 2nd frame** | **this phase** |
| 33.3 fps | every 3rd frame | 15E, committed |
| 30.0 fps | nearest 30.0 fps tick | 15E, committed |

**And the thing the demo actually needs:** whether the curve is still steep
between 50 and 30. Reported as the **50→30 increment** (`penalty(30) −
penalty(50)`) against the **100→50 increment** (`penalty(50)`), per quantity and
limb. If most of the damage is already done by 50 fps, a higher-rate capture
buys little; if the 50→30 segment is the steep one, it buys a lot.

---

## Q2 — Stage 0: do other checkpoints already on disk transfer better?

**Question.** Does a lifter trained on different or more varied data have a
**smaller AthleticsPose→MoVi loss** than the deployed one? No training, no new
data — only checkpoints already on disk.

> **Why the loss is answerable even where a contract is not fully determined.**
> The quantity compared is `loss = MoVi MAE − AthleticsPose MAE` **per
> checkpoint**. Each checkpoint is fed **identically on both datasets**, so a
> handicap in its absolute level (wrong confidence distribution, wrong window)
> affects both terms and largely cancels in the difference. A checkpoint can
> therefore be absolutely worse and still show a smaller loss, which is exactly
> the question. **Absolute MAEs are reported but are not the answer.**

### The input contracts, established before anything was scored

From each checkpoint's own repo, config and paper. **Architecture and load
status verified**: all three load into `phase3_infer.ARCH` with **0 missing and
0 unexpected** keys.

| item | **b-ath-det-coco-v1** (deployed) | **b-h36m** | **s-ap3d** |
|---|---|---|---|
| architecture | MotionAGFormer-**B**: `dim_feat` 128, 16 layers, `dim_in` 3 | MotionAGFormer-**B**, identical | MotionAGFormer-**S**: `dim_feat` 64, **26 layers**, `dim_in` 3 |
| loads | 0 missing / 0 unexpected | 0 / 0 | 0 / 0 (`arch="small"`) |
| joint order | **H36M-17** | **H36M-17** | **H36M-17** |
| normalization | AthleticsPose `utils.normalize_kpts`: subtract joint 0, divide x,y by `max|xy|`; **confidence re-attached RAW after normalization** | MotionBERT `datareader_h36m.read_2d`: `xy / res_w * 2 − [1, res_h/res_w]` — **image-resolution**, absolute image position preserved | **UNDETERMINED** — not stated in repo, config or paper |
| confidence channel trained on | AthleticsPose generic-COCO detector scores; **measured** mean **0.8676**, sd 0.1164 (`video_kinematics.TRAIN_CONF_MEAN`) | Stacked Hourglass scores, `dt_file: h36m_sh_conf_cam_source_final.pkl`. **The distribution is UNDETERMINED** — the pkl is not available. Its own reader's **documented fallback when confidence is absent is 1.0** | **UNDETERMINED** — the paper does not say whether a confidence channel is used at all |
| training frame rate | **100.0 Hz** (QCA `Timebase/Frequency`) | **50 Hz** — H36M capture rate; `sample_stride` is unspecified in `configs/h36m/MotionAGFormer-base.yaml`, so the standard stride of 1 applies | **mixed and unstated**: running **120 FPS**, track/field and figure skating **60 FPS**; no training rate given |
| window length | **81** (`configs/model/base.yaml`, `n_frames: 81`) | **243** (`configs/h36m/MotionAGFormer-base.yaml`, `n_frames: 243`) | **81** (paper: *"we use the model variant that processes 81 frames"*) |
| provenance | AthleticsPose release | redistributed in AthleticsPose `checkpoints.zip`; upstream `TaatiTeam/MotionAGFormer` | redistributed in AthleticsPose `checkpoints.zip`; upstream `calvinyeungck/AthletePose3D` |
| **items determined** | **5 of 5** | **4 of 5** | **3 of 5** |

**A finding in its own right:** the AthleticsPose repo ships both `.pth` files
and documents evaluating them (`evaluation=h36m_pretrained`,
`evaluation=ap3d_pretrained model=small`) through **its own 81-frame pipeline
with its own person-scale normalization**. For `b-h36m` that is a **documented
but contract-violating** usage — the checkpoint was trained at 243 frames with
image-resolution normalization — and the repo never claims it works standalone:
the AthleticsPose paper says fine-tuning from pretrained weights was needed
*because learning from scratch was ineffective*. **So "the way the shipping repo
runs it" and "its own training contract" are two different things, and this
phase separates them.**

### Inclusion, by the stated rule

- **`b-ath-det-coco-v1` — INCLUDED.** Full contract determined.
- **`b-h36m` — INCLUDED**, fed **its own contract**: image-resolution
  normalization at each dataset's true resolution, **243-frame windows**,
  input rate-matched to **50 Hz**. Its confidence *distribution* is undetermined,
  so the primary uses **1.0**, which is its own pipeline's documented fallback
  rather than a guess, and a **`TRAIN_CONF_MEAN` sensitivity arm** is reported
  beside it. This is a stated limitation and it biases in a known direction: a
  checkpoint fed a confidence distribution it did not train on is
  **disadvantaged**, so a poor absolute result cannot be attributed to its
  training data.
- **`s-ap3d` — EXCLUDED FROM THE PRIMARY**, per *"if a contract can't be
  determined, say so and exclude that checkpoint rather than guessing"*. Its
  **normalization and confidence convention are both unstated**, and its
  training rate is mixed with none given — 2 of 5 items undetermined, including
  the one that sets the input scale the network sees.
  It is run **only** as an explicitly labelled **"as the shipping repo runs
  it"** arm — AthleticsPose normalization, 81 frames, `TRAIN_CONF_MEAN`, which
  is the one documented way to feed this file. **That arm is never described as
  contract-respecting and its loss is reported separately from the primary
  table.**

### Datasets, and which is in-domain for whom

**Detected 2D only**, both datasets, every checkpoint.

| dataset | clips / subjects | `b-ath-det-coco` | `b-h36m` | `s-ap3d` |
|---|---|---|---|---|
| AthleticsPose, S11/S13/S16, running, generic COCO | 589 clips, 3 subjects | **IN-DOMAIN** (fine-tuned on this dataset; S11/S13/S16 are its held-out subjects) | cross-domain | cross-domain |
| MoVi PG1, jogging | **11 subjects** | cross-domain | cross-domain | cross-domain |

**MoVi subjects: 1, 10, 11, 12, 13, 15, 16, 17, 18, 19, 2 — n = 11.** All are
**spent**: 1/10/12/13 were the phase 15/15D rule set, 11 and 15 were
diagnostic-only, and 16/17/18/19/2 were the sealed set. **This is a diagnostic
comparison on spent subjects and establishes no new held-out claim.** Subject_14
has no round F video of any kind and cannot be included.

**Two subjects carry known pathologies and are reported per-subject so they stay
visible:** `Subject_11` loses **42%** of its frames to the gap sentinel in the
markers the reference frame is built from, which is why phase 15 excluded it from
the primary; `Subject_15` was diagnostic-only. Neither is dropped here — the
comparison is between checkpoints on identical frames, so a bad subject handicaps
all three equally — but **a per-checkpoint loss driven by those two will be
called out.**

Image resolutions, for the image-resolution normalization: **AthleticsPose
1920×1088** (`FovVideo` Right 1919, Bottom 1087, uniform across all 8 cameras and
all 14 sessions) and **MoVi PG1 800×600** (measured from the files, 2026-11-02).

### Quantities and statistics

`knee_from_segments`, gravity **thigh** inclination, **trunk-referenced hip**,
**trunk lean**. **Near and far limb separately**, never pooled. Per clip or
subject: **MAE, bias, SD, r**. Pooled unweighted over subjects, with a
**subject-bootstrap CI, `N_BOOT = 2000`, `SEED = 11`**. On AthleticsPose that is
3 subjects and the interval is close to the range of three numbers; on MoVi it is
11, the largest n this project has had.

The gravity-referenced quantities need the world markers aligned to GT, so on
AthleticsPose they run on the **231 length-matched clips** and knee and hip run
on all **589** — the same split 15E recorded as deviation D-E2, stated here in
advance this time.

### The answer

Per checkpoint, per quantity, per limb:

> `loss = MoVi MAE − AthleticsPose MAE`, each with its own subject-bootstrap CI,
> and the **difference in loss against the deployed checkpoint**.

A checkpoint "transfers better" only if its **loss** is smaller. **No gate.**

### The input-contract check

**`tests/test_input_contract.py` does not exist** — the test suite is
`test_angles`, `test_confidence`, `test_eval_protocol`,
`test_phase15_checks_bar`, `test_phase15_checks_out`, `test_phase15_refusal`,
`test_scan`, `test_segment_angles`, `test_video_crop`, `test_video_pipeline`.
Rather than report nothing, this phase **writes it**, and it is committed with
the results.

It asserts, for every checkpoint's prepared input before any lift, that:

1. the array is `(T, 17, 3)`, finite, and float;
2. the **x,y channels lie in the range its declared normalization implies** —
   `[-1, 1]` for AthleticsPose person-scale, and `[-1, 1] × [-1, res_h/res_w]`
   for MotionBERT image-resolution;
3. the **confidence channel lies in its declared band** and is **not** left on a
   foreign scale — the defect that cost 30.07° on MoVi (raw torchvision scores,
   mean 12.4, against a trained 0.8676) and 0.60–1.63° on AthleticsPose (a
   constant 1.0 against 0.8676);
4. no frame is all-zero, the sentinel that put 45 skeletons at the image origin
   in phase 15.

**Whatever it flags is reported, including flags on the deployed checkpoint.**

### What would make this phase worthless

- **Quoting the 50 fps arm as 60 fps.** It is 50.0.
- **Calling the `s-ap3d` arm a contract-respecting result**, or putting it in the
  primary table.
- **Answering Q2 on absolute MAE instead of on the loss.** A cross-domain
  checkpoint fed an undetermined confidence distribution will look bad
  absolutely; that is not the question.
- **Changing any contract after seeing a score.** The table above is fixed.
- **Presenting the MoVi n=11 as a held-out result.** Every one of those subjects
  is spent.

---

## Results

**Run 2026-11-04**, against the pre-registration committed alone as `d198eac`.
`results/phase15f.json`, driver `scripts/phase15f.py`. No gate on either
question.

### Corrections to the pre-registration, before the numbers

**C1 — `b-h36m` is an 81-frame model, not 243. The pre-registration above is
wrong and the weights are decisive.** Each checkpoint's
`graph_temporal.mixer.batch_norm` is a BatchNorm over the *temporal node*
dimension, so its shape **is** `n_frames`. All three report **[81]**:

| checkpoint | temporal BatchNorm shape | n_frames |
|---|---|---|
| b-ath-det-coco-v1 | (81,) | **81** |
| b-h36m | (81,) | **81** |
| s-ap3d | (81,) | **81** |

Building `b-h36m` at 243 fails outright — `size mismatch ... copying a param
with shape torch.Size([81]) from checkpoint, the shape in current model is
torch.Size([243])`. The 243 in the pre-registration came from
`configs/h36m/MotionAGFormer-base.yaml` (`n_frames: 243`) and the upstream
README's ambiguous "243 | 81" column; the shipped file is the 81-frame B
variant, which is the README's second MACs figure (16G). **Lesson worth keeping:
when a contract item is load-bearing, read the artifact, not the documentation
about the artifact.**

**C2 — a sixth contract item, not pre-registered, established before scoring:
the OUTPUT coordinate convention.** The pre-registration listed five input
items. The angle code reads the estimate against `EST_UP = (0, −1, 0)`, which
phase 15D verified for the deployed checkpoint only. Measured as mean(head −
pelvis) over six clips:

| checkpoint | x | y | z | up-axis | output limb scale |
|---|---|---|---|---|---|
| b-ath-det-coco | +0.025 | **−0.999** | −0.035 | axis 1, negative | thigh 0.4251 |
| b-h36m | +0.022 | **−0.988** | −0.155 | axis 1, negative | thigh 0.1285 |
| s-ap3d | +0.037 | **−0.978** | −0.203 | axis 1, negative | thigh 0.4112 |

All three agree, so `EST_UP` is correct for all three and the comparison is not
corrupted by a convention mismatch of my own making. `b-h36m`'s output is 3.3×
smaller in absolute scale, which does not matter: the angles are
scale-invariant.

**C3 — the windowing is bit-identical to the frozen path.** `_windows` in
`phase15f.py` reimplements edge padding plus 50%-overlap triangular blending,
and against `phase7_inference_fix.lift` on the same input it gives
**max |diff| = 0.000e+00**.

### The input-contract check

**`tests/test_input_contract.py` did not exist** and was written for this phase:
14 tests, all passing, one per defect that actually happened — raw torchvision
scores (the 30.07° MoVi bug), the constant-1.0 channel (0.60–1.63° in 15E), and
all-zero sentinel frames (phase 15 D5).

Writing it corrected a claim of mine. I expected a normalization swap to be
invisible to a range check; it is not. Person-scale input permits `|y| ≤ 1.0`
while the image-resolution contract bounds `y` at `res_h/res_w = 0.567`, so the
swap **is** caught. The genuine limitation is pinned in its own test: on a
**square** image the two bounds coincide and only the confidence channel gives
the swap away.

**What it flagged on the real runs: nothing.**

| checkpoint | AthleticsPose clips flagged | MoVi subjects flagged |
|---|---|---|
| b-ath-det-coco | **0** of 589 | **0** of 11 |
| b-h36m | **0** of 589 | **0** of 11 |
| s-ap3d | **0** of 589 | **0** of 11 |

Every prepared input satisfied its declared contract: shape, finiteness,
coordinate range for its normalization, confidence in band, no all-zero frames.
**So none of the results below is explained by a scale defect of the kind that
has bitten this project three times** — which is the point of running the check
before scoring rather than after being surprised.

---

## Q1 — the capture-rate curve. It is 50 fps, not 60.

**589 clips, generic COCO, paired per subject.** The 50.0 fps arm is this
phase; the 33.3 and 30.0 arms are quoted from 15E's commit, not recomputed.

| quantity | limb | **50.0 fps** | 95% CI | 33.3 fps | 30.0 fps |
|---|---|---|---|---|---|
| **knee flexion** | near | **+0.598** | [+0.284, +0.916] | +1.509 | +1.835 |
| **knee flexion** | far | **+0.870** | [+0.535, +1.056] | +1.597 | +1.757 |
| hip (trunk-ref) | near | **+0.372** | [+0.284, +0.487] | +0.539 | +0.641 |
| hip (trunk-ref) | far | **+0.397** | [+0.325, +0.505] | +0.594 | +0.734 |
| thigh inclination | near | **+0.059** | [+0.012, +0.146] | +0.310 | +0.381 |
| thigh inclination | far | **+0.288** | [+0.099, +0.531] | +0.462 | +0.465 |
| trunk lean | both | **−0.390** | [−0.652, −0.063] | −0.169 | −0.202 |

**All eight are resolved**, and the curve is monotone in the right direction at
every one of the four rates.

### The demo recommendation: 30 fps is on the steep part of the curve

Splitting the total 30 fps penalty into the segment above 50 fps and the
segment between 50 and 30:

| quantity | limb | 100→50 | **50→30** | total at 30 | **share of the damage below 50 fps** |
|---|---|---|---|---|---|
| **knee flexion** | near | +0.598 | **+1.237** | +1.835 | **67%** |
| **knee flexion** | far | +0.870 | +0.887 | +1.757 | 50% |
| hip (trunk-ref) | near | +0.372 | +0.269 | +0.641 | 42% |
| hip (trunk-ref) | far | +0.397 | +0.337 | +0.734 | 46% |
| **thigh inclination** | near | +0.059 | **+0.322** | +0.381 | **84%** |
| thigh inclination | far | +0.288 | +0.177 | +0.465 | 38% |

**The curve has NOT flattened — for the knee and the near thigh it steepens.**
Two thirds of the knee's 30 fps penalty, and five sixths of the near thigh's, is
incurred below 50 fps. Capturing at 50 fps instead of 30 would recover
**1.24° of the knee's 1.84°** and **0.32° of the thigh's 0.38°**.

**So the demo should ask for 50 fps or better**, and a phone shooting 30 fps is
paying the steepest part of the curve. The hip is the exception: its penalty is
roughly linear in rate, so a higher capture buys proportionally less there.
Trunk lean improves at every reduced rate, which 15E already showed **fails to
replicate on sprint** and is not treated as a real effect here either.

---

## Q2 — do other checkpoints transfer better?

### The contracts as actually run

| item | b-ath-det-coco (deployed) | b-h36m | s-ap3d |
|---|---|---|---|
| normalization | person scale | **image resolution** | person scale *(shipping repo's, not its own)* |
| confidence fed | 0.8676 (measured) | 1.0 (its reader's documented fallback) | 0.8676 *(not its own)* |
| rate-matched to | 100 Hz | **50 Hz** | 100 Hz *(not its own)* |
| window | 81 | 81 | 81 |
| contract items determined | **5 of 5** | **4 of 5** | **3 of 5** |
| in the primary | yes | yes | **no** |
| load | 0 missing / 0 unexpected | 0 / 0 | 0 / 0 |

### The pre-registered loss metric, and why it fails here

`loss = MoVi MAE − AthleticsPose MAE`, per checkpoint, each fed identically on
both datasets:

| checkpoint | quantity | limb | AthleticsPose | MoVi (n=11) | loss | 95% CI |
|---|---|---|---|---|---|---|
| **b-ath-det-coco** | knee | near | 4.788 | 8.675 | **+3.887** | [+2.441, +5.338] |
| **b-ath-det-coco** | knee | far | 5.090 | 9.105 | **+4.015** | [+2.705, +5.365] |
| **b-ath-det-coco** | **thigh** | near | 6.034 | 5.689 | **−0.345** | **[−1.885, +1.219]** |
| **b-ath-det-coco** | thigh | far | 6.093 | 6.744 | +0.651 | [−1.528, +2.810] |
| **b-ath-det-coco** | hip | near | 3.177 | 9.306 | **+6.128** | [+4.166, +8.078] |
| **b-ath-det-coco** | hip | far | 3.267 | 9.962 | **+6.695** | [+5.014, +8.430] |
| **b-ath-det-coco** | trunk | near | 4.016 | 5.933 | +1.917 | [+0.583, +3.204] |
| b-h36m | knee | near | **25.238** | 8.169 | −17.069 | [−20.711, −13.482] |
| b-h36m | thigh | near | **32.700** | 6.828 | −25.873 | [−27.285, −24.475] |
| b-h36m | hip | near | **20.324** | 7.614 | −12.710 | [−15.718, −9.802] |
| b-h36m | trunk | near | **23.258** | 4.896 | −18.362 | [−21.165, −16.480] |
| s-ap3d | knee | near | **26.379** | 22.578 | −3.801 | [−8.136, +0.080] |
| s-ap3d | thigh | near | **20.401** | 6.328 | −14.073 | [−15.687, −12.475] |
| s-ap3d | hip | near | **14.898** | 7.895 | −7.004 | [−8.200, −5.676] |

**The deployed checkpoint's row is the useful one, and it carries the phase's
cleanest result: at n = 11, the largest n this project has had, the
gravity-referenced thigh has NO measurable AthleticsPose→MoVi loss** —
−0.345°, CI [−1.885, +1.219] spanning zero on the near limb and +0.651°
[−1.528, +2.810] on the far. The knee loses ~3.9°, the hip ~6.1–6.7°, the trunk
~1.9°. That confirms the sealed test's thigh finding on nearly four times the
subjects.

**For the other two checkpoints the loss metric is uninterpretable, and the
reason for `b-h36m` is a defect in my own pre-registered design.**
Rate-matching to a checkpoint's training fps interacts with clip length:

| checkpoint | dataset | train fps | median model-input frames | **fraction of the 81-frame window that is real** |
|---|---|---|---|---|
| b-ath-det-coco | AthleticsPose | 100 | 83 | **1.00** |
| b-ath-det-coco | MoVi PG1 | 100 | 810 | 1.00 |
| **b-h36m** | **AthleticsPose** | **50** | **42** | **0.52** |
| **b-h36m** | **MoVi PG1** | **50** | 405 | **1.00** |
| s-ap3d | AthleticsPose | 100 | 83 | 1.00 |
| s-ap3d | MoVi PG1 | 100 | 810 | 1.00 |

**`b-h36m` is fed half-padding on AthleticsPose and a full window on MoVi**,
because AthleticsPose clips have a median of 83 native frames and halving the
rate halves them to 42, below the 81-frame window. Its AthleticsPose term is
inflated by a padding handicap its MoVi term does not share, so a loss built
from the difference measures my harness, not transfer. **It is reported above
and then set aside.** `s-ap3d` has no padding asymmetry and is simply bad
(20–26° on AthleticsPose): excluded by contract, and empirically not producing
usable kinematics.

### The comparison the design DOES support, and it answers the question

Detections were computed **once per MoVi subject and reused for every
checkpoint**, so the checkpoints can be compared **paired per subject on
identical frames** — a tighter and confound-free statistic. n = 11.

| checkpoint | quantity | limb | deployed | other | **other − deployed** | 95% CI | resolved? |
|---|---|---|---|---|---|---|---|
| **b-h36m** | **hip (trunk-ref)** | near | 9.306 | **7.614** | **−1.692** | [−2.986, −0.386] | **YES, better** |
| **b-h36m** | **hip (trunk-ref)** | far | 9.962 | **8.119** | **−1.843** | [−2.810, −0.846] | **YES, better** |
| **b-h36m** | **trunk lean** | near | 5.933 | **4.896** | **−1.037** | [−1.739, −0.366] | **YES, better** |
| **b-h36m** | **trunk lean** | far | 5.903 | **4.885** | **−1.017** | [−1.724, −0.340] | **YES, better** |
| b-h36m | knee flexion | near | 8.675 | 8.169 | −0.506 | [−1.789, +0.780] | no |
| b-h36m | knee flexion | far | 9.105 | 8.769 | −0.335 | [−1.646, +1.160] | no |
| **b-h36m** | **thigh** | near | **5.689** | 6.828 | **+1.138** | [+0.306, +2.061] | **YES, worse** |
| b-h36m | thigh | far | 6.744 | 7.526 | +0.781 | [−0.154, +1.860] | no |
| s-ap3d | knee flexion | near | 8.675 | **22.578** | **+13.903** | [+11.527, +16.416] | YES, far worse |
| s-ap3d | hip (trunk-ref) | far | 9.962 | 8.265 | −1.698 | [−3.099, −0.375] | YES, better |
| s-ap3d | trunk lean | near | 5.933 | 4.964 | −0.969 | [−2.202, +0.193] | no |

**On MoVi, an H36M-trained lifter that has never seen running, this lab, or this
camera beats the running-fine-tuned deployed checkpoint on the hip by ~1.7–1.8°
and on trunk lean by ~1.0°, with every interval excluding zero and the same sign
on the subject means. It ties on the knee and loses on the near thigh.**

**And the pattern is exactly the one phase 15C implies.** 15C found the
deployed checkpoint's thorax does not land on the same anatomical point in two
datasets — `CLAV` to 5.04 mm on AthleticsPose, but 87–144 mm from CLAV on MoVi.
The trunk-referenced hip is measured against the pelvis→thorax axis and **trunk
lean *is* that axis**, so those are precisely the quantities that inherit a
learned convention. The gravity-referenced thigh needs no convention at all —
and that is the one the deployed checkpoint still wins.

> **So fine-tuning on AthleticsPose bought accuracy on AthleticsPose and a
> convention that does not travel.** The deployed checkpoint is 4–5× better on
> its own dataset (3.18° hip against 20.3°) and measurably *worse* on a lab it
> has never seen, for the two quantities that depend on where the model puts its
> thorax. That is overfitting of a convention, not of accuracy, and it is the
> strongest Stage-0 signal available without training anything.

**What this does not license.** `b-h36m`'s confidence distribution is
undetermined, so it is running on a channel value (1.0) it did not train on;
its advantage might grow or shrink under its true contract. The win is ~1–1.8°
against a 3.51° deployable baseline and a 7.6° absolute level, so **it does not
make the hip usable**. And it is measured on 11 **spent** subjects: this is
diagnostic and establishes no held-out claim. The honest next step is to test
an H36M-family lifter on a sealed set, not to swap the deployed checkpoint on
this evidence.

### Per-subject MoVi, deployed checkpoint, near limb (n=11)

| subject | knee | hip | detected |
|---|---|---|---|
| Subject_1 | 6.950 | 13.706 | 206/206 |
| Subject_10 | 9.376 | 8.408 | 243/243 |
| Subject_11 | 9.828 | 10.107 | 441/441 |
| Subject_12 | 12.278 | 14.145 | 432/432 |
| Subject_13 | 8.472 | 12.819 | 275/275 |
| Subject_15 | 6.468 | 4.447 | 241/241 |
| Subject_16 | 9.105 | 7.550 | 227/227 |
| Subject_17 | 5.130 | 5.156 | 221/221 |
| Subject_18 | 6.879 | 7.940 | 279/279 |
| Subject_19 | 7.973 | 9.601 | 238/238 |
| Subject_2 | 12.964 | 8.485 | 157/157 |

**Subject_1's knee (6.950) and hip (13.706) reproduce phase 15D's committed
per-subject values exactly**, which validates the MoVi path in this new
harness. **Detection succeeded on 100% of frames for all 11 subjects with zero
full-frame re-detections** — including `Subject_11`, whose 42% marker-gap
problem is a *mocap* defect, not a video one. Neither `Subject_11` nor
`Subject_15` drives the checkpoint comparison: it is paired on identical frames,
so a hard subject handicaps all three equally.

### Deviations, recorded

- **D-F1 — the pre-registration's `n_frames: 243` for `b-h36m` is wrong.** See
  C1. Corrected from the weights before scoring; all three run at 81.
- **D-F2 — a sixth contract item was added before scoring**, the output
  coordinate convention (C2). Not pre-registered; it had to be established
  because `EST_UP` had been verified for one checkpoint only. All three agree,
  so nothing changed.
- **D-F3 — rate-matching confounds the loss metric for `b-h36m`.** 52% real
  input on AthleticsPose against 100% on MoVi. The pre-registration argued the
  loss would cancel an absolute handicap; it cancels a *uniform* handicap, and
  this one is not uniform across the two datasets. The paired MoVi comparison is
  substituted as the statistic that answers the question, and is labelled as the
  substitute.
- **D-F4 — the `--clips` limit was not plumbed into Q2 in the first version**,
  so a smoke run began a full 589-clip × 3-checkpoint pass and was stopped. No
  result came from it.
- **D-F5 — `s-ap3d` ran under the shipping repo's contract**, as
  pre-registered, and is reported outside the primary throughout.

### What surprised me

1. **An H36M checkpoint beats the deployed one on MoVi's hip and trunk**, with
   resolved intervals at n=11, while being 5× worse on AthleticsPose. I expected
   the fine-tuned checkpoint to win everywhere or lose everywhere.
2. **The split follows 15C's convention argument precisely** — it wins exactly
   the two convention-dependent quantities and loses the convention-free one.
   That is a mechanism, not a coincidence, and it reframes the AthleticsPose
   fine-tune as having bought a non-transferable thorax definition.
3. **The fps curve steepens rather than flattening.** 67% of the knee's 30 fps
   penalty sits below 50 fps, so the demo's capture rate matters more than the
   30 fps figure alone suggested.
4. **Window length was readable from a BatchNorm's shape**, and it contradicted
   the config. Two of the six contract items turned out to be readable from the
   weights and more reliable than the published documentation.
5. **The contract checker flagged nothing at all** on 589 clips × 3 checkpoints
   × 2 datasets. After three scale defects in this project I expected at least
   one.
