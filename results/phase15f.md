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

*Not yet run at the time of this commit.*
