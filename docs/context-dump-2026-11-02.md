# Context dump — 2026-11-02

Written read-only for planning in a separate chat. Nothing was run, changed or
committed to produce it. Facts in §§1–7; §8 is opinion and labelled as such.

**Status.** The phase 15 full measurement run **completed** (exit 0) and
`results/phase15_movi.json` exists. The headline is §1.5.0. This file was first
written while that run was still going; the sections it changed are patched and
marked.

> **PATCHED 2026-11-02, after the full run landed.** Changes: §1.5.0 added (the
> actual result), §1.2 item 12, §6.1 (the gap is now ~5°, not ~7°), §6.4
> (PG2 and CP1 are no longer unscored), §7.2 (timing was understated by ~2×),
> §8 (first two priorities are done or superseded). **The most important
> addition is §6.5: the CP1 figure is invalid, not a finding.**

---

## 1. SESSION SUMMARY

### 1.1 The prior session (pre-compaction, 2026-10-19 → 10-20)

Executed the approved plan at `~\.claude\plans\phase-0-effervescent-wave.md`
(steps 0a–0d, then phase 14, then phase 13 arm 1, then phase 16 speed).
Outcomes, all committed:

| step | outcome | commit |
|---|---|---|
| 0a | AthletePose3D video retry **failed permanently** — error changed from the 24-hour quota message to "Cannot retrieve the public link", so not a quota that expires | `ed9f383` |
| 0b | `MODEL_FPS` 120 → 100; `error_budget()` double-counting fixed | `78e27b6` |
| 0c | Baseline re-run bit-identical; far limb measured for the first time | `7f0e116` |
| 0d | Baseline restated as two numbers | `6d74aa5` |
| — | **Bug: `video_kinematics.py` fed COCO-ordered keypoints to an H36M-ordered lifter.** Fixed with `coco17_to_h36m17` | `9c6e3b7` |
| 14 | Per-segment confidence: **mixed**. Both shanks clear ρ 0.30 every subject (0.34); thighs do not (0.09, 0.17); only `shank_r` calibrated (slope 0.83). It is a **viewpoint meter**; ~4/5 of the skill is *between clips* | `ee1d525`, `4a5d524` |
| — | Detector headroom: perfect 2D + matched weights = 1.468°/1.626°; both detectors have the **same** mean 2D error (12.28 vs 12.18 px); realistic prize **0.33°**; cleaner input with fixed weights is monotonically **worse** (3.51 → 5.16°) | `9df8143` |
| 16 | Free speed-ups: **1,055 → 207 ms/frame (5.1×)**. **20 fps gate recorded NOT MET**, 5.3× over | `799b71c`, `b290309` |
| 13 arm 1 | Filtering the 2D is **harmful at every setting**: +0.119° [+0.019, +0.176] at the tool's own σ=0.48, +4.613° at 6 Hz. `LOWPASS_HZ` → 0.0. Input-side half of phase 13 **closed** | `bf6861d`, `e432a7e` |
| 15 | Pre-registration frozen **before any MoVi byte on disk** | `64fb0bc` |

### 1.2 This session, in order

**Goal:** open MoVi (held-out) and run phase 15.

1. **Status review** (read-only): `.claude/progress.md`, `git log`, `results/phase15.md`.
2. **User authorised the download** via AskUserQuestion → "Yes, download the slice".
3. **Licence recorded before download** — `docs/data-inventory.md` §8.5, commit
   `3e0ab69`. Licence text came from the Dataverse `termsOfUse` field, **not**
   `README.pdf`, which only points at it.
4. **Round S downloaded** (wrong round, see below): 1,255.7 MB.
   `.venv/Scripts/python.exe scripts/phase15_fetch_movi.py`
5. **Pre-registered checks written and run** → `scripts/phase15_checks.py`,
   `results/phase15_checks.json`, commit `fe247a8`.
6. **Round F downloaded**:
   `.venv/Scripts/python.exe scripts/phase15_fetch_movi.py --round F`
   (first attempt failed: the ranged tar prefix was 115 MB, 0.5 MB short of
   subject 11, and wrote a truncated `.mat`; fixed to `.part`-then-rename and
   `MOCAP_PREFIX_MB = 150.0`).
7. **Three fixes exposed by a second subject** — commit `33ffb3d` (details §6).
8. **Smoke runs on `scripts/phase15_movi.py`** → found the confidence bug.
9. **Bug fixed** — `scores_to_train_scale` in `scripts/video_kinematics.py`,
   commit `2db742f`, plus one test. 50 tests pass.
10. **Deviations and check answers recorded** — `results/phase15.md`,
    `docs/data-inventory.md` §8.5, `LEARNING.md`, commit `d48d75e`.
11. **Perfect-2D diagnostic arm added** (`project_reference`, `--perfect-2d`) —
    commit `71d1a2e`. **Not yet run.**
12. **Full run** — `.venv/Scripts/python.exe scripts/phase15_movi.py` (no
    arguments: every mocap subject on disk × every camera). **Completed, exit 0**,
    wrote `results/phase15_movi.json`. Result in §1.5.0. **Not committed.**

Also, on a direct user question, an **operator decision** was recorded in
`.claude/progress.md`: select the next detector on **convention**, with
**ONNX/OpenVINO quantisation** as the speed lever rather than architecture
shopping. RF-DETR Keypoint (preview 2026-06-22, Apache 2.0 code *and* weights,
`rfdetr-keypoint-preview`, 71.8 AP at 9.8 ms on **T4 + TensorRT FP16**; no
published CPU latency for any size) is the lead candidate, for its custom
skeletons (trainable to H36M's convention on BEDLAM → matched lifter by
construction) and its per-keypoint covariance. The 20 fps gate **stays recorded
as NOT MET**; only the remedy changed.

### 1.3 Files created this session

| path | what |
|---|---|
| `scripts/phase15_fetch_movi.py` | Dataverse fetcher; round S and round F slices; ranged-prefix tar streaming for the mocap |
| `scripts/phase15_checks.py` | The six pre-registered checks; `MOVI_TO_H36M`, `load_v3d`, `to_h36m`, `angle_mask`, `gap_report`, `camera_pose`, `to_camera`, `project_reference`, `travel_azimuth` |
| `scripts/phase15_movi.py` | The measurement; `decode_range`, `jogging_window`, `near_limb`, `run_clip`, `aggregate` |
| `results/phase15_checks.json` | Check answers, subjects 1 and 10 |
| `results/phase15_smoke.json` | **Smoke only**, 150 frames, one clip |
| `results/phase15_movi_manifest.json`, `_f.json` | Download provenance: datafile ids, byte counts, sha256. No MoVi data |
| `docs/context-dump-2026-11-02.md` | This file |

### 1.4 Files modified this session

- `scripts/video_kinematics.py` — `TRAIN_CONF_MEAN = 0.8676`, `TRAIN_CONF_SD = 0.1164`, new `scores_to_train_scale()`, called in `main()` after `coco17_to_h36m17`.
- `docs/data-inventory.md` — new §8.5 (MoVi), check table filled in.
- `results/phase15.md` — Deviations D1–D3, check answers, side-on answer appended **below** the Results heading; nothing above it edited.
- `tests/test_video_pipeline.py` — `test_the_confidence_channel_is_put_on_the_scale_the_lifter_trained_on`.
- `LEARNING.md` — two entries.
- `.claude/progress.md` — operator decision + phase 15 state.

### 1.5.0 THE ACTUAL PHASE 15 RESULT — `results/phase15_movi.json`

```
.venv/Scripts/python.exe scripts/phase15_movi.py
```

No arguments: every mocap subject present (`Subject_1`, `Subject_10`,
`Subject_11`) × every camera in `CAMERAS` (PG1, PG2, CP1), round F, jogging only,
the frozen pipeline **with** the confidence fix.

**Aggregate, subject-bootstrap CI (N_BOOT 2000, SEED 11):**

| camera | limb | hip | knee | **mean** | 95% CI | n subj |
|---|---|---|---|---|---|---|
| **PG1** (side-on) | near | 10.740 | 8.718 | **9.729°** | **[8.892, 10.328]** | **3** |
| **PG1** | far | 10.687 | 10.426 | **10.556°** | **[10.241, 10.966]** | **3** |
| PG2 (frontal) | near | 9.392 | 8.959 | 9.175° | n/a | 1 |
| PG2 | far | 10.928 | 11.808 | 11.368° | n/a | 1 |
| CP1 (iPhone) | near | 59.102 | 24.624 | 41.863° | n/a | 1 |
| CP1 | far | 67.574 | 54.834 | 61.204° | n/a | 1 |

**AthleticsPose deployable baseline, for reference only: 3.5093° near /
3.6177° far** — a different dataset, camera, population and movement.

**Per clip:**

| subject | cam | video frames | detected | fps | jog idx | scored near/far | near limb | ms/frame |
|---|---|---|---|---|---|---|---|---|
| Subject_1 | PG1 | 206 | **206** | 30.000 | 18 | 198 / 197 | l | 1056 |
| Subject_1 | PG2 | 206 | **206** | 30.000 | 18 | 197 / 198 | **r** | 904 |
| Subject_1 | CP1 | 206 | **206** | **30.009** | 18 | 197 / 198 | r | 3243 |
| Subject_10 | PG1 | 243 | **243** | 30.000 | **16** | 243 / 243 | l | 1024 |
| Subject_11 | PG1 | 441 | **441** | 30.000 | **15** | **256 / 256** | l | 892 |

**Detection was 100% on every clip** (1,096 of 1,096 frames).

| subject | cam | near hip (r) | near knee (r) | near mean | far mean |
|---|---|---|---|---|---|
| Subject_1 | PG1 | 13.706 (+0.841) | 6.950 (+0.924) | 10.328 | 10.966 |
| Subject_1 | PG2 | 9.392 (+0.798) | 8.959 (+0.875) | 9.175 | 11.368 |
| Subject_1 | CP1 | 59.102 (**−0.077**) | 24.624 (**+0.022**) | 41.863 | 61.204 |
| Subject_10 | PG1 | 8.408 (+0.665) | 9.376 (+0.930) | 8.892 | 10.241 |
| Subject_11 | PG1 | 10.107 (+0.709) | 9.828 (+0.937) | 9.967 | 10.462 |

Per-subject PG1 near: Subject_1 10.328, Subject_10 8.892, Subject_11 9.967.
PG1 far: 10.966 / 10.241 / 10.462.

**Five things to read off this:**

1. **PG1 is 9.73° near / 10.56° far on 3 subjects**, tight CIs, far limb worse
   than near — consistent in direction with the AthleticsPose deployable path
   (3.51 near / 3.62 far), and **~2.8× worse in absolute terms**.
2. **Knee correlations are high and stable** (+0.92, +0.93, +0.94). The knee is
   genuinely being tracked. **Hip r is lower and more variable** (+0.67 to +0.84)
   and hip MAE is the larger term everywhere.
3. **`jogging` index differs on all three subjects: 18, 16, 15.** Confirms the
   randomised `motions_list` order; a hardcoded index would have measured three
   different movements.
4. **Subject_11 lost 185 of 441 frames (42%)** to the gap mask — by far the worst,
   and worth checking before that subject is trusted.
5. **CP1's figure is INVALID — see §6.5.** It is not evidence about phone footage.

### 1.5 THE SMOKE RUN AND THE BISECTION: 11.45° vs 4.49° vs 3.51°

These are the *diagnostic* numbers that found the confidence bug, on **one clip,
150 frames**. They are superseded as a result by §1.5.0 but are the record of
how the bug was isolated. All three are **different things**. Stated precisely:

#### (a) 11.448° — detected 2D, after the confidence fix

```
.venv/Scripts/python.exe scripts/phase15_movi.py \
    --subjects Subject_1 --cameras PG1 --max-frames 150 \
    --out results/phase15_smoke.json
```

- **Dataset / round:** BML MoVi, **round F**, `doi:10.5683/SP2/JRHDRN`
- **Subject:** `Subject_1` (1 subject, 1 clip, 1 camera)
- **Camera:** **PG1** — Point Grey / FLIR, calibrated, **side-on** (`view_ratio` 0.720)
- **Video:** `data/movi/raw/round_f/F_PG1_Subject_1_L.avi`, 145,303,034 bytes,
  **800 × 600, 30.000 fps** read from the file, 3,441 frames total
- **Mocap:** `data/movi/raw/round_f/mocap/F_v3d_Subject_1.mat`, 32,555,139 bytes, **120 Hz**
- **Movement:** jogging = `motions_list[18]` *for this subject*
- **Window:** `flags30 = [2884, 3089]` (206 frames, 6.87 s); `flags120 = [11533, 12353]`.
  The smoke run was capped at 150 frames → video 2884–3033, mocap 11533–12129
- **Detection:** 150/150 frames, rate 1.000; 149 on a crop, 0 full-frame re-detections
- **Near limb:** `l`, from ground truth (camera is calibrated)
- **Frames scored:** near 145, far 144 (gap sentinel excluded, not interpolated)

| limb | hip | hip r | knee | knee r | mean |
|---|---|---|---|---|---|
| near | **14.694°** | +0.842 | **8.202°** | +0.909 | **11.448°** |
| far | **14.721°** | +0.815 | **8.309°** | +0.869 | **11.515°** |

GT ranges: near hip [−1.2, 47.2], near knee [−73.1, −13.7].
Cost: **551.2 ms/frame**, 82.7 s wall.

#### (b) 30.07° — the same path BEFORE the confidence fix

Same script and clip, `--max-frames 90`: near hip 23.842, knee 36.294, **mean
30.068**; far 22.042 / 36.874 / **29.458**; r hip **−0.36**, knee **−0.19**.
An earlier `--max-frames 60` run gave near mean 30.009 / far 32.613.
A separate lag scan at N=150 (scratchpad `diag_align.py`) gave **lag-0
r = +0.022 hip, +0.167 knee** and **no peak at any lag from −12 to +12**, which
is what ruled out time misalignment. Estimated knee flexion reached **+66.2°**,
outside `video_kinematics.PLAUSIBLE["knee"] = (−170, 20)`.

#### (c) 4.49° — GT 2D (projected reference), same clip

Produced by a **scratchpad** diagnostic (`diag_bisect.py`) in the session
scratchpad, which is transient. The committed equivalent is
`scripts/phase15_checks.py:project_reference` + `phase15_movi.py --perfect-2d`,
which **has not been run yet**.

- Same subject, camera, mocap, window — but the **full** 206-frame jogging
  window, of which **161 frames** had a complete projected skeleton
- 2D built by projecting the reference 3D through PG1's own intrinsics
  (intrinsics not reproduced; read them from Camera Parameters.tar), extrinsics, and radial distortion
  (distortion coefficients not reproduced). H36M-ordered by construction — no detector, no
  COCO conversion. Channel 2 set to 1.0 in the diagnostic
  (`project_reference` now uses `TRAIN_CONF_MEAN`)
- Projected skeleton height **333 px** vs the detector's **325 px**

| limb | hip | hip r | knee | knee r | mean |
|---|---|---|---|---|---|
| near | 6.251° | +0.865 | 2.735° | +0.982 | **4.493°** |
| far | 7.288° | +0.793 | 3.659° | +0.961 | **5.474°** |

For reference, `results/phase07.md` §5 measured perfect 2D into **this same
checkpoint** on AthleticsPose at **5.16°** — so 4.49°/5.47° is in line, which is
what cleared the lifter, the GT mapping, the projection, the alignment and the
angle code.

#### (d) 3.5093° — the AthleticsPose deployable baseline

A **different dataset, camera, population and movement** — not "the same clips".
`results/phase7_inference_full.json`, config `generic COCO + fixes`
(COCO detections + edge padding + overlapped windows), **592 clips**,
**3 subjects** (`TEST_SUBJECTS = ["S11", "S13", "S16"]`, `scripts/phase3_infer.py:53`).

- Pooled clip means: **3.5093° near / 3.6177° far**
- Subject-level with subject bootstrap (N_BOOT 2000, SEED 11),
  `results/phase7_baseline_ci.json`: **3.4528° [3.266, 3.695] near**,
  **3.5597° [3.130, 3.903] far**
- Other configs, subject means near/far: baseline (phase 3) 3.3603/3.3219;
  both fixes 3.0880/2.8660; generic COCO detector 3.8781/3.9832

#### What (a)–(d) together say

The 2D itself is **good**: compared joint by joint against the projected
reference, **11.8 px mean error — better than AthleticsPose's own 12.18–12.28 px**,
with hips 21.3/21.6 px (the definitional `mean(ASI,PSI)` offset), knees 5.9/6.3 px,
ankles 4.9/5.2 px. Every H36M joint matched the COCO slot it should, so
`coco17_to_h36m17` is correct. **The remaining 11.45 − 4.49 = ~7° gap is
unexplained and is the open question.**

---

## 2. UNCOMMITTED STATE

**`git status --short` is empty — the working tree is clean.**

`git status -sb`: `## main...origin/main [ahead 36]` — **36 commits unpushed.**
Nothing has been pushed this session or the prior one.

`git log --oneline -20`:

```
71d1a2e Add the perfect-2D diagnostic arm and record where phase 15 stands
d48d75e Record phase 15's deviations and the six check answers, before the result
2db742f A real bug in the deployable tool: the confidence channel was 14x out of scale
33ffb3d Fix three things the second MoVi subject exposed
fe247a8 Phase 15 checks: MoVi round S cannot carry this measurement, round F can
3e0ab69 Record MoVi's licence and accession before downloading a single byte
64fb0bc Phase 15 pre-registration: freeze the pipeline before MoVi is downloaded
e432a7e Phase 13 arm 1: filtering the 2D is harmful, and the tool had been doing it
1e74bad Learning log: per-band SNR decides a filter, and a model learns its input pipeline
bf6861d Phase 13 arm 1: the filter harness, parametrised by sigma rather than by cutoff
058a71e Phase 13 pre-registration: arm 1 runs, arm 2 is deferred on a geometric argument
b290309 Take the free 5.1x, record the 20 fps gate as failed, and defer the swap
799b71c A 5.1x speed-up is in the call path, and the 20 fps gate still cannot be met
9436442 Learning log: name the hip joint-centre problem instead of mis-citing a paper
9df8143 Bound what a better detector can buy: 0.33 deg, not 2.04, and not drop-in
e04179b Learning log: a permutation control tells you which level a model works at
4a5d524 Record the phase 14 outcome against its own gate
ee1d525 Phase 14 result: the confidence is a viewpoint meter, and it ranks clips
2930056 Phase 14 read the 2D in the wrong joint order too, and the validator caught it
9c6e3b7 The video tool fed the lifter a scrambled skeleton
```

This file (`docs/context-dump-2026-11-02.md`) is newly written and **uncommitted
by instruction**.

Note: scratchpad diagnostics (`diag_align.py`, `diag_bisect.py`, `diag_2d.py`,
`diag_knee.py`, `diag_virtual.py`, `peek_tar.py`, `read_calib.py`,
`extract_v3d.py`, `dump_move.py`, `probe_archives.py`, `range_tar_head.py`,
`stream_tar_head.py`, `fids.py`, `terms.py`, `pdftxt.py`) live only in the
session scratchpad at
`~\AppData\Local\Temp\claude\C--Users-alvin-Overstride-V2\<session-id>\scratchpad`
and are **not in the repo**. They will be lost.

---

## 3. DATA ON DISK

Totals measured with PowerShell; all under `data/`, which is gitignored via
`.gitignore:6` (`data/`), verified with `git check-ignore -v`.

| path | size | files |
|---|---|---|
| `data/ric` | **46.23 GB** | 2,526 |
| `data/athletepose3d` | **4.62 GB** | 14 |
| `data/movi` | **2.40 GB** | 33 |
| `data/athleticspose` | **2.19 GB** | 20,560 |
| `data/derived` | **1.70 GB** | 2,480 |
| `data/fukuchi` | **0.89 GB** | 631 |
| `data/kaggle_stage` | 0.00 GB | 6 |

### 3.1 AthleticsPose — role **tune**

`data/athleticspose/`: `checkpoints/`, `checkpoints.zip`, `data/`, `data.zip`, `repo/`.

`data/athleticspose/data/AthleticsPoseDataset/`:

| subdir | files | contents |
|---|---|---|
| `camera_params` | 14 | per-session JSON (`20250125_discus.json`, …) |
| `det_markers2d_by_cam_coco` | 3,998 `.npy` | generic COCO detector 2D, **H36M-ordered** |
| `det_markers2d_by_cam_ft` | 3,998 `.npy` | fine-tuned detector 2D |
| `gt_bboxes2d_by_cam` | 3,998 | GT boxes |
| `gt_markers2d_by_cam` | 3,998 | **GT 2D** |
| `gt_markers3d_by_cam` | 3,998 `.npz` | **GT 3D** (`markers_h36m`, in PIXELS with per-frame `p2mm`) |
| `raw_markers_in_world` | 500 | raw marker trajectories |

- **24 subjects**, `S00`–`S23`. **8 actions**: discus, hurdle, javelin, racewalk,
  running, sd, shotput, sprint.
- Phase 7/13 use `ACTIONS = ["running", "sprint"]` × `TEST_SUBJECTS = [S11, S13, S16]`
  = **592 clips**, 298 of them multi-window.
- **RAW VIDEO: NONE, permanently.** Upstream README (`data/athleticspose/repo/README.md:12`):
  *"We currently do not plan to release the original video files."*
- **Camera calibration: yes** (`camera_params/`).
- **Licence: CC BY-NC-SA 4.0**, on disk at `data/athleticspose/data/LICENSE-DATASET`
  and `data/athleticspose/data/LICENSES/CC-BY-NC-SA-4.0.txt`.

`data/athleticspose/checkpoints/`:

| file | bytes |
|---|---|
| `motionagformer-b-ath-det-coco-v1.ckpt` | 141,862,608 ← **the deployed one** |
| `motionagformer-b-ath-det-ft-v1.ckpt` | 141,862,544 |
| `motionagformer-b-ath-gt-v1.ckpt` | 141,862,544 |
| `motionagformer-b-h36m.pth` | 47,299,185 |
| `motionagformer-s-ap3d.pth` | 19,953,294 |

### 3.2 AthletePose3D — role **tune** (demoted from held-out in phase 12)

`data/athletepose3d/`, 14 files:

| file | bytes | what |
|---|---|---|
| `pose_3d.zip` | 1,460,224,002 | **GT 3D**, 22,155 `.pkl` under `pose_3d_v3/frame_81/{test,…}/` |
| `data.zipttgd025t.part` | **3,499,098,112** | **INCOMPLETE** download of the video archive |
| `cam_param.json` | 24,206 | **camera calibration** |
| `fetch*.log/.err` ×10, `retry.log`, `peek` | — | download attempt records |

- **RAW VIDEO ON DISK: NONE. Zero subjects, zero clips.** A `find` for
  `*.mp4 *.avi *.mov` under `data/athletepose3d` returns nothing. The only video
  artefact is the 3.5 GB `.part`, which is **not openable** and has been the same
  byte count since 2026-09-14.
- Google Drive refusal is **permanent, not a quota**: `fetch5.err` ends
  *"Cannot retrieve the public link of the file…"*; `results/phase11_fetch.json`
  records `quota_limited: false`. `pose_2d.zip` was refused outright from the
  first attempt.
- Clip index `data/derived/ap3d_clip_index.json`: `n_clips` **22,153**,
  `n_matched` **21,529**, `n_unmatched` **624**; `by_action` = fs 11,124 /
  tnf 8,533 / rm 1,872; **`running_subjects: ["S2", "S3", "S4"]`**;
  normalisation `x/w*2-1, y/w*2-h/w`; tolerance 1e-05.
- GT 2D: **not on disk** (`pose_2d.zip` never downloaded).
- **Licence as recorded in `CLAUDE.md`: non-commercial research only.**

### 3.3 Fukuchi et al. 2017 — used in phase 10

`data/fukuchi/`, 0.89 GB, 631 files, 293 top-level entries. Pattern per subject:
`RBDS###processed.txt`, `RBDS###static.txt`, and
`RBDS###run{T25,T35,T45}{markers,forces}.txt` — i.e. **marker trajectories at
150 Hz**, three treadmill speeds. Plus `_dl.sh`, `_urls*.txt`, `_manifest.json`,
`batch.log/.err`, `footab.log/.err`.

- **No video, no camera calibration, no GT 3D in a pose format** — marker text files.
- **Licence as recorded: unknown.** Fukuchi is not in `CLAUDE.md`'s pose-dataset
  table (it is a *marker* source for the MATLAB pipeline, phase 10).

### 3.4 BML MoVi — role **held-out test**, opened 2026-11-02

`doi:10.5683/SP2/JRHDRN`, Scholars Portal Dataverse (borealisdata.ca).
Record: **1,057 files, 290.2 GB**, `fileAccessRequest: false`, **0 restricted**.
1 KB range probes on datafiles 85637, 92073, 193807 each returned **206 / 1024
bytes** with no credential, so the `README.pdf` (2020-03-29) request-access
procedure is **stale**.

| path | du | contents |
|---|---|---|
| `data/movi/docs` | 140K | `README.pdf` (73,083 B), `camera_parameters.tar` (67,072 B) |
| `data/movi/calib` | 78K | `Calib/{cameraParams,Extrinsics}_{PG1,PG2}.{mat,npz,pkl}` |
| `data/movi/raw` | 2.4G | round S slice + `round_f/` |
| `data/movi/v3d` | 23M | round S subjects 22, 45, 74, 80 extracted |
| `data/movi/f_probe` | 32M | first copy of `F_v3d_Subject_1.mat` |

**Round S slice (WRONG ROUND, kept as the record), `data/movi/raw/`, 1,255.7 MB:**
`S_V3D.tar` (619,532,288 B, id 92073), `S2_PG1_Subject_{45,74,80}_L.avi`
(99.1/98.2/102.4 MB, ids 85637/85559/85654), `S2_CP{1,2}_Subject_22.{mp4}`
(176.1/160.4 MB, ids 86001/86078).

**Round F slice (what phase 15 uses), `data/movi/raw/round_f/`, ~1.14 GB:**

| file | datafile id | MB |
|---|---|---|
| `F_PG1_Subject_1_L.avi` | 85268 | 145.3 |
| `F_PG1_Subject_10_L.avi` | 85238 | 250.6 |
| `F_PG1_Subject_11_L.avi` | 85274 | 300.9 |
| `F_PG2_Subject_1_L.avi` | 85331 | 132.4 |
| `F_CP1_Subject_1.mp4` | 86002 | 309.6 |
| `mocap/F_v3d_Subject_1.mat` | 128299 (tar) | 32.6 |
| `mocap/F_v3d_Subject_10.mat` | 128299 (tar) | 43.9 |
| `mocap/F_v3d_Subject_11.mat` | 128299 (tar) | 52.4 |

The mocap came from a **ranged GET of the first 150 MB** of
`F_Subjects_1_45.tar` (1,814,554,112 B) read with `tarfile` in stream mode — the
remaining ~1.7 GB was never transferred, because subjects 1, 10 and 11 are the
archive's first three members. Subjects were chosen **by archive order**, for
streaming cost and nothing else.

**What each modality gives:**
- **Raw video: YES**, and the two camera families **differ**, contrary to
  `CLAUDE.md`'s "all 800×600 **30 fps**" (see §6.5):
  - **PG** = Point Grey / FLIR (machine vision): **800 × 600, 30.000 fps**,
    3,441 frames for Subject_1
  - **CP** = Camera Phone (hand-held iPhone 7): **1920 × 1080, 30.009 fps**,
    **4,369 frames** for Subject_1 — i.e. **928 frames longer than PG**, which is
    why the PG-referenced `flags30` window cannot be applied to it
- **GT 3D: YES** — Visual3D export, **120 Hz**, **87 markers** (67 physical +
  20 virtual) including `LFWT`/`LBWT` = **Left ASIS/PSIS** and `RFWT`/`RBWT`, so
  **H36M's own `mean(ASI,PSI)` hip is exactly computable**. Also `LKJC`/`RKJC`,
  `LAJC`/`RAJC`, `PELVIS`, `THORAX`, `HEAD`, `MHIP`, `LHIP`/`RHIP` (hip joint
  centres), and `jointsAffine_v3d` / `jointsTranslation_v3d` / `jointsExpMaps_v3d`
  for 15 segments.
- **GT 2D: no** — but computable by projection for PG cameras.
- **Camera calibration: PG1 and PG2 ONLY. Zero CP entries.** The hand-held phone
  cameras ship **no calibration**, which fires the clause frozen in
  `results/phase15.md`: CP clips carry an MAE only if sync can be verified,
  otherwise they are **traces with no error figure**.
- **Round S is unusable for this phase:** 12 physical markers (head, wrists,
  ankles), **no hip, knee or pelvis**, no `motions_list`, no frame flags.
- **Licence:** non-commercial scientific research; derivative works permitted;
  **no distribution in any form to any third party**; **explicitly prohibits
  training methods for commercial use**; citing the dataset URL instead of the
  paper is non-compliant (cite PLoS ONE 16(6) 2021, `10.1371/journal.pone.0253157`).
  **No MoVi frame, crop, overlay or render may ever be published.** Recorded
  verbatim in `docs/data-inventory.md` §8.5.

### 3.5 Ferber / Running Injury Clinic

`data/ric`, 46.23 GB, 2,526 files. Plays **no part** in the measurement track.
`data/derived` (1.70 GB) holds the generated waveforms
(`waveforms_mean.npy`, `(n_sessions, 54, 101)` float32) and `waveforms_steps/`.

### 3.6 Datasets named but NOT on disk

AMASS, BEDLAM, AddBiomechanics, dual-system set (`10.5281/zenodo.19720803`,
**CC BY 4.0**, 1.1 GB), Bath BioCV (`10.15125/BATH-01258`, **All Rights
Reserved**, ~225 GB). Existence and licence established in
`docs/data-inventory.md` §8.4; nothing downloaded.

---

## 4. CURRENT PIPELINE

### 4.1 `scripts/video_kinematics.py` — the deployable tool

Constants: `MODEL_FPS = 100.0`, `PERSON_SCORE = 0.90`, `LOWPASS_HZ = 0.0`,
`CROP_PAD = 0.35`, `PLAUSIBLE = {"hip": (-40.0, 60.0), "knee": (-170.0, 20.0)}`,
`TRAIN_CONF_MEAN = 0.8676`, `TRAIN_CONF_SD = 0.1164`.

| # | stage | implementation | detail |
|---|---|---|---|
| 1 | decode | `decode(path, every, max_frames)` | OpenCV; **always from frame 0**, no range; returns `fps / every` |
| 2 | detect | `detect_coco17(frames, crop=True, model=None)` | torchvision **`keypointrcnn_resnet50_fpn`**, `KeypointRCNN_ResNet50_FPN_Weights.DEFAULT`, `box_score_thresh=0.5`, **`min_size=min(h,w)`, `max_size=max(h,w)`** (native resolution, no upscaling). One full-frame pass, then a **padded person crop** tracking the previous frame; full-frame **re-detection** whenever a keypoint leaves the window. Rejects a frame if `scores.max() < 0.5`; picks `argmax(scores)`. **Undetected frames are left ALL ZEROS.** `PERSON_SCORE = 0.90` is used only to count `frames_crowded` |
| 3 | smooth | `smooth(kp, fps, cutoff_hz)` | **OFF** (`LOWPASS_HZ = 0.0`) — phase 13 measured it harmful |
| 4 | joint order | `coco17_to_h36m17(kp)` | COCO-17 → H36M-17. Verified against the projected MoVi reference: every joint matches the right COCO slot |
| 5 | **confidence scale** | **`scores_to_train_scale(kp)`** | **NEW this session.** Channel 2 ← `TRAIN_CONF_MEAN` (0.8676) for detected frames, **0.0** for undetected. Coordinates untouched |
| 6 | rate match | `resample(seq, n_out)` | linear `np.interp` per coordinate; `n_model = max(81, round(len(kp) * MODEL_FPS / fps))` |
| 7 | lift | `phase7_inference_fix.lift(model, kp, pad="edge", windows="overlap")` | `motionagformer-b-ath-det-coco-v1.ckpt`, MotionAGFormer-B, 11,721,795 params. Input `(1, CLIP=81, 17, 3)`: channels 0–1 are `normalize_kpts`-normalised x,y with z zeroed; **channel 2 is `det2d[:, :, 2:3]` passed through verbatim as a per-frame weight**. 50 %-overlap sliding windows blended with a **centred triangular** weight → ~40 future frames of latency |
| 8 | angles | `phase3_angles.sagittal_angles(kp, side)` | Body frame from `up = thorax − pelvis`, `ml = r_hip − l_hip`, `fwd = cross(ml, up)`, re-orthogonalised. `hip = signed_angle(−up, thigh)`, `knee = signed_angle(thigh, shank)`, wrapped to (−180, 180]. **`ankle` is always NaN** — H36M-17 has no toe |
| 9 | guard | `check(angles, kp, meta, n)` | refuses to write output when the subject can't be found or angles leave `PLAUSIBLE` |
| 10 | budget | `error_budget()` | reads the deployable row from the results JSON |

**The confidence-channel contract.** The checkpoint was trained on
AthleticsPose's released `det_markers2d_by_cam_coco`, whose channel 2 is a
**normalised confidence**: measured over 60 clips / 131,121 values —
**mean 0.8676, sd 0.1164, min 0.0592, max 1.1812, p1/p25/p50/p75/p99 =
0.423 / 0.837 / 0.899 / 0.940 / 1.017**. torchvision's `keypoints_scores` are
**raw heatmap scores**: on MoVi **[3.914, 18.382], mean 12.424, median 11.590,
p99 17.785** — **~14× the trained scale.**

### 4.2 `scripts/phase15_movi.py` — the smoke-run path

Differences from `video_kinematics.py`, all deliberate:

| | `video_kinematics.py` | `phase15_movi.py` |
|---|---|---|
| decode | `decode()`, from frame 0, no range | **`decode_range(path, a, b)`** — reads sequentially and discards up to `a`. Deliberately **not** `CAP_PROP_POS_FRAMES`, which lands on a keyframe and would shift the clip against the mocap |
| frame window | all / `max_frames` | **the jogging window only**, from `flags30` |
| detector model | built per call | `det_model = None` on purpose, so `detect_coco17` rebuilds with `min_size` = that camera's own shortest side (PG is 800×600, CP is phone) |
| undetected frames | `check()` refuses to write | **excluded from scoring** via `detected = abs(kp[:,:,:2]).sum(axis=(1,2)) > 0`; the clip is **refused** if fewer than 20 remain |
| after lifting | angles at `MODEL_FPS` | `resample(est_model, n_v)` **back onto the video grid** |
| scoring | no GT | GT at **exactly** mocap frame `f120[0] + 4k`; **the reference 3D is never resampled** |
| near/far | — | from GT depth in camera coords when calibrated, else from the estimate (phase 14: 592/592) |
| extras | — | Pearson **r** per joint, GT range per joint, subject-bootstrap CIs, `--perfect-2d` arm |

`scripts/phase15_checks.py` holds the MoVi-side mapping: `MOVI_TO_H36M`
(hips = `mean(LFWT,LBWT)` / `mean(RFWT,RBWT)`, knees `LKJC`/`RKJC`, ankles
`LAJC`/`RAJC`, thorax `THORAX`, spine = midpoint(pelvis, thorax), shoulders/
elbows/wrists the `*JC` virtuals, head `HEAD`, neck `CLAV`), `angle_mask`
(the 7 joints `sagittal_angles` actually reads: pelvis, thorax, both hips, and
that side's hip/knee/ankle), and `project_reference`.

**Time alignment is exact, not estimated.** `flags120 / flags30` median
**3.9981**, max deviation from 4 = **0.0112**. This also satisfies the
pre-registered 30 Hz control *on the MoVi side* by construction.

---

## 5. ENVIRONMENTS

### Local machine
- **OS:** Windows 11 Home, `10.0.26300` (`Windows-10-10.0.26300-SP0`)
- **Shell:** PowerShell 5.1 primary; Git Bash available
- **CPU:** Intel Core Ultra 7 255H — **16 cores, 16 logical**
- **RAM:** 31.4 GB
- **GPU:** Intel Arc 140T (16 GB), driver 32.0.101.8860 — **not CUDA**, unusable by torch
- **Free disk:** **612.1 GB free of 930.5 GB** on C:
- **Python:** 3.11.9, venv at `~\Overstride_V2\.venv` (`.venv/Scripts/python.exe`)

### Packages
| package | version |
|---|---|
| torch | **2.14.0+cpu** |
| torchvision | **0.29.0+cpu** |
| CUDA | **None** — `torch.cuda.is_available() == False`, `torch.version.cuda is None` |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |
| opencv (cv2) | 4.14.0 |
| pypdf | installed |
| torch threads | 16 |

**Not installed:** `mmpose`, `onnxruntime`, `openvino`, `fitz`/PyMuPDF.

**Is MMPose installable here? Unknown, and likely difficult.** MMPose requires
`mmcv`, which ships prebuilt wheels only for specific torch × CUDA pairs and
otherwise compiles from source; torch **2.14.0+cpu** is far newer than the
versions `mmcv` publishes wheels for, so a source build (needing MSVC toolchain)
is the probable path. Not attempted this session.

**Colab:** no Colab setup exists in the repo. `notebooks/` contains only
`phase3_kaggle.ipynb` and `phase3_kaggle.py`, the **superseded** route.
Per `CLAUDE.md`, phase 3 was planned for Kaggle, two runs were given neither GPU
nor Internet (account not phone-verified, Kaggle withholds both silently), and
phase 3 ran **locally on CPU** in ~90 min instead. `docs/phase3-setup.md` is the
superseded doc.

### Known fragile
- **No GPU of any kind usable by torch.** Everything is CPU. `CLAUDE.md` forbids
  proposing solutions requiring a local GPU, and forbids uploading the Ferber
  archive to cloud storage.
- **mediapipe / BlazePose is blocked on this machine by Smart App Control**
  (`docs/AGENT_CONTEXT.md` §8).
- **Heredocs through the Bash tool eat backslashes** — a recurring failure this
  project has hit repeatedly. Workaround: write patch scripts with the Write tool.
- **Python stdout is buffered** when not a tty, so a backgrounded run shows an
  empty output file until it exits (currently masking the phase 15 full run).
- `git` warns `LF will be replaced by CRLF` on most commits; harmless.
- The GitHub MCP server **failed to connect** this session:
  `AUTH_HEADER_REJECTED`, *"credential expired"*.

---

## 6. KNOWN ISSUES AND OPEN QUESTIONS

### 6.1 The live one: ~5° is unexplained

**Updated against the full run.** On the whole PG1 jogging window for Subject_1
the detected path gives **10.328° near**, and the projected-reference (perfect)
2D on that same window gives **4.493° near** — so the gap is **~5.8°**, not the
~7° the 150-frame smoke run suggested. Across all three PG1 subjects the
detected figure is **9.729° [8.892, 10.328]**; the perfect-2D arm has **not**
been run on subjects 10 and 11, so the gap is only established on Subject_1.

The puzzle is unchanged and is the main open question: the detected 2D measures
**11.8 px** mean error against the projected reference — *better* than
AthleticsPose's own 12.18–12.28 px — and correlations are healthy (knee r +0.92
to +0.94 across subjects). **Nothing yet explains the gap**, and **the hip is
the larger term everywhere** (near hip 8.4–13.7° vs near knee 7.0–9.8°, and hip
r is both lower and more variable at +0.67 to +0.84).

Note that **perfect 2D already gives 6.25° hip** against a 3.5° baseline, so a
substantial part of the hip term exists with **no detector involved at all** —
which points at a convention/definition term rather than detector noise.

Hypotheses **not** tested:
- a systematic offset rather than noise (r is high, so an offset is plausible) —
  the per-joint bias/offset has not been separated from the MAE;
- the trunk convention: MoVi's `THORAX` virtual marker vs H36M's thorax joint.
  Note **perfect 2D also gives 6.25° hip**, above the 3.5° baseline, so part of
  the hip term is already present without any detector;
- the constant confidence being a poor substitute for the real per-keypoint
  weight;
- phase 11's measured **1.84× lifter degradation** on an independent lab;
- `TRAIN_CONF_SD` is defined but **unused** — only the mean is applied.

### 6.2 Bugs found and fixed this session

1. **Confidence channel 14× out of scale** (`2db742f`) — §4.1. Fixed with a
   constant whose choice is *a priori*; **zero parameters fitted to held-out data**.
   A quantile map onto the tune-set distribution is the obvious upgrade but
   **cannot be validated without a tune set that ships video**.
2. **Gap exclusion was global, should be per quantity** (`33ffb3d`) —
   `markerGaps == 1` marks a missing marker and Visual3D writes `[0,0,0]`.
   Read literally it put the knee joint centre at the lab origin and drove knee
   flexion to **+148.7°**. Excluding every frame with a gap in *any* mapped
   marker cost **21.2%** of subject 1's window, mostly `LEJC` (121), `REJC` (69),
   `LWJC` (27) — elbows and a wrist, which no sagittal hip/knee angle reads.
   Correctly scoped: **3.9% left / 4.5% right**; subject 10 loses **nothing**.
3. **`travel_azimuth` averaged direction vectors instead of angles** (`33ffb3d`)
   — circular statistics. Read 88.6° and 36.3° for two clips whose `view_ratio`
   was 0.72 and 0.71. Now per-frame over an 83 ms central difference, median:
   **75.7°** and **75.0°**.
4. **`fetch_mocap_prefix` wrote a truncated `.mat`** (`33ffb3d`) when the ranged
   prefix cut off mid-member — a plausible filename with no marker. Now
   `.part`-then-rename with a size check against the tar header.

### 6.3 Claims of mine that were wrong and are corrected on the record

- **`motions_list` order is randomised per subject.** Jogging is index **18** for
  subject 1 and **16** for subject 10; `running_in_spot` moves from **7** to
  **19**. An earlier commit message stated the index as a dataset constant.
- **`LHJC`/`RHJC` are the _hand_ joint centres, not the hip.**
  `markerDescription` reads "Left Hand Joint" and their height matches the elbow.
  Hip joint centres are **`LHIP`/`RHIP`**. Wrong marker → 325 mm offset; right
  marker → **104/110 mm** (subject 1) and **90/92 mm** (subject 10), which
  independently reproduces the **81 mm** phase 12 measured on AthleticsPose.
- Earlier in the prior session: a self-cancelling azimuth proxy and a COCO/H36M
  indexing bug voided **phase 14 attempt 1**, preserved at
  `results/phase14_confidence_frozen_protocol.json`.

### 6.4 Still open, named so it is not mistaken for done

- **The AthleticsPose-side 30 fps control has not been run.** MoVi's comparison
  is already on a 30 Hz grid, so the MoVi half is answered by construction, but
  the tune set has **not** been re-scored at 30 fps, so the frame-rate penalty is
  **not** isolated.
- **`--perfect-2d` has never been run** through `phase15_movi.py`. The 4.49°
  figure is from a transient scratchpad script, on Subject_1 / PG1 only.
- **CP1 was scored but the number is INVALID — see §6.5.** Whether the tool works
  on phone footage is **still completely open**, and nothing measured so far
  bears on it.
- **PG2 (frontal) was scored on Subject_1 only**: 9.175° near / 11.368° far,
  against PG1's 10.328° / 10.966° on the same subject. **The frontal camera is
  not worse near-limb**, which runs *against* phase 11's viewpoint curve
  (3.50° frontal → 1.77° side-on). n = 1, so this is suggestive only, and
  `F_PG2_Subject_{10,11}_L.avi` (ids unknown, not fetched) would settle it.
- **n = 3 MoVi subjects on PG1, n = 1 on PG2 and CP1**, and the AthleticsPose
  side of every comparison rests on S11, S13, S16.
- **Subject_11 lost 185 of 441 frames (42%)** to the gap mask, far more than the
  other two. Unexplained; should be checked before that subject is trusted.
- **`results/phase15_movi.json` is uncommitted**, and `results/phase15.md` still
  says *"The full run is in progress at the time of this commit"* under Results.
  **No phase 15 result has been written into the report yet.**
- `docs/AGENT_CONTEXT.md` is **stale** (user deferred): §177 still says 3.4°, and
  its "capture rate not documented anywhere" line is now false.
- `results/phase12_scan.json` is **4 MB in git**.
- Phase 13's gate in `CLAUDE.md` still describes the old single-shot plan rather
  than the three arms.
- **Phase 13 arms 2 and 3 were never run** (scan-as-consistency-constraint,
  and both).
- **36 commits unpushed.**

### 6.5 THE CP1 NUMBER IS INVALID — a bug in `phase15_movi.py`, found while patching this dump

**Do not read 41.863° / 61.204° as "the tool fails on phone footage". It is not
a measurement of anything.**

`run_clip` takes the jogging window from `flags30` and applies it to whichever
video it is given. But the videos are **not the same length or the same format**:

| camera | resolution | total frames | fps |
|---|---|---|---|
| `F_PG1_Subject_1_L.avi` | **800 × 600** | **3,441** | 30.000 |
| `F_PG2_Subject_1_L.avi` | 800 × 600 | 3,441 | 30.000 |
| `F_CP1_Subject_1.mp4` | **1920 × 1080** | **4,369** | **30.009** |

`flags30 = [2884, 3089]` is a **PG-referenced** frame index. The CP recording is
**928 frames longer** (4,369 vs 3,441, a ratio of 1.27) and runs at a slightly
different rate, so **CP frame 2884 is not the same instant as PG frame 2884.**
The CP1 arm therefore scored a *different part of the recording* against the
jogging mocap. That is exactly what its numbers look like: **r = −0.077 hip and
+0.022 knee**, i.e. no relationship at all — the same signature the confidence
bug produced, and for the same reason (comparing two unrelated signals).

**This also corrects two things stated elsewhere:**

1. **`CLAUDE.md` says all four MoVi cameras are "800×600 **30 fps**".** That is
   **wrong for CP**: the phone cameras are **1920 × 1080**. The claim should be
   corrected there and in `docs/data-inventory.md` §8.5.
2. **It is further evidence for the already-recorded finding that CP has no
   calibration and no verified synchronisation.** `results/phase15.md`'s frozen
   clause — CP clips carry an MAE *only if* sync can be verified, otherwise they
   are "traces with no error figure" — now has a concrete reason to fire: the
   sync **cannot** be taken from `flags30`, because those flags are indexed to a
   different camera's frame count.

**What would be needed to score CP at all:** an independent CP-to-mocap
alignment. Options, none verified: a per-camera flag set (does MoVi ship one?
**unknown**), a cross-correlation of CP against PG or against the mocap to
recover the offset (that is *estimating* sync, which the pre-registration says
disqualifies the MAE), or treating CP as traces only. **My reading: CP1 should
be dropped from the MAE and reported as traces**, which is what the frozen
pre-registration already prescribes.

### 6.6 `docs/REVIEW.md` §7 — still open

| # | item | state |
|---|---|---|
| 1 | Dose-response used a proxy (`InjDefn` measures training disruption, not mechanical severity). ρ = +0.044, CI spans zero — "not a knockout alone" | **OPEN** |
| 2 | Repeatability rests on non-independent pairs — 258 pairs from 72 subjects; 0.667 has real uncertainty | **OPEN** |
| 3 | The +0.54° detector cost was a proxy on a biased subsample | **CLOSED (A2)** — +0.45° on all 592 clips; fix effect −0.25° |
| 4 | 120 fps was inferred from cadence | **CLOSED (phase 11)** — AP3D records 120 fps in metadata |
| 5 | **The video tool has never seen real footage** | **OPEN in REVIEW.md — but this is what phase 15 is now closing.** The blocker named there ("dataset distribution") is resolved by MoVi, which ships 30 fps video synchronised to 120 Hz mocap. **REVIEW.md has not been updated.** |
| 6 | One dataset, one protocol; no external validation of generality | **OPEN** for the cohort/injury half. Accuracy half qualified by phase 11: 1.25° AthleticsPose vs 2.29° AthletePose3D under matched GT-2D = **1.84× degradation** |
| 7 | Two documented protocol deviations (labelling rule vs README; phase 1 kill criterion fired and work continued) | **OPEN by design** — recorded, not hidden |
| 8 | Waveforms come from a patched MATLAB pipeline | **LARGELY CLOSED (phase 10)** — reproduces `dv_r` exactly; driven by Fukuchi's markers, speed to 1.71%, hip \|r\| 0.993 |

---

## 7. TIMING

### 7.1 Detector speed profile — `results/phase16_detector_speed.json`

Synthetic frames, 16 threads, 6 frames per config. Budget 50 ms/frame (20 fps);
lifter measured at **11.2 ms** per frame of video.

| configuration | ms/frame | fps | × over 50 ms |
|---|---|---|---|
| current path, 640×480 (`min_size=800`) | **1054.7** | 0.95 | 27.2 |
| current path, 1280×720 (`min_size=800`) | 1252.0 | 0.80 | 32.3 |
| native resolution, 640×480 (`min_size=480`) | 464.1 | 2.15 | 12.0 |
| half resolution, 320×240 (`min_size=320`) | 332.0 | 3.01 | 8.6 |
| tight person crop, 192×256 (`min_size=256`) | 248.2 | 4.03 | 6.4 |
| crop + batch of 4 | 237.9 | 4.20 | 6.1 |
| crop, 4 threads | 469.6 | 2.13 | 12.1 |
| crop, 8 threads | 264.8 | 3.78 | 6.8 |
| **crop, 16 threads** | **206.7** | **4.84** | **5.3** |

So **1,055 → 207 ms/frame, 5.1×**, with the **same weights** — no retraining and
no shift in the keypoint distribution. Still **5.3× over** the detector's ~39 ms
share of a 50 ms frame. Earlier figures in the plan: 2,790 ms at 640×480 and
3,280 ms at 1280×720, and the lifter at 455 ms per 81-frame window = 11.2 ms per
frame of video.

### 7.2 Measured on MoVi — **full run, supersedes the smoke figures**

| clip | resolution | frames | **ms/frame** |
|---|---|---|---|
| Subject_1 PG1 | 800 × 600 | 206 | **1056** |
| Subject_1 PG2 | 800 × 600 | 206 | **904** |
| Subject_1 **CP1** | **1920 × 1080** | 206 | **3243** |
| Subject_10 PG1 | 800 × 600 | 243 | **1024** |
| Subject_11 PG1 | 800 × 600 | 441 | **892** |

**PG is 892–1,056 ms/frame on real 800×600 footage — roughly 4.3–5.1× the
206.7 ms the synthetic profile in §7.1 measured**, and 18–21× the 50 ms budget.
Two reasons: §7.1 used synthetic frames on a *tight 192×256 person crop*,
whereas these are real 800×600 frames where the subject is ~325 px tall and the
padded crop is correspondingly larger; and the per-clip model build amortises
over only a few hundred frames.

**CP1 at 1920×1080 costs 3,243 ms/frame — 3.1× the PG cost**, which is the
expected consequence of `min_size = min(h, w) = 1080` instead of 600.

Superseded smoke figures, kept for the record: 551.2 ms/frame (150 frames),
562 ms (60), 573 ms (90) — all **understated**, because a short window was
truncating the expensive part of the clip.

### 7.3 Download throughput

Dataverse `api/access/datafile/<id>`: **6.5–8.7 MB/s**. Round S slice
1,255.7 MB in ~157 s total. Round F mocap: 150 MB ranged prefix instead of
1,814.6 MB.

### 7.4 Not measured

Per-stage breakdown (decode / detect / convert / resample / lift / angles)
**within** a MoVi clip — **unknown**. Only the whole-clip ms/frame exists.

---

## 8. WHAT I WOULD DO NEXT AND WHY

**This section is my opinion, not a finding.**

**First — DONE, and the result is 9.73° near / 10.56° far on PG1, n=3.** What
remains is to *write it up*: `results/phase15.md` still says the run is in
progress, and `results/phase15_movi.json` is uncommitted. I would commit the
JSON and append Results before doing anything else, because the pre-registration
is explicit that a loss goes in the README as prominently as a win, and
**~2.8× worse than the AthleticsPose baseline is a loss that needs stating.**

**Second — fix the CP1 window bug before CP is mentioned anywhere.** §6.5: the
`flags30` indices are PG-referenced and CP is 928 frames longer at a different
resolution and rate, so CP1 scored the wrong moment. My view is that CP should
be **dropped from the MAE and reported as traces**, exactly as the frozen
pre-registration prescribes, rather than aligned by cross-correlation — because
estimating the sync is what the pre-registration says disqualifies the MAE.
`CLAUDE.md`'s "800×600 30 fps" claim about all four cameras also needs fixing.

**Second-and-a-half, run `--perfect-2d` on all three PG1 subjects.** It is cheap
— no detector — and it turns a bare 9.73° into a decomposition: detector cost vs
the lifter arriving in a lab it has never seen. Right now that split exists only
for Subject_1 (10.33° detected vs 4.49° perfect). Without it the headline number
is hard to act on.

**Third, chase the ~5.8°, and I would start with bias, not noise.** Knee r is
+0.92 to +0.94 across all three subjects while knee MAE is 7–10°, and hip r is
+0.67 to +0.84 with hip MAE 8–14°. High correlation with a large MAE is the
signature of a **systematic offset**, not scatter.
I would compute per-joint mean signed error alongside the MAE, and separately
test the trunk convention (MoVi `THORAX` vs H36M thorax) — because perfect 2D
*also* gives 6.25° hip against a 3.5° baseline, which says part of the hip term
exists with no detector involved and is therefore a **convention** term, exactly
like the 21–22 px hip offset. If that is confirmed, the honest reporting is a
convention-corrected number beside the raw one, not a single figure.

**Fourth, I would not start the detector swap yet.** The confidence bug just
demonstrated that the deployable path had an unvalidated seam for its entire
life; there may be more, and MoVi is now the only instrument that can find them.
I would spend the next effort on *measuring what is there* rather than on
RF-DETR, however attractive it is. When it does start, the convention-first
order the operator chose is right, and `--perfect-2d` plus the per-joint pixel
comparison is the harness it needs.

**Fifth, a process point I think matters more than any single number.** Both
defects found this week — the COCO/H36M joint order and the confidence scale —
lived in the one code path with no ground truth to check it against, and both
survived every prior phase. Finding two is not evidence that path is now clean.
I would add a standing check that compares **every input channel's distribution**
against the distribution the weights were trained on, and run it as a test
rather than as a diagnostic.

**Fifth-and-a-half, a new one from the full run: the viewpoint curve may not
replicate.** PG2 is the near-frontal camera (`view_ratio` 0.468, azimuth 10.5°)
and it scored **9.175° near** against PG1's **10.328°** on the same subject —
i.e. frontal was *not* worse. Phase 11 measured 3.50° frontal → 1.77° side-on,
so this is the opposite direction. n = 1, and it could easily be noise or the
near/far assignment flipping (PG1 near limb is `l`, PG2's is `r`, which is
correct for opposed cameras). But it is cheap to test — two more PG2 videos —
and if it holds it weakens a measured claim the project currently leans on.

**Sixth, housekeeping I would not defer much longer:** 36 unpushed commits,
`docs/REVIEW.md` §7.5 now materially out of date (its blocker is resolved),
`docs/AGENT_CONTEXT.md` stale, and the scratchpad diagnostics that found this
bug will be lost unless the useful ones are moved into `scripts/`.
