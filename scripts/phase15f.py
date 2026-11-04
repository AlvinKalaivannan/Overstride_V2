"""Phase 15F: Q1 the 50 fps arm, Q2 do other checkpoints transfer better?

Pre-registered in `results/phase15f.md`, committed alone as `d198eac`. No gate
on either question.

Q1 extends the 15E harness with one decimation rule, every 2nd frame = exactly
**50.0 fps** (not 60 -- 100/60 is not an integer).

Q2 scores each checkpoint on AthleticsPose and on all 11 spent MoVi PG1
subjects, feeding each its own contract (`scripts/input_contract.py`), and
compares the **loss** = MoVi MAE - AthleticsPose MAE. Each checkpoint is fed
identically on both datasets, so an absolute handicap largely cancels in the
difference.

Run: .venv/Scripts/python.exe scripts/phase15f.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data" / "athleticspose" / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from athleticspose.models.MotionAGFormer.model import MotionAGFormer  # noqa: E402
from athleticspose.utils import mocap_to_h36m  # noqa: E402
from input_contract import CONTRACTS, check, normalize, summarize  # noqa: E402
from phase12_scan import N_BOOT, SEED, subject_bootstrap_ci  # noqa: E402
from phase15_checks import (angle_mask, camera_pose, load_v3d,  # noqa: E402
                            to_camera, to_h36m)
from phase15_movi import (CAMERAS, MOCAP, MOCAP_PER_VIDEO, MOVI,  # noqa: E402
                          decode_range, jogging_window)
from phase15c_thorax import align_per_frame, gt3d_mm  # noqa: E402
from phase15d_run import EST_UP, stats, world_up_athleticspose  # noqa: E402
from phase15d_segments import segment_angles  # noqa: E402
from phase15e_fps import (NATIVE_HZ, clips, every_nth,  # noqa: E402
                          paired_penalty, pool as pool_e, run_clip,
                          tick_indices)
from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import ARCH, CKPT, DATA, TEST_SUBJECTS  # noqa: E402
from phase7_inference_fix import pad_to  # noqa: E402
from video_kinematics import (coco17_to_h36m17, detect_coco17,  # noqa: E402
                              resample)

OUT = REPO / "results" / "phase15f.json"
CAM = "PG1"
RES_AP = (1920, 1088)     # FovVideo Right 1919 / Bottom 1087, all 8 cameras
RES_PG1 = (800, 600)      # measured from the files, 2026-11-02

# All 11 SPENT MoVi PG1 subjects. Diagnostic: none is held out any more.
MOVI_SUBJECTS = ("Subject_1", "Subject_10", "Subject_11", "Subject_12",
                 "Subject_13", "Subject_15", "Subject_16", "Subject_17",
                 "Subject_18", "Subject_19", "Subject_2")
QS = ("knee_from_segments", "thigh", "trunk_referenced_hip", "trunk")
LABEL = {"knee_from_segments": "knee flexion",
         "thigh": "thigh incl (gravity)",
         "trunk_referenced_hip": "hip (trunk-ref)",
         "trunk": "trunk lean"}


def build(key: str):
    """Load a checkpoint into its own architecture."""
    c = CONTRACTS[key]
    model = MotionAGFormer(**dict(ARCH[c.arch], n_frames=c.n_frames))
    blob = torch.load(CKPT / c.name, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.eval()
    return model, {"missing": len(miss), "unexpected": len(unexp)}


@torch.no_grad()
def _windows(model, x: np.ndarray, clip: int) -> np.ndarray:
    """Edge padding + 50%-overlap triangular blend, copied from
    `phase7_inference_fix.lift` so the two cannot drift. Verified identical to
    it in `main` before anything is scored."""
    t = len(x)
    stride = clip // 2
    acc = np.zeros((t, 17, 3))
    wts = np.zeros((t, 1, 1))
    starts = list(range(0, max(t - clip, 0) + 1, stride))
    if starts and starts[-1] + clip < t:
        starts.append(t - clip)
    if not starts:
        starts = [0]
    tri = np.maximum(1.0 - np.abs(np.linspace(-1.0, 1.0, clip)),
                     1e-3).reshape(clip, 1, 1)
    for s in starts:
        chunk = x[s:s + clip]
        valid = len(chunk)
        inp = torch.from_numpy(pad_to(chunk, clip, "edge")[None]).float()
        got = model(inp)[0].numpy()
        acc[s:s + valid] += got[:valid] * tri[:valid]
        wts[s:s + valid] += tri[:valid]
    return acc / np.maximum(wts, 1e-9)


def lift_contract(model, kp_px: np.ndarray, key: str, src_fps: float,
                  res: tuple[int, int]) -> tuple[np.ndarray, list[str], dict]:
    """Rate-match to the checkpoint's training fps, normalize to its contract,
    lift at its window, resample back to the input grid."""
    c = CONTRACTS[key]
    n = len(kp_px)
    n_model = max(int(round(n * (c.train_fps or src_fps) / src_fps)), 2)
    x = normalize(resample(np.asarray(kp_px, float), n_model), c, res)
    flags = check(x, c, res)
    return resample(_windows(model, x, c.n_frames), n), flags, summarize(x)


# ------------------------------------------------------------- scoring -----

def score_pair(est: np.ndarray, ref: np.ndarray, near: str,
               up) -> dict:
    out = {}
    for limb, side in (("near", near), ("far", "r" if near == "l" else "l")):
        m = angle_mask(ref, side)
        ea = segment_angles(est, side, EST_UP)
        ga = segment_angles(ref, side, up)
        cell = {q: stats(ea[q][m], ga[q][m])
                for q in ("knee_from_segments", "thigh", "trunk")}
        cell["trunk_referenced_hip"] = stats(
            sagittal_angles(est, side)["hip"][m],
            sagittal_angles(ref, side)["hip"][m])
        out[limb] = cell
    return out


def athleticspose_rows(models, keys, limit: int = 0) -> list[dict]:
    recs = clips("running", TEST_SUBJECTS, need_raw_match=False)
    if limit:
        recs = recs[:limit]
    rows = []
    for i, rec in enumerate(recs, 1):
        _a, subj, stem, rawp, gtp, _g2p, detp = rec
        gt = gt3d_mm(gtp)
        det = np.load(detp).astype(float)
        n = min(len(gt), len(det))
        if n < 34:
            continue
        gt, det = gt[:n], det[:n]
        ref_up = None
        raw = np.load(rawp).astype(float)
        if len(raw) == len(gt):
            wup = world_up_athleticspose(raw)
            _rc, tfs = align_per_frame(mocap_to_h36m(raw), gt,
                                       list(range(17)), scale=False)
            ref_up = np.stack([R @ wup for R, _t, _s in tfs])
        near = ("l" if np.nanmean(gt[:, H36M["l_hip"], 2])
                < np.nanmean(gt[:, H36M["r_hip"], 2]) else "r")
        row = {"subject": subj, "stem": stem, "n": int(n),
               "has_gravity": ref_up is not None, "ckpt": {}}
        for key in keys:
            est, flags, summ = lift_contract(models[key], det, key,
                                             NATIVE_HZ, RES_AP)
            cells = score_pair(est, gt, near,
                               ref_up if ref_up is not None else EST_UP)
            if ref_up is None:
                for limb in cells:
                    for q in ("thigh", "trunk"):
                        cells[limb].pop(q, None)
            row["ckpt"][key] = {"cells": cells, "flags": flags,
                                "summary": summ}
        rows.append(row)
        if i % 50 == 0:
            print(f"      {i}/{len(recs)} AthleticsPose clips")
    return rows


def movi_rows(models, keys, limit: int = 0) -> list[dict]:
    R_cv, t_cv, _s = camera_pose(CAM)
    mats = {p.stem.split("_v3d_")[-1]: p for p in sorted(MOCAP.glob("F_v3d_*.mat"))}
    rows = []
    subjects = MOVI_SUBJECTS[:limit] if limit else MOVI_SUBJECTS
    for subject in subjects:
        if subject not in mats:
            print(f"      {subject}: no mocap, skipped")
            continue
        video = MOVI / CAMERAS[CAM]["pattern"].format(s=subject.split("_")[-1])
        if not video.exists():
            print(f"      {subject}: no PG1 video, skipped")
            continue
        v = load_v3d(mats[subject])
        _ji, f30, f120 = jogging_window(v)
        ref_all = to_h36m(v)
        frames, fps, _vm = decode_range(video, int(f30[0]), int(f30[1]))
        mk = int(f120[0]) + MOCAP_PER_VIDEO * np.arange(len(frames))
        mk = mk[mk < ref_all.shape[0]]
        frames = frames[:len(mk)]
        ref = ref_all[mk]
        gc = to_camera(ref, R_cv, t_cv)
        near = ("l" if np.nanmean(gc[:, H36M["l_hip"], 2])
                < np.nanmean(gc[:, H36M["r_hip"], 2]) else "r")
        # DETECT ONCE, reuse for every checkpoint -- the 2D input is identical
        kp = coco17_to_h36m17(detect_coco17(frames, crop=True, model=None)[0])
        det_ok = np.abs(kp[:, :, :2]).sum(axis=(1, 2)) > 0
        row = {"subject": subject, "n": int(len(mk)),
               "n_detected": int(det_ok.sum()), "near_limb": near, "ckpt": {}}
        for key in keys:
            est, flags, summ = lift_contract(models[key], np.nan_to_num(kp),
                                             key, float(fps), RES_PG1)
            cells = score_pair(est[det_ok], ref[det_ok], near,
                               np.array([0.0, 0.0, 1.0]))
            row["ckpt"][key] = {"cells": cells, "flags": flags,
                                "summary": summ}
        rows.append(row)
        c = row["ckpt"][keys[0]]["cells"]["near"]
        print(f"      {subject:<12} knee {c['knee_from_segments']['mae']:6.3f} "
              f"hip {c['trunk_referenced_hip']['mae']:6.3f}  "
              f"({row['n_detected']}/{row['n']} detected)")
    return rows


def pool(rows, key: str, q: str, limb: str) -> dict:
    per = defaultdict(list)
    for r in rows:
        s = r["ckpt"].get(key, {}).get("cells", {}).get(limb, {}).get(q)
        if s and s.get("n") and np.isfinite(s["mae"]):
            per[r["subject"]].append(s)
    if not per:
        return {"n_subjects": 0}
    maes = {s: float(np.mean([x["mae"] for x in v])) for s, v in per.items()}
    lo, hi = subject_bootstrap_ci(list(maes.values()))
    return {"mae": float(np.mean(list(maes.values()))), "ci95": [lo, hi],
            "bias": float(np.mean([x["bias"] for v in per.values() for x in v])),
            "sd": float(np.mean([x["sd"] for v in per.values() for x in v])),
            "r": float(np.nanmean([x["r"] for v in per.values() for x in v])),
            "per_subject_mae": maes, "n_subjects": len(maes),
            "n_clips": sum(len(v) for v in per.values())}


def loss_ci(ap: dict, mv: dict) -> dict:
    """MoVi MAE - AthleticsPose MAE. The subject sets are disjoint, so this is
    an unpaired difference of two pooled means; the CI is the bootstrap of the
    difference, resampling subjects within each side."""
    if not (ap.get("n_subjects") and mv.get("n_subjects")):
        return {"loss": None}
    a = np.array(list(ap["per_subject_mae"].values()))
    m = np.array(list(mv["per_subject_mae"].values()))
    rng = np.random.default_rng(SEED)
    d = (m[rng.integers(0, len(m), size=(N_BOOT, len(m)))].mean(axis=1)
         - a[rng.integers(0, len(a), size=(N_BOOT, len(a)))].mean(axis=1))
    return {"loss": float(m.mean() - a.mean()),
            "ci95": [float(np.percentile(d, 2.5)),
                     float(np.percentile(d, 97.5))],
            "ap_mae": float(a.mean()), "movi_mae": float(m.mean()),
            "n_ap_subjects": len(a), "n_movi_subjects": len(m)}


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", type=Path, default=OUT)
    ap_.add_argument("--clips", type=int, default=0)
    ap_.add_argument("--movi", type=int, default=0,
                     help="0 = all 11 spent subjects")
    args = ap_.parse_args()

    out: dict = {
        "what": "Phase 15F: Q1 the 50 fps arm; Q2 do other checkpoints "
                "transfer better?",
        "pre_registration": "results/phase15f.md, committed alone as d198eac",
        "gates": "none",
    }

    # ---- contracts, and the window length read from the weights ----------
    print("=== contracts, window length READ FROM THE WEIGHTS ===")
    contracts = {}
    models = {}
    for key, c in CONTRACTS.items():
        model, load = build(key)
        models[key] = model
        contracts[key] = {"name": c.name, "arch": c.arch, "joints": c.joints,
                          "normalization": c.norm, "conf_value": c.conf_value,
                          "conf_label": c.conf_label, "train_fps": c.train_fps,
                          "n_frames": c.n_frames, "determined_of_5": c.determined,
                          "undetermined": list(c.undetermined),
                          "in_primary": c.include_in_primary,
                          "load": load, "note": c.note}
        print(f"  {key:16s} {c.norm:14s} conf {c.conf_value}  "
              f"{c.train_fps} Hz  {c.n_frames} frames  "
              f"{c.determined}/5  primary={c.include_in_primary}  "
              f"load {load['missing']}/{load['unexpected']}")
    out["contracts"] = contracts

    # the windowing must match the frozen lift() exactly
    from phase7_inference_fix import lift as frozen_lift
    rng = np.random.default_rng(0)
    probe = rng.normal(500, 100, size=(130, 17, 3))
    probe[:, :, 2] = 0.8676
    a = frozen_lift(models["b-ath-det-coco"], probe, "edge", "overlap")
    from athleticspose.utils import normalize_kpts
    co = np.concatenate([probe[:, :, :2], np.zeros_like(probe[:, :, :1])], -1)
    cn, _ = normalize_kpts(co)
    b = _windows(models["b-ath-det-coco"],
                 np.concatenate([cn[:, :, :2], probe[:, :, 2:3]], -1), 81)
    same = float(np.abs(a - b).max())
    out["windowing_matches_frozen_lift_max_abs_diff"] = same
    print(f"\n  _windows vs frozen lift(): max |diff| {same:.3e}")

    keys = list(CONTRACTS)
    primary = [k for k in keys if CONTRACTS[k].include_in_primary]

    # ---- Q1 ---------------------------------------------------------------
    print("\n=== Q1: the 50.0 fps arm (every 2nd frame). NOT 60. ===")
    recs = clips("running", TEST_SUBJECTS, need_raw_match=False)
    if args.clips:
        recs = recs[:args.clips]
    rows50 = []
    for i, rec in enumerate(recs, 1):
        r = run_clip(models["b-ath-det-coco"], rec, ("coco_det",),
                     lambda n: every_nth(n, 2))
        if r:
            rows50.append(r)
        if i % 100 == 0:
            print(f"      {i}/{len(recs)}")
    pen50 = {q: {limb: paired_penalty(rows50, "coco_det", q, limb)
                 for limb in ("near", "far")} for q in QS}
    nat50 = {q: {limb: pool_e(rows50, "coco_det|native", q, limb)
                 for limb in ("near", "far")} for q in QS}
    f50 = {q: {limb: pool_e(rows50, "coco_det|fps30", q, limb)
               for limb in ("near", "far")} for q in QS}
    out["q1_50fps"] = {"rule": "every 2nd frame of 100.0 Hz = exactly 50.0 fps",
                       "not_60": "100/60 is not an integer; this is 50, not 60",
                       "n_clips": len(rows50),
                       "native": nat50, "fps50": f50, "penalty": pen50}

    # ---- Q2 ---------------------------------------------------------------
    print("\n=== Q2: AthleticsPose (in-domain for the deployed ckpt) ===")
    ap_rows = athleticspose_rows(models, keys, args.clips)
    print(f"\n=== Q2: MoVi PG1, {len(MOVI_SUBJECTS)} SPENT subjects ===")
    mv_rows = movi_rows(models, keys, args.movi)

    pooled = {k: {"athleticspose": {q: {limb: pool(ap_rows, k, q, limb)
                                        for limb in ("near", "far")}
                                    for q in QS},
                  "movi": {q: {limb: pool(mv_rows, k, q, limb)
                               for limb in ("near", "far")} for q in QS}}
              for k in keys}
    losses = {k: {q: {limb: loss_ci(pooled[k]["athleticspose"][q][limb],
                                    pooled[k]["movi"][q][limb])
                      for limb in ("near", "far")} for q in QS} for k in keys}
    out["q2"] = {"pooled": pooled, "loss": losses,
                 "movi_subjects": [r["subject"] for r in mv_rows],
                 "movi_is_diagnostic": ("all 11 subjects are spent; this "
                                        "establishes no held-out claim"),
                 "n_athleticspose_clips": len(ap_rows),
                 "n_movi_subjects": len(mv_rows),
                 "primary_checkpoints": primary}

    # contract flags
    flags = {}
    for k in keys:
        fa = sorted({f for r in ap_rows for f in r["ckpt"][k]["flags"]})
        fm = sorted({f for r in mv_rows for f in r["ckpt"][k]["flags"]})
        flags[k] = {"athleticspose": fa, "movi": fm,
                    "n_ap_clips_flagged":
                        sum(1 for r in ap_rows if r["ckpt"][k]["flags"]),
                    "n_movi_subjects_flagged":
                        sum(1 for r in mv_rows if r["ckpt"][k]["flags"])}
    out["input_contract_flags"] = flags

    # ---- tables -----------------------------------------------------------
    print("\n=== Q1: the fps penalty curve (COCO, paired per subject) ===")
    print(f"  {'quantity':<22}{'limb':<6}{'50.0':>9}{'95% CI':>20}")
    for q in QS:
        for limb in ("near", "far"):
            p = pen50[q][limb]
            if not p.get("n_subjects"):
                continue
            print(f"  {LABEL[q]:<22}{limb:<6}{p['penalty']:>+9.3f}"
                  f"   [{p['ci95'][0]:>+6.3f}, {p['ci95'][1]:>+6.3f}]")

    print("\n=== Q2: loss = MoVi MAE - AthleticsPose MAE ===")
    print(f"  {'checkpoint':<16}{'quantity':<22}{'limb':<6}{'AP':>8}"
          f"{'MoVi':>8}{'loss':>9}{'95% CI':>20}")
    for k in keys:
        for q in QS:
            for limb in ("near", "far"):
                L = losses[k][q][limb]
                if L.get("loss") is None:
                    continue
                print(f"  {k:<16}{LABEL[q]:<22}{limb:<6}{L['ap_mae']:>8.3f}"
                      f"{L['movi_mae']:>8.3f}{L['loss']:>+9.3f}"
                      f"   [{L['ci95'][0]:>+6.3f}, {L['ci95'][1]:>+6.3f}]")

    print("\n=== input-contract flags ===")
    for k in keys:
        f = flags[k]
        print(f"  {k}: {f['n_ap_clips_flagged']} AP clips, "
              f"{f['n_movi_subjects_flagged']} MoVi subjects flagged")
        for s in f["athleticspose"][:3] + f["movi"][:3]:
            print(f"      {s}")

    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    try:
        shown = args.out.resolve().relative_to(REPO)
    except ValueError:
        shown = args.out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
