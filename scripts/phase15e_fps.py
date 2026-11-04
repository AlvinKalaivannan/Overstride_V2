"""Phase 15E: how much error comes from 30 fps input alone, no change of lab?

Pre-registered in `results/phase15e.md`, committed alone as `45dc30a`. Tune set
only -- AthleticsPose. **No MoVi file is opened by this script.**

AthleticsPose is 100.0 Hz (shipped Qualisys QCA metadata) and that is exactly
`MODEL_FPS`, so the native arm through `lift_2d` reduces to the committed
baseline's `lift(...)` call bit-for-bit: `resample(x, len(x))` is identity. The
30 fps arm decimates the INPUT to a true 30.0 fps grid, then the same path
resamples it back up to the same model-frame count. Only temporal information
differs; the window structure, the edge padding and the scored frames are
identical.

No gate. Run: .venv/Scripts/python.exe scripts/phase15e_fps.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data" / "athleticspose" / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from athleticspose.utils import mocap_to_h36m  # noqa: E402
from phase12_scan import N_BOOT, SEED, subject_bootstrap_ci  # noqa: E402
from phase15_checks import angle_mask  # noqa: E402
from phase15b_conventions import lift_2d  # noqa: E402
from phase15c_thorax import align_per_frame, gt3d_mm  # noqa: E402
from phase15d_run import EST_UP, stats, world_up_athleticspose  # noqa: E402
from phase15d_segments import segment_angles  # noqa: E402
from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import DATA, TEST_SUBJECTS, load_model  # noqa: E402
from phase7_inference_fix import CKPT_FOR  # noqa: E402
from video_kinematics import MODEL_FPS, TRAIN_CONF_MEAN  # noqa: E402

OUT = REPO / "results" / "phase15e.json"

NATIVE_HZ = 100.0          # shipped QCA Timebase/Frequency, all 14 sessions
TARGET_FPS = 30.0          # MoVi PG1's rate
MIN_NATIVE = 34            # pre-registered: >= 10 real frames at 30 fps

# Pre-registered, fixed in 45dc30a. Never recomputed here.
SEALED = {
    "knee_from_segments": {"near": 8.410, "far": 8.685},
    "thigh": {"near": 5.401, "far": 7.372},
    "trunk_referenced_hip": {"near": 7.746, "far": 9.257},
    "trunk": {"near": 5.158, "far": 5.099},
}
# Quantities needing the gravity vertical, hence the length-matched raw/gt subset
GRAVITY_Q = ("thigh", "trunk")
# Quantities invariant to the vertical (knee is a within-frame difference) or
# not referenced to it at all (the hip is referenced to the trunk)
FREE_Q = ("knee_from_segments", "trunk_referenced_hip")
LABEL = {"knee_from_segments": "knee flexion",
         "thigh": "thigh inclination (gravity)",
         "trunk_referenced_hip": "hip (trunk-referenced)",
         "trunk": "trunk lean"}


def tick_indices(n: int, native: float = NATIVE_HZ,
                 target: float = TARGET_FPS) -> np.ndarray:
    """Nearest-tick decimation to a true `target` fps grid. Real frames only."""
    k = np.arange(int(np.floor((n - 1) * target / native)) + 1)
    return np.unique(np.rint(k * native / target).astype(int))


def every_nth(n: int, step: int = 3) -> np.ndarray:
    """Strict integer decimation: the 33.3 fps sensitivity arm."""
    return np.arange(0, n, step)


def clips(action: str, subjects, need_raw_match: bool):
    """Clips with all four inputs. `need_raw_match` adds phase 15C's
    length-matching, which the gravity vertical requires."""
    out = []
    for subj in subjects:
        d = DATA / "det_markers2d_by_cam_coco" / action / subj
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.npy")):
            gt = DATA / "gt_markers3d_by_cam" / action / subj / f"{f.stem}.npz"
            g2 = DATA / "gt_markers2d_by_cam" / action / subj / f"{f.stem}.npy"
            rawp = (DATA / "raw_markers_in_world" / action / subj
                    / f"{f.stem.rsplit('_', 1)[0]}.npy")
            if not (gt.exists() and g2.exists() and rawp.exists()):
                continue
            with np.load(gt) as z:
                n_gt = len(z["p2mm"])
            n_raw = len(np.load(rawp, mmap_mode="r"))
            if need_raw_match and n_raw != n_gt:
                continue
            if min(n_gt, n_raw, len(np.load(g2, mmap_mode="r")),
                   len(np.load(f, mmap_mode="r"))) < MIN_NATIVE:
                continue
            out.append((action, subj, f.stem, rawp, gt, g2, f))
    return out


def inputs_for(stem_paths) -> dict[str, np.ndarray]:
    """The three pre-registered input arms, at native rate."""
    _a, _s, _st, _rawp, _gt, g2p, detp = stem_paths
    det = np.load(detp).astype(float)
    g2 = np.load(g2p).astype(float)
    g2_train = g2.copy()
    # GT 2D ships a CONSTANT 1.0 in channel 2, not the training confidence
    # scale. Feeding that raw is the class of defect the sealed test found at
    # 14x; `project` already overwrites it for MoVi's diagnostic arm.
    g2_train[:, :, 2] = TRAIN_CONF_MEAN
    return {"coco_det": det, "gt2d_trainconf": g2_train, "gt2d_raw1.0": g2}


def run_clip(model, rec, arms, decim) -> dict | None:
    action, subj, stem, rawp, gtp, _g2p, _detp = rec
    gt = gt3d_mm(gtp)
    ins = inputs_for(rec)
    n = min([len(gt)] + [len(v) for v in ins.values()])
    if n < MIN_NATIVE:
        return None
    gt = gt[:n]
    ins = {k: v[:n] for k, v in ins.items() if k in arms}

    idx = decim(n)
    idx = idx[idx < n]
    if len(idx) < 10:
        return None

    # --- the gravity vertical, exactly as 15D's G1 built it ----------------
    ref_up = None
    raw = np.load(rawp).astype(float)
    if len(raw) == len(gt):
        wup = world_up_athleticspose(raw)
        rebuilt = mocap_to_h36m(raw)
        _rc, tfs = align_per_frame(rebuilt, gt, list(range(17)), scale=False)
        ref_up = np.stack([R @ wup for R, _t, _s in tfs])

    # GT 3D is already in camera coordinates: axis 2 is depth.
    near = ("l" if np.nanmean(gt[:, H36M["l_hip"], 2])
            < np.nanmean(gt[:, H36M["r_hip"], 2]) else "r")

    row = {"action": action, "subject": subj, "stem": stem,
           "n_native": int(n), "n_decimated": int(len(idx)),
           "has_gravity": ref_up is not None, "near_limb": near, "arms": {}}

    for arm, kp in ins.items():
        est_native_full = lift_2d(model, kp, NATIVE_HZ)
        est_native = est_native_full[idx]
        est_30 = lift_2d(model, kp[idx], TARGET_FPS)
        for rate, est in (("native", est_native), ("fps30", est_30)):
            g = gt[idx]
            up = ref_up[idx] if ref_up is not None else EST_UP
            cell = {}
            for limb, side in (("near", near),
                               ("far", "r" if near == "l" else "l")):
                m = angle_mask(g, side)
                ea = segment_angles(est, side, EST_UP)
                ga = segment_angles(g, side, up)
                c = {}
                for q in FREE_Q + GRAVITY_Q:
                    if q == "trunk_referenced_hip":
                        c[q] = stats(sagittal_angles(est, side)["hip"][m],
                                     sagittal_angles(g, side)["hip"][m])
                        continue
                    if q in GRAVITY_Q and ref_up is None:
                        continue
                    c[q] = stats(ea[q][m], ga[q][m])
                # the knee identity, as a running check on this new path
                k1 = ga["knee_from_segments"][m]
                k2 = sagittal_angles(g, side)["knee"][m]
                c["_knee_identity_max_dev"] = float(np.nanmax(
                    np.abs(((k1 - k2) + 180) % 360 - 180)))
                cell[limb] = c
            row["arms"][f"{arm}|{rate}"] = cell
        # the full-rate figure, scored on every native frame, as the check that
        # restricting to the decimated subset does not itself move the number
        row["arms"][f"{arm}|native_allframes"] = {}
        for limb, side in (("near", near), ("far", "r" if near == "l" else "l")):
            m = angle_mask(gt, side)
            c = {"trunk_referenced_hip": stats(
                sagittal_angles(est_native_full, side)["hip"][m],
                sagittal_angles(gt, side)["hip"][m])}
            ea = segment_angles(est_native_full, side, EST_UP)
            upf = ref_up if ref_up is not None else EST_UP
            ga = segment_angles(gt, side, upf)
            c["knee_from_segments"] = stats(ea["knee_from_segments"][m],
                                            ga["knee_from_segments"][m])
            if ref_up is not None:
                for q in GRAVITY_Q:
                    c[q] = stats(ea[q][m], ga[q][m])
            row["arms"][f"{arm}|native_allframes"][limb] = c
    return row


def pool(rows, key: str, q: str, limb: str) -> dict:
    per = defaultdict(list)
    for r in rows:
        cell = r["arms"].get(key, {}).get(limb, {})
        s = cell.get(q)
        if s and s.get("n") and np.isfinite(s["mae"]):
            per[r["subject"]].append(s)
    if not per:
        return {"n_subjects": 0}
    maes = {s: float(np.mean([x["mae"] for x in v])) for s, v in per.items()}
    bias = {s: float(np.mean([x["bias"] for x in v])) for s, v in per.items()}
    sd = {s: float(np.mean([x["sd"] for x in v])) for s, v in per.items()}
    rr = {s: float(np.nanmean([x["r"] for x in v])) for s, v in per.items()}
    lo, hi = subject_bootstrap_ci(list(maes.values()))
    return {"mae": float(np.mean(list(maes.values()))), "ci95": [lo, hi],
            "bias": float(np.mean(list(bias.values()))),
            "sd": float(np.mean(list(sd.values()))),
            "r": float(np.mean(list(rr.values()))),
            "per_subject_mae": maes, "n_subjects": len(maes),
            "n_clips": sum(len(v) for v in per.values()),
            "n_boot": N_BOOT, "seed": SEED}


def paired_penalty(rows, arm: str, q: str, limb: str) -> dict:
    """30 fps minus native, PAIRED per subject, subject-bootstrap CI."""
    per = defaultdict(lambda: [[], []])
    for r in rows:
        a = r["arms"].get(f"{arm}|native", {}).get(limb, {}).get(q)
        b = r["arms"].get(f"{arm}|fps30", {}).get(limb, {}).get(q)
        if not (a and b and a.get("n") and b.get("n")):
            continue
        if not (np.isfinite(a["mae"]) and np.isfinite(b["mae"])):
            continue
        per[r["subject"]][0].append(a["mae"])
        per[r["subject"]][1].append(b["mae"])
    if not per:
        return {"n_subjects": 0}
    d = {s: float(np.mean(v[1]) - np.mean(v[0])) for s, v in per.items()}
    lo, hi = subject_bootstrap_ci(list(d.values()))
    return {"penalty": float(np.mean(list(d.values()))), "ci95": [lo, hi],
            "per_subject": d, "n_subjects": len(d),
            "resolved": bool((lo > 0) or (hi < 0))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=0, help="0 = every clip")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--per-clip-out", type=Path, default=None,
                    help="optional path OUTSIDE the repository for the "
                         "per-clip rows; they are 8 MB and nothing "
                         "reads them, so they are not committed")
    args = ap.parse_args()

    model = load_model(CKPT_FOR["coco"], "base")
    if model is None:
        raise SystemExit("checkpoint missing")

    assert MODEL_FPS == NATIVE_HZ, (MODEL_FPS, NATIVE_HZ)
    arms = ("coco_det", "gt2d_trainconf", "gt2d_raw1.0")

    recs = clips("running", TEST_SUBJECTS, need_raw_match=False)
    if args.clips:
        recs = recs[:args.clips]
    print(f"=== PHASE 15E: the 30 fps control, AthleticsPose running, "
          f"{list(TEST_SUBJECTS)} ===")
    print(f"    native {NATIVE_HZ:.0f} Hz == MODEL_FPS, target {TARGET_FPS:.0f} "
          f"fps, {len(recs)} clips\n")

    rows = []
    for i, rec in enumerate(recs, 1):
        r = run_clip(model, rec, arms, tick_indices)
        if r:
            rows.append(r)
        if i % 25 == 0:
            print(f"    {i}/{len(recs)} clips")
    n_grav = sum(1 for r in rows if r["has_gravity"])
    print(f"\n  {len(rows)} clips scored; {n_grav} have the gravity vertical "
          f"(length-matched raw/gt)")

    # ---- sensitivity: strict every-third-frame (33.3 fps), COCO only -------
    print("\n  sensitivity arm: strict every-3rd frame (33.3 fps), COCO only")
    rows33 = []
    for i, rec in enumerate(recs, 1):
        r = run_clip(model, rec, ("coco_det",), every_nth)
        if r:
            rows33.append(r)
        if i % 50 == 0:
            print(f"    {i}/{len(recs)}")

    # ---- secondary: sprint, on ITS OWN TRAIN SUBJECTS, penalty only -------
    sprint_subs = sorted({p.name for p in
                          (DATA / "det_markers2d_by_cam_coco" / "sprint").iterdir()
                          if p.is_dir()}) if (
        DATA / "det_markers2d_by_cam_coco" / "sprint").is_dir() else []
    sprint_recs = clips("sprint", sprint_subs, need_raw_match=False)
    if args.clips:
        sprint_recs = sprint_recs[:args.clips]
    print(f"\n  SECONDARY: sprint, {len(sprint_subs)} TRAIN subjects, "
          f"{len(sprint_recs)} clips, COCO only, penalty only")
    rows_sp = []
    for i, rec in enumerate(sprint_recs, 1):
        r = run_clip(model, rec, ("coco_det",), tick_indices)
        if r:
            rows_sp.append(r)
        if i % 50 == 0:
            print(f"    {i}/{len(sprint_recs)}")

    # ---- tables -----------------------------------------------------------
    QS = FREE_Q + GRAVITY_Q
    pooled = {k: {q: {limb: pool(rows, k, q, limb) for limb in ("near", "far")}
                  for q in QS}
              for k in [f"{a}|{r}" for a in arms
                        for r in ("native", "fps30", "native_allframes")]}
    pen = {a: {q: {limb: paired_penalty(rows, a, q, limb)
                   for limb in ("near", "far")} for q in QS} for a in arms}
    pen33 = {q: {limb: paired_penalty(rows33, "coco_det", q, limb)
                 for limb in ("near", "far")} for q in QS}
    pen_sp = {q: {limb: paired_penalty(rows_sp, "coco_det", q, limb)
                  for limb in ("near", "far")} for q in QS}

    print("\n=== PENALTY: 30 fps minus native, paired per subject ===")
    print(f"  {'input':<16}{'quantity':<30}{'limb':<6}{'native':>8}"
          f"{'30 fps':>9}{'penalty':>9}{'95% CI':>20}{'res?':>6}")
    for a in arms:
        for q in QS:
            for limb in ("near", "far"):
                nat = pooled[f"{a}|native"][q][limb]
                f30 = pooled[f"{a}|fps30"][q][limb]
                p = pen[a][q][limb]
                if not nat.get("n_subjects") or not p.get("n_subjects"):
                    continue
                print(f"  {a:<16}{LABEL[q]:<30}{limb:<6}{nat['mae']:>8.3f}"
                      f"{f30['mae']:>9.3f}{p['penalty']:>+9.3f}"
                      f"   [{p['ci95'][0]:>+6.3f}, {p['ci95'][1]:>+6.3f}]"
                      f"{('YES' if p['resolved'] else 'no'):>6}")

    print("\n=== SHARE OF THE SEALED LOSS, if the penalty transferred "
          "unchanged ===")
    print(f"  {'quantity':<30}{'limb':<6}{'AP native':>10}{'sealed':>8}"
          f"{'loss':>8}{'penalty':>9}{'share':>9}")
    share = {}
    for q in QS:
        share[q] = {}
        for limb in ("near", "far"):
            nat = pooled["coco_det|native"][q][limb]
            p = pen["coco_det"][q][limb]
            if not nat.get("n_subjects") or not p.get("n_subjects"):
                continue
            loss = SEALED[q][limb] - nat["mae"]
            if loss <= 0.2:          # pre-registered: undefined near/below zero
                share[q][limb] = {"ap_native": nat["mae"],
                                  "sealed": SEALED[q][limb], "loss": loss,
                                  "penalty": p["penalty"], "share": None,
                                  "why": ("total loss is at or below zero, so a "
                                          "share of it is undefined")}
                txt = "undefined"
            else:
                sh = p["penalty"] / loss
                share[q][limb] = {"ap_native": nat["mae"],
                                  "sealed": SEALED[q][limb], "loss": loss,
                                  "penalty": p["penalty"], "share": sh}
                txt = f"{100 * sh:+.1f}%"
            print(f"  {LABEL[q]:<30}{limb:<6}{nat['mae']:>10.3f}"
                  f"{SEALED[q][limb]:>8.3f}{loss:>8.3f}"
                  f"{p['penalty']:>+9.3f}{txt:>9}")

    ident = max((c.get("_knee_identity_max_dev", 0.0)
                 for r in rows for a in r["arms"].values()
                 for c in a.values() if isinstance(c, dict)), default=0.0)
    print(f"\n  knee identity on this path: max |deviation| {ident:.2e} deg")

    out = {
        "what": ("Phase 15E: the 30 fps penalty, isolated on AthleticsPose. "
                 "Tune set only; no MoVi file is opened."),
        "pre_registration": "results/phase15e.md, committed alone as 45dc30a",
        "gates": "none",
        "native_hz": NATIVE_HZ,
        "native_hz_source": ("shipped Qualisys QCA metadata, "
                             "camera_params/*.json Timebase/Frequency, all 14 "
                             "sessions; supersedes phase 7's 113-136 fps "
                             "cadence inference"),
        "model_fps": MODEL_FPS,
        "native_arm_is_identity": ("MODEL_FPS == NATIVE_HZ and "
                                   "resample(x, len(x)) is bit-identical, so "
                                   "the native arm reduces exactly to the "
                                   "committed baseline's lift() call"),
        "target_fps": TARGET_FPS,
        "decimation": {"primary": "nearest 30.0 fps tick, real frames only",
                       "sensitivity": "strict every-3rd frame (33.3 fps)"},
        "subjects": list(TEST_SUBJECTS), "action": "running",
        "sprint_excluded_from_primary": (
            "sprint and running have disjoint subject sets; TEST_SUBJECTS "
            "appear in running only, so sprint would be a train-subject score"),
        "n_clips_scored": len(rows),
        "n_clips_with_gravity_vertical": n_grav,
        "bootstrap": {"n_boot": N_BOOT, "seed": SEED,
                      "resampled": "subjects, not clips"},
        "sealed_comparison_fixed_in_prereg": SEALED,
        "pooled": pooled,
        "penalty_paired": pen,
        "penalty_sensitivity_33fps_coco": pen33,
        "secondary_sprint_train_subjects": {
            "subjects": sprint_subs, "n_clips": len(rows_sp),
            "note": ("TRAIN subjects for this checkpoint. Penalty only -- the "
                     "absolute MAE is not a baseline and is not quoted as one."),
            "penalty_paired": pen_sp},
        "share_of_sealed_loss": share,
        "share_caveat": ("if-it-transferred arithmetic, not a causal "
                         "decomposition: the penalty is measured on "
                         "AthleticsPose running at 100 Hz and divided into a "
                         "difference between two datasets"),
        "knee_identity_max_abs_deviation_deg": ident,
    }
    if args.per_clip_out:
        pc = Path(args.per_clip_out).resolve()
        if REPO in pc.parents or pc.parent == REPO:
            print("  REFUSED: --per-clip-out is inside the repository")
            return 3
        pc.parent.mkdir(parents=True, exist_ok=True)
        pc.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"  per-clip detail -> {pc}  (outside the repo)")
        out["per_clip"] = f"{len(rows)} rows written to {pc.name}"
    else:
        out["per_clip"] = (f"{len(rows)} clip rows, not committed: "
                           f"8 MB of detail nothing reads. Reproduce "
                           f"with --per-clip-out PATH.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # relative_to raises for any path outside the repo -- the same trap
    # that bit phase15_checks and was fixed in 2b5b74d.
    try:
        shown = args.out.resolve().relative_to(REPO)
    except ValueError:
        shown = args.out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
