"""Phase 15C -- what is this lifter's "thorax", anatomically? AthleticsPose only.

Pre-registered in `results/phase15c.md`, committed alone in 49428c0 before any
of this ran. The decision rule and the candidate mapping are fixed there.

PART 2 lives here (AthleticsPose). PART 3 (applying the winner to MoVi once) is
`scripts/phase15c_movi.py`.

Nothing is chosen on MoVi. The angle code and the lifter are untouched.

Run: .venv/Scripts/python.exe scripts/phase15c_thorax.py [--clips N]
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

from athleticspose.statics.joints import (  # noqa: E402
    joint_index_h36m_to_mocap, mocap_joints_name_to_index as MK)
from athleticspose.utils import mocap_to_h36m  # noqa: E402
from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import DATA, TEST_SUBJECTS, load_model  # noqa: E402
from phase7_inference_fix import CKPT_FOR, lift  # noqa: E402

OUT = REPO / "results" / "phase15c.json"
ACTIONS = ("running", "sprint")

# Candidates, fixed in the pre-registration. Values are AthleticsPose marker
# names; each is averaged if more than one.
CANDIDATES = {
    "CLAV": ("CLAV",),
    "C7": ("C7",),
    "STRN": ("Xiphoid",),                   # AP has no STRN; see 15C part 1
    "mid_CLAV_C7": ("CLAV", "C7"),
    "mean_Xiphoid_T10": ("Xiphoid", "T10"),  # control: the recipe's SPINE joint
}
RECIPE_CHOICE = "CLAV"                       # tie-break, from the recipe
TIE_DEG = 1.0

LOWER = ("pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle")


def similarity(src: np.ndarray, dst: np.ndarray, scale: bool):
    """Rigid (or similarity) transform taking `src` onto `dst`. Umeyama.

    src, dst: (n, 3) corresponding points. Returns (R, t, s) with
    dst ~= s * src @ R.T + t.
    """
    ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    src, dst = src[ok], dst[ok]
    if len(src) < 3:
        return np.eye(3), np.zeros(3), 1.0
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sv, Vt = np.linalg.svd(D.T @ S / len(S))
    sign = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ sign @ Vt
    s = (sv @ np.diag(sign)).sum() / (S ** 2).sum() * len(S) if scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def apply_tf(x: np.ndarray, R, t, s) -> np.ndarray:
    return s * x @ R.T + t


def align_per_frame(src: np.ndarray, dst: np.ndarray, idx, scale: bool):
    """Per-frame transform taking `src` onto `dst`, fitted on joints `idx`.

    PER FRAME because `p2mm` varies per frame: dividing all three axes by it is
    a perspective normalisation, not a unit conversion, so no single transform
    fits a whole sequence. Fitted on `idx` only, so a joint under test never
    helps place itself.
    """
    out = np.full_like(src, np.nan)
    tfs = []
    for f in range(len(src)):
        R, t, sc = similarity(src[f, idx, :], dst[f, idx, :], scale=scale)
        out[f] = apply_tf(src[f], R, t, sc)
        tfs.append((R, t, sc))
    return out, tfs


def gt3d_mm(path: Path) -> np.ndarray:
    """gt_markers3d_by_cam is in PIXELS with a per-frame p2mm the released
    evaluator DIVIDES by (CLAUDE.md, results/phase03b.md). Return mm."""
    with np.load(path) as z:
        m = np.asarray(z["markers_h36m"], float)
        p = np.asarray(z["p2mm"], float)
    return m / p[:, None, None]


def clip_list(limit: int | None):
    out = []
    for action in ACTIONS:
        for subj in TEST_SUBJECTS:
            d = DATA / "det_markers2d_by_cam_coco" / action / subj
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.npy")):
                gt = DATA / "gt_markers3d_by_cam" / action / subj / f"{f.stem}.npz"
                raw_stem = f.stem.rsplit("_", 1)[0]
                raw = DATA / "raw_markers_in_world" / action / subj / f"{raw_stem}.npy"
                if not (gt.exists() and raw.exists()):
                    continue
                # raw_markers_in_world is per CLIP; gt_markers3d_by_cam is per
                # CAMERA and differently trimmed, and the dataset records no
                # frame correspondence. Only length-matched pairs are used, so
                # frame i means frame i. 231 of 592 qualify. This is data
                # plumbing, not a measurement-driven selection.
                with np.load(gt) as z:
                    n_gt = len(z["p2mm"])
                if len(np.load(raw, mmap_mode="r")) != n_gt:
                    continue
                out.append((action, subj, f.stem, raw))
    out.sort(key=lambda r: r[2])
    return out[:limit] if limit else out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=0, help="0 = every clip")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    clips = clip_list(args.clips or None)
    print(f"=== phase 15C: {len(clips)} AthleticsPose clips, "
          f"subjects {TEST_SUBJECTS}, actions {list(ACTIONS)} ===")
    model = load_model(CKPT_FOR["coco"], "base")
    if model is None:
        raise SystemExit("checkpoint missing")

    inv = {i: n for n, i in H36M.items()}
    recipe_resid = defaultdict(list)          # step 1: per joint, mm
    offsets = {c: defaultdict(list) for c in CANDIDATES}   # step 2: deg
    dists = {c: defaultdict(list) for c in CANDIDATES}     # step 2: mm
    hip_err = {c: defaultdict(lambda: {"mae": [], "bias": []})
               for c in CANDIDATES}                        # step 4
    n_frames = 0

    for i, (action, subj, stem, rawp) in enumerate(clips):
        raw = np.load(rawp).astype(float)                  # (T, 84, 3) world mm
        gt = gt3d_mm(DATA / "gt_markers3d_by_cam" / action / subj / f"{stem}.npz")
        det = np.load(DATA / "det_markers2d_by_cam_coco" / action / subj
                      / f"{stem}.npy").astype(float)
        n = min(len(raw), len(gt), len(det))
        if n < 30:
            continue
        raw, gt, det = raw[:n], gt[:n], det[:n]

        # ---- step 1: the authors' own recipe, rebuilt and compared ---------
        rebuilt = mocap_to_h36m(raw)                       # world mm
        # PER-FRAME rigid alignment on all 17 joints: a shape comparison of the
        # rebuilt recipe against the released array.
        rebuilt_cam, _tfs = align_per_frame(rebuilt, gt, list(range(17)),
                                            scale=False)
        for j in range(17):
            d = np.linalg.norm(rebuilt_cam[:, j] - gt[:, j], axis=1)
            d = d[np.isfinite(d)]
            if d.size:
                recipe_resid[inv[j]].append(float(d.mean()))

        # Candidate markers into the released frame, by a per-frame transform
        # fitted on the LOWER BODY ONLY -- so the trunk never helps place the
        # trunk candidates it is being compared against.
        li = [H36M[k] for k in LOWER]
        _rc, tfs_low = align_per_frame(rebuilt, gt, li, scale=False)
        cand_cam = {}
        for name, markers in CANDIDATES.items():
            idx = [MK[m] for m in markers]
            pts = raw[:, idx, :].mean(axis=1)
            cand_cam[name] = np.stack(
                [apply_tf(pts[f], *tfs_low[f]) for f in range(len(pts))])

        # ---- step 2: the lifter's own trunk axis --------------------------
        est = lift(model, det, "edge", "overlap")
        # similarity, not rigid: `lift` returns a scale-free root-relative
        # skeleton, so scale has to be solved for. Lower body only again.
        est_cam, _te = align_per_frame(est, gt, li, scale=True)

        ml = gt[:, H36M["r_hip"]] - gt[:, H36M["l_hip"]]
        ml /= np.maximum(np.linalg.norm(ml, axis=1, keepdims=True), 1e-9)
        trunk_est = est_cam[:, H36M["thorax"]] - est_cam[:, H36M["pelvis"]]

        def sag_angle_between(a, b, normal):
            """Signed angle from a to b inside the plane perpendicular to
            `normal`. Needs no 'up' -- which matters, because the usual up is
            the quantity under test."""
            ap_ = a - normal * np.sum(a * normal, axis=1, keepdims=True)
            bp = b - normal * np.sum(b * normal, axis=1, keepdims=True)
            cross = np.cross(ap_, bp)
            sin = np.sum(cross * normal, axis=1)
            cos = np.sum(ap_ * bp, axis=1)
            return np.degrees(np.arctan2(sin, cos))

        hi = [H36M[k] for k in ("pelvis", "l_hip", "r_hip")]
        eh, _th = align_per_frame(est, gt, hi, scale=True)
        for name in CANDIDATES:
            trunk_c = cand_cam[name] - gt[:, H36M["pelvis"]]
            ang = sag_angle_between(trunk_est, trunk_c, ml)
            ang = ang[np.isfinite(ang)]
            if ang.size:
                offsets[name][subj].extend(ang.tolist())
            # 3D distance after aligning on pelvis and hips only, per frame
            dd = np.linalg.norm(eh[:, H36M["thorax"]] - cand_cam[name], axis=1)
            dd = dd[np.isfinite(dd)]
            if dd.size:
                dists[name][subj].extend(dd.tolist())

            # ---- step 4: hip error if the GT used this thorax -------------
            gt_var = gt.copy()
            gt_var[:, H36M["thorax"]] = cand_cam[name]
            for side in ("l", "r"):
                ga = sagittal_angles(gt_var, side)["hip"]
                ea = sagittal_angles(est, side)["hip"]
                ok = np.isfinite(ga) & np.isfinite(ea)
                if ok.sum() > 5:
                    d = (ea - ga)[ok]
                    hip_err[name][subj]["mae"].append(float(np.abs(d).mean()))
                    hip_err[name][subj]["bias"].append(float(d.mean()))
        n_frames += n
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(clips)} clips, {n_frames} frames")

    # ---------------------------------------------------------------- report
    print(f"\n=== STEP 1: recipe rebuilt from raw_markers_in_world vs "
          f"released gt_markers3d_by_cam, per joint (mm) ===")
    step1 = {}
    for j, v in sorted(recipe_resid.items(), key=lambda kv: -np.mean(kv[1])):
        step1[j] = float(np.mean(v))
        print(f"  {j:<12}{np.mean(v):>9.2f}")

    print(f"\n=== STEP 2: sagittal offset, lifter trunk axis vs "
          f"pelvis->candidate (deg) ===")
    print(f"  {'candidate':<20}{'mean':>9}{'|mean|':>9}{'sd':>8}"
          + "".join(f"{s:>9}" for s in TEST_SUBJECTS))
    step2 = {}
    for name in CANDIDATES:
        allv = [x for s in offsets[name] for x in offsets[name][s]]
        per = {s: float(np.mean(offsets[name][s])) for s in offsets[name]}
        m = float(np.mean(allv)) if allv else float("nan")
        step2[name] = {
            "mean_deg": m, "abs_mean_deg": abs(m),
            "sd_deg": float(np.std(allv, ddof=1)) if len(allv) > 1 else float("nan"),
            "per_subject_mean_deg": per,
            "n_frames": len(allv),
            "thorax_to_candidate_mm": float(np.mean(
                [x for s in dists[name] for x in dists[name][s]]))}
        print(f"  {name:<20}{m:>9.3f}{abs(m):>9.3f}{step2[name]['sd_deg']:>8.3f}"
              + "".join(f"{per.get(s, float('nan')):>9.3f}" for s in TEST_SUBJECTS))

    print(f"\n  3D distance, lifter thorax -> candidate, aligned on pelvis+hips (mm)")
    for name in CANDIDATES:
        print(f"    {name:<20}{step2[name]['thorax_to_candidate_mm']:>9.1f}")

    # ---- step 3: the rule, applied mechanically ------------------------
    ranked = sorted(CANDIDATES, key=lambda c: step2[c]["abs_mean_deg"])
    best, second = ranked[0], ranked[1]
    gap = step2[second]["abs_mean_deg"] - step2[best]["abs_mean_deg"]
    tie = gap < TIE_DEG
    chosen = best
    reason = (f"smallest |mean| sagittal offset "
              f"({step2[best]['abs_mean_deg']:.3f} deg)")
    if tie:
        within = [c for c in ranked
                  if step2[c]["abs_mean_deg"] - step2[best]["abs_mean_deg"] < TIE_DEG]
        if RECIPE_CHOICE in within:
            chosen = RECIPE_CHOICE
            reason = (f"tie within {TIE_DEG} deg among {within}; tie-break to "
                      f"the recipe's choice, {RECIPE_CHOICE}")
        else:
            reason = (f"tie within {TIE_DEG} deg among {within}, none of which "
                      f"is the recipe's {RECIPE_CHOICE}; smallest kept")
    print(f"\n=== STEP 3: DECISION ===")
    print(f"  ranking by |mean| offset: "
          + ", ".join(f"{c} {step2[c]['abs_mean_deg']:.3f}" for c in ranked))
    print(f"  gap between 1st and 2nd: {gap:.3f} deg "
          f"({'TIE' if tie else 'no tie'}, threshold {TIE_DEG})")
    print(f"  CHOSEN: {chosen}  -- {reason}")

    print(f"\n=== STEP 4: AthleticsPose hip error if the GT used each thorax ===")
    print(f"  {'candidate':<20}{'hip MAE':>10}{'hip bias':>10}")
    step4 = {}
    for name in CANDIDATES:
        mae = [x for s in hip_err[name] for x in hip_err[name][s]["mae"]]
        bias = [x for s in hip_err[name] for x in hip_err[name][s]["bias"]]
        step4[name] = {"hip_mae": float(np.mean(mae)) if mae else float("nan"),
                       "hip_bias": float(np.mean(bias)) if bias else float("nan"),
                       "per_subject": {
                           s: {"mae": float(np.mean(hip_err[name][s]["mae"])),
                               "bias": float(np.mean(hip_err[name][s]["bias"]))}
                           for s in hip_err[name]}}
        print(f"  {name:<20}{step4[name]['hip_mae']:>10.3f}"
              f"{step4[name]['hip_bias']:>10.3f}")

    out = {"what": "Phase 15C part 2: the thorax definition, decided on "
                   "AthleticsPose only. No MoVi subject involved.",
           "pre_registration": "results/phase15c.md, committed alone in 49428c0",
           "subjects": list(TEST_SUBJECTS), "actions": list(ACTIONS),
           "n_clips": len(clips), "n_frames": n_frames,
           "clip_restriction": "only clips where raw_markers_in_world and "
                               "gt_markers3d_by_cam have equal frame counts "
                               "(231 of 592); the dataset records no frame "
                               "correspondence for the rest",
           "alignment": "per frame, because p2mm varies per frame and so the "
                        "released array is perspective-normalised rather than "
                        "metric; candidate placement and est alignment are "
                        "fitted on the lower body only",
           "candidates_athleticspose": {k: list(v)
                                        for k, v in CANDIDATES.items()},
           "step1_recipe_residual_mm": step1,
           "step2_candidate_offsets": step2,
           "step3_decision": {"ranking": ranked, "gap_deg": gap,
                              "tie": bool(tie), "tie_threshold_deg": TIE_DEG,
                              "chosen": chosen, "reason": reason},
           "step4_hip_error_by_thorax": step4}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
