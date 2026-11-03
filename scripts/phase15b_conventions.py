"""Phase 15B -- is "perfect 2D" even in distribution, and whose convention is wrong?

Pre-registered in `results/phase15b.md`, committed alone in b13b48f before any
of this ran. Three hypotheses with outcome rules fixed there:

  H1  INPUT convention: the projected reference is out-of-distribution input at
      the hip, because the checkpoint was trained on COCO-detector hips.
  H2  REFERENCE convention: the MoVi reference uses a different joint recipe
      from the one AthleticsPose used.
  H3  Within-subject posture: hip error tracks pelvic tilt and trunk angle.

Standing rules, from the pre-registration: nothing is fitted to MoVi; the angle
code and the lifter are untouched; every original figure stays and new ones sit
beside it labelled by convention; the sealed test set is not opened; the phase
15 headline (9.610 near / 10.604 far, PG1, n=2) does not change.

Run: .venv/Scripts/python.exe scripts/phase15b_conventions.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data" / "athleticspose" / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from phase15_checks import (CALIB, angle_mask, camera_pose,  # noqa: E402
                             load_v3d, to_camera, to_h36m)
from phase15_movi import (CAMERAS, MOCAP, MOCAP_PER_VIDEO, MOVI,  # noqa: E402
                           decode_range, jogging_window)
from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import DATA, TEST_SUBJECTS, load_model  # noqa: E402
from phase7_inference_fix import CKPT_FOR, lift  # noqa: E402
from video_kinematics import (MODEL_FPS, TRAIN_CONF_MEAN,  # noqa: E402
                               coco17_to_h36m17, detect_coco17, resample,
                               scores_to_train_scale)

OUT = REPO / "results" / "phase15b.json"
CAM = "PG1"
SUBJECTS = ("Subject_1", "Subject_10", "Subject_11", "Subject_12",
            "Subject_13", "Subject_15")
# The four the H2 and H3 outcome rules are written against.
RULE_SUBJECTS = ("Subject_1", "Subject_10", "Subject_12", "Subject_13")
JOINTS = ("hip", "knee")
LIMBS = ("near", "far")

# --------------------------------------------------------------------- H2 ---
# The AthleticsPose recipe, read from the authors' own code:
#   data/athleticspose/repo/athleticspose/utils.py:mocap_to_h36m
#   data/athleticspose/repo/athleticspose/statics/joints.py:
#       joint_index_h36m_to_mocap
# Each H36M joint is the MEAN of a named marker set. Reproduced verbatim:
AP_RECIPE = {
    "pelvis": ("PELVIS",),
    "r_hip": ("RASI", "RPSI"),
    "r_knee": ("RKNE",),
    "r_ankle": ("RANK",),
    "l_hip": ("LASI", "LPSI"),
    "l_knee": ("LKNE",),
    "l_ankle": ("LANK",),
    "spine": ("Xiphoid", "T10"),
    "thorax": ("CLAV",),
    "neck": ("CLAV", "Nasion"),
    "head": ("Vertex", "EARC"),
    "l_shoulder": ("L_Shoulder Joint",),
    "l_elbow": ("LELB",),
    "l_wrist": ("LWRT",),
    "r_shoulder": ("R_Shoulder Joint",),
    "r_elbow": ("RELB",),
    "r_wrist": ("RWRT",),
}

# The same recipe expressed in MoVi's marker names. FIXED BEFORE SCORING, from
# the marker descriptions in the Visual3D export, never by which choice scores
# better. Four substitutions are needed and each is named in SUBSTITUTIONS.
AP_RECIPE_IN_MOVI = {
    "pelvis": ("PELVIS",),                 # same virtual pelvis centre
    "r_hip": ("RFWT", "RBWT"),             # RFWT = Right ASIS, RBWT = Right PSIS
    "r_knee": ("RKNE",),                   # lateral knee marker, not the centre
    "r_ankle": ("RANK",),                  # lateral ankle marker, not the centre
    "l_hip": ("LFWT", "LBWT"),
    "l_knee": ("LKNE",),
    "l_ankle": ("LANK",),
    "spine": ("STRN", "T10"),              # substitution: STRN for Xiphoid
    "thorax": ("CLAV",),                   # NOT the Visual3D THORAX point
    "neck": ("CLAV", "LFHD", "RFHD"),      # substitution: front-head mid for Nasion
    "head": ("HEAD",),                     # substitution: head centre for Vertex+EARC
    "l_shoulder": ("LSJC",),
    "l_elbow": ("LELB",),
    "l_wrist": ("LWRA", "LWRB"),           # substitution: wrist bar mid for LWRT
    "r_shoulder": ("RSJC",),
    "r_elbow": ("RELB",),
    "r_wrist": ("RWRA", "RWRB"),
}

SUBSTITUTIONS = {
    "spine": "AthleticsPose uses mean(Xiphoid, T10). MoVi has no Xiphoid "
             "marker; STRN ('Sternum') is the closest anatomical equivalent, "
             "the xiphoid process lying at the inferior sternum.",
    "neck": "AthleticsPose uses mean(CLAV, Nasion). MoVi has no Nasion; the "
            "midpoint of LFHD and RFHD ('Left/Right Front Head') is the "
            "closest available anterior head landmark.",
    "head": "AthleticsPose uses mean(Vertex, EARC). MoVi has neither; HEAD "
            "('Head Center') is the dataset's own head centre.",
    "l_wrist/r_wrist": "AthleticsPose uses LWRT/RWRT. MoVi ships the wrist bar "
                       "pair LWRA/LWRB ('Wrist Bar Thumb/Pinkie Side'); their "
                       "midpoint is the wrist centre.",
}


def to_h36m_recipe(v: dict) -> np.ndarray:
    """MoVi markers -> H36M-17 using the ATHLETICSPOSE recipe.

    Same gap handling as `phase15_checks.to_h36m`: `markerGaps == 1` marks a
    missing marker and Visual3D writes [0,0,0] there, so gapped markers become
    NaN and `np.mean` propagates -- a partial average would be an invention.
    """
    idx = {n: i for i, n in enumerate(v["marker_names"])}
    loc = v["marker_location"].copy()
    loc[np.asarray(v["marker_gaps"]) == 1] = np.nan
    out = np.full((loc.shape[0], 17, 3), np.nan)
    for joint, names in AP_RECIPE_IN_MOVI.items():
        missing = [n for n in names if n not in idx]
        if missing:
            raise KeyError(f"{joint}: absent from this export: {missing}")
        out[:, H36M[joint]] = np.mean([loc[:, idx[n]] for n in names], axis=0)
    return out


# --------------------------------------------------------------------- H1 ---
def _pelvis_frame_2d(kp: np.ndarray):
    """Origin at mid-hip, axis 1 along hip->hip, axis 2 perpendicular to it.

    Returns (origin, along, perp, torso) with torso the mid-shoulder to mid-hip
    distance -- the scale the offset is expressed in, so it transfers across
    camera views and body sizes instead of being a pixel count for one view.
    """
    xy = kp[:, :, :2]
    mid_hip = 0.5 * (xy[:, H36M["l_hip"]] + xy[:, H36M["r_hip"]])
    mid_sho = 0.5 * (xy[:, H36M["l_shoulder"]] + xy[:, H36M["r_shoulder"]])
    ml = xy[:, H36M["r_hip"]] - xy[:, H36M["l_hip"]]
    n = np.linalg.norm(ml, axis=1, keepdims=True)
    along = ml / np.maximum(n, 1e-9)
    perp = np.stack([-along[:, 1], along[:, 0]], axis=1)
    torso = np.linalg.norm(mid_sho - mid_hip, axis=1)
    return mid_hip, along, perp, torso


def measure_hip_offset(limit: int | None = None) -> dict:
    """H1: where do COCO-detector hips sit vs GT 2D hips? AthleticsPose only.

    Measured on the NON-held-out subjects, so the subjects the frozen baseline
    is reported on (S11, S13, S16) contribute nothing to the offset. Expressed
    in the pelvis frame and in torso-length units.
    """
    rows = {"l_hip": {"along": [], "perp": []},
            "r_hip": {"along": [], "perp": []}}
    clips = 0
    for action in ("running", "sprint"):
        det_dir = DATA / "det_markers2d_by_cam_coco" / action
        if not det_dir.is_dir():
            continue
        for subj_dir in sorted(det_dir.iterdir()):
            if subj_dir.name in TEST_SUBJECTS:
                continue                      # held out; excluded on purpose
            for f in sorted(subj_dir.glob("*.npy")):
                gt = DATA / "gt_markers2d_by_cam" / action / subj_dir.name / f.name
                if not gt.exists():
                    continue
                d = np.load(f).astype(float)
                g = np.load(gt).astype(float)
                n = min(len(d), len(g))
                if n < 10:
                    continue
                d, g = d[:n], g[:n]
                _o, along, perp, torso = _pelvis_frame_2d(g)
                ok = torso > 1e-6
                for name in ("l_hip", "r_hip"):
                    off = d[:, H36M[name], :2] - g[:, H36M[name], :2]
                    a = np.sum(off * along, axis=1) / np.maximum(torso, 1e-9)
                    p = np.sum(off * perp, axis=1) / np.maximum(torso, 1e-9)
                    rows[name]["along"].extend(a[ok].tolist())
                    rows[name]["perp"].extend(p[ok].tolist())
                clips += 1
                if limit and clips >= limit:
                    break
            if limit and clips >= limit:
                break
        if limit and clips >= limit:
            break

    out = {"source": "AthleticsPose det_markers2d_by_cam_coco vs "
                     "gt_markers2d_by_cam",
           "subjects": "all non-held-out (TEST_SUBJECTS excluded)",
           "n_clips": clips, "units": "fraction of 2D torso length",
           "frame": "pelvis frame: 'along' is the hip->hip axis (+ = toward "
                    "the right hip), 'perp' is perpendicular to it in the image",
           "hips": {}}
    for name, v in rows.items():
        a, p = np.asarray(v["along"]), np.asarray(v["perp"])
        out["hips"][name] = {
            "n_frames": int(a.size),
            "along_mean": float(a.mean()), "along_sd": float(a.std(ddof=1)),
            "perp_mean": float(p.mean()), "perp_sd": float(p.std(ddof=1))}
    return out


def apply_hip_offset(kp2d: np.ndarray, offset: dict) -> np.ndarray:
    """Displace the hips of a 2D array by the AthleticsPose-measured offset.

    The offset arrives in torso-length units in the pelvis frame, so it is
    re-expressed in this clip's own pixels using this clip's own torso length
    and hip axis. No MoVi quantity enters the offset itself.
    """
    out = kp2d.copy()
    _o, along, perp, torso = _pelvis_frame_2d(kp2d)
    for name in ("l_hip", "r_hip"):
        o = offset["hips"][name]
        d = (along * (o["along_mean"] * torso)[:, None]
             + perp * (o["perp_mean"] * torso)[:, None])
        out[:, H36M[name], :2] = kp2d[:, H36M[name], :2] + d
    return out


# ------------------------------------------------------------------ arms ----
def project(gt_world: np.ndarray, cam: str) -> np.ndarray:
    """Reference 3D -> pixels through the camera's own calibration."""
    R_cv, t_cv, _c = camera_pose(cam)
    cc = to_camera(gt_world, R_cv, t_cv)
    z = np.load(CALIB / f"cameraParams_{cam}.npz")
    K = np.asarray(z["IntrinsicMatrix"], float)
    k1, k2 = np.asarray(z["RadialDistortion"], float).ravel()[:2]
    u, w = cc[..., 0] / cc[..., 2], cc[..., 1] / cc[..., 2]
    r2 = u ** 2 + w ** 2
    f = 1.0 + k1 * r2 + k2 * r2 ** 2
    out = np.stack([u * f * K[0, 0] + K[2, 0],
                    w * f * K[1, 1] + K[2, 1],
                    np.full_like(u, TRAIN_CONF_MEAN)], axis=-1)
    out[~np.isfinite(out).all(axis=-1)] = np.nan
    return out


def lift_2d(model, kp2d: np.ndarray, fps: float = 30.0) -> np.ndarray:
    n = len(kp2d)
    n_model = max(int(round(n * MODEL_FPS / fps)), 2)
    return resample(lift(model, resample(kp2d, n_model), "edge", "overlap"), n)


def score(est: np.ndarray, ref: np.ndarray, near: str) -> dict:
    """MAE / bias / SD / r per limb per joint. The angle code is untouched."""
    out: dict = {}
    for limb, side in (("near", near), ("far", "r" if near == "l" else "l")):
        m = angle_mask(ref, side)
        ga, ea = sagittal_angles(ref, side), sagittal_angles(est, side)
        out[limb] = {}
        for j in JOINTS:
            gv, ev = ga[j][m], ea[j][m]
            ok = np.isfinite(gv) & np.isfinite(ev)
            if ok.sum() < 5:
                out[limb][j] = {k: float("nan")
                                for k in ("mae", "bias", "sd", "r")}
                out[limb][j]["n"] = int(ok.sum())
                continue
            d = (ev - gv)[ok]
            out[limb][j] = {
                "mae": float(np.abs(d).mean()), "bias": float(d.mean()),
                "sd": float(d.std(ddof=1)), "n": int(ok.sum()),
                "r": (float(np.corrcoef(gv[ok], ev[ok])[0, 1])
                      if gv[ok].std() > 1e-9 and ev[ok].std() > 1e-9
                      else float("nan"))}
        out[limb]["mean"] = float(np.nanmean(
            [out[limb][j]["mae"] for j in JOINTS]))
    return out


# -------------------------------------------------------------------- H3 ----
def pelvic_tilt(v: dict, mk: np.ndarray, up_axis: int = 2) -> np.ndarray:
    """ANATOMICAL pelvic tilt from MoVi's own ASIS and PSIS markers.

    The ASIS->PSIS axis against horizontal, in the sagittal plane: positive is
    an anterior tilt (ASIS below PSIS). Taken from the raw markers rather than
    from the H36M-17 reference, because that reference carries one point per hip
    and so has no pelvis orientation at all -- an earlier version of this used
    pelvis->spine as a stand-in, which is both mislabelled and collinear with
    trunk angle.
    """
    idx = {n: i for i, n in enumerate(v["marker_names"])}
    loc = v["marker_location"].copy()
    loc[np.asarray(v["marker_gaps"]) == 1] = np.nan
    asis = 0.5 * (loc[mk, idx["LFWT"]] + loc[mk, idx["RFWT"]])
    psis = 0.5 * (loc[mk, idx["LBWT"]] + loc[mk, idx["RBWT"]])
    lw, rw = loc[mk, idx["LFWT"]], loc[mk, idx["RFWT"]]
    up = np.zeros(3)
    up[up_axis] = 1.0
    ml = rw - lw
    ml = ml / np.maximum(np.linalg.norm(ml, axis=1, keepdims=True), 1e-9)
    fwd = np.cross(ml, up)
    fwd = fwd / np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
    ap = asis - psis
    # angle of the pelvis fore-aft axis above horizontal
    return np.degrees(np.arctan2(np.sum(ap * up, axis=1),
                                 np.sum(ap * fwd, axis=1)))


def trunk_angle(ref: np.ndarray, up_axis: int = 2) -> np.ndarray:
    """pelvis->thorax against vertical, in the sagittal plane.

    Built from the reference being scored against, so it describes that
    reference's own trunk convention -- which is the thing under suspicion.
    """
    pelvis, thorax = ref[:, H36M["pelvis"]], ref[:, H36M["thorax"]]
    lh, rh = ref[:, H36M["l_hip"]], ref[:, H36M["r_hip"]]
    up = np.zeros(3)
    up[up_axis] = 1.0
    ml = rh - lh
    ml = ml / np.maximum(np.linalg.norm(ml, axis=1, keepdims=True), 1e-9)
    fwd = np.cross(ml, up)
    fwd = fwd / np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
    trunk = thorax - pelvis
    return np.degrees(np.arctan2(np.sum(trunk * fwd, axis=1),
                                 np.sum(trunk * up, axis=1)))


def regress(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS with an intercept; slopes, 95% CIs and R^2."""
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    if len(y) < 20:
        return {"n": int(len(y)), "r2": float("nan")}
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    names = ("intercept", "tilt", "trunk")
    return {"n": int(len(y)), "r2": float(r2),
            "terms": {names[i]: {"slope": float(beta[i]),
                                 "ci95": [float(beta[i] - 1.96 * se[i]),
                                          float(beta[i] + 1.96 * se[i])]}
                      for i in range(len(beta))}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--offset-clips", type=int, default=0,
                    help="cap AthleticsPose clips for the H1 offset (0 = all)")
    ap.add_argument("--subjects", nargs="*", default=list(SUBJECTS))
    ap.add_argument("--max-frames", type=int, default=0,
                    help="smoke test: cap the jogging window")
    args = ap.parse_args()

    print("=== H1: COCO-detector hip offset, AthleticsPose tune data only ===")
    offset = measure_hip_offset(args.offset_clips or None)
    for name, o in offset["hips"].items():
        print(f"  {name}: along {o['along_mean']:+.4f} (sd {o['along_sd']:.4f}) "
              f"perp {o['perp_mean']:+.4f} (sd {o['perp_sd']:.4f})  "
              f"torso-lengths, n={o['n_frames']}")
    print(f"  from {offset['n_clips']} clips, {offset['subjects']}")

    model = load_model(CKPT_FOR["coco"], "base")
    if model is None:
        raise SystemExit("checkpoint missing")

    mats = {p.stem.split("_v3d_")[-1]: p
            for p in sorted(MOCAP.glob("F_v3d_*.mat"))}
    results: dict = {}
    joint_dist: dict = {j: [] for j in AP_RECIPE_IN_MOVI}

    for subject in args.subjects:
        if subject not in mats:
            print(f"\n{subject}: no mocap on disk, skipping")
            continue
        video = MOVI / CAMERAS[CAM]["pattern"].format(s=subject.split("_")[-1])
        if not video.exists():
            print(f"\n{subject}: no {CAM} video on disk, skipping")
            continue
        print(f"\n--- {subject} ---")
        v = load_v3d(mats[subject])
        _ji, f30, f120 = jogging_window(v)
        ref_orig_all = to_h36m(v)
        ref_recipe_all = to_h36m_recipe(v)

        b30 = int(f30[1])
        if args.max_frames:
            b30 = min(b30, int(f30[0]) + args.max_frames - 1)
        frames, fps, vmeta = decode_range(video, int(f30[0]), b30)
        n_v = len(frames)
        mk = int(f120[0]) + MOCAP_PER_VIDEO * np.arange(n_v)
        mk = mk[mk < ref_orig_all.shape[0]]
        n_v = len(mk)
        frames = frames[:n_v]
        ref = {"original": ref_orig_all[mk], "recipe": ref_recipe_all[mk]}

        # joint-by-joint 3D distance between the two reference conventions
        for j in AP_RECIPE_IN_MOVI:
            a = ref["original"][:, H36M[j]]
            b = ref["recipe"][:, H36M[j]]
            d = np.linalg.norm(a - b, axis=1)
            d = d[np.isfinite(d)]
            if d.size:
                joint_dist[j].append(float(np.mean(d)))

        R_cv, t_cv, _c = camera_pose(CAM)
        gc = to_camera(ref["original"], R_cv, t_cv)
        near = ("l" if np.nanmean(gc[:, H36M["l_hip"], 2])
                < np.nanmean(gc[:, H36M["r_hip"], 2]) else "r")

        # ---- arms --------------------------------------------------------
        arms: dict = {}

        kp = coco17_to_h36m17(detect_coco17(frames, crop=True, model=None)[0])
        det_ok = np.abs(kp[:, :, :2]).sum(axis=(1, 2)) > 0
        kp = scores_to_train_scale(kp)
        arms["detected"] = {"est": lift_2d(model, np.nan_to_num(kp), fps),
                            "keep": det_ok}

        p_orig = project(ref["original"], CAM)
        keep = np.isfinite(p_orig).all(axis=(1, 2))
        arms["perfect_original"] = {
            "est_sub": lift_2d(model, p_orig[keep], fps), "keep": keep}

        p_h1 = apply_hip_offset(p_orig, offset)
        keep1 = np.isfinite(p_h1).all(axis=(1, 2))
        arms["perfect_convention_matched"] = {
            "est_sub": lift_2d(model, p_h1[keep1], fps), "keep": keep1}

        # ---- grid: arms x references -------------------------------------
        grid: dict = {}
        for arm, a in arms.items():
            grid[arm] = {}
            for rname, rr in ref.items():
                if "est" in a:
                    est, rsub = a["est"][a["keep"]], rr[a["keep"]]
                else:
                    est, rsub = a["est_sub"], rr[a["keep"]]
                grid[arm][rname] = score(est, rsub, near)
                s = grid[arm][rname]
                print(f"  {arm:<28}{rname:<9} near hip "
                      f"MAE {s['near']['hip']['mae']:6.3f} "
                      f"bias {s['near']['hip']['bias']:+7.3f} "
                      f"r {s['near']['hip']['r']:+.3f} | knee "
                      f"MAE {s['near']['knee']['mae']:6.3f} "
                      f"r {s['near']['knee']['r']:+.3f}")

        # ---- H3: within-subject posture regression -----------------------
        a = arms["perfect_original"]
        rsub = ref["original"][a["keep"]]
        est = a["est_sub"]
        m = angle_mask(rsub, near)
        tilt = pelvic_tilt(v, mk[a["keep"]])
        trunk = trunk_angle(rsub)
        err = (sagittal_angles(est, near)["hip"]
               - sagittal_angles(rsub, near)["hip"])
        X = np.column_stack([tilt, trunk])
        h3 = regress(err[m], X[m])
        h3["regressors"] = {
            "tilt": "anatomical pelvic tilt, mean(LFWT,RFWT) -> "
                    "mean(LBWT,RBWT) against horizontal",
            "trunk": "pelvis -> thorax against vertical, in the original "
                     "reference's own convention"}
        h3["collinearity_r"] = (
            float(np.corrcoef(tilt[m][np.isfinite(tilt[m])],
                              trunk[m][np.isfinite(tilt[m])])[0, 1])
            if np.isfinite(tilt[m]).sum() > 5 else float("nan"))
        print(f"  H3 posture -> near hip error: R2 {h3.get('r2', float('nan')):.3f} "
              f"(n={h3.get('n')})")

        results[subject] = {
            "near_limb": near, "n_video_frames": n_v,
            "video_fps": fps, "resolution": [vmeta["width"], vmeta["height"]],
            "in_rule_set": subject in RULE_SUBJECTS,
            "grid": grid, "h3": h3}

    out = {"what": "DIAGNOSTIC. Does not change the phase 15 headline "
                   "(9.610 near / 10.604 far, PG1, n=2).",
           "pre_registration": "results/phase15b.md, committed alone in b13b48f",
           "camera": CAM,
           "h1_offset": offset,
           "h2_recipe_athleticspose": {k: list(v) for k, v in AP_RECIPE.items()},
           "h2_recipe_in_movi": {k: list(v)
                                 for k, v in AP_RECIPE_IN_MOVI.items()},
           "h2_substitutions": SUBSTITUTIONS,
           "h2_joint_distance_mm": {
               j: (float(np.mean(d)) if d else float("nan"))
               for j, d in joint_dist.items()},
           "subjects": results}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
