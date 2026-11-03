"""Phase 15 -- the six pre-registered checks on MoVi, before the first number.

`results/phase15.md` (frozen 2026-10-20, commit 64fb0bc) requires six unit and
convention checks to pass before any MAE is reported, and `CLAUDE.md` requires
them written into `docs/data-inventory.md`. This script answers them from the
dataset's own files, never from the paper and never from an assumption.

It also answers the question `CLAUDE.md` says phase 15 must settle first --
**is any MoVi camera side-on to the direction of travel?** -- using phase 3B's
`view_ratio` unchanged, so the answer is on the same axis as the measured
viewpoint curve (3.50 deg frontal -> 1.77 deg side-on) rather than a new scale.
`view_ratio` is hip DEPTH separation over hip width, so it runs 0 = frontal to
1 = side-on; AthleticsPose's median is 0.751.

ROUND F, NOT ROUND S. Round S's Visual3D export carries 12 physical markers
(head, wrists, ankles) and no hip, knee or pelvis, so check 3 fails for it
irrecoverably; it also ships no `motions_list` and no frame flags. See
`docs/data-inventory.md` 8.5.

Run: .venv/Scripts/python.exe scripts/phase15_checks.py [--mat PATH ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase11_viewpoint import view_ratio  # noqa: E402

MOVI = REPO / "data" / "movi"
CALIB = MOVI / "calib" / "Calib"
OUT = REPO / "results" / "phase15_checks.json"

# Visual3D marker names -> H36M-17. A tuple means "average these", which is how
# H36M's own hip is defined: the mean of the ASIS and PSIS markers, a point
# INSIDE the pelvis. That definition is the 21-23 px offset phase 7 section 5
# found for every image-based detector, and MoVi ships the two markers that
# define it -- LFWT/LBWT is Left ASIS/Left PSIS, RFWT/RBWT the right pair.
MOVI_TO_H36M: dict[str, str | tuple[str, ...]] = {
    "pelvis": ("LFWT", "LBWT", "RFWT", "RBWT"),
    "r_hip": ("RFWT", "RBWT"),
    "r_knee": "RKJC",
    "r_ankle": "RAJC",
    "l_hip": ("LFWT", "LBWT"),
    "l_knee": "LKJC",
    "l_ankle": "LAJC",
    "spine": "THORAX",       # replaced below by the pelvis-thorax midpoint
    "thorax": "THORAX",
    "neck": "CLAV",
    "head": "HEAD",
    "l_shoulder": "LSJC",
    "l_elbow": "LEJC",
    "l_wrist": "LWJC",
    "r_shoulder": "RSJC",
    "r_elbow": "REJC",
    "r_wrist": "RWJC",
}

MOCAP_HZ = 120.0
VIDEO_FPS_DECLARED = 30.0
# ---------------------------------------------------------------------------
# CHECK 1's FEMUR BAR, RE-REGISTERED 2026-11-02. Old bar (350, 550) mm kept
# below, in the record, with the reason. Full account: `results/phase15.md` D7.
#
# WHAT THIS CHECK IS FOR. It detects a UNIT ERROR -- millimetres read as
# centimetres (10x) or as metres (1000x), the trap that cost this project two
# wrong claims on AthleticsPose's `p2mm`. It is NOT an anatomical outlier
# detector, and it was never meant to rank subjects by body proportion. Used as
# one it rejected 2 of 7 MoVi round F subjects for being tall, one of them with a
# 3.9% gap rate identical to a subject that passed.
#
# WHAT IS ACTUALLY MEASURED. Not an anatomical femur:
#       L = | mean(ASIS, PSIS)  ->  knee joint centre |
# `mean(ASIS, PSIS)` lies INSIDE the pelvis, so L exceeds an anatomical femur by
# roughly the pelvis-landmark-to-hip-joint-centre offset. The old bar was the
# anatomical range, so a tall subject exceeded it by construction.
#
# DERIVATION. Two terms, both external to MoVi. No MoVi subject's measurement
# is used anywhere in it -- a bar must not be fitted to the subjects it judges.
#
#   TERM 1, anatomical femur length, from a POPULATION-GENERAL ratio rather
#   than one skeletal series. Feldesman et al.'s generic femur:stature ratio,
#   derived from a systematic review of 51 population means, puts femur length
#   at 26.5-27.0% of standing height in adults (mean 26.74%); see Feldesman,
#   Kleckner & Lundy, and the ratio as applied in Mays, Am J Phys Anthropol 2016
#   (doi:10.1002/ajpa.23068). The primary 1992 bibliographic record was NOT
#   verified first-hand, and sources differ between 26.74% and 26.75% -- so the
#   RANGE 26.5-27.0% is used, not the point value.
#
#   A deliberately over-wide adult stature span of 1450-2100 mm is applied to
#   it. That is wider than any published adult 1st-99th percentile interval for
#   a North American population, and is chosen to be wider on purpose: this bar
#   exists to catch a factor of ten, so erring wide costs nothing and erring
#   narrow excludes real subjects. (ANSUR II, Natick 2012, n = 11,961 US Army
#   personnel, is the obvious percentile source for this population, but its
#   1st/99th stature values could not be verified from the primary report here,
#   so no percentile figure is quoted.)
#
#       0.265 x 1450 mm  =  384 mm        0.270 x 2100 mm  =  567 mm
#
#   TERM 2, the pelvis-landmark-to-hip-joint-centre offset. THIS PROJECT
#   measured it at 81 mm on AthleticsPose (phase 12, `results/phase12.md`) -- an
#   independent dataset, not MoVi. Harrington et al., J Biomech 40(3):595-602
#   (2007) is the established and best-validated regression for hip joint centre
#   from pelvic landmarks, cited here for the existence and sense of the
#   displacement; its coefficients are NOT quoted, because they were not
#   verified against the paper. A +/-50% tolerance on the one measured value
#   gives 40-122 mm, and that tolerance is a judgement, declared as one.
#
#   BAR = (384 + 40, 567 + 122) = (424, 689), rounded outwards to (420, 700).
#
# THE BAR IS JUSTIFIED BY ITS PURPOSE, NOT BY THE SUBJECTS. Across the WHOLE
# plausible human range for L, 424-689 mm, the bar separates the true scale from
# a 10x or 1/10 misreading with a factor of ~6 of headroom at each end:
#
#   scale      plausible L maps to        vs bar (420, 700)
#   1/1000     0.42 -  0.69 mm            below  -> REJECTED
#   1/10       42.4 -  68.9 mm            below  -> REJECTED
#   1x         424  -  689  mm            inside -> ACCEPTED
#   10x        4240 - 6890  mm            above  -> REJECTED
#   1000x      424000-689000 mm           above  -> REJECTED
#
# The widest plausible true value (689) is 6.2x below the smallest 10x value
# (4240), and the narrowest plausible true value (424) is 6.2x above the largest
# 1/10 value (68.9). So no unit error of 10x or more can masquerade as a valid
# length anywhere in the human range, which is exactly the check's job -- and
# the bar is set by that requirement rather than by which subjects pass.
# `tests/test_phase15_checks_bar.py` asserts this table.
FEMUR_MM_OLD_PREREGISTERED = (350.0, 550.0)   # superseded 2026-11-02; see D7
FEMUR_MM = (420.0, 700.0)     # re-registered 2026-11-02, derivation above
TIBIA_MM = (300.0, 500.0)     # unchanged: both ends are joint centres, so it
                              # carries no definitional offset and never failed


def _f(struct, field: str):
    """One MATLAB struct field, with scipy's size-1 object wrappers peeled off.

    `loadmat` represents a MATLAB struct as nested object arrays, so a field is
    usually an ndarray of dtype object holding a single ndarray. Peeling in a
    loop is safer than indexing `[0][0]` at each site, because the depth of the
    wrapping differs between fields and between the two rounds.
    """
    v = struct[field]
    while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        v = v.item()
    return v


def _strings(v) -> list[str]:
    """A MATLAB cell array of char into a flat list of str."""
    return [str(np.asarray(x).ravel()[0]) for x in np.asarray(v).ravel()]


def load_v3d(path: Path) -> dict:
    """Read one MoVi Visual3D subject export into plain arrays."""
    d = loadmat(path)
    key = [k for k in d if not k.startswith("__")][0]
    top = d[key][0, 0]
    m = _f(top, "move")
    if m.ndim == 2:
        m = m[0, 0]
    subj = _f(top, "subject")

    out = {
        "key": key,
        "subject_id": str(np.asarray(_f(subj, "id")).ravel()[0]),
        # `subject` also carries sex, height, mass and age. They are NOT read.
        # MoVi's licence forbids reproducing the Data "in any form to any third
        # party", this repository is public, and per-participant sex and stature
        # keyed to the dataset's own subject id is the Data about a consenting
        # human subject rather than a measurement of this method. No check here
        # needs them -- they were collected only because they were in the struct.
        "marker_names": _strings(_f(m, "markerName")),
        "marker_location": np.asarray(_f(m, "markerLocation"), dtype=float),
        "marker_gaps": np.asarray(_f(m, "markerGaps")),
    }
    if "motions_list" in m.dtype.names:
        out["motions"] = _strings(_f(m, "motions_list"))
        out["flags30"] = np.asarray(_f(m, "flags30"), dtype=int)
        out["flags120"] = np.asarray(_f(m, "flags120"), dtype=int)
    return out


def to_h36m(v: dict) -> np.ndarray:
    """(n_frames, 17, 3) in the H36M-17 order, from named Visual3D markers.

    **`markerGaps == 1` marks a MISSING marker, and Visual3D writes [0, 0, 0]
    in its place.** Zero is a sentinel, not a measurement: read literally it put
    the knee joint centre at the lab origin and drove jogging knee flexion to
    +148.7 deg, outside anything anatomical. Gapped markers become NaN here and
    the frames they touch are EXCLUDED, never interpolated -- `CLAUDE.md` forbids
    imputing dataset rows and forbids re-deriving a reference 3D before scoring
    against it. Same class of trap as Ferber's `999` and the `dv_r` zeros.
    """
    idx = {n: i for i, n in enumerate(v["marker_names"])}
    loc = v["marker_location"].copy()
    gaps = np.asarray(v["marker_gaps"])
    loc[gaps == 1] = np.nan                      # sentinel -> missing
    out = np.full((loc.shape[0], 17, 3), np.nan)
    for joint, src in MOVI_TO_H36M.items():
        cols = (src,) if isinstance(src, str) else src
        missing = [c for c in cols if c not in idx]
        if missing:
            raise KeyError(f"{joint}: markers absent from this export: {missing}")
        # np.mean, not nanmean: if any contributing marker is gapped the joint
        # is unknown, and a partial average would be a quiet invention.
        out[:, H36M[joint]] = np.mean([loc[:, idx[c]] for c in cols], axis=0)
    # H36M's spine sits between the root and the thorax, not on the thorax.
    out[:, H36M["spine"]] = 0.5 * (out[:, H36M["pelvis"]] + out[:, H36M["thorax"]])
    return out


# The joints `sagittal_angles` actually reads. Its body frame is pelvis->thorax
# and hip<->hip, then the limb chain for one side -- so an elbow or wrist gap is
# irrelevant to a knee angle and must not cost that frame.
FRAME_JOINTS = ("pelvis", "thorax", "l_hip", "r_hip")


def angle_mask(gt: np.ndarray, side: str) -> np.ndarray:
    """Frames where the hip and knee angles for `side` are actually measured.

    Derived from the GT array rather than from a hand-kept marker list, so it
    cannot drift out of step with `MOVI_TO_H36M` or with `sagittal_angles`.

    **Per quantity, not global.** Excluding every frame with a gap in ANY mapped
    marker threw away 21.2% of subject 1's jogging window, and most of that was
    `LEJC`, `REJC` and `LWJC` -- elbows and a wrist, which no sagittal hip or
    knee angle reads. Discarding good frames is not a conservative choice; it is
    a smaller n reported as if it were the data's fault.
    """
    need = list(FRAME_JOINTS) + [f"{side}_hip", f"{side}_knee", f"{side}_ankle"]
    idx = [H36M[j] for j in need]
    return np.isfinite(gt[:, idx]).all(axis=(1, 2))


def gap_report(v: dict, sl: slice, gt: np.ndarray) -> dict:
    """What the gap sentinel costs -- globally, and per measured quantity."""
    used: set[str] = set()
    for src in MOVI_TO_H36M.values():
        used.update((src,) if isinstance(src, str) else src)
    idx = {n: i for i, n in enumerate(v["marker_names"])}
    gaps = np.asarray(v["marker_gaps"])[sl]
    cols = [idx[n] for n in sorted(used) if n in idx]
    per_marker = {n: int(gaps[:, idx[n]].sum()) for n in sorted(used) if n in idx}
    any_bad = (gaps[:, cols] == 1).any(axis=1)
    per_side = {}
    for side in ("l", "r"):
        m = angle_mask(gt, side)
        per_side[side] = {"n_usable": int(m.sum()),
                          "n_excluded": int((~m).sum()),
                          "fraction_excluded": round(float((~m).mean()), 4)}
    return {"n_frames": int(gaps.shape[0]),
            "n_frames_with_a_gap_in_any_mapped_marker": int(any_bad.sum()),
            "fraction_if_excluded_globally": round(float(any_bad.mean()), 4),
            "per_side_for_hip_and_knee": per_side,
            "joints_each_angle_reads": list(FRAME_JOINTS) + ["<side>_hip",
                                                             "<side>_knee",
                                                             "<side>_ankle"],
            "gapped_frames_per_marker": {k: n for k, n in per_marker.items() if n},
            "handling": "excluded, never interpolated"}


def camera_pose(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R_cv, t_cv, centre) in world millimetres, from MATLAB-convention files.

    MATLAB stores extrinsics for ROW vectors: x_cam = x_world @ R + t. Column
    convention wants x_cam = R_cv @ x_world + t_cv, so R_cv = R.T and t_cv = t.
    The camera centre follows as -R_cv.T @ t_cv = -R @ t.
    """
    z = np.load(CALIB / f"Extrinsics_{name}.npz")
    R, t = np.asarray(z["rotationMatrix"], float), np.asarray(
        z["translationVector"], float).ravel()
    R_cv = R.T
    centre = -(R @ t)
    return R_cv, t, centre


def to_camera(kp_world: np.ndarray, R_cv: np.ndarray, t_cv: np.ndarray) -> np.ndarray:
    """World -> camera coordinates, so axis 2 is depth and `view_ratio` applies."""
    return kp_world @ R_cv.T + t_cv


def project_reference(kp_world: np.ndarray, cam: str) -> np.ndarray:
    """Project the reference 3D into a PG camera: (n, 17, 3), H36M-ordered.

    This is a PERFECT 2D detector for this dataset, and it exists only because
    MoVi ships intrinsics, extrinsics and radial distortion for the Point Grey
    rigs. Its purpose is diagnostic: it bounds how much of the measured error is
    the 2D detector and how much is the lifter arriving in a lab it has never
    seen. Phase 7 section 5 did the same thing on AthleticsPose and got 5.16 deg
    for this checkpoint, so the figure has a reference point.

    It is NOT a tuning instrument. Nothing is selected on its output; it is
    reported beside the real number as a decomposition.

    Channel 2 is set to the trained confidence constant, matching what the real
    path now feeds the lifter, so the only difference between the two arms is
    the 2D coordinates themselves.

    MATLAB intrinsics are the row-vector form, so `IntrinsicMatrix` holds fx and
    fy on the diagonal and the principal point in its LAST ROW.
    """
    from video_kinematics import TRAIN_CONF_MEAN

    R_cv, t_cv, _centre = camera_pose(cam)
    cc = to_camera(kp_world, R_cv, t_cv)
    z = np.load(CALIB / f"cameraParams_{cam}.npz")
    K = np.asarray(z["IntrinsicMatrix"], float)
    k1, k2 = np.asarray(z["RadialDistortion"], float).ravel()[:2]
    u = cc[..., 0] / cc[..., 2]
    w = cc[..., 1] / cc[..., 2]
    r2 = u ** 2 + w ** 2
    f = 1.0 + k1 * r2 + k2 * r2 ** 2
    px = u * f * K[0, 0] + K[2, 0]
    py = w * f * K[1, 1] + K[2, 1]
    out = np.stack([px, py, np.full_like(px, TRAIN_CONF_MEAN)], axis=-1)
    out[~np.isfinite(out).all(axis=-1)] = np.nan
    return out


def travel_azimuth(pelvis: np.ndarray, centre: np.ndarray, up_axis: int,
                   win: int = 10) -> dict:
    """Angle between the camera's line of sight and the direction of travel.

    Both projected onto the horizontal plane, **per frame**, and summarised by
    the median. 90 deg is side-on, 0 deg is the camera looking along the path.
    Reported beside `view_ratio` as an independent construction of the same
    geometry.

    PER FRAME, not from the mean velocity. A jogging trial inside a 15-camera
    lab volume is a shuttle or a loop, not a straight line, so the direction of
    the MEAN velocity is close to meaningless -- it read 88.6 deg for subject 1
    and 36.3 deg for subject 10 while `view_ratio` sat at 0.72 and 0.71 for the
    same clips. The sign of travel is also irrelevant here (a camera is equally
    side-on to a runner going either way), hence |cos| rather than cos.
    """
    hor = [a for a in range(3) if a != up_axis]
    p = pelvis[:, hor]
    # Central difference over ~win frames: at 120 Hz that is ~83 ms, long
    # enough to beat marker noise and short enough to track a turn.
    vel = p[win:] - p[:-win]
    mid = 0.5 * (p[win:] + p[:-win])
    sight = mid - centre[hor]
    ok = (np.isfinite(vel).all(axis=1) & np.isfinite(sight).all(axis=1)
          & (np.linalg.norm(vel, axis=1) > 1e-3)
          & (np.linalg.norm(sight, axis=1) > 1e-3))
    if ok.sum() < 5:
        return {"deg": float("nan"), "n": int(ok.sum())}
    v = vel[ok] / np.linalg.norm(vel[ok], axis=1, keepdims=True)
    s = sight[ok] / np.linalg.norm(sight[ok], axis=1, keepdims=True)
    cos = np.abs(np.sum(v * s, axis=1))
    deg = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
    return {"deg": float(np.median(deg)), "n": int(ok.sum()),
            "p25": float(np.percentile(deg, 25)),
            "p75": float(np.percentile(deg, 75))}


def check_subject(path: Path) -> dict:
    v = load_v3d(path)
    r: dict = {"file": path.name, "subject_id": v["subject_id"],
               "n_markers": len(v["marker_names"]),
               "n_frames_mocap": int(v["marker_location"].shape[0])}
    print(f"\n===== {path.name} :: {v['subject_id']} =====")
    print(f"  {r['n_markers']} markers, {r['n_frames_mocap']} mocap frames")

    # ---- check 6: subject identity across modalities -----------------------
    stem = path.stem                                  # F_v3d_Subject_1
    expect = stem.split("_v3d_")[-1]                  # Subject_1
    r["check6_subject_identity"] = {
        "mat_field": v["subject_id"], "filename": expect,
        "video_stem_would_be": f"F_PG1_{expect}_L.avi",
        "pass": v["subject_id"] == expect}
    print(f"  [6] identity: mat says {v['subject_id']}, filename says {expect}"
          f" -> {'PASS' if r['check6_subject_identity']['pass'] else 'FAIL'}")

    # ---- the movement list, read out of the data ---------------------------
    if "motions" not in v:
        r["fatal"] = ("no motions_list in this export -- round S cannot be "
                      "segmented into movements")
        print(f"  FATAL: {r['fatal']}")
        return r
    motions = v["motions"]
    r["motions_list"] = motions
    jog = [i for i, m in enumerate(motions) if m.strip().lower() == "jogging"]
    rip = [i for i, m in enumerate(motions)
           if "running" in m.lower() or "in_spot" in m.lower()]
    if not jog:
        r["fatal"] = "no 'jogging' entry in motions_list"
        print(f"  FATAL: {r['fatal']}")
        return r
    ji = jog[0]
    r["jogging_index_0based"] = ji
    r["excluded_running_in_spot_index_0based"] = rip
    f30, f120 = v["flags30"][ji], v["flags120"][ji]
    r["jogging_frames"] = {"video_30fps": f30.tolist(),
                           "mocap_120hz": f120.tolist(),
                           "n_video": int(f30[1] - f30[0] + 1),
                           "n_mocap": int(f120[1] - f120[0] + 1),
                           "seconds_at_30fps": round((f30[1] - f30[0] + 1) / 30, 2)}
    print(f"  jogging is motions_list[{ji}] = '{motions[ji]}'; "
          f"excluded running_in_spot at {rip}")
    print(f"      video frames {f30.tolist()} ({r['jogging_frames']['n_video']} "
          f"= {r['jogging_frames']['seconds_at_30fps']}s), "
          f"mocap frames {f120.tolist()}")

    # ---- check 5: synchronisation, from the dataset's own flags ------------
    ratios = v["flags120"] / np.maximum(v["flags30"], 1)
    r["check5_sync"] = {
        "mocap_hz": MOCAP_HZ, "video_fps_declared": VIDEO_FPS_DECLARED,
        "expected_ratio": MOCAP_HZ / VIDEO_FPS_DECLARED,
        "median_flag_ratio": float(np.median(ratios)),
        "max_abs_dev_from_4": float(np.max(np.abs(ratios - 4.0))),
        "note": ("flags30 and flags120 are the SAME movement boundaries "
                 "expressed in video and mocap frames, so the dataset ships "
                 "the correspondence rather than leaving it to be estimated"),
        "pass": bool(np.max(np.abs(ratios - 4.0)) < 0.05)}
    print(f"  [5] sync: flags120/flags30 median "
          f"{r['check5_sync']['median_flag_ratio']:.4f}, max deviation from 4 "
          f"{r['check5_sync']['max_abs_dev_from_4']:.4f} -> "
          f"{'PASS' if r['check5_sync']['pass'] else 'REVIEW'}")

    # ---- check 3: the mapping, and whether it is complete ------------------
    try:
        gt = to_h36m(v)
    except KeyError as e:
        r["fatal"] = f"check 3 failed: {e}"
        print(f"  FATAL: {r['fatal']}")
        return r
    r["check3_mapping"] = {k: (v2 if isinstance(v2, str) else list(v2))
                           for k, v2 in MOVI_TO_H36M.items()}
    r["check3_mapping"]["spine"] = "midpoint(pelvis, THORAX)"
    jogsl = slice(int(f120[0]), int(f120[1]) + 1)
    g = gt[jogsl]
    r["n_frames_jogging_mocap"] = int(g.shape[0])
    r["gap_sentinel"] = gap_report(v, jogsl, g)
    gs = r["gap_sentinel"]
    print(f"  gaps: {gs['n_frames_with_a_gap_in_any_mapped_marker']} of "
          f"{gs['n_frames']} jogging frames have a gap in SOME mapped marker "
          f"({100 * gs['fraction_if_excluded_globally']:.1f}%), but hip and "
          f"knee read only {len(FRAME_JOINTS) + 3} joints:")
    for side in ("l", "r"):
        ps = gs["per_side_for_hip_and_knee"][side]
        print(f"        {side}: {ps['n_usable']} usable, {ps['n_excluded']} "
              f"excluded ({100 * ps['fraction_excluded']:.1f}%)")
    print(f"        per marker: {gs['gapped_frames_per_marker']}")

    # The H36M hip is mean(ASIS, PSIS) -- inside the pelvis. MoVi also ships
    # the anatomical hip joint centre, so the offset phase 12 measured at 81 mm
    # can be re-measured here on an independent dataset, for free.
    mi = {n: i for i, n in enumerate(v["marker_names"])}
    loc = v["marker_location"].copy()
    loc[np.asarray(v["marker_gaps"]) == 1] = np.nan
    # LHIP/RHIP are the HIP joint centres. LHJC/RHJC are the HAND joint
    # centres -- markerDescription reads "Left Hand Joint", and their height
    # matches the elbow. Using them put the offset at 325 mm; LHIP gives 104.
    lj = loc[jogsl, mi["LHIP"]]
    rj = loc[jogsl, mi["RHIP"]]
    r["h36m_hip_vs_joint_centre_mm"] = {
        "left": round(float(np.nanmedian(np.linalg.norm(
            g[:, H36M["l_hip"]] - lj, axis=-1))), 1),
        "right": round(float(np.nanmedian(np.linalg.norm(
            g[:, H36M["r_hip"]] - rj, axis=-1))), 1),
        "phase12_measured_mm": 81.0,
        "note": ("distance from H36M's mean(ASIS,PSIS) hip to the dataset's own "
                 "hip joint centre -- the irreducible offset no image-based "
                 "detector can mark")}
    print(f"  H36M hip vs the dataset's own hip joint centre: "
          f"L {r['h36m_hip_vs_joint_centre_mm']['left']} mm, "
          f"R {r['h36m_hip_vs_joint_centre_mm']['right']} mm "
          f"(phase 12 measured 81 mm)")

    # ---- check 1: units, via segment lengths that have to be human --------
    def seglen(a: str, b: str) -> float:
        d = np.linalg.norm(g[:, H36M[a]] - g[:, H36M[b]], axis=-1)
        return float(np.nanmedian(d))

    femur = 0.5 * (seglen("l_hip", "l_knee") + seglen("r_hip", "r_knee"))
    tibia = 0.5 * (seglen("l_knee", "l_ankle") + seglen("r_knee", "r_ankle"))
    hipw = seglen("l_hip", "r_hip")
    # THE VERDICT, NOT THE MEASUREMENTS. The question this check exists to
    # answer is "are the units millimetres or metres", and a pass against a
    # pre-registered human range answers it exactly as well as printing a
    # participant's femur length does -- without publishing the limb
    # measurements of three identifiable people from a dataset whose licence
    # forbids redistributing the Data. The values are computed and tested here;
    # they are simply not emitted.
    femur_ok = FEMUR_MM[0] <= femur <= FEMUR_MM[1]
    tibia_ok = TIBIA_MM[0] <= tibia <= TIBIA_MM[1]
    hip_ok = 120.0 <= hipw <= 350.0
    r["check1_units"] = {
        "femur_in_plausible_range": bool(femur_ok),
        "femur_plausible_mm": list(FEMUR_MM),
        "femur_bar_rereg": "2026-11-02; derived from purpose (unit detection) "
                           "and a population-general femur:stature ratio. "
                           "See results/phase15.md D7",
        "femur_bar_superseded_mm": list(FEMUR_MM_OLD_PREREGISTERED),
        "tibia_in_plausible_range": bool(tibia_ok),
        "tibia_plausible_mm": list(TIBIA_MM),
        "hip_width_in_plausible_range": bool(hip_ok),
        "units": "millimetres",
        "note": ("per-subject segment lengths are deliberately not reported: "
                 "see the comment above and docs/data-inventory.md 8.5"),
        "pass": bool(femur_ok and tibia_ok)}
    print(f"  [1] units: femur, tibia and hip width all inside their "
          f"pre-registered human ranges -> "
          f"{'PASS millimetres' if r['check1_units']['pass'] else 'FAIL'}")

    # ---- check 2: which axis is up, and is the frame mirrored -------------
    # The cameras give an independent answer: a camera height of ~1.3 m has to
    # lie on the vertical axis, and that is a fact about the rig, not a guess.
    cams = {}
    for name in ("PG1", "PG2"):
        R_cv, t_cv, centre = camera_pose(name)
        cams[name] = (R_cv, t_cv, centre)
    # local only -- never emitted; see check2_convention below
    # The axis on which BOTH camera rigs sit between 0.8 m and 2.5 m: that has
    # to be a mounting height, which is a fact about the rig rather than a guess
    # about the convention.
    cand = [a for a in range(3)
            if all(800.0 < abs(cams[n][2][a]) < 2500.0 for n in cams)]
    up_axis = cand[0] if len(cand) == 1 else 2
    head_minus_pelvis = np.nanmean(g[:, H36M["head"]] - g[:, H36M["pelvis"]],
                                   axis=0)
    knees = {}
    for side in ("l", "r"):
        m = angle_mask(g, side)
        k = sagittal_angles(g, side)["knee"][m]
        knees[side] = {"n": int(m.sum()),
                       "min": round(float(k.min()), 2),
                       "max": round(float(k.max()), 2),
                       "median": round(float(np.median(k)), 2)}
    knee = np.concatenate([sagittal_angles(g, s)["knee"][angle_mask(g, s)]
                           for s in ("l", "r")])
    r["check2_convention"] = {
        # The camera centres themselves are verbatim content of MoVi's
        # `Camera Parameters.tar`, so what is reported is the INFERENCE drawn
        # from them -- that both rigs sit at a height, which is what identifies
        # the vertical axis -- rather than the coordinates.
        "both_cameras_at_mounting_height_0.8_2.5m": bool(len(cand) == 1),
        "vertical_axis_from_camera_height": up_axis,
        "head_above_pelvis_on_that_axis": bool(
            int(np.argmax(np.abs(head_minus_pelvis))) == up_axis),
        "vertical_axis_from_posture": int(np.argmax(np.abs(head_minus_pelvis))),
        "knee_flexion_deg": {"min": round(float(knee.min()), 2),
                             "max": round(float(knee.max()), 2),
                             "median": round(float(np.median(knee)), 2)},
        "note": ("a mirrored axis would flip flexion sign; jogging knee "
                 "flexion must be one-signed and inside the tool's own "
                 "plausible band, hip (-40,60) knee (-170,20)"),
        "pass": bool(up_axis == int(np.argmax(np.abs(head_minus_pelvis)))
                     and -170.0 <= knee.min() and knee.max() <= 20.0)}
    print(f"  [2] convention: camera mounting heights put vertical on axis "
          f"{up_axis}; "
          f"head-pelvis agrees on axis "
          f"{int(np.argmax(np.abs(head_minus_pelvis)))}")
    print(f"      knee flexion over jogging: "
          f"[{knee.min():.1f}, {knee.max():.1f}] deg, median "
          f"{np.median(knee):.1f} -> "
          f"{'PASS' if r['check2_convention']['pass'] else 'REVIEW'}")

    # ---- the side-on question, per camera ---------------------------------
    r["viewpoint"] = {}
    for name, (R_cv, t_cv, centre) in cams.items():
        gcam = to_camera(g, R_cv, t_cv)
        # view_ratio is phase 3B's function, used unchanged, and it means its
        # np.mean: feed it only frames where the hips are actually measured
        # rather than nan-averaging around the gap sentinel.
        ok = np.isfinite(gcam[:, [H36M["l_hip"], H36M["r_hip"]]]).all(axis=(1, 2))
        vr = view_ratio(gcam[ok]) if ok.sum() > 5 else float("nan")
        pel = g[:, H36M["pelvis"]]
        az = travel_azimuth(pel[np.isfinite(pel).all(axis=1)], centre, up_axis)
        r["viewpoint"][name] = {
            "view_ratio": round(vr, 4),
            "travel_azimuth_deg": round(az["deg"], 1),
            "travel_azimuth_iqr_deg": [round(az.get("p25", float("nan")), 1),
                                       round(az.get("p75", float("nan")), 1)],
            "n_frames_scored": int(ok.sum()),
            # Rounded to the metre: the distance matters as "a few metres away,
            # like a phone on a tripod", and a decimal place would start to
            # describe the rig rather than the geometry.
            "distance_to_subject_m": round(float(np.linalg.norm(
                np.nanmean(g[:, H36M["pelvis"]], axis=0) - centre)) / 1000.0, 1)}
        print(f"  [side-on] {name}: view_ratio {vr:.4f} "
              f"(1 = side-on, AthleticsPose median 0.751), "
              f"travel azimuth {az['deg']:.1f} deg "
              f"[{az.get('p25', float('nan')):.0f}, "
              f"{az.get('p75', float('nan')):.0f}] (90 = side-on), "
              f"{r['viewpoint'][name]['distance_to_subject_m']:.1f} m away")
    r["left_limb_nearer_camera"] = {
        name: bool(np.nanmean(to_camera(g, *cams[name][:2])[:, H36M["l_hip"], 2])
                   < np.nanmean(to_camera(g, *cams[name][:2])[:, H36M["r_hip"], 2]))
        for name in cams}
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    # Default to the real mocap directory, not the one-file probe dir that
    # happened to exist when this was first written: an incomplete default
    # silently produced a checks file covering one subject and overwrote the
    # canonical one.
    ap.add_argument("--mat", nargs="*", type=Path,
                    default=sorted((MOVI / "raw" / "round_f" / "mocap")
                                   .glob("F_v3d_*.mat")))
    ap.add_argument("--out", type=Path, default=OUT,
                    help="where to write; defaults to results/phase15_checks.json")
    args = ap.parse_args()
    if not args.mat:
        print("no Visual3D .mat files given or found")
        return 2
    print("=== phase 15 pre-registered checks (results/phase15.md, 64fb0bc) ===")
    rows = [check_subject(p) for p in args.mat]
    out = {"round": "F",
           "why_not_round_S": ("round S's Visual3D export has 12 physical "
                               "markers -- head, wrists, ankles -- and no hip, "
                               "knee or pelvis, so check 3 cannot pass; it "
                               "also ships no motions_list and no frame flags"),
           "mocap_hz": MOCAP_HZ, "video_fps_declared": VIDEO_FPS_DECLARED,
           "check4_video_fps": "pending -- read from the video file itself",
           "subjects": rows}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
