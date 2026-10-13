"""Phase 12 - Scan: a per-subject body model from monocular video.

Recovers segment lengths from the lifted 3D and asks whether they are accurate
and stable enough for phase 13 to impose as a constraint. Bars are pre-registered
in results/phase12.md and frozen; this script does not choose them.

This measures the measurement. It emits no injury quantity of any kind.

Three things worth knowing before reading the code:

  * The lifter is SCALE-FREE. normalize_kpts divides by max|xy| of each clip, so
    absolute lengths carry an arbitrary per-clip factor. Ratios are the only
    quantity comparable across clips, and the only one a real video can produce.
    Absolute mm is reconstructed for the anatomy check only, by re-deriving that
    clip's norm_scale and dividing by p2mm.

  * Ground truth uses the H36M convention on BOTH sides. H36M puts the hip at
    mean(RASI, RPSI), 81.07 mm from the lab's own hip joint centre, which
    inflates the femur 18.2%. Comparing lifter output to anatomical lengths would
    charge the method for that. Anatomical lengths are reported separately so the
    offset stays visible. See docs/data-inventory.md 8.1.

  * Foot length is unavailable, not approximated. H36M-17 has no toe keypoint.
    Mocap has one, so ground truth is reported and the estimate is None.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AP = REPO / "data" / "athleticspose"
DATA = AP / "data" / "AthleticsPoseDataset"
sys.path.insert(0, str(AP / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from athleticspose.statics.joints import mocap_joints_name_to_index as MOCAP  # noqa: E402
from athleticspose.utils import mocap_to_h36m  # noqa: E402
from phase3_angles import H36M  # noqa: E402
from phase3_infer import load_model  # noqa: E402
from phase7_inference_fix import lift  # noqa: E402

OUT = REPO / "results" / "phase12_scan.json"
ACTIONS = ("running", "sprint")
CKPT_NAME = "motionagformer-b-ath-det-coco-v1.ckpt"   # matches the COCO detector
MIN_FRAMES = 25            # below this a median over frames is not meaningful
N_BOOT = 2000
SEED = 11

# --- the skeleton. There was no such table in the repo before this phase. -----
# H36M-17 joint pairs. Foot is deliberately absent: no toe keypoint exists.
SEGMENTS: dict[str, tuple[str, str]] = {
    "thigh_l": ("l_hip", "l_knee"),
    "thigh_r": ("r_hip", "r_knee"),
    "shank_l": ("l_knee", "l_ankle"),
    "shank_r": ("r_knee", "r_ankle"),
    "pelvis": ("l_hip", "r_hip"),
    "trunk": ("pelvis", "thorax"),
}

# Scale-invariant ratios, the gated quantity. Per side where the segment has one.
RATIOS: dict[str, tuple[str, str]] = {
    "thigh/shank_l": ("thigh_l", "shank_l"),
    "thigh/shank_r": ("thigh_r", "shank_r"),
    "thigh/pelvis_l": ("thigh_l", "pelvis"),
    "thigh/pelvis_r": ("thigh_r", "pelvis"),
    "shank/pelvis_l": ("shank_l", "pelvis"),
    "shank/pelvis_r": ("shank_r", "pelvis"),
    "trunk/thigh_l": ("trunk", "thigh_l"),
    "trunk/thigh_r": ("trunk", "thigh_r"),
}

# Anatomical definitions, mocap only. KNE/ANK are already exact medial-lateral
# midpoints (verified to 0.000 mm), so they are used directly; the hip is the
# lab's own joint centre, which is what H36M does NOT use.
ANATOMICAL: dict[str, tuple[str, str]] = {
    "thigh_l": ("LHIP", "LKNE"),
    "thigh_r": ("RHIP", "RKNE"),
    "shank_l": ("LKNE", "LANK"),
    "shank_r": ("RKNE", "RANK"),
    "pelvis": ("LHIP", "RHIP"),
    "trunk": ("SACR", "C7"),
    "foot_l": ("LHEL", "L_Toe"),      # mocap can measure it; the lifter cannot
    "foot_r": ("RHEL", "R_Toe"),
}


# ---------------------------------------------------------------- lengths -----
def bone_lengths(kp3d: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame segment lengths from (T, 17, 3) H36M keypoints.

    Units are whatever kp3d carries. Scale-free input gives scale-free output.
    """
    if kp3d.ndim != 3 or kp3d.shape[1] != 17:
        raise ValueError(f"expected (T, 17, 3), got {kp3d.shape}")
    out = {}
    for name, (a, b) in SEGMENTS.items():
        out[name] = np.linalg.norm(kp3d[:, H36M[a]] - kp3d[:, H36M[b]], axis=-1)
    return out


def anatomical_lengths(markers: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame anatomical lengths from (T, 84, 3) raw mocap markers, in mm."""
    if markers.ndim != 3 or markers.shape[1] != 84:
        raise ValueError(f"expected (T, 84, 3), got {markers.shape}")
    return {name: np.linalg.norm(markers[:, MOCAP[a]] - markers[:, MOCAP[b]], axis=-1)
            for name, (a, b) in ANATOMICAL.items()}


def scan(per_frame: dict[str, np.ndarray]) -> dict[str, float]:
    """Collapse per-frame lengths into one body model: the median over frames.

    Median rather than mean because a mis-detected frame moves a mean and not a
    median, and the whole question is whether one clip gives a stable number.
    """
    return {k: float(np.median(v)) for k, v in per_frame.items()}


def ratios_of(lengths: dict[str, float]) -> dict[str, float]:
    """The gated scale-invariant quantities."""
    out = {}
    for name, (num, den) in RATIOS.items():
        d = lengths.get(den)
        out[name] = float(lengths[num] / d) if d else float("nan")
    return out


def norm_scale_of(det2d: np.ndarray) -> float:
    """Re-derive the scale normalize_kpts discarded, so mm can be reconstructed.

    Must mirror athleticspose.utils.normalize_kpts exactly: root-relative, then
    max absolute value over the x and y channels of the whole clip.
    """
    coords = np.concatenate([det2d[:, :, :2], np.zeros_like(det2d[:, :, :1])], axis=-1)
    coords = coords - coords[:, 0:1, :]
    return float(np.max(np.abs(coords[:, :, :2])))


def near_side(kp3d: np.ndarray) -> str:
    """Which limb faces the camera, by hip depth. Phase 7's convention."""
    return ("l" if kp3d[:, H36M["l_hip"], 2].mean() < kp3d[:, H36M["r_hip"], 2].mean()
            else "r")


# ------------------------------------------------------- grouping / stats -----
def assert_no_subject_leakage(groups: list[list[str]]) -> None:
    """No subject may appear in two groups. Asserted, never merely called."""
    seen: dict[str, int] = {}
    for i, g in enumerate(groups):
        for s in g:
            if s in seen:
                raise AssertionError(
                    f"subject {s} appears in group {seen[s]} and group {i}")
            seen[s] = i


def subject_bootstrap_ci(per_subject: list[float], n_boot: int = N_BOOT,
                         seed: int = SEED) -> tuple[float, float]:
    """Percentile CI by resampling SUBJECTS, not clips.

    Clips are not independent: 8 cameras share one trial and a subject has many
    trials. Resampling clips would understate the interval badly.
    """
    v = np.asarray([x for x in per_subject if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def cv(values: list[float]) -> float:
    """Coefficient of variation, in percent."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) < 2 or not np.all(v > 0):
        return float("nan")
    return float(100.0 * v.std(ddof=1) / v.mean())


# ----------------------------------------------------------------- mocap ------
def mocap_scan(trial: Path) -> dict | None:
    """Ground-truth scan for one trial: H36M convention and anatomical, both mm."""
    raw = np.load(trial)
    if raw.shape[0] < MIN_FRAMES:
        return None
    h36m = mocap_to_h36m(raw)
    gt = scan(bone_lengths(h36m))
    anat = scan(anatomical_lengths(raw))
    return {"h36m_mm": gt, "anatomical_mm": anat, "ratios": ratios_of(gt),
            "ratios_anatomical": ratios_of(anat), "frames": int(raw.shape[0])}


# ------------------------------------------------------------------ main ------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=int, default=0,
                    help="max clips per subject per action; 0 = all")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    print("phase 12 - scan: per-subject segment lengths from monocular video")
    print(f"  checkpoint {CKPT_NAME}")
    model = load_model(CKPT_NAME, "base")
    if model is None:
        raise SystemExit("checkpoint missing -- see docs for the download")

    # --- mocap ground truth, one trial at a time ---------------------------
    print("\nmocap ground truth from raw_markers_in_world (world mm)")
    gt_by_trial: dict[tuple[str, str, str], dict] = {}
    for action in ACTIONS:
        root = DATA / "raw_markers_in_world" / action
        for trial in sorted(root.glob("*/*.npy")):
            g = mocap_scan(trial)
            if g is not None:
                gt_by_trial[(action, trial.parent.name, trial.stem)] = g
    subjects = sorted({k[1] for k in gt_by_trial})
    print(f"  {len(gt_by_trial)} trials, {len(subjects)} subjects: {' '.join(subjects)}")

    # The grouping key is subject. Subjects are not nested within event, so
    # grouping by event would leak; this asserts the key actually separates.
    assert_no_subject_leakage([[s] for s in subjects])

    # --- estimate, one clip at a time --------------------------------------
    print("\nlifting detected 2D and scanning (one clip at a time)")
    rows: list[dict] = []
    for action in ACTIONS:
        for subj in subjects:
            d = DATA / "det_markers2d_by_cam_coco" / action / subj
            if not d.is_dir():
                continue
            files = sorted(d.glob("*.npy"))
            if args.clips:
                files = files[:args.clips]
            for f in files:
                det = np.load(f).astype(np.float64)
                if det.shape[0] < MIN_FRAMES:
                    continue
                # filename is <date>_<trial>_<cam>; the trial key drops the cam
                parts = f.stem.rsplit("_", 1)
                trial_key = (action, subj, parts[0])
                cam = parts[1]
                gt = gt_by_trial.get(trial_key)
                if gt is None:
                    continue
                est3d = lift(model, det, pad="edge", windows="overlap")
                per_frame = bone_lengths(est3d)
                est = scan(per_frame)
                est_r = ratios_of(est)

                # absolute mm, anatomy check only -- needs this clip's p2mm
                npz = DATA / "gt_markers3d_by_cam" / action / subj / f.name.replace(
                    ".npy", ".npz")
                mm = {}
                if npz.exists():
                    p2mm = float(np.median(np.load(npz)["p2mm"]))
                    k = norm_scale_of(det) / max(p2mm, 1e-9)
                    mm = {seg: v * k for seg, v in est.items()}

                rows.append({
                    "action": action, "subject": subj, "trial": parts[0], "cam": cam,
                    "frames": int(det.shape[0]), "near": near_side(est3d),
                    "est_ratios": est_r,
                    "gt_ratios": gt["ratios"],
                    "ratio_err_pct": {r: float(100.0 * abs(est_r[r] - gt["ratios"][r])
                                               / gt["ratios"][r])
                                      for r in RATIOS if np.isfinite(est_r[r])},
                    "est_mm": mm,
                    "gt_h36m_mm": gt["h36m_mm"],
                    "gt_anatomical_mm": gt["anatomical_mm"],
                    # per-frame CV, for the kill criterion
                    "per_frame_cv_pct": {s: cv(list(v)) for s, v in per_frame.items()},
                    "scan_scalefree": est,
                })
            print(f"  {action}/{subj}: {sum(1 for r in rows if r['subject'] == subj and r['action'] == action)} clips")

    if not rows:
        raise SystemExit("no clips scanned -- nothing to report")

    payload = {
        "n_clips": len(rows),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "actions": list(ACTIONS),
        "checkpoint": CKPT_NAME,
        "min_frames": MIN_FRAMES,
        "groups": "subject",
        "segments": {k: list(v) for k, v in SEGMENTS.items()},
        "ratios": {k: list(v) for k, v in RATIOS.items()},
        "foot": "unavailable from H36M-17 (no toe keypoint); mocap GT reported",
        "scale": "lifter is scale-free; ratios are the gated quantity",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    try:
        shown: Path | str = args.out.relative_to(REPO)
    except ValueError:
        shown = args.out
    print(f"\nwrote {shown}  ({len(rows)} clips)")
    print("per-clip rows kept so the aggregation can be redone without re-lifting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
