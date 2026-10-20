"""How much is a better 2D detector actually worth? The bound, measured.

The approved plan calls the detector replacement "the single highest-value
change": it carries the largest quantified accuracy term (+0.43 deg near,
+0.42 deg far, paired, both excluding zero), the whole of phase 16's 56-66x
speed problem, and the only unquantified error in the tool's budget. That
reasoning has a hole in it. **Nobody measured the ceiling.** If the lifter's own
error with a PERFECT detector were already 3.4 deg, then replacing the detector
could buy at most 0.1 deg and the plan's highest-value change would be worth
almost nothing.

AthleticsPose ships `gt_markers2d_by_cam` -- the mocap markers projected into
each camera, in the same H36M-17 convention as the detections -- and a
checkpoint trained on that input, `motionagformer-b-ath-gt-v1.ckpt`. So the
ceiling is measurable with data already on disk and no video at all:

  input 2D          checkpoint   what it measures
  ---------------   ----------   -----------------------------------------------
  COCO detections   det-coco     3.51 deg, the deployable baseline
  ground-truth 2D   gt           THE FLOOR: a perfect detector, matched weights
  ground-truth 2D   det-coco     same model, perfect input (unmatched pairing,
                                 reported as such -- pairing input and weights
                                 wrongly is the mistake phase 7 documented)
  blend, alpha=0.5  det-coco     half the detector's error, same model

The blend gives a sensitivity curve: angle error against 2D keypoint error in
pixels. A candidate detector can then be placed on that curve from its 2D error
alone, which is measurable on any dataset with 2D ground truth -- and therefore
WITHOUT the video AthleticsPose does not release.

Near and far limb are reported separately throughout, per the measurement-track
rule. No injury output; this is a measurement of a measurement.

Run: .venv/Scripts/python.exe scripts/phase7_detector_headroom.py [--clips N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AP = REPO / "data" / "athleticspose"
sys.path.insert(0, str(AP / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import DATA, load_model  # noqa: E402
from phase7_inference_fix import clip_list, lift  # noqa: E402
from phase12_scan import subject_bootstrap_ci  # noqa: E402

OUT = REPO / "results" / "phase7_detector_headroom.json"

CKPT = {"coco": "motionagformer-b-ath-det-coco-v1.ckpt",
        "gt": "motionagformer-b-ath-gt-v1.ckpt"}

# (label, 2D source, blend toward ground truth, checkpoint)
#   alpha = 1.0 keeps the detector's own 2D; 0.0 is ground-truth 2D
CONFIGS = [
    ("deployable (COCO det)",      "det", 1.00, "coco"),
    ("half the 2D error",          "det", 0.50, "coco"),
    ("quarter the 2D error",       "det", 0.25, "coco"),
    ("perfect 2D, same weights",   "det", 0.00, "coco"),
    ("perfect 2D, matched weights", "gt", 0.00, "gt"),
]

# the joints the sagittal angles are built from, for the 2D error figure
LOWER = ["pelvis", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
         "thorax"]
PAD, WINDOWS = "edge", "overlap"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=0, help="0 = every held-out clip")
    args = ap.parse_args()

    models = {k: load_model(v, "base") for k, v in CKPT.items()}
    for k, m in models.items():
        if m is None:
            raise SystemExit(f"checkpoint missing: {CKPT[k]}")
        print(f"  {k:5} -> {CKPT[k]}")

    clips = clip_list(args.clips or None)
    print(f"\n=== detector headroom on {len(clips)} held-out clips ===")

    err: dict = {name: {"near": defaultdict(list), "far": defaultdict(list)}
                 for name, *_ in CONFIGS}
    px: dict = {name: [] for name, *_ in CONFIGS}
    rows = []
    for i, (action, subj, stem, _n) in enumerate(clips):
        gt3d = np.asarray(np.load(
            DATA / "gt_markers3d_by_cam" / action / subj / f"{stem}.npz")["markers_h36m"])
        det = np.load(DATA / "det_markers2d_by_cam_coco" / action / subj
                      / f"{stem}.npy").astype(float)
        gt2d_path = DATA / "gt_markers2d_by_cam" / action / subj / f"{stem}.npy"
        if not gt2d_path.exists():
            continue
        gt2d = np.load(gt2d_path).astype(float)
        n = min(len(gt3d), len(det), len(gt2d))
        if n < 20:
            continue
        gt3d, det, gt2d = gt3d[:n], det[:n], gt2d[:n]

        near = ("l" if gt3d[:, H36M["l_hip"], 2].mean()
                < gt3d[:, H36M["r_hip"], 2].mean() else "r")
        far = "r" if near == "l" else "l"
        ga = {"near": sagittal_angles(gt3d, near),
              "far": sagittal_angles(gt3d, far)}

        for name, src, alpha, ck in CONFIGS:
            if src == "gt":
                x = gt2d.copy()
            else:
                # blend the POSITIONS toward truth; keep the detector's own
                # confidence channel, since a better detector is not a more
                # confident one and the model reads that channel
                x = det.copy()
                x[:, :, :2] = gt2d[:, :, :2] + alpha * (det[:, :, :2]
                                                        - gt2d[:, :, :2])
            d2 = np.linalg.norm(x[:, [H36M[j] for j in LOWER], :2]
                                - gt2d[:, [H36M[j] for j in LOWER], :2], axis=-1)
            px[name].append(float(d2.mean()))

            est = lift(models[ck], x, PAD, WINDOWS)
            ea = {"near": sagittal_angles(est, near),
                  "far": sagittal_angles(est, far)}
            row = {"config": name, "action": action, "subject": subj,
                   "clip": stem, "px_err": float(d2.mean())}
            for limb in ("near", "far"):
                for joint in ("hip", "knee"):
                    e = np.abs(ea[limb][joint] - ga[limb][joint])
                    ok = bool(np.all(np.isfinite(e)))
                    row[f"{limb}_{joint}_mae"] = float(e.mean()) if ok else None
                    if ok:
                        err[name][limb][joint].append(
                            (subj, float(e.mean())))
            rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(clips)} clips")

    def agg(pairs: list[tuple[str, float]]) -> dict:
        by: dict[str, list[float]] = defaultdict(list)
        for s, v in pairs:
            by[s].append(v)
        per = {s: float(np.mean(v)) for s, v in sorted(by.items())}
        vals = list(per.values())
        lo, hi = subject_bootstrap_ci(vals)
        return {"pooled_clip_mean": float(np.mean([v for _, v in pairs])),
                "subject_mean": float(np.mean(vals)) if vals else float("nan"),
                "ci95_nominal": [lo, hi], "per_subject": per,
                "n_clips": len(pairs)}

    out = {"configs": {}, "pad": PAD, "windows": WINDOWS,
           "n_clips": len(clips), "checkpoints": CKPT,
           "note": "2D error is the mean pixel distance to projected mocap over "
                   "the eight joints the sagittal angles are built from"}
    for name, src, alpha, ck in CONFIGS:
        entry = {"source": src, "alpha": alpha, "checkpoint": CKPT[ck],
                 "px_err_mean": float(np.mean(px[name])), "limbs": {}}
        for limb in ("near", "far"):
            hip = agg(err[name][limb]["hip"])
            knee = agg(err[name][limb]["knee"])
            both = [(s, v) for s, v in err[name][limb]["hip"]]
            entry["limbs"][limb] = {
                "hip": hip, "knee": knee,
                "mean_pooled": 0.5 * (hip["pooled_clip_mean"]
                                      + knee["pooled_clip_mean"]),
                "mean_subject": 0.5 * (hip["subject_mean"] + knee["subject_mean"]),
                "n_clips": len(both)}
        out["configs"][name] = entry

    print(f"\n=== VERDICT ===")
    print(f"  {'configuration':<28}{'2D px':>8}{'near':>8}{'far':>8}"
          f"{'near vs deployable':>21}")
    base = out["configs"]["deployable (COCO det)"]
    for name, *_ in CONFIGS:
        e = out["configs"][name]
        dn = (e["limbs"]["near"]["mean_pooled"]
              - base["limbs"]["near"]["mean_pooled"])
        print(f"  {name:<28}{e['px_err_mean']:>8.2f}"
              f"{e['limbs']['near']['mean_pooled']:>8.3f}"
              f"{e['limbs']['far']['mean_pooled']:>8.3f}{dn:>+21.3f}")

    floor = out["configs"]["perfect 2D, matched weights"]
    head_near = (base["limbs"]["near"]["mean_pooled"]
                 - floor["limbs"]["near"]["mean_pooled"])
    head_far = (base["limbs"]["far"]["mean_pooled"]
                - floor["limbs"]["far"]["mean_pooled"])
    out["headroom_deg"] = {"near": head_near, "far": head_far}
    print(f"\n  A PERFECT 2D detector is worth at most {head_near:.3f} deg near "
          f"and {head_far:.3f} deg far.")
    print(f"  Everything below that is the lifter's own error, not the "
          f"detector's.")

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    rows_path = OUT.with_name(OUT.stem + "_rows.json")
    rows_path.write_text(json.dumps({"n_rows": len(rows), "rows": rows}, indent=2),
                         encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)} and {rows_path.name} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
