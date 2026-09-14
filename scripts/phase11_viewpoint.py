"""Phase 11 (B1) step 3 -- measure the viewpoint penalty instead of extrapolating it.

WHAT THIS REPLACES
------------------
`docs/REVIEW.md` §4 records the far-limb occlusion penalty as **+0.07 deg
extrapolated**. Phase 3B could not measure it: AthleticsPose's held-out clips
cluster tightly around one viewing geometry (median view ratio 0.751, no
near-frontal clips), so the penalty had to be read off a regression extended past
the data. An extrapolated number is exactly the kind this project has elsewhere
insisted on measuring.

AthletePose3D supplies what was missing -- **four calibrated running cameras**,
spread 43.8 to 169.5 degrees apart:

    rm_camera_1  xyz [-2725, -3827, 1371]
    rm_camera_2  xyz [  217,  4101, 1303]
    rm_camera_3  xyz [ 3098,  2847, 1352]
    rm_camera_4  xyz [ 2948, -4321, 1370]

THE METRIC IS PHASE 3B'S, UNCHANGED
-----------------------------------
    view ratio = |z_L - z_R| / ||p_L - p_R||

the fraction of the subject's own hip-to-hip vector lying along the camera depth
axis: 0 = perfectly frontal, 1 = perfectly lateral, and arcsin of it is the
pelvis angle out of the image plane. Being a ratio of two lengths measured in the
same units, it is immune to the scale and representation questions that made
AthletePose3D's `data_label` unusable for positional work.

Using the identical metric is the point -- the result has to be comparable with
phase 3B's, not merely similar in spirit.

A CROSS-CHECK THE ORIGINAL COULD NOT DO
---------------------------------------
The view ratio is derived from the pose. The camera azimuth is derived from the
published extrinsics. They are independent routes to the same geometry, so
agreement between them validates both -- and disagreement would mean the clip
index has mismatched a clip to its camera.

GROUND-TRUTH 2D INPUT, as everywhere in phase 11: this isolates the LIFTING
stage. The detector's own viewpoint sensitivity is not measured here.

Run: .venv/Scripts/python.exe scripts/phase11_viewpoint.py
"""

from __future__ import annotations

import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data" / "athleticspose" / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from phase3_angles import H36M  # noqa: E402
from phase3_infer import load_model  # noqa: E402
from phase11_lifter_transfer import CKPT, JOINTS, clip_mae  # noqa: E402

AP3D_ZIP = REPO / "data" / "athletepose3d" / "pose_3d.zip"
CLIP_INDEX = REPO / "data" / "derived" / "ap3d_clip_index.json"
CAM = REPO / "data" / "athletepose3d" / "cam_param.json"
OUT = REPO / "results" / "phase11_viewpoint.json"

PHASE3B_MEDIAN_RATIO = 0.751          # AthleticsPose, for comparison
PHASE3B_EXTRAPOLATED_PENALTY = 0.07   # deg, the number being replaced


def view_ratio(gt3d: np.ndarray) -> float:
    """Phase 3B's metric, unchanged: hip depth separation over hip width."""
    lh, rh = gt3d[:, H36M["l_hip"]], gt3d[:, H36M["r_hip"]]
    width = np.linalg.norm(lh - rh, axis=-1)
    depth = np.abs(lh[:, 2] - rh[:, 2])
    return float(np.mean(depth / np.maximum(width, 1e-9)))


def camera_azimuths() -> dict[str, float]:
    """Azimuth of each running camera about the vertical, from the extrinsics."""
    d = json.loads(CAM.read_text())
    pos = {k: np.asarray(d[k]["xyz"], dtype=float)
           for k in d if k.startswith("rm_camera_")}
    ctr = np.mean(list(pos.values()), axis=0)
    return {k: float(np.degrees(np.arctan2(v[1] - ctr[1], v[0] - ctr[0])))
            for k, v in pos.items()}


def main() -> int:
    for p in (AP3D_ZIP, CLIP_INDEX, CAM):
        if not p.exists():
            print(f"ERROR: {p} missing")
            return 2

    az = camera_azimuths()
    print("running camera azimuths about the vertical (deg):")
    for k, v in sorted(az.items()):
        print(f"  {k}: {v:+7.1f}")

    idx = json.loads(CLIP_INDEX.read_text())["clips"]
    running = {k: v for k, v in idx.items() if v["action"] == "rm"}
    print(f"\n{len(running)} running clips / "
          f"{len({v['subject'] for v in running.values()})} subjects")

    model = load_model(CKPT, "base")
    if model is None:
        return 2

    z = zipfile.ZipFile(AP3D_ZIP)
    rows = []
    with torch.no_grad():
        for i, (k, v) in enumerate(sorted(running.items())):
            with z.open(v["zip_entry"]) as f:
                c = pickle.load(f)
            in2d = np.asarray(c["data_input"], dtype=float)
            gt3d = np.asarray(c["data_label"], dtype=float)
            m = clip_mae(model, in2d, gt3d)
            if not m:
                continue
            rows.append({"clip": k, "subject": v["subject"],
                         "camera": v["cameraid"],
                         "view_ratio": view_ratio(gt3d),
                         "camera_azimuth": az.get(v["cameraid"], float("nan")),
                         "mean_mae": float(np.mean([m[j] for j in JOINTS])), **m})
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(running)}")

    if not rows:
        print("no clips produced a result")
        return 2
    r = np.array([x["view_ratio"] for x in rows])
    e = np.array([x["mean_mae"] for x in rows])

    print(f"\n=== view ratio coverage ({len(rows)} clips) ===")
    print(f"  AthletePose3D : median {np.median(r):.3f}  "
          f"range [{r.min():.3f}, {r.max():.3f}]  "
          f"= {np.degrees(np.arcsin(np.clip(r, 0, 1))).min():.0f}-"
          f"{np.degrees(np.arcsin(np.clip(r, 0, 1))).max():.0f} deg out of plane")
    print(f"  AthleticsPose : median {PHASE3B_MEDIAN_RATIO:.3f} (phase 3B)")

    print(f"\n=== per camera ===")
    print(f"  {'camera':<14}{'clips':>7}{'view ratio':>12}{'mean MAE':>11}")
    per_cam = {}
    for cam in sorted({x["camera"] for x in rows}):
        sub = [x for x in rows if x["camera"] == cam]
        vr = float(np.median([x["view_ratio"] for x in sub]))
        mae = float(np.median([x["mean_mae"] for x in sub]))
        per_cam[cam] = {"n": len(sub), "view_ratio": vr, "mae": mae,
                        "azimuth": az.get(cam)}
        print(f"  {cam:<14}{len(sub):>7}{vr:>12.3f}{mae:>10.2f}")

    # --- the measurement that replaces the extrapolation -------------------
    slope, intercept = np.polyfit(r, e, 1)
    corr = float(np.corrcoef(r, e)[0, 1])
    # Penalty going from fully lateral (ratio 1) to the dataset's own median.
    penalty = float(slope * (1.0 - np.median(r)))
    print(f"\n=== viewpoint penalty, MEASURED ===")
    print(f"  regression: MAE = {slope:+.3f} x view_ratio {intercept:+.3f}")
    print(f"  correlation r = {corr:+.3f}")
    print(f"  fitted change, median view -> fully lateral: {penalty:+.3f} deg")
    print(f"  phase 3B extrapolated: {PHASE3B_EXTRAPOLATED_PENALTY:+.2f} deg")

    # quartiles, the same presentation phase 3 used
    q = np.quantile(r, [0, .25, .5, .75, 1.0])
    print(f"\n  by view-ratio quartile (phase 3 reported 4.66 -> 2.11 deg):")
    quart = []
    for i in range(4):
        m = (r >= q[i]) & (r <= q[i + 1] if i == 3 else r < q[i + 1])
        if m.sum():
            quart.append({"lo": float(q[i]), "hi": float(q[i + 1]),
                          "n": int(m.sum()), "mae": float(np.median(e[m]))})
            print(f"    [{q[i]:.2f},{q[i+1]:.2f})  {m.sum():>5} clips  "
                  f"{np.median(e[m]):.2f} deg")

    payload = {"n_clips": len(rows), "checkpoint": CKPT,
               "input_2d": "ground truth; lifting stage only",
               "metric": "phase 3B view ratio |z_L-z_R| / ||p_L-p_R||",
               "camera_azimuths": az, "per_camera": per_cam,
               "view_ratio_median": float(np.median(r)),
               "view_ratio_min": float(r.min()), "view_ratio_max": float(r.max()),
               "regression_slope": float(slope),
               "regression_intercept": float(intercept),
               "correlation": corr, "measured_penalty_deg": penalty,
               "phase3b_extrapolated_penalty_deg": PHASE3B_EXTRAPOLATED_PENALTY,
               "quartiles": quart}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
