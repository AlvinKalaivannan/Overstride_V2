"""Phase 11 (B1) step 2 -- does the lifter transfer to another lab's data?

THE QUESTION, AND WHAT IT IS NOT
--------------------------------
`docs/REVIEW.md` §7.6 records that every accuracy number in this project rests on
one dataset. This is the first measurement taken on an independent one:
AthletePose3D (Yeung et al., CVSports at CVPR 2025) -- different lab, different
athletes, 120 fps, four calibrated cameras.

**This does NOT validate the headline 3.4 deg.** That figure is a DETECTED-2D
number. AthletePose3D's shipped clips carry GROUND-TRUTH 2D: `data_input`'s x and
y are bit-identical to `data_label`'s, with the third channel a constant 1.0. So
what is measured here is the LIFTING STAGE ALONE, under perfect 2D input. The
detector's contribution was measured separately in phase 7 (+0.45 deg at full n),
and validating the two together needs raw video, which Google Drive has so far
refused to serve.

Reported as such, and it is still worth having: the error budget has never been
decomposed on data this project did not also train on.

MATCHED CONDITIONS, OR THE COMPARISON MEANS NOTHING
---------------------------------------------------
The `ath-gt` checkpoint is used -- the one trained for GROUND-TRUTH 2D input.
Pairing GT 2D with the `det-ft` weights is exactly the checkpoint/detector
mismatch phase 7 caught, which cost 1.67 deg of imaginary error.

Phase 3 never ran a GT-2D condition, so the AthleticsPose comparator does not
exist and is produced here. Both datasets go through the same checkpoint, the same
`lift()`, and the same `sagittal_angles()`, so the only thing differing is the
data.

WHY THE ANGLES ARE COMPARABLE ACROSS THE TWO REPRESENTATIONS
------------------------------------------------------------
AthletePose3D's `data_label` is an image-normalised perspective representation --
its bone lengths vary at CV 6-12%, which a rigid bone cannot do. That looked
disqualifying. But the quantity that matters for angles is the WITHIN-FRAME bone
length RATIO, and AthleticsPose's own ground truth is no better:

    AP3D data_label        ratio CV  2.4 - 5.2 %
    AthleticsPose h36m     ratio CV  2.4 - 6.5 %

(One AthleticsPose file shows thigh and shank CV identical at 27.24 %, the
signature of the per-frame `p2mm` uniform scale -- which cancels in angles.) The
two are on equal footing, so the cross-dataset comparison is fair. Positional
quantities would NOT be, and use `joint_3d_camera` instead.

Verified separately: `normalize_kpts(pixels) == normalize_kpts(image-normalised)`
to 2.2e-16, so AP3D's pre-normalisation passes through `lift()` harmlessly.

PRE-DECLARED READING, fixed before running:

    AP3D MAE within 1.5x AthleticsPose MAE   the lifter transfers
    1.5 - 3x                                 degrades; report factor and joint
    > 3x                                     does not transfer; 3.4 deg is
                                             dataset-specific

Per-subject, never pooled: three running subjects is exactly the case where a
frame-count-weighted figure overstates precision.

Run: .venv/Scripts/python.exe scripts/phase11_lifter_transfer.py [--limit N]
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
AP = REPO / "data" / "athleticspose"
sys.path.insert(0, str(AP / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import DATA, TEST_SUBJECTS, load_model  # noqa: E402
from phase7_inference_fix import lift  # noqa: E402

AP3D_ZIP = REPO / "data" / "athletepose3d" / "pose_3d.zip"
CLIP_INDEX = REPO / "data" / "derived" / "ap3d_clip_index.json"
OUT = REPO / "results" / "phase11_lifter_transfer.json"

CKPT = "motionagformer-b-ath-gt-v1.ckpt"     # GT-2D input -> GT-2D weights
ACTIONS = ["running", "sprint"]
JOINTS = ("hip", "knee")                      # ankle needs a toe; H36M-17 has none
BAND_TRANSFERS, BAND_DEGRADES = 1.5, 3.0


def near_side(gt3d: np.ndarray) -> str:
    """Which limb faces the camera, by hip depth. Matches phase 7's convention."""
    return ("l" if gt3d[:, H36M["l_hip"], 2].mean() < gt3d[:, H36M["r_hip"], 2].mean()
            else "r")


def clip_mae(model, in2d: np.ndarray, gt3d: np.ndarray) -> dict[str, float] | None:
    """Sagittal MAE for one clip: lift the 2D, compare angles against the 3D."""
    est = lift(model, in2d, pad="edge", windows="overlap")
    side = near_side(gt3d)
    ga, ea = sagittal_angles(gt3d, side), sagittal_angles(est, side)
    out = {}
    for j in JOINTS:
        d = np.abs(ea[j] - ga[j])
        if not np.all(np.isfinite(d)):
            return None
        out[j] = float(d.mean())
    return out


def run_athleticspose(model, limit: int | None) -> list[dict]:
    """The internal comparator: same checkpoint, same code path, own dataset."""
    rows = []
    for action in ACTIONS:
        for subj in TEST_SUBJECTS:
            d = DATA / "gt_markers2d_by_cam" / action / subj
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.npy")):
                gt = DATA / "gt_markers3d_by_cam" / action / subj / f"{f.stem}.npz"
                if not gt.exists():
                    continue
                in2d = np.load(f).astype(float)
                gt3d = np.asarray(np.load(gt)["markers_h36m"], dtype=float)
                n = min(len(in2d), len(gt3d))
                if n < 20:
                    continue
                m = clip_mae(model, in2d[:n], gt3d[:n])
                if m:
                    rows.append({"dataset": "AthleticsPose", "subject": subj,
                                 "action": action, "clip": f.stem, **m})
                if limit and len(rows) >= limit:
                    return rows
    return rows


def run_ap3d(model, limit: int | None) -> list[dict]:
    """AthletePose3D running clips, selected through the phase 11 clip index."""
    idx = json.loads(CLIP_INDEX.read_text())["clips"]
    running = {k: v for k, v in idx.items() if v["action"] == "rm"}
    z = zipfile.ZipFile(AP3D_ZIP)
    rows = []
    for k, v in sorted(running.items()):
        with z.open(v["zip_entry"]) as f:
            c = pickle.load(f)
        in2d = np.asarray(c["data_input"], dtype=float)
        gt3d = np.asarray(c["data_label"], dtype=float)
        m = clip_mae(model, in2d, gt3d)
        if m:
            rows.append({"dataset": "AthletePose3D", "subject": v["subject"],
                         "action": "running", "clip": k,
                         "camera": v["cameraid"], **m})
        if limit and len(rows) >= limit:
            break
    return rows


def summarise(rows: list[dict], label: str) -> dict:
    print(f"\n=== {label}: {len(rows)} clips / "
          f"{len({r['subject'] for r in rows})} subjects ===")
    print(f"  {'subject':<10}{'clips':>7}" + "".join(f"{j:>10}" for j in JOINTS)
          + f"{'mean':>10}")
    per_subject = {}
    for s in sorted({r["subject"] for r in rows}):
        sub = [r for r in rows if r["subject"] == s]
        vals = {j: float(np.median([r[j] for r in sub])) for j in JOINTS}
        mean = float(np.mean(list(vals.values())))
        per_subject[s] = {"n_clips": len(sub), **vals, "mean": mean}
        print(f"  {s:<10}{len(sub):>7}"
              + "".join(f"{vals[j]:>10.2f}" for j in JOINTS) + f"{mean:>10.2f}")
    # Across SUBJECTS, not clips -- three subjects must not be weighted by how
    # many clips each happens to contribute.
    subj_means = [v["mean"] for v in per_subject.values()]
    agg = float(np.mean(subj_means))
    print(f"  {'-> across subjects':<10}{'':>7}{'':>20}{agg:>10.2f} deg"
          f"   (sd {np.std(subj_means):.2f})")
    return {"n_clips": len(rows), "per_subject": per_subject,
            "across_subject_mean_deg": agg,
            "across_subject_sd_deg": float(np.std(subj_means))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap clips per dataset")
    args = ap.parse_args()
    limit = args.limit or None

    if not CLIP_INDEX.exists():
        print(f"ERROR: {CLIP_INDEX} missing. Run phase11_clip_index.py first.")
        return 2

    print(f"loading {CKPT} (ground-truth-2D weights, matched to this input)")
    model = load_model(CKPT, "base")
    if model is None:
        return 2

    with torch.no_grad():
        a_rows = run_athleticspose(model, limit)
        b_rows = run_ap3d(model, limit)

    if not a_rows or not b_rows:
        print("one dataset produced no clips")
        return 2

    a = summarise(a_rows, "AthleticsPose (own data, held-out subjects)")
    b = summarise(b_rows, "AthletePose3D (independent lab, running)")

    ratio = b["across_subject_mean_deg"] / a["across_subject_mean_deg"]
    verdict = ("the lifter TRANSFERS" if ratio <= BAND_TRANSFERS else
               "DEGRADES on unseen data" if ratio <= BAND_DEGRADES else
               "does NOT transfer -- 3.4 deg is dataset-specific")
    print(f"\n=== transfer ===")
    print(f"  AthleticsPose {a['across_subject_mean_deg']:.2f} deg  ->  "
          f"AthletePose3D {b['across_subject_mean_deg']:.2f} deg")
    print(f"  ratio {ratio:.2f}x   (pre-declared: <={BAND_TRANSFERS} transfers, "
          f"<={BAND_DEGRADES} degrades, else fails)")
    print(f"  VERDICT: {verdict}")
    print(f"\n  NOTE: ground-truth 2D input on both sides. This measures the")
    print(f"  LIFTING STAGE ONLY and does not validate the detected-2D 3.4 deg.")

    payload = {"checkpoint": CKPT, "joints": list(JOINTS),
               "input_2d": "ground truth on both datasets",
               "measures": "lifting stage only; not the detected-2D 3.4 deg",
               "athleticspose": a, "athletepose3d": b,
               "ratio": ratio, "verdict": verdict,
               "bands": {"transfers": BAND_TRANSFERS, "degrades": BAND_DEGRADES},
               "ap3d_source": "Yeung et al., CVSports at CVPR 2025; "
                              "non-commercial research only"}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
