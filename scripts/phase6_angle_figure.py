"""What 3.4 degrees of error actually looks like.

Every error figure in this project is a number in a table. This plots the
recovered sagittal hip and knee angles against marker-based ground truth for a
few held-out clips, so the quantity the whole degradation curve rests on is
visible rather than asserted.

Side-on clips are chosen deliberately: phase 3B showed that is both the geometry
Overstride prescribes and the one the estimator handles best.

Runs the fine-tuned checkpoint on 3 clips only -- about a minute on CPU.

Run: .venv/Scripts/python.exe scripts/phase6_angle_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
AP = REPO / "data" / "athleticspose"
sys.path.insert(0, str(AP / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import DATA, TEST_SUBJECTS, load_model, predict  # noqa: E402
from phase3b_viewpoint import clip_geometry  # noqa: E402

OUT = REPO / "figures" / "phase6_angle_recovery.png"
N_CLIPS = 3
GT = "#111111"
EST = "#c0392b"


def main() -> int:
    geo = clip_geometry()
    # Most side-on clips available, longest first for a readable trace.
    cands = sorted(((g["view_ratio"], a, s, c) for (a, s, c), g in geo.items()),
                   reverse=True)
    model = load_model("motionagformer-b-ath-det-ft-v1.ckpt", "base")
    if model is None:
        raise SystemExit("fine-tuned checkpoint missing")

    picked = []
    for ratio, action, subj, stem in cands:
        det = DATA / "det_markers2d_by_cam_ft" / action / subj / f"{stem}.npy"
        gtf = DATA / "gt_markers3d_by_cam" / action / subj / f"{stem}.npz"
        if not (det.exists() and gtf.exists()):
            continue
        d2 = np.load(det).astype(np.float64)
        g3 = np.asarray(np.load(gtf)["markers_h36m"])
        n = min(len(d2), len(g3))
        if n < 40:
            continue
        with torch.no_grad():
            est = predict(model, d2[:n], g3[:n])
        g3 = g3[:n]
        near = ("l" if g3[:, H36M["l_hip"], 2].mean()
                < g3[:, H36M["r_hip"], 2].mean() else "r")
        if any(q["subj"] == subj for q in picked):
            continue          # one clip per subject, so the panels are not
                              # three views of the same runner
        picked.append({"ratio": ratio, "subj": subj, "stem": stem, "n": n,
                       "est": sagittal_angles(est, near),
                       "gt": sagittal_angles(g3, near), "limb": near})
        print(f"  {subj}/{stem}  ratio {ratio:.2f}  {n} frames  near limb {near}")
        if len(picked) == N_CLIPS:
            break

    fig, axes = plt.subplots(2, N_CLIPS, figsize=(11.2, 5.9), sharex="col")
    for col, p in enumerate(picked):
        t = np.arange(p["n"])
        for row, joint in enumerate(("hip", "knee")):
            ax = axes[row, col]
            gt, es = p["gt"][joint], p["est"][joint]
            mae = float(np.nanmean(np.abs(es - gt)))
            ax.fill_between(t, gt, es, color=EST, alpha=0.16, lw=0)
            ax.plot(t, gt, color=GT, lw=1.9, label="marker mocap (ground truth)")
            ax.plot(t, es, color=EST, lw=1.6, ls="--",
                    label="monocular pose estimate")
            lo, hi = np.nanmin([gt, es]), np.nanmax([gt, es])
            ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.26 * (hi - lo))
            ax.text(0.975, 0.955, f"MAE {mae:.1f}°", transform=ax.transAxes,
                    fontsize=9, ha="right", va="top", color=EST,
                    fontweight="bold")
            ax.grid(alpha=0.16)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if col == 0:
                ax.set_ylabel(f"{joint} flexion (°)", fontsize=9.5)
            if row == 0:
                ax.set_title(f"{p['subj']} · side-on view "
                             f"({np.degrees(np.arcsin(min(p['ratio'], 1.0))):.0f}° "
                             "out of plane)", fontsize=9.5)
            if row == 1:
                ax.set_xlabel("frame", fontsize=9)

    axes[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.suptitle("What the degradation curve rests on: recovered vs true "
                 "sagittal angles", fontsize=12.5, y=0.985)
    fig.text(0.008, 0.012,
             "Fine-tuned MotionAGFormer on AthleticsPose held-out subjects "
             "(S11/S13/S16), near limb shown. Mean sagittal error across all "
             "592 held-out clips is 3.4° (hip 2.9°, knee 3.8°).\n"
             "This is 2D-to-3D lifting only — AthleticsPose does not release "
             "the source videos, so video decode and person detection error are "
             "excluded and these curves are a LOWER BOUND on real-world error.",
             fontsize=7.2, color="0.35", va="bottom", linespacing=1.5)
    fig.subplots_adjust(left=0.062, right=0.99, top=0.90, bottom=0.175,
                        hspace=0.18, wspace=0.17)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
