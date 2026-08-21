"""The degradation curve, assembled from phases 2, 3 and 4.

This is the project's stated deliverable in one panel: how much of the
within-subject injured-limb signal survives as angular error rises, at full
temporal resolution and at 30 fps, with the MEASURED monocular operating points
marked on the same axis.

UNITS -- the one thing that has to be right here
  Phase 2 injected Gaussian noise parameterised by a standard deviation; phase 3
  and phase 4 report mean absolute error. Those are not the same number, and
  putting a sigma on an MAE axis would misplace every phase 2 point.

  The analytic conversion E|N(0,s)| = s*sqrt(2/pi) is also wrong here, because
  phase 2 does not add noise and stop: it decimates, adds noise at the sampled
  points, low-pass filters, then re-interpolates to 101 points. Filtering
  suppresses part of the injected noise while decimation adds reconstruction
  error of its own. So the realized error is MEASURED by re-running phase 2's own
  `degrade()` on the same cohort and taking mean |degraded - clean|, which is
  exactly the quantity phase 4's error bank reports. Cached to
  results/phase2_realized_error.json; delete that file to recompute.

Run: .venv/Scripts/python.exe scripts/phase4_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "figures" / "phase4_degradation.png"
CACHE = RESULTS / "phase2_realized_error.json"

BAR = 0.60          # the pre-registered phase 5 bar, mean AUC
CHANCE = 0.5


def realized_error() -> dict[str, float]:
    """Mean |degraded - clean| in degrees for each (k, sigma) phase 2 plotted.

    Re-runs phase 2's own degrade() on phase 2's own cohort, so the x-axis is a
    measurement rather than an assumption.
    """
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))

    sys.path.insert(0, str(REPO / "scripts"))
    import pandas as pd

    from phase1_cohort import build_cohort
    from phase2_degradation import N_POINTS, degrade
    from phase5_limb import JOINTS3, SAGITTAL_PLANE

    derived = REPO / "data" / "derived"
    idx = pd.read_parquet(derived / "waveforms_var_index.parquet")
    full = pd.read_parquet(derived / "waveforms_index.parquet")
    for f in (idx, full):
        f["sub_id"] = f["sub_id"].astype(str)
        f["filename"] = f["filename"].astype(str)
    ch = (derived / "waveform_channels.txt").read_text(encoding="utf-8").split("\n")
    pos = {c: i for i, c in enumerate(ch)}
    order = {k: i for i, k in enumerate(full["sub_id"] + "|" + full["filename"])}
    sel = np.array([order[k] for k in (idx["sub_id"] + "|" + idx["filename"])])
    mean = np.load(derived / "waveforms_mean.npy")[sel]

    cohort, _oa, _n = build_cohort()
    dvr = pd.read_parquet(derived / "dvr_features.parquet")
    for f in (cohort, dvr):
        f["sub_id"] = f["sub_id"].astype(str)
        f["filename"] = f["filename"].astype(str)
    meta = (idx.merge(cohort, on=["sub_id", "filename"], how="inner")
            .merge(dvr[["sub_id", "filename", "left_STANCE_TIME",
                        "right_STANCE_TIME"]], on=["sub_id", "filename"],
                   how="inner"))
    keep = ((meta["label"] == 1)
            & meta["InjSide"].isin(["Right", "Left"])).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    mean = mean[keep]
    stance = (meta[["left_STANCE_TIME", "right_STANCE_TIME"]].replace(0, np.nan)
              .mean(axis=1).fillna(0.295).to_numpy())

    p2 = json.loads((RESULTS / "phase2_degradation.json").read_text(encoding="utf-8"))
    want = {(s["k"], s["sigma_near"]) for s in p2["summary"]
            if s["sigma_near"] == s["sigma_far"]}

    out = {}
    for k, sigma in sorted(want):
        rng = np.random.default_rng(4242)
        errs = []
        for j in JOINTS3:
            for side in ("L", "R"):
                clean = mean[:, pos[f"ang_{side}_{j}_p{SAGITTAL_PLANE}"], :]
                got = degrade(clean, k, float(sigma), stance, rng)
                errs.append(np.abs(got - clean).mean())
        out[f"k{k}_s{sigma:g}"] = float(np.mean(errs))
        print(f"  realized error: k={k:>3} sigma={sigma:>4.1f} "
              f"-> {out[f'k{k}_s{sigma:g}']:.2f} deg"
              + ("   (decimation only)" if sigma == 0 else ""))
    CACHE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  cached to {CACHE.relative_to(REPO)}")
    return out


def phase2_series(summary: list[dict], k: int,
                  err: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric-noise conditions at one temporal resolution, sorted by error."""
    rows = [s for s in summary
            if s["k"] == k and s["sigma_near"] == s["sigma_far"]]
    rows.sort(key=lambda s: s["sigma_near"])
    x = np.array([err[f"k{s['k']}_s{s['sigma_near']:g}"] for s in rows])
    y = np.array([s["auc_mean"] for s in rows])
    ci = np.array([s["ci"] for s in rows])
    return x, y, ci


def main() -> int:
    p2 = json.loads((RESULTS / "phase2_degradation.json").read_text(encoding="utf-8"))
    p4 = json.loads((RESULTS / "phase4_real_delta.json").read_text(encoding="utf-8"))
    p3 = json.loads((RESULTS / "phase3_angle_errors_labelled.json").read_text(encoding="utf-8"))

    measured = {}
    for m in {r["model"] for r in p3}:
        measured[m] = float(np.mean([r["mae_deg"] for r in p3 if r["model"] == m]))

    err = realized_error()
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    for k, label, colour in ((101, "full resolution (101 pts)", "#1f6fb4"),
                             (9, "30 fps (9 pts)", "#c0392b")):
        x, y, ci = phase2_series(p2["summary"], k, err)
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=colour, alpha=0.10, lw=0)
        ax.plot(x, y, "-o", color=colour, lw=2, ms=5,
                label=f"phase 2, assumed Gaussian noise — {label}")

    # Phase 4: measured residuals injected, plotted at each condition's OWN
    # realized mean |degraded - clean| -- the same quantity as the phase 2 x
    # positions, so the two are directly comparable.
    style = {"mocap": "#1f6fb4", "30fps": "#c0392b"}
    for s in p4["summary"]:
        if s["fps"] not in style or s["features"] != "wave2":
            continue
        ax.errorbar(s["realized_error_deg"], s["auc_mean"],
                    yerr=[[s["auc_mean"] - s["ci"][0]], [s["ci"][1] - s["auc_mean"]]],
                    fmt="D", color=style[s["fps"]], ms=8.5, mfc="white", mew=2.2,
                    capsize=4, elinewidth=1.6, zorder=6)
    ax.plot([], [], "D", color="0.25", ms=8.5, mfc="white", mew=2.2, ls="none",
            label="phase 4, MEASURED monocular error (wave2: hip+knee)")

    for name, x in sorted(measured.items(), key=lambda kv: kv[1]):
        ax.axvline(x, color="0.45", ls=":", lw=1.3, zorder=0)
        ax.text(x + 0.3, 0.474, f"{name.split(' ')[0]}\n{x:.1f}$\\degree$",
                fontsize=8, color="0.3", va="bottom")

    ax.axhline(CHANCE, color="0.2", lw=1.2)
    ax.text(22.6, CHANCE + 0.003, "chance", fontsize=8, color="0.2", ha="right")
    ax.axhline(BAR, color="0.45", ls="--", lw=1.2)
    ax.text(22.6, BAR + 0.003, "pre-registered bar (mean AUC 0.60)",
            fontsize=8, color="0.45", ha="right")

    ax.set_xlabel("mean absolute sagittal joint-angle error (degrees)")
    ax.set_ylabel("AUC — identifying which limb is injured")
    ax.set_title("Overstride: how the injured-limb signal degrades with monocular "
                 "angular error", fontsize=11.5)
    ax.set_ylim(0.455, 0.72)
    ax.set_xlim(-0.5, 23)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    ax.grid(alpha=0.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.text(0.012, 0.015,
             "Within-subject task: which limb is injured, given an injured "
             "runner. Not injury screening.\n"
             "818 unilateral sessions / 675 subjects. StratifiedGroupKFold, "
             "groups=sub_id, 5 splits x 5 seeds = 25 folds shared across every "
             "point; bands and bars are cross-fold 95% intervals.\n"
             "Phase 2 x-positions are the MEASURED mean |degraded - clean| of "
             "its own degrade(), not its nominal sigma; its sweep stopped at "
             "sigma 15.\n"
             "Measured error is a LOWER BOUND: 2D-to-3D lifting only, and no "
             "side-on views exist in the source clips -- so the occlusion "
             "penalty is still unmeasured.",
             fontsize=7.2, color="0.35", va="bottom", linespacing=1.5)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.235)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("measured operating points: "
          + ", ".join(f"{k} {v:.1f} deg" for k, v in sorted(measured.items(),
                                                            key=lambda kv: kv[1])))
    print(f"phase 4 error bank mean |error|: {p4['bank_mae_deg']:.2f} deg")
    for s in p4["summary"]:
        if s["features"] == "wave2":
            print(f"  wave2 @ {s['fps']:>5}: realized "
                  f"{s['realized_error_deg']:.2f} deg -> AUC {s['auc_mean']:.3f}")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
