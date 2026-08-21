"""The selective-classification curve: what you buy by refusing to answer.

Run: .venv/Scripts/python.exe scripts/phase5d_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "phase5d_selective.json"
OUT = REPO / "figures" / "phase5d_selective.png"

STYLE = {
    "clean mocap, limbsag (upper bound)":
        ("#1f6fb4", "-", "clean mocap (upper bound)"),
    "clean mocap, limbsag + strides (5C best)":
        ("#2e9e5b", "-", "clean mocap + stride distributions"),
    "deployment: wave2, side-on bank, 30 fps":
        ("#c0392b", "-", "deployment: video-derived error, 30 fps"),
    "CONTROL: provenance only (must stay FLAT)":
        ("#8a8a8a", ":", "CONTROL: provenance only (must stay flat)"),
}


def main() -> int:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    base = d["majority_accuracy"]
    fig, ax = plt.subplots(figsize=(9.0, 5.4))

    for key, (colour, ls, label) in STYLE.items():
        rows = d["curves"][key]
        x = np.array([r["coverage"] for r in rows]) * 100
        yv = np.array([r["accuracy"] for r in rows])
        cis = np.array([r["accuracy_ci"] for r in rows])
        ax.fill_between(x, cis[:, 0], cis[:, 1], color=colour, alpha=0.10, lw=0)
        ax.plot(x, yv, ls, color=colour, lw=2.2, marker="o", ms=4.5,
                label=label)

    ax.axhline(base, color="0.2", lw=1.3, ls="--")
    # x is reversed (100% on the left), so ha="left" makes the label run
    # inward from the left edge rather than off the figure.
    ax.text(101, base + 0.005, f"always guess the majority side ({base:.3f})",
            fontsize=8.5, color="0.2", ha="left")

    ax.set_xlabel("coverage — share of scans the system answers at all (%)")
    ax.set_ylabel("accuracy among the scans it answered")
    ax.set_title("Overstride: what abstention buys on the injured-limb task",
                 fontsize=12)
    ax.set_xlim(103, 5)          # reversed: full coverage on the left
    ax.set_ylim(0.40, 0.92)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.text(0.012, 0.015,
             "Within-subject task: which limb is injured, given an injured "
             "runner. Not injury screening. 818 sessions / 675 subjects, "
             "StratifiedGroupKFold on sub_id, 25 folds.\n"
             "Selection is by the model's own confidence |p-0.5| within each "
             "test fold; labels are never used to select, and coverage is a "
             "design parameter rather than a fitted threshold.\n"
             "Bands are cross-fold 95% intervals; they widen sharply at low "
             "coverage because only ~16 sessions per fold remain.\n"
             "The grey control curve must stay flat for the rest to mean "
             "anything. It does.",
             fontsize=7.2, color="0.35", va="bottom", linespacing=1.5)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.235)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
