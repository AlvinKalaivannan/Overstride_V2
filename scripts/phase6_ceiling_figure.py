"""Every feature family ever tried on the injured-limb task, in one panel.

Phases 5, 5B and 5C attacked this task with angle differences, stride summaries,
velocities, a different clinical measurement modality, stride distributions and
per-condition splits. This plots all of them against the pre-registered 0.60 bar
so the flatness is visible at a glance: everything that carries any signal lands
in a narrow band around 0.61, and the negative controls sit at chance.

Reads existing result JSONs only -- nothing is refitted here.

Run: .venv/Scripts/python.exe scripts/phase6_ceiling_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
OUT = REPO / "figures" / "phase6_ceiling.png"

BAR = 0.60
CHANCE = 0.5

CTRL = "#9a9a9a"
KIN = "#1f6fb4"
PROBE = "#7b4fa8"
VIDEO = "#c0392b"


def load(name: str) -> dict:
    return json.loads((RES / name).read_text(encoding="utf-8"))


def main() -> int:
    p5 = load("phase5_limb.json")["results"]
    p5b = load("phase5b_ceiling.json")
    p5c = load("phase5c_features.json")
    p4 = load("phase4_real_delta.json")

    def r5(k):
        return p5[k]["auc_mean"], p5[k]["auc_ci"]

    rows: list[tuple] = []

    rows.append(("NEGATIVE CONTROLS", None, None, None))
    for label, key in (("provenance only", "NEG provenance"),
                       ("demographics only", "NEG demographics"),
                       ("file structure", "NEG structure"),
                       ("dominant leg", "DominantLeg")):
        a, c = r5(key)
        rows.append((label, a, c, CTRL))

    rows.append(("KINEMATIC FAMILIES  (clean mocap)", None, None, None))
    fam = [("stride distributions + sagittal", None), ("stride distributions", None)]
    for nm, _ in fam:
        e = next(x for x in p5c["feature_sets"] if nm.split(" +")[0] in x["name"]
                 and ("+" in x["name"]) == ("+" in nm))
        rows.append((nm, e["auc_mean"], e["ci"], KIN))
    for label, key in (("sagittal 3 channels (phase 5 winner)", "limbsag_mean [matched]"),
                       ("9 channels", "limb9_mean [matched]"),
                       ("15 channels", "limb15_mean [matched]"),
                       ("stride quartiles", "limb9_dist"),
                       ("mean + SD + quartiles", "limb9_all"),
                       ("stride-to-stride SD only", "limb9_sd")):
        a, c = r5(key)
        rows.append((label, a, c, KIN))
    e = next(x for x in p5c["feature_sets"] if x["name"] == "limbsag + limbvel")
    rows.append(("sagittal + velocities", e["auc_mean"], e["ci"], KIN))
    e = next(x for x in p5c["feature_sets"] if x["name"].startswith("limbvel "))
    rows.append(("joint velocities (24 channels)", e["auc_mean"], e["ci"], KIN))

    rows.append(("CEILING PROBE  (not video-recoverable)", None, None, None))
    rows.append(("clinical dv_r + sagittal", p5b["D2"]["combined_auc"],
                 p5b["D2"]["combined_ci"], PROBE))
    rows.append(("39 clinical metrics (dv_r)", p5b["D2"]["dvr_auc"],
                 p5b["D2"]["dvr_ci"], PROBE))

    rows.append(("VIDEO-CONSTRAINED  (what a camera gives)", None, None, None))
    rows.append(("hip+knee, clean mocap", p4["clean"]["wave2"],
                 p4["results"]["wave2 clean (hip+knee)"]["auc_ci"], VIDEO))
    s = next(x for x in p4["summary"] if x["features"] == "wave2"
             and x["bank"] == "lateral only" and x["fps"] == "30fps")
    rows.append(("hip+knee + measured error, 30 fps", s["auc_mean"], s["ci"], VIDEO))

    fig, ax = plt.subplots(figsize=(9.6, 8.8))
    ypos, labels, ticks = [], [], []
    y = 0.0
    for label, auc, cint, colour in rows:
        if auc is None:
            y -= 0.85
            ax.text(0.398, y, label, fontsize=8.6, fontweight="bold",
                    color="0.25", va="center")
            ticks.append(y)
            labels.append("")
            y -= 0.55
            continue
        lo, hi = cint
        ax.plot([lo, hi], [y, y], color=colour, lw=2.0, alpha=0.55,
                solid_capstyle="round")
        ax.plot([auc], [y], "o", color=colour, ms=7.5, mfc="white", mew=2.0,
                zorder=5)
        ax.text(hi + 0.006, y, f"{auc:.3f}", fontsize=7.8, color=colour,
                va="center")
        ticks.append(y)
        labels.append(label)
        ypos.append(y)
        y -= 1.0

    ax.axvline(CHANCE, color="0.2", lw=1.3)
    ax.axvline(BAR, color="0.45", lw=1.3, ls="--")
    ax.text(CHANCE, y - 0.4, "chance", fontsize=8.5, color="0.2", ha="center",
            va="top")
    ax.text(BAR, y - 0.4, "pre-registered bar", fontsize=8.5, color="0.45",
            ha="center", va="top")

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_ylim(y - 1.4, 0.7)
    ax.set_xlim(0.39, 0.78)
    ax.set_xlabel("AUC — identifying which limb is injured  (cross-fold 95% CI)")
    ax.set_title("Every feature family tried, and the ceiling none of them clears",
                 fontsize=12.5)
    ax.grid(axis="x", alpha=0.18)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.text(0.012, 0.014,
             "Within-subject task: which limb is injured, given an injured "
             "runner. Not injury screening. 818 unilateral sessions / 675 "
             "subjects, chance 0.517.\n"
             "StratifiedGroupKFold, groups=sub_id, 5 splits x 5 seeds = 25 "
             "folds shared across every row. Sources: results/phase4, phase5, "
             "phase5b, phase5c JSONs.\n"
             "The clinical dv_r probe uses frontal/transverse variables a "
             "side-on camera cannot recover; it is a diagnostic, not a "
             "candidate feature set.",
             fontsize=7.2, color="0.35", va="bottom", linespacing=1.5)

    fig.subplots_adjust(left=0.30, right=0.975, top=0.95, bottom=0.155)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT.relative_to(REPO)}  ({len(ypos)} families)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
