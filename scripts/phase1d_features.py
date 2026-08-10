"""Phase 1D -- stride-to-stride variability and left/right asymmetry.

Everything modelled so far used session-MEAN curves, which discard two things by
construction:

  1. Within-session variability. Two runners can have identical mean curves and
     very different stride-to-stride consistency, and variability is a documented
     injury correlate.
  2. Left/right asymmetry. A mean curve per side exists, but no model has been
     given the difference between them.

Both are computed from the per-step curves already generated in phase 1C
(data/derived/waveforms_steps/, ~1.0 GB). Regenerating those means re-running
~1.7 h of MATLAB, which is why they were written once.

ASYMMETRY AND LEAKAGE
---------------------
Asymmetry is computed as |left - right| per channel. It is side-AGNOSTIC: it
never consults `InjSide`. That is deliberate. Aligning features to the injured
side would encode the label, because uninjured subjects have no injured side --
the alignment itself would separate the classes. |L-R| carries the physiological
motivation (unilateral injury plausibly shows greater asymmetry) without ever
looking at which side is hurt.

Run: .venv/Scripts/python.exe scripts/phase1d_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
STEPS = DERIVED / "waveforms_steps"

# An SD over fewer than this many strides is not a variability estimate, it is
# noise. Median retained is 29; this costs 24 of 1745 sessions (1.4%).
MIN_STEPS = 8

ANG = ["L_ankle", "L_knee", "L_hip", "L_foot", "L_pelvis",
       "R_ankle", "R_knee", "R_hip", "R_foot", "R_pelvis"]
VEL = ["L_ankle", "L_knee", "L_hip", "L_pelvis",
       "R_ankle", "R_knee", "R_hip", "R_pelvis"]
JOINTS3 = ["ankle", "knee", "hip"]

SD_CHANNELS = ([f"sd_ang_{j}_p{p}" for j in ANG for p in range(3)]
               + [f"sd_vel_{j}_p{p}" for j in VEL for p in range(3)])
ASYM_CHANNELS = [f"asym_{j}_p{p}" for j in JOINTS3 for p in range(3)]


def main() -> int:
    index = pd.read_parquet(DERIVED / "waveforms_index.parquet")
    index["sub_id"] = index["sub_id"].astype(str)
    index["filename"] = index["filename"].astype(str)
    print(f"index: {len(index)} sessions")

    sd_rows, asym_rows, kept_rows, dropped = [], [], [], []

    for r in index.itertuples():
        path = STEPS / f"{r.sub_id}__{r.session}.mat"
        if not path.exists():
            dropped.append((r.sub_id, r.session, "no curve file"))
            continue
        m = loadmat(path, squeeze_me=False)

        n_l = m["ang_L_ankle"].shape[1]
        n_r = m["ang_R_ankle"].shape[1]
        if min(n_l, n_r) < MIN_STEPS:
            dropped.append((r.sub_id, r.session,
                            f"only {min(n_l, n_r)} strides (< {MIN_STEPS})"))
            continue

        # --- across-step SD, same (54, 101) layout as the means --------------
        sd = []
        for name, joints in (("ang", ANG), ("vel", VEL)):
            for j in joints:
                a = m[f"{name}_{j}"]                      # (101, n_steps, 3)
                sd.append(np.nanstd(a, axis=1, ddof=1).T)  # (3, 101)
        sd = np.concatenate(sd, axis=0).astype(np.float32)

        # --- |L - R| on the step-mean curves, 3 joints x 3 planes ------------
        asym = []
        for joint in JOINTS3:
            left = np.nanmean(m[f"ang_L_{joint}"], axis=1).T   # (3, 101)
            right = np.nanmean(m[f"ang_R_{joint}"], axis=1).T
            asym.append(np.abs(left - right))
        asym = np.concatenate(asym, axis=0).astype(np.float32)

        if sd.shape != (len(SD_CHANNELS), 101) or \
                asym.shape != (len(ASYM_CHANNELS), 101):
            dropped.append((r.sub_id, r.session,
                            f"shapes {sd.shape} / {asym.shape}"))
            continue

        sd_rows.append(sd)
        asym_rows.append(asym)
        kept_rows.append({"sub_id": r.sub_id, "filename": r.filename,
                          "session": r.session, "n_steps_L": n_l,
                          "n_steps_R": n_r,
                          "stride_imbalance": abs(n_l - n_r) / max(n_l, n_r)})

    SD = np.stack(sd_rows).astype(np.float32)
    ASYM = np.stack(asym_rows).astype(np.float32)
    kept = pd.DataFrame(kept_rows)

    np.save(DERIVED / "waveforms_sd.npy", SD)
    np.save(DERIVED / "waveforms_asym.npy", ASYM)
    kept.to_parquet(DERIVED / "waveforms_var_index.parquet", index=False)
    (DERIVED / "waveform_sd_channels.txt").write_text("\n".join(SD_CHANNELS),
                                                      encoding="utf-8")
    (DERIVED / "waveform_asym_channels.txt").write_text("\n".join(ASYM_CHANNELS),
                                                        encoding="utf-8")

    print(f"\nSD   {SD.shape} -> waveforms_sd.npy ({SD.nbytes / 1e6:.1f} MB)")
    print(f"ASYM {ASYM.shape} -> waveforms_asym.npy ({ASYM.nbytes / 1e6:.1f} MB)")
    print(f"kept {len(kept)} / {len(index)}   dropped {len(dropped)}")
    for d in dropped[:12]:
        print(f"    {d[0]}/{d[1]}: {d[2]}")
    if len(dropped) > 12:
        print(f"    ... and {len(dropped) - 12} more")

    print(f"\nstrides retained: median {kept[['n_steps_L', 'n_steps_R']].min(axis=1).median():.0f}"
          f", min {kept[['n_steps_L', 'n_steps_R']].min(axis=1).min()}")
    print(f"SD   range [{np.nanmin(SD):.3f}, {np.nanmax(SD):.1f}]  "
          f"NaN channels {int(np.isnan(SD).any(axis=(0, 2)).sum())}/{len(SD_CHANNELS)}")
    print(f"ASYM range [{np.nanmin(ASYM):.3f}, {np.nanmax(ASYM):.1f}]  "
          f"NaN channels {int(np.isnan(ASYM).any(axis=(0, 2)).sum())}/{len(ASYM_CHANNELS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
