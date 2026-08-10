"""Phase 5 -- across-stride quantile curves, per limb.

CLAUDE.md names "within-session stride distributions" as a phase 5 component.
Phase 1D already built mean and SD curves; SD turned out to be the best of the
kinematic families in 3 of 5 conditions, which is weak evidence that the shape of
the stride distribution carries more than its centre. This adds the quartiles.

Output layout matches waveforms_mean.npy / waveforms_sd.npy exactly -- (n, 54,
101), same channel order, same row order as waveforms_var_index.parquet -- so the
arrays are interchangeable downstream.

Run: .venv/Scripts/python.exe scripts/phase5_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1d_features import ANG, MIN_STEPS, SD_CHANNELS, VEL  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
STEPS = DERIVED / "waveforms_steps"

QUANTILES = {"q25": 25, "q50": 50, "q75": 75}


def main() -> int:
    vidx = pd.read_parquet(DERIVED / "waveforms_var_index.parquet")
    vidx["sub_id"] = vidx["sub_id"].astype(str)
    vidx["filename"] = vidx["filename"].astype(str)
    print(f"index: {len(vidx)} sessions (MIN_STEPS={MIN_STEPS} already applied)")

    out = {k: [] for k in QUANTILES}
    missing = []

    for r in vidx.itertuples():
        path = STEPS / f"{r.sub_id}__{r.session}.mat"
        if not path.exists():
            missing.append(f"{r.sub_id}/{r.session}")
            continue
        m = loadmat(path, squeeze_me=False)
        per_q = {k: [] for k in QUANTILES}
        for name, joints in (("ang", ANG), ("vel", VEL)):
            for j in joints:
                a = m[f"{name}_{j}"]                    # (101, n_steps, 3)
                for k, q in QUANTILES.items():
                    per_q[k].append(np.nanpercentile(a, q, axis=1).T)  # (3, 101)
        for k in QUANTILES:
            out[k].append(np.concatenate(per_q[k], axis=0).astype(np.float32))

    if missing:
        raise SystemExit(f"{len(missing)} curve files missing, e.g. {missing[:3]}"
                         " -- the index and the step files disagree; investigate.")

    for k in QUANTILES:
        arr = np.stack(out[k]).astype(np.float32)
        assert arr.shape == (len(vidx), len(SD_CHANNELS), 101), arr.shape
        np.save(DERIVED / f"waveforms_{k}.npy", arr)
        print(f"{k}: {arr.shape} -> waveforms_{k}.npy ({arr.nbytes / 1e6:.1f} MB)"
              f"  range [{np.nanmin(arr):.2f}, {np.nanmax(arr):.1f}]")

    # Sanity: the median curve should sit between the quartiles everywhere.
    q25 = np.load(DERIVED / "waveforms_q25.npy")
    q50 = np.load(DERIVED / "waveforms_q50.npy")
    q75 = np.load(DERIVED / "waveforms_q75.npy")
    bad = int(((q25 > q50 + 1e-4) | (q50 > q75 + 1e-4)).sum())
    print(f"\nquantile ordering violations (q25<=q50<=q75): {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
