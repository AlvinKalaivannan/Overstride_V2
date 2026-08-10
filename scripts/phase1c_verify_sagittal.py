"""Phase 1C -- verify the sagittal channel indexing against the archive.

The whole degradation curve rests on "plane 0 is flexion/extension". That comes
from the Cardan order documented in docs/ferber-schema.md S3 (flex/ext ->
ab/adduction -> int/ext rotation), which is a claim from the paper, not an
observation. This checks it against data.

Three sagittal discrete variables ARE live in the stored dv_r:
HIP_EXT_PEAK_ANGLE, KNEE_FLEX_PEAK_ANGLE, ANKLE_DF_PEAK_ANGLE. Each should track
the extremum of the corresponding generated curve. They will not match exactly --
dv_r takes a per-step peak then medians, we average curves across steps first --
but a correct indexing should correlate strongly, and a wrong one should not.

Run: .venv/Scripts/python.exe scripts/phase1c_verify_sagittal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"

# stored dv_r name -> (joint, extremum of the sagittal curve that should track it)
CHECKS = [
    ("KNEE_FLEX_PEAK_ANGLE", "knee", "max"),
    ("HIP_EXT_PEAK_ANGLE", "hip", "max"),
    ("ANKLE_DF_PEAK_ANGLE", "ankle", "min"),
]

# MEASURED, not assumed. docs/ferber-schema.md S3 gives the Cardan order as
# flex/ext -> ab/adduction -> int/ext rotation, which implies index 0. The data
# says otherwise: index 2 tracks the stored sagittal peaks at |r| > 0.99, and
# index 0 at |r| < 0.17. The stored dv_r also uses the opposite sign convention.
SAGITTAL_PLANE = 2
MIN_ABS_R = 0.9


def main() -> int:
    X = np.load(DERIVED / "waveforms_mean.npy")
    index = pd.read_parquet(DERIVED / "waveforms_index.parquet")
    channels = (DERIVED / "waveform_channels.txt").read_text(
        encoding="utf-8").split("\n")
    dvr = pd.read_parquet(DERIVED / "dvr_features.parquet")

    for frame in (index, dvr):
        frame["sub_id"] = frame["sub_id"].astype(str)
        frame["filename"] = frame["filename"].astype(str)
    df = index.merge(dvr, on=["sub_id", "filename"], how="inner")
    pos = {c: i for i, c in enumerate(channels)}
    # Align X rows to the merged frame.
    key = index["sub_id"].astype(str) + "|" + index["filename"].astype(str)
    want = df["sub_id"].astype(str) + "|" + df["filename"].astype(str)
    row = {k: i for i, k in enumerate(key)}
    sel = np.array([row[k] for k in want])
    X = X[sel]
    print(f"{len(df)} sessions aligned\n")

    print(f"{'variable':<26}{'r|p0':>9}{'r|p1':>9}{'r|p2':>9}  best  verdict")
    ok_all = True
    for var, joint, how in CHECKS:
        for side in ("L", "R"):
            stored = df[f"{'left' if side == 'L' else 'right'}_{var}"].to_numpy()
            live = stored != 0
            rs = {}
            for plane in (0, 1, 2):
                curve = X[:, pos[f"ang_{side}_{joint}_p{plane}"], :]
                val = curve.max(axis=1) if how == "max" else curve.min(axis=1)
                m = live & np.isfinite(val) & np.isfinite(stored)
                rs[plane] = (float(np.corrcoef(val[m], stored[m])[0, 1])
                             if m.sum() >= 30 else float("nan"))
            best = max(rs, key=lambda p: abs(rs[p]) if rs[p] == rs[p] else -1)
            good = best == SAGITTAL_PLANE and abs(rs[best]) >= MIN_ABS_R
            ok_all &= good
            print(f"{side}_{var:<24}" + "".join(f"{rs[p]:>9.3f}" for p in (0, 1, 2))
                  + f"{best:>6}  {'OK' if good else 'MISMATCH'}")

    print(f"\n{'PASS' if ok_all else 'FAIL'} — plane {SAGITTAL_PLANE} "
          + (f"is flexion/extension (|r| >= {MIN_ABS_R} against the stored "
             "sagittal peaks). Note the sign convention is inverted relative to "
             "dv_r, which does not affect a classifier."
             if ok_all else "does NOT track the stored sagittal peaks. "
             "Re-index before modelling."))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
