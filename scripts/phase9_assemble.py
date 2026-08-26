"""Phase 9 (A1) step 2 -- assemble walking curve files into a modelling array.

The walking twin of phase1c_assemble.py. Same channel layout, same 101-point
convention, same (n_sessions, 54, 101) float32 output -- written to
`walking_*` filenames so it can never be confused with, or overwrite, the
running arrays that phases 1C-8 are built on.

TWO THINGS THIS ENFORCES THAT THE RUNNING VERSION DID NOT HAVE TO

  UNRESOLVED SESSIONS ARE DROPPED. A session is kept only if the regenerated
  DISCRETE_VARIABLES reproduce the archive's own stored dv_w exactly. That is
  what makes these curves verified rather than assumed. The running batch
  resolved 1745/1745, so the filter never fired there; walking is a path the
  pipeline has not been exercised on, so it is enforced explicitly and the
  count is reported.

  SAMPLING RATE IS CARRIED THROUGH. Walking is a mix of 120 Hz and 200 Hz where
  running was almost entirely 200 Hz (1,822 vs 10). The 101-point stance
  normalisation should absorb that, but filter cutoffs and step detection
  interact with sample rate, so `hz` is kept per session and the analysis
  asserts it is not recoverable from the limb-difference features. Decimation
  is exactly where phase 4's framerate artefact came from.

CHANNEL ORDER is identical to the running arrays:
  0..29   angles     L/R x {ankle, knee, hip, foot, pelvis} x 3 planes
  30..53  velocities L/R x {ankle, knee, hip, pelvis}       x 3 planes

**Plane 2 is flexion/extension, not plane 0.** phase1c_assemble.py's docstring
says plane 0, following docs/ferber-schema.md S3; the emitted array disagrees
and CLAUDE.md records the correction. Measured against the live sagittal dv_r
peaks, plane 2 gives |r| >= 0.99 and plane 0 gives |r| <= 0.17.

Run: .venv/Scripts/python.exe scripts/phase9_assemble.py
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
STEPS = DERIVED / "walk_steps"
MANIFEST = DERIVED / "walk_manifest.csv"

ANG = ["L_ankle", "L_knee", "L_hip", "L_foot", "L_pelvis",
       "R_ankle", "R_knee", "R_hip", "R_foot", "R_pelvis"]
VEL = ["L_ankle", "L_knee", "L_hip", "L_pelvis",
       "R_ankle", "R_knee", "R_hip", "R_pelvis"]

CHANNELS = ([f"ang_{j}_p{p}" for j in ANG for p in range(3)]
            + [f"vel_{j}_p{p}" for j in VEL for p in range(3)])

SAGITTAL_PLANE = 2
MIN_STEPS = 5     # matches the running pipeline's filter


def main() -> int:
    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} not found. Run batch_waveforms_walk.m first.")
        return 2

    man = pd.read_csv(MANIFEST)
    print(f"manifest: {len(man)} rows")
    print(f"  label used: {man['label_used'].value_counts().to_dict()}")
    print(f"  sampling rate: {man['hz'].value_counts().to_dict()}")

    errs = man[man["err"].notna() & (man["err"].astype(str).str.strip() != "")]
    print(f"  errors: {len(errs)}")
    for e in errs["err"].astype(str).head(5):
        print(f"    {e}")

    n_res = int(man["resolved"].sum())
    print(f"  resolved (reproduce stored dv_w exactly): {n_res}/{len(man)} "
          f"({100 * n_res / max(len(man), 1):.1f}%)")
    unresolved = man[man["resolved"] != 1]
    if len(unresolved):
        print(f"  DROPPING {len(unresolved)} unresolved sessions")

    rows, arrays, skipped = [], [], []
    for r in man[man["resolved"] == 1].itertuples():
        path = STEPS / f"{r.sub_id}__{r.session}.mat"
        if not path.exists():
            skipped.append((r.sub_id, r.session, "no curve file"))
            continue
        m = loadmat(path, squeeze_me=False)
        try:
            per_channel, n_steps = [], None
            for name, joints in (("ang", ANG), ("vel", VEL)):
                for j in joints:
                    a = m[f"{name}_{j}"]          # (101, n_steps, 3)
                    if a.ndim != 3 or a.shape[0] != 101:
                        raise ValueError(f"{name}_{j} shape {a.shape}")
                    n_steps = a.shape[1] if n_steps is None else n_steps
                    per_channel.append(np.nanmean(a, axis=1).T)   # (3, 101)
            arr = np.concatenate(per_channel, axis=0).astype(np.float32)
        except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
            skipped.append((r.sub_id, r.session, f"{type(exc).__name__}: {exc}"))
            continue
        if arr.shape != (len(CHANNELS), 101):
            skipped.append((r.sub_id, r.session, f"assembled {arr.shape}"))
            continue
        if n_steps is None or n_steps < MIN_STEPS:
            skipped.append((r.sub_id, r.session, f"only {n_steps} steps"))
            continue
        arrays.append(arr)
        rows.append({"sub_id": str(r.sub_id), "filename": f"{r.session}.json",
                     "session": r.session, "label_used": r.label_used,
                     "resolved": r.resolved, "n_steps": n_steps,
                     "hz": r.hz, "speed_pipeline": r.speed,
                     "eventsflag_mean": r.eventsflag_mean})

    X = np.stack(arrays) if arrays else np.zeros((0, len(CHANNELS), 101), np.float32)
    index = pd.DataFrame(rows)

    np.save(DERIVED / "walking_mean.npy", X)
    index.to_parquet(DERIVED / "walking_index.parquet", index=False)
    (DERIVED / "walking_channels.txt").write_text("\n".join(CHANNELS),
                                                  encoding="utf-8")

    print(f"\nassembled {X.shape} float32 -> walking_mean.npy "
          f"({X.nbytes / 1e6:.1f} MB)")
    print(f"  subjects: {index['sub_id'].nunique() if len(index) else 0}")
    print(f"  skipped {len(skipped)}")
    for s in skipped[:10]:
        print(f"    {s}")
    if len(X):
        nan_ch = int(np.isnan(X).any(axis=(0, 2)).sum())
        print(f"  channels containing any NaN: {nan_ch}/{len(CHANNELS)}")
        print(f"  value range: [{np.nanmin(X):.2f}, {np.nanmax(X):.2f}]")
        print(f"  steps/session: median {index['n_steps'].median():.0f} "
              f"(min {index['n_steps'].min()}, max {index['n_steps'].max()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
