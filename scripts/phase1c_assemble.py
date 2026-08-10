"""Phase 1C -- assemble per-session curve files into modelling arrays.

MATLAB writes one .mat per session holding (101, n_steps, 3) arrays per joint.
This reduces them to a session-mean array for phases 1C-4 and records which
sessions are usable.

Channel layout, fixed here and used everywhere downstream:

  0..29   angles     L/R x {ankle, knee, hip, foot, pelvis} x 3 planes
  30..53  velocities L/R x {ankle, knee, hip, pelvis}       x 3 planes

Plane 0 is flexion/extension (the Cardan order is flex/ext -> ab/adduction ->
rotation, per docs/ferber-schema.md S3), so the sagittal subset -- the channels
recoverable from a single side-on camera -- is plane 0 of ankle/knee/hip.

Run: .venv/Scripts/python.exe scripts/phase1c_assemble.py
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
MANIFEST = DERIVED / "waveform_manifest.csv"

ANG = ["L_ankle", "L_knee", "L_hip", "L_foot", "L_pelvis",
       "R_ankle", "R_knee", "R_hip", "R_foot", "R_pelvis"]
VEL = ["L_ankle", "L_knee", "L_hip", "L_pelvis",
       "R_ankle", "R_knee", "R_hip", "R_pelvis"]

CHANNELS = ([f"ang_{j}_p{p}" for j in ANG for p in range(3)]
            + [f"vel_{j}_p{p}" for j in VEL for p in range(3)])
SAGITTAL = [f"ang_{j}_p0" for j in
            ["L_ankle", "L_knee", "L_hip", "R_ankle", "R_knee", "R_hip"]]


def main() -> int:
    man = pd.read_csv(MANIFEST)
    print(f"manifest: {len(man)} rows")
    print(f"  resolved (reproduce stored dv_r): {int(man['resolved'].sum())}")
    print(f"  label used: {man['label_used'].value_counts().to_dict()}")
    errs = man[man["err"].notna() & (man["err"].astype(str) != "")]
    print(f"  errors: {len(errs)}")
    for e in errs["err"].head(5):
        print(f"    {e}")

    rows, arrays, skipped = [], [], []
    for r in man.itertuples():
        path = STEPS / f"{r.sub_id}__{r.session}.mat"
        if not path.exists():
            skipped.append((r.sub_id, r.session, "no curve file"))
            continue
        m = loadmat(path, squeeze_me=False)
        try:
            per_channel = []
            n_steps = None
            for name, joints in (("ang", ANG), ("vel", VEL)):
                for j in joints:
                    a = m[f"{name}_{j}"]          # (101, n_steps, 3)
                    if a.ndim != 3 or a.shape[0] != 101:
                        raise ValueError(f"{name}_{j} shape {a.shape}")
                    n_steps = a.shape[1] if n_steps is None else n_steps
                    per_channel.append(np.nanmean(a, axis=1).T)  # (3, 101)
            arr = np.concatenate(per_channel, axis=0).astype(np.float32)
        except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
            skipped.append((r.sub_id, r.session, f"{type(exc).__name__}: {exc}"))
            continue
        if arr.shape != (len(CHANNELS), 101):
            skipped.append((r.sub_id, r.session, f"assembled {arr.shape}"))
            continue
        arrays.append(arr)
        rows.append({"sub_id": r.sub_id, "filename": f"{r.session}.json",
                     "session": r.session, "label_used": r.label_used,
                     "resolved": r.resolved, "n_steps": n_steps,
                     "speed_pipeline": r.speed,
                     "eventsflag_mean": r.eventsflag_mean})

    X = np.stack(arrays) if arrays else np.zeros((0, len(CHANNELS), 101), np.float32)
    index = pd.DataFrame(rows)

    np.save(DERIVED / "waveforms_mean.npy", X)
    index.to_parquet(DERIVED / "waveforms_index.parquet", index=False)
    (DERIVED / "waveform_channels.txt").write_text("\n".join(CHANNELS),
                                                   encoding="utf-8")

    print(f"\nassembled {X.shape} float32 -> waveforms_mean.npy "
          f"({X.nbytes / 1e6:.1f} MB)")
    print(f"  channels {len(CHANNELS)}  (sagittal subset: {SAGITTAL})")
    print(f"  skipped {len(skipped)}")
    for s in skipped[:10]:
        print(f"    {s}")
    if len(X):
        nan_ch = np.isnan(X).any(axis=(0, 2)).sum()
        print(f"  channels containing any NaN: {nan_ch}/{len(CHANNELS)}")
        print(f"  value range: [{np.nanmin(X):.2f}, {np.nanmax(X):.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
