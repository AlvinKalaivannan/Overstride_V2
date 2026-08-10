"""Phase 1B -- stream the cohort's session JSONs once and extract two things.

1. `dv_r`: 76 discrete kinematic scalars per side. The cheap read on whether any
   kinematic signal exists, without running the MATLAB pipeline.
2. Structural metadata per session: marker-channel count, `joints` landmark
   count, sampling rate, frame count.

(2) is the reason this runs first. Phase 1A found collection year predicts injury
at AUC 0.720 -- the dataset is a concatenation of studies with injury rates from
0.00 to 0.93. If the marker set also tracks collection wave, then a kinematic
model can score by detecting which lab protocol was used rather than anything
about the runner, and that would contaminate every later phase including the
video work. Cheap to measure now, expensive to retrofit.

Rules honoured (CLAUDE.md):
  - One file open at a time, released before the next. Never all 1,745 at once.
  - $DATA_ROOT read from the environment, read-only.
  - Nothing synthesized. Unreadable files are counted and named, never skipped
    silently or filled in.

Run: .venv/Scripts/python.exe scripts/phase1b_extract.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_cohort import build_cohort, data_root  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DERIVED = REPO / "data" / "derived"
OUT_FEATURES = DERIVED / "dvr_features.parquet"
OUT_STRUCTURE = DERIVED / "session_structure.parquet"


def flatten_dv(dv: object) -> dict[str, float]:
    """dv_r is {'left': {...76 scalars...}, 'right': {...}}. Prefix and flatten.

    Anything that is not a scalar is dropped and counted -- phase 0 observed
    0 of 1520 sampled values were arrays, and a violation of that should surface
    rather than crash or be coerced.
    """
    out: dict[str, float] = {}
    if not isinstance(dv, dict):
        return out
    for side in ("left", "right"):
        block = dv.get(side)
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"{side}_{key}"] = float(value)
            elif isinstance(value, list) and len(value) == 1 \
                    and isinstance(value[0], (int, float)):
                out[f"{side}_{key}"] = float(value[0])
    return out


def main() -> int:
    root = data_root()
    ric = root / "ric_data"
    cohort, _oa, notes = build_cohort(root)
    print(f"cohort: {len(cohort)} sessions, {cohort['sub_id'].nunique()} subjects")
    print(f"streaming from {ric}\n")

    feature_rows: list[dict] = []
    structure_rows: list[dict] = []
    failures: list[str] = []
    non_scalar: list[str] = []
    t0 = time.perf_counter()

    for i, row in enumerate(cohort.itertuples(index=False), start=1):
        path = ric / str(row.sub_id) / str(row.filename)
        if not path.exists():
            failures.append(f"{row.sub_id}/{row.filename} - not found")
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - reported, never silently skipped
            failures.append(f"{row.sub_id}/{row.filename} - "
                            f"{type(exc).__name__}: {exc}")
            continue

        running = data.get("running") or {}
        joints = data.get("joints") or {}
        neutral = data.get("neutral") or {}
        dv_r = data.get("dv_r")

        n_frames = 0
        if isinstance(running, dict) and running:
            first = next(iter(running.values()))
            n_frames = len(first) if isinstance(first, list) else 0

        structure_rows.append({
            "sub_id": row.sub_id, "filename": row.filename,
            "n_marker_channels": len(running) if isinstance(running, dict) else 0,
            "n_joints_landmarks": len(joints) if isinstance(joints, dict) else 0,
            "n_neutral_markers": len(neutral) if isinstance(neutral, dict) else 0,
            "n_frames": n_frames,
            "hz_r": data.get("hz_r") if not isinstance(data.get("hz_r"), list) else None,
            "size_mb": path.stat().st_size / 1e6,
            "has_walking": bool(data.get("walking")),
        })

        flat = flatten_dv(dv_r)
        if isinstance(dv_r, dict):
            expected = sum(len(v) for v in dv_r.values() if isinstance(v, dict))
            if expected and len(flat) != expected:
                non_scalar.append(f"{row.sub_id}/{row.filename}: "
                                  f"{expected - len(flat)} non-scalar value(s)")
        flat["sub_id"] = row.sub_id
        flat["filename"] = row.filename
        feature_rows.append(flat)

        del data, running, joints, neutral, dv_r  # release before the next file

        if i % 200 == 0:
            rate = i / (time.perf_counter() - t0)
            print(f"  {i}/{len(cohort)}  {rate:.1f} files/s  "
                  f"eta {(len(cohort) - i) / rate / 60:.1f} min")

    elapsed = time.perf_counter() - t0
    print(f"\nstreamed {len(structure_rows)} files in {elapsed / 60:.1f} min "
          f"({len(structure_rows) / elapsed:.1f} files/s)")

    features = pd.DataFrame(feature_rows)
    structure = pd.DataFrame(structure_rows)
    DERIVED.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT_FEATURES, index=False)
    structure.to_parquet(OUT_STRUCTURE, index=False)

    print(f"\nfeatures : {features.shape[0]} rows x {features.shape[1] - 2} dv_r cols"
          f" -> {OUT_FEATURES.relative_to(REPO)} "
          f"({OUT_FEATURES.stat().st_size / 1e6:.1f} MB)")
    print(f"structure: {structure.shape[0]} rows "
          f"-> {OUT_STRUCTURE.relative_to(REPO)}")

    # --- reconciliation: report, never paper over -------------------------
    print("\n=== reconciliation ===")
    print(f"  cohort sessions          {len(cohort)}")
    print(f"  extracted                {len(structure)}")
    print(f"  failures                 {len(failures)}")
    for f in failures[:10]:
        print(f"    {f}")
    if len(failures) > 10:
        print(f"    ... and {len(failures) - 10} more")
    print(f"  non-scalar dv_r values   {len(non_scalar)}")
    for f in non_scalar[:5]:
        print(f"    {f}")

    dv_cols = [c for c in features.columns if c not in ("sub_id", "filename")]
    print(f"  dv_r columns             {len(dv_cols)}")
    all_null = [c for c in dv_cols if features[c].isna().all()]
    print(f"  columns entirely absent  {len(all_null)}")
    ragged = features[dv_cols].isna().sum(axis=1)
    print(f"  rows missing >=1 dv_r    {int((ragged > 0).sum())}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
