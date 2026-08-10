"""Phase 1C -- regression-test the re-run pipeline against the archive itself.

`gait_steps` returns DISCRETE_VARIABLES (77x3). The session JSONs already store
`dv_r`, produced by this same pipeline before the archive was published. If our
re-run reproduces those numbers, the pipeline is running correctly on R2026a
despite the two patches, and the 101-point curves it emits alongside them can be
trusted. If it does not, everything downstream is unverifiable and we stop.

This is the archive's own regression test. Nothing here is synthesized.

Run: .venv/Scripts/python.exe scripts/phase1c_validate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
MAT = REPO / "data" / "derived" / "validate_out.mat"

# Relative tolerance for "reproduces". The pipeline is deterministic, so exact
# agreement is the expectation; this allows only float round-trip through JSON.
RTOL, ATOL = 1e-4, 1e-6


def main() -> int:
    mat = loadmat(MAT, squeeze_me=True, struct_as_record=False)
    results = np.atleast_1d(mat["results"])
    print(f"{len(results)} sessions in {MAT.name}\n")

    n_ok = agreed = compared = 0
    per_session = []

    for rec in results:
        if not rec.ok:
            print(f"FAILED  {Path(rec.path).name}: {rec.err}")
            continue
        n_ok += 1
        dv_mat = np.asarray(rec.dv, dtype=float)  # (77, 3)

        with open(rec.path, encoding="utf-8") as fh:
            stored = json.load(fh)["dv_r"]
        left = stored["left"]
        right = stored["right"]
        keys = list(left.keys())  # JSON preserves the pipeline's own order

        # Row 1 of DISCRETE_VARIABLES is unused; rows 2..77 map to the 76 keys.
        # Column 2 is left, column 3 is right (1-indexed in MATLAB).
        if dv_mat.shape != (77, 3) or len(keys) != 76:
            print(f"SHAPE MISMATCH {Path(rec.path).name}: dv={dv_mat.shape} "
                  f"keys={len(keys)}")
            continue

        def as_float(value: object) -> float:
            """Stored dv_r carries JSON nulls in places. Absent, not zero."""
            if value is None or isinstance(value, (list, dict)):
                return np.nan
            return float(value)

        mine_l = dv_mat[1:, 1]
        mine_r = dv_mat[1:, 2]
        theirs_l = np.array([as_float(left[k]) for k in keys])
        theirs_r = np.array([as_float(right[k]) for k in keys])

        live = ~(((theirs_l == 0) | np.isnan(theirs_l))
                 & ((theirs_r == 0) | np.isnan(theirs_r)))
        ok_l = np.isclose(mine_l, theirs_l, rtol=RTOL, atol=ATOL, equal_nan=True)
        ok_r = np.isclose(mine_r, theirs_r, rtol=RTOL, atol=ATOL, equal_nan=True)
        both = (ok_l & ok_r)

        n_live = int(live.sum())
        n_agree = int(both[live].sum())
        agreed += n_agree
        compared += n_live
        per_session.append((Path(rec.path).name, n_agree, n_live,
                            rec.label, float(rec.speed),
                            int(rec.nsteps_L), int(rec.nsteps_R),
                            tuple(np.asarray(rec.ang_size).tolist())))

        print(f"{Path(rec.path).name}  live vars {n_agree}/{n_live} agree  "
              f"label={rec.label}  speed={rec.speed:.2f}  "
              f"steps L/R={rec.nsteps_L}/{rec.nsteps_R}  "
              f"norm_ang={tuple(np.asarray(rec.ang_size).tolist())}")

        if n_agree < n_live:
            bad = np.where(live & ~both)[0]
            for i in bad[:6]:
                print(f"    MISMATCH {keys[i]:<34} "
                      f"mine L/R {mine_l[i]:+.5g}/{mine_r[i]:+.5g}   "
                      f"stored L/R {theirs_l[i]:+.5g}/{theirs_r[i]:+.5g}")

    print(f"\n=== verdict ===")
    print(f"  sessions run           {n_ok}/{len(results)}")
    print(f"  live variables agreed  {agreed}/{compared}"
          + (f"  ({agreed / compared:.1%})" if compared else ""))

    # Did the patched pipeline populate anything the archive left at zero?
    if n_ok:
        rec = next(r for r in results if r.ok)
        dv_mat = np.asarray(rec.dv, dtype=float)
        with open(rec.path, encoding="utf-8") as fh:
            stored = json.load(fh)["dv_r"]
        keys = list(stored["left"].keys())
        theirs = np.array([[float(stored["left"][k]), float(stored["right"][k])]
                           for k in keys])
        dead = (theirs == 0).all(axis=1)
        revived = dead & (np.abs(dv_mat[1:, 1:]) > ATOL).any(axis=1)
        print(f"  stored-dead variables  {int(dead.sum())}")
        print(f"  ...that we populated   {int(revived.sum())}")
        if revived.any():
            print("    " + ", ".join(keys[i] for i in np.where(revived)[0][:8]))

    ok = compared > 0 and agreed == compared
    print(f"\n{'PASS' if ok else 'FAIL'} — "
          + ("pipeline reproduces the archive; waveforms are trustworthy."
             if ok else "does NOT reproduce the archive. Stop; do not batch."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
