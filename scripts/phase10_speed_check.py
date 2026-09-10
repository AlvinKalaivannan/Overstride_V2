"""Phase 10 (C) step 2 -- validate the Fukuchi adapter against prescribed speed.

THE TEST, AND WHY IT IS WORTH ANYTHING
--------------------------------------
Fukuchi ran each subject on a treadmill at PRESCRIBED speeds -- 2.5, 3.5 and
4.5 m/s -- recorded in the filename. `gait_steps` computes speed independently
from marker kinematics and never sees the filename.

So agreement between the two is an EXTERNAL check on the entire adapter at once:
the derived coordinate frame, the marker mapping, the cluster assignment, the
synthetic fourth markers and the units. It is not a self-consistency check and it
cannot be satisfied by a wrong adapter that happens to be smooth.

That matters because the failure mode here is silent. A wrong rotation still
produces continuous, physiological-looking curves -- the first version of this
adapter did exactly that, and its only visible symptom was a speed of 1.85 m/s
on a 2.5 m/s trial. Angles alone would not have caught it.

Three prescribed speeds per subject also make this a SLOPE test, not just an
offset test. A residual scale error would show as a systematic drift in relative
error across speeds even if one speed happened to match.

PRE-DECLARED READING, fixed before the batch was run:

  median |relative error| <= 3%    the adapter transfers; proceed to angles
  3-8%                             usable, but report the bias and its direction
  > 8%                             something is wrong; do not report angles

Treadmill belt speed and centre-of-mass speed are not identical, and the pipeline
estimates the latter, so a small systematic offset is expected and is not itself
a failure. A LARGE or speed-DEPENDENT error is.

Run: .venv/Scripts/python.exe scripts/phase10_speed_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FUK = REPO / "data" / "fukuchi"
MANIFEST = FUK / "ric_format" / "batch_manifest.csv"
OUT = REPO / "results" / "phase10_speed.json"

BAND_GOOD, BAND_USABLE = 3.0, 8.0    # percent, declared in advance


def main() -> int:
    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} not found. Run fukuchi_batch.m first.")
        return 2

    m = pd.read_csv(MANIFEST)
    err = m["err"].astype(str).str.strip().replace("nan", "")
    failed = m[err != ""]
    ok = m[(err == "") & m["speed_computed"].notna()].copy()

    print(f"trials in manifest: {len(m)}  |  pipeline errors: {len(failed)}")
    for e in failed["err"].astype(str).head(5):
        print(f"    {e[:100]}")
    if not len(ok):
        print("no successful trials")
        return 2

    ok["abs_err"] = ok["speed_computed"] - ok["speed_nominal"]
    ok["rel_err"] = 100 * ok["abs_err"] / ok["speed_nominal"]

    print(f"\n=== speed agreement, {len(ok)} trials / "
          f"{ok['subject'].nunique()} subjects ===")
    print(f"  {'nominal':>8}{'n':>5}{'computed mean':>15}{'sd':>8}"
          f"{'mean err':>10}{'median |rel|':>14}")
    per_speed = {}
    for sp, g in ok.groupby("speed_nominal"):
        med = float(np.median(np.abs(g["rel_err"])))
        print(f"  {sp:>8.1f}{len(g):>5}{g['speed_computed'].mean():>15.3f}"
              f"{g['speed_computed'].std():>8.3f}{g['abs_err'].mean():>+10.3f}"
              f"{med:>13.2f}%")
        per_speed[str(sp)] = {
            "n": int(len(g)),
            "computed_mean": float(g["speed_computed"].mean()),
            "computed_sd": float(g["speed_computed"].std()),
            "mean_abs_err_ms": float(g["abs_err"].mean()),
            "median_rel_err_pct": med}

    overall = float(np.median(np.abs(ok["rel_err"])))
    r = float(np.corrcoef(ok["speed_nominal"], ok["speed_computed"])[0, 1])

    # Slope test: a residual scale error shows up as speed-dependent relative
    # error even when a single speed happens to agree.
    slope, intercept = np.polyfit(ok["speed_nominal"], ok["speed_computed"], 1)
    drift = (per_speed[max(per_speed)]["mean_abs_err_ms"]
             - per_speed[min(per_speed)]["mean_abs_err_ms"])

    print(f"\n  overall median |relative error|: {overall:.2f}%")
    print(f"  correlation nominal vs computed: r = {r:.4f}")
    print(f"  regression: computed = {slope:.4f} x nominal {intercept:+.4f}")
    print(f"  error drift, slowest -> fastest: {drift:+.3f} m/s")

    verdict = ("PASS -- the adapter transfers" if overall <= BAND_GOOD else
               "USABLE, with a reported bias" if overall <= BAND_USABLE else
               "FAIL -- do not report angles")
    print(f"\n  pre-declared bands: <={BAND_GOOD}% pass, "
          f"<={BAND_USABLE}% usable, else fail")
    print(f"  VERDICT: {verdict}")

    worst = ok.reindex(ok["rel_err"].abs().sort_values(ascending=False).index)
    print(f"\n  worst 5 trials:")
    for t in worst.head(5).itertuples():
        print(f"    {t.trial:<20} nominal {t.speed_nominal:.1f}  "
              f"computed {t.speed_computed:.3f}  ({t.rel_err:+.1f}%)")

    payload = {
        "n_trials": int(len(ok)), "n_subjects": int(ok["subject"].nunique()),
        "n_pipeline_errors": int(len(failed)),
        "per_speed": per_speed,
        "median_rel_err_pct": overall, "correlation": r,
        "regression_slope": float(slope), "regression_intercept": float(intercept),
        "error_drift_ms": float(drift),
        "bands": {"pass_pct": BAND_GOOD, "usable_pct": BAND_USABLE},
        "verdict": verdict,
        "eventsflag_median": float(ok["eventsflag"].median()),
        "steps_median": float(ok[["nsteps_L", "nsteps_R"]].median().mean()),
        "source": "Fukuchi et al. 2017, figshare 10.6084/m9.figshare.4543435 v5",
        "licence": "CC BY 4.0", "retrieved": "2026-09-10",
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
