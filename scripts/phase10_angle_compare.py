"""Phase 10 (C) step 3 -- compare recovered joint angles against Fukuchi's own.

THE TEST
--------
Phase 10's speed check established that the RIC pipeline's GEOMETRY transfers to
Fukuchi's markers (median 1.71% error, r = 0.9995). It said nothing about the
angles, because speed does not exercise the angle convention at all.

This compares, for the same trials, the sagittal hip/knee/ankle waveforms that
the RIC pipeline recovers from Fukuchi's raw markers against the waveforms
Fukuchi themselves published from the same recordings, computed independently in
Visual3D with their own model. A different pipeline, a different marker model,
the same underlying motion.

THE NORMALISATION MISMATCH, AND HOW IT IS RESOLVED
-------------------------------------------------
The two are not comparable point-for-point as shipped:

  RIC `gait_steps`  ->  101 points of STANCE
  Fukuchi processed ->  101 points of the FULL GAIT CYCLE (`PercGcycle` 0-100)

Verified that Fukuchi's cycle starts at initial contact, from the curve shape
itself: knee 13.4 deg at 0%, a 40.6 deg midstance peak near 20%, an 84.6 deg
swing peak near 70%, returning to 13.4 deg at 100%.

So Fukuchi's curve is sliced to its stance portion and resampled to 101 points.
The stance fraction is NOT assumed -- it is measured per trial from Fukuchi's own
instrumented-treadmill vertical force, which is independent of both pipelines.

SIGN CONVENTIONS: DECLARED PER JOINT FROM THE POPULATION, NOT PER CURVE
----------------------------------------------------------------------
Two labs' flexion/extension signs need not agree. Measured here, they agree for
the KNEE and are inverted for the HIP and ANKLE between the RIC pipeline and
Fukuchi's Visual3D model.

This matters for the error metric, and an earlier version of this script got it
wrong in a way worth recording. A pure sign inversion makes the offset-removed
RMS equal TWICE the signal amplitude -- so hip reported |r| = 0.991 (near-perfect
shape) beside a 28 deg RMS, which cannot both describe the same pair of curves.
The metric was conflating "the convention differs" with "the magnitudes differ".

The sign is therefore resolved ONCE PER JOINT from the median correlation across
all trials and subjects, and applied as a declared convention -- never chosen per
curve, which would be a fudge that guarantees agreement. Both the signed
correlation and the convention are reported, and RMS is computed on sign-aligned
curves and labelled as such.

AMPLITUDE IS CHECKED SEPARATELY. The range ratio (ours / theirs) tests scale
agreement with no sign or offset confound, which neither r nor RMS does cleanly
on its own.

WHAT A MISMATCH WOULD AND WOULD NOT MEAN
---------------------------------------
A disagreement here is ambiguous between an adapter error and a genuine
difference between two marker models, and that ambiguity cannot be resolved from
this comparison alone. It is declared up front so the result is not over-read in
either direction. Systematic OFFSETS are expected -- different marker models
define neutral differently -- so RMS is computed after removing a constant
offset, and the offset is reported separately because it is the quantity that
actually differs between models.

PRE-DECLARED READING, fixed before running:

  |r| >= 0.95 and offset-removed RMS <= 3 deg   the angle path transfers
  |r| 0.85-0.95                                 shape recovered, detail differs
  |r| < 0.85                                    does not transfer; 7.8 worsens

Run: .venv/Scripts/python.exe scripts/phase10_angle_compare.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

REPO = Path(__file__).resolve().parents[1]
FUK = REPO / "data" / "fukuchi"
CURVES = FUK / "ric_format" / "curves"
OUT = REPO / "results" / "phase10_angles.json"

SAGITTAL = 2          # plane index: flexion/extension (CLAUDE.md; gait_kinematics.m:62)
FUK_AXIS = "Z"        # Fukuchi's flexion axis, measured in phase 10's first pass
JOINTS = ["hip", "knee", "ankle"]
SIDES = {"R": "R", "L": "L"}

GRF_THRESHOLD_N = 50.0        # contact detection; running peaks are ~1400 N
MIN_CONTACTS = 6
# Forces are sampled at 300 Hz. A running contact is ~0.15-0.30 s, so anything
# under 0.05 s is threshold noise splitting one contact in two -- which breaks
# the strict alternation that stride = onset[i] -> onset[i+2] depends on.
MIN_CONTACT_SAMPLES = 15
BAND_R_GOOD, BAND_R_PART = 0.95, 0.85
BAND_RMS_GOOD = 3.0


def stance_fraction(trial: str) -> tuple[float, dict]:
    """Measure stance as a fraction of the stride from Fukuchi's own force plate.

    The treadmill reports ONE combined vertical force, so each supra-threshold
    interval is one foot's stance and consecutive intervals alternate feet. A
    stride for a given foot therefore spans onset[i] -> onset[i+2].
    """
    f = pd.read_csv(FUK / f"{trial}forces.txt", sep="\t")
    fy = f["Fy"].to_numpy(dtype=float)
    contact = fy > GRF_THRESHOLD_N
    edges = np.diff(contact.astype(int))
    onsets = np.flatnonzero(edges == 1) + 1
    offsets = np.flatnonzero(edges == -1) + 1
    if len(onsets) < MIN_CONTACTS or len(offsets) < MIN_CONTACTS:
        raise ValueError(f"only {len(onsets)} contacts detected")
    if offsets[0] < onsets[0]:
        offsets = offsets[1:]
    n = min(len(onsets), len(offsets))
    onsets, offsets = onsets[:n], offsets[:n]

    keep = (offsets - onsets) >= MIN_CONTACT_SAMPLES
    onsets, offsets = onsets[keep], offsets[keep]
    if len(onsets) < MIN_CONTACTS:
        raise ValueError(f"only {len(onsets)} contacts survive the duration filter")

    stance = offsets - onsets
    stride = onsets[2:] - onsets[:-2]          # same foot
    k = min(len(stance) - 2, len(stride))
    if k < 2:
        raise ValueError("too few complete strides")
    frac = stance[:k] / stride[:k]
    frac = frac[(frac > 0.15) & (frac < 0.75)]     # physiological for running
    if len(frac) < 2:
        raise ValueError("no plausible stride fractions")
    return float(np.median(frac)), {
        "n_contacts": int(len(onsets)),
        "n_strides_used": int(len(frac)),
        "stance_frac_sd": float(np.std(frac))}


def fukuchi_stance_curve(proc: pd.DataFrame, side: str, joint: str,
                         speed_tag: str, frac: float) -> np.ndarray:
    """Fukuchi's cycle-normalised curve, sliced to stance and resampled to 101."""
    col = f"{side}{joint}Ang{FUK_AXIS}{speed_tag}"
    if col not in proc.columns:
        raise KeyError(col)
    full = proc[col].to_numpy(dtype=float)          # 101 points, 0-100% cycle
    end = frac * 100.0
    src = np.linspace(0.0, 100.0, len(full))
    want = np.linspace(0.0, end, 101)
    return np.interp(want, src, full)


def ric_stance_curve(mat: dict, side: str, joint: str) -> np.ndarray:
    """Our pipeline's stance curve: mean across steps, sagittal plane."""
    a = mat[f"ang_{side}_{joint}"]                  # (101, n_steps, 3)
    return np.nanmean(a[:, :, SAGITTAL], axis=1).astype(float)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default="curves",
                    help="subdirectory of ric_format holding the .mat curves")
    ap.add_argument("--manifest", default="batch_manifest.csv")
    ap.add_argument("--out", default="", help="output json filename")
    ap.add_argument("--label", default="", help="tag printed with the results")
    args = ap.parse_args()
    global CURVES, OUT
    CURVES = FUK / "ric_format" / args.curves
    if args.out:
        OUT = REPO / "results" / args.out
    if args.label:
        print(f"### {args.label}")

    man_p = FUK / "ric_format" / args.manifest
    if not man_p.exists():
        print(f"ERROR: {man_p} not found. Run fukuchi_batch.m first.")
        return 2
    man = pd.read_csv(man_p)
    err = man["err"].astype(str).str.strip().replace("nan", "")
    man = man[(err == "") & man["speed_computed"].notna()]

    # Duplicate-content trials carry an unassignable nominal speed; phase 10's
    # speed check excludes them and the same exclusion applies here, because the
    # Fukuchi column to compare against is selected BY nominal speed.
    sp = json.loads((REPO / "results" / "phase10_speed.json").read_text())
    dup = {t for v in sp["duplicate_groups"].values() for t in v}
    man = man[~man["trial"].isin(dup)]
    print(f"trials available: {len(man)} (after excluding {len(dup)} duplicates)")

    rows, skipped = [], []
    for t in man.itertuples():
        m = re.match(r"^(RBDS\d+)runT(\d+)$", t.trial)
        subj, tag = m.group(1), m.group(2)
        cur_p = CURVES / f"{t.trial}.mat"
        proc_p = FUK / f"{subj}processed.txt"
        if not cur_p.exists() or not proc_p.exists():
            skipped.append((t.trial, "missing curve or processed file"))
            continue
        try:
            frac, fmeta = stance_fraction(t.trial)
        except (ValueError, KeyError, FileNotFoundError) as e:
            skipped.append((t.trial, f"stance: {e}"))
            continue

        mat = loadmat(cur_p)
        proc = pd.read_csv(proc_p, sep="\t")
        for side in SIDES:
            for joint in JOINTS:
                try:
                    f_curve = fukuchi_stance_curve(proc, side, joint, tag, frac)
                except KeyError as e:
                    skipped.append((t.trial, f"no column {e}"))
                    continue
                r_curve = ric_stance_curve(mat, side, joint)
                if not (np.all(np.isfinite(f_curve)) and np.all(np.isfinite(r_curve))):
                    skipped.append((t.trial, f"{side}_{joint}: non-finite"))
                    continue
                r = float(np.corrcoef(r_curve, f_curve)[0, 1])
                offset = float(np.mean(r_curve) - np.mean(f_curve))
                rng_r = float(r_curve.max() - r_curve.min())
                rng_f = float(f_curve.max() - f_curve.min())
                rows.append({"trial": t.trial, "subject": subj,
                             "speed": float(tag) / 10, "side": side,
                             "joint": joint, "r": r, "offset_deg": offset,
                             "range_ours_deg": rng_r, "range_fuk_deg": rng_f,
                             "amp_ratio": (rng_r / rng_f) if rng_f else np.nan,
                             "curve_ours": r_curve, "curve_fuk": f_curve,
                             "stance_frac": frac,
                             "n_strides": fmeta["n_strides_used"]})

    if not rows:
        print("no comparable curves")
        return 2
    df = pd.DataFrame(rows)

    # ONE convention per joint, from the population median -- never per curve.
    sign = {j: (-1.0 if df.loc[df["joint"] == j, "r"].median() < 0 else 1.0)
            for j in JOINTS}
    aligned = []
    for i in range(len(df)):
        j = df["joint"].iloc[i]
        rc = sign[j] * np.asarray(df["curve_ours"].iloc[i], dtype=float)
        fc = np.asarray(df["curve_fuk"].iloc[i], dtype=float)
        rc = rc - rc.mean()
        fc = fc - fc.mean()
        aligned.append(float(np.sqrt(np.mean((rc - fc) ** 2))))
    df["rms_aligned_deg"] = aligned
    df = df.drop(columns=["curve_ours", "curve_fuk"])

    print("\n=== sign convention, resolved per joint from the population ===")
    for j in JOINTS:
        med = float(df.loc[df["joint"] == j, "r"].median())
        tag_ = "INVERTED vs Fukuchi" if sign[j] < 0 else "same as Fukuchi"
        print(f"  {j:<7} median r = {med:+.3f}  ->  {tag_}")

    print(f"\n=== stance fraction, from Fukuchi's own force plate ===")
    for s, g in df.groupby("speed"):
        u = g.drop_duplicates("trial")
        print(f"  {s:.1f} m/s: median {u['stance_frac'].median():.3f} "
              f"(n={len(u)} trials)")

    print(f"\n=== per joint, {df['trial'].nunique()} trials / "
          f"{df['subject'].nunique()} subjects ===")
    print(f"  {'joint':<8}{'n':>5}{'median r':>11}{'|r|':>8}"
          f"{'RMS aligned':>15}{'amp ratio':>11}{'offset':>9}")
    per_joint = {}
    for (joint,), g in df.groupby(["joint"]):
        mr = float(g["r"].median())
        per_joint[joint] = {
            "n": int(len(g)), "median_r": mr,
            "median_abs_r": float(g["r"].abs().median()),
            "median_rms_aligned_deg": float(g["rms_aligned_deg"].median()),
            "median_amp_ratio": float(g["amp_ratio"].median()),
            "median_range_ours_deg": float(g["range_ours_deg"].median()),
            "median_range_fuk_deg": float(g["range_fuk_deg"].median()),
            "median_offset_deg": float(g["offset_deg"].median()),
            "convention_inverted": bool(sign[joint] < 0)}
        p = per_joint[joint]
        print(f"  {joint:<8}{p['n']:>5}{mr:>11.3f}{p['median_abs_r']:>8.3f}"
              f"{p['median_rms_aligned_deg']:>11.2f} deg"
              f"{p['median_amp_ratio']:>11.3f}{p['median_offset_deg']:>+9.1f}")

    overall_absr = float(df["r"].abs().median())
    overall_rms = float(df["rms_aligned_deg"].median())
    verdict = ("PASS -- the angle path transfers"
               if overall_absr >= BAND_R_GOOD and overall_rms <= BAND_RMS_GOOD else
               "PARTIAL -- shape recovered, detail differs"
               if overall_absr >= BAND_R_PART else
               "FAIL -- the angle path does not transfer")
    print(f"\n  overall median |r| = {overall_absr:.3f}  |  "
          f"median offset-removed RMS = {overall_rms:.2f} deg")
    print(f"  pre-declared: |r|>={BAND_R_GOOD} & RMS<={BAND_RMS_GOOD} deg pass, "
          f"|r|>={BAND_R_PART} partial, else fail")
    print(f"  VERDICT: {verdict}")

    print(f"\n=== by speed (is agreement speed-dependent?) ===")
    per_speed = {}
    for s, g in df.groupby("speed"):
        per_speed[str(s)] = {"n": int(len(g)),
                             "median_abs_r": float(g["r"].abs().median()),
                             "median_rms_deg": float(g["rms_aligned_deg"].median()),
                             "median_amp_ratio": float(g["amp_ratio"].median())}
        q = per_speed[str(s)]
        print(f"  {s:.1f} m/s: |r| {q['median_abs_r']:.3f}  "
              f"RMS {q['median_rms_deg']:.2f} deg  "
              f"amp ratio {q['median_amp_ratio']:.3f}  (n={int(len(g))})")

    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for t, why in skipped[:8]:
            print(f"    {t}: {why}")

    payload = {"n_comparisons": int(len(df)),
               "n_trials": int(df["trial"].nunique()),
               "n_subjects": int(df["subject"].nunique()),
               "per_joint": per_joint, "per_speed": per_speed,
               "overall_median_abs_r": overall_absr,
               "overall_median_rms_aligned_deg": overall_rms,
               "sign_convention": {j: ("inverted" if v < 0 else "same")
                                   for j, v in sign.items()},
               "overall_median_amp_ratio": float(df["amp_ratio"].median()),
               "stance_frac_median": float(df.drop_duplicates("trial")["stance_frac"].median()),
               "bands": {"r_pass": BAND_R_GOOD, "r_partial": BAND_R_PART,
                         "rms_pass_deg": BAND_RMS_GOOD},
               "verdict": verdict, "n_skipped": len(skipped),
               "source": "Fukuchi et al. 2017, figshare 10.6084/m9.figshare.4543435 v5",
               "licence": "CC BY 4.0", "retrieved": "2026-09-10"}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
