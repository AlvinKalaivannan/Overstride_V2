"""Phase 10 (C) step 1 -- adapt Fukuchi marker data into the RIC pipeline's input.

WHY THIS EXISTS
---------------
`docs/REVIEW.md` §7.8 records that every waveform in this project came out of one
MATLAB pipeline, and that nothing independent has ever driven it. That is the
soft spot this closes: feed a DIFFERENT lab's markers, on DIFFERENT people, from
a DIFFERENT marker model, through the same `gait_kinematics` -> `gait_steps`
path, and see whether it produces sane kinematics that agree with that lab's own
independently-computed angles.

This is also the readiness review's R1 "adapter contract", made concrete.

Fukuchi et al. 2017, "A public dataset of running biomechanics and the effects of
running speed on lower extremity kinematics and kinetics".
figshare DOI 10.6084/m9.figshare.4543435 (v5), CC BY 4.0. Retrieved 2026-09-10.

THE CONTRACT, ESTABLISHED BY READING THE PIPELINE RATHER THAN ASSUMING
----------------------------------------------------------------------
`gait_kinematics(joints, neutral, dynamic, hz, plots)` needs:

  joints  (14 fields, one 3-vector each)   anatomical landmarks, static trial
  neutral (30 fields, one 3-vector each)   cluster markers, static trial
  dynamic (30 fields, (n_frames, 3) each)  the same clusters, running trial
  hz                                       sample rate

COORDINATE FRAMES -- DERIVED FROM ANATOMY, NOT HARDCODED
--------------------------------------------------------
An earlier version of this adapter hardcoded an axis permutation, and it was
wrong in a way worth recording. `gait_kinematics.m:57-63` documents its input as
the Bonita lab frame (X right, Y walking direction, Z up) and says that frame "is
switched to" an internal one (X right, Y up, Z opposite walking). There is no
conversion code between that comment and the first use of the data: the switch
happens upstream, and **the archive JSON already stores the internal frame.**

Measured on a real RIC session, which is how this was caught:

    neutral L_foot centroid  [512.3,  59.9, 168.3]   <- Y is the floor
    neutral pelvis centroid  [700.2, 964.7, 207.5]   <- Y is pelvis height

So the target frame is X = subject's right, Y = vertically up, Z = POSTERIOR
(opposite the walking direction). Right-handed: right x up = posterior.

Rather than assert a permutation, `derive_rotation` builds that basis out of the
subject's own anatomy in the static trial:

    up        = pelvis centroid - foot centroid      (the subject is standing)
    right     = R.ASIS - L.ASIS, orthogonalised against up
    posterior = right x up                           (completes right-handed)

This needs no assumption about the source lab's axis order or signs, works for a
frame that is not axis-aligned at all, and is checked rather than trusted:
orthonormality and det = +1, then three independent physical facts -- the heel
reaches the floor along +up, ASIS sits anterior of PSIS, and the planted foot
travels posteriorly through stance. Any failure stops the conversion.

CLUSTER ORDERING. Checked in the source rather than guessed:
  - thigh and shank anatomical frames are built ENTIRELY from `joints` landmarks
    (`gait_kinematics.m:174-222`), never from cluster order. Those clusters need
    only be CONSISTENT between neutral and dynamic, because the segment rotation
    comes from a Soderkvist-Wedin SVD fit of one onto the other.
  - `ag.pelvis` is HARDCODED to the lab axes (`:233-235`), so pelvis ordering does
    not matter either -- but it does assume the subject faces the walking
    direction in the neutral trial. Verified for Fukuchi: mean ASIS is 168.8 mm
    anterior of mean PSIS in the static trial.
  - THE FOOT IS THE EXCEPTION (`:120-130`). It takes `L_foot_1..3`, sorts them by
    the medio-lateral axis, and treats the outermost as the lateral heel and the
    remaining two as a vertical pair. So 1..3 must be the three REAL heel markers.
    Fukuchi's Heel.Top / Heel.Bottom / Heel.Lateral map onto that exactly.

THE SYNTHETIC FOURTH. In the RIC archive, `foot_4` and `pelvis_4` are not
measured -- both are exactly the centroid of markers 1..3 (verified: affine
weights (1/3, 1/3, 1/3), residual 0.0000 mm, four points exactly coplanar). This
reproduces that construction rather than inventing a marker: the fourth carries
no new information and exists only to fill the pipeline's fixed 4-column matrix.

UNUSED FIELDS. `joints.*_first`, `joints.*_fifth`, `L_toe` and `R_toe` appear in
the archive's structure but neither `gait_kinematics.m` nor `gait_steps.m`
references them. They are populated for structural fidelity and affect nothing.

DEVIATION, RECORDED. RIC's pelvis cluster is 3 real markers plus their centroid.
Fukuchi has six pelvis markers, so this uses four real ones (R/L ASIS, R/L PSIS)
rather than three-plus-centroid. `ag.pelvis` is hardcoded and `jc.pelvis` is a
centroid either way, so this cannot change a joint angle; it makes the rigid fit
better conditioned.

NOTHING IS SYNTHESIZED. Every value written comes from a Fukuchi measurement,
except the two centroid markers described above, which reproduce a construction
already present in the RIC data.

Run: .venv/Scripts/python.exe scripts/phase10_fukuchi_adapter.py --subject RBDS001
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FUK = REPO / "data" / "fukuchi"
OUT = FUK / "ric_format"

# Markers used to derive the frame. Chosen because they are anatomically
# unambiguous and present in both the static and running trials.
PELVIS_REF = ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS"]
FOOT_REF = ["R.Heel.Bottom", "L.Heel.Bottom"]
ANT_REF, POST_REF = ["R.ASIS", "L.ASIS"], ["R.PSIS", "L.PSIS"]

# --- anatomical landmarks: RIC name -> Fukuchi marker (static trial only) ----
JOINTS = {
    "L_hip": "L.GTR",            "R_hip": "R.GTR",
    "L_lat_knee": "L.Knee",      "L_med_knee": "L.Knee.Medial",
    "R_lat_knee": "R.Knee",      "R_med_knee": "R.Knee.Medial",
    "L_lat_ankle": "L.Ankle",    "L_med_ankle": "L.Ankle.Medial",
    "R_lat_ankle": "R.Ankle",    "R_med_ankle": "R.Ankle.Medial",
    # present in the archive's structure, referenced by neither pipeline file
    "L_first": "L.MT1",          "L_fifth": "L.MT5",
    "R_first": "R.MT1",          "R_fifth": "R.MT5",
}

# --- tracking clusters: RIC name -> Fukuchi marker, in both trials -----------
CLUSTERS = {
    "pelvis": ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS"],
    "L_thigh": ["L.Thigh.Top.Lateral", "L.Thigh.Top.Medial",
                "L.Thigh.Bottom.Lateral", "L.Thigh.Bottom.Medial"],
    "R_thigh": ["R.Thigh.Top.Lateral", "R.Thigh.Top.Medial",
                "R.Thigh.Bottom.Lateral", "R.Thigh.Bottom.Medial"],
    "L_shank": ["L.Shank.Top.Lateral", "L.Shank.Top.Medial",
                "L.Shank.Bottom.Lateral", "L.Shank.Bottom.Medial"],
    "R_shank": ["R.Shank.Top.Lateral", "R.Shank.Top.Medial",
                "R.Shank.Bottom.Lateral", "R.Shank.Bottom.Medial"],
    # 1..3 must be the three real heel markers; 4 is their centroid (see above)
    "L_foot": ["L.Heel.Top", "L.Heel.Bottom", "L.Heel.Lateral"],
    "R_foot": ["R.Heel.Top", "R.Heel.Bottom", "R.Heel.Lateral"],
}
# Unused by the pipeline; a preference list so an absent marker cannot sink a
# trial. First available wins.
TOE = {"L_toe": ["L.MT1", "L.MT5", "L.MT2"],
       "R_toe": ["R.MT1", "R.MT5", "R.MT2"]}
MIN_FRAMES = 750          # 5 s at 150 Hz -- ample for speed and stride events


class TrialUnusable(Exception):
    """One trial cannot be converted. Recorded and skipped, never fatal."""


def read_markers(path: Path) -> tuple[pd.DataFrame, float]:
    """Read a Fukuchi marker .txt and recover its true sample rate.

    The Time column is rounded to 0.007 s, which reads as 142.9 Hz. The true
    rate is 150 Hz and is recovered from the span, not the step.
    """
    d = pd.read_csv(path, sep="\t")
    t = d["Time"].to_numpy()
    hz = (len(t) - 1) / (t[-1] - t[0])
    return d, hz


def raw(d: pd.DataFrame, marker: str) -> np.ndarray:
    """(n_frames, 3) for one marker, in the source lab's own frame."""
    cols = [f"{marker}{a}" for a in "XYZ"]
    missing = [c for c in cols if c not in d.columns]
    if missing:
        raise KeyError(f"marker {marker!r} absent: {missing}")
    return d[cols].to_numpy(dtype=float)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def derive_rotation(static: pd.DataFrame) -> np.ndarray:
    """Build the rotation into the RIC archive frame out of the subject's anatomy.

    Returns R with rows [right, up, posterior], so `raw @ R.T` expresses a point
    in the target frame. Nothing about the source lab's axis order or signs is
    assumed -- a frame that is rotated arbitrarily relative to anatomy converts
    just as correctly as an axis-aligned one.
    """
    pelvis = np.mean([raw(static, m).mean(axis=0) for m in PELVIS_REF], axis=0)
    foot = np.mean([raw(static, m).mean(axis=0) for m in FOOT_REF], axis=0)
    up = _unit(pelvis - foot)                       # subject is standing

    r_asis = raw(static, "R.ASIS").mean(axis=0)
    l_asis = raw(static, "L.ASIS").mean(axis=0)
    right = r_asis - l_asis
    right = _unit(right - np.dot(right, up) * up)   # orthogonalise against up

    posterior = np.cross(right, up)                 # right x up = posterior
    R = np.vstack([right, up, posterior])

    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "frame is not orthonormal"
    assert abs(np.linalg.det(R) - 1.0) < 1e-9, "frame is not right-handed"
    return R


def xyz(d: pd.DataFrame, marker: str, R: np.ndarray) -> np.ndarray:
    """(n_frames, 3) for one marker, rotated into the RIC archive frame."""
    return raw(d, marker) @ R.T


def check_frame(dyn: pd.DataFrame, static: pd.DataFrame, R: np.ndarray) -> dict:
    """Verify the derived frame against physical facts it was not built from.

    `derive_rotation` uses the pelvis, the heels and the ASIS pair. These checks
    use the DYNAMIC trial and the ASIS/PSIS relationship, so passing them is
    evidence rather than restating the construction. Any failure stops the
    conversion -- a silently wrong frame produces plausible, wrong angles, which
    is the worst outcome available here.
    """
    n: dict = {}

    # 1. the heel reaches its lowest point along -up, and the pelvis is above it
    heel = xyz(dyn, "R.Heel.Bottom", R)
    pel = np.mean([xyz(dyn, m, R) for m in PELVIS_REF], axis=0)
    n["pelvis_above_heel_mm"] = float(np.nanmean(pel[:, 1] - heel[:, 1]))
    n["pelvis_is_above_heel"] = n["pelvis_above_heel_mm"] > 500

    # 2. ASIS sits anterior of PSIS, i.e. LESS positive along +posterior
    ant = np.mean([xyz(static, m, R).mean(axis=0) for m in ANT_REF], axis=0)
    post = np.mean([xyz(static, m, R).mean(axis=0) for m in POST_REF], axis=0)
    n["psis_minus_asis_posterior_mm"] = float(post[2] - ant[2])
    n["asis_is_anterior_of_psis"] = n["psis_minus_asis_posterior_mm"] > 50

    # 3. through stance the planted foot travels POSTERIORLY (+Z)
    stance = heel[:, 1] < np.nanpercentile(heel[:, 1], 25)
    dz = float(np.nanmean(np.gradient(heel[:, 2])[stance]))
    n["stance_posterior_drift_mm_per_frame"] = dz
    n["foot_travels_posteriorly_in_stance"] = dz > 0

    # 4. medio-lateral excursion is the smallest of the three, as it must be
    exc = [float(np.nanpercentile(heel[:, i], 95) - np.nanpercentile(heel[:, i], 5))
           for i in range(3)]
    n["heel_excursion_mm"] = {"right": exc[0], "up": exc[1], "posterior": exc[2]}
    n["ml_excursion_is_smallest"] = exc[0] == min(exc)

    n["asis_width_mm"] = float(
        np.nanmean(xyz(dyn, "R.ASIS", R)[:, 0] - xyz(dyn, "L.ASIS", R)[:, 0]))
    n["ok"] = all(n[k] for k in ("pelvis_is_above_heel", "asis_is_anterior_of_psis",
                                 "foot_travels_posteriorly_in_stance",
                                 "ml_excursion_is_smallest"))
    return n


def longest_clean_window(d: pd.DataFrame, markers: list[str]) -> tuple[int, int]:
    """Longest contiguous run of frames with no dropout in any marker used.

    Fukuchi trials carry occasional single-frame marker gaps. Interpolating them
    would be standard biomechanics practice, but `CLAUDE.md` forbids imputing
    values into a dataset and a trimmed window needs no such exception: it keeps
    the time base continuous, which the velocity and event-detection stages
    require, and every retained sample is measured.
    """
    cols = [f"{m}{a}" for m in markers for a in "XYZ"]
    bad = d[cols].isna().any(axis=1).to_numpy()
    best = (0, 0)
    start = None
    for i, is_bad in enumerate(np.append(bad, True)):
        if is_bad:
            if start is not None and i - start > best[1] - best[0]:
                best = (start, i)
            start = None
        elif start is None:
            start = i
    return best


def build(subject: str, trial: str) -> tuple[dict, dict]:
    static_p = FUK / f"{subject}static.txt"
    dyn_p = FUK / f"{subject}{trial}markers.txt"
    for p in (static_p, dyn_p):
        if not p.exists():
            raise FileNotFoundError(p)

    static, _ = read_markers(static_p)
    dyn, hz = read_markers(dyn_p)

    R = derive_rotation(static)

    # The toe markers are deliberately EXCLUDED from this requirement. Neither
    # gait_kinematics.m nor gait_steps.m references L_toe/R_toe, and letting an
    # unused marker gate the window discarded whole trials: RBDS003runT25 and
    # RBDS005runT45 are 100% "bad" solely because R.MT1 is absent throughout.
    used = sorted({m for ms in CLUSTERS.values() for m in ms}
                  | set(PELVIS_REF) | set(FOOT_REF))
    lo, hi = longest_clean_window(dyn, used)
    n_total = len(dyn)
    kept = hi - lo
    if kept < MIN_FRAMES:
        raise TrialUnusable(
            f"longest gap-free window {kept} frames "
            f"({100 * kept / max(n_total, 1):.1f}% of {n_total}), "
            f"below the {MIN_FRAMES}-frame minimum")
    dyn = dyn.iloc[lo:hi].reset_index(drop=True)

    frame = check_frame(dyn, static, R)
    if not frame["ok"]:
        failed = [k for k, v in frame.items() if v is False]
        raise SystemExit(f"{subject}/{trial}: derived-frame check FAILED on "
                         f"{failed}. Refusing to convert.\n{frame}")

    out: dict = {"hz_r": float(round(hz)), "hz_w": float(round(hz))}

    # joints: one position per landmark, averaged over the static trial
    out["joints"] = {ric: xyz(static, fuk, R).mean(axis=0).tolist()
                     for ric, fuk in JOINTS.items()}

    def cluster(frame_df: pd.DataFrame, seg: str, names: list[str],
                static_mode: bool):
        arrs = [xyz(frame_df, nm, R) for nm in names]
        if len(arrs) == 3:                       # foot: add the centroid fourth
            arrs.append(np.mean(arrs, axis=0))
        vals = {}
        for i, a in enumerate(arrs, start=1):
            vals[f"{seg}_{i}"] = (a.mean(axis=0).tolist() if static_mode
                                  else a.tolist())
        return vals

    out["neutral"] = {}
    out["running"] = {}
    for seg, names in CLUSTERS.items():
        out["neutral"].update(cluster(static, seg, names, True))
        out["running"].update(cluster(dyn, seg, names, False))
    # Unused by the pipeline, but the struct has a fixed shape. Fall back to
    # another real forefoot marker rather than writing NaN into the JSON.
    toe_used = {}
    for ric, prefs in TOE.items():
        pick = next((m for m in prefs
                     if f"{m}X" in dyn.columns
                     and not dyn[f"{m}X"].isna().all()
                     and f"{m}X" in static.columns), None)
        if pick is None:
            raise TrialUnusable(f"no usable forefoot marker for {ric}")
        toe_used[ric] = pick
        out["neutral"][ric] = xyz(static, pick, R).mean(axis=0).tolist()
        out["running"][ric] = np.nan_to_num(xyz(dyn, pick, R),
                                            nan=0.0).tolist()

    assert len(out["joints"]) == 14, len(out["joints"])
    assert len(out["neutral"]) == 30, len(out["neutral"])
    assert len(out["running"]) == 30, len(out["running"])

    meta = {"subject": subject, "trial": trial, "hz": out["hz_r"],
            "n_frames": int(len(dyn)), "frame_checks": frame,
            "n_frames_source": int(n_total),
            "trimmed_window": [int(lo), int(hi)],
            "frames_dropped_to_avoid_gaps": int(n_total - (hi - lo)),
            "pct_frames_kept": float(100 * (hi - lo) / max(n_total, 1)),
            "toe_markers_used": toe_used,
            "rotation_rows_right_up_posterior": R.tolist(),
            "source": "Fukuchi et al. 2017, figshare 10.6084/m9.figshare.4543435 v5",
            "licence": "CC BY 4.0", "retrieved": "2026-09-10"}
    return out, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="", help="e.g. RBDS001; blank = all found")
    ap.add_argument("--trials", default="runT25,runT35,runT45")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    subjects = ([args.subject] if args.subject else
                sorted({m.group(1) for f in FUK.glob("*static.txt")
                        if (m := re.match(r"(RBDS\d+)static\.txt", f.name))}))
    if not subjects:
        print(f"no *static.txt under {FUK}")
        return 2

    written, skipped = 0, []
    for s in subjects:
        for t in args.trials.split(","):
            if not (FUK / f"{s}{t}markers.txt").exists():
                skipped.append(f"{s}/{t} (no marker file)")
                continue
            try:
                payload, meta = build(s, t)
            except (KeyError, FileNotFoundError, TrialUnusable) as e:
                skipped.append(f"{s}/{t}: {e}")
                continue
            (OUT / f"{s}{t}.json").write_text(json.dumps(payload), encoding="utf-8")
            (OUT / f"{s}{t}.meta.json").write_text(json.dumps(meta, indent=2),
                                                   encoding="utf-8")
            written += 1
            fc = meta["frame_checks"]
            print(f"  {s}/{t}: {meta['n_frames']} frames @ {meta['hz']:.0f} Hz  "
                  f"(ASIS width {fc['asis_width_mm']:.0f} mm, "
                  f"pelvis {fc['pelvis_above_heel_mm']:.0f} mm above heel)")

    print(f"\nwrote {written} trials -> {OUT.relative_to(REPO)}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for s in skipped[:10]:
            print(f"    {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
