"""READ-ONLY scoping: is there a usable STANDING pose in MoVi round F?

No model is run, no new subject is opened -- this reads only the twelve round F
subjects already on disk, all of which have been opened by phases 15/15B/15D or
the sealed test. It answers, from the mocap alone:

  1. is there a motion named standing / neutral / T-pose / calibration?  (no)
  2. which motions are nonetheless LOWER-LIMB STATIONARY, i.e. performed from
     a standing posture with the legs still, and how long are they?
  3. are the inter-motion gaps -- the frames belonging to no motion window --
     stationary, and how long are they?

Stationarity is measured as the speed of the lower-limb markers the angle
pipeline actually reads, in mm/s at the mocap's 120 Hz, plus the pelvis's
horizontal drift. Nothing is fitted and no angle error is computed.

Output: results/phase15_standing_scope.json

Run: .venv/Scripts/python.exe scripts/phase15_standing_scope.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from phase15_checks import load_v3d  # noqa: E402

MOCAP = REPO / "data" / "movi" / "raw" / "round_f" / "mocap"
OUT = REPO / "results" / "phase15_standing_scope.json"
MOCAP_HZ = 120.0
VIDEO_FPS = 30.0

# The markers the sagittal pipeline's reference frame and limb segments read.
# Hips are mean(ASI, PSI) per side, as H36M defines them.
LOWER = ["LFWT", "LBWT", "RFWT", "RBWT", "LKJC", "RKJC", "LAJC", "RAJC"]
PELVIS = ["LFWT", "LBWT", "RFWT", "RBWT"]

# A pose is only useful as a calibration reference if the legs are still for
# long enough to average several frames. These are descriptive thresholds for
# the report, NOT a pre-registered gate -- nothing is accepted or rejected here.
STILL_MM_S = 100.0      # lower-limb marker speed below this = "legs still"
MIN_SECONDS = 1.0


def marker_idx(v: dict, names: list[str]) -> dict[str, int]:
    lut = {n.strip(): i for i, n in enumerate(v["marker_names"])}
    return {n: lut[n] for n in names if n in lut}


def gapped(v: dict) -> np.ndarray:
    """(frames, markers) True where Visual3D wrote the [0,0,0] sentinel."""
    g = np.asarray(v["marker_gaps"])
    return g == 1


def speeds(v: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame lower-limb marker speed (mm/s) and pelvis horizontal speed."""
    loc = v["marker_location"]                      # (frames, markers, 3)
    idx = marker_idx(v, LOWER)
    gap = gapped(v)

    xyz = loc[:, list(idx.values()), :].astype(float)
    bad = gap[:, list(idx.values())]
    xyz[bad] = np.nan

    d = np.diff(xyz, axis=0) * MOCAP_HZ             # mm/s
    sp = np.linalg.norm(d, axis=2)                  # (frames-1, markers)
    limb = np.nanmedian(sp, axis=1)                 # median over markers

    pidx = marker_idx(v, PELVIS)
    p = loc[:, list(pidx.values()), :].astype(float)
    p[gap[:, list(pidx.values())]] = np.nan
    centre = np.nanmean(p, axis=1)                  # (frames, 3)
    # axis 2 is vertical (phase 15 check 2), so 0 and 1 are the floor plane
    pel = np.linalg.norm(np.diff(centre[:, :2], axis=0), axis=1) * MOCAP_HZ

    return limb, pel


def longest_still_run(limb: np.ndarray, lo: int, hi: int) -> dict:
    """Longest consecutive stretch inside [lo, hi) with legs below threshold."""
    seg = limb[lo:max(lo + 1, min(hi, len(limb)))]
    ok = np.isfinite(seg) & (seg < STILL_MM_S)
    best = cur = 0
    best_end = 0
    for i, flag in enumerate(ok):
        cur = cur + 1 if flag else 0
        if cur > best:
            best, best_end = cur, i
    return {"frames_120hz": int(best),
            "seconds": round(best / MOCAP_HZ, 2),
            "start_120hz": int(lo + best_end - best + 1) if best else None,
            "fraction_of_window_still": (round(float(ok.mean()), 3)
                                         if len(ok) else None)}


def analyse(path: Path) -> dict:
    v = load_v3d(path)
    sid = v["subject_id"]
    n = v["marker_location"].shape[0]
    limb, pel = speeds(v)

    row: dict = {"subject_id": sid, "n_frames_mocap": int(n),
                 "n_frames_video_implied": int(round(n / 4)),
                 "has_motions_list": "motions" in v}
    if "motions" not in v:
        return row

    motions = [m.strip() for m in v["motions"]]
    f120 = np.asarray(v["flags120"], dtype=int)
    f30 = np.asarray(v["flags30"], dtype=int)

    per = []
    for i, name in enumerate(motions):
        lo, hi = int(f120[i][0]), int(f120[i][1])
        seg_limb = limb[lo:min(hi, len(limb))]
        seg_pel = pel[lo:min(hi, len(pel))]
        per.append({
            "index_0based": i, "motion": name,
            "mocap_120hz": [lo, hi],
            "video_30fps": [int(f30[i][0]), int(f30[i][1])],
            "seconds": round((hi - lo) / MOCAP_HZ, 2),
            "limb_speed_mm_s": {
                "median": round(float(np.nanmedian(seg_limb)), 1),
                "p10": round(float(np.nanpercentile(seg_limb, 10)), 1),
                "p90": round(float(np.nanpercentile(seg_limb, 90)), 1)},
            "pelvis_floor_speed_mm_s_median":
                round(float(np.nanmedian(seg_pel)), 1),
            "longest_legs_still": longest_still_run(limb, lo, hi),
        })
    row["motions"] = per

    # ---- the frames belonging to NO motion window -------------------------
    covered = np.zeros(len(limb), dtype=bool)
    for i in range(len(motions)):
        lo, hi = int(f120[i][0]), int(f120[i][1])
        covered[lo:min(hi, len(covered))] = True
    gaps = []
    i = 0
    while i < len(covered):
        if covered[i]:
            i += 1
            continue
        j = i
        while j < len(covered) and not covered[j]:
            j += 1
        seg = limb[i:j]
        if (j - i) / MOCAP_HZ >= MIN_SECONDS:
            gaps.append({
                "mocap_120hz": [int(i), int(j)],
                "video_30fps": [int(round(i / 4)), int(round(j / 4))],
                "seconds": round((j - i) / MOCAP_HZ, 2),
                "limb_speed_mm_s_median":
                    (round(float(np.nanmedian(seg)), 1)
                     if np.isfinite(seg).any() else None),
                "longest_legs_still": longest_still_run(limb, i, j),
                "before_motion": (motions[int(np.argmin(
                    [abs(int(f120[k][0]) - j) for k in range(len(motions))]))]
                    if len(motions) else None),
            })
        i = j
    row["uncovered_runs_over_1s"] = gaps
    row["n_uncovered_frames"] = int((~covered).sum())
    row["fraction_uncovered"] = round(float((~covered).mean()), 3)
    row["longest_still_anywhere"] = longest_still_run(limb, 0, len(limb))
    return row


def aggregate(rows: list[dict]) -> dict:
    """Cross-subject aggregates. NOTHING per-subject is returned."""
    import collections
    per_motion = collections.defaultdict(list)
    for r in rows:
        for m in r.get("motions", []):
            per_motion[m["motion"]].append(
                (m["longest_legs_still"]["seconds"],
                 m["limb_speed_mm_s"]["median"], m["seconds"]))

    motions = {}
    for name, v in per_motion.items():
        still = [x[0] for x in v]
        speed = [x[1] for x in v]
        dur = [x[2] for x in v]
        motions[name] = {
            "n_subjects": len(v),
            "still_seconds": {"median": round(float(np.median(still)), 2),
                              "min": round(float(min(still)), 2),
                              "max": round(float(max(still)), 2)},
            "limb_speed_mm_s_median_of_medians":
                round(float(np.median(speed)), 1),
            "window_seconds_median": round(float(np.median(dur)), 2)}

    unc = [r["fraction_uncovered"] for r in rows if "fraction_uncovered" in r]
    rest = [max((g["longest_legs_still"]["seconds"]
                 for g in r.get("uncovered_runs_over_1s", [])), default=0.0)
            for r in rows]
    anywhere = [r["longest_still_anywhere"]["seconds"] for r in rows
                if "longest_still_anywhere" in r]
    return {
        "n_subjects": len(rows),
        "per_motion": dict(sorted(
            motions.items(),
            key=lambda kv: -kv[1]["still_seconds"]["median"])),
        "fraction_of_frames_in_no_motion_window": {
            "min": round(float(min(unc)), 3),
            "max": round(float(max(unc)), 3)},
        "longest_inter_motion_rest_still_seconds": {
            "min": round(float(min(rest)), 2),
            "max": round(float(max(rest)), 2)},
        "longest_legs_still_anywhere_seconds": {
            "min": round(float(min(anywhere)), 2),
            "max": round(float(max(anywhere)), 2)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject-out", type=Path, default=None,
                    help="optional path OUTSIDE the repository for the "
                         "per-subject detail; MoVi's licence keeps it out of "
                         "this repo")
    args = ap.parse_args()

    mats = sorted(MOCAP.glob("F_v3d_*.mat"))
    if not mats:
        print("no round F mocap on disk")
        return 2
    print(f"=== READ-ONLY standing-pose scope, {len(mats)} opened subjects ===")
    print(f"    still = lower-limb marker median speed < {STILL_MM_S:.0f} mm/s "
          f"at {MOCAP_HZ:.0f} Hz\n")
    rows = []
    for p_ in mats:                     # one subject at a time, never all
        r = analyse(p_)
        rows.append(r)
        if "motions" in r:
            best = max(r["motions"],
                       key=lambda m: m["longest_legs_still"]["frames_120hz"])
            print(f"  {r['subject_id']:<12} longest still inside a motion: "
                  f"{best['longest_legs_still']['seconds']:5.2f} s "
                  f"({best['motion']}), "
                  f"uncovered {r['fraction_uncovered'] * 100:4.1f}%, "
                  f"longest still anywhere "
                  f"{r['longest_still_anywhere']['seconds']:5.2f} s")

    agg = aggregate(rows)
    print(f"\n  {'motion':<22}{'n':>3}{'median still s':>16}{'min':>7}"
          f"{'limb mm/s':>11}")
    for name, a in list(agg["per_motion"].items())[:9]:
        print(f"  {name:<22}{a['n_subjects']:>3}"
              f"{a['still_seconds']['median']:>16.2f}"
              f"{a['still_seconds']['min']:>7.2f}"
              f"{a['limb_speed_mm_s_median_of_medians']:>11.1f}")

    out = {"what": ("READ-ONLY scope for a per-subject offset calibration: is "
                    "there a standing / neutral pose in MoVi round F? No model "
                    "was run and no new subject was opened."),
           "licence": ("MoVi forbids distributing the Data in any form. This "
                       "file carries CROSS-SUBJECT AGGREGATES ONLY -- no "
                       "per-subject measurement, no frame window, no marker "
                       "coordinate, no camera parameter. Per-subject detail is "
                       "reproducible with --per-subject-out, which must point "
                       "outside this repository."),
           "still_threshold_mm_s": STILL_MM_S,
           "mocap_hz": MOCAP_HZ, "video_fps": VIDEO_FPS,
           "markers_used": LOWER,
           "no_motion_named_standing": True,
           "motion_names_present": sorted(agg["per_motion"]),
           "aggregate": agg}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}  (aggregates only)")

    if args.per_subject_out:
        dest = Path(args.per_subject_out).resolve()
        if REPO in dest.parents or dest.parent == REPO:
            print("  REFUSED: --per-subject-out is inside the repository")
            return 3
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"subjects": rows}, indent=2),
                        encoding="utf-8")
        print(f"  per-subject detail -> {dest}  (outside the repo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
