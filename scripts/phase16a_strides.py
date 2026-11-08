"""M3 validation: does the stride segmentation find the right gait cycles?

`docs/phase16-demo-design.md` milestone M3. Stride segmentation is **the only
component of the demo that no earlier phase measured**, so it is validated here
before it is allowed into a report, and the design's fallback stands if it
fails: per-frame traces only.

Three things are measured.

  **1. The rejected candidate: heel strike from the anterior ankle offset.**
  The design asked for heel-strike detection. The standard monocular substitute
  for a force plate is Zeni et al. (2008) -- heel strike is the frame of maximum
  *anterior* ankle offset from the pelvis -- and "anterior" needs the direction
  of travel. This part measures whether that direction is readable from
  running's duty factor (stance is the shorter half-cycle, because running has
  a flight phase). **It is not, on this footage**, and the numbers are recorded
  rather than the attempt being deleted.

  **2. The shipped method, against the mocap, all 11 spent subjects (cheap).**
  Cycles bounded at peak swing-phase knee flexion, found with a hysteresis
  trigger, run on the **mocap** 3D. The left/right count agreement is the
  consistency check that needs no ground truth at all: two legs necessarily
  take the same number of steps.

  **3. The estimate, 3 clips (expensive: ~680 ms/frame).**
  The full demo path (detect -> lift -> segment) on three subjects spanning the
  sealed test's range, scored against part 2's count. This is the M3 bar:
  **within +/-1 stride on 3 clips.**

The second M3 bar -- per-stride knee peaks varying less within a subject than
knee peaks vary across subjects -- is measured too. It is what makes a stride
average worth showing: if stride-to-stride spread were as large as
between-subject spread, the average would be averaging noise.

All 11 subjects are SPENT (phase 15 and the sealed test opened them), so this
establishes no held-out claim. Output is AGGREGATE ONLY: no per-subject MoVi
value, frame or anthropometric quantity is written to the repository.

Run: .venv/Scripts/python.exe scripts/phase16a_strides.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from overstride.analyze import analyze_frames  # noqa: E402
from overstride.strides import (CYCLE_SIGNAL, HYST_FRAC,  # noqa: E402
                                MAX_COUNT_DIFF, MIN_KNEE_SWING_DEG,
                                MIN_PLAUSIBLE_SHARE, MIN_STRIDES,
                                STRIDE_S_MAX, STRIDE_S_MIN,
                                cv, cycle_events, dominant_period,
                                fill_short_gaps, peaks_per_stride, segment)
from phase15_checks import (camera_pose, load_v3d, to_camera,  # noqa: E402
                            to_h36m)
from phase15_movi import (CAMERAS, MOCAP, MOCAP_PER_VIDEO, MOVI,  # noqa: E402
                          decode_range, jogging_window)
from phase15d_segments import segment_angles  # noqa: E402
from phase3_angles import H36M  # noqa: E402

OUT = REPO / "results" / "phase16a_strides.json"
CAM = "PG1"
SUBJECTS = ("Subject_1", "Subject_10", "Subject_11", "Subject_12",
            "Subject_13", "Subject_15", "Subject_16", "Subject_17",
            "Subject_18", "Subject_19", "Subject_2")

# Fixed before the run: the best-behaved sealed subject, the worst, and one
# diagnostic subject in between.
ESTIMATE_CLIPS = ("Subject_17", "Subject_2", "Subject_1")

REF_UP = np.array([0.0, 0.0, 1.0])      # MoVi world vertical, phase 15 check 2
BARS = "_.-~=*#@"


def sparkline(x: np.ndarray, width: int = 100) -> str:
    """A text trace, so cycles can be counted by eye from the log."""
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return "(no signal)"
    y = y[np.linspace(0, y.size - 1, min(width, y.size)).astype(int)]
    lo, hi = float(y.min()), float(y.max())
    if hi - lo < 1e-12:
        return BARS[0] * y.size
    q = ((y - lo) / (hi - lo) * (len(BARS) - 1)).round().astype(int)
    return "".join(BARS[i] for i in q)


# ------------------------------------- the REJECTED candidate -------------

def anterior_offset(kp3d: np.ndarray, side: str, sign: int = 1) -> np.ndarray:
    """Zeni's signal: the ankle's horizontal offset from the pelvis.

    Kept here, not in `overstride/strides.py`, because it is not what ships.
    """
    kp = np.asarray(kp3d, dtype=float)
    a = kp[:, H36M[f"{side}_ankle"], 0] - kp[:, H36M["pelvis"], 0]
    trunk = np.linalg.norm(kp[:, H36M["thorax"], :] - kp[:, H36M["pelvis"], :],
                           axis=1)
    scale = float(np.nanmedian(trunk))
    return int(sign) * a / (scale if np.isfinite(scale) and scale > 1e-9 else 1.0)


def duty_fraction(kp3d: np.ndarray, fps: float) -> float:
    """Median share of the cycle from the anterior maximum to the next minimum.

    ~0.35 if the raw signal already points anterior (stance is shorter), ~0.65
    if it points the other way. **Near 0.5 means the direction of travel cannot
    be read**, which is what the measured footage gives.
    """
    fracs = []
    for side in ("l", "r"):
        x = fill_short_gaps(anterior_offset(kp3d, side), fps)
        period = dominant_period(x, fps)
        if not np.isfinite(period):
            continue
        mx = cycle_events(x, fps)
        mn = cycle_events(-x, fps)
        for i in mx:
            nxt = mn[mn > i]
            if nxt.size:
                fracs.append(float(nxt[0] - i) / period)
    return float(np.median(fracs)) if fracs else float("nan")


def pelvis_excursion(kp3d: np.ndarray, fps: float) -> float:
    """Pelvis travel across the camera's x axis, in trunk lengths."""
    kp = np.asarray(kp3d, dtype=float)
    trunk = float(np.nanmedian(np.linalg.norm(
        kp[:, H36M["thorax"]] - kp[:, H36M["pelvis"]], axis=1)))
    if not np.isfinite(trunk) or trunk < 1e-9:
        return float("nan")
    px = fill_short_gaps(kp[:, H36M["pelvis"], 0] / trunk, fps, 1e9)
    px = px[np.isfinite(px)]
    return float(px.max() - px.min()) if px.size else float("nan")


def movi_window(subject: str):
    """The subject's jogging window: frames, fps, mocap reference (world)."""
    mat = MOCAP / f"F_v3d_{subject}.mat"
    video = MOVI / CAMERAS[CAM]["pattern"].format(s=subject.split("_")[-1])
    if not (mat.exists() and video.exists()):
        return None
    v = load_v3d(mat)
    _ji, f30, f120 = jogging_window(v)
    ref_all = to_h36m(v)
    frames, fps, _m = decode_range(video, int(f30[0]), int(f30[1]))
    mk = int(f120[0]) + MOCAP_PER_VIDEO * np.arange(len(frames))
    mk = mk[mk < ref_all.shape[0]]
    return list(frames)[:len(mk)], float(fps), ref_all[mk]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject-out", type=Path, default=None,
                    help="optional path OUTSIDE the repo for per-subject rows")
    ap.add_argument("--clips", type=int, default=len(ESTIMATE_CLIPS),
                    help="how many of the 3 estimate clips to run")
    args = ap.parse_args()

    R_cv, t_cv, _s = camera_pose(CAM)
    rows: list[dict] = []

    print("=== PART 1+2: the mocap ground truth, all spent subjects ===")
    print("    cycles bounded at peak swing-phase knee flexion (shipped),")
    print("    against Zeni's anterior-offset heel strike (rejected)\n")
    print(f"  {'subject':<12}{'secs':>6}{'cycL':>5}{'cycR':>5}{'|d|':>4}"
          f"{'used':>6}{'dur':>6}{'cv':>6}{'amp':>6}{'excur':>7}{'duty':>6}"
          f"{'apdiff':>7}{'ok':>5}")

    for subject in SUBJECTS:
        w = movi_window(subject)
        if w is None:
            print(f"  {subject}: missing files, skipped")
            continue
        frames, fps, ref = w
        del frames                                 # part 1+2 need no pixels
        gc = to_camera(ref, R_cv, t_cv)
        gt_ang = {s: segment_angles(ref, s, REF_UP) for s in ("l", "r")}
        st = segment(gt_ang, fps)

        # the rejected candidate, on the same clip
        duty = duty_fraction(gc, fps)
        ap_counts = {s: max(cycle_events(
            fill_short_gaps(anterior_offset(gc, s), fps), fps).size - 1, 0)
            for s in ("l", "r")}
        ap_diff = max(abs(ap_counts[s] - st.count[s]) for s in ("l", "r"))

        knee_peaks = {s: peaks_per_stride(gt_ang[s][CYCLE_SIGNAL], st.used[s])
                      for s in ("l", "r")}
        durs = [d for s in ("l", "r") for d in st.durations_s[s]]
        row = {
            "subject": subject, "secs": ref.shape[0] / fps,
            "count": dict(st.count), "n_used": dict(st.n_used),
            "count_diff": abs(st.count["l"] - st.count["r"]),
            "ok": bool(st.ok), "reasons": st.reasons,
            "mean_duration_s": float(np.mean(durs)) if durs else float("nan"),
            "duration_cv": {s: st.duration_cv(s) for s in ("l", "r")},
            "plausible_share": {s: st.plausible_share(s) for s in ("l", "r")},
            "amplitude_deg": {s: float(st.amplitude_deg[s]) for s in ("l", "r")},
            "cadence_spm": st.cadence_spm,
            "knee_peak_mean": {s: (float(np.nanmean(knee_peaks[s]))
                                   if knee_peaks[s].size else float("nan"))
                               for s in ("l", "r")},
            "knee_peak_cv": {s: cv(knee_peaks[s]) for s in ("l", "r")},
            # the rejected candidate
            "duty_fraction": duty,
            "pelvis_excursion_trunks": pelvis_excursion(gc, fps),
            "anterior_offset_count_diff": int(ap_diff),
        }
        rows.append(row)
        print(f"  {subject:<12}{row['secs']:>6.1f}"
              f"{st.count['l']:>5}{st.count['r']:>5}{row['count_diff']:>4}"
              f"{st.n_used['l']:>3}/{st.n_used['r']:<2}"
              f"{row['mean_duration_s']:>6.2f}"
              f"{row['duration_cv']['l']:>6.2f}"
              f"{row['amplitude_deg']['l']:>6.0f}"
              f"{row['pelvis_excursion_trunks']:>7.1f}{duty:>6.2f}"
              f"{ap_diff:>7}{('yes' if st.ok else 'NO'):>5}")
        if not st.ok:
            for r in st.reasons:
                print(f"       - {r}")

    print("\n  the mocap knee-flexion trace, left leg, so cycles can be")
    print("  counted by hand from this log:")
    for subject in ESTIMATE_CLIPS:
        w = movi_window(subject)
        if w is None:
            continue
        _f, fps, ref = w
        sig = segment_angles(ref, "l", REF_UP)[CYCLE_SIGNAL]
        ev = cycle_events(sig, fps)
        print(f"    {subject:<12}{sparkline(sig)}  ({ev.size} events)")

    # ---------------- part 3: the estimate on three clips -----------------
    print(f"\n=== PART 3: the demo path on {args.clips} clip(s) "
          f"(~680 ms/frame) ===")
    lifter = None
    if args.clips > 0:
        from phase3_infer import load_model
        from phase7_inference_fix import CKPT_FOR
        lifter = load_model(CKPT_FOR["coco"], "base")

    est_rows: list[dict] = []
    for subject in ESTIMATE_CLIPS[:args.clips]:
        w = movi_window(subject)
        if w is None:
            continue
        frames, fps, ref = w
        gc = to_camera(ref, R_cv, t_cv)
        gt = segment({s: segment_angles(ref, s, REF_UP) for s in ("l", "r")},
                     fps)
        near = "left" if (np.nanmean(gc[:, H36M["l_hip"], 2])
                          < np.nanmean(gc[:, H36M["r_hip"], 2])) else "right"
        res = analyze_frames(frames, fps, lifter, near_side=near)
        st = segment(res.angles, fps, detected=res.detected)
        knee = {s: peaks_per_stride(res.series(s, CYCLE_SIGNAL), st.used[s])
                for s in ("l", "r")}
        err = {s: int(st.count[s] - gt.count[s]) for s in ("l", "r")}
        est_rows.append({
            "subject": subject,
            "gt_count": dict(gt.count), "est_count": dict(st.count),
            "count_error": err,
            "within_1": all(abs(v) <= 1 for v in err.values()),
            "est_ok": bool(st.ok), "est_reasons": st.reasons,
            "n_used": dict(st.n_used),
            "duration_cv": {s: st.duration_cv(s) for s in ("l", "r")},
            "amplitude_deg": {s: float(st.amplitude_deg[s]) for s in ("l", "r")},
            "knee_peak_cv": {s: cv(knee[s]) for s in ("l", "r")},
            "knee_peak_mean": {s: (float(np.nanmean(knee[s]))
                                   if knee[s].size else float("nan"))
                               for s in ("l", "r")},
            "cadence_spm": st.cadence_spm,
        })
        e = est_rows[-1]
        print(f"  {subject:<12}gt L{gt.count['l']} R{gt.count['r']}   "
              f"est L{st.count['l']} R{st.count['r']}   "
              f"err {err['l']:+d}/{err['r']:+d}   "
              f"{'WITHIN 1' if e['within_1'] else 'OUTSIDE 1'}   "
              f"amp {e['amplitude_deg']['l']:.0f}/"
              f"{e['amplitude_deg']['r']:.0f} deg   "
              f"ok {'yes' if st.ok else 'NO'}")
        if not st.ok:
            for r in st.reasons:
                print(f"       - {r}")

    # ---------------- aggregates ------------------------------------------
    def both(key):
        return [float(r[key][s]) for r in rows for s in ("l", "r")
                if np.isfinite(r[key][s])]

    def one(key, src=None):
        return [float(r[key]) for r in (src or rows) if np.isfinite(r[key])]

    counts = [float(r["count"][s]) for r in rows for s in ("l", "r")]
    cvs, shares = both("duration_cv"), both("plausible_share")
    amps, knee_cv = both("amplitude_deg"), both("knee_peak_cv")
    peak_means = both("knee_peak_mean")
    agree = sum(1 for r in rows if r["count_diff"] <= MAX_COUNT_DIFF)
    duties, excur = one("duty_fraction"), one("pelvis_excursion_trunks")
    # a value above 1 means the max->min pairing spans more than a whole cycle,
    # i.e. the method is incoherent on that clip, not merely ambiguous
    readable = [f for f in duties if f <= 1.0 and abs(f - 0.5) >= 0.07]
    ap_diffs = one("anterior_offset_count_diff")

    across = (float(np.std(peak_means, ddof=1) / abs(np.mean(peak_means)))
              if len(peak_means) > 1 else float("nan"))
    within = float(np.median(knee_cv)) if knee_cv else float("nan")
    hits = sum(1 for r in est_rows if r["within_1"])

    print("\n=== AGGREGATES ===")
    print(f"  left/right cycle counts agree within {MAX_COUNT_DIFF} on "
          f"{agree}/{len(rows)} subjects")
    print(f"  cycles per leg {int(min(counts))}-{int(max(counts))}, "
          f"mean stride {min(one('mean_duration_s')):.2f}-"
          f"{max(one('mean_duration_s')):.2f} s, "
          f"knee swing {min(amps):.0f}-{max(amps):.0f} deg")
    print(f"  duration CV {min(cvs):.2f}-{max(cvs):.2f} (reported, not a gate)")
    print(f"  plausible share {min(shares):.2f}-{max(shares):.2f} "
          f"(bar {MIN_PLAUSIBLE_SHARE})")
    print(f"  knee-peak CV within a subject, median {within:.4f} vs ACROSS "
          f"subjects {across:.4f} -> {'PASS' if within < across else 'FAIL'}")
    print(f"\n  REJECTED candidate (Zeni anterior offset):")
    print(f"    duty fraction {min(duties):.2f}-{max(duties):.2f}; "
          f"only {len(readable)}/{len(duties)} readable "
          f"(0.35 or 0.65 reads the direction; ~0.5 cannot, and >1 means the "
          f"pairing spans more than a cycle)")
    print(f"    pelvis excursion {min(excur):.1f}-{max(excur):.1f} trunk "
          f"lengths -- subjects traverse the volume and turn around")
    print(f"    its cycle count differs from the shipped method by up to "
          f"{int(max(ap_diffs))} strides")
    print(f"\n  M3 count bar: {hits}/{len(est_rows)} clips within +/-1 stride")

    out = {
        "what": ("M3 stride segmentation, validated on the 11 SPENT MoVi PG1 "
                 "subjects. Ground truth from the mocap 3D; the estimate path "
                 "on 3 clips."),
        "shipped_method": {
            "cycle_signal": CYCLE_SIGNAL,
            "boundary": "peak swing-phase knee flexion",
            "detector": ("hysteresis (Schmitt) trigger, band "
                         f"{HYST_FRAC} of the 5-95 percentile amplitude; no "
                         "period estimate, so a changing cadence cannot break "
                         "it"),
            "why_not_heel_strike": (
                "H36M-17 has no foot or heel keypoint, so there is no "
                "ground-contact signal; the standard substitute (Zeni et al. "
                "2008, maximum anterior ankle offset) needs the direction of "
                "travel, and that direction is not readable on this footage "
                "-- see rejected_candidate below"),
        },
        "rejected_candidate": {
            "method": ("Zeni et al. 2008: heel strike = maximum anterior "
                       "ankle offset from the pelvis"),
            "needs": "the direction of travel across the image",
            "direction_read_from": ("running's duty factor -- stance is the "
                                    "shorter half-cycle, because running has "
                                    "a flight phase"),
            "duty_fraction_range": [round(min(duties), 3), round(max(duties), 3)],
            "duty_fraction_median": round(float(np.median(duties)), 3),
            "readable_would_be": "about 0.35 or 0.65; ~0.5 is unreadable",
            "n_readable": len(readable),
            "n_subjects": len(duties),
            "above_one_means": ("the max->min pairing spans more than a whole "
                                "cycle, so the method is incoherent on that "
                                "clip rather than merely ambiguous"),
            "pelvis_excursion_trunk_lengths": [round(min(excur), 2),
                                               round(max(excur), 2)],
            "why_it_fails": ("MoVi subjects traverse the capture volume and "
                             "turn around inside the window, so anterior "
                             "reverses mid-clip and the statistic collapses "
                             "to ~0.5"),
            "max_cycle_count_difference_vs_shipped": int(max(ap_diffs)),
        },
        "diagnostic_only": ("all 11 subjects are spent; this establishes no "
                            "held-out claim"),
        "aggregate_only": ("no per-subject MoVi value is written here, same "
                           "standard as the scoping files"),
        "n_subjects": len(rows),
        "ground_truth": {
            "source": "120 Hz Qualisys mocap, sampled at the video frame times",
            "consistency_check": ("left/right cycle counts, which need no "
                                  "ground truth: two legs necessarily take the "
                                  "same number of steps"),
            "subjects_agreeing_within_1": agree,
            "cycles_per_leg": [int(min(counts)), int(max(counts))],
            "mean_stride_duration_s": [round(min(one("mean_duration_s")), 3),
                                       round(max(one("mean_duration_s")), 3)],
            "knee_swing_amplitude_deg": [round(min(amps), 1),
                                         round(max(amps), 1)],
            "duration_cv": [round(min(cvs), 3), round(max(cvs), 3)],
            "plausible_share": [round(min(shares), 3), round(max(shares), 3)],
            "cadence_spm": [round(min(one("cadence_spm")), 1),
                            round(max(one("cadence_spm")), 1)],
        },
        "bars_in_overstride_strides": {
            "MIN_STRIDES": MIN_STRIDES,
            "MAX_COUNT_DIFF": MAX_COUNT_DIFF,
            "MIN_KNEE_SWING_DEG": MIN_KNEE_SWING_DEG,
            "MIN_PLAUSIBLE_SHARE": MIN_PLAUSIBLE_SHARE,
            "STRIDE_S_RANGE": [STRIDE_S_MIN, STRIDE_S_MAX],
            "note": ("each bar sits outside the measured spread above, so it "
                     "rejects a mis-segmentation rather than normal stride "
                     "variability. Duration CV is REPORTED, not gated: a "
                     "runner who decelerates inside the clip has a high CV "
                     "and nothing is wrong"),
        },
        "knee_peak_spread": {
            "within_subject_cv_median": round(within, 4),
            "within_subject_cv_range": ([round(min(knee_cv), 4),
                                         round(max(knee_cv), 4)]
                                        if knee_cv else None),
            "across_subject_cv": round(across, 4),
            "bar_met": bool(within < across),
            "why_it_matters": ("if stride-to-stride spread were as large as "
                               "between-subject spread, a stride average "
                               "would be averaging noise"),
        },
        "estimate_vs_ground_truth": {
            "n_clips": len(est_rows),
            "clips_within_1_stride": hits,
            "bar": "+/-1 stride on 3 clips",
            "bar_met": bool(est_rows and hits == len(est_rows)),
            "count_error_range": ([min(v for r in est_rows
                                       for v in r["count_error"].values()),
                                   max(v for r in est_rows
                                       for v in r["count_error"].values())]
                                  if est_rows else None),
            "clips_passing_their_own_check": sum(1 for r in est_rows
                                                 if r["est_ok"]),
            "knee_swing_amplitude_deg": ([round(min(float(r["amplitude_deg"][s])
                                                    for r in est_rows
                                                    for s in ("l", "r")), 1),
                                          round(max(float(r["amplitude_deg"][s])
                                                    for r in est_rows
                                                    for s in ("l", "r")), 1)]
                                         if est_rows else None),
        },
        "fallback": ("when Strides.ok is False the report shows per-frame "
                     "traces only and no stride table; stride averaging is a "
                     "convenience, never part of the error budget"),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    try:
        shown = OUT.resolve().relative_to(REPO)
    except ValueError:
        shown = OUT
    print(f"\nwrote {shown}  (aggregate only)")

    if args.per_subject_out:
        dest = Path(args.per_subject_out).resolve()
        if REPO in dest.parents or dest.parent == REPO:
            print("  REFUSED: --per-subject-out is inside the repository")
            return 3
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"ground_truth": rows,
                                    "estimate": est_rows}, indent=2,
                                   default=float), encoding="utf-8")
        print(f"  per-subject rows -> {dest}  (outside the repo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
