"""M3: stride segmentation -- gait cycles, stride count, stride averages.

`docs/phase16-demo-design.md` section 5 stage 9, milestone M3. **This is the
only component of the demo that no earlier phase measured**, so it carries its
own check and its own fallback: when the check fails, `Strides.ok` is False and
the report shows **per-frame traces only**. Stride averaging is a convenience,
never part of the error budget.

## `[M3]` The cycle boundary is peak swing-phase knee flexion, NOT heel strike

The design asked for heel-strike detection. **It is not available on this
pipeline, and the attempt is recorded rather than hidden.**

* There is **no foot or heel keypoint.** H36M-17's most distal joint is the
  ankle, so there is no foot-contact signal to threshold -- the same limitation
  phase 14 recorded for the foot segment.
* The standard substitute is the **coordinate-based method of Zeni et al.
  (2008)**: heel strike is the frame where the foot is maximally *anterior*
  relative to the pelvis. Anterior is a direction, so that method needs to know
  which way the runner travels across the image.
* **That direction is not reliably knowable here.** `overstride.nearside`
  abstains on 5 of 11 MoVi clips. Reading it instead from running's duty factor
  -- stance is the shorter half-cycle, because running has a flight phase --
  failed on the measured footage: MoVi subjects traverse the capture volume and
  **turn around inside the window** (measured pelvis excursion 3.5-6.9 trunk
  lengths), so anterior reverses mid-clip and the statistic lands at ~0.5.
  `scripts/phase16a_strides.py` keeps that candidate and reports its numbers.

So the cycle is bounded at the **peak of knee flexion**, which is the sharpest,
highest-amplitude event in the running cycle (50-70 deg of swing) and needs no
direction at all. Measured against the mocap on the 11 spent MoVi PG1 subjects,
this boundary gives **left/right stride counts agreeing within one on 11 of
11** -- a consistency check that needs no ground truth, because two legs
necessarily take the same number of steps -- where the anterior-offset method
disagreed by up to 3 strides.

**The consequence is stated in the report, not papered over.** The stride axis
reads *"% of stride cycle (0 = peak swing-phase knee flexion)"*, and the demo
never claims a heel strike, a ground-contact time, or a stance/swing split.

## The detector: a hysteresis trigger, not peak picking

Peak picking with a refractory period needs a period estimate, and the period
is not constant when a runner decelerates and turns. A **Schmitt trigger**
needs neither: with a band placed around the signal's own mid-range, one cycle
is counted each time the signal rises above the high edge after having fallen
below the low edge, and the event is the signal's maximum inside that
excursion. Small wobbles and double bumps cannot re-trigger it, and a changing
cadence does not matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------- the bars -------
# Every bar below is set from the distribution measured over the 11 SPENT MoVi
# PG1 subjects in `scripts/phase16a_strides.py`, recorded as aggregates in
# `results/phase16a_strides.json`. No MoVi value appears in this file.

# A plausible stride cycle (same leg to same leg) at jogging and running pace.
# Deliberately wide: a plausibility bound, not a fitted window. Measured mean
# stride duration on the spent subjects falls inside it with margin at both
# ends, and it excludes a detector artefact at 3 Hz or a standing pause.
STRIDE_S_MIN, STRIDE_S_MAX = 0.40, 1.40

# Fewer than this and a stride average is averaging almost nothing.
MIN_STRIDES = 3

# The two legs necessarily take the same number of steps, so a bigger gap than
# this means one leg's events are wrong. MEASURED: within one on 11 of 11.
MAX_COUNT_DIFF = 1

# The trigger band, as a fraction of the signal's 5-95 percentile amplitude.
# A quarter of the amplitude is wide enough to ignore the secondary bump in the
# knee trace (stance flexion, roughly a tenth of the swing peak) and narrow
# enough that a real cycle always crosses it.
HYST_FRAC = 0.25

# The knee must actually swing, or the clip is not running and the trace's
# maxima are noise. Set well below the measured range, which is recorded in
# results/phase16a_strides.json for both the mocap and the estimate.
MIN_KNEE_SWING_DEG = 25.0

# Cycles whose duration falls outside STRIDE_S_* are counted but not averaged.
# If most of them are implausible, the signal is not gait: this is the share
# that must survive.
MIN_PLAUSIBLE_SHARE = 0.60

# NOTE ON TRIMMING, which was tried and dropped. Excluding strides more than
# 25% from the clip's median duration -- to keep transition strides out of the
# average -- removed so many that 5 of the 11 measured subjects fell below
# MIN_STRIDES, because MoVi's own footage is not steady-state (duration CV
# 0.14-0.47 as subjects traverse the volume and turn). Every stride inside
# STRIDE_S_* is therefore averaged, and because each one is time-normalized to
# its OWN cycle, a longer stride is not misaligned -- it only widens the SD,
# which the report shows rather than hides.

# Stride averages are time-normalized to this many points. 101, not 100, the
# same convention as the project's stance-phase waveforms (CLAUDE.md).
N_POINTS = 101

# ------------------------------- M3's BAR, AND IT WAS NOT MET -------------
# `scripts/phase16a_strides.py`, results/phase16a_strides.json. The design's
# M3 test was "the detected stride count matches ground truth within +/-1, on
# 3 clips". **It came out 2 of 3.** On the third clip the right leg was short
# by 2 cycles, and the count error ranged -2 to 0.
#
# The bar stands as written and is recorded as FAILED. It is not lowered, not
# redefined, and not re-read as "2 of 3 is close". The design's stated
# consequence therefore applies: **the demo ships per-frame traces only**, and
# stride averaging is opt-in (`--stride-average`) rather than the default.
#
# What can honestly be said beyond that is very little, and it is said here
# rather than in a summary: the clip whose count was wrong ALSO failed this
# module's own check, so the fallback would have caught it -- but only **1** of
# the 3 clips passed that check, so the check's screening value rests on a
# single clip and is not evidence of anything.
STRIDE_COUNT_BAR = "+/-1 stride on 3 clips"
STRIDE_COUNT_CLIPS_PASSED = 2
STRIDE_COUNT_CLIPS = 3
STRIDE_COUNT_BAR_MET = False
STRIDE_SELFCHECK_PASSED_N = 1
# The second M3 bar, which WAS met: per-stride knee peaks vary less within a
# subject (median CV 0.282) than knee peaks vary across subjects (0.433), so a
# stride average is averaging gait rather than noise.
STRIDE_AVERAGE_IS_WORTH_SHOWING = True
# The consequence of the first bar failing, in code.
STRIDE_AVERAGING_IS_DEFAULT = False

SIDES = ("l", "r")

# The signal the cycle is read from. Knee flexion is the project's own primary
# quantity and the largest-amplitude periodic signal in the clip.
CYCLE_SIGNAL = "knee_from_segments"


# ------------------------------------------------------- the signal --------

def fill_short_gaps(x: np.ndarray, fps: float,
                    max_gap_s: float = 0.20) -> np.ndarray:
    """Interpolate across NaN runs up to `max_gap_s`; leave longer ones NaN.

    **This is not cosmetic.** Mocap drops markers and the detector drops
    frames, and a single NaN frame splits the signal into two islands, so the
    trigger re-arms in the middle of a cycle and counts it twice. A gap shorter
    than a fifth of a second cannot hide a gait event at 30 fps, so filling it
    adds no cycle and removes a false one.
    """
    y = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(y)
    if not bad.any() or bad.all():
        return y
    idx = np.arange(y.size)
    limit = max(1, int(round(max_gap_s * fps)))
    edges = np.flatnonzero(np.diff(bad.astype(np.int8)))
    starts = np.r_[0, edges + 1]
    ends = np.r_[edges, y.size - 1]
    filled = np.interp(idx, idx[~bad], y[~bad])
    out = y.copy()
    for a, b in zip(starts, ends):
        if bad[a] and (b - a + 1) <= limit and a > 0 and b < y.size - 1:
            out[a:b + 1] = filled[a:b + 1]
    return out


def swing_amplitude(series: np.ndarray) -> float:
    """The signal's 5-95 percentile range. Robust to a single bad frame."""
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 4:
        return float("nan")
    lo, hi = np.percentile(y, [5.0, 95.0])
    return float(hi - lo)


def cycle_events(series: np.ndarray, fps: float,
                 hyst_frac: float = HYST_FRAC) -> np.ndarray:
    """Frames of peak knee flexion, one per gait cycle. The Schmitt trigger.

    Walks the signal once with two thresholds placed symmetrically about its
    mid-range. Rising past the high edge opens an excursion; falling past the
    low edge closes it and emits the excursion's maximum. Needs no period
    estimate, so a changing cadence cannot break it.
    """
    x = fill_short_gaps(np.asarray(series, dtype=float), fps)
    amp = swing_amplitude(x)
    if not np.isfinite(amp) or amp < 1e-9:
        return np.empty(0, dtype=int)
    lo_p, hi_p = np.percentile(x[np.isfinite(x)], [5.0, 95.0])
    mid = 0.5 * (lo_p + hi_p)
    half = 0.5 * hyst_frac * amp
    high, low = mid + half, mid - half

    events: list[int] = []
    armed, start = True, 0
    for i, v in enumerate(x):
        if not np.isfinite(v):
            continue
        if armed and v > high:
            armed, start = False, i
        elif not armed and v < low:
            seg = x[start:i]
            if seg.size and np.isfinite(seg).any():
                events.append(start + int(np.nanargmax(seg)))
            armed = True
    return np.asarray(events, dtype=int)


def dominant_period(x: np.ndarray, fps: float,
                    lo_s: float = 0.30, hi_s: float = 2.00) -> float:
    """The dominant cycle length of `x` in frames, by autocorrelation.

    Reported as a cross-check on the trigger's own stride durations; the
    segmentation does not depend on it.
    """
    y = fill_short_gaps(np.asarray(x, dtype=float), fps)
    y = y[np.isfinite(y)]
    lo, hi = int(round(lo_s * fps)), int(round(hi_s * fps))
    if y.size < lo + 3 or lo < 2:
        return float("nan")
    hi = min(hi, y.size - 2)
    if hi <= lo:
        return float("nan")
    v = y - y.mean()
    denom = float(v @ v)
    if denom < 1e-18:
        return float("nan")
    ac = np.correlate(v, v, mode="full")[v.size - 1:] / denom
    return float(int(lo + np.argmax(ac[lo:hi + 1])))


# -------------------------------------------------- the segmentation -------

@dataclass
class Strides:
    """Gait cycles for both legs, with its own verdict and its own fallback."""

    fps: float
    events: dict = field(default_factory=dict)      # side -> (k,) int frames
    strides: dict = field(default_factory=dict)     # side -> [(lo, hi), ...]
    used: dict = field(default_factory=dict)        # side -> [(lo, hi), ...]
    durations_s: dict = field(default_factory=dict)
    amplitude_deg: dict = field(default_factory=dict)
    ok: bool = False
    reasons: list = field(default_factory=list)
    boundary: str = "peak swing-phase knee flexion"

    @property
    def count(self) -> dict:
        """Cycles detected per leg, before the duration filter."""
        return {s: max(len(self.events.get(s, [])) - 1, 0) for s in SIDES}

    @property
    def n_used(self) -> dict:
        return {s: len(self.used.get(s, [])) for s in SIDES}

    def duration_cv(self, side: str) -> float:
        """Stride-to-stride duration spread. REPORTED, never a gate.

        It is a property of the running, not of the measurement: a runner who
        decelerates inside the clip has a high CV and nothing is wrong.
        """
        d = np.asarray(self.durations_s.get(side, []), dtype=float)
        if d.size < 2 or not np.isfinite(d).all() or d.mean() <= 0:
            return float("nan")
        return float(d.std(ddof=1) / d.mean())

    def plausible_share(self, side: str) -> float:
        n = self.count[side]
        if n <= 0:
            return float("nan")
        return float(len(self.strides.get(side, [])) / n)

    @property
    def cadence_spm(self) -> float:
        """Steps per minute from the strides actually used. Descriptive."""
        d = [(hi - lo) / self.fps for s in SIDES for lo, hi in self.used.get(s, [])]
        if not d:
            return float("nan")
        return float(2.0 * 60.0 / np.mean(d))


def segment(angles: dict, fps: float,
            detected: np.ndarray | None = None) -> Strides:
    """Segment both legs into gait cycles, and judge whether to trust it.

    `angles` is `AnalysisResult.angles`: side -> quantity -> (T,) series. The
    returned object's `ok` is the design's fallback switch -- when it is False
    the report shows per-frame traces only and no stride table.
    """
    out = Strides(fps=float(fps))
    for side in SIDES:
        series = np.asarray(angles[side][CYCLE_SIGNAL], dtype=float)
        if detected is not None:
            ok = np.asarray(detected, dtype=bool)
            n = min(series.size, ok.size)
            series = series[:n].copy()
            series[~ok[:n]] = np.nan
        ev = cycle_events(series, fps)
        pairs = [(int(a), int(b)) for a, b in zip(ev[:-1], ev[1:])]
        dur = [(b - a) / float(fps) for a, b in pairs]
        keep = [i for i, d in enumerate(dur)
                if STRIDE_S_MIN <= d <= STRIDE_S_MAX]
        out.events[side] = ev
        out.strides[side] = [pairs[i] for i in keep]
        out.durations_s[side] = [dur[i] for i in keep]
        out.used[side] = list(out.strides[side])
        out.amplitude_deg[side] = swing_amplitude(series)

    why: list[str] = []
    leg = {"l": "left", "r": "right"}
    for side in SIDES:
        if len(out.used[side]) < MIN_STRIDES:
            why.append(f"{leg[side]} leg: only {len(out.used[side])} usable "
                       f"stride(s), fewer than {MIN_STRIDES}")
        amp = out.amplitude_deg[side]
        if not np.isfinite(amp) or amp < MIN_KNEE_SWING_DEG:
            why.append(f"{leg[side]} leg: the knee swings only {amp:.0f} deg, "
                       f"under {MIN_KNEE_SWING_DEG:.0f}; this does not look "
                       f"like running")
        share = out.plausible_share(side)
        if np.isfinite(share) and share < MIN_PLAUSIBLE_SHARE:
            why.append(f"{leg[side]} leg: only {share:.0%} of the detected "
                       f"cycles last a plausible "
                       f"{STRIDE_S_MIN}-{STRIDE_S_MAX} s")
    if abs(out.count["l"] - out.count["r"]) > MAX_COUNT_DIFF:
        why.append(f"the legs disagree on the cycle count "
                   f"({out.count['l']} left vs {out.count['r']} right); both "
                   f"legs take the same number of steps, so one is wrong")

    out.ok = not why
    out.reasons = why
    return out


# ---------------------------------------------------- stride averages ------

def normalize_stride(series: np.ndarray, lo: int, hi: int,
                     n: int = N_POINTS) -> np.ndarray:
    """One stride resampled to `n` points over 0-100% of the cycle."""
    y = np.asarray(series, dtype=float)[lo:hi + 1]
    if y.size < 2:
        return np.full(n, np.nan)
    t = np.linspace(0.0, 1.0, y.size)
    good = np.isfinite(y)
    if good.sum() < 2:
        return np.full(n, np.nan)
    return np.interp(np.linspace(0.0, 1.0, n), t[good], y[good])


def stride_average(series: np.ndarray, strides: list[tuple[int, int]],
                   n: int = N_POINTS) -> dict:
    """Mean and SD across strides, time-normalized to the cycle.

    The SD is **this clip's stride-to-stride spread**, a repeatability figure.
    It is NOT the measurement error, which comes from the error budget
    (`overstride.budget`) and is a different quantity measured against mocap.
    The report labels them separately and never adds them.
    """
    if not strides:
        return {"n": 0, "mean": np.full(n, np.nan), "sd": np.full(n, np.nan),
                "points": n}
    rows = np.vstack([normalize_stride(series, lo, hi, n)
                      for lo, hi in strides])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(rows, axis=0)
        sd = (np.nanstd(rows, axis=0, ddof=1) if rows.shape[0] > 1
              else np.zeros(n))
    return {"n": int(rows.shape[0]), "mean": mean, "sd": sd, "points": n}


def peaks_per_stride(series: np.ndarray,
                     strides: list[tuple[int, int]]) -> np.ndarray:
    """The maximum of `series` within each stride. One value per stride."""
    y = np.asarray(series, dtype=float)
    out = []
    for lo, hi in strides:
        seg = y[lo:hi + 1]
        seg = seg[np.isfinite(seg)]
        out.append(float(seg.max()) if seg.size else float("nan"))
    return np.asarray(out, dtype=float)


def cv(values) -> float:
    """Coefficient of variation, |sd / mean|. nan when undefined."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2 or abs(v.mean()) < 1e-12:
        return float("nan")
    return float(abs(v.std(ddof=1) / v.mean()))
