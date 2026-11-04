"""M2: the input checks of `docs/phase16-demo-design.md` section 4.

Each check returns **pass**, **warn** (processing continues and the warning is
carried into the report) or **refuse** (nothing is processed and no number is
emitted). Refusing is the existing behaviour of the tool -- `video_kinematics`
already refuses to write output when it cannot find the subject -- and this
extends it to conditions the error budget does not cover.

**Every threshold below is derived from a measured number and names its
source.** None is a taste judgement.

Checks 1-3 run on the clip's properties, before decoding. Checks 4-5 need the
detected 2D, so they run after detection and before anything is reported.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from phase14_confidence import R_POP, azimuth_proxy  # noqa: E402

PASS, WARN, REFUSE = "pass", "warn", "refuse"

# --- 1. frame rate ---------------------------------------------------------
# results/phase15e.md + results/phase15f.md: the penalty curve STEEPENS below
# 50 fps -- 67% of the knee's 30 fps penalty is incurred between 50 and 30.
# Below 25 fps the behaviour is off the measured curve entirely (the lowest
# rate ever measured is 30.0).
FPS_WARN, FPS_REFUSE = 50.0, 25.0

# --- 2. resolution --------------------------------------------------------
# The detector runs at the input's short side, and every number in Row B was
# measured at short side 600 (MoVi PG1, 800x600). Below 480 the resize has to
# UPSCALE to reach 600, which invents detail the budget never saw.
RES_WARN, RES_REFUSE = 480, 360

# --- 3. clip length -------------------------------------------------------
# The lifter's window is 81 frames. Phase 15F measured what heavy edge padding
# costs: a checkpoint fed 52% real input scored 20 deg+. At 30 fps a 3 s clip is
# 90 frames, barely one window.
LEN_WARN_S, LEN_REFUSE_S = 8.0, 3.0

# --- 4. person found ------------------------------------------------------
# The sealed test detected 100% of frames on 11 of 11 subjects with zero
# full-frame re-detections (results/phase15_sealed.md, results/phase15f.json),
# so a miss rate is an input problem, not a tool limitation.
MISS_REFUSE_FRAC = 0.20

# --- 5. viewpoint ---------------------------------------------------------
# The design doc's [A5] bars were derived in GROUND-TRUTH view_ratio units:
# MoVi PG1 jogging spans [0.536, 0.720] and Row B's sealed subjects run
# 0.537-0.698, so PASS had to sit below 0.536 or the budget's own footage would
# be flagged. Those GT-unit bars were 0.50 / 0.35.
GT_VIEW_WARN, GT_VIEW_REFUSE = 0.50, 0.35
MEASURED_GT_VIEW_RANGE = (0.536, 0.720)

# BUT THE DEMO DOES NOT MEASURE GROUND TRUTH. It computes
# `estimate_view_ratio` from the detected 2D, and that estimate is
# **systematically optimistic**: measured against calibrated ground truth on the
# 11 spent MoVi PG1 subjects (scripts/phase16a_viewratio.py,
# results/phase16a_viewratio.json),
#
#     est - gt:  mean +0.139,  sd 0.048,  range [+0.0245, +0.1868]
#     Pearson(gt, est) = 0.879
#     est range [0.633, 0.887]   against gt [0.537, 0.719]
#
# Applying a GT-unit bar to that estimate under-refuses: a clip whose TRUE
# view_ratio is 0.35 estimates near 0.49 and would only warn. So the bars are
# restated in ESTIMATE units:
#
#   REFUSE below 0.49  = GT 0.35 + the measured +0.139 bias
#   PASS at or above 0.60, which is below the LOWEST measured estimate (0.633)
#     with margin, so no clip the budget was measured on is ever flagged --
#     the binding constraint of [A5], now checked in the units actually used.
#
# The estimate's noise (sd 0.048) makes the effective GT cut fuzzy by roughly
# +/-0.05; that is stated in the report rather than hidden behind a sharp bar.
VIEW_WARN, VIEW_REFUSE = 0.60, 0.49
VIEW_BIAS_EST_MINUS_GT = 0.139
MEASURED_EST_VIEW_RANGE = (0.633, 0.887)
# kept for the tests and the report: the range the budget's footage occupies
MEASURED_VIEW_RANGE = MEASURED_EST_VIEW_RANGE


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    value: float | None = None
    thresholds: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status != REFUSE


class ClipRefused(Exception):
    """Raised instead of emitting numbers for a clip outside the budget."""

    def __init__(self, results):
        self.results = [r for r in results if r.status == REFUSE]
        super().__init__("; ".join(r.message for r in self.results))


def _res(name, status, message, value, **thr) -> CheckResult:
    return CheckResult(name=name, status=status, message=message, value=value,
                       thresholds=thr)


def check_fps(fps: float) -> CheckResult:
    if not np.isfinite(fps) or fps <= 0:
        return _res("fps", REFUSE, "could not read the clip's frame rate",
                    None, warn=FPS_WARN, refuse=FPS_REFUSE)
    if fps < FPS_REFUSE:
        return _res("fps", REFUSE,
                    f"{fps:.1f} fps is below {FPS_REFUSE:.0f}; the lowest rate "
                    f"this pipeline has ever been measured at is 30.0 fps, so "
                    f"no error figure applies", fps,
                    warn=FPS_WARN, refuse=FPS_REFUSE)
    if fps < FPS_WARN:
        return _res("fps", WARN,
                    f"{fps:.1f} fps is below {FPS_WARN:.0f}. The frame-rate "
                    f"penalty steepens here: 67% of the knee's 30 fps penalty "
                    f"is incurred between 50 and 30 fps (tune set, "
                    f"results/phase15f.md). Film at 50 fps or better", fps,
                    warn=FPS_WARN, refuse=FPS_REFUSE)
    return _res("fps", PASS, f"{fps:.1f} fps", fps,
                warn=FPS_WARN, refuse=FPS_REFUSE)


def check_resolution(width: int, height: int) -> CheckResult:
    short = int(min(width, height))
    if short < RES_REFUSE:
        return _res("resolution", REFUSE,
                    f"short side {short} px is below {RES_REFUSE}; the budget "
                    f"was measured at short side 600 and this cannot be "
                    f"upscaled to it honestly", short,
                    warn=RES_WARN, refuse=RES_REFUSE)
    if short < RES_WARN:
        return _res("resolution", WARN,
                    f"short side {short} px is below {RES_WARN}, so the resize "
                    f"to 600 px will UPSCALE, inventing detail the budget "
                    f"never saw", short, warn=RES_WARN, refuse=RES_REFUSE)
    return _res("resolution", PASS, f"{width}x{height}, short side {short}",
                short, warn=RES_WARN, refuse=RES_REFUSE)


def check_length(n_frames: int, fps: float) -> CheckResult:
    secs = (n_frames / fps) if (fps and np.isfinite(fps) and fps > 0) else 0.0
    if secs < LEN_REFUSE_S:
        return _res("length", REFUSE,
                    f"{secs:.1f} s is below {LEN_REFUSE_S:.0f} s; the lifter's "
                    f"window is 81 frames and a clip this short is mostly edge "
                    f"padding", secs,
                    warn=LEN_WARN_S, refuse=LEN_REFUSE_S)
    if secs < LEN_WARN_S:
        return _res("length", WARN,
                    f"{secs:.1f} s is below {LEN_WARN_S:.0f} s; expect fewer "
                    f"complete strides and more edge padding", secs,
                    warn=LEN_WARN_S, refuse=LEN_REFUSE_S)
    return _res("length", PASS, f"{secs:.1f} s, {n_frames} frames", secs,
                warn=LEN_WARN_S, refuse=LEN_REFUSE_S)


def check_person(detected, meta: dict | None = None) -> CheckResult:
    detected = np.asarray(detected, dtype=bool)
    n = int(detected.size)
    if n == 0:
        return _res("person", REFUSE, "no frames", None,
                    miss_refuse=MISS_REFUSE_FRAC)
    miss = 1.0 - float(detected.mean())
    crowded = int((meta or {}).get("frames_crowded", 0) or 0)
    if crowded:
        return _res("person", REFUSE,
                    f"more than one person in {crowded} frames; the pipeline "
                    f"tracks one runner and cannot tell whose limb is whose",
                    miss, miss_refuse=MISS_REFUSE_FRAC)
    if miss > MISS_REFUSE_FRAC:
        return _res("person", REFUSE,
                    f"no person found in {miss:.0%} of frames, over the "
                    f"{MISS_REFUSE_FRAC:.0%} limit; the sealed test detected "
                    f"100% of frames on 11 of 11 subjects, so this is an input "
                    f"problem", miss, miss_refuse=MISS_REFUSE_FRAC)
    if miss > 0:
        return _res("person", WARN,
                    f"no person found in {miss:.1%} of frames; those frames "
                    f"are excluded", miss, miss_refuse=MISS_REFUSE_FRAC)
    return _res("person", PASS, f"person found in all {n} frames", 0.0,
                miss_refuse=MISS_REFUSE_FRAC)


def estimate_view_ratio(kp2d) -> float:
    """`view_ratio` from the detected 2D alone, via phase 14's azimuth proxy.

    The proxy gives apparent hip separation over trunk length, which equals
    `R_POP * |cos(theta)|`; phase 14 records that `view_ratio ~ |sin(theta)|`,
    so `view_ratio ~ sqrt(1 - (ratio / R_POP)^2)`. No depth, no calibration.

    Phase 14 measured the proxy at **Pearson 0.974** against the ground-truth
    depth metric. Its SCALE against ground truth is checked separately -- see
    `scripts/phase16a_viewratio.py` -- because the thresholds are absolute.
    """
    kp2d = np.asarray(kp2d, dtype=float)
    ok = np.abs(kp2d[:, :, :2]).sum(axis=(1, 2)) > 0
    if ok.sum() < 5:
        return float("nan")
    ratio, _deg = azimuth_proxy(kp2d[ok])
    cos_t = np.clip(ratio / R_POP, 0.0, 1.0)
    vr = np.sqrt(np.clip(1.0 - cos_t ** 2, 0.0, 1.0))
    return float(np.nanmedian(vr))


def check_viewpoint(kp2d) -> CheckResult:
    vr = estimate_view_ratio(kp2d)
    thr = {"warn": VIEW_WARN, "refuse": VIEW_REFUSE,
           "units": "estimated view_ratio, not ground truth",
           "bias_est_minus_gt": VIEW_BIAS_EST_MINUS_GT,
           "measured_est_range": list(MEASURED_EST_VIEW_RANGE),
           "gt_equivalents": {"warn": GT_VIEW_WARN,
                              "refuse": GT_VIEW_REFUSE}}
    if not np.isfinite(vr):
        return _res("viewpoint", WARN,
                    "could not estimate the viewpoint from the detections",
                    None, **thr)
    if vr < VIEW_REFUSE:
        return _res("viewpoint", REFUSE,
                    f"estimated view_ratio {vr:.3f} is below {VIEW_REFUSE}: "
                    f"the runner is nearer frontal than side-on. Phase 11 "
                    f"measured roughly double the error at the frontal end "
                    f"(3.50 deg against 1.77 side-on) and phase 15 found that "
                    f"curve did not replicate on its one frontal clip, so "
                    f"there is no model of the error here. (This bar is "
                    f"{VIEW_REFUSE} in estimated units, which is the GT "
                    f"{GT_VIEW_REFUSE} bar plus the measured "
                    f"{VIEW_BIAS_EST_MINUS_GT:+.3f} estimator bias)", vr, **thr)
    if vr < VIEW_WARN:
        return _res("viewpoint", WARN,
                    f"estimated view_ratio {vr:.3f} is below {VIEW_WARN}, "
                    f"outside the range "
                    f"[{MEASURED_VIEW_RANGE[0]}, {MEASURED_VIEW_RANGE[1]}] the "
                    f"error budget's own footage occupies. Error rises toward a "
                    f"frontal view and the amount is unmeasured for this "
                    f"pipeline. Note this estimate runs about "
                    f"{VIEW_BIAS_EST_MINUS_GT:+.2f} optimistic against "
                    f"calibrated ground truth, with a spread of +/-0.05", vr,
                    **thr)
    return _res("viewpoint", PASS,
                f"estimated view_ratio {vr:.3f}, side-on", vr, **thr)


def pre_decode_checks(info) -> list[CheckResult]:
    """Checks 1-3, from the clip's properties. Run before decoding."""
    return [check_fps(info.fps),
            check_resolution(info.width, info.height),
            check_length(info.n_frames, info.fps)]


def post_detect_checks(kp2d, detected, meta=None) -> list[CheckResult]:
    """Checks 4-5, from the detected 2D. Run before anything is reported."""
    return [check_person(detected, meta), check_viewpoint(kp2d)]


def verdict(results) -> str:
    """The worst status across a list of checks."""
    statuses = {r.status for r in results}
    if REFUSE in statuses:
        return REFUSE
    if WARN in statuses:
        return WARN
    return PASS


def raise_if_refused(results) -> None:
    if verdict(results) == REFUSE:
        raise ClipRefused(results)
