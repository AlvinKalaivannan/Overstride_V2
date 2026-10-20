"""Phase 14 -- per-segment measurement confidence, and whether it is honest.

Pre-registration: `results/phase14.md`, frozen 2026-10-19. Read it first; the
bars, the estimators and the two corrections made before any result are there.

What this does, in one line: for every frame of every held-out clip it asks how
wrong the thigh and shank directions are, then asks whether a model that has
never seen the ground truth can say so in advance.

THE RULE THAT SHAPES THE CODE
-----------------------------
Every confidence input must be computable from the video alone -- no ground
truth, no `norm_scale`, no `p2mm`, nothing a phone clip cannot produce. The
extraction below is therefore split in two, deliberately, so the separation is
visible rather than promised: `features_from_video()` is handed the detections
and the lifted estimate and nothing else, and `realized_errors()` is the only
function that touches ground truth. A feature that needs GT cannot be written
without moving it across that line.

This produces no injury output. Confidence describes the MEASUREMENT.

Run: .venv/Scripts/python.exe scripts/phase14_confidence.py --stage all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AP = REPO / "data" / "athleticspose"
sys.path.insert(0, str(AP / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from phase3_angles import H36M, sagittal_angles  # noqa: E402
from phase3_infer import DATA, load_model  # noqa: E402
from phase7_inference_fix import CKPT_FOR, clip_list, lift  # noqa: E402
from phase12_scan import SEED, subject_bootstrap_ci  # noqa: E402

FRAMES = REPO / "data" / "derived" / "phase14_frames.npz"
OUT = REPO / "results" / "phase14_confidence.json"
FIG = REPO / "figures" / "phase14_confidence.png"

# The DEPLOYABLE configuration, 3.51 deg: generic COCO detections, edge padding,
# overlapped windows. Fine-tuned detections are an AthleticsPose artifact and a
# confidence fitted on them would be fitted on something no clip can reproduce.
DET = "coco"
PAD, WINDOWS = "edge", "overlap"

# The four gated segments. Pelvis and trunk define the body frame, so their
# sagittal error is definitionally zero; foot has no toe keypoint. Both are
# reported as not measurable rather than estimated. See the pre-registration.
SEGMENTS = {"thigh_l": ("l_hip", "l_knee"), "thigh_r": ("r_hip", "r_knee"),
            "shank_l": ("l_knee", "l_ankle"), "shank_r": ("r_knee", "r_ankle")}

# COCO-17, as torchvision emits and the lifter consumes.
COCO = {"l_shoulder": 5, "r_shoulder": 6, "l_hip": 11, "r_hip": 12,
        "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16}

FEATURES = ["score_prox", "score_dist", "score_min", "len_dev", "lr_asym",
            "ml_compression", "view_proxy_deg", "blend_weight", "speed",
            "is_far", "is_shank"]
# AthleticsPose holds out THREE subjects (S11, S13, S16 -- phase3_infer.py:53),
# and the other 21 trained the lifter, so its error on them is training error and
# cannot be used here. Three subjects means leave-one-subject-out, and it means a
# 3-point bootstrap CI is nominal. The gate is therefore stated per subject
# rather than as a mean with an interval around three numbers: see the third
# correction in the pre-registration.
N_FOLDS = 3
EPS = 1e-6


# ------------------------------------------------------------------ geometry --
def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def body_frame(kp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The fwd/up basis of `sagittal_angles`, lifted out verbatim.

    Built from pelvis->thorax and hip<->hip, so a rotated camera does not change
    it -- and so the pelvis and trunk cannot have a sagittal error in it.
    """
    p, thorax = kp[:, H36M["pelvis"]], kp[:, H36M["thorax"]]
    up = _unit(thorax - p)
    ml = _unit(kp[:, H36M["r_hip"]] - kp[:, H36M["l_hip"]])
    fwd = _unit(np.cross(ml, up))
    up = _unit(np.cross(fwd, ml))
    return fwd, up


def seg_angle(kp: np.ndarray, a: str, b: str) -> np.ndarray:
    """Sagittal direction of segment a->b, in degrees, in the body's own frame."""
    fwd, up = body_frame(kp)
    d = _unit(kp[:, H36M[b]] - kp[:, H36M[a]])
    return np.degrees(np.arctan2(np.sum(d * fwd, axis=-1),
                                 np.sum(d * up, axis=-1)))


def wrap(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def realized_errors(est: np.ndarray, gt: np.ndarray) -> dict[str, np.ndarray]:
    """THE ONLY FUNCTION HERE THAT TOUCHES GROUND TRUTH.

    Per segment, per frame: the absolute difference between the estimated and
    true sagittal direction of that segment, each measured in its own skeleton's
    body frame -- the same construction the published angle MAE uses.
    """
    return {name: np.abs(wrap(seg_angle(est, a, b) - seg_angle(gt, a, b)))
            for name, (a, b) in SEGMENTS.items()}


# ------------------------------------------------------------------ features --
def features_from_video(det2d: np.ndarray, est: np.ndarray,
                        blend: np.ndarray) -> tuple[dict, str, np.ndarray]:
    """Per-segment, per-frame inputs, from the detections and the estimate only.

    det2d: (n, 17, 3) COCO-17 x, y, score -- what the detector produced.
    est:   (n, 17, 3) the lifter's own output, scale-free.
    blend: (n,) the per-frame window blend weight `lift` built the frame from.

    Ground truth is not a parameter of this function, by design.
    """
    n = len(est)
    xy, score = det2d[:, :, :2], det2d[:, :, 2]

    # Camera azimuth, video-only: the medio-lateral axis as the camera sees it
    # against the same axis as the body model has it. Azimuth compresses ml and
    # leaves the sagittal plane roughly intact, so dividing by a sagittal length
    # cancels the image scale and the subject's size alike.
    mid_hip_2d = 0.5 * (xy[:, COCO["l_hip"]] + xy[:, COCO["r_hip"]])
    mid_sho_2d = 0.5 * (xy[:, COCO["l_shoulder"]] + xy[:, COCO["r_shoulder"]])
    ml_2d = np.linalg.norm(xy[:, COCO["l_hip"]] - xy[:, COCO["r_hip"]], axis=-1)
    trunk_2d = np.linalg.norm(mid_sho_2d - mid_hip_2d, axis=-1)
    apparent_ml = ml_2d / np.maximum(trunk_2d, 1e-9)

    mid_sho_3d = 0.5 * (est[:, H36M["l_shoulder"]] + est[:, H36M["r_shoulder"]])
    ml_3d = np.linalg.norm(est[:, H36M["l_hip"]] - est[:, H36M["r_hip"]], axis=-1)
    trunk_3d = np.linalg.norm(mid_sho_3d - est[:, H36M["pelvis"]], axis=-1)
    # the scan: one ratio for the clip, the median over its frames
    scanned_ml = float(np.median(ml_3d / np.maximum(trunk_3d, 1e-9)))
    ml_compression = apparent_ml / max(scanned_ml, 1e-9)
    view_proxy_deg = np.degrees(np.arccos(np.clip(ml_compression, 0.0, 1.0)))

    # motion: how far the lower-body keypoints moved since the previous frame,
    # in units of trunk length so it does not depend on the image scale
    lower = [COCO[k] for k in ("l_hip", "r_hip", "l_knee", "r_knee",
                               "l_ankle", "r_ankle")]
    disp = np.zeros(n)
    if n > 1:
        d = np.linalg.norm(np.diff(xy[:, lower], axis=0), axis=-1).mean(axis=1)
        disp[1:] = d / np.maximum(trunk_2d[1:], 1e-9)
        disp[0] = disp[1]

    # which limb is nearer the camera, from the ESTIMATE's own depth, never the
    # ground truth's -- a phone clip has only the estimate
    near_est = ("l" if est[:, H36M["l_hip"], 2].mean()
                < est[:, H36M["r_hip"], 2].mean() else "r")

    lengths = {name: np.linalg.norm(est[:, H36M[b]] - est[:, H36M[a]], axis=-1)
               for name, (a, b) in SEGMENTS.items()}

    out = {}
    for name, (a, b) in SEGMENTS.items():
        side = name[-1]
        partner = name[:-1] + ("r" if side == "l" else "l")
        med = float(np.median(lengths[name]))
        pair_mean = 0.5 * (lengths[name] + lengths[partner])
        out[name] = {
            "score_prox": score[:, COCO[a]],
            "score_dist": score[:, COCO[b]],
            "score_min": np.minimum(score[:, COCO[a]], score[:, COCO[b]]),
            # phase 12: a subject's length is stable across trials to 1.7% but
            # wanders 14.5% frame to frame, so a frame whose limb is the wrong
            # length is a frame the lifter lost
            "len_dev": np.abs(lengths[name] - med) / max(med, 1e-9),
            "lr_asym": (np.abs(lengths[name] - lengths[partner])
                        / np.maximum(pair_mean, 1e-9)),
            "ml_compression": ml_compression,
            "view_proxy_deg": view_proxy_deg,
            "blend_weight": blend,
            "speed": disp,
            "is_far": np.full(n, 0.0 if side == near_est else 1.0),
            "is_shank": np.full(n, 1.0 if name.startswith("shank") else 0.0),
        }
    return out, near_est, view_proxy_deg


# ---------------------------------------------------------------- extraction --
def extract(limit: int | None) -> dict:
    """One pass over the held-out clips, saving per-frame rows."""
    from phase11_viewpoint import view_ratio  # GT-based VALIDATOR, not an input

    model = load_model(CKPT_FOR[DET], "base")
    if model is None:
        raise SystemExit(f"checkpoint missing: {CKPT_FOR[DET]}")

    clips = clip_list(limit)
    print(f"=== extracting {len(clips)} clips, {DET} + {PAD} + {WINDOWS} ===")

    cols: dict[str, list] = defaultdict(list)
    clip_rows = []
    near_agree = 0
    for i, (action, subj, stem, _n) in enumerate(clips):
        gt = np.asarray(np.load(
            DATA / "gt_markers3d_by_cam" / action / subj / f"{stem}.npz")["markers_h36m"])
        det = np.load(DATA / f"det_markers2d_by_cam_{DET}" / action / subj
                      / f"{stem}.npy").astype(float)
        n = min(len(gt), len(det))
        if n < 20:
            continue
        gt, det = gt[:n], det[:n]
        est, blend = lift(model, det, PAD, WINDOWS, with_weights=True)

        feats, near_est, view_proxy = features_from_video(det, est, blend)
        errs = realized_errors(est, gt)

        near_gt = ("l" if gt[:, H36M["l_hip"], 2].mean()
                   < gt[:, H36M["r_hip"], 2].mean() else "r")
        near_agree += int(near_gt == near_est)

        # the GT validator for the azimuth proxy, per clip, reported not gated
        vr = view_ratio(gt)
        ga = sagittal_angles(gt, near_gt)
        ea = sagittal_angles(est, near_gt)
        clip_rows.append({
            "action": action, "subject": subj, "clip": stem, "n_frames": int(n),
            "near_gt": near_gt, "near_est": near_est,
            "view_ratio_gt": float(vr),
            "out_of_plane_deg_gt": float(np.degrees(np.arcsin(min(vr, 1.0)))),
            "view_proxy_deg_est": float(np.median(view_proxy)),
            "near_hip_mae": float(np.mean(np.abs(ea["hip"] - ga["hip"]))),
            "near_knee_mae": float(np.mean(np.abs(ea["knee"] - ga["knee"]))),
        })

        for name in SEGMENTS:
            cols["segment"].append(np.full(n, name))
            cols["subject"].append(np.full(n, subj))
            cols["clip"].append(np.full(n, f"{action}/{subj}/{stem}"))
            cols["error"].append(errs[name])
            for f in FEATURES:
                cols[f].append(feats[name][f])
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(clips)} clips")

    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    FRAMES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FRAMES, **arrays,
                        clip_rows=json.dumps(clip_rows),
                        near_agreement=np.array([near_agree, len(clip_rows)]))
    print(f"\nwrote {FRAMES.relative_to(REPO)}: "
          f"{len(arrays['error'])} segment-frames over {len(clip_rows)} clips")
    print(f"  near-limb call from the estimate agrees with ground truth on "
          f"{near_agree}/{len(clip_rows)} clips")
    return {"n_clips": len(clip_rows), "n_rows": int(len(arrays["error"]))}


# ----------------------------------------------------------------- the model --
def fit_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions of log error. Everything fitted inside the fold."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        pipe = make_pipeline(StandardScaler(),
                             RidgeCV(alphas=np.logspace(-3, 3, 13)))
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict(X[te])
    return oof


def spearman_by_subject(stated: np.ndarray, realized: np.ndarray,
                        subject: np.ndarray) -> dict:
    """Within subject, then averaged. Pooling would let between-subject
    differences in overall error pose as within-clip ranking skill."""
    from scipy.stats import spearmanr

    per = {}
    for s in sorted(set(subject)):
        m = subject == s
        if m.sum() < 50:
            continue
        rho = spearmanr(stated[m], realized[m]).statistic
        if np.isfinite(rho):
            per[str(s)] = float(rho)
    vals = list(per.values())
    lo, hi = subject_bootstrap_ci(vals)
    return {"n_subjects": len(vals),
            "mean_rho": float(np.mean(vals)) if vals else float("nan"),
            "min_rho": float(np.min(vals)) if vals else float("nan"),
            # reported for completeness; with three subjects it is nominal and
            # the gate does not rest on it
            "ci95_nominal": [lo, hi],
            "per_subject": per}


def _ols(stated: np.ndarray, realized: np.ndarray) -> tuple[float, float, float]:
    A = np.c_[stated, np.ones_like(stated)]
    (b, a), *_ = np.linalg.lstsq(A, realized, rcond=None)
    pred = A @ np.array([b, a])
    ss = float(np.sum((realized - realized.mean()) ** 2))
    r2 = float(1.0 - np.sum((realized - pred) ** 2) / ss) if ss else float("nan")
    return float(b), float(a), r2


def calibration(stated: np.ndarray, realized: np.ndarray,
                subject: np.ndarray | None = None) -> dict:
    """OLS of realized on stated: slope in [0.8, 1.2] is the pre-registered bar."""
    b, a, r2 = _ols(stated, realized)
    out = {"slope": b, "intercept": a, "r2": r2,
           "mean_abs_gap_deg": float(np.mean(np.abs(realized - stated)))}
    if subject is not None:
        per = {}
        for sb in sorted(set(subject.tolist())):
            m = subject == sb
            if m.sum() >= 50:
                per[str(sb)] = _ols(stated[m], realized[m])[0]
        out["slope_per_subject"] = per
        if per:
            out["slope_worst"] = float(max(per.values(), key=lambda v: abs(v - 1.0)))
    return out


def fit_and_score() -> dict:
    if not FRAMES.exists():
        raise SystemExit(f"missing {FRAMES.relative_to(REPO)} -- run --stage extract")
    z = np.load(FRAMES, allow_pickle=False)
    seg, subj = z["segment"], z["subject"]
    clip = z["clip"]
    y_err = z["error"].astype(float)
    X = np.column_stack([z[f].astype(float) for f in FEATURES])
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y_err)
    seg, subj, clip, y_err, X = seg[finite], subj[finite], clip[finite], \
        y_err[finite], X[finite]
    y = np.log(y_err + EPS)

    rng = np.random.default_rng(SEED)
    out = {"pre_registration": "results/phase14.md, frozen 2026-10-19",
           "configuration": {"detector": DET, "padding": PAD, "windows": WINDOWS,
                             "note": "the deployable 3.51 deg configuration"},
           "n_rows": int(len(y)), "n_subjects": len(set(subj.tolist())),
           "features": FEATURES, "n_folds": N_FOLDS,
           "not_measurable": {
               "pelvis": "defines the frame's medio-lateral axis -- sagittal "
                         "error is definitionally zero",
               "trunk": "defines the frame's vertical axis -- same",
               "foot": "H36M-17 has no toe keypoint"},
           "models": {}}

    models = {"full": FEATURES,
              "detector_score_only": ["score_min"]}
    stated_full = None
    for tag, feats in models.items():
        idx = [FEATURES.index(f) for f in feats]
        oof = fit_oof(X[:, idx], y, subj)
        stated = np.exp(oof) - EPS
        if tag == "full":
            stated_full = stated
        entry = {"inputs": feats, "segments": {}}
        for name in sorted(set(seg.tolist())):
            m = seg == name
            entry["segments"][name] = {
                "n_frames": int(m.sum()),
                "realized_mean_deg": float(np.mean(y_err[m])),
                "stated_mean_deg": float(np.mean(stated[m])),
                "spearman": spearman_by_subject(stated[m], y_err[m], subj[m]),
                "calibration": calibration(stated[m], y_err[m], subj[m]),
            }
        out["models"][tag] = entry

    # control: shuffle the stated error WITHIN each clip. If the ranking is
    # really per-frame this collapses to rho ~ 0; if it was ranking clips all
    # along, it survives.
    perm = stated_full.copy()
    for c in set(clip.tolist()):
        m = clip == c
        perm[m] = rng.permutation(perm[m])
    out["control_within_clip_permutation"] = {
        name: spearman_by_subject(perm[seg == name], y_err[seg == name],
                                  subj[seg == name])
        for name in sorted(set(seg.tolist()))}

    # the azimuth proxy against the ground-truth viewpoint metric: reported,
    # never gated -- it says what not having depth costs
    from scipy.stats import spearmanr
    rows = json.loads(str(z["clip_rows"]))
    pr = np.array([r["view_proxy_deg_est"] for r in rows])
    gtv = np.array([r["out_of_plane_deg_gt"] for r in rows])
    ok = np.isfinite(pr) & np.isfinite(gtv)
    agree, total = (int(x) for x in z["near_agreement"])
    out["azimuth_proxy_vs_ground_truth"] = {
        "n_clips": int(ok.sum()),
        "spearman": float(spearmanr(pr[ok], gtv[ok]).statistic),
        "pearson": float(np.corrcoef(pr[ok], gtv[ok])[0, 1]),
        "note": "video-only proxy vs the GT depth metric; ungated, it measures "
                "the cost of having no depth"}
    out["near_limb_call"] = {
        "agreement": f"{agree}/{total}",
        "rate": agree / max(total, 1),
        "note": "the near limb from the estimate's own depth vs from ground "
                "truth; the feature uses the estimate, reporting uses GT"}

    verdict = {}
    for name, e in out["models"]["full"]["segments"].items():
        sp, cal = e["spearman"], e["calibration"]
        per_subject_rho = list(sp["per_subject"].values())
        per_subject_slope = list(cal.get("slope_per_subject", {}).values())
        verdict[name] = {
            # bar 1: EVERY subject clears 0.30, which at n=3 is stronger and
            # more honest than a mean with a 3-point interval around it
            "bar1_rho_met": bool(per_subject_rho
                                 and min(per_subject_rho) >= 0.30),
            "bar2_slope_met": bool(0.80 <= cal["slope"] <= 1.20),
            "bar2_slope_met_every_subject": bool(
                per_subject_slope
                and all(0.80 <= v <= 1.20 for v in per_subject_slope)),
            "mean_rho": sp["mean_rho"], "min_rho": sp["min_rho"],
            "slope": cal["slope"]}
    out["verdict"] = verdict
    out["kill_criterion_fired"] = not any(v["bar1_rho_met"] for v in verdict.values())

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_report(out)
    return out


def print_report(out: dict) -> None:
    print("\n=== phase 14: per-segment measurement confidence ===")
    print(f"  {out['n_rows']} segment-frames, {out['n_subjects']} subjects, "
          f"{out['n_folds']}-fold grouped by subject")
    for tag, e in out["models"].items():
        print(f"\n  model: {tag}  ({len(e['inputs'])} inputs)")
        print(f"    {'segment':<10}{'realized':>10}{'stated':>9}{'rho':>7}"
              f"{'min rho':>9}{'slope':>8}{'gate':>14}")
        for name, s in e["segments"].items():
            sp, cal = s["spearman"], s["calibration"]
            gate = ("met" if sp["min_rho"] >= 0.30 else "NOT met") + (
                "/cal met" if 0.8 <= cal["slope"] <= 1.2 else "/cal NOT")
            print(f"    {name:<10}{s['realized_mean_deg']:>9.2f}°"
                  f"{s['stated_mean_deg']:>8.2f}°{sp['mean_rho']:>7.2f}"
                  f"{sp['min_rho']:>9.2f}{cal['slope']:>8.2f}{gate:>14}")
    print("\n  control -- stated error shuffled within each clip:")
    for name, sp in out["control_within_clip_permutation"].items():
        print(f"    {name:<10}rho {sp['mean_rho']:+.3f}")
    a = out["azimuth_proxy_vs_ground_truth"]
    print(f"\n  azimuth proxy vs GT depth metric: Spearman {a['spearman']:.3f}, "
          f"Pearson {a['pearson']:.3f} over {a['n_clips']} clips (ungated)")
    print(f"  near-limb call agrees with GT on {out['near_limb_call']['agreement']}")
    print(f"\n  kill criterion fired: {out['kill_criterion_fired']}")
    print(f"\nwrote {OUT.relative_to(REPO)}")


def figure(out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segs = list(out["models"]["full"]["segments"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(segs))
    rho = [out["models"]["full"]["segments"][s]["spearman"]["mean_rho"] for s in segs]
    # the spread over the three held-out subjects, not a bootstrap interval:
    # with n=3 the bootstrap is nominal and the gate is per subject
    lows = [min(out["models"]["full"]["segments"][s]["spearman"]["per_subject"]
                .values()) for s in segs]
    highs = [max(out["models"]["full"]["segments"][s]["spearman"]["per_subject"]
                 .values()) for s in segs]
    err = [[rho[i] - lows[i] for i in range(len(segs))],
           [highs[i] - rho[i] for i in range(len(segs))]]
    ctl = [out["control_within_clip_permutation"][s]["mean_rho"] for s in segs]
    ax[0].bar(x - 0.18, rho, 0.36, yerr=err, capsize=3,
              label="confidence (bar = per-subject range, n=3)")
    ax[0].bar(x + 0.18, ctl, 0.36, label="shuffled within clip")
    ax[0].axhline(0.30, ls="--", c="k", lw=1, label="pre-registered bar 0.30")
    ax[0].set_xticks(x, segs)
    ax[0].set_ylabel("Spearman rho, within subject")
    ax[0].set_title("Does confidence rank the error?")
    ax[0].legend(fontsize=8)

    slope = [out["models"]["full"]["segments"][s]["calibration"]["slope"] for s in segs]
    ax[1].bar(x, slope, 0.5)
    ax[1].axhspan(0.8, 1.2, color="k", alpha=0.10,
                  label="pre-registered band [0.8, 1.2]")
    ax[1].axhline(1.0, ls="--", c="k", lw=1)
    ax[1].set_xticks(x, segs)
    ax[1].set_ylabel("calibration slope")
    ax[1].set_title("Is the stated error the right size?")
    ax[1].legend(fontsize=8)

    fig.suptitle("Phase 14 -- confidence in the MEASUREMENT, per segment "
                 "(never an injury quantity)", fontsize=10)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)
    print(f"wrote {FIG.relative_to(REPO)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("extract", "fit", "all"), default="all")
    ap.add_argument("--clips", type=int, default=0, help="0 = every held-out clip")
    args = ap.parse_args()
    if args.stage in ("extract", "all"):
        extract(args.clips or None)
    if args.stage in ("fit", "all"):
        out = fit_and_score()
        figure(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
