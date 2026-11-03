"""Phase 15 DIAGNOSTIC rows -- why the hip is biased, and what the detector costs.

**These are diagnostic rows. They do NOT change the phase 15 headline**, which
stays 9.610 deg near / 10.604 deg far on PG1 at n = 2 (`results/phase15.md`).
This script exists because that result rests on two subjects which disagree by
2.3x on the detector's share of the error and 2x on the perfect-2D hip bias --
too thin to decide anything on.

Subject roles were fixed in commit f549bfd **before** any of this data was
downloaded. `Subject_12`, `Subject_13` and `Subject_15` are the DIAGNOSTIC set,
spent on arrival. `Subject_16`, `_17`, `_18`, `_19` and `_2` are the SEALED TEST
set and are **not touched here**.

Every subject still passes through the pre-registered check gate, so a failing
subject is excluded before its video is decoded -- that is the frozen rule, and
it is why Subject_11 and Subject_15 carry no error figure.

Reported per subject, per limb, per joint, for BOTH arms: MAE, signed bias, SD
of the signed error, Pearson r, and n. Then the detector's cost as
(detected - perfect). Then pooled over the diagnostic-eligible subjects with a
subject bootstrap.

Also reported: each subject's **trunk lean** -- the mean sagittal angle of the
pelvis->THORAX axis against vertical during jogging -- beside that subject's
perfect-2D hip bias, with the correlation across subjects. **No correction is
applied.** The hip angle is measured against that axis, so if MoVi's THORAX
marker sits differently from H36M's thorax the bias should track lean; this
tests that, it does not act on it.

Run: .venv/Scripts/python.exe scripts/phase15_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data" / "athleticspose" / "repo"))
sys.path.insert(0, str(REPO / "scripts"))

from phase15_checks import load_v3d, to_h36m  # noqa: E402
from phase15_movi import (CAMERAS, JOINTS, LIMBS, MOCAP, MOCAP_PER_VIDEO,  # noqa: E402
                           MOVI, jogging_window, preregistered_checks, run_clip)
from phase3_angles import H36M  # noqa: E402
from phase3_infer import load_model  # noqa: E402
from phase7_inference_fix import CKPT_FOR  # noqa: E402
from phase12_scan import subject_bootstrap_ci  # noqa: E402

OUT = REPO / "results" / "phase15_diagnostic.json"
CAM = "PG1"

# Fixed in commit f549bfd, before download. Not editable to suit a result.
SPENT = ("Subject_1", "Subject_10")
DIAGNOSTIC = ("Subject_12", "Subject_13", "Subject_15")
SEALED = ("Subject_16", "Subject_17", "Subject_18", "Subject_19", "Subject_2")


def trunk_lean(mat: Path) -> dict:
    """Mean sagittal angle of pelvis->THORAX against vertical, during jogging.

    Vertical is axis 2 in MoVi's world frame (check 2, agreed two ways). The
    sagittal plane is spanned by vertical and the direction of travel, so the
    lean reported is the component of the trunk axis along travel -- a forward
    lean is positive.

    This is a property of how the subject runs, computed from the reference
    mocap alone. It is reported BESIDE the hip bias, never subtracted from it.
    """
    v = load_v3d(mat)
    _ji, _f30, f120 = jogging_window(v)
    gt = to_h36m(v)[int(f120[0]):int(f120[1]) + 1]

    pelvis, thorax = gt[:, H36M["pelvis"]], gt[:, H36M["thorax"]]
    ok = np.isfinite(pelvis).all(1) & np.isfinite(thorax).all(1)
    if ok.sum() < 20:
        return {"deg": float("nan"), "n": int(ok.sum())}
    trunk = thorax[ok] - pelvis[ok]

    # direction of travel, horizontal, per frame over ~83 ms
    hor = [0, 1]
    p = pelvis[ok][:, hor]
    w = 10
    vel = p[w:] - p[:-w]
    good = np.isfinite(vel).all(1) & (np.linalg.norm(vel, axis=1) > 1e-3)
    if good.sum() < 5:
        return {"deg": float("nan"), "n": int(good.sum())}
    fwd2 = vel[good].mean(0)
    # travel sign is irrelevant to "how far forward is the trunk", but the
    # per-frame direction is needed, so use each frame's own velocity sign
    ang = []
    tr = trunk[w // 2: w // 2 + len(vel)][good]
    for t, vv in zip(tr, vel[good]):
        f = np.array([vv[0], vv[1], 0.0])
        f /= np.linalg.norm(f)
        up = np.array([0.0, 0.0, 1.0])
        # components of the trunk axis in the sagittal plane (fwd, up)
        a = float(np.dot(t, f))
        b = float(np.dot(t, up))
        ang.append(np.degrees(np.arctan2(a, b)))
    ang = np.asarray(ang)
    return {"deg": float(np.mean(ang)), "sd_deg": float(np.std(ang, ddof=1)),
            "n": int(len(ang)),
            "note": "pelvis->THORAX vs vertical, in the sagittal plane; "
                    "positive = leaning forward along travel"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    mats = {p.stem.split("_v3d_")[-1]: p
            for p in sorted(MOCAP.glob("F_v3d_*.mat"))}
    pool = [s for s in SPENT + DIAGNOSTIC if s in mats]
    for s in SEALED:
        assert s not in pool, "a sealed-test subject must never be scored here"

    lift_model = load_model(CKPT_FOR["coco"], "base")
    if lift_model is None:
        raise SystemExit("checkpoint missing")

    print(f"=== phase 15 diagnostic rows, camera {CAM} ===")
    print(f"  pool: {', '.join(pool)}")
    print(f"  sealed and untouched: {', '.join(SEALED)}\n")

    rows: dict = {}
    for subject in pool:
        mat = mats[subject]
        ok, chk = preregistered_checks(mat)
        entry = {"subject": subject,
                 "designation": ("spent (phase 15 headline)" if subject in SPENT
                                 else "diagnostic (spent on arrival)"),
                 "passes_prereg_checks": ok,
                 "failed_checks": chk["failed_checks"],
                 "trunk_lean": trunk_lean(mat)}
        if not ok:
            print(f"  {subject}: EXCLUDED before decoding -- "
                  f"{'; '.join(chk['failed_checks'].values())}")
            entry["arms"] = None
            rows[subject] = entry
            continue

        video = MOVI / CAMERAS[CAM]["pattern"].format(s=subject.split("_")[-1])
        if not video.exists():
            print(f"  {subject}: no {CAM} video on disk, skipping")
            entry["arms"] = None
            rows[subject] = entry
            continue

        entry["arms"] = {}
        for arm, perfect in (("detected", False), ("perfect", True)):
            print(f"  {subject} / {arm}:")
            r = run_clip(subject, CAM, mat, video, None, lift_model, 0,
                         perfect=perfect)
            entry["arms"][arm] = (r or {}).get("err")
        rows[subject] = entry

    # ---- per-subject table ------------------------------------------------
    print(f"\n=== PER SUBJECT ({CAM}) ===")
    hdr = (f"  {'subject':<12}{'arm':<9}{'limb':<6}{'joint':<6}"
           f"{'MAE':>8}{'bias':>8}{'SD':>8}{'r':>7}{'n':>6}")
    print(hdr)
    for s, e in rows.items():
        if not e.get("arms"):
            continue
        for arm in ("detected", "perfect"):
            a = e["arms"].get(arm)
            if not a:
                continue
            for limb in LIMBS:
                for j in JOINTS:
                    print(f"  {s:<12}{arm:<9}{limb:<6}{j:<6}"
                          f"{a[limb][j]:>8.3f}{a[limb][j + '_bias']:>8.3f}"
                          f"{a[limb][j + '_sd']:>8.3f}"
                          f"{a[limb][j + '_r']:>7.3f}{a[limb][j + '_n']:>6d}")

    # ---- detector cost ----------------------------------------------------
    print(f"\n=== DETECTOR COST = detected - perfect, per subject ===")
    cost = {}
    for s, e in rows.items():
        if not e.get("arms") or not e["arms"].get("perfect"):
            continue
        cost[s] = {}
        for limb in LIMBS:
            d = e["arms"]["detected"][limb]["mean"]
            p = e["arms"]["perfect"][limb]["mean"]
            cost[s][limb] = {"detected": d, "perfect": p, "cost": d - p}
            print(f"  {s:<12}{limb:<6}{d:>8.3f} - {p:>7.3f} = {d - p:>7.3f}")

    # ---- pooled over the eligible subjects --------------------------------
    print(f"\n=== POOLED over diagnostic-eligible subjects, "
          f"subject bootstrap (N_BOOT 2000, SEED 11) ===")
    elig = [s for s, e in rows.items() if e.get("arms")]
    pooled = {"subjects": elig, "n": len(elig), "arms": {}}
    for arm in ("detected", "perfect"):
        pooled["arms"][arm] = {}
        for limb in LIMBS:
            vals = [rows[s]["arms"][arm][limb]["mean"] for s in elig
                    if rows[s]["arms"].get(arm)]
            lo, hi = subject_bootstrap_ci(vals)
            pooled["arms"][arm][limb] = {
                "mae_deg": float(np.mean(vals)), "ci95": [lo, hi],
                "per_subject": {s: rows[s]["arms"][arm][limb]["mean"]
                                for s in elig if rows[s]["arms"].get(arm)}}
            print(f"  {arm:<9}{limb:<6}{np.mean(vals):>8.3f}  "
                  f"[{lo:>7.3f}, {hi:>7.3f}]  n={len(vals)}")
    cvals = [cost[s][lb]["cost"] for s in cost for lb in ("near",)]
    if len(cvals) > 1:
        lo, hi = subject_bootstrap_ci(cvals)
        pooled["detector_cost_near"] = {"mean": float(np.mean(cvals)),
                                        "ci95": [lo, hi], "n": len(cvals)}
        print(f"  detector cost, near limb: {np.mean(cvals):.3f} "
              f"[{lo:.3f}, {hi:.3f}]  n={len(cvals)}")

    # ---- trunk lean vs hip bias ------------------------------------------
    print(f"\n=== TRUNK LEAN vs PERFECT-2D HIP BIAS (no correction applied) ===")
    print(f"  {'subject':<12}{'trunk lean':>12}{'near hip bias':>15}"
          f"{'far hip bias':>14}")
    lean, bias = [], []
    for s, e in rows.items():
        tl = e["trunk_lean"]["deg"]
        a = (e.get("arms") or {}).get("perfect")
        nb = a["near"]["hip_bias"] if a else float("nan")
        fb = a["far"]["hip_bias"] if a else float("nan")
        flag = "" if a else "   (excluded -- lean shown for the record)"
        print(f"  {s:<12}{tl:>12.3f}{nb:>15.3f}{fb:>14.3f}{flag}")
        if a and np.isfinite(tl):
            lean.append(tl)
            bias.append(nb)
    corr = {"n": len(lean)}
    if len(lean) > 2:
        corr["pearson_r"] = float(np.corrcoef(lean, bias)[0, 1])
        print(f"\n  Pearson r across {len(lean)} subjects: "
              f"{corr['pearson_r']:+.3f}")
    else:
        print(f"\n  only {len(lean)} scored subjects -- a correlation across "
              f"this many is not interpretable, and none is reported")
    corr["lean_deg"] = lean
    corr["near_hip_bias_deg"] = bias
    corr["note"] = ("reported side by side only. No correction is applied and "
                    "none should be derived from this n.")

    out = {"what": "DIAGNOSTIC ROWS -- these do not change the phase 15 "
                   "headline (9.610 near / 10.604 far, PG1, n=2)",
           "camera": CAM,
           "designation_commit": "f549bfd, fixed before download",
           "spent": list(SPENT), "diagnostic": list(DIAGNOSTIC),
           "sealed_test_untouched": list(SEALED),
           "subjects": rows, "detector_cost": cost, "pooled": pooled,
           "trunk_lean_vs_hip_bias": corr}
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
