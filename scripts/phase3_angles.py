"""Phase 3 -- joint angles from 3D keypoints, and the error decomposition
phase 2 needs.

Runs on Kaggle (GPU) next to the AthleticsPose checkpoints, and locally (CPU) on
the exported predictions. Imports nothing from the Ferber pipeline, so it is
self-contained -- CLAUDE.md keeps the archive off any cloud environment.

WHAT THIS MUST PRODUCE, and why the shape matters
-------------------------------------------------
Phase 2 measured how the injured-limb signal degrades under two axes: temporal
resolution and FAR-LIMB angular error. It found the signal survives 30 fps and
survives 8 deg of far-limb error, but was not swept beyond 8 deg. Published
monocular knee-flexion MAE against marker-based mocap is 14.1-25.8 deg for
generic models on clinical gait, so the realistic operating point may sit outside
that range.

So phase 3 does not just report an average MAE. It reports:

  1. Sagittal joint-angle MAE per joint (hip / knee / ankle), in degrees.
  2. The SAME split by NEAR vs FAR limb relative to the camera -- this is the
     parameter phase 2 sweeps, and the whole reason the limb task is fragile.
  3. The systematic / random decomposition. Phase 2 injects noise AFTER
     stride-averaging, which models error that is systematic across strides.
     If real error is mostly random it will average out over ~28 strides and
     phase 2 is pessimistic; if it is mostly bias, phase 2 is calibrated.

CONVENTION CAVEAT, for phase 4
------------------------------
Keypoint-derived joint angles are three-point angles between segment vectors.
The Ferber waveforms are Cardan angles from marker-cluster segment coordinate
systems (gait_kinematics.m). These are NOT the same quantity. Within phase 3
that does not matter -- estimate and ground truth use the identical convention,
so the MAE is valid. It matters a great deal in phase 4, where video-derived
angles are fed to a model trained on Cardan angles, and it must be handled there
rather than assumed away.
"""

from __future__ import annotations

import numpy as np

# Human3.6M 17-joint convention, which AthleticsPose / AthletePose3D follow.
H36M = {"pelvis": 0, "r_hip": 1, "r_knee": 2, "r_ankle": 3,
        "l_hip": 4, "l_knee": 5, "l_ankle": 6, "spine": 7, "thorax": 8,
        "neck": 9, "head": 10, "l_shoulder": 11, "l_elbow": 12, "l_wrist": 13,
        "r_shoulder": 14, "r_elbow": 15, "r_wrist": 16}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def sagittal_angles(kp: np.ndarray, side: str) -> dict[str, np.ndarray]:
    """Hip, knee and ankle flexion/extension from 3D keypoints.

    kp: (n_frames, 17, 3) in millimetres or metres -- scale-invariant here.
    side: 'l' or 'r'.

    Angles are signed in the sagittal plane, defined by the body's own frame
    (pelvis->thorax as up, hip-to-hip as the medio-lateral axis) rather than the
    world, so a rotated camera does not change them. Returned in degrees.
    """
    p = kp[:, H36M["pelvis"]]
    thorax = kp[:, H36M["thorax"]]
    l_hip, r_hip = kp[:, H36M["l_hip"]], kp[:, H36M["r_hip"]]

    up = _unit(thorax - p)
    ml = _unit(r_hip - l_hip)                     # medio-lateral
    fwd = _unit(np.cross(ml, up))                 # anterior-posterior
    up = _unit(np.cross(fwd, ml))                 # re-orthogonalise

    hip = kp[:, H36M[f"{side}_hip"]]
    knee = kp[:, H36M[f"{side}_knee"]]
    ankle = kp[:, H36M[f"{side}_ankle"]]

    thigh = _unit(knee - hip)
    shank = _unit(ankle - knee)

    def signed_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Angle from a to b measured in the sagittal plane (normal = ml)."""
        ax = np.sum(a * fwd, axis=-1)
        ay = np.sum(a * up, axis=-1)
        bx = np.sum(b * fwd, axis=-1)
        by = np.sum(b * up, axis=-1)
        return np.degrees(np.arctan2(bx, by) - np.arctan2(ax, ay))

    def wrap(d: np.ndarray) -> np.ndarray:
        return (d + 180.0) % 360.0 - 180.0

    down = -up
    hip_flex = wrap(signed_angle(down, thigh))     # thigh vs trunk-down
    knee_flex = wrap(signed_angle(thigh, shank))   # shank vs thigh
    # Ankle needs a foot vector; H36M has no toe, so ankle dorsiflexion is not
    # recoverable from this skeleton. Reported as unavailable rather than faked.
    return {"hip": hip_flex, "knee": knee_flex, "ankle": np.full_like(hip_flex, np.nan)}


def error_decomposition(est: np.ndarray, gt: np.ndarray,
                        stride_ids: np.ndarray | None = None) -> dict:
    """Split estimation error into systematic bias and random scatter.

    Phase 2 adds noise to the stride-averaged curve, which models error that
    persists across strides. This measures how much of the real error actually
    behaves that way.

    est, gt: (n_frames,) angle series for one joint and one limb.
    stride_ids: (n_frames,) stride index, if strides have been segmented.
    """
    err = est - gt
    out = {
        "mae_deg": float(np.nanmean(np.abs(err))),
        "rmse_deg": float(np.sqrt(np.nanmean(err ** 2))),
        "bias_deg": float(np.nanmean(err)),
        "sd_deg": float(np.nanstd(err)),
    }
    if stride_ids is not None:
        # Per-stride mean error. Its spread is the part that does NOT average
        # out; the residual within strides is what averaging suppresses.
        strides = [err[stride_ids == s] for s in np.unique(stride_ids)]
        per_stride = np.array([np.nanmean(s) for s in strides if len(s)])
        within = np.nanmean([np.nanstd(s) for s in strides if len(s) > 1])
        out["n_strides"] = int(len(per_stride))
        out["systematic_sd_deg"] = float(np.nanstd(per_stride))
        out["within_stride_sd_deg"] = float(within)
        # What phase 2's post-averaging sigma should be set to.
        out["effective_sigma_for_phase2"] = float(
            np.sqrt(np.nanstd(per_stride) ** 2
                    + (within ** 2) / max(len(per_stride), 1)))
    return out


def summarise(rows: list[dict]) -> str:
    """Format the table phase 4 needs: MAE by joint x near/far limb."""
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        return "(no rows)"
    piv = df.pivot_table(index="joint", columns="limb",
                         values=["mae_deg", "bias_deg",
                                 "effective_sigma_for_phase2"],
                         aggfunc="mean")
    return piv.round(2).to_string()


if __name__ == "__main__":
    # Self-check on synthetic data: a known knee flexion must be recovered.
    n = 200
    kp = np.zeros((n, 17, 3))
    kp[:, H36M["pelvis"]] = [0, 0, 0]
    kp[:, H36M["thorax"]] = [0, 0, 500]
    kp[:, H36M["l_hip"]] = [-100, 0, 0]
    kp[:, H36M["r_hip"]] = [100, 0, 0]
    kp[:, H36M["r_knee"]] = [100, 0, -400]
    ang = np.linspace(0, 90, n)
    kp[:, H36M["r_ankle"], 0] = 100
    kp[:, H36M["r_ankle"], 1] = -400 * np.sin(np.radians(ang))
    kp[:, H36M["r_ankle"], 2] = -400 - 400 * np.cos(np.radians(ang))
    got = sagittal_angles(kp, "r")["knee"]
    err = np.max(np.abs(np.abs(got) - ang))
    print(f"synthetic knee sweep 0-90 deg: max recovery error {err:.3f} deg")
    print("PASS" if err < 0.5 else "FAIL -- sagittal angle geometry is wrong")
