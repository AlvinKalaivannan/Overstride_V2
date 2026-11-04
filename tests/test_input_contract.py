"""The input-contract checker must catch the scale defects that actually
happened, and must not fire on correct input.

Three defects in this project came from a channel on the wrong scale, and none
of them raised at the time:

* torchvision's raw `keypoints_scores` (mean 12.4) in a channel trained at
  0.8676 -- 30.07 deg MAE on MoVi at r = +0.02;
* `gt_markers2d_by_cam`'s constant 1.0 in the same channel -- 0.60-1.63 deg on
  AthleticsPose (phase 15E);
* `np.nan_to_num` turning undetected frames into all-zero skeletons at the
  image origin, which the 50%-overlap windows smeared into their neighbours
  (phase 15, D5).

Each has a test here. Written for phase 15F; `tests/test_input_contract.py` did
not exist before it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from input_contract import (CONTRACTS, check, normalize,  # noqa: E402
                            summarize)

ATH = CONTRACTS["b-ath-det-coco"]
H36M = CONTRACTS["b-h36m"]
RES_AP = (1920, 1088)
RES_PG1 = (800, 600)


def pixels(n: int = 40, w: int = 1920, h: int = 1088, seed: int = 0):
    """A plausible block of pixel keypoints inside the image."""
    rng = np.random.default_rng(seed)
    cx = rng.uniform(0.3 * w, 0.7 * w, size=(n, 1))
    cy = rng.uniform(0.3 * h, 0.7 * h, size=(n, 1))
    xy = np.stack([cx + rng.normal(0, 0.05 * w, (n, 17)),
                   cy + rng.normal(0, 0.10 * h, (n, 17))], axis=-1)
    xy[:, :, 0] = np.clip(xy[:, :, 0], 0, w)
    xy[:, :, 1] = np.clip(xy[:, :, 1], 0, h)
    return np.concatenate([xy, np.full((n, 17, 1), 0.9)], axis=-1)


# --------------------------------------------------------------- clean -----

def test_person_scale_prepared_input_is_clean():
    x = normalize(pixels(), ATH)
    assert check(x, ATH) == []
    assert np.abs(x[:, :, :2]).max() <= 1.0 + 1e-9


def test_image_res_prepared_input_is_clean():
    x = normalize(pixels(w=RES_AP[0], h=RES_AP[1]), H36M, RES_AP)
    assert check(x, H36M, RES_AP) == []


def test_image_res_respects_the_aspect_ratio_bound():
    """y is divided by res_w, not res_h, so its bound is res_h/res_w."""
    w, h = RES_PG1
    xy = np.zeros((5, 17, 2))
    xy[:, :, 1] = h                      # bottom edge of the image
    kp = np.concatenate([xy, np.ones((5, 17, 1))], axis=-1)
    x = normalize(kp, H36M, RES_PG1)
    assert x[:, :, 1].max() == pytest.approx(h / w, abs=1e-9)
    assert check(x, H36M, RES_PG1) == []


def test_person_scale_is_root_relative():
    kp = pixels()
    x = normalize(kp, ATH)
    assert np.allclose(x[:, 0, :2], 0.0)


# ------------------------------------------------- the real defects --------

def test_flags_raw_torchvision_scores():
    """The 30.07 deg MoVi defect: scores on a 0-30 scale, trained at 0.8676."""
    x = normalize(pixels(), ATH)
    x[:, :, 2] = 12.4
    flags = check(x, ATH)
    assert any("outside [0, 1]" in f for f in flags), flags
    assert any("30.07" in f for f in flags), flags


def test_flags_constant_one_confidence():
    """The phase 15E defect: gt_markers2d_by_cam ships a constant 1.0."""
    x = normalize(pixels(), ATH)
    x[:, :, 2] = 1.0
    flags = check(x, ATH)
    assert any("confidence mean" in f for f in flags), flags
    # and it must NOT fire for the checkpoint whose contract IS 1.0
    y = normalize(pixels(), H36M, RES_AP)
    assert check(y, H36M, RES_AP) == []


def test_flags_all_zero_frames():
    """The phase 15 D5 sentinel: skeletons at the image origin."""
    x = normalize(pixels(), ATH)
    x[3:8, :, :2] = 0.0
    flags = check(x, ATH)
    assert any("all-zero frames" in f for f in flags), flags
    assert any("5 all-zero" in f for f in flags), flags


def test_flags_wrong_normalization_fed_to_image_res_contract():
    """Person-scale input handed to the image-resolution contract IS caught.

    Person-scale allows |y| up to 1.0; image_res bounds y at res_h/res_w, which
    is 0.567 for 1920x1088. So the aspect ratio catches the swap.
    """
    x = normalize(pixels(), ATH)
    flags = check(x, H36M, RES_AP)
    assert any("max|y|" in f for f in flags), flags


def test_the_aspect_bound_cannot_catch_the_swap_on_a_SQUARE_image():
    """The honest limit of the check above.

    When res_h == res_w the image_res bound is also 1.0, so person-scale input
    sits inside it and only the confidence channel gives the swap away. Pinned
    so nobody reads the previous test as "normalization swaps are always
    caught" -- they are caught by a RANGE, not by a check of what the numbers
    mean.
    """
    square = (1000, 1000)
    x = normalize(pixels(w=1000, h=1000), ATH)
    flags = [f for f in check(x, H36M, square) if "confidence" not in f]
    assert flags == [], flags


def test_flags_out_of_image_coordinates():
    w, h = RES_AP
    kp = pixels(w=w, h=h)
    kp[:, :, 0] += 3 * w                 # far outside the frame
    x = normalize(kp, H36M, RES_AP)
    flags = check(x, H36M, RES_AP)
    assert any("max|x|" in f for f in flags), flags


def test_flags_bad_shape_and_nonfinite():
    assert any("not (T, 17, 3)" in f
               for f in check(np.zeros((10, 17, 2)), ATH))
    x = normalize(pixels(), ATH)
    x[0, 0, 0] = np.nan
    assert any("non-finite" in f for f in check(x, ATH))


# ------------------------------------------------------- the contracts -----

def test_every_contract_declares_its_window_from_the_weights():
    """All three checkpoints are 81-frame models, read from the temporal
    BatchNorm shape -- not 243, which the h36m config implies."""
    for c in CONTRACTS.values():
        assert c.n_frames == 81, (c.name, c.n_frames)


def test_undetermined_items_are_excluded_from_the_primary():
    for c in CONTRACTS.values():
        if c.undetermined and "normalization" in c.undetermined:
            assert not c.include_in_primary, c.name
    assert CONTRACTS["s-ap3d"].include_in_primary is False
    assert CONTRACTS["b-ath-det-coco"].determined == 5


def test_summarize_reports_the_channels():
    s = summarize(normalize(pixels(), ATH))
    assert s["conf_mean"] == pytest.approx(0.8676, abs=1e-6)
    assert s["n_frames"] == 40
    assert -1.0 <= s["x_min"] <= s["x_max"] <= 1.0
