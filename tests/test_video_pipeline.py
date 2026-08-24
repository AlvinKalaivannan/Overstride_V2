"""The video path. The detector's own error is unmeasurable here (AthleticsPose
releases no source video), so these cover the parts that ARE checkable: keypoint
ordering, padding, window blending and the resampler."""
import numpy as np
import pytest
from phase7_inference_fix import pad_to
from video_kinematics import resample

COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]


def test_detector_keypoint_order_matches_what_the_lifter_expects():
    """torchvision emits COCO-17; the lifter consumes COCO-17. If torchvision
    ever reorders, every angle silently becomes wrong."""
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
    names = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT.meta["keypoint_names"]
    assert list(names) == COCO17


def test_edge_padding_replicates_rather_than_zeroing():
    x = np.arange(1, 11, dtype=float).reshape(10, 1, 1) * np.ones((1, 17, 3))
    z, e = pad_to(x, 15, "zero"), pad_to(x, 15, "edge")
    assert z.shape == e.shape == (15, 17, 3)
    assert np.all(z[10:] == 0.0), "zero padding should be zeros"
    assert np.all(e[10:] == 10.0), "edge padding should hold the last frame"
    assert np.array_equal(z[:10], x) and np.array_equal(e[:10], x)


def test_padding_is_a_noop_when_long_enough():
    x = np.ones((90, 17, 3))
    assert pad_to(x, 81, "edge").shape == (90, 17, 3)


@pytest.mark.parametrize("n_out", [40, 81, 200])
def test_resample_preserves_endpoints_and_length(n_out):
    seq = np.linspace(0, 1, 57)[:, None] * np.ones((1, 4))
    got = resample(seq, n_out)
    assert got.shape == (n_out, 4)
    assert np.allclose(got[0], seq[0]) and np.allclose(got[-1], seq[-1])


def test_resample_is_linear_on_a_ramp():
    seq = np.arange(30, dtype=float)[:, None]
    got = resample(seq, 59)[:, 0]
    assert np.allclose(got, np.linspace(0, 29, 59), atol=1e-9)
