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


def test_torchvision_still_emits_the_coco_order_the_mapping_assumes():
    """The mapping is index arithmetic over this exact list. If torchvision ever
    reorders its output, the mapping silently sends every joint somewhere else.

    This test used to also claim "the lifter consumes COCO-17". It does not, and
    the claim is why `video_kinematics.py` fed it a permuted skeleton -- see the
    two tests below.
    """
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
    names = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT.meta["keypoint_names"]
    assert list(names) == COCO17


def test_the_lifters_input_convention_is_h36m_read_from_the_released_data():
    """Not from a comment, and not from a dataset card: from the files the
    checkpoint was actually fed.

    In H36M-17 the ankles are joints 3 and 6 and the head is joint 10, so in a
    standing or running frame the ankles are the lowest points in the image and
    the head the highest. In COCO-17 those indices are an ear, a shoulder and a
    wrist. Upstream does no reordering between loading these files and the
    model (`datasets/dynamic_dataset.py.__getitem__`), so the file order *is*
    the model's input convention.
    """
    from conftest import DATASET

    root = DATASET / "det_markers2d_by_cam_coco" / "running"
    files = sorted(root.glob("*/*.npy")) if root.is_dir() else []
    if not files:
        pytest.skip(f"AthleticsPose detections not on disk at {root}")
    a = np.load(files[0]).astype(float)
    y = a[:, :, 1].mean(axis=0)                      # image y grows downward
    assert set(np.argsort(-y)[:2]) == {3, 6}, "joints 3 and 6 should be ankles"
    assert int(np.argmin(y)) == 10, "joint 10 should be the head"


def test_coco_to_h36m_puts_every_joint_where_the_lifter_looks_for_it():
    """A round trip over the joints that correspond one-to-one, which is where a
    permutation bug lives. The constructed joints -- pelvis, spine, thorax, neck
    -- are checked against their definitions instead, since mocap's pelvis is
    not the mid-hip and never was (phase 12 measured that offset)."""
    from phase3_angles import H36M
    from video_kinematics import COCO, coco17_to_h36m17

    rng = np.random.default_rng(11)
    h = rng.normal(size=(30, 17, 3)) * 100.0
    h[:, :, 2] = rng.uniform(0.4, 1.0, size=(30, 17))

    # the same skeleton written out in COCO order
    coco = np.zeros_like(h)
    for name, ci in COCO.items():
        if name in H36M:
            coco[:, ci] = h[:, H36M[name]]
    coco[:, COCO["nose"]] = h[:, H36M["head"]]

    got = coco17_to_h36m17(coco)
    for name in ("r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
                 "l_shoulder", "l_elbow", "l_wrist",
                 "r_shoulder", "r_elbow", "r_wrist", "head"):
        assert np.allclose(got[:, H36M[name], :2], h[:, H36M[name], :2]), name
    assert np.allclose(got[:, H36M["pelvis"], :2],
                       0.5 * (h[:, H36M["l_hip"], :2] + h[:, H36M["r_hip"], :2]))
    assert np.allclose(got[:, H36M["thorax"], :2],
                       0.5 * (h[:, H36M["l_shoulder"], :2]
                              + h[:, H36M["r_shoulder"], :2]))
    # and it is not a no-op: feeding COCO order straight in moves 15 of 17
    moved = ~np.all(np.isclose(got[:, :, :2], coco[:, :, :2]), axis=(0, 2))
    assert moved.sum() >= 15, f"only {moved.sum()} joints moved"


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


def test_model_fps_matches_the_training_set_declared_capture_rate():
    """MODEL_FPS is not a free parameter: it is the rate the lifter's training
    data was captured at, and AthleticsPose declares it in every QTM export.
    The previous value, 120.0, was inferred from stride cadence instead -- the
    documented figure outranks the inference, and nothing may silently drift
    back. Read from the dataset, never from a second hardcoded constant."""
    import json

    from conftest import DATASET
    from video_kinematics import MODEL_FPS

    params = sorted((DATASET / "camera_params").glob("*.json"))
    if not params:
        pytest.skip(f"AthleticsPose camera_params not on disk at {DATASET}")
    rates = {}
    for p in params:
        rates[p.name] = json.loads(p.read_text(encoding="utf-8"))["Timebase"]["Frequency"]
    assert len(set(rates.values())) == 1, f"mixed capture rates: {rates}"
    declared = next(iter(set(rates.values())))
    assert MODEL_FPS == declared, (
        f"MODEL_FPS={MODEL_FPS} but the lifter's training set was captured at "
        f"{declared} Hz across {len(rates)} QTM exports")


def test_blend_weights_are_one_per_frame_and_peak_at_window_centres():
    """phase 14 reads the per-frame blend weight out of `lift` rather than
    reimplementing the window arithmetic beside it, where it could drift. This
    pins the contract: one weight per frame, and a frame covered by two windows
    is weighted more than a frame at the very edge."""
    import torch
    from phase7_inference_fix import CLIP, lift

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    n = 200
    rng = np.random.default_rng(3)
    det = np.concatenate([rng.normal(size=(n, 17, 2)) * 40.0 + 200.0,
                          np.ones((n, 17, 1))], axis=-1)
    out, w = lift(Identity(), det, "edge", "overlap", with_weights=True)
    assert out.shape == (n, 17, 3) and w.shape == (n,)
    assert np.all(w > 0)
    assert w[CLIP // 2] > w[0]

    out2, w2 = lift(Identity(), det, "edge", "consecutive", with_weights=True)
    assert np.allclose(w2, 1.0), "consecutive windows cover each frame once"
    assert np.allclose(out2, lift(Identity(), det, "edge", "consecutive"))
