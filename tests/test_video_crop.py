"""The crop path: a 5.1x speed-up whose correctness lives in one coordinate map.

`detect_coco17` runs the detector on a padded crop that tracks the previous
frame, so every keypoint it returns has been translated from crop coordinates
back into the full frame. That translation is a seam, and this pipeline has
broken at a seam twice in one day -- COCO indices into H36M-ordered data, in two
different files. So the map is tested rather than trusted, with a stub that
behaves like a real detector: it is handed pixels, it finds the subject in
*those* pixels, and it answers in their coordinate frame, knowing nothing about
where in the full frame they came from. If `detect_coco17` forgets to add the
crop's origin back, the recovered positions move when the crop moves.

The last test runs main() end to end on a real video file. Nothing exercised
that wiring before, which is how a joint-ordering bug survived inside it.
"""
import numpy as np
import pytest

from video_kinematics import CROP_PAD, crop_window, detect_coco17

# a tall, thin bright shape standing in for a runner, at a known position
BOX = (280, 150, 320, 350)       # x0, y0, x1, y1 in the full frame
FRAME_H, FRAME_W = 480, 640


def painted_frame() -> np.ndarray:
    f = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    f[BOX[1]:BOX[3], BOX[0]:BOX[2]] = 255
    return f


def skeleton_in(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """17 keypoints down the vertical centre line of a box."""
    return np.column_stack([np.full(17, 0.5 * (x0 + x1)),
                            np.linspace(y0, y1 - 1, 17)])


class _PixelStub:
    """Finds the bright shape in whatever sub-image it is handed.

    Answers in the sub-image's own coordinates, exactly as a real detector
    does -- it has no idea what the crop's origin was.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, batch):
        import torch
        t = batch[0]
        self.calls.append((int(t.shape[1]), int(t.shape[2])))
        mask = t.numpy().max(axis=0) > 0.5
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return [{"scores": torch.zeros(0), "keypoints": torch.zeros(0, 17, 3),
                     "keypoints_scores": torch.zeros(0, 17),
                     "boxes": torch.zeros(0, 4)}]
        local = skeleton_in(int(xs.min()), int(ys.min()),
                            int(xs.max()) + 1, int(ys.max()) + 1)
        k = np.concatenate([local, np.ones((17, 1))], axis=1)
        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                       dtype=np.float32)
        return [{"scores": torch.tensor([0.99]),
                 "keypoints": torch.tensor(k[None], dtype=torch.float32),
                 "keypoints_scores": torch.full((1, 17), 0.95),
                 "boxes": torch.tensor(box[None], dtype=torch.float32)}]


def test_crop_window_pads_proportionally_and_stays_inside_the_frame():
    x0, y0, x1, y1 = crop_window(np.array([100.0, 200.0, 200.0, 400.0]),
                                 FRAME_H, FRAME_W)
    assert (x0, y0, x1, y1) == (100 - int(CROP_PAD * 100), 200 - int(CROP_PAD * 200),
                                200 + int(CROP_PAD * 100), 400 + int(CROP_PAD * 200))
    # a box against the edge clips rather than going negative or past the frame
    assert crop_window(np.array([0.0, 0.0, 50.0, 50.0]), FRAME_H, FRAME_W)[:2] == (0, 0)
    assert crop_window(np.array([600.0, 440.0, 640.0, 480.0]),
                       FRAME_H, FRAME_W)[2:] == (FRAME_W, FRAME_H)


def test_keypoints_come_back_in_full_frame_coordinates():
    frames = [painted_frame() for _ in range(5)]
    stub = _PixelStub()
    kp, meta = detect_coco17(frames, crop=True, model=stub)

    want = skeleton_in(*BOX)
    for i in range(len(frames)):
        assert np.allclose(kp[i, :, :2], want, atol=1.0), (
            f"frame {i} is off by {np.abs(kp[i, :, :2] - want).max():.1f} px -- "
            "the crop's origin was not added back")
    assert meta["frames_with_person"] == len(frames)
    assert meta["frames_on_crop"] >= len(frames) - 1, meta
    assert meta["full_frame_redetections"] == 0, meta
    assert meta["detector_min_size"] == min(FRAME_H, FRAME_W)


def test_the_first_pass_sees_the_whole_frame_and_the_rest_do_not():
    frames = [painted_frame() for _ in range(4)]
    stub = _PixelStub()
    detect_coco17(frames, crop=True, model=stub)
    assert stub.calls[0] == (FRAME_H, FRAME_W), "the first pass must see it all"
    h, w = stub.calls[-1]
    assert h * w < FRAME_H * FRAME_W * 0.5, f"crop {h}x{w} is not much smaller"


def test_no_crop_runs_every_frame_full_size():
    frames = [painted_frame() for _ in range(3)]
    stub = _PixelStub()
    _, meta = detect_coco17(frames, crop=False, model=stub)
    assert all(c == (FRAME_H, FRAME_W) for c in stub.calls), stub.calls
    assert meta["frames_on_crop"] == 0


def test_losing_the_subject_costs_a_full_frame_redetection_not_a_guess():
    """The tracker is allowed to fail; it is not allowed to invent a skeleton."""
    frames = [painted_frame(), np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8),
              painted_frame()]
    stub = _PixelStub()
    kp, meta = detect_coco17(frames, crop=True, model=stub)
    assert np.all(kp[1] == 0.0), "the empty frame must stay empty"
    assert meta["frames_with_person"] == 2
    # and the frame after the loss is recovered by a full-frame pass
    assert np.allclose(kp[2, :, :2], skeleton_in(*BOX), atol=1.0)


def test_a_clip_with_no_subject_is_refused_and_writes_nothing(tmp_path):
    """End-to-end through main(): decode, detect, refuse, write nothing."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("torchvision")
    import sys

    import video_kinematics as vk

    path = tmp_path / "noise.mp4"
    rng = np.random.default_rng(1)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    if not wr.isOpened():
        pytest.skip("no mp4 writer available in this OpenCV build")
    for _ in range(24):
        wr.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    wr.release()

    out = tmp_path / "out"
    out.mkdir()
    argv = sys.argv
    sys.argv = ["video_kinematics.py", "--video", str(path), "--out", str(out),
                "--max-frames", "24"]
    try:
        code = vk.main()
    finally:
        sys.argv = argv
    assert code == 2, "noise contains no runner; the tool must refuse"
    assert not list(out.iterdir()), f"refused but still wrote {list(out.iterdir())}"


def test_smoothing_is_off_by_default_and_opt_in():
    """Phase 13 arm 1 measured the 2D low-pass as harmful: +0.119 deg near with
    a CI excluding zero at the sigma this tool applied, and up to +4.6 deg at
    stronger settings. It stays available so the measurement can be reproduced,
    but it must not be on unless asked for."""
    from video_kinematics import LOWPASS_HZ, smooth

    kp = np.zeros((40, 17, 3))
    kp[:, :, 0] = np.arange(40)[:, None] % 7          # something jagged
    assert LOWPASS_HZ == 0.0, "the default must be off"
    assert np.array_equal(smooth(kp, 30.0), kp), "default must not alter the 2D"
    assert np.array_equal(smooth(kp, 30.0, 0.0), kp)
    changed = smooth(kp, 100.0, 6.0)
    assert not np.array_equal(changed, kp), "an explicit cutoff must still filter"
    assert np.array_equal(changed[:, :, 2], kp[:, :, 2]), "scores are not a track"
