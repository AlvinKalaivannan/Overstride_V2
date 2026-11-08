"""M5: the HTML report. One self-contained file, no network, no JavaScript.

Every report in this file is rendered from `scripts/synthetic_clip.py` -- a
procedurally generated stick figure whose angles are known by construction.
**No dataset frame is decoded, rendered or written here**, which is what lets
an example report be committed and looked at.

The M5 bars from `docs/phase16-demo-design.md`:

* no `http://` or `https://` resource reference except the attribution
  citations;
* it opens with JavaScript disabled;
* under 25 MB for a 15 s clip.

Plus the content the design requires of it: a headline readout per leg with
its stated error, the overlay strip, stride-averaged traces, the trunk panel
marked experimental, the checks as a pass/warn list, and a scope section.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from overstride.budget import UNMEASURED, load_budget  # noqa: E402
from overstride.checks import post_detect_checks, pre_decode_checks  # noqa: E402
from overstride.preprocess import ClipInfo  # noqa: E402
from overstride.report import overlay_frames, render, write_report  # noqa: E402
from overstride.strides import segment  # noqa: E402
from synthetic_clip import FRAME_H, FRAME_W, synthetic_result  # noqa: E402


@pytest.fixture(scope="module")
def clip():
    res, frames, _kp3d = synthetic_result(n_frames=300, fps=60.0)
    info = ClipInfo(path=Path("synthetic-runner.mp4"), fps=60.0,
                    width=FRAME_W, height=FRAME_H, n_frames=300,
                    duration_s=5.0)
    checks = (pre_decode_checks(info)
              + post_detect_checks(res.kp2d, res.detected, res.detect_meta))
    return res, frames, checks


@pytest.fixture(scope="module")
def html(clip):
    res, frames, checks = clip
    return render(res, strides=segment(res.angles, res.fps), checks=checks,
                  budget=load_budget(), clip_name="synthetic-runner.mp4",
                  frames=frames, today="2026-11-08")


# ------------------------------------------------- self-contained ---------

def test_no_external_resource_is_fetched(html):
    """The only http strings may be attribution links a reader may follow."""
    for m in re.finditer(r'(src|href)\s*=\s*"([^"]+)"', html):
        attr, url = m.group(1), m.group(2)
        if attr == "src":
            assert url.startswith("data:"), f"fetched resource: {url}"
        elif url.startswith("http"):
            assert ("arxiv.org" in url or "doi.org" in url), (
                f"non-attribution link: {url}")


def test_it_opens_with_javascript_disabled(html):
    assert "<script" not in html.lower()
    assert "onclick" not in html.lower()
    assert "javascript:" not in html.lower()
    assert "<noscript" not in html.lower(), "nothing should need a fallback"


def test_nothing_is_loaded_from_a_cdn_or_a_font_service(html):
    for host in ("cdn.", "googleapis", "unpkg", "jsdelivr", "fonts."):
        assert host not in html


def test_the_file_is_small_enough_for_a_15_second_clip(tmp_path):
    """The M5 bar is 25 MB; a 15 s clip at 60 fps is 900 frames."""
    res, frames, _ = synthetic_result(n_frames=900, fps=60.0)
    out = write_report(tmp_path / "big.html", res, frames=frames,
                       strides=segment(res.angles, res.fps))
    mb = out.stat().st_size / 1024 / 1024
    assert mb < 25.0, f"{mb:.1f} MB"


def test_it_is_one_file(tmp_path, clip):
    res, frames, checks = clip
    out = write_report(tmp_path / "r.html", res, frames=frames, checks=checks)
    assert [p.name for p in tmp_path.iterdir()] == ["r.html"]
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


# ------------------------------------------------------- the content ------

def test_the_headline_shows_both_quantities_on_both_legs(html):
    assert html.count("Knee flexion, peak") == 2
    assert html.count("Thigh inclination against gravity, peak") == 2
    assert "left leg" in html and "right leg" in html
    assert "nearer the camera" in html
    assert "further from the camera" in html


def test_each_reading_carries_its_stated_error(html, clip):
    """The error is part of the reading, which is the document's whole thesis."""
    res, _f, _c = clip
    b = load_budget()
    for quantity, limb in (("knee_from_segments", "near"),
                           ("knee_from_segments", "far"),
                           ("thigh", "near"), ("thigh", "far")):
        cell = b.cell("B", quantity, limb)
        assert f"{cell.value:.2f}" in html, f"{quantity} {limb} error missing"
        assert f"{cell.ci[0]:.2f}" in html


def test_the_angles_are_the_ones_the_clip_was_built_with(clip):
    """A renderer that silently changed a number would pass every other test.

    The synthetic runner's knee peaks at 35 + 32 = 67 deg and its thigh at
    10 + 25 = 35, by construction in `scripts/synthetic_clip.py`.
    """
    res, _f, _c = clip
    import numpy as np
    for side in ("l", "r"):
        assert np.nanmax(res.series(side, "knee_from_segments")) == (
            pytest.approx(67.0, abs=0.2))
        assert np.nanmax(res.series(side, "thigh")) == pytest.approx(
            35.0, abs=0.2)


def test_the_two_uncertainties_are_kept_apart(html):
    assert "stride-to-stride spread" in html
    assert "measurement error" in html
    assert "never added" in html


def test_the_stride_traces_name_their_cycle_boundary(html):
    assert "peak swing-phase knee flexion" in html
    assert "not at foot contact" in html


def test_a_failed_segmentation_falls_back_to_per_frame_traces(clip):
    """M3's fallback: no stride table, and the reason is printed."""
    res, frames, checks = clip
    bad = segment(res.angles, res.fps)
    bad.ok = False
    bad.reasons = ["left leg: only 1 usable stride(s), fewer than 3"]
    out = render(res, strides=bad, checks=checks, frames=None)
    assert "Strides were not segmented" in out
    assert "only 1 usable stride" in out
    assert "Stride cycles found" not in out
    assert "highest value in the clip" in out


def test_no_strides_at_all_still_renders(clip):
    res, _frames, checks = clip
    out = render(res, strides=None, checks=checks, frames=None)
    assert "no stride segmentation was run" in out
    assert "<title>" in out


def test_the_overlay_strip_is_inline_png(html):
    assert html.count('src="data:image/png;base64,') >= 4
    assert "What the detector saw" in html


def test_overlay_frames_are_all_the_same_size(clip):
    """A fixed crop box, centred per frame, so the strip is a grid not a mess."""
    import base64
    import cv2
    import numpy as np
    res, frames, _ = clip
    shots = overlay_frames(frames, res.kp2d, res.detected)
    assert len(shots) >= 4
    sizes = set()
    for _i, b64 in shots:
        buf = np.frombuffer(base64.b64decode(b64), np.uint8)
        sizes.add(cv2.imdecode(buf, cv2.IMREAD_COLOR).shape[:2])
    assert len(sizes) == 1, f"ragged thumbnails: {sizes}"


def test_overlay_handles_no_frames_and_no_keypoints(clip):
    res, frames, _ = clip
    assert overlay_frames([], res.kp2d) == []
    assert overlay_frames(frames, None) == []


def test_the_checks_are_listed_with_their_state(html):
    assert "Input checks" in html
    for name in ("fps", "resolution", "length", "person", "viewpoint"):
        assert f">{name}<" in html
    assert 'class="state pass"' in html


def test_the_trunk_panel_is_marked_experimental(html):
    assert "experimental" in html
    i = html.index("Trunk lean")
    panel = html[i - 400:i + 2200]
    assert "experimental" in panel
    assert "direction, not a number" in panel
    assert "-2.34" in panel or "−2.34" in panel


def test_the_scope_section_says_what_is_not_measured(html):
    assert "What this does and does not measure" in html
    assert "not a clinical assessment" in html
    assert "no health claim" in html
    assert "--near-side" in html


def test_the_budget_section_cites_its_sources(html):
    assert "results/phase15_sealed.json" in html
    assert "pooled/detected/knee_from_segments/near/mae" in html
    assert "results/phase15e.json" in html


def test_row_c_is_present_and_marked_as_tune_set_context(html):
    assert "Why the filming instructions ask for 60 fps" in html
    assert "in-domain on the tune set" in html
    assert "a correction you can apply to the error above" in html
    assert "never combines them" in html


def test_unmeasured_conditions_are_printed(html):
    assert html.count(UNMEASURED) >= 4
    assert "60 fps, anywhere" in html
    assert "phone or consumer footage" in html


def test_the_footer_attributes_every_component(html):
    for who in ("MotionAGFormer", "AthleticsPose", "MoVi", "Keypoint R-CNN"):
        assert who in html
    assert "No frame of any research dataset appears in it" in html


def test_both_themes_are_defined(html):
    """The viewer has three states: light, dark, and an unstamped default."""
    assert "prefers-color-scheme:dark" in html
    assert ':root:not([data-theme="light"])' in html
    assert ':root[data-theme="dark"]' in html
    assert "body{margin:0;background:var(--paper)" in html


def test_it_declares_a_viewport_and_a_title(html):
    assert 'name="viewport"' in html
    assert "<title>Overstride measurement report" in html
    assert 'lang="en"' in html
