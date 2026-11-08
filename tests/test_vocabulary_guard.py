"""M6: the vocabulary and output guard, including the `[A4]` hip check.

`docs/phase16-demo-design.md` milestone M6. A guard that never fires is a
guard nobody has tested, so every rule here is checked twice: once by planting
a violation and watching it caught, and once by running the guard over the
repository's real demo surface and watching it pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from check_vocabulary import (BANNED, check_html, check_outputs,  # noqa: E402
                              check_pragmas, main, scan_text, surface)

PAGE = ("<!doctype html><html><head><title>t</title></head><body>"
        "<p>{}</p></body></html>")


# ------------------------------------------------- every term is caught ---

@pytest.mark.parametrize("term,text", [
    ("risk", "the risk of this is low"),
    ("risk", "RISKS are not reported"),
    ("predict", "this predicts the outcome"),
    ("predict", "a prediction over time"),
    ("injury", "an injury was detected"),
    ("injury", "the injured limb"),
    ("diagnosis", "a diagnosis follows"),
    ("diagnosis", "diagnostic output"),
    ("abnormal", "an abnormal gait"),
    ("asymmetry index", "the asymmetry index is 1.2"),
    ("asymmetry index", "the asymmetry-index is 1.2"),
    ("normal range", "inside the normal range"),
    ("normal range", "inside the normal ranges"),
])
def test_a_banned_term_is_caught_in_source(term, text, tmp_path):
    p = tmp_path / "m.py"
    p.write_text(f'"""{text}"""\n', encoding="utf-8")
    hits = scan_text(p.read_text(encoding="utf-8"), p)
    assert hits, f"{term!r} not caught in {text!r}"
    assert hits[0].term == term


@pytest.mark.parametrize("text", [t for _k, t in [
    ("risk", "the risk of this is low"), ("predict", "this predicts it"),
    ("injury", "an injury"), ("diagnosis", "a diagnosis"),
    ("abnormal", "abnormal gait")]])
def test_a_banned_term_is_caught_in_a_report(text, tmp_path):
    p = tmp_path / "report.html"
    p.write_text(PAGE.format(text), encoding="utf-8")
    assert check_html(p), f"{text!r} not caught in a report"


def test_clean_text_passes(tmp_path):
    p = tmp_path / "m.py"
    p.write_text('"""Knee flexion, measured, with its error."""\n',
                 encoding="utf-8")
    assert scan_text(p.read_text(encoding="utf-8"), p) == []


# ---------------------------------------------------------- the pragma ----

def test_the_pragma_allows_an_engineering_use_with_a_reason(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("x = 1  # diagnostic  # vocab-ok: internal detail only\n",
                 encoding="utf-8")
    text = p.read_text(encoding="utf-8")
    assert scan_text(text, p) == []
    assert check_pragmas(text, p) == []


def test_a_pragma_without_a_reason_fails(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("x = 1  # diagnostic  # vocab-ok:\n", encoding="utf-8")
    hits = check_pragmas(p.read_text(encoding="utf-8"), p)
    assert hits and "no reason" in hits[0].why


def test_a_pragma_on_a_line_that_needs_none_fails(tmp_path):
    """Otherwise pragmas accumulate and nobody notices when one stops mattering."""
    p = tmp_path / "m.py"
    p.write_text("x = 1  # vocab-ok: nothing banned here\n", encoding="utf-8")
    hits = check_pragmas(p.read_text(encoding="utf-8"), p)
    assert hits and "needs no exception" in hits[0].why


def test_a_report_gets_no_pragma_exemption(tmp_path):
    """The reader sees the report, so the ban there is absolute."""
    p = tmp_path / "report.html"
    p.write_text(PAGE.format("injury risk  <!-- vocab-ok: please -->"),
                 encoding="utf-8")
    hits = check_html(p)
    assert len(hits) >= 2
    assert all("in output" in h.why for h in hits)


# ------------------------------------------------- `[A4]` the hip check ---

def test_a_hip_trace_in_a_report_fails(tmp_path):
    for text in ("hip flexion", "Hip", "the hips", "<text>hip angle</text>"):
        p = tmp_path / "report.html"
        p.write_text(PAGE.format(text), encoding="utf-8")
        assert any(h.term == "hip" for h in check_html(p)), text


def test_words_containing_hip_are_not_flagged(tmp_path):
    p = tmp_path / "report.html"
    p.write_text(PAGE.format("the relationship between ship and chip"),
                 encoding="utf-8")
    assert check_html(p) == []


def test_the_committed_example_report_has_no_hip_and_no_script():
    out = REPO / "figures" / "example-report.html"
    if not out.exists():
        pytest.skip("run scripts/synthetic_clip.py")
    assert check_html(out) == []


# --------------------------------------------------- the other output ----

def test_a_script_tag_in_a_report_fails(tmp_path):
    p = tmp_path / "report.html"
    p.write_text(PAGE.format("fine") + "<script>alert(1)</script>",
                 encoding="utf-8")
    hits = check_html(p)
    assert any(h.term == "script" for h in hits)


def test_a_debug_file_beside_a_report_fails(tmp_path):
    report = tmp_path / "report.html"
    report.write_text(PAGE.format("fine"), encoding="utf-8")
    assert check_outputs(report) == []
    (tmp_path / "report-debug.html").write_text("x", encoding="utf-8")
    hits = check_outputs(report)
    assert hits and "never published" in hits[0].why


def test_the_debug_file_is_not_itself_part_of_the_surface(tmp_path):
    _py, htm = surface([str(tmp_path / "r-debug.html")])
    assert not any(p.name.endswith("-debug.html") for p in htm)


# ------------------------------------------- the real surface is clean ----

def test_the_demo_surface_is_covered():
    py, _htm = surface()
    names = {p.name for p in py}
    for expect in ("analyze.py", "budget.py", "checks.py", "nearside.py",
                   "report.py", "strides.py", "overstride_analyze.py"):
        assert expect in names, f"{expect} is not guarded"


def test_the_repository_passes_its_own_guard(capsys):
    assert main([]) == 0, capsys.readouterr().out


def test_every_term_in_the_design_is_in_the_guard():
    for term in ("risk", "predict", "injury", "diagnosis", "abnormal",
                 "asymmetry index", "normal range"):
        assert term in BANNED
