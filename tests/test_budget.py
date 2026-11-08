"""M4: every displayed number traces to the result file it cites.

`docs/phase16-demo-design.md` section 3 and milestone M4:

* no number may be hardcoded without a matching source -- proved twice, once
  by resolving every cell's cited JSON pointer and once by loading a PATCHED
  copy of the result file and watching the output move;
* an unmeasured condition renders the word `UNMEASURED`;
* `[A2]` Row B and Row C are never combined -- for a 60 fps clip the displayed
  expected error must equal a Row B value **exactly**;
* `[A3]` Row A is labelled as the best case, in-domain;
* `[A4]` no hip figure appears anywhere in the budget.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from overstride.budget import (EXPERIMENTAL, NOT_MEASURED,  # noqa: E402
                               PRIMARY, QUANTITIES, ROW_B_CAVEATS,
                               ROW_B_FPS, ROW_LABELS, UNMEASURED,
                               Budget, MissingSource, load_budget, resolve)

RESULTS = REPO / "results"
SOURCES = ("phase15_sealed.json", "phase15e.json", "phase15f.json")


@pytest.fixture(scope="module")
def budget() -> Budget:
    return load_budget()


# ------------------------------------------- every number has a source ----

def test_every_cell_resolves_to_the_path_it_cites(budget):
    """The M4 bar: no displayed number without a matching source."""
    docs = {}
    for cell in budget.cells:
        name = Path(cell.source_file).name
        if name not in docs:
            docs[name] = json.loads((RESULTS / name).read_text(
                encoding="utf-8"))
        got = float(resolve(docs[name], cell.source_path))
        assert got == pytest.approx(cell.value, abs=1e-12), (
            f"{cell.row}/{cell.quantity}/{cell.limb} displays {cell.value} "
            f"but {cell.source_file}:{cell.source_path} holds {got}")


def test_every_cell_cites_a_file_that_exists(budget):
    for cell in budget.cells:
        assert (REPO / cell.source_file).exists(), cell.source_file
        assert cell.source_file.startswith("results/")
        assert cell.conditions, f"{cell.row} cell has no conditions"


def test_the_values_are_not_hardcoded(tmp_path):
    """Patch the source file; the rendered number must move with it.

    Resolving a pointer proves the path exists. This proves the module READS
    it: if any figure were written into the code, this test would not notice a
    changed result file, which is exactly the failure mode M4 guards against.
    """
    for name in SOURCES:
        shutil.copy(RESULTS / name, tmp_path / name)
    sealed = json.loads((tmp_path / "phase15_sealed.json").read_text(
        encoding="utf-8"))
    before = sealed["pooled"]["detected"]["knee_from_segments"]["near"]["mae"]
    sealed["pooled"]["detected"]["knee_from_segments"]["near"]["mae"] = (
        before + 1.0)
    (tmp_path / "phase15_sealed.json").write_text(json.dumps(sealed),
                                                  encoding="utf-8")

    patched = load_budget(tmp_path)
    cell = patched.cell("B", "knee_from_segments", "near")
    assert cell.value == pytest.approx(before + 1.0, abs=1e-9)


def test_a_missing_source_refuses_rather_than_remembering(tmp_path):
    with pytest.raises(MissingSource):
        load_budget(tmp_path)


# ------------------------------------------------- the rows themselves ----

def test_all_three_rows_are_present(budget):
    assert len(budget.row("A")) == len(QUANTITIES) * 2
    assert len(budget.row("B")) == len(QUANTITIES) * 2
    assert len(budget.row("C")) == len(QUANTITIES) * 2 * 3   # 50, 33.3, 30 fps


def test_row_a_is_labelled_the_best_case_in_domain(budget):
    assert "best case" in ROW_LABELS["A"]
    assert "in-domain" in ROW_LABELS["A"]
    for cell in budget.row("A"):
        assert cell.fps == 100.0
        assert "fine-tuned" in cell.conditions


def test_row_b_is_the_sealed_test_at_30_fps(budget):
    for cell in budget.row("B"):
        assert cell.fps == ROW_B_FPS
        assert "sealed" in cell.source_file or "never opened" in cell.conditions
        assert cell.ci is not None and cell.ci[0] < cell.value < cell.ci[1]
        assert cell.bias is not None


def test_row_c_says_it_is_tune_set_context(budget):
    assert "tune-set" in ROW_LABELS["C"]
    for cell in budget.row("C"):
        assert cell.kind == "penalty"
        assert "IN-DOMAIN on the tune set" in cell.conditions


def test_no_hip_anywhere_in_the_budget(budget):
    """`[A4]` its error is a learned convention, so it is absent, not labelled."""
    assert "trunk_referenced_hip" not in QUANTITIES
    for cell in budget.cells:
        assert "hip" not in cell.quantity
        assert "hip" not in cell.label.lower()
        assert "hip" not in cell.source_path


def test_the_quantities_are_the_two_primaries_plus_the_experimental():
    assert PRIMARY == ("knee_from_segments", "thigh")
    assert EXPERIMENTAL == ("trunk",)
    assert set(QUANTITIES) == set(PRIMARY) | set(EXPERIMENTAL)


# ------------------------------------- the near/far columns from the flag --

@pytest.mark.parametrize("near_side,expect", [("l", {"l": "near", "r": "far"}),
                                              ("r", {"l": "far", "r": "near"})])
def test_each_leg_gets_its_own_column_from_the_declared_side(budget, near_side,
                                                             expect):
    for side, want in expect.items():
        q = budget.expected("thigh", side, near_side, ROW_B_FPS)
        assert q.limb == want
        assert q.cell is budget.cell("B", "thigh", want)


def test_the_flag_buys_back_the_measured_thigh_gap(budget):
    """The whole reason --near-side is required: a 36% difference on the thigh."""
    near = budget.cell("B", "thigh", "near").value
    far = budget.cell("B", "thigh", "far").value
    assert far > near
    assert (far - near) / near > 0.30


# ----------------------------------------- `[A2]` B and C never combined --

def test_a_60_fps_clip_gets_row_b_exactly(budget):
    """Not adjusted, not interpolated: the identical float, plus words."""
    for side, limb in (("l", "near"), ("r", "far")):
        q = budget.expected("knee_from_segments", side, "l", 60.0)
        assert q.cell.value == budget.cell("B", "knee_from_segments",
                                           limb).value
        assert not budget.is_combination(q.cell.value)
        assert any(UNMEASURED in n for n in q.unmeasured)


def test_the_60_fps_note_says_the_number_is_not_adjusted(budget):
    q = budget.expected("knee_from_segments", "l", "l", 60.0)
    note = " ".join(q.unmeasured)
    assert "NOT adjusted downward" in note
    assert "not a correction you can apply" in note


def test_a_30_fps_clip_carries_no_unmeasured_note(budget):
    q = budget.expected("knee_from_segments", "l", "l", 30.0)
    assert q.unmeasured == []
    assert ROW_LABELS["B"] in q.notes


def test_30_009_fps_is_still_30_fps(budget):
    """MoVi's own iPhones run at 30.009; that is not a different condition."""
    assert budget.expected("thigh", "l", "l", 30.009).unmeasured == []


def test_a_slower_clip_is_not_adjusted_upward_either(budget):
    q = budget.expected("knee_from_segments", "l", "l", 27.0)
    assert q.cell.value == budget.cell("B", "knee_from_segments", "near").value
    assert any("NOT adjusted upward" in n for n in q.unmeasured)


def test_is_combination_catches_the_forbidden_arithmetic(budget):
    b = budget.cell("B", "knee_from_segments", "near").value
    c = budget.cell("C", "knee_from_segments", "near", fps=30.0).value
    assert budget.is_combination(b + c)
    assert budget.is_combination(b - c)
    assert budget.is_combination(b + 0.5 * c)
    assert not budget.is_combination(b)
    assert not budget.is_combination(b + 100.0)


def test_no_quote_the_report_can_make_is_a_combination(budget):
    """Mechanical enforcement over every number the report can display."""
    for fps in (25.0, 30.0, 30.009, 50.0, 60.0, 120.0):
        for q in budget.quotes(near_side="l", fps=fps):
            assert q.cell is not None
            assert not budget.is_combination(q.cell.value), (
                f"{q.quantity} at {fps} fps displays an arithmetic "
                f"combination of Row B and Row C")


# -------------------------------------------- the unmeasured conditions ---

def test_the_unmeasured_table_is_stated_verbatim():
    text = " ".join(f"{k}: {v}" for k, v in NOT_MEASURED)
    assert text.count(UNMEASURED) >= 4
    assert "60 fps" in text
    assert "phone" in text
    assert "treadmill" in text
    assert "Row C must not be used to adjust it" in text


def test_row_b_carries_both_measured_caveats():
    text = " ".join(ROW_B_CAVEATS)
    assert "per-subject offset" in text
    assert "NOT a removable one" in text
    assert "17%" in text and "11%" in text
    assert "SHAPE is tracked well" in text


def test_a_quote_without_a_cell_renders_unmeasured(budget):
    q = budget.expected("shank", "l", "l", 30.0)      # not a reported quantity
    assert q.cell is None
    assert q.text == UNMEASURED
    assert q.ci_text == UNMEASURED
    assert any(UNMEASURED in u or "no Row B cell" in u for u in q.unmeasured)


def test_every_quote_has_bit_identical_row_b_provenance(budget):
    """`[A2]` the primary form of the rule: provenance, not arithmetic.

    A displayed expected error must BE a Row B float, not merely sit near one.
    Nothing added to, scaled or interpolated from Row B can satisfy this.
    """
    for fps in (25.0, 30.0, 30.009, 50.0, 60.0, 120.0):
        for q in budget.quotes(near_side="r", fps=fps):
            src = budget.quoted_from(q.cell.value)
            assert src is not None, f"{q.quantity} at {fps} fps is not a Row B figure"
            assert src.quantity == q.quantity and src.limb == q.limb


def test_quoted_from_rejects_a_nudged_value(budget):
    cell = budget.cell("B", "knee_from_segments", "near")
    assert budget.quoted_from(cell.value) is cell
    assert budget.quoted_from(cell.value + 1e-9) is None
