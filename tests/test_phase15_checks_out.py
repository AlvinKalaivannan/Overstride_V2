"""`--out` must decide where the checks file is written.

This is a regression test for the defect found DURING the sealed test and
recorded as D-S1 in `results/phase15_sealed.md`: `phase15_checks.main` parsed
`--out` and then wrote to the module-level `OUT` anyway, so running the script
on the five sealed subjects silently overwrote the canonical diagnostic-set
`results/phase15_checks.json`. Nothing raised; the file simply had the wrong
contents, and the only reason it was caught was `git status`.

The script was frozen by the sealed-test pre-registration at the time, so the
fix had to wait until that test was committed. These tests pin the fixed
behaviour so it cannot regress.

No MoVi file is read: `check_subject` is stubbed, so this runs without the
dataset on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import phase15_checks as pc  # noqa: E402

STUB = {"subject_id": "stub", "n_markers": 87}


def _stub_subject(path: Path) -> dict:
    return dict(STUB, file=Path(path).name)


def _fake_mat(tmp_path: Path, name: str = "F_v3d_Subject_99.mat") -> Path:
    p = tmp_path / name
    p.write_bytes(b"")          # never parsed; check_subject is stubbed
    return p


def test_out_flag_is_written_and_canonical_is_untouched(tmp_path, monkeypatch):
    """The exact regression: --out is honoured, OUT is left alone."""
    monkeypatch.setattr(pc, "check_subject", _stub_subject)
    canonical = tmp_path / "canonical.json"
    monkeypatch.setattr(pc, "OUT", canonical)

    dest = tmp_path / "results" / "phase15_sealed_checks.json"
    mat = _fake_mat(tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["phase15_checks.py", "--mat", str(mat),
                         "--out", str(dest)])

    assert pc.main() == 0
    assert dest.exists(), "--out path was not written"
    assert not canonical.exists(), (
        "the module-level OUT was written even though --out was given -- this "
        "is the D-S1 defect: it overwrites the canonical checks file")

    d = json.loads(dest.read_text(encoding="utf-8"))
    assert d["subjects"] == [dict(STUB, file="F_v3d_Subject_99.mat")]
    assert d["round"] == "F"


def test_out_creates_missing_parent_directory(tmp_path, monkeypatch):
    """A nested --out must not fail for want of a directory."""
    monkeypatch.setattr(pc, "check_subject", _stub_subject)
    monkeypatch.setattr(pc, "OUT", tmp_path / "canonical.json")
    dest = tmp_path / "a" / "b" / "c" / "checks.json"
    mat = _fake_mat(tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["phase15_checks.py", "--mat", str(mat),
                         "--out", str(dest)])
    assert pc.main() == 0
    assert dest.exists()


def test_default_still_writes_to_the_canonical_path(tmp_path, monkeypatch):
    """Omitting --out must keep writing where every earlier phase expects."""
    monkeypatch.setattr(pc, "check_subject", _stub_subject)
    canonical = tmp_path / "phase15_checks.json"
    # argparse's default is bound at parse time from the module attribute, so
    # patch the attribute the default reads.
    monkeypatch.setattr(pc, "OUT", canonical)
    mat = _fake_mat(tmp_path)
    monkeypatch.setattr(sys, "argv",
                        ["phase15_checks.py", "--mat", str(mat)])
    assert pc.main() == 0
    assert canonical.exists(), "the default output path regressed"


def test_no_mat_files_writes_nothing(tmp_path, monkeypatch):
    """An empty subject list must not truncate whatever is already there."""
    monkeypatch.setattr(pc, "check_subject", _stub_subject)
    dest = tmp_path / "checks.json"
    dest.write_text('{"sentinel": true}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["phase15_checks.py", "--mat", "--out", str(dest)])
    assert pc.main() == 2
    assert json.loads(dest.read_text(encoding="utf-8")) == {"sentinel": True}
