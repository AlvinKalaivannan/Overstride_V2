"""M6: the vocabulary and output guard. Runs in CI and fails the build.

`docs/phase16-demo-design.md` section 2 and milestone M6, enforcing
`CLAUDE.md`'s measurement-track rule: the demo has **no output of any kind**
about health, and the words that would imply one do not appear in its code,
its strings or its reports.

## What is banned, and why each one

| term | why |
|---|---|
| `risk` | `CLAUDE.md`: the word does not appear in any output, UI, docstring or plot of this track |
| `predict` | the data supports screening at a point in time, never prediction over time |
| `injur*` | phases 4B and 5D settled that a per-user verdict is unsupportable |
| `diagnos*` | a joint-angle measurement is not a clinical finding |
| `abnormal` | there is no normative comparison; AddBiomechanics ranges are plausibility bounds inside the code |
| `asymmetry index` | left-right differences are shown as two numbers, never reduced to one |
| `normal range` | same reason |

## The `# vocab-ok:` pragma, and why HTML does not get one

A Python line may carry `# vocab-ok: <reason>` to allow a banned word where it
is genuinely an engineering term rather than a claim -- "internal diagnostic
output", say. The reason is **required**, so each exception is a decision
someone wrote down rather than a silenced warning.

**HTML gets no pragma.** A report is the thing a reader sees, so the ban there
is absolute: no exceptions, no reasons, no escape hatch.

## `[A4]` The hip check

`report.html` must contain **no hip trace, no hip number and no hip axis** --
not collapsed, not greyed, not behind a toggle. Phase 15F showed an
H36M-trained lifter that has never seen running, this lab or this camera beats
this pipeline on that joint on held-out footage while being 5x worse on the
tune set, so its error is a learned convention rather than a measurement
limit, and a number whose error is a convention is not a measurement. The word
is matched on a word boundary, so "ship" and "relationship" are safe.

Run: .venv/Scripts/python.exe scripts/check_vocabulary.py [extra.html ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BANNED = {
    "risk": r"\brisks?\b",
    "predict": r"\bpredict\w*\b",
    "injury": r"\binjur\w*\b",
    "diagnosis": r"\bdiagnos\w*\b",
    "abnormal": r"\babnormal\w*\b",
    "asymmetry index": r"asymmetry[\s\-]+index",
    "normal range": r"normal[\s\-]+ranges?\b",
}
HIP = r"\bhips?\b"
PRAGMA = re.compile(r"#\s*vocab-ok:\s*(\S.*)$")

# The demo's surface. Everything a user can run, read or be shown.
PY_SURFACE = ("overstride/*.py", "scripts/overstride_analyze.py",
              "scripts/synthetic_clip.py")
HTML_SURFACE = ("figures/*.html",)

# The debug file is explicitly internal, never produced by a normal run and
# never published, so it is not part of the surface. A normal run producing one
# is itself a failure, and `check_outputs` looks for exactly that.
DEBUG_SUFFIX = "-debug.html"


class Hit:
    def __init__(self, path: Path, line_no: int, term: str, line: str,
                 why: str):
        self.path, self.line_no = path, line_no
        self.term, self.line, self.why = term, line.strip(), why

    def __str__(self) -> str:
        try:
            shown = self.path.resolve().relative_to(REPO)
        except ValueError:
            shown = self.path
        return (f"{shown}:{self.line_no}: {self.why} [{self.term}]\n"
                f"    {self.line[:110]}")


def scan_text(text: str, path: Path, allow_pragma: bool = True,
              terms: dict | None = None) -> list:
    """Every banned term, line by line, honouring the pragma if allowed."""
    hits = []
    for n, line in enumerate(text.splitlines(), start=1):
        pragma = PRAGMA.search(line) if allow_pragma else None
        for term, pattern in (terms or BANNED).items():
            if not re.search(pattern, line, re.IGNORECASE):
                continue
            if pragma:
                continue
            hits.append(Hit(path, n, term,
                            line.replace(term, term.upper()),
                            "banned word with no `# vocab-ok:` reason"
                            if allow_pragma else "banned word in output"))
    return hits


def check_pragmas(text: str, path: Path) -> list:
    """A pragma must give a reason, and must sit on a line that needs one."""
    out = []
    for n, line in enumerate(text.splitlines(), start=1):
        m = re.search(r"#\s*vocab-ok:?(.*)$", line)
        if not m:
            continue
        reason = m.group(1).strip().lstrip(":").strip()
        if not reason:
            out.append(Hit(path, n, "vocab-ok", line,
                           "`# vocab-ok:` with no reason"))
            continue
        bare = line[:m.start()]
        if not any(re.search(p, bare, re.IGNORECASE)
                   for p in BANNED.values()):
            out.append(Hit(path, n, "vocab-ok", line,
                           "`# vocab-ok:` on a line that needs no exception"))
    return out


def check_html(path: Path) -> list:
    """A report: no banned word, no hip, no script, no pragma escape."""
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = scan_text(text, path, allow_pragma=False)
    hits += scan_text(text, path, allow_pragma=False, terms={"hip": HIP})
    for n, line in enumerate(text.splitlines(), start=1):
        if "<script" in line.lower():
            hits.append(Hit(path, n, "script", line,
                            "the report must read with JavaScript disabled"))
    return hits


def check_outputs(report: Path) -> list:
    """A normal run must not have produced a debug file beside the report."""
    dbg = report.with_name(report.stem + DEBUG_SUFFIX)
    if dbg.exists():
        return [Hit(dbg, 0, "debug", str(dbg.name),
                    "a debug file sits beside a published report; "
                    "`--debug` output is internal and never published")]
    return []


def surface(extra=()) -> tuple[list, list]:
    py = sorted({p for pat in PY_SURFACE for p in REPO.glob(pat)})
    htm = sorted({p for pat in HTML_SURFACE for p in REPO.glob(pat)}
                 | {Path(p) for p in extra})
    htm = [p for p in htm if not p.name.endswith(DEBUG_SUFFIX)]
    return py, htm


def main(argv=None) -> int:
    extra = list(argv if argv is not None else sys.argv[1:])
    py, htm = surface(extra)
    hits: list = []
    for p in py:
        text = p.read_text(encoding="utf-8", errors="replace")
        hits += scan_text(text, p) + check_pragmas(text, p)
    for p in htm:
        hits += check_html(p) + check_outputs(p)

    print(f"vocabulary guard: {len(py)} source file(s), "
          f"{len(htm)} report(s)")
    for p in py + htm:
        try:
            print(f"  checked {p.resolve().relative_to(REPO)}")
        except ValueError:
            print(f"  checked {p}")
    if not hits:
        print("\nclean: no banned vocabulary, no hip trace in any report")
        return 0
    print(f"\nFAILED: {len(hits)} problem(s)\n")
    for h in hits:
        print(h)
    print("\nThe demo makes no claim about health. If a word here is an "
          "engineering term rather than a claim, add `# vocab-ok: <reason>` "
          "to that line -- reports get no such exemption.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
