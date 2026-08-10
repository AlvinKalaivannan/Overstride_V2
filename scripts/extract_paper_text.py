"""Dump the dataset paper to plain text so Table 1 can be transcribed exactly.

Read-only helper for phase 0 step B. Writes to the scratchpad, not the repo.
"""

import sys
from pathlib import Path

from pypdf import PdfReader

PDF = Path(__file__).resolve().parents[1] / "docs" / "s41597-024-04011-7.pdf"


def main() -> int:
    if not PDF.exists():
        print(f"MISSING: {PDF}", file=sys.stderr)
        return 1

    reader = PdfReader(PDF)
    print(f"# pages: {len(reader.pages)}")
    for i, page in enumerate(reader.pages, start=1):
        print(f"\n{'=' * 70}\n=== PAGE {i}\n{'=' * 70}")
        print(page.extract_text() or "(no extractable text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
