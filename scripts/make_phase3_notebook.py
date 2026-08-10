"""Emit notebooks/phase3_kaggle.ipynb from notebooks/phase3_kaggle.py.

The .py is the source of truth -- it is reviewable in diffs, which .ipynb JSON is
not. This splits it on `# %% CELL n` markers and writes a notebook Kaggle can
import directly (File -> Import Notebook).

Run: .venv/Scripts/python.exe scripts/make_phase3_notebook.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "notebooks" / "phase3_kaggle.py"
DST = REPO / "notebooks" / "phase3_kaggle.ipynb"

INTRO = """## Phase 3 — monocular 3D kinematics error

Measures sagittal hip/knee angle error from the released AthleticsPose
checkpoints, in the form phase 2 needs: **split near vs far limb**, plus the
systematic/random decomposition.

**Before running:** Settings → Accelerator **GPU**, Settings → Internet **ON**.
Upload `scripts/phase3_angles.py` as a Kaggle dataset named `overstride-scripts`.

No Ferber data is used or needed here.
"""


def main() -> int:
    text = SRC.read_text(encoding="utf-8")

    # Drop the module docstring; it becomes the notebook's intro markdown.
    body = re.sub(r'^""".*?"""\n', "", text, count=1, flags=re.S)

    parts = re.split(r"^# %% (CELL [^\n]*)\n", body, flags=re.M)
    if len(parts) < 3:
        raise SystemExit("no `# %% CELL` markers found -- check the source file")

    cells = [{"cell_type": "markdown", "id": "intro", "metadata": {},
              "source": INTRO.splitlines(keepends=True)}]
    # parts = [preamble, title1, code1, title2, code2, ...]
    for title, code in zip(parts[1::2], parts[2::2]):
        n = len(cells)
        cells.append({"cell_type": "markdown", "id": f"md{n}", "metadata": {},
                      "source": [f"### {title.strip()}"]})
        src = code.strip("\n")
        cells.append({"cell_type": "code", "id": f"code{n}",
                      "execution_count": None,
                      "metadata": {}, "outputs": [],
                      "source": src.splitlines(keepends=True)})

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"},
                       "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_code = sum(c["cell_type"] == "code" for c in cells)
    print(f"wrote {DST.relative_to(REPO)}  ({n_code} code cells, "
          f"{DST.stat().st_size/1e3:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
