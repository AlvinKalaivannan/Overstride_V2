"""M4: the error budget as data, not prose.

`docs/phase16-demo-design.md` section 3. Three rows, every one of them measured,
each cell naming its source file, the exact path inside it, and the conditions
it was measured under.

**No number in this module is written down.** Each cell carries a JSON pointer
and reads its value out of the committed result file at load time, so a figure
cannot drift away from the measurement that produced it: change the result file
and the report changes, delete the result file and the report refuses to claim
a number. `tests/test_budget.py` asserts that every rendered value resolves to
its cited path, and proves the values are not hardcoded by loading a patched
copy of the JSON and watching the output move.

## The three rows

* **Row A -- BEST CASE: in-domain lab data at 100 Hz.** `[A3]` The ceiling the
  method reaches on held-out subjects of the dataset the checkpoint was
  fine-tuned on. It exists so the in-domain/cross-lab gap is visible, and for
  no other purpose. **Never shown as the expected error for a user clip.**
* **Row B -- held-out lab footage at 30 fps. The honest headline.** The sealed
  test: 5 subjects never opened, pipeline pinned by content hash, run once.
  **This is the expected error on new footage.**
* **Row C -- frame-rate penalties, measured on the tune set.** Shown in its own
  panel as context for the filming instructions.

## `[A2]` Rows B and C are never combined

Not added, not subtracted, not scaled, not interpolated. Row C is measured
**in-domain on the tune set** against a 100 Hz native reference; Row B is a
**different lab**. Combining them manufactures a number for a condition nobody
measured -- the move phase 15E labelled "if it transferred unchanged", whose
arithmetic produced three undefined cells out of eight. `is_combination()`
makes the rule checkable, and the renderer runs it over every number it is
about to display.

## `[A4]` The hip is not here

`trunk_referenced_hip` has no cell in any row. Phase 15F showed an H36M-trained
lifter that has never seen running, this lab or this camera **beats** this
pipeline on the hip while being 5x worse on the tune set, so the hip error is a
learned convention rather than a measurement limit. A number whose error is a
convention is not a measurement, and it is absent rather than labelled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

UNMEASURED = "UNMEASURED"

# The quantities the demo reports, and how they are named on screen. The hip is
# deliberately absent -- see the module docstring.
QUANTITIES = ("knee_from_segments", "thigh", "trunk")
LABELS = {"knee_from_segments": "knee flexion",
          "thigh": "thigh inclination (against gravity)",
          "trunk": "trunk lean"}
PRIMARY = ("knee_from_segments", "thigh")
EXPERIMENTAL = ("trunk",)

# Row B is measured at this rate, and it is the only rate held-out footage
# exists at. MoVi PG1 is 30.000 fps.
ROW_B_FPS = 30.0
FPS_TOLERANCE = 0.5          # 30.009 (MoVi's iPhones) is still "30 fps"

# Row C's measured rates, each with the file and the path its penalties live at.
ROW_C_RATES = {
    50.0: ("phase15f.json", "q1_50fps/penalty/{q}/{limb}/penalty",
           "q1_50fps/penalty/{q}/{limb}/ci95"),
    33.3: ("phase15e.json",
           "penalty_sensitivity_33fps_coco/{q}/{limb}/penalty",
           "penalty_sensitivity_33fps_coco/{q}/{limb}/ci95"),
    30.0: ("phase15e.json", "penalty_paired/coco_det/{q}/{limb}/penalty",
           "penalty_paired/coco_det/{q}/{limb}/ci95"),
}

ROW_A_CONDITIONS = (
    "a calibrated 8-camera lab, athletes, 100 Hz, and the dataset this "
    "checkpoint was fine-tuned on -- in-domain performance on held-out "
    "subjects of the TRAINING dataset")
ROW_B_CONDITIONS = (
    "MoVi PG1, 800x600, 30.000 fps, side-on, jogging; 5 subjects never opened "
    "before the run, pipeline pinned by content hash, pre-registration pushed "
    "before a single byte was downloaded, run once")
ROW_C_CONDITIONS = (
    "AthleticsPose, generic COCO detections, paired per subject against a "
    "native 100 Hz reference. IN-DOMAIN on the tune set, not on new footage")

ROW_LABELS = {
    "A": "best case: in-domain lab data at 100 Hz",
    "B": "expected error: measured on held-out lab footage at 30 fps",
    "C": "why we ask for 60 fps: tune-set context",
}


class MissingSource(Exception):
    """A cited result file or path is absent. The demo quotes no number."""


def resolve(doc, path: str):
    """Walk a '/'-separated pointer into a parsed JSON document.

    Result-file keys contain '|' (phase 15E's arms) but never '/', so '/' is a
    safe separator.
    """
    node = doc
    for part in path.split("/"):
        if not isinstance(node, dict) or part not in node:
            raise MissingSource(f"{path}: no key {part!r}")
        node = node[part]
    return node


@dataclass(frozen=True)
class Cell:
    """One measured figure, with everything needed to audit it."""

    row: str                     # "A" | "B" | "C"
    quantity: str
    limb: str                    # "near" | "far"
    value: float
    ci: tuple | None
    source_file: str
    source_path: str
    conditions: str
    fps: float | None = None     # Row C only: the rate the penalty is for
    bias: float | None = None
    kind: str = "mae"            # "mae" | "penalty"

    @property
    def label(self) -> str:
        return LABELS[self.quantity]

    @property
    def is_experimental(self) -> bool:
        return self.quantity in EXPERIMENTAL


@dataclass
class Quote:
    """What the report displays for one quantity on one leg."""

    quantity: str
    side: str                    # "l" | "r"
    limb: str                    # "near" | "far" -- from --near-side
    cell: Cell | None
    notes: list = field(default_factory=list)
    unmeasured: list = field(default_factory=list)

    @property
    def text(self) -> str:
        if self.cell is None:
            return UNMEASURED
        return f"{self.cell.value:.2f}°"

    @property
    def ci_text(self) -> str:
        if self.cell is None or self.cell.ci is None:
            return UNMEASURED
        return f"[{self.cell.ci[0]:.2f}, {self.cell.ci[1]:.2f}]"


@dataclass
class Budget:
    """Rows A, B and C, loaded from the committed result files."""

    cells: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)

    def row(self, row: str) -> list:
        return [c for c in self.cells if c.row == row]

    def cell(self, row: str, quantity: str, limb: str,
             fps: float | None = None) -> Cell | None:
        for c in self.cells:
            if (c.row == row and c.quantity == quantity and c.limb == limb
                    and (fps is None or c.fps == fps)):
                return c
        return None

    # ---------------------------------------------------- the display -----

    def expected(self, quantity: str, side: str, near_side: str,
                 fps: float) -> Quote:
        """`[A2]` Row B for the leg, unadjusted, whatever the clip's rate.

        The near leg gets Row B's near column and the far leg its far column --
        the precision the required `--near-side` flag buys back. The clip's
        frame rate changes the WORDS, never the number.
        """
        limb = "near" if side == near_side else "far"
        cell = self.cell("B", quantity, limb)
        q = Quote(quantity=quantity, side=side, limb=limb, cell=cell)
        if cell is None:
            q.unmeasured.append(
                f"no Row B cell for {LABELS.get(quantity, quantity)} "
                f"({limb} limb)")
            return q
        q.notes.append(ROW_LABELS["B"])
        if fps > ROW_B_FPS + FPS_TOLERANCE:
            q.unmeasured.append(
                f"your clip is faster than the footage this was measured on "
                f"({fps:.3g} fps against {ROW_B_FPS:.0f}); the effect of that "
                f"is {UNMEASURED} on new footage, and the number above is NOT "
                f"adjusted downward. The tune-set frame-rate penalties below "
                f"are not a correction you can apply to it.")
        elif fps < ROW_B_FPS - FPS_TOLERANCE:
            q.unmeasured.append(
                f"your clip is slower than the footage this was measured on "
                f"({fps:.3g} fps against {ROW_B_FPS:.0f}); 30.0 fps is the "
                f"lowest rate ever measured for this pipeline, so the extra "
                f"cost is {UNMEASURED} and the number above is NOT adjusted "
                f"upward.")
        return q

    def quotes(self, near_side: str, fps: float,
               quantities=QUANTITIES) -> list:
        """One quote per quantity per leg, in report order."""
        return [self.expected(q, side, near_side, fps)
                for q in quantities for side in ("l", "r")]

    # ------------------------------------------- the no-combination rule --

    def penalty_values(self) -> list:
        return [c.value for c in self.row("C")]

    def quoted_from(self, value: float) -> Cell | None:
        """The Row B cell a displayed number came from, bit-identically.

        **Provenance, not arithmetic.** This is the primary form of the
        no-combination rule: a displayed expected error must be the exact float
        of one Row B cell. Nothing that has been added to, scaled or
        interpolated can satisfy it.
        """
        for c in self.row("B"):
            if value == c.value:
                return c
        return None

    def is_combination(self, value: float, tol: float = 5e-3) -> bool:
        """`[A2]` Is `value` a Row B figure arithmetically moved by a Row C one?

        The secondary, numeric form of the rule: it catches a value that sits
        on a Row B +/- Row C sum. It is checked AFTER provenance, because with
        6 Row B values, 18 Row C penalties and 6 multipliers there are 648
        candidate sums and one of them lands within 5e-3 of the legitimate
        thigh figure by coincidence -- a birthday-paradox false positive, not a
        combination. An exact Row B value is therefore never a combination by
        definition, and anything else near a B +/- C sum is.
        """
        if self.quoted_from(value) is not None:
            return False
        for b in self.row("B"):
            for p in self.penalty_values():
                for k in (1.0, -1.0, 0.5, -0.5, 0.1, -0.1):
                    if abs(value - (b.value + k * p)) <= tol and abs(p) > tol:
                        return True
        return False


# ------------------------------------------------------------ loading ------

def load_budget(results: Path | str = RESULTS) -> Budget:
    """Read rows A, B and C out of the committed result files.

    Raises `MissingSource` if a cited file or path is gone -- the demo shows
    `UNMEASURED` rather than a remembered number.
    """
    results = Path(results)
    docs: dict[str, dict] = {}

    def doc(name: str) -> dict:
        if name not in docs:
            p = results / name
            if not p.exists():
                raise MissingSource(f"{name} is not in {results}")
            docs[name] = json.loads(p.read_text(encoding="utf-8"))
        return docs[name]

    # Load every cited file up front: a missing file is a broken install, not
    # a missing cell, and it must not be swallowed by the per-cell handler
    # below. (It was, before `tests/test_budget.py` caught it.)
    for name in ("phase15_sealed.json", "phase15e.json", "phase15f.json"):
        doc(name)

    cells: list[Cell] = []

    # --- Row A: phase 15E's native arm, which reproduced phase 15/15D ------
    for q in QUANTITIES:
        for limb in ("near", "far"):
            base = f"pooled/coco_det|native/{q}/{limb}"
            try:
                value = float(resolve(doc("phase15e.json"), f"{base}/mae"))
                ci = resolve(doc("phase15e.json"), f"{base}/ci95")
            except MissingSource:
                continue
            cells.append(Cell(
                row="A", quantity=q, limb=limb, value=value,
                ci=(float(ci[0]), float(ci[1])),
                source_file="results/phase15e.json",
                source_path=f"{base}/mae", conditions=ROW_A_CONDITIONS,
                fps=100.0, kind="mae"))

    # --- Row B: the sealed test -------------------------------------------
    for q in QUANTITIES:
        for limb in ("near", "far"):
            base = f"pooled/detected/{q}/{limb}"
            try:
                value = float(resolve(doc("phase15_sealed.json"),
                                      f"{base}/mae"))
                ci = resolve(doc("phase15_sealed.json"), f"{base}/ci95")
                bias = float(resolve(doc("phase15_sealed.json"),
                                     f"{base}/bias"))
            except MissingSource:
                continue
            cells.append(Cell(
                row="B", quantity=q, limb=limb, value=value,
                ci=(float(ci[0]), float(ci[1])), bias=bias,
                source_file="results/phase15_sealed.json",
                source_path=f"{base}/mae", conditions=ROW_B_CONDITIONS,
                fps=ROW_B_FPS, kind="mae"))

    # --- Row C: the frame-rate penalties ----------------------------------
    for rate, (fname, vpath, cpath) in ROW_C_RATES.items():
        for q in QUANTITIES:
            for limb in ("near", "far"):
                vp, cp = vpath.format(q=q, limb=limb), cpath.format(q=q,
                                                                    limb=limb)
                try:
                    value = float(resolve(doc(fname), vp))
                    ci = resolve(doc(fname), cp)
                except MissingSource:
                    continue
                cells.append(Cell(
                    row="C", quantity=q, limb=limb, value=value,
                    ci=(float(ci[0]), float(ci[1])),
                    source_file=f"results/{fname}", source_path=vp,
                    conditions=ROW_C_CONDITIONS, fps=rate, kind="penalty"))

    return Budget(cells=cells, sources={k: f"results/{k}" for k in docs})


# ------------------------------------- the conditions nobody measured ------
# Prose, printed verbatim. `[A2]` requires these to be stated rather than
# interpolated, so they are strings and not arithmetic.

NOT_MEASURED = (
    ("60 fps, anywhere",
     f"{UNMEASURED}. 100/60 is not an integer, so no clean 60 fps decimation "
     f"exists on the tune set and none was run. The nearest measured rate on "
     f"the tune set is 50 fps, and no 60 fps figure is shown for a user clip."),
    ("any rate above 30 fps on new footage",
     f"{UNMEASURED}. MoVi PG1 is 30.000 fps and no higher-rate held-out "
     f"footage exists for this pipeline. A faster clip gets Row B unchanged "
     f"plus this statement. Row C must not be used to adjust it."),
    ("phone or consumer footage, at any rate",
     f"{UNMEASURED}, and this is the project's largest open gap "
     f"(docs/REVIEW.md 7.5). MoVi's iPhone cameras ship no calibration and "
     f"their sync cannot be verified, so the pipeline refuses to score them. "
     f"No error figure applies to phone footage."),
    ("running faster than jogging, outdoors, on a treadmill, in low light",
     f"{UNMEASURED}."),
)

# What the user must be told alongside Row B. Both halves are measured, and
# the second one is the correction committed on 2026-11-03.
ROW_B_CAVEATS = (
    "The error is largely a per-subject offset -- one sealed subject carried a "
    "+0.56° knee bias and another +12.11° -- but it is NOT a "
    "removable one: within-subject SD is 4.5-9.7°, and even an oracle "
    "per-subject correction would cut MAE by only 17% (knee) and 11% (thigh), "
    "leaving 4.8-7.0°.",
    "The SHAPE is tracked well -- knee r +0.77 to +0.92 -- so the trace's form "
    "is more trustworthy than its absolute level.",
)
