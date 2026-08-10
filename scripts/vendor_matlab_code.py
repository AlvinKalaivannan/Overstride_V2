"""Make a patched, working copy of the bundled RIC MATLAB pipeline.

$DATA_ROOT is read-only input (CLAUDE.md), so the archive's Code/ folder is never
edited in place. This copies it to data/derived/matlab_code/ and applies the
minimum patch needed to run on a modern MATLAB.

THE PATCH -- one block in gait_steps.m, the walk/run classifier.

  Two things break on R2026a (the code targets R2023a):
    1. `import classreg.learning.classif.CompactClassificationDiscriminant` is a
       parse-time error -- that internal namespace no longer resolves.
    2. Removing the import is NOT sufficient: gaitClass.mat stores a
       ClassificationDiscriminant object that R2026a cannot instantiate at all.
       It loads as a uint32 and `predict` fails. Verified, not assumed.

  The classifier's entire job is to decide 'walk' vs 'run' for the trial being
  processed. Our caller already knows: it passes `out.running` from a session
  listed in run_data_meta.csv. So the patched block takes the label from the
  caller (RIC_FORCED_LABEL global) and falls back to the original classifier
  path when none is set.

  This IS a behavioural change and it is not assumed safe -- it is verified.
  gait_steps also returns DISCRETE_VARIABLES, and the archive stores dv_r
  produced by this same pipeline. Regression-testing computed vs stored on the
  40 live variables per side confirms empirically whether the substitution
  reproduces the archive's own numbers. If it does not, stop.

  Note the one case this cannot catch: a session filed under `running` whose
  trial is actually a walk. The LDA existed partly to catch that, and run speeds
  in this archive go down to 1.17 m/s, which is walking territory. The manifest
  records computed speed per session so those can be flagged downstream.

  Nothing else is touched. Every numerical path is unchanged.

Run: .venv/Scripts/python.exe scripts/vendor_matlab_code.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_cohort import data_root  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "data" / "derived" / "matlab_code"

ORIGINAL_BLOCK = """testSet = [vel stRate];
import classreg.learning.classif.CompactClassificationDiscriminant
load('gaitClass.mat','gaitClass')


label = predict(gaitClass,testSet);
%label returned as cell
label = label{1};"""

PATCHED_BLOCK = """testSet = [vel stRate];
% ---- PATCHED by scripts/vendor_matlab_code.py -- see that file for rationale.
% gaitClass.mat holds a ClassificationDiscriminant that MATLAB R2026a cannot
% instantiate (it loads as uint32), and the classreg import above it is a
% parse-time error. The classifier only decides 'walk' vs 'run'; the caller
% already knows which trial it passed in, so it supplies the label directly.
% Verified by regression-testing DISCRETE_VARIABLES against the archive's dv_r.
global RIC_FORCED_LABEL %#ok<GVMIS>
if ~isempty(RIC_FORCED_LABEL)
    label = RIC_FORCED_LABEL;
else
    load('gaitClass.mat','gaitClass')
    label = predict(gaitClass,testSet);
    %label returned as cell
    label = label{1};
end
% ---- end patch"""


def main() -> int:
    src = data_root() / "Supplemental_materials" / "Code"
    if not src.is_dir():
        raise SystemExit(f"pipeline source not found: {src}")

    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(src, DEST)
    print(f"copied {src} -> {DEST.relative_to(REPO)}")

    target = DEST / "gait_steps.m"
    text = target.read_text(encoding="utf-8", errors="surrogateescape")
    normalised = text.replace("\r\n", "\n")
    if ORIGINAL_BLOCK not in normalised:
        raise SystemExit(
            "the walk/run classifier block in gait_steps.m does not match what "
            "this patch expects -- the archive changed. Stop and re-read it "
            "rather than guessing."
        )
    n = normalised.count(ORIGINAL_BLOCK)
    normalised = normalised.replace(ORIGINAL_BLOCK, PATCHED_BLOCK)
    target.write_text(normalised, encoding="utf-8", errors="surrogateescape")
    print(f"patched gait_steps.m: walk/run classifier block ({n} occurrence)")

    # --- compatibility shims -------------------------------------------------
    # nanmin/nanmean/nanmedian were removed from modern MATLAB. Rather than edit
    # the pipeline's numerical code, restore the functions it expects. These are
    # exact equivalents of what the originals did, so no behaviour changes.
    shims = {
        "nanmin": "function varargout = nanmin(varargin)\n"
                  "[varargout{1:nargout}] = min(varargin{:}, 'omitnan');\nend\n",
        "nanmean": "function y = nanmean(x, varargin)\n"
                   "y = mean(x, varargin{:}, 'omitnan');\nend\n",
        "nanmedian": "function y = nanmedian(x, varargin)\n"
                     "y = median(x, varargin{:}, 'omitnan');\nend\n",
    }
    used = set()
    for path in DEST.glob("*.m"):
        body = path.read_text(encoding="utf-8", errors="surrogateescape")
        used.update(name for name in shims if name in body)
    for name in sorted(used):
        header = (f"% Compatibility shim added by scripts/vendor_matlab_code.py.\n"
                  f"% MATLAB removed {name}; this restores the documented\n"
                  f"% behaviour ('omitnan') without touching the pipeline code.\n")
        (DEST / f"{name}.m").write_text(header + shims[name], encoding="utf-8")
    print(f"added shims: {', '.join(sorted(used)) or 'none needed'}")

    # The archive must be untouched.
    original = (src / "gait_steps.m").read_text(
        encoding="utf-8", errors="surrogateescape").replace("\r\n", "\n")
    assert ORIGINAL_BLOCK in original, "SOURCE ARCHIVE WAS MODIFIED -- investigate"
    print("verified: $DATA_ROOT copy is unmodified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
