import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))          # so `import overstride` works
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "data" / "athleticspose" / "repo"))

DERIVED = REPO / "data" / "derived"
CKPT = REPO / "data" / "athleticspose" / "checkpoints"
DATASET = REPO / "data" / "athleticspose" / "data" / "AthleticsPoseDataset"


# --------------------------------------------------------------------------
# The AthleticsPose package is vendored at `data/athleticspose/repo`, which is
# gitignored, so it is absent from any clean checkout -- including CI. The
# frozen pipeline scripts import it at module scope and are pinned by content
# hash for the sealed test's provenance, so they cannot be made lazy.
#
# Only THREE test modules are left that cannot even be COLLECTED without
# it: two import a frozen blob directly, and `phase12_scan` is built on
# the package's own marker tables rather than merely importing past them. A collection error fails the whole
# run (pytest exit code 2) rather than skipping, so they are skipped here.
#
# Everything else now imports with nothing but numpy. The unfrozen scripts
# -- phase11_viewpoint, phase12_scan, phase14_confidence -- import their
# heavy dependencies inside the functions that use them, and so do
# `overstride.analyze` and `overstride.checks`. That is what keeps this list
# to three entries instead of eight.
import importlib.util  # noqa: E402

NEEDS_ATHLETICSPOSE = (
    "test_scan.py",             # phase12_scan is built ON the package
    "test_video_crop.py",       # imports video_kinematics, a FROZEN blob
    "test_video_pipeline.py",   # imports phase7_inference_fix, a FROZEN blob
)

if importlib.util.find_spec("athleticspose") is None:
    collect_ignore = list(NEEDS_ATHLETICSPOSE)
