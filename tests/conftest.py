import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "data" / "athleticspose" / "repo"))

DERIVED = REPO / "data" / "derived"
CKPT = REPO / "data" / "athleticspose" / "checkpoints"
