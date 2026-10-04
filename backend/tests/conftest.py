import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "shared", ROOT / "backend"):
    sys.path.insert(0, str(p))
