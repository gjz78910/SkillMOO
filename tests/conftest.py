from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS = [REPO_ROOT / "src"]
for path in PATHS:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
