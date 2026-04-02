from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "inspection_system" / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
