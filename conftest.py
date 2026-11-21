"""Root conftest: ensure the repo root is importable by pytest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))