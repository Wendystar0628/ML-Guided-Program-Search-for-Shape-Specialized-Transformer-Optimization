"""Launch the shared verifier for the NVIDIA GeForce RTX 4080 bundle."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    from runner.verified_hardware import main_for_bundle

    raise SystemExit(main_for_bundle(Path(__file__).resolve().parent))
