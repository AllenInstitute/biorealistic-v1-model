#!/usr/bin/env python3
"""Legacy full correlation final-panel generator (Figure 5).

The canonical implementation historically lived at repo root as `generate_corr_final_panel.py`.
We keep this wrapper so all Figure 5 entrypoints are discoverable under `figure_scripts/figure5/`.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "generate_corr_final_panel.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
