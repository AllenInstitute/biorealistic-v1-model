#!/usr/bin/env python3
"""Compatibility shim for Figure 5 correlation sim-vs-EM panel plot.

The maintained script lives in `figure_scripts/figure5/generate_corr_sim_em_panels.py`.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "figure_scripts" / "figure5" / "generate_corr_sim_em_panels.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
