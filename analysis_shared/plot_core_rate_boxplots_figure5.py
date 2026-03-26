#!/usr/bin/env python3
"""
Backward-compatibility wrapper.

The actual entrypoint has moved to figure_scripts/figure5/plot_core_rate_boxplots_figure5.py.
This wrapper invokes the new location so old commands still work.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEW_SCRIPT = ROOT / "figure_scripts" / "figure5" / "plot_core_rate_boxplots_figure5.py"

if not NEW_SCRIPT.exists():
    raise FileNotFoundError(
        f"Could not find the main script at {NEW_SCRIPT}. "
        "It may have been moved or renamed."
    )

# Load and execute the new script in this namespace
with open(NEW_SCRIPT, encoding="utf-8") as f:
    code = compile(f.read(), str(NEW_SCRIPT), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(NEW_SCRIPT)})
