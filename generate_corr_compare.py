#!/usr/bin/env python3
"""Compatibility shim.

Figure 5 scripts are organized under `figure_scripts/figure5/`.
This module re-exports the public helpers so older imports keep working,
and delegates CLI execution to the new location.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_scripts.figure5.generate_corr_compare import (  # noqa: E402
    compute_corr_effect_size,
    main,
    plot_corr_effect_size,
)

__all__ = ["compute_corr_effect_size", "plot_corr_effect_size", "main"]

if __name__ == "__main__":
    main()
