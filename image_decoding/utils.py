from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


# Path relative to project root where naming scheme csv resides
NAMING_CSV = Path("base_props") / "cell_type_naming_scheme.csv"


def pop_to_cell_type() -> Dict[str, str]:
    """Return dict mapping `pop_name` → 19-way `cell_type`."""
    df = pd.read_csv(NAMING_CSV, delim_whitespace=True)
    return dict(zip(df["pop_name"], df["cell_type"])) 