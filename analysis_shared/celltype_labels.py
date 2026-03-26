"""Shared utilities for naming / labeling cell types in plots.

Important: These helpers are *display-only* transformations. They should not be used
to change analysis grouping logic or to map between naming schemes in data files.
For mapping between network naming variants, refer to `base_props/cell_type_naming_scheme.csv`
and higher-level utilities such as `network_utils.py`.
"""

from __future__ import annotations


def abbrev_cell_type(cell_type: str, *, l1_label: str = "I1") -> str:
    """Return an abbreviated label for a cell type string (for plots).

    Conventions (default):
    - Exc: L2/3_Exc->E23, L4_Exc->E4, L6_Exc->E6
    - L5 split: L5_IT->E5IT, L5_ET->E5ET, L5_NP->E5NP
    - Simplified inh: PV/SST/VIP unchanged
    - Full inh (layered): L2/3_PV->PV23, L4_SST->SST4, etc.
    - L1 inhibitory: L1_Inh->I1 (configurable via l1_label)
    """
    if not isinstance(cell_type, str):
        return str(cell_type)

    exc_map = {
        "L2/3_Exc": "E23",
        "L4_Exc": "E4",
        "L5_Exc": "E5",
        "L6_Exc": "E6",
        "L5_IT": "E5IT",
        "L5_ET": "E5ET",
        "L5_NP": "E5NP",
    }
    if cell_type in exc_map:
        return exc_map[cell_type]

    # Inhibitory (already simplified)
    if cell_type == "L1_Inh":
        return l1_label
    if cell_type in {"PV", "SST", "VIP"}:
        return cell_type

    # Inhibitory (full 19): L2/3_PV -> PV23, etc.
    if "_" in cell_type:
        layer, subtype = cell_type.split("_", 1)
        layer = layer.strip()
        subtype = subtype.strip()
        if layer.startswith("L"):
            layer = layer[1:]
        if layer == "2/3":
            layer = "23"
        if subtype in {"PV", "SST", "VIP"}:
            return f"{subtype}{layer}"

    return cell_type


def abbrev_cell_types(cell_types: list[str], *, l1_label: str = "I1") -> list[str]:
    """Vectorized wrapper around `abbrev_cell_type` for convenience."""
    return [abbrev_cell_type(ct, l1_label=l1_label) for ct in cell_types]
