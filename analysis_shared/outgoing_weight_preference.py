#!/usr/bin/env python3
"""Analyze outgoing connectivity preferences for high/low weight cohorts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "core_nll_0"
NODE_SET_DIR = BASE_DIR / "node_sets"
OUTPUT_DIR = BASE_DIR / "figures" / "selectivity_outgoing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_FILE = BASE_DIR / "network" / "v1_v1_edges_bio_trained.h5"
NODE_FILE = BASE_DIR / "network" / "v1_nodes.h5"
NODE_TYPES_FILE = BASE_DIR / "network" / "v1_node_types.csv"

NODE_SETS: Dict[str, str] = {
    "exc_high": "high_outgoing_exc_nodes.json",
    "exc_low": "low_outgoing_exc_nodes.json",
    "exc_high_core": "high_outgoing_exc_core_nodes.json",
    "exc_low_core": "low_outgoing_exc_core_nodes.json",
    "exc_high_periphery": "high_outgoing_exc_periphery_nodes.json",
    "exc_low_periphery": "low_outgoing_exc_periphery_nodes.json",
    "inh_high": "high_outgoing_inh_nodes.json",
    "inh_low": "low_outgoing_inh_nodes.json",
    "inh_high_core": "high_outgoing_inh_core_nodes.json",
    "inh_low_core": "low_outgoing_inh_core_nodes.json",
    "inh_high_periphery": "high_outgoing_inh_periphery_nodes.json",
    "inh_low_periphery": "low_outgoing_inh_periphery_nodes.json",
}


def load_node_attributes() -> pd.DataFrame:
    node_types = pd.read_csv(NODE_TYPES_FILE, sep="\s+")
    node_types["is_inhibitory"] = node_types["ei"].str.lower().str.startswith("i")
    type_lookup = node_types.set_index("node_type_id")[["pop_name", "is_inhibitory"]]

    with h5py.File(NODE_FILE, "r") as f:
        node_ids = f["nodes"]["v1"]["node_id"][:]
        node_type_ids = f["nodes"]["v1"]["node_type_id"][:]

    df = pd.DataFrame({"node_type_id": node_type_ids}, index=node_ids)
    df.index.name = "node_id"
    df = df.join(type_lookup, on="node_type_id", how="left")
    df["is_inhibitory"] = df["is_inhibitory"].astype(bool)
    return df


def load_node_sets() -> Dict[str, np.ndarray]:
    sets: Dict[str, np.ndarray] = {}
    for name, filename in NODE_SETS.items():
        data = json.loads((NODE_SET_DIR / filename).read_text())
        sets[name] = np.array(data["node_id"], dtype=np.int64)
    return sets


def load_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(EDGE_FILE, "r") as f:
        grp = f["edges"]["v1_to_v1"]
        sources = grp["source_node_id"][:].astype(np.int64)
        targets = grp["target_node_id"][:].astype(np.int64)
        weights = grp["0"]["syn_weight"][:].astype(np.float64)
    return sources, targets, weights


def main() -> None:
    node_attrs = load_node_attributes()
    node_sets = load_node_sets()
    sources, targets, weights = load_edges()

    target_attrs = node_attrs.loc[targets]
    target_is_inh = target_attrs["is_inhibitory"].to_numpy()

    abs_weights = np.abs(weights)

    rows = []
    for label, node_ids in node_sets.items():
        if node_ids.size == 0:
            continue
        mask = np.isin(sources, node_ids)
        if not mask.any():
            continue

        src_weights = abs_weights[mask]
        src_targets_inh = target_is_inh[mask]

        exc_weight = src_weights[~src_targets_inh].sum()
        inh_weight = src_weights[src_targets_inh].sum()
        total_weight = src_weights.sum()

        exc_fraction = exc_weight / total_weight if total_weight > 0 else np.nan
        inh_fraction = inh_weight / total_weight if total_weight > 0 else np.nan

        rows.append(
            {
                "group": label,
                "n_sources": int(node_ids.size),
                "total_connections": int(mask.sum()),
                "exc_connections": int((~src_targets_inh).sum()),
                "inh_connections": int(src_targets_inh.sum()),
                "total_weight_abs": float(total_weight),
                "exc_weight_abs": float(exc_weight),
                "inh_weight_abs": float(inh_weight),
                "exc_weight_fraction": float(exc_fraction),
                "inh_weight_fraction": float(inh_fraction),
            }
        )

    df = pd.DataFrame(rows).sort_values("group")
    out_path = OUTPUT_DIR / "suppressed_outgoing_weight_preference.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
