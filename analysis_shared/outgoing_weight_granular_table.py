#!/usr/bin/env python3
"""Build granular outgoing weight fraction table with detailed target breakdowns."""
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

SOURCE_GROUPS = {
    # Aggregated inhibitory groups (existing)
    "inh_high_core": "high_outgoing_inh_core_nodes.json",
    "inh_high_periphery": "high_outgoing_inh_periphery_nodes.json",
    "inh_low_core": "low_outgoing_inh_core_nodes.json",
    "inh_low_periphery": "low_outgoing_inh_periphery_nodes.json",
    # New: PV/SST/VIP split for core and periphery high/low sources
    "pv_high_core": "pv_high_outgoing_core_nodes.json",
    "pv_low_core": "pv_low_outgoing_core_nodes.json",
    "pv_high_periphery": "pv_high_outgoing_periphery_nodes.json",
    "pv_low_periphery": "pv_low_outgoing_periphery_nodes.json",
    "sst_high_core": "sst_high_outgoing_core_nodes.json",
    "sst_low_core": "sst_low_outgoing_core_nodes.json",
    "sst_high_periphery": "sst_high_outgoing_periphery_nodes.json",
    "sst_low_periphery": "sst_low_outgoing_periphery_nodes.json",
    "vip_high_core": "vip_high_outgoing_core_nodes.json",
    "vip_low_core": "vip_low_outgoing_core_nodes.json",
    "vip_high_periphery": "vip_high_outgoing_periphery_nodes.json",
    "vip_low_periphery": "vip_low_outgoing_periphery_nodes.json",
}


def load_node_attributes() -> pd.DataFrame:
    """Load node attributes including cell type and E/I status."""
    node_types = pd.read_csv(NODE_TYPES_FILE, sep=r"\s+")
    node_types["is_inhibitory"] = node_types["ei"].str.lower().str.startswith("i")

    # Extract cell type marker (Pvalb, Sst, Vip, etc.)
    node_types["cell_marker"] = node_types["pop_name"].str.extract(r"[ei]\d+(\w+)")
    type_lookup = node_types.set_index("node_type_id")[["pop_name", "is_inhibitory", "cell_marker"]]

    with h5py.File(NODE_FILE, "r") as f:
        node_ids = f["nodes"]["v1"]["node_id"][:]
        node_type_ids = f["nodes"]["v1"]["node_type_id"][:]

    df = pd.DataFrame({"node_type_id": node_type_ids}, index=node_ids)
    df.index.name = "node_id"
    df = df.join(type_lookup, on="node_type_id", how="left")
    df["is_inhibitory"] = df["is_inhibitory"].astype(bool)
    return df


def load_node_set(filename: str) -> set[int]:
    """Load a single node set from JSON."""
    data = json.loads((NODE_SET_DIR / filename).read_text())
    return {int(x) for x in data["node_id"]}


def load_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load edge data: source, target, weight."""
    with h5py.File(EDGE_FILE, "r") as f:
        grp = f["edges"]["v1_to_v1"]
        sources = grp["source_node_id"][:].astype(np.int64)
        targets = grp["target_node_id"][:].astype(np.int64)
        weights = grp["0"]["syn_weight"][:].astype(np.float64)
    return sources, targets, weights


def main() -> None:
    print("Loading node attributes...")
    node_attrs = load_node_attributes()

    print("Loading edges...")
    sources, targets, weights = load_edges()
    abs_weights = np.abs(weights)

    # Load target groups
    print("Loading target node sets...")
    target_groups = {
        "exc_high": load_node_set("high_outgoing_exc_nodes.json"),
        "exc_low": load_node_set("low_outgoing_exc_nodes.json"),
        "inh_high": load_node_set("high_outgoing_inh_nodes.json"),
        "inh_low": load_node_set("low_outgoing_inh_nodes.json"),
    }

    # Create cell-type specific target groups for inhibitory cells
    inh_nodes = node_attrs[node_attrs["is_inhibitory"]]
    target_groups["inh_pv"] = set(inh_nodes[inh_nodes["cell_marker"] == "Pvalb"].index)
    target_groups["inh_sst"] = set(inh_nodes[inh_nodes["cell_marker"] == "Sst"].index)
    target_groups["inh_vip"] = set(inh_nodes[inh_nodes["cell_marker"] == "Vip"].index)
    target_groups["inh_htr3a"] = set(inh_nodes[inh_nodes["cell_marker"] == "Htr3a"].index)

    # Combine high/low with cell types
    for weight_level in ["high", "low"]:
        base_set = target_groups[f"inh_{weight_level}"]
        target_groups[f"inh_{weight_level}_pv"] = base_set & target_groups["inh_pv"]
        target_groups[f"inh_{weight_level}_sst"] = base_set & target_groups["inh_sst"]
        target_groups[f"inh_{weight_level}_vip"] = base_set & target_groups["inh_vip"]
        target_groups[f"inh_{weight_level}_htr3a"] = base_set & target_groups["inh_htr3a"]

    # Load source groups and compute statistics
    print("Computing connectivity statistics...")
    rows = []

    for src_label, src_filename in SOURCE_GROUPS.items():
        print(f"  Processing {src_label}...")
        src_nodes = load_node_set(src_filename)
        src_mask = np.isin(sources, list(src_nodes))

        if not src_mask.any():
            continue

        total_weight = abs_weights[src_mask].sum()

        row = {"source_group": src_label, "total_weight": total_weight}

        # Compute fraction for each target group
        for tgt_label, tgt_nodes in target_groups.items():
            mask = src_mask & np.isin(targets, list(tgt_nodes))
            weight = abs_weights[mask].sum()
            fraction = weight / total_weight if total_weight > 0 else 0.0
            row[f"{tgt_label}_weight"] = weight
            row[f"{tgt_label}_fraction"] = fraction

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save full table
    out_full = OUTPUT_DIR / "outgoing_weight_granular_full.csv"
    df.to_csv(out_full, index=False)
    print(f"\nSaved full table to {out_full}")

    # Create a summary table with just the fractions (matching TODOS.md format)
    fraction_cols = [col for col in df.columns if col.endswith("_fraction")]
    summary_df = df[["source_group"] + fraction_cols].copy()
    summary_df.columns = ["group"] + [col.replace("_fraction", "") for col in fraction_cols]

    out_summary = OUTPUT_DIR / "outgoing_weight_granular_summary.csv"
    summary_df.to_csv(out_summary, index=False)
    print(f"Saved summary table to {out_summary}")

    # Print the summary in a readable format
    print("\n" + "="*80)
    print("GRANULAR CONNECTIVITY PATTERNS")
    print("="*80)
    print("\nWeight fractions by target type:")
    print(summary_df.to_string(index=False))

    # Print key comparisons
    print("\n" + "-"*80)
    print("KEY COMPARISONS:")
    print("-"*80)
    for _, row in summary_df.iterrows():
        group = row["group"]
        print(f"\n{group.upper()}:")
        print(f"  E/I balance:")
        print(f"    → Excitatory (all):     {row.get('exc_high', 0) + row.get('exc_low', 0):.3f}")
        print(f"    → Inhibitory (all):     {row.get('inh_high', 0) + row.get('inh_low', 0):.3f}")
        print(f"  Inhibitory breakdown:")
        print(f"    → High-weight inh:      {row.get('inh_high', 0):.3f}")
        print(f"    → Low-weight inh:       {row.get('inh_low', 0):.3f}")
        print(f"  Cell type breakdown (all inh):")
        print(f"    → PV:                   {row.get('inh_pv', 0):.3f}")
        print(f"    → SST:                  {row.get('inh_sst', 0):.3f}")
        print(f"    → VIP:                  {row.get('inh_vip', 0):.3f}")
        print(f"    → Htr3a:                {row.get('inh_htr3a', 0):.3f}")


if __name__ == "__main__":
    main()
