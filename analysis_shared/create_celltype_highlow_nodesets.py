#!/usr/bin/env python3
"""Create node sets for high/low outgoing weight neurons within each cell type (PV, SST, VIP).

Each cell type is now split within layers so that the same fraction of neurons
is selected from every layer-specific subgroup. We take the top third of each
layer for the "high" node set and the bottom third for the "low" node set. The
middle third remains unused by these node sets.
"""

import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "core_nll_0"
NODE_SET_DIR = BASE_DIR / "node_sets"
NODE_FILE = BASE_DIR / "network" / "v1_nodes.h5"
NODE_TYPES_FILE = BASE_DIR / "network" / "v1_node_types.csv"
EDGE_FILE = BASE_DIR / "network" / "v1_v1_edges_bio_trained.h5"
CORE_RADIUS_UM = 200.0

LAYER_MAP = {
    "1": "L1",
    "23": "L2/3",
    "4": "L4",
    "5": "L5",
    "6": "L6",
}


def compute_outgoing_weights(
    node_ids: set[int], sources: np.ndarray, weights: np.ndarray
) -> dict:
    """Compute total outgoing weight for each node."""
    outgoing = {}
    for node_id in node_ids:
        mask = sources == node_id
        total_weight = np.abs(weights[mask]).sum()
        outgoing[node_id] = total_weight
    return outgoing


def _select_high_low_by_layer(
    subset: pd.DataFrame,
    sources: np.ndarray,
    weights: np.ndarray,
) -> tuple[list[int], list[int], list[dict]]:
    """Select high/low by layer; return node ids and per-layer stats records."""
    if subset.empty:
        return [], [], []

    outgoing_weights = compute_outgoing_weights(set(subset.index.tolist()), sources, weights)

    high_cells: list[int] = []
    low_cells: list[int] = []
    stats: list[dict] = []

    for layer, layer_df in subset.groupby("layer"):
        layer_ids = [int(node_id) for node_id in layer_df.index]
        sorted_ids = sorted(layer_ids, key=lambda nid: outgoing_weights[nid], reverse=True)
        n_cells = len(sorted_ids)
        n_third = n_cells // 3

        if n_third == 0:
            stats.append(
                {
                    "layer": layer,
                    "total": n_cells,
                    "high": 0,
                    "low": 0,
                    "high_mean_weight": np.nan,
                    "low_mean_weight": np.nan,
                }
            )
            continue

        layer_high = sorted_ids[:n_third]
        layer_low = sorted_ids[-n_third:]

        high_cells.extend(layer_high)
        low_cells.extend(layer_low)

        high_weights = [outgoing_weights[cell_id] for cell_id in layer_high]
        low_weights = [outgoing_weights[cell_id] for cell_id in layer_low]

        stats.append(
            {
                "layer": layer,
                "total": n_cells,
                "high": len(layer_high),
                "low": len(layer_low),
                "high_mean_weight": (np.mean(high_weights) if high_weights else np.nan),
                "low_mean_weight": (np.mean(low_weights) if low_weights else np.nan),
            }
        )

    return high_cells, low_cells, stats


def main():
    # Load node types
    node_types = pd.read_csv(NODE_TYPES_FILE, sep=r"\s+")
    node_types["cell_marker"] = node_types["pop_name"].str.extract(r"[ei]\d+(\w+)")
    node_types["layer_label"] = node_types["pop_name"].apply(_extract_layer_label)
    cell_marker_lookup = node_types.set_index("node_type_id")["cell_marker"]
    pop_lookup = node_types.set_index("node_type_id")["pop_name"]
    layer_lookup = node_types.set_index("node_type_id")["layer_label"]

    # Load nodes
    with h5py.File(NODE_FILE, "r") as f:
        node_ids = f["nodes"]["v1"]["node_id"][:]
        node_type_ids = f["nodes"]["v1"]["node_type_id"][:]
        x_coord = f["nodes"]["v1"]["0"]["x"][:]
        z_coord = f["nodes"]["v1"]["0"]["z"][:]

    # Create dataframe
    df = pd.DataFrame({"node_type_id": node_type_ids}, index=node_ids)
    df["x"] = x_coord
    df["z"] = z_coord
    df["core"] = (df["x"] ** 2 + df["z"] ** 2) < CORE_RADIUS_UM**2
    df["cell_marker"] = df["node_type_id"].map(cell_marker_lookup)
    df["pop_name"] = df["node_type_id"].map(pop_lookup)
    df["layer"] = df["node_type_id"].map(layer_lookup)

    # Load edges
    print("Loading edges...")
    with h5py.File(EDGE_FILE, "r") as f:
        sources = f["edges"]["v1_to_v1"]["source_node_id"][:].astype(np.int64)
        weights = f["edges"]["v1_to_v1"]["0"]["syn_weight"][:].astype(np.float64)

    # Process each cell type
    cell_types = {"pv": "Pvalb", "sst": "Sst", "vip": "Vip"}

    results = []

    for abbrev, marker in cell_types.items():
        print(f"\nProcessing {abbrev.upper()} ({marker})...")

        # Get all cells of this type that have a known layer assignment
        subset_core = df[(df["cell_marker"] == marker) & df["layer"].notna() & df["core"]]
        subset_periph = df[(df["cell_marker"] == marker) & df["layer"].notna() & (~df["core"])]

        print(
            f"  Cells with layer labels: core={len(subset_core)}, periphery={len(subset_periph)}"
        )

        # Compute selections
        print("  Selecting high/low within layers (core)...")
        high_core, low_core, stats_core = _select_high_low_by_layer(subset_core, sources, weights)
        print("  Selecting high/low within layers (periphery)...")
        high_periph, low_periph, stats_periph = _select_high_low_by_layer(subset_periph, sources, weights)

        # Save stats for summary
        for rec in stats_core:
            results.append({"cell_type": abbrev.upper(), "region": "core", **rec})
        for rec in stats_periph:
            results.append({"cell_type": abbrev.upper(), "region": "periphery", **rec})

        # Legacy filenames (core-only) for backward compatibility
        legacy_high_file = NODE_SET_DIR / f"{abbrev}_high_outgoing_nodes.json"
        legacy_low_file = NODE_SET_DIR / f"{abbrev}_low_outgoing_nodes.json"
        with open(legacy_high_file, "w") as f:
            json.dump({"population": "v1", "node_id": high_core}, f, indent=2)
        with open(legacy_low_file, "w") as f:
            json.dump({"population": "v1", "node_id": low_core}, f, indent=2)
        print(f"  → Created: {legacy_high_file}")
        print(f"  → Created: {legacy_low_file}")

        # New explicit core/periphery files
        files_to_write = {
            f"{abbrev}_high_outgoing_core_nodes.json": high_core,
            f"{abbrev}_low_outgoing_core_nodes.json": low_core,
            f"{abbrev}_high_outgoing_periphery_nodes.json": high_periph,
            f"{abbrev}_low_outgoing_periphery_nodes.json": low_periph,
        }
        for fname, node_ids in files_to_write.items():
            out_path = NODE_SET_DIR / fname
            with open(out_path, "w") as f:
                json.dump({"population": "v1", "node_id": node_ids}, f, indent=2)
            print(f"  → Created: {out_path} ({len(node_ids)} cells)")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    df_results = pd.DataFrame(results)
    print(df_results.sort_values(["cell_type", "layer"]).to_string(index=False))
    print("=" * 80)


def _extract_layer_label(pop_name: str) -> str | None:
    match = re.search(r"[ei](\d+)", pop_name)
    if not match:
        return None
    return LAYER_MAP.get(match.group(1))


if __name__ == "__main__":
    main()
