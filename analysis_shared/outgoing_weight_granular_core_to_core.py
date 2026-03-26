#!/usr/bin/env python3
"""Build granular outgoing weight fraction table with CORE-TO-CORE connections only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "core_nll_0"
NODE_SET_DIR = BASE_DIR / "node_sets"
OUTPUT_DIR = BASE_DIR / "figures" / "selectivity_outgoing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NETWORK_EDGE_FILES: Dict[str, Path] = {
    "bio_trained": BASE_DIR / "network" / "v1_v1_edges_bio_trained.h5",
    "plain": BASE_DIR / "network" / "v1_v1_edges.h5",
    "naive": BASE_DIR / "network" / "v1_v1_edges_naive.h5",
    "checkpoint": BASE_DIR / "network" / "v1_v1_edges_checkpoint.h5",
    "adjusted": BASE_DIR / "network" / "v1_v1_edges_adjusted.h5",
}
NODE_FILE = BASE_DIR / "network" / "v1_nodes.h5"
NODE_TYPES_FILE = BASE_DIR / "network" / "v1_node_types.csv"

BASE_TYPES: Tuple[str, ...] = ("exc", "inh", "pv", "sst", "vip")
REGIONS: Tuple[str, ...] = ("core", "periphery")
WEIGHT_ORDER: Tuple[str, ...] = ("high", "mid", "low")
CELL_MARKER_MAP: Dict[str, str] = {"pv": "Pvalb", "sst": "Sst", "vip": "Vip"}


def load_node_attributes() -> pd.DataFrame:
    """Load node attributes including position, E/I status, and cell markers."""
    node_types = pd.read_csv(NODE_TYPES_FILE, sep=r"\s+")
    node_types["is_inhibitory"] = node_types["ei"].str.lower().str.startswith("i")
    node_types["cell_marker"] = node_types["pop_name"].str.extract(r"[ei]\d+(\w+)")

    type_lookup = node_types.set_index("node_type_id")[
        ["pop_name", "is_inhibitory", "cell_marker"]
    ]

    with h5py.File(NODE_FILE, "r") as f:
        node_ids = f["nodes"]["v1"]["node_id"][:]
        node_type_ids = f["nodes"]["v1"]["node_type_id"][:]
        x = f["nodes"]["v1"]["0"]["x"][:]
        y = f["nodes"]["v1"]["0"]["y"][:]
        z = f["nodes"]["v1"]["0"]["z"][:]

    df = pd.DataFrame(
        {"node_type_id": node_type_ids, "x": x, "y": y, "z": z}, index=node_ids
    )
    df.index.name = "node_id"
    df = df.join(type_lookup, on="node_type_id", how="left")
    df["is_inhibitory"] = df["is_inhibitory"].astype(bool)

    center_x, center_z = np.median(x), np.median(z)
    df["distance_from_center"] = np.sqrt(
        (df["x"] - center_x) ** 2 + (df["z"] - center_z) ** 2
    )

    return df


def define_spatial_core_threshold(
    node_attrs: pd.DataFrame, threshold_um: float = 200.0
) -> float:
    """Print diagnostics about the spatial core region and return the threshold."""
    distances = node_attrs["distance_from_center"]
    n_core = (distances < threshold_um).sum()
    pct_core = 100.0 * n_core / len(distances)

    print(f"Spatial core threshold: {threshold_um:.1f} µm from center (in XZ space)")
    print(f"Spatial core contains {n_core} / {len(distances)} nodes ({pct_core:.1f}%)")
    print(f"Distance range: [{distances.min():.1f}, {distances.max():.1f}] µm")
    print(f"Expected ~25% if total radius is 400µm, actual: {pct_core:.1f}%")
    return threshold_um


def load_edges(edge_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load edge data (source, target, weight) from an edge file."""
    with h5py.File(edge_file, "r") as f:
        grp = f["edges"]["v1_to_v1"]
        sources = grp["source_node_id"][:].astype(np.int64)
        targets = grp["target_node_id"][:].astype(np.int64)
        weights = grp["0"]["syn_weight"][:].astype(np.float64)
    return sources, targets, weights


def load_json_node_set(filename: str) -> set[int]:
    path = NODE_SET_DIR / filename
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    return {int(x) for x in data.get("node_id", [])}


def node_set_filename(base: str, weight: str, region: str) -> str:
    if base in {"pv", "sst", "vip"}:
        return f"{base}_{weight}_outgoing_{region}_nodes.json"
    return f"{weight}_outgoing_{base}_{region}_nodes.json"


def build_base_sets(
    node_attrs: pd.DataFrame, core_mask: pd.Series
) -> Dict[Tuple[str, str], set[int]]:
    """Return base node sets (by cell type and region) used to derive mid groups."""
    periphery_mask = ~core_mask
    base_sets: Dict[Tuple[str, str], set[int]] = {}

    for base in BASE_TYPES:
        if base == "exc":
            base_mask = ~node_attrs["is_inhibitory"]
        elif base == "inh":
            base_mask = node_attrs["is_inhibitory"]
        else:
            marker = CELL_MARKER_MAP[base]
            base_mask = node_attrs["cell_marker"] == marker

        core_ids = node_attrs.index[base_mask & core_mask]
        periphery_ids = node_attrs.index[base_mask & periphery_mask]
        base_sets[(base, "core")] = set(int(x) for x in core_ids)
        base_sets[(base, "periphery")] = set(int(x) for x in periphery_ids)

    return base_sets


def generate_source_groups(
    base_sets: Dict[Tuple[str, str], set[int]]
) -> List[Dict[str, object]]:
    """Construct source group definitions including high/mid/low cohorts."""
    source_groups: List[Dict[str, object]] = []

    for (base, region), base_set in base_sets.items():
        if not base_set:
            continue

        high_file = node_set_filename(base, "high", region)
        low_file = node_set_filename(base, "low", region)

        high_set = load_json_node_set(high_file) & base_set
        low_set = load_json_node_set(low_file) & base_set
        mid_set = base_set - high_set - low_set

        group_map = {
            "high": high_set,
            "mid": mid_set,
            "low": low_set,
        }

        for weight, nodes in group_map.items():
            if not nodes:
                continue
            label = f"{base}_{weight}_{region}"
            node_array = np.fromiter(sorted(nodes), dtype=np.int64)
            source_groups.append(
                {
                    "label": label,
                    "base": base,
                    "region": region,
                    "weight": weight,
                    "node_ids": node_array,
                }
            )

    return source_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate outgoing weight tables for core-to-core connectivity."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Base directory containing network/ and node_sets/ (default: core_nll_0).",
    )
    parser.add_argument(
        "--network",
        choices=sorted(NETWORK_EDGE_FILES),
        default="bio_trained",
        help="Which network edge file to use (default: bio_trained).",
    )
    parser.add_argument(
        "--edge-file",
        type=Path,
        help="Explicit path to a v1_v1_edges*.h5 file. Overrides --network.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix for output filenames (e.g., 'plain' -> *_plain.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global BASE_DIR, NODE_SET_DIR, OUTPUT_DIR, NETWORK_EDGE_FILES, NODE_FILE, NODE_TYPES_FILE
    BASE_DIR = args.base_dir
    NODE_SET_DIR = BASE_DIR / "node_sets"
    OUTPUT_DIR = BASE_DIR / "figures" / "selectivity_outgoing"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    NETWORK_EDGE_FILES = {
        "bio_trained": BASE_DIR / "network" / "v1_v1_edges_bio_trained.h5",
        "plain": BASE_DIR / "network" / "v1_v1_edges.h5",
        "naive": BASE_DIR / "network" / "v1_v1_edges_naive.h5",
        "checkpoint": BASE_DIR / "network" / "v1_v1_edges_checkpoint.h5",
        "adjusted": BASE_DIR / "network" / "v1_v1_edges_adjusted.h5",
    }
    NODE_FILE = BASE_DIR / "network" / "v1_nodes.h5"
    NODE_TYPES_FILE = BASE_DIR / "network" / "v1_node_types.csv"

    edge_file = args.edge_file or NETWORK_EDGE_FILES.get(args.network)
    if edge_file is None:
        raise ValueError(f"No edge file mapping for network '{args.network}'")
    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")

    suffix = f"_{args.tag}" if args.tag else ""

    print("Loading node attributes...")
    node_attrs = load_node_attributes()

    print("Defining spatial core region threshold...")
    core_threshold = define_spatial_core_threshold(node_attrs, threshold_um=200.0)
    core_mask = node_attrs["distance_from_center"] <= core_threshold
    core_node_ids = set(node_attrs.index[core_mask])
    print(f"Core region has {len(core_node_ids)} nodes")

    print(f"Loading edges from {edge_file}...")
    sources, targets, weights = load_edges(edge_file)
    abs_weights = np.abs(weights)

    core_to_core_mask = np.isin(sources, list(core_node_ids)) & np.isin(
        targets, list(core_node_ids)
    )
    print(f"Total edges: {len(sources):,}")
    print(
        f"Core-to-core edges: {core_to_core_mask.sum():,} ({100 * core_to_core_mask.sum() / len(sources):.1f}%)"
    )

    sources_c2c = sources[core_to_core_mask]
    targets_c2c = targets[core_to_core_mask]
    weights_c2c = abs_weights[core_to_core_mask]

    print("Building base node sets...")
    base_sets = build_base_sets(node_attrs, core_mask)

    print("Generating source cohorts (high/mid/low)...")
    source_groups = generate_source_groups(base_sets)
    if not source_groups:
        raise RuntimeError("No source cohorts generated. Verify node sets.")

    print("Loading target node sets...")
    target_groups = {
        "exc_high": load_json_node_set("high_outgoing_exc_nodes.json") & core_node_ids,
        "exc_low": load_json_node_set("low_outgoing_exc_nodes.json") & core_node_ids,
        "inh_high": load_json_node_set("high_outgoing_inh_nodes.json") & core_node_ids,
        "inh_low": load_json_node_set("low_outgoing_inh_nodes.json") & core_node_ids,
    }

    inh_nodes = node_attrs[(node_attrs["is_inhibitory"]) & core_mask]
    target_groups["inh_pv"] = set(inh_nodes[inh_nodes["cell_marker"] == "Pvalb"].index)
    target_groups["inh_sst"] = set(inh_nodes[inh_nodes["cell_marker"] == "Sst"].index)
    target_groups["inh_vip"] = set(inh_nodes[inh_nodes["cell_marker"] == "Vip"].index)
    target_groups["inh_htr3a"] = set(
        inh_nodes[inh_nodes["cell_marker"] == "Htr3a"].index
    )

    for weight in ("high", "low"):
        base_set = target_groups[f"inh_{weight}"]
        target_groups[f"inh_{weight}_pv"] = base_set & target_groups["inh_pv"]
        target_groups[f"inh_{weight}_sst"] = base_set & target_groups["inh_sst"]
        target_groups[f"inh_{weight}_vip"] = base_set & target_groups["inh_vip"]
        target_groups[f"inh_{weight}_htr3a"] = base_set & target_groups["inh_htr3a"]

    print("Computing connectivity statistics (core-to-core only)...")
    rows: List[Dict[str, float]] = []

    for group in source_groups:
        label = group["label"]
        node_ids = group["node_ids"]
        print(f"  Processing {label} (n={len(node_ids)})...")

        if node_ids.size == 0:
            print(f"    WARNING: Empty node set for {label}, skipping")
            continue

        src_mask = np.isin(sources_c2c, node_ids)
        if not src_mask.any():
            print(f"    WARNING: No core-to-core connections found for {label}")
            continue

        total_weight = weights_c2c[src_mask].sum()
        row = {
            "source_group": label,
            "base_type": group["base"],
            "region": group["region"],
            "weight_group": group["weight"],
            "total_weight": total_weight,
            "n_connections": int(src_mask.sum()),
        }

        for tgt_label, tgt_nodes in target_groups.items():
            mask = src_mask & np.isin(targets_c2c, list(tgt_nodes))
            weight = weights_c2c[mask].sum()
            fraction = weight / total_weight if total_weight > 0 else 0.0
            row[f"{tgt_label}_weight"] = weight
            row[f"{tgt_label}_fraction"] = fraction

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No connectivity rows generated.")

    out_full = OUTPUT_DIR / f"outgoing_weight_granular_core_to_core_full{suffix}.csv"
    df.to_csv(out_full, index=False)
    print(f"\nSaved full table to {out_full}")

    fraction_cols = [col for col in df.columns if col.endswith("_fraction")]
    summary_df = df[["source_group", "n_connections"] + fraction_cols].copy()
    summary_df.columns = ["group", "n_connections"] + [
        col.replace("_fraction", "") for col in fraction_cols
    ]
    summary_df["inh_total"] = (
        summary_df["inh_pv"]
        + summary_df["inh_sst"]
        + summary_df["inh_vip"]
        + summary_df["inh_htr3a"]
    )
    summary_df["exc_total"] = 1.0 - summary_df["inh_total"]

    out_summary = (
        OUTPUT_DIR / f"outgoing_weight_granular_core_to_core_summary{suffix}.csv"
    )
    summary_df.to_csv(out_summary, index=False)
    print(f"Saved summary table to {out_summary}")

    print("\n" + "=" * 80)
    print("GRANULAR CONNECTIVITY PATTERNS (CORE-TO-CORE ONLY)")
    print("=" * 80)
    print("\nWeight fractions by target type:")
    print(summary_df.to_string(index=False))

    print("\n" + "-" * 80)
    print("KEY COMPARISONS (CORE-TO-CORE):")
    print("-" * 80)

    for base in BASE_TYPES:
        for weight in WEIGHT_ORDER:
            label = f"{base}_{weight}_core"
            row = summary_df[summary_df["group"] == label]
            if row.empty:
                continue

            row = row.iloc[0]
            n_conn = int(row["n_connections"])
            exc_total = row["exc_total"]
            inh_total = (
                row["inh_pv"] + row["inh_sst"] + row["inh_vip"] + row["inh_htr3a"]
            )

            print(f"\n{label.upper()} ({n_conn:,} connections):")
            print("  E/I balance:")
            print(f"    → Excitatory (all):     {exc_total:.3f}")
            print(f"    → Inhibitory (all):     {inh_total:.3f}")
            print("  Inhibitory breakdown:")
            print(f"    → High-weight inh:      {row['inh_high']:.3f}")
            print(f"    → Low-weight inh:       {row['inh_low']:.3f}")
            print("  Cell type breakdown (all inh):")
            print(f"    → PV:                   {row['inh_pv']:.3f}")
            print(f"    → SST:                  {row['inh_sst']:.3f}")
            print(f"    → VIP:                  {row['inh_vip']:.3f}")


if __name__ == "__main__":
    main()
