#!/usr/bin/env python3
"""Build core-to-core outgoing target fractions using *synapse count* (n_syns_) as mass.

This mirrors the Fig. 5 stacked-bar target panels, but replaces outgoing-weight fractions
with outgoing *synapse-count* fractions. Cohorts are defined by the synapse-count node sets
(`*_outgoing_synapsecount_*_nodes.json`) created by
`analysis_shared/create_highlow_outgoing_synapsecount_nodesets.py`.

Outputs:
- `core_nll_0/figures/selectivity_outgoing/outgoing_synapsecount_complete_targets{suffix}.csv`
  with columns compatible with `plot_target_fraction_figure5.py` style:
  group,n_connections,exc_total,inh_pv,inh_sst,inh_vip,inh_htr3a,total_check
"""

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

NODE_FILE = BASE_DIR / "network" / "v1_nodes.h5"
NODE_TYPES_FILE = BASE_DIR / "network" / "v1_node_types.csv"

NETWORK_EDGE_FILES: Dict[str, Path] = {
    "bio_trained": BASE_DIR / "network" / "v1_v1_edges_bio_trained.h5",
    "plain": BASE_DIR / "network" / "v1_v1_edges.h5",
    "naive": BASE_DIR / "network" / "v1_v1_edges_naive.h5",
    "checkpoint": BASE_DIR / "network" / "v1_v1_edges_checkpoint.h5",
    "adjusted": BASE_DIR / "network" / "v1_v1_edges_adjusted.h5",
}

BASE_TYPES: Tuple[str, ...] = ("exc", "inh", "pv", "sst", "vip")
WEIGHT_ORDER: Tuple[str, ...] = ("high", "mid", "low")
CELL_MARKER_MAP: Dict[str, str] = {"pv": "Pvalb", "sst": "Sst", "vip": "Vip"}


def load_node_attributes() -> pd.DataFrame:
    node_types = pd.read_csv(NODE_TYPES_FILE, sep=r"\s+")
    node_types["is_inhibitory"] = node_types["ei"].str.lower().str.startswith("i")
    node_types["cell_marker"] = node_types["pop_name"].str.extract(r"[ei]\d+(\w+)")
    type_lookup = node_types.set_index("node_type_id")[
        ["pop_name", "is_inhibitory", "cell_marker"]
    ]

    with h5py.File(NODE_FILE, "r") as f:
        node_ids = f["nodes"]["v1"]["node_id"][:].astype(np.int64)
        node_type_ids = f["nodes"]["v1"]["node_type_id"][:].astype(np.int64)
        x = f["nodes"]["v1"]["0"]["x"][:].astype(np.float64)
        z = f["nodes"]["v1"]["0"]["z"][:].astype(np.float64)

    df = pd.DataFrame({"node_type_id": node_type_ids, "x": x, "z": z}, index=node_ids)
    df.index.name = "node_id"
    df = df.join(type_lookup, on="node_type_id", how="left")
    df["is_inhibitory"] = df["is_inhibitory"].fillna(False).astype(bool)
    df["distance_from_center"] = np.sqrt((df["x"] - np.median(x)) ** 2 + (df["z"] - np.median(z)) ** 2)
    return df


def load_edges(edge_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(edge_file, "r") as f:
        grp = f["edges"]["v1_to_v1"]
        sources = grp["source_node_id"][:].astype(np.int64)
        targets = grp["target_node_id"][:].astype(np.int64)
        if "n_syns_" in grp["0"]:
            n_syns = grp["0"]["n_syns_"][:].astype(np.float64)
        elif "nsyns" in grp["0"]:
            n_syns = grp["0"]["nsyns"][:].astype(np.float64)
        else:
            raise KeyError("Neither 'n_syns_' nor 'nsyns' found in edges/v1_to_v1/0")
    return sources, targets, n_syns


def load_json_node_set(filename: str) -> set[int]:
    path = NODE_SET_DIR / filename
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    return {int(x) for x in data.get("node_id", [])}


def node_set_filename(base: str, weight: str, region: str) -> str:
    if base in {"pv", "sst", "vip"}:
        return f"{base}_{weight}_outgoing_synapsecount_{region}_nodes.json"
    return f"{weight}_outgoing_synapsecount_{base}_{region}_nodes.json"


def build_base_sets(node_attrs: pd.DataFrame, core_mask: pd.Series) -> Dict[Tuple[str, str], set[int]]:
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


def generate_source_groups(base_sets: Dict[Tuple[str, str], set[int]]) -> List[Dict[str, object]]:
    source_groups: List[Dict[str, object]] = []
    for (base, region), base_set in base_sets.items():
        if region != "core":
            continue  # Fig 5 panels use core groups
        if not base_set:
            continue

        high_set = load_json_node_set(node_set_filename(base, "high", region)) & base_set
        low_set = load_json_node_set(node_set_filename(base, "low", region)) & base_set
        mid_set = base_set - high_set - low_set

        group_map = {"high": high_set, "mid": mid_set, "low": low_set}
        for weight, nodes in group_map.items():
            if not nodes:
                continue
            source_groups.append(
                {
                    "label": f"{base}_{weight}_{region}",
                    "base": base,
                    "region": region,
                    "weight": weight,
                    "node_ids": np.fromiter(sorted(nodes), dtype=np.int64),
                }
            )
    return source_groups


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute outgoing target fractions using synapse-count mass (core-to-core)."
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Base directory containing network/ and node_sets/ (default: core_nll_0).",
    )
    ap.add_argument(
        "--network",
        choices=sorted(NETWORK_EDGE_FILES),
        default="bio_trained",
        help="Which network edge file to use (default: bio_trained).",
    )
    ap.add_argument(
        "--edge-file",
        type=Path,
        help="Explicit path to an edge file. Overrides --network.",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix for output filename (e.g. 'plain' -> *_plain.csv).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    global BASE_DIR, NODE_SET_DIR, OUTPUT_DIR, NODE_FILE, NODE_TYPES_FILE, NETWORK_EDGE_FILES
    BASE_DIR = args.base_dir
    NODE_SET_DIR = BASE_DIR / "node_sets"
    OUTPUT_DIR = BASE_DIR / "figures" / "selectivity_outgoing"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    NODE_FILE = BASE_DIR / "network" / "v1_nodes.h5"
    NODE_TYPES_FILE = BASE_DIR / "network" / "v1_node_types.csv"
    NETWORK_EDGE_FILES = {
        "bio_trained": BASE_DIR / "network" / "v1_v1_edges_bio_trained.h5",
        "plain": BASE_DIR / "network" / "v1_v1_edges.h5",
        "naive": BASE_DIR / "network" / "v1_v1_edges_naive.h5",
        "checkpoint": BASE_DIR / "network" / "v1_v1_edges_checkpoint.h5",
        "adjusted": BASE_DIR / "network" / "v1_v1_edges_adjusted.h5",
    }

    edge_file = args.edge_file or NETWORK_EDGE_FILES.get(args.network)
    if edge_file is None:
        raise ValueError(f"No edge file mapping for network '{args.network}'")
    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")

    suffix = f"_{args.tag}" if args.tag else ""

    print("Loading node attributes...")
    node_attrs = load_node_attributes()

    # Core definition: match existing 200um radius convention.
    core_mask = node_attrs["distance_from_center"] <= 200.0
    core_node_ids = set(node_attrs.index[core_mask])
    print(f"Core nodes: {len(core_node_ids)}")

    print(f"Loading edges from {edge_file} ...")
    sources, targets, n_syns = load_edges(edge_file)
    core_to_core_mask = np.isin(sources, list(core_node_ids)) & np.isin(targets, list(core_node_ids))
    sources = sources[core_to_core_mask]
    targets = targets[core_to_core_mask]
    n_syns = n_syns[core_to_core_mask]
    print(f"Core-to-core edges: {len(sources):,}")

    print("Building cohort source groups (synapsecount-defined)...")
    base_sets = build_base_sets(node_attrs, core_mask)
    source_groups = generate_source_groups(base_sets)
    if not source_groups:
        raise RuntimeError("No source cohorts generated. Did you run create_highlow_outgoing_synapsecount_nodesets?")

    inh_nodes = node_attrs[core_mask & node_attrs["is_inhibitory"]]
    target_sets = {
        "inh_pv": set(inh_nodes[inh_nodes["cell_marker"] == "Pvalb"].index),
        "inh_sst": set(inh_nodes[inh_nodes["cell_marker"] == "Sst"].index),
        "inh_vip": set(inh_nodes[inh_nodes["cell_marker"] == "Vip"].index),
        "inh_htr3a": set(inh_nodes[inh_nodes["cell_marker"] == "Htr3a"].index),
    }

    # Precompute inhibitory-target flags for fast slicing
    tgt_is_inh = node_attrs.loc[targets, "is_inhibitory"].to_numpy(dtype=bool)
    tgt_marker = node_attrs.loc[targets, "cell_marker"].astype(str).to_numpy()

    rows: List[Dict[str, float]] = []
    for group in source_groups:
        label = str(group["label"])
        node_ids = group["node_ids"]
        src_mask = np.isin(sources, node_ids)
        if not src_mask.any():
            continue

        mass_total = float(np.sum(n_syns[src_mask]))
        if mass_total <= 0:
            continue

        # Exc targets are all non-inhibitory targets
        exc_mass = float(np.sum(n_syns[src_mask & (~tgt_is_inh)]))

        pv_mass = float(np.sum(n_syns[src_mask & tgt_is_inh & (tgt_marker == "Pvalb")]))
        sst_mass = float(np.sum(n_syns[src_mask & tgt_is_inh & (tgt_marker == "Sst")]))
        vip_mass = float(np.sum(n_syns[src_mask & tgt_is_inh & (tgt_marker == "Vip")]))
        htr3a_mass = float(np.sum(n_syns[src_mask & tgt_is_inh & (tgt_marker == "Htr3a")]))

        row = {
            "group": label,
            # name kept for compatibility with plotting scripts; this is total synapse-count mass
            "n_connections": mass_total,
            "exc_total": exc_mass / mass_total,
            "inh_pv": pv_mass / mass_total,
            "inh_sst": sst_mass / mass_total,
            "inh_vip": vip_mass / mass_total,
            "inh_htr3a": htr3a_mass / mass_total,
        }
        row["total_check"] = (
            row["exc_total"] + row["inh_pv"] + row["inh_sst"] + row["inh_vip"] + row["inh_htr3a"]
        )
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("group")
    out_path = OUTPUT_DIR / f"outgoing_synapsecount_complete_targets{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Diagnostics for core groups only
    core_df = df[df["group"].str.contains("_core")].copy()
    if not core_df.empty:
        print("\nTotal fractions (should be ~1.0):")
        for _, r in core_df.iterrows():
            print(f"{r['group']:20s}: {r['total_check']:.3f}")


if __name__ == "__main__":
    main()


