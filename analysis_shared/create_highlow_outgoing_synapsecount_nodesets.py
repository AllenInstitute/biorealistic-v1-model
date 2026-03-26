#!/usr/bin/env python3
"""Create node sets for high/low outgoing *synapse-count* cohorts (Figure 5 degree-style analysis).

This is analogous to the existing outgoing-weight cohorts, but uses the sum of `n_syns_`
per source neuron (i.e. total outgoing synapse count) as the cohort-defining quantity.

We keep the same core/periphery split and (for fair comparison across depth) we select
high/low *within each layer* (top third and bottom third) for each base group.

Outputs are written to `core_nll_0/node_sets/` with filenames that include
`outgoing_synapsecount`, so existing weight-based node sets are not overwritten.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analysis_shared.array_utils import safe_bincount_sum

PROJECT_ROOT = Path(__file__).resolve().parent.parent


CORE_RADIUS_UM = 200.0

LAYER_MAP = {
    "1": "L1",
    "23": "L2/3",
    "4": "L4",
    "5": "L5",
    "6": "L6",
}


def _extract_layer_label(pop_name: str) -> str | None:
    match = re.search(r"[ei](\d+)", str(pop_name))
    if not match:
        return None
    return LAYER_MAP.get(match.group(1))


def load_node_table(base_dir: Path) -> pd.DataFrame:
    node_types_file = base_dir / "network" / "v1_node_types.csv"
    node_file = base_dir / "network" / "v1_nodes.h5"

    node_types = pd.read_csv(node_types_file, sep=r"\s+")
    node_types["is_inhibitory"] = node_types["ei"].str.lower().str.startswith("i")
    node_types["cell_marker"] = node_types["pop_name"].str.extract(r"[ei]\d+(\w+)")
    node_types["layer_label"] = node_types["pop_name"].apply(_extract_layer_label)

    type_lookup = node_types.set_index("node_type_id")[
        ["pop_name", "is_inhibitory", "cell_marker", "layer_label"]
    ]

    with h5py.File(node_file, "r") as f:
        node_ids = f["nodes"]["v1"]["node_id"][:].astype(np.int64)
        node_type_ids = f["nodes"]["v1"]["node_type_id"][:].astype(np.int64)
        x = f["nodes"]["v1"]["0"]["x"][:].astype(np.float64)
        z = f["nodes"]["v1"]["0"]["z"][:].astype(np.float64)

    df = pd.DataFrame({"node_type_id": node_type_ids, "x": x, "z": z}, index=node_ids)
    df.index.name = "node_id"
    df = df.join(type_lookup, on="node_type_id", how="left")
    df["is_inhibitory"] = df["is_inhibitory"].fillna(False).astype(bool)
    df["core"] = (df["x"] ** 2 + df["z"] ** 2) < CORE_RADIUS_UM**2
    df = df.rename(columns={"layer_label": "layer"})
    return df


def load_outgoing_synapsecount(edge_file: Path) -> pd.Series:
    with h5py.File(edge_file, "r") as f:
        grp = f["edges"]["v1_to_v1"]
        sources = grp["source_node_id"][:].astype(np.int64)
        # Prefer n_syns_ (this repo uses it); fall back to nsyns if present
        if "n_syns_" in grp["0"]:
            n_syns = grp["0"]["n_syns_"][:].astype(np.float64)
        elif "nsyns" in grp["0"]:
            n_syns = grp["0"]["nsyns"][:].astype(np.float64)
        else:
            raise KeyError("Neither 'n_syns_' nor 'nsyns' found in edges/v1_to_v1/0")

    n_nodes = int(sources.max()) + 1 if sources.size else 0
    sums = safe_bincount_sum(sources, n_syns, minlength=n_nodes)
    return pd.Series(sums, name="outgoing_synapsecount")


def _node_set_filename(base: str, weight: str, region: str) -> str:
    if base in {"pv", "sst", "vip"}:
        return f"{base}_{weight}_outgoing_synapsecount_{region}_nodes.json"
    return f"{weight}_outgoing_synapsecount_{base}_{region}_nodes.json"


def _select_high_low_by_layer(
    subset: pd.DataFrame,
    metric: pd.Series,
) -> tuple[list[int], list[int], list[dict]]:
    if subset.empty:
        return [], [], []

    high_cells: list[int] = []
    low_cells: list[int] = []
    stats: list[dict] = []

    # Ensure metric is aligned on node_id
    metric = metric.reindex(subset.index).fillna(0.0)

    for layer, layer_df in subset.groupby("layer"):
        layer_ids = [int(nid) for nid in layer_df.index]
        layer_vals = metric.loc[layer_ids].to_dict()
        sorted_ids = sorted(layer_ids, key=lambda nid: layer_vals[nid], reverse=True)
        n_cells = len(sorted_ids)
        n_third = n_cells // 3
        if n_third == 0:
            stats.append(
                {
                    "layer": layer,
                    "total": n_cells,
                    "high": 0,
                    "low": 0,
                    "high_mean_synapsecount": np.nan,
                    "low_mean_synapsecount": np.nan,
                }
            )
            continue

        layer_high = sorted_ids[:n_third]
        layer_low = sorted_ids[-n_third:]
        high_cells.extend(layer_high)
        low_cells.extend(layer_low)

        high_vals = [layer_vals[c] for c in layer_high]
        low_vals = [layer_vals[c] for c in layer_low]
        stats.append(
            {
                "layer": layer,
                "total": n_cells,
                "high": len(layer_high),
                "low": len(layer_low),
                "high_mean_synapsecount": float(np.mean(high_vals)) if high_vals else np.nan,
                "low_mean_synapsecount": float(np.mean(low_vals)) if low_vals else np.nan,
            }
        )

    return high_cells, low_cells, stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create high/low outgoing synapse-count node sets for Fig. 5 (degree-style cohorts)."
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path("core_nll_0"),
        help="Network base directory (default: core_nll_0).",
    )
    ap.add_argument(
        "--edge-file",
        type=Path,
        default=None,
        help="Edge file to compute outgoing synapse-count from (default: <base_dir>/network/v1_v1_edges_bio_trained.h5).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    node_set_dir = base_dir / "node_sets"
    node_set_dir.mkdir(parents=True, exist_ok=True)

    edge_file = args.edge_file or (base_dir / "network" / "v1_v1_edges_bio_trained.h5")
    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")

    print("Loading node metadata...")
    nodes = load_node_table(base_dir)

    print(f"Computing outgoing synapse counts from: {edge_file}")
    syn_count = load_outgoing_synapsecount(edge_file)

    # Bases for cohort definition. PV/SST/VIP are by marker; exc/inh are broad E/I.
    base_defs = {
        "exc": nodes[~nodes["is_inhibitory"]],
        "inh": nodes[nodes["is_inhibitory"]],
        "pv": nodes[nodes["cell_marker"] == "Pvalb"],
        "sst": nodes[nodes["cell_marker"] == "Sst"],
        "vip": nodes[nodes["cell_marker"] == "Vip"],
    }

    results: list[dict] = []
    for base, base_df in base_defs.items():
        # Only consider nodes where we can assign a layer label (for fair per-layer selection).
        base_df = base_df[base_df["layer"].notna()].copy()
        if base_df.empty:
            continue

        for region, mask in (("core", base_df["core"]), ("periphery", ~base_df["core"])):
            subset = base_df[mask].copy()
            if subset.empty:
                continue

            print(f"\nSelecting {base} {region}: n={len(subset)} (per-layer top/bottom third)")
            high_ids, low_ids, stats = _select_high_low_by_layer(subset, syn_count)
            for rec in stats:
                results.append({"base": base, "region": region, **rec})

            files_to_write = {
                _node_set_filename(base, "high", region): sorted(set(high_ids)),
                _node_set_filename(base, "low", region): sorted(set(low_ids)),
            }
            for fname, node_ids in files_to_write.items():
                out_path = node_set_dir / fname
                with open(out_path, "w") as f:
                    json.dump({"population": "v1", "node_id": node_ids}, f, indent=2)
                print(f"  → Wrote {out_path} ({len(node_ids)} cells)")

            # Convenience aliases (core-only) mirroring legacy naming style (but with synapsecount tag).
            if region == "core":
                if base in {"pv", "sst", "vip"}:
                    legacy_high = node_set_dir / f"{base}_high_outgoing_synapsecount_nodes.json"
                    legacy_low = node_set_dir / f"{base}_low_outgoing_synapsecount_nodes.json"
                else:
                    legacy_high = node_set_dir / f"high_outgoing_synapsecount_{base}_nodes.json"
                    legacy_low = node_set_dir / f"low_outgoing_synapsecount_{base}_nodes.json"
                with open(legacy_high, "w") as f:
                    json.dump({"population": "v1", "node_id": sorted(set(high_ids))}, f, indent=2)
                with open(legacy_low, "w") as f:
                    json.dump({"population": "v1", "node_id": sorted(set(low_ids))}, f, indent=2)
                print(f"  → Wrote {legacy_high.name} / {legacy_low.name} (core convenience aliases)")

    if results:
        print("\n" + "=" * 80)
        print("SUMMARY (per-layer selection diagnostics)")
        print("=" * 80)
        df = pd.DataFrame(results).sort_values(["base", "region", "layer"])
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()


