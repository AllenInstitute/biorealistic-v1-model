#!/usr/bin/env python3
"""Analyze reciprocal connections and distance dependence of SST targeting."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "core_nll_0"
NODE_SET_DIR = BASE_DIR / "node_sets"
OUTPUT_DIR = BASE_DIR / "figures" / "selectivity_outgoing"

EDGE_FILE = BASE_DIR / "network" / "v1_v1_edges_bio_trained.h5"
NODE_FILE = BASE_DIR / "network" / "v1_nodes.h5"
NODE_TYPES_FILE = BASE_DIR / "network" / "v1_node_types.csv"


def load_node_attributes() -> pd.DataFrame:
    """Load node attributes including positions and cell types."""
    node_types = pd.read_csv(NODE_TYPES_FILE, sep=r"\s+")
    node_types["is_inhibitory"] = node_types["ei"].str.lower().str.startswith("i")
    node_types["cell_marker"] = node_types["pop_name"].str.extract(r"[ei]\d+(\w+)")
    type_lookup = node_types.set_index("node_type_id")[["pop_name", "is_inhibitory", "cell_marker"]]

    with h5py.File(NODE_FILE, "r") as f:
        node_ids = f["nodes"]["v1"]["node_id"][:]
        node_type_ids = f["nodes"]["v1"]["node_type_id"][:]
        x = f["nodes"]["v1"]["0"]["x"][:]
        y = f["nodes"]["v1"]["0"]["y"][:]
        z = f["nodes"]["v1"]["0"]["z"][:]

    df = pd.DataFrame({
        "node_type_id": node_type_ids,
        "x": x, "y": y, "z": z
    }, index=node_ids)
    df.index.name = "node_id"
    df = df.join(type_lookup, on="node_type_id", how="left")
    df["is_inhibitory"] = df["is_inhibitory"].astype(bool)

    return df


def load_node_set(filename: str) -> set[int]:
    """Load node set from JSON."""
    data = json.loads((NODE_SET_DIR / filename).read_text())
    return {int(x) for x in data["node_id"]}


def main() -> None:
    print("="*80)
    print("RECIPROCAL CONNECTIONS & DISTANCE DEPENDENCE ANALYSIS")
    print("="*80)

    print("\nLoading data...")
    node_attrs = load_node_attributes()

    # Load node sets
    inh_high = load_node_set("high_outgoing_inh_nodes.json")
    inh_low = load_node_set("low_outgoing_inh_nodes.json")

    # Get SST nodes
    sst_nodes = set(node_attrs[node_attrs["cell_marker"] == "Sst"].index)

    # Load edges
    with h5py.File(EDGE_FILE, "r") as f:
        grp = f["edges"]["v1_to_v1"]
        sources = grp["source_node_id"][:].astype(np.int64)
        targets = grp["target_node_id"][:].astype(np.int64)
        weights = np.abs(grp["0"]["syn_weight"][:].astype(np.float64))

    print(f"Loaded {len(sources):,} edges")

    # ========================================================================
    # 1. RECIPROCAL CONNECTIONS
    # ========================================================================

    print("\n1. RECIPROCAL CONNECTION ANALYSIS:")
    print("-"*80)

    # Check: Do high-weight inh preferentially target low-weight inh back?

    # Low → High
    mask_low_to_high = np.isin(sources, list(inh_low)) & np.isin(targets, list(inh_high))
    weight_low_to_high = weights[mask_low_to_high].sum()
    n_low_to_high = mask_low_to_high.sum()

    # High → Low
    mask_high_to_low = np.isin(sources, list(inh_high)) & np.isin(targets, list(inh_low))
    weight_high_to_low = weights[mask_high_to_low].sum()
    n_high_to_low = mask_high_to_low.sum()

    # For comparison: total outgoing from each
    mask_low_out = np.isin(sources, list(inh_low))
    mask_high_out = np.isin(sources, list(inh_high))

    total_weight_low = weights[mask_low_out].sum()
    total_weight_high = weights[mask_high_out].sum()

    frac_low_to_high = weight_low_to_high / total_weight_low
    frac_high_to_low = weight_high_to_low / total_weight_high

    print(f"  Low-weight → High-weight inhibitory:")
    print(f"    Connections: {n_low_to_high:,}")
    print(f"    Total weight: {weight_low_to_high:.1f}")
    print(f"    Fraction of low-weight output: {frac_low_to_high:.4f} ({100*frac_low_to_high:.2f}%)")

    print(f"\n  High-weight → Low-weight inhibitory:")
    print(f"    Connections: {n_high_to_low:,}")
    print(f"    Total weight: {weight_high_to_low:.1f}")
    print(f"    Fraction of high-weight output: {frac_high_to_low:.4f} ({100*frac_high_to_low:.2f}%)")

    reciprocity_ratio = frac_high_to_low / frac_low_to_high
    print(f"\n  Reciprocity ratio (high→low / low→high): {reciprocity_ratio:.2f}×")

    if reciprocity_ratio < 1:
        print(f"  → Connection is ASYMMETRIC: Low preferentially targets high {1/reciprocity_ratio:.2f}× more")
    else:
        print(f"  → Connection is RECIPROCAL: High also targets low")

    # ========================================================================
    # 2. DISTANCE DEPENDENCE OF SST TARGETING
    # ========================================================================

    print("\n2. DISTANCE DEPENDENCE OF SST TARGETING:")
    print("-"*80)

    # For low-weight inh → SST, calculate distance for each connection

    # Get connections from low-weight to SST
    mask_low_to_sst = np.isin(sources, list(inh_low)) & np.isin(targets, list(sst_nodes))
    src_ids_low_sst = sources[mask_low_to_sst]
    tgt_ids_low_sst = targets[mask_low_to_sst]
    weights_low_sst = weights[mask_low_to_sst]

    # Calculate distances (XZ space)
    src_coords = node_attrs.loc[src_ids_low_sst, ["x", "z"]].values
    tgt_coords = node_attrs.loc[tgt_ids_low_sst, ["x", "z"]].values
    distances_low_sst = np.sqrt(np.sum((src_coords - tgt_coords)**2, axis=1))

    # Same for high-weight inh → SST
    mask_high_to_sst = np.isin(sources, list(inh_high)) & np.isin(targets, list(sst_nodes))
    src_ids_high_sst = sources[mask_high_to_sst]
    tgt_ids_high_sst = targets[mask_high_to_sst]
    weights_high_sst = weights[mask_high_to_sst]

    src_coords_high = node_attrs.loc[src_ids_high_sst, ["x", "z"]].values
    tgt_coords_high = node_attrs.loc[tgt_ids_high_sst, ["x", "z"]].values
    distances_high_sst = np.sqrt(np.sum((src_coords_high - tgt_coords_high)**2, axis=1))

    print(f"\n  Low-weight → SST connections:")
    print(f"    Count: {len(distances_low_sst):,}")
    print(f"    Mean distance: {distances_low_sst.mean():.1f} µm")
    print(f"    Median distance: {np.median(distances_low_sst):.1f} µm")

    print(f"\n  High-weight → SST connections:")
    print(f"    Count: {len(distances_high_sst):,}")
    print(f"    Mean distance: {distances_high_sst.mean():.1f} µm")
    print(f"    Median distance: {np.median(distances_high_sst):.1f} µm")

    # Binned analysis
    bins = np.array([0, 50, 100, 150, 200, 300, 500])
    bin_labels = ['0-50', '50-100', '100-150', '150-200', '200-300', '300+']

    print(f"\n  Distance-binned enrichment:")
    print(f"  {'Distance (µm)':15s} {'Low→SST %':12s} {'High→SST %':12s} {'Enrichment':12s}")
    print("-"*60)

    for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
        # Low-weight
        mask_low_bin = (distances_low_sst >= bin_start) & (distances_low_sst < bin_end)
        weight_low_bin = weights_low_sst[mask_low_bin].sum()
        frac_low_bin = weight_low_bin / total_weight_low

        # High-weight
        mask_high_bin = (distances_high_sst >= bin_start) & (distances_high_sst < bin_end)
        weight_high_bin = weights_high_sst[mask_high_bin].sum()
        frac_high_bin = weight_high_bin / total_weight_high

        enrichment = frac_low_bin / frac_high_bin if frac_high_bin > 0 else 0

        print(f"  {bin_labels[i]:15s} {100*frac_low_bin:11.3f}% {100*frac_high_bin:11.3f}% {enrichment:11.2f}×")

    # ========================================================================
    # 3. VISUALIZATION
    # ========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Reciprocal connections
    ax = axes[0]

    groups = ['Low→High', 'High→Low']
    fractions = [frac_low_to_high, frac_high_to_low]
    colors = ['#e74c3c', '#3498db']

    bars = ax.bar(groups, [100*f for f in fractions], color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)

    for bar, frac in zip(bars, fractions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
               f'{100*frac:.2f}%', ha='center', va='bottom',
               fontsize=11, fontweight='bold')

    ax.set_ylabel('% of Total Outgoing Weight', fontsize=12, fontweight='bold')
    ax.set_title(f'A. Asymmetric Reciprocity\n(Low→High is {1/reciprocity_ratio:.2f}× stronger)',
                fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, max([100*f for f in fractions]) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Distance dependence
    ax = axes[1]

    # Plot enrichment vs distance
    bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
    enrichments = []

    for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
        mask_low_bin = (distances_low_sst >= bin_start) & (distances_low_sst < bin_end)
        weight_low_bin = weights_low_sst[mask_low_bin].sum()
        frac_low_bin = weight_low_bin / total_weight_low

        mask_high_bin = (distances_high_sst >= bin_start) & (distances_high_sst < bin_end)
        weight_high_bin = weights_high_sst[mask_high_bin].sum()
        frac_high_bin = weight_high_bin / total_weight_high

        enrichment = frac_low_bin / frac_high_bin if frac_high_bin > 0 else 0
        enrichments.append(enrichment)

    ax.plot(bin_centers, enrichments, 'o-', linewidth=2.5, markersize=10,
            color='#e67e22', markeredgecolor='black', markeredgewidth=1.5)

    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No preference')

    ax.set_xlabel('Distance (µm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SST Enrichment (Low/High)', fontsize=12, fontweight='bold')
    ax.set_title('B. Distance Dependence of SST Targeting',
                fontsize=13, fontweight='bold', pad=15)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    out_path = OUTPUT_DIR / 'reciprocal_and_distance_analysis.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n\nSaved figure to {out_path}")

    out_path_svg = OUTPUT_DIR / 'reciprocal_and_distance_analysis.svg'
    plt.savefig(out_path_svg, bbox_inches='tight')
    print(f"Saved vector version to {out_path_svg}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
