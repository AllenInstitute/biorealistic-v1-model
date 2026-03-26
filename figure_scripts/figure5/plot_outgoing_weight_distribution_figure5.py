#!/usr/bin/env python3
"""Distribution of outgoing weights for L2/3 Exc cells (Figure 5)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_DIR = PROJECT_ROOT / "core_nll_0"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "paper" / "figure5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_FILE = BASE_DIR / "network" / "v1_v1_edges_bio_trained.h5"
FEATURES_PATH = (
    PROJECT_ROOT / "cell_categorization" / "core_nll_0_neuron_features.parquet"
)
NODE_SETS_DIR = BASE_DIR / "node_sets"


def load_edges_sum_by_source() -> pd.Series:
    """Load edge data and sum weights by source node."""
    print(f"Loading edges from {EDGE_FILE}...")
    with h5py.File(EDGE_FILE, "r") as f:
        grp = f["edges"]["v1_to_v1"]
        sources = grp["source_node_id"][:].astype(np.int64)
        weights = np.abs(grp["0"]["syn_weight"][:].astype(np.float64))

    print("Summing outgoing weights...")
    # Use bincount for fast summation
    max_id = sources.max()
    sums = np.bincount(sources, weights=weights, minlength=max_id + 1)
    return pd.Series(sums, name="outgoing_weight")


def load_node_set(filename: str) -> set[int]:
    path = NODE_SETS_DIR / filename
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    return {int(x) for x in data["node_id"]}


def main() -> None:
    # 1. Load outgoing weights
    outgoing_weights = load_edges_sum_by_source()

    # 2. Load features to filter for L2/3 Exc
    print("Loading features...")
    features = pd.read_parquet(FEATURES_PATH)
    # Ensure unique node_id
    features = features.drop_duplicates(subset="node_id").set_index("node_id")

    # Filter for L2/3 Exc
    target_type = "L2/3_Exc"
    if target_type not in features["cell_type"].unique():
        target_type = "L2/3 Exc"

    l23_exc_nodes = features[features["cell_type"] == target_type].index
    print(f"Found {len(l23_exc_nodes)} {target_type} neurons")

    # 3. Get weights for these neurons
    weights = outgoing_weights.reindex(l23_exc_nodes).fillna(0.0)

    # 4. Identify cohorts
    high_ids = load_node_set("high_outgoing_exc_core_nodes.json")
    low_ids = load_node_set("low_outgoing_exc_core_nodes.json")

    # Filter to core
    if "radius" in features.columns:
        core_radius = 200.0
        is_core = features.loc[l23_exc_nodes, "radius"] <= core_radius
        weights = weights[is_core]
        print(f"Filtered to {len(weights)} core neurons")

    # Determine thresholds
    # Assuming cohorts are defined by weight thresholds
    low_weights = weights[weights.index.isin(low_ids)]
    high_weights = weights[weights.index.isin(high_ids)]

    # Thresholds: max of low group, min of high group
    # (Note: Mid group is between)
    thresh_low_mid = low_weights.max()
    thresh_mid_high = high_weights.min()

    print(f"Low/Mid Threshold: {thresh_low_mid:.4f}")
    print(f"Mid/High Threshold: {thresh_mid_high:.4f}")

    # 5. Plot with log scale and colored bins
    plt.figure(figsize=(3.5, 2.5))

    # Create log-spaced bins
    # Handle zero weights if any (log(0) is -inf)
    # If there are zero weights, we might want to plot them separately or start bins from a small epsilon
    # But weights are usually > 0 for connected neurons
    min_w = weights[weights > 0].min()
    max_w = weights.max()

    # Extend slightly for nicer plot
    bins = np.logspace(np.log10(min_w), np.log10(max_w), 100)

    counts, bin_edges = np.histogram(weights, bins=bins)

    # Plot bars individually to color them
    colors = []
    width = np.diff(bin_edges)
    centers = bin_edges[:-1] + width / 2

    # Match cohort colors used in boxplots: low=blue, mid=gray, high=red
    palette = {"Low": "#2b7bba", "Mid": "#888888", "High": "#c73635"}

    for center in centers:
        if center <= thresh_low_mid:
            colors.append(palette["Low"])
        elif center >= thresh_mid_high:
            colors.append(palette["High"])
        else:
            colors.append(palette["Mid"])

    plt.bar(
        bin_edges[:-1],
        counts,
        width=width,
        color=colors,
        align="edge",
        edgecolor="none",
    )

    # Add threshold lines
    plt.axvline(thresh_low_mid, color="black", linestyle="--", linewidth=1, alpha=0.7)
    plt.axvline(thresh_mid_high, color="black", linestyle="--", linewidth=1, alpha=0.7)

    plt.xscale("log")
    plt.xlabel("Total Outgoing Synaptic Weight (pA)")
    plt.ylabel("Count")

    # Force sparse labeled ticks at 3e3, 5e3, 1e4; keep minor tick marks elsewhere
    ax = plt.gca()
    xtick_vals = [3e3, 5e3, 1e4]
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(["3×10³", "5×10³", "10⁴"])
    # Add unlabeled minor ticks for additional log positions
    from matplotlib.ticker import LogLocator, NullFormatter

    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(1, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    # remove spine
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Custom legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=palette["Low"], label="Low"),
        Patch(facecolor=palette["Mid"], label="Mid"),
        Patch(facecolor=palette["High"], label="High"),
    ]
    plt.legend(handles=legend_elements, title="Cohort", loc="upper right")

    # Remove title as requested
    # plt.title(f"Outgoing Weight Distribution ({target_type})")

    output_path = OUTPUT_DIR / "outgoing_weight_distribution_L23Exc.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"))
    print(f"Saved {output_path} and PDF")


if __name__ == "__main__":
    main()
