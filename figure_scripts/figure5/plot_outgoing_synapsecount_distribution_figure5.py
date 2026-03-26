#!/usr/bin/env python3
"""Histogram of outgoing synapse counts for L2/3 Exc cells (Figure 5, degree-style panel).

This mirrors `plot_outgoing_weight_distribution_figure5.py` but replaces
total outgoing |weight| with total outgoing synapse count (sum of n_syns_).

Cohort thresholds are inferred from the synapsecount-based node sets:
`high_outgoing_synapsecount_exc_core_nodes.json` and
`low_outgoing_synapsecount_exc_core_nodes.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_shared.style import apply_pub_style, trim_spines

BASE_DIR = PROJECT_ROOT / "core_nll_0"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "paper" / "figure5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_FILE = BASE_DIR / "network" / "v1_v1_edges_bio_trained.h5"
FEATURES_PATH = (
    PROJECT_ROOT / "cell_categorization" / "core_nll_0_neuron_features.parquet"
)
NODE_SETS_DIR = BASE_DIR / "node_sets"


def load_edges_synapsecount_by_source() -> pd.Series:
    with h5py.File(EDGE_FILE, "r") as f:
        grp = f["edges"]["v1_to_v1"]
        sources = grp["source_node_id"][:].astype(np.int64)
        if "n_syns_" in grp["0"]:
            n_syns = grp["0"]["n_syns_"][:].astype(np.float64)
        elif "nsyns" in grp["0"]:
            n_syns = grp["0"]["nsyns"][:].astype(np.float64)
        else:
            raise KeyError("Neither 'n_syns_' nor 'nsyns' found in edges/v1_to_v1/0")

    max_id = int(sources.max()) if sources.size else -1
    sums = np.bincount(sources, weights=n_syns, minlength=max_id + 1).astype(np.float64)
    return pd.Series(sums, name="outgoing_synapsecount")


def load_node_set(filename: str) -> set[int]:
    path = NODE_SETS_DIR / filename
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    return {int(x) for x in data.get("node_id", [])}


def main() -> None:
    apply_pub_style()

    # 1) Outgoing synapse-count per source
    syncount = load_edges_synapsecount_by_source()

    # 2) Load features to filter for L2/3 Exc
    features = pd.read_parquet(FEATURES_PATH)
    features = features.drop_duplicates(subset="node_id").set_index("node_id")

    target_type = "L2/3_Exc"
    if target_type not in features["cell_type"].unique():
        target_type = "L2/3 Exc"

    l23_exc_nodes = features[features["cell_type"] == target_type].index
    values = syncount.reindex(l23_exc_nodes).fillna(0.0)

    # Core filter (keep consistent with Fig 5)
    if "radius" in features.columns:
        values = values[features.loc[values.index, "radius"] <= 200.0]

    # 3) Cohort thresholds from synapsecount node sets (core)
    high_ids = load_node_set("high_outgoing_synapsecount_exc_core_nodes.json")
    low_ids = load_node_set("low_outgoing_synapsecount_exc_core_nodes.json")
    low_vals = values[values.index.isin(low_ids)]
    high_vals = values[values.index.isin(high_ids)]
    if low_vals.empty or high_vals.empty:
        raise RuntimeError(
            "Synapsecount cohort node sets not found or empty. "
            "Run analysis_shared/create_highlow_outgoing_synapsecount_nodesets.py first."
        )

    thresh_low_mid = float(low_vals.max())
    thresh_mid_high = float(high_vals.min())

    # 4) Plot histogram (log-x like the weight panel)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    positive = values[values > 0]
    if positive.empty:
        raise RuntimeError("No positive outgoing synapse counts found for L2/3 Exc.")

    min_v = float(positive.min())
    max_v = float(values.max())
    bins = np.logspace(np.log10(min_v), np.log10(max_v), 100)
    counts, bin_edges = np.histogram(values, bins=bins)

    palette = {"Low": "#2b7bba", "Mid": "#888888", "High": "#c73635"}
    width = np.diff(bin_edges)
    centers = bin_edges[:-1] + width / 2
    colors = []
    for c in centers:
        if c <= thresh_low_mid:
            colors.append(palette["Low"])
        elif c >= thresh_mid_high:
            colors.append(palette["High"])
        else:
            colors.append(palette["Mid"])

    ax.bar(
        bin_edges[:-1],
        counts,
        width=width,
        color=colors,
        align="edge",
        edgecolor="none",
    )
    ax.axvline(thresh_low_mid, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(thresh_mid_high, color="black", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Total outgoing synapse count (∑ n_syns)")
    ax.set_ylabel("Count")
    trim_spines(ax)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=palette["Low"], label="Low"),
            Patch(facecolor=palette["Mid"], label="Mid"),
            Patch(facecolor=palette["High"], label="High"),
        ],
        title="Cohort",
        loc="upper right",
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )

    out_png = OUTPUT_DIR / "outgoing_synapsecount_distribution_L23Exc.png"
    out_pdf = OUTPUT_DIR / "outgoing_synapsecount_distribution_L23Exc.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
