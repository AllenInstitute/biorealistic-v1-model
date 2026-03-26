#!/usr/bin/env python3
"""
Plot cell-type suppression effects following the existing pattern.
Compares % delta rate vs % delta OSI/DSI, organized by cell type.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py

BASE_DIR = Path("core_nll_0")
METRICS_DIR = BASE_DIR / "metrics"
NETWORK_DIR = BASE_DIR / "network"
OUTPUT_DIR = BASE_DIR / "figures" / "celltype_suppression"

# Load baseline (bio_trained)
baseline_df = pd.read_csv(METRICS_DIR / "OSI_DSI_DF_bio_trained.csv", sep=" ")

# Load cell-type suppression experiments
experiments = {
    "pv_high": "PV High",
    "pv_low": "PV Low",
    "sst_high": "SST High",
    "sst_low": "SST Low",
    "vip_high": "VIP High",
    "vip_low": "VIP Low",
}

# Load node types to get cell type labels
with h5py.File(NETWORK_DIR / "v1_nodes.h5", "r") as f:
    node_ids = f["nodes"]["v1"]["node_id"][:]
    node_type_ids = f["nodes"]["v1"]["node_type_id"][:]

node_types_df = pd.read_csv(NETWORK_DIR / "v1_node_types.csv", delimiter=" ")
node_info = pd.DataFrame({"node_id": node_ids, "node_type_id": node_type_ids})
node_info = node_info.merge(node_types_df[["node_type_id", "pop_name", "ei"]], on="node_type_id")

# Calculate deltas for each experiment
results = []

for exp_key, exp_label in experiments.items():
    exp_df = pd.read_csv(METRICS_DIR / f"OSI_DSI_DF_{exp_key}_neg1000.csv", sep=" ")

    # Merge with baseline
    merged = baseline_df.merge(exp_df, on="node_id", suffixes=("_baseline", "_exp"))
    merged = merged.merge(node_info[["node_id", "pop_name", "ei"]], on="node_id")

    # Calculate percent deltas (clip extreme values)
    # Only calculate percentage for neurons with reasonable baseline values
    merged["pct_delta_rate"] = np.where(
        merged["Ave_Rate(Hz)_baseline"] > 0.5,
        100 * (merged["Ave_Rate(Hz)_exp"] - merged["Ave_Rate(Hz)_baseline"]) / merged["Ave_Rate(Hz)_baseline"],
        np.nan
    )
    merged["pct_delta_osi"] = np.where(
        merged["OSI_baseline"] > 0.05,
        100 * (merged["OSI_exp"] - merged["OSI_baseline"]) / merged["OSI_baseline"],
        np.nan
    )
    merged["pct_delta_dsi"] = np.where(
        merged["DSI_baseline"] > 0.05,
        100 * (merged["DSI_exp"] - merged["DSI_baseline"]) / merged["DSI_baseline"],
        np.nan
    )

    merged["experiment"] = exp_key
    merged["exp_label"] = exp_label

    results.append(merged)

combined_df = pd.concat(results, ignore_index=True)

# Filter to only excitatory neurons
exc_df = combined_df[combined_df["ei"] == "e"].copy()

# Define cell type groups for plotting
cell_type_groups = {
    "L2/3 Exc": ["e23Cux2", "e23Rorb"],
    "L4 Exc": ["e4Cux2", "e4Rorb", "e4Scnn1a"],
    "L5 Exc": ["e5Rbp4"],
    "L5 ET": ["e5Chrna6"],
    "L5 IT": ["e5Tlx3"],
    "L6 Exc": ["e6Ntsr1"],
    "L2/3 PV": ["i23Pvalb"],
    "L4 PV": ["i4Pvalb"],
    "L5 PV": ["i5Pvalb"],
    "L6 PV": ["i6Pvalb"],
    "L2/3 SST": ["i23Sst"],
    "L4 SST": ["i4Sst"],
    "L5 SST": ["i5Sst"],
    "L6 SST": ["i6Sst"],
    "L2/3 VIP": ["i23Vip"],
    "L4 VIP": ["i4Vip"],
    "L5 VIP": ["i5Vip"],
    "L6 VIP": ["i6Vip"],
    "L1 Inh": ["i1Htr3a"],
}

# Add grouped labels
def assign_group(pop_name):
    for group_name, pop_list in cell_type_groups.items():
        if pop_name in pop_list:
            return group_name
    return "Other"

exc_df["cell_group"] = exc_df["pop_name"].apply(assign_group)

# =============================================================================
# Figure 1: % Delta Rate vs % Delta DSI (like the reference figure)
# =============================================================================

# Get unique cell groups (only those with data)
cell_groups = sorted([g for g in exc_df["cell_group"].unique() if g != "Other"])

n_groups = len(cell_groups)
n_cols = 5
n_rows = int(np.ceil(n_groups / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_groups == 1 else axes

for idx, cell_group in enumerate(cell_groups):
    ax = axes[idx]

    group_data = exc_df[exc_df["cell_group"] == cell_group]

    # Get baseline FR and DSI for this group
    baseline_fr = group_data["Ave_Rate(Hz)_baseline"].mean()
    baseline_dsi = group_data["DSI_baseline"].mean()

    # Plot each experiment
    colors = {"pv_high": "C3", "pv_low": "C3", "sst_high": "C0",
              "sst_low": "C0", "vip_high": "C2", "vip_low": "C2"}
    markers = {"pv_high": "o", "pv_low": "x", "sst_high": "o",
               "sst_low": "x", "vip_high": "o", "vip_low": "x"}

    for exp_key in experiments.keys():
        exp_data = group_data[group_data["experiment"] == exp_key]

        if len(exp_data) > 0:
            mean_pct_rate = exp_data["pct_delta_rate"].mean()
            mean_pct_dsi = exp_data["pct_delta_dsi"].mean()

            ax.scatter(mean_pct_rate, mean_pct_dsi,
                      c=colors[exp_key], marker=markers[exp_key],
                      s=100, alpha=0.8, edgecolors='k', linewidth=0.5)

    # Draw connecting lines between high/low pairs
    for cell_type in ["pv", "sst", "vip"]:
        high_data = group_data[group_data["experiment"] == f"{cell_type}_high"]
        low_data = group_data[group_data["experiment"] == f"{cell_type}_low"]

        if len(high_data) > 0 and len(low_data) > 0:
            high_x = high_data["pct_delta_rate"].mean()
            high_y = high_data["pct_delta_dsi"].mean()
            low_x = low_data["pct_delta_rate"].mean()
            low_y = low_data["pct_delta_dsi"].mean()

            ax.plot([high_x, low_x], [high_y, low_y],
                   c=colors[f"{cell_type}_high"], alpha=0.3, linewidth=1)

    ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel("% Δ rate", fontsize=8)
    ax.set_ylabel("% Δ DSI", fontsize=8)
    ax.set_title(f"{cell_group} (FR={baseline_fr:.1f} Hz, DSI={baseline_dsi:.2f})",
                fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-50, 150)
    ax.set_ylim(-50, 50)

# Hide unused subplots
for idx in range(n_groups, len(axes)):
    axes[idx].set_visible(False)

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='C3', markersize=8, label='PV High', markeredgecolor='k'),
    plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='C3', markersize=8, label='PV Low', markeredgecolor='k'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='C0', markersize=8, label='SST High', markeredgecolor='k'),
    plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='C0', markersize=8, label='SST Low', markeredgecolor='k'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='C2', markersize=8, label='VIP High', markeredgecolor='k'),
    plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='C2', markersize=8, label='VIP Low', markeredgecolor='k'),
]
fig.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=3)

plt.suptitle("Cell-Type Specific Suppression: FR vs DSI Changes (Excitatory Neurons)",
            fontsize=12, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "celltype_suppression_fr_vs_dsi_pct.png", dpi=300, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "celltype_suppression_fr_vs_dsi_pct.svg", bbox_inches="tight")
print(f"✓ Saved FR vs DSI plots")

# =============================================================================
# Figure 2: % Delta Rate vs % Delta OSI
# =============================================================================

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_groups == 1 else axes

for idx, cell_group in enumerate(cell_groups):
    ax = axes[idx]

    group_data = exc_df[exc_df["cell_group"] == cell_group]

    baseline_fr = group_data["Ave_Rate(Hz)_baseline"].mean()
    baseline_osi = group_data["OSI_baseline"].mean()

    for exp_key in experiments.keys():
        exp_data = group_data[group_data["experiment"] == exp_key]

        if len(exp_data) > 0:
            mean_pct_rate = exp_data["pct_delta_rate"].mean()
            mean_pct_osi = exp_data["pct_delta_osi"].mean()

            ax.scatter(mean_pct_rate, mean_pct_osi,
                      c=colors[exp_key], marker=markers[exp_key],
                      s=100, alpha=0.8, edgecolors='k', linewidth=0.5)

    # Draw connecting lines
    for cell_type in ["pv", "sst", "vip"]:
        high_data = group_data[group_data["experiment"] == f"{cell_type}_high"]
        low_data = group_data[group_data["experiment"] == f"{cell_type}_low"]

        if len(high_data) > 0 and len(low_data) > 0:
            high_x = high_data["pct_delta_rate"].mean()
            high_y = high_data["pct_delta_osi"].mean()
            low_x = low_data["pct_delta_rate"].mean()
            low_y = low_data["pct_delta_osi"].mean()

            ax.plot([high_x, low_x], [high_y, low_y],
                   c=colors[f"{cell_type}_high"], alpha=0.3, linewidth=1)

    ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel("% Δ rate", fontsize=8)
    ax.set_ylabel("% Δ OSI", fontsize=8)
    ax.set_title(f"{cell_group} (FR={baseline_fr:.1f} Hz, OSI={baseline_osi:.2f})",
                fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-50, 150)
    ax.set_ylim(-50, 50)

for idx in range(n_groups, len(axes)):
    axes[idx].set_visible(False)

fig.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=3)

plt.suptitle("Cell-Type Specific Suppression: FR vs OSI Changes (Excitatory Neurons)",
            fontsize=12, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "celltype_suppression_fr_vs_osi_pct.png", dpi=300, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "celltype_suppression_fr_vs_osi_pct.svg", bbox_inches="tight")
print(f"✓ Saved FR vs OSI plots")

# Save summary CSV (use nanmean to handle filtered values)
summary_by_group = exc_df.groupby(["cell_group", "experiment"]).agg({
    "pct_delta_rate": lambda x: np.nanmean(x),
    "pct_delta_osi": lambda x: np.nanmean(x),
    "pct_delta_dsi": lambda x: np.nanmean(x),
    "Ave_Rate(Hz)_baseline": "mean",
    "OSI_baseline": "mean",
    "DSI_baseline": "mean",
}).reset_index()

summary_by_group.to_csv(OUTPUT_DIR / "celltype_suppression_summary_by_group.csv", index=False)
print(f"✓ Saved summary CSV")

print("\nAll figures saved to:", OUTPUT_DIR)
