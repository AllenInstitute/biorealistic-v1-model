#!/usr/bin/env python3
"""
Plot cell-type specific suppression effects.
Creates:
1. % delta rate vs % delta OSI scatter plots
2. % delta rate vs % delta DSI scatter plots
3. Effect size comparison to full inhibitory suppression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load metrics
df = pd.read_csv("analysis_shared/celltype_suppression_metrics.csv")

# Define experiment groups and colors
CELLTYPE_EXP = ["pv_high", "pv_low", "sst_high", "sst_low", "vip_high", "vip_low"]
FULL_INH_EXP = ["inh_high", "inh_low"]

COLORS = {
    "pv_high": "#E24A33",  # Red
    "pv_low": "#FFA07A",   # Light red
    "sst_high": "#348ABD", # Blue
    "sst_low": "#87CEEB",  # Light blue
    "vip_high": "#988ED5", # Purple
    "vip_low": "#D8BFD8",  # Light purple
    "inh_high": "#777777", # Gray
    "inh_low": "#AAAAAA",  # Light gray
}

LABELS = {
    "pv_high": "PV High",
    "pv_low": "PV Low",
    "sst_high": "SST High",
    "sst_low": "SST Low",
    "vip_high": "VIP High",
    "vip_low": "VIP Low",
    "inh_high": "All Inh High",
    "inh_low": "All Inh Low",
}

# ============================================================================
# Figure 1: % Delta Rate vs % Delta OSI/DSI
# ============================================================================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Cell-Type Specific Suppression Effects", fontsize=14, fontweight="bold")

# Plot each cell-type experiment
for idx, exp_name in enumerate(CELLTYPE_EXP):
    row = idx // 3
    col = idx % 3

    exp_data = df[df["experiment"] == exp_name]

    # OSI plot
    ax = axes[row, col]
    ax.scatter(exp_data["pct_delta_rate"], exp_data["pct_delta_osi"],
               alpha=0.3, s=10, color=COLORS[exp_name])

    # Add reference lines
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)

    # Calculate correlation
    valid_mask = np.isfinite(exp_data["pct_delta_rate"]) & np.isfinite(exp_data["pct_delta_osi"])
    if np.sum(valid_mask) > 10:
        r, p = stats.pearsonr(exp_data["pct_delta_rate"][valid_mask],
                             exp_data["pct_delta_osi"][valid_mask])
        ax.text(0.05, 0.95, f"r={r:.3f}\np={p:.3e}",
               transform=ax.transAxes, va="top", fontsize=8,
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_title(LABELS[exp_name], fontweight="bold")
    ax.set_xlabel("% Δ Rate")
    ax.set_ylabel("% Δ OSI")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-100, 100)
    ax.grid(True, alpha=0.3)

# DSI plots
for idx, exp_name in enumerate(CELLTYPE_EXP):
    row = idx // 3
    col = idx % 3 + 3  # Offset by 3 for DSI column

    if col >= 4:  # Wrap to second row
        row = 1
        col = col - 4

    exp_data = df[df["experiment"] == exp_name]

    ax = axes[row, col + 1] if row == 0 and col < 2 else axes[1, col - 2]
    ax.scatter(exp_data["pct_delta_rate"], exp_data["pct_delta_dsi"],
               alpha=0.3, s=10, color=COLORS[exp_name])

    ax.axhline(0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5, alpha=0.3)

    valid_mask = np.isfinite(exp_data["pct_delta_rate"]) & np.isfinite(exp_data["pct_delta_dsi"])
    if np.sum(valid_mask) > 10:
        r, p = stats.pearsonr(exp_data["pct_delta_rate"][valid_mask],
                             exp_data["pct_delta_dsi"][valid_mask])
        ax.text(0.05, 0.95, f"r={r:.3f}\np={p:.3e}",
               transform=ax.transAxes, va="top", fontsize=8,
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_title(LABELS[exp_name], fontweight="bold")
    ax.set_xlabel("% Δ Rate")
    ax.set_ylabel("% Δ DSI")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-100, 100)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("analysis_shared/celltype_suppression_scatter.png", dpi=300, bbox_inches="tight")
plt.savefig("analysis_shared/celltype_suppression_scatter.svg", bbox_inches="tight")
print("✓ Saved scatter plots")

# ============================================================================
# Figure 2: Effect Size Comparison
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Effect Sizes: Cell-Type Specific vs Full Inhibitory Suppression",
             fontsize=14, fontweight="bold")

# Calculate mean effects for each experiment
summary = []
for exp_name in list(CELLTYPE_EXP) + list(FULL_INH_EXP):
    exp_data = df[df["experiment"] == exp_name]
    summary.append({
        "experiment": exp_name,
        "label": LABELS[exp_name],
        "mean_delta_rate": exp_data["delta_rate"].mean(),
        "mean_pct_delta_rate": exp_data["pct_delta_rate"].mean(),
        "mean_delta_osi": exp_data["delta_osi"].mean(),
        "mean_pct_delta_osi": exp_data["pct_delta_osi"].mean(),
        "mean_delta_dsi": exp_data["delta_dsi"].mean(),
        "mean_pct_delta_dsi": exp_data["pct_delta_dsi"].mean(),
        "is_full_inh": exp_name in FULL_INH_EXP,
    })

summary_df = pd.DataFrame(summary)

# Panel A: Delta Rate
ax = axes[0]
x = np.arange(len(summary_df))
colors = [COLORS[exp] for exp in summary_df["experiment"]]
bars = ax.bar(x, summary_df["mean_delta_rate"], color=colors, alpha=0.8, edgecolor="k")

# Highlight full inhibitory
for i, is_full in enumerate(summary_df["is_full_inh"]):
    if is_full:
        bars[i].set_linewidth(3)

ax.axhline(0, color="k", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(summary_df["label"], rotation=45, ha="right")
ax.set_ylabel("Mean Δ Rate (Hz)")
ax.set_title("A. Firing Rate Change")
ax.grid(True, alpha=0.3, axis="y")

# Panel B: Delta OSI
ax = axes[1]
bars = ax.bar(x, summary_df["mean_delta_osi"], color=colors, alpha=0.8, edgecolor="k")
for i, is_full in enumerate(summary_df["is_full_inh"]):
    if is_full:
        bars[i].set_linewidth(3)

ax.axhline(0, color="k", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(summary_df["label"], rotation=45, ha="right")
ax.set_ylabel("Mean Δ OSI")
ax.set_title("B. OSI Change")
ax.grid(True, alpha=0.3, axis="y")

# Panel C: Delta DSI
ax = axes[2]
bars = ax.bar(x, summary_df["mean_delta_dsi"], color=colors, alpha=0.8, edgecolor="k")
for i, is_full in enumerate(summary_df["is_full_inh"]):
    if is_full:
        bars[i].set_linewidth(3)

ax.axhline(0, color="k", linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(summary_df["label"], rotation=45, ha="right")
ax.set_ylabel("Mean Δ DSI")
ax.set_title("C. DSI Change")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("analysis_shared/celltype_effect_size_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig("analysis_shared/celltype_effect_size_comparison.svg", bbox_inches="tight")
print("✓ Saved effect size comparison")

# ============================================================================
# Figure 3: Normalized Effect Sizes (relative to full inhibitory)
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Normalized Effect Sizes (Relative to Full Inhibitory Suppression)",
             fontsize=14, fontweight="bold")

# Get full inhibitory reference values
inh_high_effects = summary_df[summary_df["experiment"] == "inh_high"].iloc[0]
inh_low_effects = summary_df[summary_df["experiment"] == "inh_low"].iloc[0]

# Calculate normalized effects (cell-type / corresponding full inh)
norm_summary = []
for exp_name in CELLTYPE_EXP:
    exp_row = summary_df[summary_df["experiment"] == exp_name].iloc[0]

    # Match to high or low reference
    if "high" in exp_name:
        ref = inh_high_effects
    else:
        ref = inh_low_effects

    norm_summary.append({
        "experiment": exp_name,
        "label": LABELS[exp_name],
        "norm_delta_rate": exp_row["mean_delta_rate"] / ref["mean_delta_rate"] if ref["mean_delta_rate"] != 0 else 0,
        "norm_delta_osi": exp_row["mean_delta_osi"] / ref["mean_delta_osi"] if ref["mean_delta_osi"] != 0 else 0,
        "norm_delta_dsi": exp_row["mean_delta_dsi"] / ref["mean_delta_dsi"] if ref["mean_delta_dsi"] != 0 else 0,
    })

norm_df = pd.DataFrame(norm_summary)

# Plot normalized effects
x = np.arange(len(norm_df))
colors = [COLORS[exp] for exp in norm_df["experiment"]]

for ax, metric, title in zip(axes,
                             ["norm_delta_rate", "norm_delta_osi", "norm_delta_dsi"],
                             ["A. Firing Rate", "B. OSI", "C. DSI"]):
    ax.bar(x, norm_df[metric], color=colors, alpha=0.8, edgecolor="k")
    ax.axhline(1.0, color="r", linestyle="--", linewidth=2, label="Full Inh Suppression")
    ax.set_xticks(x)
    ax.set_xticklabels(norm_df["label"], rotation=45, ha="right")
    ax.set_ylabel("Normalized Effect Size")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

plt.tight_layout()
plt.savefig("analysis_shared/celltype_normalized_effects.png", dpi=300, bbox_inches="tight")
plt.savefig("analysis_shared/celltype_normalized_effects.svg", bbox_inches="tight")
print("✓ Saved normalized effects")

# ============================================================================
# Print Summary Statistics
# ============================================================================

print("\n" + "=" * 80)
print("EFFECT SIZE SUMMARY")
print("=" * 80)

print("\nCell-Type Specific Suppression:")
print(summary_df[~summary_df["is_full_inh"]][["label", "mean_delta_rate", "mean_delta_osi", "mean_delta_dsi"]].to_string(index=False))

print("\nFull Inhibitory Suppression (Reference):")
print(summary_df[summary_df["is_full_inh"]][["label", "mean_delta_rate", "mean_delta_osi", "mean_delta_dsi"]].to_string(index=False))

print("\nNormalized Effects (Cell-Type / Full Inh):")
print(norm_df[["label", "norm_delta_rate", "norm_delta_osi", "norm_delta_dsi"]].to_string(index=False))

print("\n✓ All figures saved to analysis_shared/")
