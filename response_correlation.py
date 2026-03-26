# %% Calculate response correlation between neurons.

# %% import libraries
import network_utils as nu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from response_correlation_calculations import calculate_edge_df_core
import time  # Add import for time measurement

# %% Load the pre-calculated data
base_dir = "core_nll_0"
input_type = "checkpoint"  # or "plain"
correlations = np.load(f"{base_dir}/metrics/correlations_{input_type}.npy")
distances = np.load(f"{base_dir}/metrics/distances.npy")

# Calculate edge_df_core using the function from response_correlation_calculations.py
edge_df_core = calculate_edge_df_core(base_dir)

# Add the correlations and distances to the DataFrame
edge_df_core["Response Correlation"] = correlations
edge_df_core["Lateral Distance"] = distances

# %% I'd like to make this figure for each cell type pair.
ctdf = nu.get_cell_type_table()
node_lf = nu.load_nodes_pl(base_dir, core_radius=200)  # polars lazy frame
node_df = node_lf.collect().to_pandas()

edge_df_core["source_type"] = ctdf["cell_type"][
    node_df["pop_name"][edge_df_core["source_id"]]
].values
edge_df_core["target_type"] = ctdf["cell_type"][
    node_df["pop_name"][edge_df_core["target_id"]]
].values

# Group the DataFrame by source_type and target_type
grouped_edge_df_core = edge_df_core.groupby(["source_type", "target_type"])


# %% Function to plot scatter and block average for each cell type pair
def plot_cell_type_pair(
    edge_typed_df,
    source_type,
    target_type,
    ax,
    show_xlabel,
    show_ylabel,
):
    start_time = time.time()  # Start time measurement

    # Scatter plot
    # ax.scatter(
    #     edge_typed_df["Response Correlation"],
    #     edge_typed_df["syn_weight"],
    #     alpha=0.02,
    #     s=1,
    # )
    if show_xlabel:
        ax.set_xlabel("Res. corr.")
    if show_ylabel:
        ax.set_ylabel(source_type, fontsize=8)

    # Block average plot
    block_size = 0.03
    bins = np.arange(-1, 1 + block_size, block_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate block averages
    calc_start_time = time.time()  # Start time for block average calculation
    block_means = []
    block_sems = []
    for i in range(len(bins) - 1):
        bin_mask = (edge_typed_df["Response Correlation"] >= bins[i]) & (
            edge_typed_df["Response Correlation"] < bins[i + 1]
        )
        block_mean = edge_typed_df[bin_mask]["syn_weight"].mean()
        block_means.append(block_mean)
        block_sem = edge_typed_df[bin_mask]["syn_weight"].sem()
        block_sems.append(block_sem)
    calc_end_time = time.time()  # End time for block average calculation
    # print(
    #     f"Time taken for block average calculation: {calc_end_time - calc_start_time:.2f} seconds"
    # )

    # Plot block averages
    plot_start_time = time.time()  # Start time for plotting
    ax.errorbar(
        bin_centers,
        block_means,
        yerr=block_sems,
        fmt="o",
        color="blue",
        ecolor="green",
        markersize=3,
        capsize=3,
    )
    plot_end_time = time.time()  # End time for plotting
    # print(f"Time taken for plotting: {plot_end_time - plot_start_time:.2f} seconds")

    # Set x-axis ticks
    ticks = np.arange(-0.4, 0.7, 0.4)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:.1f}" for tick in ticks])

    # draw a faint line at x=0
    vl = ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    # set zorder to make sure the line is behind the data points
    vl.set_zorder(0)

    # Calculate and display correlation coefficient with significance level
    corr_start_time = time.time()  # Start time for correlation calculation
    correlation_coef = edge_typed_df["Response Correlation"].corr(
        edge_typed_df["syn_weight"]
    )
    from scipy.stats import pearsonr

    correlation_coef, p_value = pearsonr(
        edge_typed_df["Response Correlation"], edge_typed_df["syn_weight"]
    )
    significance = ""
    if p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    ax.text(
        0.05,
        0.95,
        f"r={correlation_coef:.2f}{significance}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
    )
    corr_end_time = time.time()  # End time for correlation calculation
    # print(
    #     f"Time taken for correlation calculation: {corr_end_time - corr_start_time:.2f} seconds"
    # )

    end_time = time.time()  # End time measurement
    # print(
    #     f"Total time taken for {source_type} to {target_type}: {end_time - start_time:.2f} seconds"
    # )


# %% Get unique cell types
source_types = sorted(edge_df_core["source_type"].unique())[0:]
target_types = sorted(edge_df_core["target_type"].unique())[0:]

# %% Create subplots for all cell type pairs
from tqdm import tqdm

ls = len(source_types)
lt = len(target_types)

fig, axes = plt.subplots(ls, lt, figsize=(lt * 2, ls * 2), sharex=False, sharey=False)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

for i, source_type in tqdm(enumerate(source_types), total=len(source_types)):
    for j, target_type in enumerate(target_types):
        show_xlabel = i == len(source_types) - 1
        show_ylabel = j == 0
        try:
            edge_typed_df = grouped_edge_df_core.get_group((source_type, target_type))
            plot_cell_type_pair(
                edge_typed_df,
                source_type,
                target_type,
                axes[i, j],
                show_xlabel,
                show_ylabel,
            )
        except KeyError:
            axes[i, j].axis("off")  # Turn off the axis if the group does not exist
        if not show_xlabel:
            axes[i, j].set_xticklabels([])
        if not show_ylabel:
            axes[i, j].set_yticklabels([])
        if i == 0:
            axes[i, j].set_title(target_type)

# plt.tight_layout()
# plt.show()

# %%
