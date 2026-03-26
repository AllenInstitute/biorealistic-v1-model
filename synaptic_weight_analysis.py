#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze synaptic weight distributions and their extremums across 19 cell types.
This script evaluates basic statistics (mean, std, min, max) for synaptic weights
before and after training, and visualizes them as 19x19 matrix plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import network_utils as nu
import polars as pl
from pathlib import Path
import argparse


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze synaptic weight distributions for a given network directory."
    )
    parser.add_argument(
        "base_dir",
        type=str,
        help='The base directory containing the network data (e.g., "core_nll_1").',
    )
    return parser.parse_args()


# --- Global Setup based on args ---
args = parse_args()
BASE_DIR = args.base_dir
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")

# Create directories if they don't exist
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Define plot subdirectories
PLOT_SUBDIRS = {
    "mean": "mean_weights",
    "std": "std_weights",
    "min": "min_weights",
    "max": "max_weights",
    "count": "connection_counts",
    "extremum": "max_extremums",
    "comparison_mean": "comparison_mean",
    "comparison_max": "comparison_max",
}

# Create plot subdirectories
for subdir in PLOT_SUBDIRS.values():
    os.makedirs(os.path.join(FIGURE_DIR, subdir), exist_ok=True)


# Define conditions (file suffixes)
conditions = {
    "original": "",
    "checkpoint": "_checkpoint",
    "noweightloss": "_noweightloss",
    "adjusted": "_adjusted",  # Adding the adjusted condition
}


def calculate_stats_polars(edges_df, nodes_df, cell_type_df, metric="syn_weight"):
    """Calculate statistics for the weights matrix using polars operations."""
    # Create a pandas series of unique cell types in the original order from cell_type_df
    unique_cell_types_pd = cell_type_df["cell_type"].drop_duplicates()
    # Create a mapping from cell type to its original index for sorting
    cell_type_order_map = {name: i for i, name in enumerate(unique_cell_types_pd)}
    # Get the unique cell types as a numpy array, sorted by their original appearance
    cell_types = unique_cell_types_pd.to_numpy()

    print(
        f"Using {len(cell_types)} unique cell types for analysis in their original order."
    )

    # Get layer information corresponding to the unique cell types
    layer_info = cell_type_df.drop_duplicates(subset=["cell_type"])
    layer_info = layer_info.set_index("cell_type")["layer"].reindex(cell_types)

    # Make sure we're working with unique cell type names (19 types)
    # cell_type_df is a pandas DataFrame, so cell_type_df["cell_type"] is a pandas Series
    # .unique() on a pandas Series returns a numpy array

    # Create a dictionary mapping from pop_name to cell_type
    pop_name_map = {}
    for idx, cell_type in zip(cell_type_df.index, cell_type_df["cell_type"]):
        pop_name_map[idx] = cell_type

    # Add cell_type column to nodes based on pop_name
    # First collect the nodes dataframe
    nodes_collected = nodes_df.collect()

    # Create a mapping function for pop_name to cell_type
    def map_pop_to_cell(pop_name):
        return pop_name_map.get(pop_name, pop_name)

    # Add cell_type column to nodes
    nodes_with_cell_type = nodes_collected.with_columns(
        [pl.col("pop_name").map_elements(map_pop_to_cell).alias("cell_type")]
    )

    # Join edges with source and target information
    # Collect edges here to make the join operations more reliable
    print("Collecting edges dataframe...")
    edges_collected = edges_df.collect()
    print(f"Processing {edges_collected.height} synaptic connections...")

    print("Joining with source node information...")
    source_info = edges_collected.join(
        nodes_with_cell_type.select(["node_id", "cell_type"]),
        left_on="source_id",
        right_on="node_id",
    )

    print("Joining with target node information...")
    all_info = source_info.join(
        nodes_with_cell_type.select(["node_id", "cell_type"]).rename(
            {"cell_type": "cell_type_target"}
        ),
        left_on="target_id",
        right_on="node_id",
    )

    # Use absolute values for weights
    all_info = all_info.with_columns(pl.col(metric).abs().alias(metric))

    # Initialize matrices
    n = len(cell_types)
    mean_matrix = np.zeros((n, n)) * np.nan
    std_matrix = np.zeros((n, n)) * np.nan
    min_matrix = np.zeros((n, n)) * np.nan
    max_matrix = np.zeros((n, n)) * np.nan
    count_matrix = np.zeros((n, n), dtype=int)

    print(f"Computing statistics across {n}x{n} cell type pairs...")

    # Calculate statistics for each source-target cell type combination
    for i, src in enumerate(cell_types):
        for j, tgt in enumerate(cell_types):
            # Filter for this source-target cell type pair
            subset = all_info.filter(
                (pl.col("cell_type") == src) & (pl.col("cell_type_target") == tgt)
            )

            # Skip empty combinations
            if subset.height == 0:
                continue

            # Calculate statistics
            stats = subset.select(
                [
                    pl.col(metric).mean().alias("mean"),
                    pl.col(metric).std().alias("std"),
                    pl.col(metric).min().alias("min"),
                    pl.col(metric).max().alias("max"),
                    pl.count().alias("count"),
                ]
            )

            # Store in matrices
            if stats.height > 0:
                mean_matrix[i, j] = stats.item(0, "mean")
                std_matrix[i, j] = stats.item(0, "std")
                min_matrix[i, j] = stats.item(0, "min")
                max_matrix[i, j] = stats.item(0, "max")
                count_matrix[i, j] = stats.item(0, "count")

    print("Statistics computation complete.")

    return {
        "mean": mean_matrix,
        "std": std_matrix,
        "min": min_matrix,
        "max": max_matrix,
        "count": count_matrix,
        "cell_types": cell_types,  # Now sorted by original appearance
        "layer_info": layer_info,  # Add layer info to the returned dict
    }


# --- Helper function for plotting ---
def _plot_heatmap_base(
    df,
    layer_info,
    title,
    filepath,
    annot,
    fmt,
    cmap,
    vmin=None,
    vmax=None,
    center=None,
    annot_kws=None,
    linewidths=0.5,
    linecolor="lightgray",
    highlight_cells=None,  # For extremum plot
):
    """Base function to create and save a heatmap plot with layer lines."""
    plt.figure(figsize=(17, 15))
    ax = sns.heatmap(
        df,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        square=True,
        annot_kws=annot_kws,
        linewidths=linewidths,
        linecolor=linecolor,
    )

    # Add Layer Lines
    layer_boundaries = np.where(layer_info.ne(layer_info.shift()))[0]
    for boundary in layer_boundaries:
        if boundary > 0:
            ax.axhline(boundary, color="white", lw=2.5)
            ax.axvline(boundary, color="white", lw=2.5)

    # Highlight specific cells (for extremum plot)
    if highlight_cells:
        for pos in highlight_cells:
            ax.add_patch(
                plt.Rectangle((pos[1], pos[0]), 1, 1, fill=False, edgecolor="red", lw=2)
            )

    plt.title(title, fontsize=16)
    plt.xlabel("Target Cell Type", fontsize=13)
    plt.ylabel("Source Cell Type", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_facecolor("white")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(filepath, dpi=300)
    plt.close()


# --- Refactored Plotting Functions ---


def plot_matrix(
    matrix,
    cell_types,
    layer_info,
    title,
    filename,
    subdir,
    cmap="viridis",
    annot=True,
    fmt=".1e",
    vmin=None,
    vmax=None,
    annot_kws=None,  # Pass annot_kws down
):
    """Plot a matrix of values for cell type pairs using the base plotter."""
    if annot_kws is None:
        annot_kws = {"size": 8}
    df = pd.DataFrame(matrix, index=cell_types, columns=cell_types)
    filepath = os.path.join(FIGURE_DIR, subdir, filename)
    _plot_heatmap_base(
        df,
        layer_info,
        title,
        filepath,
        annot,
        fmt,
        cmap,
        vmin,
        vmax,
        annot_kws=annot_kws,
    )


def plot_comparison_matrix(
    matrix1,
    matrix2,
    cell_types,
    layer_info,
    title,
    filename,
    subdir,
    cmap="coolwarm",
    fmt=".1f",
    annot_kws=None,  # Pass annot_kws down
):
    """Plot the percentage change between two matrices using the base plotter."""
    if annot_kws is None:
        annot_kws = {"size": 8}
    # Calculate percentage change
    with np.errstate(divide="ignore", invalid="ignore"):  # type: ignore
        diff_matrix = np.divide(matrix2 - matrix1, matrix1) * 100
        diff_matrix[np.isinf(diff_matrix)] = np.nan

    df = pd.DataFrame(diff_matrix, index=cell_types, columns=cell_types)
    abs_max = np.nanmax(np.abs(diff_matrix))
    if abs_max == 0 or np.isnan(abs_max):
        abs_max = 10  # Default range

    # Create annotation strings manually
    annot_df = df.applymap(lambda x: f"{x:{fmt}}%" if pd.notna(x) else "")

    filepath = os.path.join(FIGURE_DIR, subdir, filename)
    _plot_heatmap_base(
        df,
        layer_info,
        title,
        filepath,
        annot=annot_df,  # Pass the formatted annotations
        fmt="",  # Pass empty fmt as annotations are pre-formatted
        cmap=cmap,
        vmin=-abs_max,
        vmax=abs_max,
        center=0,
        annot_kws=annot_kws,
    )


def plot_extremum_matrix(
    max_matrix,
    cell_types,
    layer_info,
    title,
    filename,
    subdir,
    annot_kws=None,  # Pass annot_kws down
):
    """Create a special plot highlighting the maximum values using the base plotter."""
    if annot_kws is None:
        annot_kws = {"size": 8}
    df = pd.DataFrame(max_matrix, index=cell_types, columns=cell_types)

    # Get positions of maximum values
    max_positions = []
    for j in range(df.shape[1]):
        col_values = df.iloc[:, j].values
        if np.all(np.isnan(col_values)):
            continue
        col_max = np.nanmax(col_values)
        if np.isnan(col_max):
            continue
        max_rows = np.where(col_values == col_max)[0]
        for row in max_rows:
            max_positions.append((row, j))

    filepath = os.path.join(FIGURE_DIR, subdir, filename)
    _plot_heatmap_base(
        df,
        layer_info,
        title,
        filepath,
        annot=True,
        fmt=".1e",
        cmap="viridis",
        annot_kws=annot_kws,
        highlight_cells=max_positions,  # Pass highlight info
    )


def main():
    """Main function to analyze synaptic weights."""
    print(
        f"Analyzing synaptic weight distributions for {BASE_DIR}..."
    )  # Use BASE_DIR from args

    # Get cell types and layer info
    cell_type_df = nu.get_cell_type_table()
    unique_cell_types_pd = cell_type_df["cell_type"].drop_duplicates()
    cell_types_ordered = unique_cell_types_pd.to_numpy()
    layer_info_ordered = cell_type_df.drop_duplicates(subset=["cell_type"])
    layer_info_ordered = layer_info_ordered.set_index("cell_type")["layer"].reindex(
        cell_types_ordered
    )
    print(f"Found {len(cell_types_ordered)} unique cell types.")

    # Load nodes (needed for calculation if metrics don't exist)
    nodes = nu.load_nodes_pl(BASE_DIR)

    # Process each condition
    edge_data = {}
    stats_to_plot = ["mean", "std", "min", "max", "count"]
    stat_titles = {
        "mean": "Mean Synaptic Weights",
        "std": "Std Dev Synaptic Weights",
        "min": "Min Synaptic Weights",
        "max": "Max Synaptic Weights",
        "count": "Number of Connections",
    }
    stat_fmts = {"count": "d", "default": ".1e"}
    stat_cmaps = {"count": "YlGnBu", "default": "viridis"}
    stat_annot_kws = {"count": {"size": 6}, "default": {"size": 8}}

    for condition_name, suffix in conditions.items():
        print(f"--- Processing condition: {condition_name} ---")

        # --- Check if metrics already exist ---
        metrics_exist = True
        stats = {
            "cell_types": cell_types_ordered,
            "layer_info": layer_info_ordered,
        }  # Pre-populate
        for stat_name in stats_to_plot:
            matrix_filename = os.path.join(
                METRICS_DIR, f"{condition_name}_{stat_name}.npy"
            )
            if os.path.exists(matrix_filename):
                print(f"  Loading existing metric: {matrix_filename}")
                try:
                    stats[stat_name] = np.load(matrix_filename)
                except Exception as e:
                    print(f"  Error loading {matrix_filename}: {e}. Will recalculate.")
                    metrics_exist = False
                    stats = {
                        "cell_types": cell_types_ordered,
                        "layer_info": layer_info_ordered,
                    }  # Reset stats
                    break  # Stop checking if one file fails to load
            else:
                print(f"  Metric file not found: {matrix_filename}. Will calculate.")
                metrics_exist = False
                stats = {
                    "cell_types": cell_types_ordered,
                    "layer_info": layer_info_ordered,
                }  # Reset stats
                break  # Stop checking if one file is missing

        # --- Calculate if metrics don't exist ---
        if not metrics_exist:
            print(f"Calculating metrics for {condition_name}...")
            try:
                # Load edges
                print(f"Loading edges for {condition_name}...")
                edges = nu.load_edges_pl(BASE_DIR, "v1", "v1", suffix)
                print(f"Loaded edges for {condition_name}. Processing statistics...")

                # Calculate statistics
                calculated_stats = calculate_stats_polars(edges, nodes, cell_type_df)
                # Update stats dict, ensuring cell_types and layer_info are consistent
                stats.update(calculated_stats)
                # Re-assign just in case calculate_stats_polars modifies them
                stats["cell_types"] = cell_types_ordered
                stats["layer_info"] = layer_info_ordered

                print(f"Successfully processed statistics for {condition_name}.")

                # Save Matrices
                print(f"Saving calculated matrices for {condition_name}...")
                for stat_name, matrix in stats.items():
                    if stat_name not in ["cell_types", "layer_info"]:
                        matrix_filename = os.path.join(
                            METRICS_DIR, f"{condition_name}_{stat_name}.npy"
                        )
                        np.save(matrix_filename, matrix)  # Overwrite if recalculating
                        print(f"  Saved {matrix_filename}")

            except Exception as e:
                print(f"Error processing {condition_name}: {e}")
                import traceback

                traceback.print_exc()
                print(
                    f"Skipping plotting for {condition_name} due to calculation error."
                )
                continue  # Skip plotting for this condition if calculation failed
        else:
            print(f"Loaded all metrics for {condition_name} from existing files.")

        # Store the loaded/calculated stats
        edge_data[condition_name] = stats
        current_cell_types = stats["cell_types"]
        current_layer_info = stats["layer_info"]

        # --- Plotting (always happens if stats are available) ---
        print(f"Generating plots for {condition_name}...")
        for stat_name in stats_to_plot:
            plot_matrix(
                stats[stat_name],
                current_cell_types,
                current_layer_info,
                f"{stat_titles[stat_name]} - {condition_name}",
                f"{condition_name}_{stat_name}.png",
                PLOT_SUBDIRS[stat_name],
                cmap=stat_cmaps.get(stat_name, stat_cmaps["default"]),
                fmt=stat_fmts.get(stat_name, stat_fmts["default"]),  # type: ignore
                annot_kws=stat_annot_kws.get(stat_name, stat_annot_kws["default"]),  # type: ignore
            )

        # Plot extremum matrix
        plot_extremum_matrix(
            stats["max"],
            current_cell_types,
            current_layer_info,
            f"Maximum Synaptic Weights - {condition_name} (highlighted = column maximum)",
            f"{condition_name}_max_extremums.png",
            PLOT_SUBDIRS["extremum"],
        )
        print(f"Finished plots for {condition_name}.")

    # --- Compare conditions ---
    # ... (comparison plotting remains the same) ...
    print("--- Comparing conditions ---")
    comparisons = [
        ("checkpoint", "Checkpoint vs Original"),
        ("noweightloss", "NoWeightLoss vs Original"),
        ("adjusted", "Adjusted vs Original"),
    ]
    stats_to_compare = ["mean", "max"]

    if "original" in edge_data:
        comp_cell_types = edge_data["original"]["cell_types"]
        comp_layer_info = edge_data["original"]["layer_info"]

        for condition_name, title_suffix in comparisons:
            if condition_name in edge_data:
                print(f"Comparing original and {condition_name} conditions...")
                for stat_name in stats_to_compare:
                    # Check if both matrices needed for comparison exist
                    if (
                        stat_name in edge_data["original"]
                        and stat_name in edge_data[condition_name]
                    ):
                        plot_comparison_matrix(
                            edge_data["original"][stat_name],
                            edge_data[condition_name][stat_name],
                            comp_cell_types,
                            comp_layer_info,
                            f"Percentage Change in {stat_titles[stat_name]} ({title_suffix})",
                            f"change_{stat_name}_{condition_name}_vs_original_percent.png",
                            PLOT_SUBDIRS[f"comparison_{stat_name}"],
                        )
                    else:
                        print(
                            f"  Skipping comparison for {stat_name} - missing data for 'original' or '{condition_name}'."
                        )
            else:
                print(
                    f"Skipping comparison for {condition_name} - data not loaded/calculated."
                )
    else:
        print("Skipping comparisons - 'original' condition data not available.")

    print(f"Analysis complete. Results saved to {FIGURE_DIR} and {METRICS_DIR}")


if __name__ == "__main__":
    main()
