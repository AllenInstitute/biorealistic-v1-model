import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import network_utils as nu
import seaborn as sns
from matplotlib.patches import Rectangle

# Argument parser for base directory and input type
parser = argparse.ArgumentParser(
    description="Plot sparsity measurements for each cell type."
)
parser.add_argument(
    "--base_dirs",
    type=str,
    nargs="+",
    help="Base directories for data",
    default=["core_nll_1"],
)
parser.add_argument(
    "--input_types",
    type=str,
    nargs="+",
    choices=["checkpoint", "plain", "adjusted", "noweightloss"],
    default=["checkpoint", "plain", "adjusted", "noweightloss"],
    help="Types of input data to compare",
)
parser.add_argument(
    "--output_dir", type=str, default=".", help="Directory to save the plots"
)
parser.add_argument(
    "--radius", type=float, default=200.0, help="Radius for core selection"
)
parser.add_argument(
    "--data_labels",
    type=str,
    nargs="+",
    help="Custom labels for the input types (must match number of input_types)",
    default=None,
)
parser.add_argument(
    "--e_only", action="store_true", help="Plot only excitatory neurons"
)
parser.add_argument(
    "--combined_plot",
    action="store_true",
    help="Create a single combined plot for all input types",
    default=True,
)


def get_borders(ticklabel):
    """Extract layer borders from tick labels"""
    prev_layer = "1"
    borders = [-0.5]
    for i in ticklabel:
        x = i.get_position()[0]
        text = i.get_text()
        if text[1] != prev_layer:
            # detected layer change
            borders.append(x - 0.5)
            prev_layer = text[1]
    borders.append(x + 0.5)
    return borders


def draw_borders(ax, borders, ylim):
    """Draw shaded background for layers"""
    for i in range(0, len(borders) - 1, 2):
        w = borders[i + 1] - borders[i]
        h = ylim[1] - ylim[0]
        ax.add_patch(
            Rectangle((borders[i], ylim[0]), w, h, alpha=0.08, color="k", zorder=-10)
        )
    return ax


# Function to load sparsity data and create a dataframe
def load_sparsity_data(base_dirs, input_type, radius=200.0, data_labels=None):
    if data_labels is None:
        data_labels = [f"Dataset {i+1}" for i in range(len(base_dirs))]

    if len(data_labels) != len(base_dirs):
        raise ValueError("Number of data labels must match number of base directories")

    all_data = []

    for base_dir, label in zip(base_dirs, data_labels):
        # Load lifetime sparsity data
        lifetime_file = f"{base_dir}/metrics/lifetime_sparsity_{input_type}.npy"
        if not os.path.exists(lifetime_file):
            print(f"Warning: {lifetime_file} not found")
            continue

        lifetime_sparsity = np.load(lifetime_file)

        # Load node information
        v1_nodes = nu.load_nodes_pl(base_dir, core_radius=radius).collect().to_pandas()
        ctdf = nu.get_cell_type_table()
        v1_nodes["cell_type"] = ctdf.loc[v1_nodes["pop_name"]]["cell_type"].values
        v1_nodes["ei"] = ctdf.loc[v1_nodes["pop_name"]]["ei"].values

        # Create dataframe for this dataset
        df = pd.DataFrame(
            {
                "node_id": v1_nodes["node_id"],
                "cell_type": v1_nodes["cell_type"],
                "ei": v1_nodes["ei"],
                "lifetime_sparsity": lifetime_sparsity[v1_nodes["node_id"]],
                "data_type": label,
            }
        )

        # Try to load population sparsity if available
        pop_file = f"{base_dir}/metrics/population_sparsity_{input_type}.npy"
        if os.path.exists(pop_file):
            pop_sparsity = np.load(pop_file, allow_pickle=True).item()

            # Map population sparsity to each cell
            pop_sparsity_values = np.zeros(len(df))
            for cell_type, sparsity in pop_sparsity.items():
                pop_sparsity_values[df["cell_type"] == cell_type] = sparsity

            df["population_sparsity"] = pop_sparsity_values

        all_data.append(df)

    if not all_data:
        raise ValueError(f"No valid sparsity data found for input type: {input_type}")

    # Combine all datasets
    combined_df = pd.concat(
        all_data, ignore_index=True
    )  # Use ignore_index=True to avoid duplicate indices

    # Replace spaces in cell_type with underscores
    combined_df["cell_type"] = combined_df["cell_type"].str.replace(" ", "_")
    # Replace L1_Htr3a with L1_Inh if present
    combined_df["cell_type"] = combined_df["cell_type"].str.replace(
        "L1_Htr3a", "L1_Inh"
    )

    return combined_df


# Function to load sparsity data and create a dataframe for comparing input types
def load_sparsity_data_comparison(
    base_dir, input_types, radius=200.0, data_labels=None
):
    if data_labels is None:
        data_labels = [input_type.capitalize() for input_type in input_types]

    if len(data_labels) != len(input_types):
        raise ValueError("Number of data labels must match number of input types")

    all_data = []

    for input_type, label in zip(input_types, data_labels):
        # Load lifetime sparsity data
        lifetime_file = f"{base_dir}/metrics/lifetime_sparsity_{input_type}.npy"
        if not os.path.exists(lifetime_file):
            print(f"Warning: {lifetime_file} not found")
            continue

        lifetime_sparsity = np.load(lifetime_file)

        # Load node information - only needs to be done once per base_dir
        if not all_data:  # Do this only for the first input_type
            v1_nodes = (
                nu.load_nodes_pl(base_dir, core_radius=radius).collect().to_pandas()
            )
            ctdf = nu.get_cell_type_table()
            v1_nodes["cell_type"] = ctdf.loc[v1_nodes["pop_name"]]["cell_type"].values
            v1_nodes["ei"] = ctdf.loc[v1_nodes["pop_name"]]["ei"].values

        # Create dataframe for this dataset
        df = pd.DataFrame(
            {
                "node_id": v1_nodes["node_id"],
                "cell_type": v1_nodes["cell_type"],
                "ei": v1_nodes["ei"],
                "lifetime_sparsity": lifetime_sparsity[v1_nodes["node_id"]],
                "data_type": label,
            }
        )

        # Try to load population sparsity if available
        pop_file = f"{base_dir}/metrics/population_sparsity_{input_type}.npy"
        if os.path.exists(pop_file):
            pop_sparsity = np.load(pop_file, allow_pickle=True).item()

            # Map population sparsity to each cell
            pop_sparsity_values = np.zeros(len(df))
            for cell_type, sparsity in pop_sparsity.items():
                pop_sparsity_values[df["cell_type"] == cell_type] = sparsity

            df["population_sparsity"] = pop_sparsity_values

        all_data.append(df)

    if not all_data:
        raise ValueError(f"No valid sparsity data found for any input type")

    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)

    # Replace spaces in cell_type with underscores
    combined_df["cell_type"] = combined_df["cell_type"].str.replace(" ", "_")
    # Replace L1_Htr3a with L1_Inh if present
    combined_df["cell_type"] = combined_df["cell_type"].str.replace(
        "L1_Htr3a", "L1_Inh"
    )

    return combined_df


# Function to load sparsity data for multiple input types and create a combined dataframe
def load_combined_sparsity_data(base_dir, input_types, radius=200.0, data_labels=None):
    if data_labels is None:
        data_labels = [input_type.capitalize() for input_type in input_types]

    if len(data_labels) != len(input_types):
        raise ValueError("Number of data labels must match number of input types")

    all_data = []

    # Load node information - only needs to be done once
    v1_nodes = nu.load_nodes_pl(base_dir, core_radius=radius).collect().to_pandas()
    ctdf = nu.get_cell_type_table()
    v1_nodes["cell_type"] = ctdf.loc[v1_nodes["pop_name"]]["cell_type"].values
    v1_nodes["ei"] = ctdf.loc[v1_nodes["pop_name"]]["ei"].values

    for input_type, label in zip(input_types, data_labels):
        # Load lifetime sparsity data
        lifetime_file = f"{base_dir}/metrics/lifetime_sparsity_{input_type}.npy"
        if not os.path.exists(lifetime_file):
            print(f"Warning: {lifetime_file} not found")
            continue

        lifetime_sparsity = np.load(lifetime_file)

        # Create dataframe for this dataset
        df = pd.DataFrame(
            {
                "node_id": v1_nodes["node_id"],
                "cell_type": v1_nodes["cell_type"],
                "ei": v1_nodes["ei"],
                "lifetime_sparsity": lifetime_sparsity[v1_nodes["node_id"]],
                "data_type": label,
                "input_type": input_type,  # Store original input type for reference
            }
        )

        # Try to load population sparsity if available
        pop_file = f"{base_dir}/metrics/population_sparsity_{input_type}.npy"
        if os.path.exists(pop_file):
            pop_sparsity = np.load(pop_file, allow_pickle=True).item()

            # Map population sparsity to each cell
            pop_sparsity_values = np.zeros(len(df))
            for cell_type, sparsity in pop_sparsity.items():
                pop_sparsity_values[df["cell_type"] == cell_type] = sparsity

            df["population_sparsity"] = pop_sparsity_values

        all_data.append(df)

    if not all_data:
        raise ValueError(f"No valid sparsity data found for any input type")

    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)

    # Replace spaces in cell_type with underscores
    combined_df["cell_type"] = combined_df["cell_type"].str.replace(" ", "_")
    # Replace L1_Htr3a with L1_Inh if present
    combined_df["cell_type"] = combined_df["cell_type"].str.replace(
        "L1_Htr3a", "L1_Inh"
    )

    return combined_df


def plot_sparsity_boxplot(
    df, metric, output_dir, input_type, e_only=False, ylim=None, figsize=(12, 6)
):
    """Create a boxplot for sparsity metrics"""
    # Make a copy to avoid modifying the original dataframe
    plot_df = df.copy().reset_index(drop=True)

    if e_only:
        # Filter out inhibitory neurons and NaNs
        plot_df = plot_df[~plot_df["cell_type"].isna()]
        plot_df = plot_df[plot_df["ei"] == "e"]

    # Remove any NaN values that could cause plotting issues
    plot_df = plot_df.dropna(subset=["cell_type", metric])

    # Sort by cell type to ensure consistent ordering
    plot_df = plot_df.sort_values(by="cell_type")

    # Create color palette
    data_types = plot_df["data_type"].unique()
    color_pal = {dt: plt.cm.tab10(i) for i, dt in enumerate(data_types)}

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create boxplot
    sns.boxplot(
        x="cell_type",
        y=metric,
        hue="data_type",
        data=plot_df,
        ax=ax,
        width=0.7,
        palette=color_pal,
    )

    # Format plot
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(loc="upper right")

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # Set reasonable limits for sparsity metrics (0-1)
        ax.set_ylim([0, 1])

    ax.set_xlabel("")

    metric_title = metric.replace("_", " ").title()
    ax.set_ylabel(metric_title)
    ax.set_title(f"{metric_title} by Cell Type - {input_type.title()} Data")

    # Apply layer shadings
    ticklabel = ax.get_xticklabels()
    if ticklabel:  # Ensure there are tick labels
        borders = get_borders(ticklabel)
        draw_borders(ax, borders, ax.get_ylim())

    # Save figure
    population_str = "_excitatory" if e_only else ""
    output_file = os.path.join(output_dir, f"{metric}{population_str}_{input_type}.png")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return fig


def plot_combined_sparsity_boxplot(
    df, metrics, output_dir, e_only=False, figsize=(12, 10)
):
    """Create a combined boxplot for multiple sparsity metrics"""
    # Make a copy to avoid modifying the original dataframe
    plot_df = df.copy().reset_index(drop=True)

    if e_only:
        # Filter out inhibitory neurons and NaNs
        plot_df = plot_df[~plot_df["cell_type"].isna()]
        plot_df = plot_df[plot_df["ei"] == "e"]

    # Sort by cell type to ensure consistent ordering
    plot_df = plot_df.sort_values(by="cell_type")

    # Create color palette
    data_types = plot_df["data_type"].unique()
    color_pal = {dt: plt.cm.tab10(i) for i, dt in enumerate(data_types)}

    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    # Get network name from first row
    network_name = os.path.basename(args.base_dirs[0].rstrip("/"))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Remove any NaN values that could cause plotting issues
        plot_data = plot_df.dropna(subset=["cell_type", metric])

        # Create boxplot
        sns.boxplot(
            x="cell_type",
            y=metric,
            hue="data_type",
            data=plot_data,
            ax=ax,
            width=0.7,
            palette=color_pal,
        )

        # Format plot
        ax.tick_params(axis="x", labelrotation=45)
        if i < len(metrics) - 1:  # Only show legend in the last subplot
            ax.get_legend().remove()
        else:
            ax.legend(loc="upper right", bbox_to_anchor=(1.05, 1))

        # Set reasonable limits for sparsity metrics (0-1)
        ax.set_ylim([0, 1])

        ax.set_xlabel("")

        metric_title = metric.replace("_", " ").title()
        ax.set_ylabel(metric_title)

        if i == 0:  # Add title only on the first subplot
            ax.set_title(f"Sparsity Comparison for {network_name}")

        # Apply layer shadings
        ticklabel = ax.get_xticklabels()
        if ticklabel:  # Ensure there are tick labels
            borders = get_borders(ticklabel)
            draw_borders(ax, borders, ax.get_ylim())

    # Save figure
    population_str = "_excitatory" if e_only else ""
    output_file = os.path.join(
        output_dir, f"combined_sparsity{population_str}_{network_name}.png"
    )
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return fig


if __name__ == "__main__":
    args = parser.parse_args()

    try:
        # Create directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        if len(args.base_dirs) > 1:
            print(
                "Warning: Multiple base directories specified. Using first one only for combined plot."
            )

        base_dir = args.base_dirs[0]

        # Load combined data for different input types
        combined_df = load_combined_sparsity_data(
            base_dir,
            args.input_types,
            radius=args.radius,
            data_labels=args.data_labels,
        )

        # Determine available metrics
        available_metrics = []
        if "lifetime_sparsity" in combined_df.columns:
            available_metrics.append("lifetime_sparsity")
        if "population_sparsity" in combined_df.columns:
            available_metrics.append("population_sparsity")

        if available_metrics:
            # Create combined plot with all metrics as subplots
            plot_combined_sparsity_boxplot(
                combined_df,
                available_metrics,
                args.output_dir,
                e_only=args.e_only,
            )

            print(f"Combined plot saved to {args.output_dir}")
        else:
            print("No sparsity metrics found in the data.")

    except Exception as e:
        print(f"Error: {e}")
