# %% Calculate sparsity measures for neurons.

# %% import libraries
import numpy as np
import pandas as pd
import network_utils as nu
import argparse

# %% Argument parser for optional base_dir and input type
parser = argparse.ArgumentParser(description="Calculate sparsity measures for neurons.")
parser.add_argument(
    "--base_dir", type=str, default="core_nll_0", help="Base directory for data"
)
parser.add_argument(
    "--input_type",
    type=str,
    choices=["checkpoint", "plain", "adjusted", "noweightloss"],
    default="checkpoint",
    help="Type of input data",
)


# %% Function to calculate lifetime sparsity
def calculate_lifetime_sparsity(activity):
    mean_activity = np.mean(activity, axis=1)
    squared_mean_activity = np.mean(activity**2, axis=1)
    sparsity = np.full(activity.shape[0], np.nan)
    valid_indices = mean_activity != 0
    sparsity[valid_indices] = (
        1 - (mean_activity[valid_indices] ** 2 / squared_mean_activity[valid_indices])
    ) / (1 - (1 / activity.shape[1]))
    return sparsity


# %% Function to calculate population sparsity for each cell type
def calculate_population_sparsity(activity, ctdf):
    sparsity_dict = {}
    for cell_type in ctdf["cell_type"].unique():
        cell_indices = ctdf[ctdf["cell_type"] == cell_type].index
        cell_activity = activity[cell_indices]
        mean_activity = np.mean(cell_activity, axis=(0, 1))
        squared_mean_activity = np.mean(cell_activity**2, axis=(0, 1))
        if mean_activity == 0:
            sparsity_dict[cell_type] = np.nan
        else:
            sparsity = (1 - (mean_activity**2 / squared_mean_activity)) / (
                1 - (1 / (cell_activity.shape[0] * cell_activity.shape[1]))
            )
            sparsity_dict[cell_type] = sparsity
    return sparsity_dict


# %% Calculate sparsity measures
if __name__ == "__main__":
    args = parser.parse_args()

    base_dir = args.base_dir
    input_type = args.input_type

    if input_type == "plain":
        activity_path = f"{base_dir}/metrics/stim_spikes_output_imagenet.npz"
    else:
        activity_path = (
            f"{base_dir}/metrics/stim_spikes_output_imagenet_{input_type}.npz"
        )

    activity = np.load(activity_path)["arr_0"]

    v1_nodes = nu.load_nodes_pl(base_dir, core_radius=200).collect().to_pandas()
    ctdf = nu.get_cell_type_table()
    v1_nodes["cell_type"] = ctdf.loc[v1_nodes["pop_name"]]["cell_type"].values

    lifetime_sparsity = calculate_lifetime_sparsity(activity)
    population_sparsity = calculate_population_sparsity(activity, v1_nodes)

    # Save the results to numpy files
    np.save(f"{base_dir}/metrics/lifetime_sparsity_{input_type}.npy", lifetime_sparsity)
    np.save(
        f"{base_dir}/metrics/population_sparsity_{input_type}.npy", population_sparsity
    )
