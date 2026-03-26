# %% Calculate response correlation between neurons.

# %% import libraries
import network_utils as nu
import numpy as np
from pandarallel import pandarallel
import polars as pl
import argparse

# %% Argument parser for optional base_dir and input type
parser = argparse.ArgumentParser(
    description="Calculate response correlation between neurons."
)
parser.add_argument(
    "--base_dir", type=str, default="core_nll_0", help="Base directory for data"
)
parser.add_argument(
    "--input_type",
    type=str,
    default="adjusted",
    help="Type of input data",
)


# %% Function to calculate edge_df_core
def calculate_edge_df_core(base_dir):
    edge_lf = nu.load_edges_pl(base_dir)  # polars lazy frame
    node_lf = nu.load_nodes_pl(base_dir, core_radius=200)  # polars lazy frame

    cores = node_lf.select("core").collect().to_series()
    source_ids = edge_lf.select("source_id").collect().to_series()
    target_ids = edge_lf.select("target_id").collect().to_series()

    source_in_core = cores[source_ids]
    target_in_core = cores[target_ids]
    both_in_core = source_in_core & target_in_core
    both_in_core.sum()

    edge_df_core = edge_lf.filter(both_in_core).collect().to_pandas()
    return edge_df_core


# %% Calculate edge_df_core
if __name__ == "__main__":
    args = parser.parse_args()

    base_dir = args.base_dir
    input_type = args.input_type

    pandarallel.initialize(progress_bar=True)

    edge_df_core = calculate_edge_df_core(base_dir)

    activity_path = f"{base_dir}/metrics/stim_spikes_output_imagenet_{input_type}.npz"

    # if input_type == "checkpoint":
    #     activity_path = f"{base_dir}/metrics/stim_spikes_output_imagenet_checkpoint.npz"
    # else:
    #     activity_path = f"{base_dir}/metrics/stim_spikes_output_imagenet.npz"

    activity = np.load(activity_path)["arr_0"]

    #  Calculate correlation coefficient for each pair of source_id and target_id
    def calculate_correlation(row):
        source_id = row["source_id"]
        target_id = row["target_id"]
        source_activity = activity[source_id]
        target_activity = activity[target_id]
        correlation = np.corrcoef(source_activity, target_activity)[0, 1]
        return correlation

    correlations = edge_df_core.parallel_apply(calculate_correlation, axis=1)

    #  store the response
    correlations_array = correlations.to_numpy()

    # Save the results to numpy files
    np.save(
        f"{base_dir}/metrics/response_correlations_{input_type}.npy", correlations_array
    )
