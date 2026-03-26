# %% Calculate distances between neurons.

# %% import libraries
import network_utils as nu
import numpy as np
from pandarallel import pandarallel
import polars as pl
import argparse

pandarallel.initialize(progress_bar=True)

# %% Argument parser for optional base_dir
parser = argparse.ArgumentParser(description="Calculate distances between neurons.")
parser.add_argument(
    "--base_dir", type=str, default="core_nll_0", help="Base directory for data"
)
args = parser.parse_args()

base_dir = args.base_dir

# %% load the connectivity data and filter out for the core neuron.
edge_lf = nu.load_edges_pl(base_dir)  # polars lazy frame
node_lf = nu.load_nodes_pl(base_dir, core_radius=200)  # polars lazy frame

# %% filter edge_lf for the core neurons.
cores = node_lf.select("core").collect().to_series()
source_ids = edge_lf.select("source_id").collect().to_series()
target_ids = edge_lf.select("target_id").collect().to_series()

source_in_core = cores[source_ids]
target_in_core = cores[target_ids]
both_in_core = source_in_core & target_in_core
both_in_core.sum()

# make a sub frame with core neurons.
edge_df_core = edge_lf.filter(both_in_core).collect().to_pandas()

# %% Calculate the lateral and euclidean distances between the cells.
pos_x = node_lf.select("x").collect().to_series()
pos_z = node_lf.select("z").collect().to_series()


def calculate_distance(row):
    source_id = row["source_id"]
    target_id = row["target_id"]
    dx = pos_x[source_id] - pos_x[target_id]
    dz = pos_z[source_id] - pos_z[target_id]
    lateral_distance = np.sqrt(dx**2 + dz**2)
    return lateral_distance


distances = edge_df_core.parallel_apply(calculate_distance, axis=1)

# %% store the distances
distances_array = distances.to_numpy()

# Save the results to numpy files
np.save(f"{base_dir}/metrics/distances.npy", distances_array)
