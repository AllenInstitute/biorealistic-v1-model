# %%
from sonata.circuit import File
import h5py
import numpy as np
import pandas as pd
import sys

# identify the v1 nodes.
# basedir = "core"

basedir = sys.argv[1]


node_file = basedir + "/network/v1_nodes.h5"
node_types = basedir + "/network/v1_node_types.csv"
# make a sonata network.
net = File(data_files=node_file, data_type_files=node_types)
node_df = net.nodes["v1"].to_dataframe()

modulation_df = pd.read_csv(basedir + "/metrics/modulation.csv", sep=" ")
modulation_df_e = modulation_df.query("src_substring == 'e'")

# open the edge file manually.
edge_file = basedir + "/network/v1_v1_edges.h5"


print("Reading the origianal edge file...")
with h5py.File(edge_file, "r") as f:
    sources_id = f["/edges/v1_to_v1/source_node_id"][:]
    target_id = f["/edges/v1_to_v1/target_node_id"][:]
    weights = f["/edges/v1_to_v1/0/syn_weight"][:]

print("Calculating the modulation factor...")
# basically, target_id and weights are all I need.
all_pops = node_df["pop_name"].unique()
# for each pop_name, set the value
pop_name_to_value = {}
for name in all_pops:
    for row in modulation_df_e.itertuples():
        if row.trg_substring in name:
            pop_name_to_value[name] = row.value
            break

# make a map in the node_df
target_id_modulation = node_df["pop_name"].map(pop_name_to_value).to_numpy()
mod_factor_full = target_id_modulation[target_id]

# determine the inhibitory connections.
inh_conn = weights < 0

# for inhibitory connections, invert the modulation.
mod_factor_full = np.where(inh_conn, 1 / mod_factor_full, mod_factor_full)

# calculate the modified weights.
mod_weights = weights * mod_factor_full

# save the mod weights to the new edge file.
new_file_name = basedir + "/network/v1_v1_edges_adjusted.h5"

# first, copy the original file.
import shutil

print("Copying the original edge file...")
shutil.copy(edge_file, new_file_name)

# then, open and overwrite the syn_weight.
print("Overwriting the new syn_weight...")
with h5py.File(new_file_name, "r+") as f:
    f["/edges/v1_to_v1/0/syn_weight"][:] = mod_weights

print("Done.")
