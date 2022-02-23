# %% code for calculating the total amount of inputs to each of V1 cells from the LGN

# open up the two files

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as pltticker
import numpy as np
from sonata.circuit import File
import h5py
import sys

# basedir = "miniature"
basedir = sys.argv[1]
# d = "original_mini/"
dnet = basedir + "/network/"
dfiles = [dnet + "lgn_nodes.h5", dnet + "v1_nodes.h5", dnet + "lgn_v1_edges.h5"]
dtfiles = [
    dnet + "lgn_node_types.csv",
    dnet + "v1_node_types.csv",
    dnet + "lgn_v1_edge_types.csv",
]

# %%
# %%time

"""
# This is a slow method that uses pysonata. It takes ~1s for 10 v1 neurons
# for 100 neurons, it was 12s, so ~2000s for 17000 neurons
net = File(dfiles, dtfiles)
v1df = net.nodes["v1"].to_dataframe()
lgndf = net.nodes["lgn"].to_dataframe()
inputsums = []

for n in range(10):
    sum = 0
    for i in net.edges["lgn_to_v1"].get_target(n):
        sum += i["syn_weight"] * i["nsyns"]
    inputsums.append(sum)
inputsums
"""

# %% use h5py directly instead?
# %%
# %%time
with h5py.File(dfiles[2], "r") as f:
    target_id = np.array(f["edges/lgn_to_v1/target_node_id"])
    edge_type_id = np.array(f["edges/lgn_to_v1/edge_type_id"])
    nsyns = np.array(f["edges/lgn_to_v1/0/nsyns"])

edge_type_df = pd.read_csv(dtfiles[2], sep=" ", index_col=1)
allweights = edge_type_df["syn_weight"].loc[edge_type_id]

df = pd.DataFrame({"target_id": target_id, "nsyns": nsyns})

df["weight_sum"] = df["nsyns"] * np.array(allweights)
total_nsyns_lgn = df.groupby("target_id")["nsyns"].sum()
total_weights_lgn = df.groupby("target_id")["weight_sum"].sum()

# the entire thing is ~720ms, >1e3 speed-up
# I checked that the first elements are the same as the previous method

# %% let's write this into the node file

savedic = {}
savedic["nodes/v1/0/total_input_nsyns_lgn"] = total_nsyns_lgn
savedic["nodes/v1/0/total_input_weights_lgn"] = total_weights_lgn
savedic["nodes/v1/0/nsyns_correction_lgn"] = total_nsyns_lgn / np.mean(total_nsyns_lgn)
savedic["nodes/v1/0/weights_correction_lgn"] = total_weights_lgn / np.mean(
    total_weights_lgn
)


# %% dictionary test
# a = {}
# a["dog"] = "bowwow"
# a["cat"] = "meow"

# for key in a:
#     print(key)

# %%


# TODO: currently, it does not overwirte existing variables in the h5 file.
#       It'll be nice to check if they exist, and create/overwrite appropriately.
with h5py.File(dfiles[1], "r+") as f:
    for key in savedic:
        if key in f:
            del f[key]
        f[key] = np.double(savedic[key])
        # f["nodes/v1/0/total_input_nsyns_lgn"] = total_nsyns_lgn
        # f["nodes/v1/0/total_input_weights_lgn"] = total_weights_lgn
        # # del f["nodes/v1/0/nsyns_correction_lgn"]
        # # del f["nodes/v1/0/weights_correction_lgn"]
        # f["nodes/v1/0/nsyns_correction_lgn"] = total_nsyns_lgn / np.mean(
        #     total_nsyns_lgn
        # )
        # f["nodes/v1/0/weights_correction_lgn"] = total_weights_lgn / np.mean(
        #     total_weights_lgn
        # )
    # del f["nodes/v1/total_input_nsyns_lgn"]
    # del f["nodes/v1/total_input_weights_lgn"]
    # del f["nodes/v1/0/total_input_weights_v1"]

# %%
# f = h5py.File(dfiles[1], "r+")
# f["/nodes/v1/0/nsyns_correction_lgn"]

# key = list(savedic.keys())[0]
# f.require_dataset(key, savedic[key].shape, np.double)
# v = f[key]
# v = np.double(savedic[key])

# key in f

# f.close()


# %%
"""
f = h5py.File(dfiles[1], "r+")

list(f['nodes/v1/0'])
del f['nodes/v1/0/total_input_weights_v1']
f.close()
"""
