# %%
#
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as pltticker
import numpy as np
from sonata.circuit import File
import h5py

basedir = "core"
d = f"{basedir}/network/"
dfiles = [d + "lgn_nodes.h5", d + "v1_nodes.h5", d + "lgn_v1_edges.h5"]
dtfiles = [
    d + "lgn_node_types.csv",
    d + "v1_node_types.csv",
    d + "lgn_v1_edge_types.csv",
]


net = File(dfiles, dtfiles)

lgndf = net.nodes["lgn"].to_dataframe()

# %% read the rates

filternet_dir = f"{basedir}/filternet/"
ratefile = "rates.h5"

f = h5py.File(filternet_dir + ratefile, "r")

rates = np.array(f["/firing_rates/lgn/firing_rates_Hz"][1500, :])
lgndf["rate1500"] = rates


tOFF = lgndf["pop_name"].str.contains("tOFF_TF4")  # most numerous tOFF
sON = lgndf["pop_name"].str.contains("sON_TF8")  # most numerous tOFF
lgndf[tOFF].plot.scatter("x", "y", c="rate1500")
lgndf[sON].plot.scatter("x", "y", c="rate1500")


lgndf.value_counts("pop_name")

f.close()
# %% read the edges in a dataframe
# read the edges file from h5 file.

with h5py.File(dfiles[2], "r") as f:
    source_id = f["/edges/lgn_to_v1/source_node_id"][()]
    target_id = f["/edges/lgn_to_v1/target_node_id"][()]
    weights = f["/edges/lgn_to_v1/0/syn_weight"][()]

# make it into a dataframe
edges = pd.DataFrame(
    {"source_id": source_id, "target_id": target_id, "weight": weights}
)

# %% get some stats.
edges.value_counts("source_id")
edges.value_counts("target_id")

# semilog-x plot of the value counts histogram of the source_ids
np.log10(edges.value_counts("source_id")).plot.hist()
edges.value_counts("target_id").plot.hist(logx=True)
num_targets = edges.value_counts("source_id")
# add this to lgn table
lgndf["num_targets"] = num_targets

lgndf.groupby("node_type_id").mean()
