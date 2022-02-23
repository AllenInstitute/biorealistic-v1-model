# %%
#
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as pltticker
import numpy as np
from sonata.circuit import File
import h5py

d = "miniature/"
dfiles = [d + "lgn_nodes.h5", d + "v1_nodes.h5", d + "lgn_v1_edges.h5"]
dtfiles = [
    d + "lgn_node_types.csv",
    d + "v1_node_types.csv",
    d + "lgn_v1_edge_types.csv",
]


net = File(dfiles, dtfiles)

lgndf = net.nodes["lgn"].to_dataframe()

# %% read the rates

filternet_dir = "miniature_filternet/"
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
# %%
