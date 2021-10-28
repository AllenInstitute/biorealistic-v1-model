# %%
# check the LGN rates to estimate inputs
import h5py
import numpy as np
from sonata.circuit import File

f = h5py.File("miniature_filternet/rates.h5", "r")


times = np.array(f["firing_rates/lgn/times"])
node_id = np.array(f["firing_rates/lgn/node_id"])
rates = np.array(f["firing_rates/lgn/firing_rates_Hz"])

f.close()

# %% get the LGN cell identity

d = "miniature/"
dfiles = [d + "lgn_nodes.h5", d + "v1_nodes.h5", d + "lgn_v1_edges.h5"]
dtfiles = [
    d + "lgn_node_types.csv",
    d + "v1_node_types.csv",
    d + "lgn_v1_edge_types.csv",
]


net = File(dfiles, dtfiles)

lgndf = net.nodes["lgn"].to_dataframe()


core_types = ["sON_", "tOFF_"]

for typename in core_types:
    type = lgndf.pop_name.str.contains(typename)
    spont_time = times < 0.5
    stim_time = times > 1.0
    rate_type = rates[spont_time, :][:, type]
    avg_spont_rate = rate_type.mean()
    rate_type = rates[stim_time, :][:, type]
    avg_stim_rate = rate_type.mean()
    print(f"{typename}")
    print(
        f"Mean Spont Rate: {avg_spont_rate:0.1f}, Mean Stim Rate: {avg_stim_rate:0.1f}"
    )

