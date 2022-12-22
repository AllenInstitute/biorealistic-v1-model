# %%
# check the LGN rates to estimate inputs
import h5py
import numpy as np
from sonata.circuit import File
import pandas as pd

f = h5py.File("small/filternet/rates.h5", "r")


times = np.array(f["firing_rates/lgn/times"])
node_id = np.array(f["firing_rates/lgn/node_id"])
rates = np.array(f["firing_rates/lgn/firing_rates_Hz"])

f.close()

# %% get the LGN cell identity

d = "small/network/"
dfiles = [d + "lgn_nodes.h5", d + "v1_nodes.h5", d + "lgn_v1_edges.h5"]
dtfiles = [
    d + "lgn_node_types.csv",
    d + "v1_node_types.csv",
    d + "lgn_v1_edge_types.csv",
]


net = File(dfiles, dtfiles)

lgndf = net.nodes["lgn"].to_dataframe()


core_types = ["sON_", "tOFF_", ""]

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
        f"Mean Spont Rate: {avg_spont_rate:0.2f}, Mean Stim Rate: {avg_stim_rate:0.2f}"
    )


# %% Let's also check the generated spikes to see if they are consistent with the rates
spikes = pd.read_csv("small/filternet/spikes.csv", sep=" ")

spont_spikes = spikes[spikes["timestamps"] < 500]
evoked_spikes = spikes[spikes["timestamps"] > 1000]

len(spont_spikes) * 2 / 17400  # gave 3.83
len(evoked_spikes) / 2 / 17400  # gave 6.15
# OK. rates seem to be correct.