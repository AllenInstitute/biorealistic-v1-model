# %% investigate if all the OS rules are in place.
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt

import network_utils as nu
from importlib import reload

reload(nu)


# %% load the infos!
nodes = nu.load_nodes("core", core_radius=200)
edges = nu.load_edges("core")

# %%

network_to_determine = [
    "e4_to_e4",
    "e5it_to_e5it",
    "e6_to_e6",
    "e4_to_e5it",
    "e4_to_e6",
    "e4_to_pv",
]

# make a loop to calculate the delta theta for each network.
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))  # for the average weights
for i, network in enumerate(network_to_determine):
    dt_ori, dt_dir, new_edges = nu.get_delta_theta(f"{network}.json", edges, nodes)
    nu.plot_delta_theta(dt_dir, network, ax=axs.flatten()[i])
    # plot the average weights as a function of delta theta.
    blk_ave = nu.block_ave_weights(
        dt_dir, new_edges["syn_weight"], np.arange(0, 181, 5)
    )
    nu.plot_block_ave_weights(*blk_ave, np.arange(0, 181, 5), ax=axs2.flatten()[i])


fig.suptitle("New core network")
fig2.suptitle("New core network")


# %% Rossi rule investigation 1. Look at some cells
# pick up a random excitatory neuron in L4 and see who connects to it.

reload(nu)
fig = nu.plot_rossi_one(nodes, edges, seed=3000, core=True)
fig = nu.plot_rossi_one(nodes, edges, seed=2003, core=False)
# relatively close to the core.

# %% Rossi rule visualization 2 (for fixing the horizontal/vertical bias)
nodes["tuning_angle"]
# pick up the OSI values from the directory.
osi_df = pd.read_csv("core/metrics/OSI_DSI_DF.csv", sep=" ", index_col=0)
osi_df["tuning_angle"] = nodes["tuning_angle"]
osi_df["pop_name"] = nodes["types"]["pop_name"][nodes["node_type_id"]].values
# digitize by tuning angle (15 degree increments)
incr = 45
osi_df["tuning_angle digitized"] = np.digitize(
    osi_df["tuning_angle"], np.arange(-incr / 2, 361 + incr / 2, incr)
)

# the last index should be the same as the first one, so update the value.
osi_df["tuning_angle digitized"] = np.where(
    osi_df["tuning_angle digitized"] == 360 // incr + 1,
    1,
    osi_df["tuning_angle digitized"],
)

osi_df["tuning_angle digitized"].value_counts()

osi_df["core"] = nodes["core"]

osi_df

# query excitatory neurons in the core, and plot the mean firing rates for each angle
# grp = osi_df.query("pop_name.str.contains('e') & core").groupby(
grp = osi_df.query("pop_name.str.contains('e') & core").groupby(
    "tuning_angle digitized"
)["max_mean_rate(Hz)"]
# plot mean and sem, with correct x axis
grp.size()


plt.figure()
grp.mean().plot(yerr=grp.sem(), marker="o")
plt.xticks(np.arange(1, 360 // incr + 1), labels=np.arange(0, 360, incr))
plt.xlabel("Tuning angle (degrees)")
plt.ylabel("Mean firing rate (Hz)")
plt.title("Excitatory neurons in the core")


# %% do it for the Billeh network
nodes_b = nu.load_nodes("billeh", core_radius=200)
edges_b = nu.load_edges("billeh")

# %%

networks_to_determine_b = [
    "e4_to_e4",
    "e5_to_e5",
    "e6_to_e6",
    "e4_to_e5",
    "e4_to_e6",
    "e4_to_Pv",
]

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))  # for the average weights
for i, network in enumerate(networks_to_determine_b):
    dt_ori, dt_dir, new_edges = nu.get_delta_theta(
        f"{network}.json", edges_b, nodes_b, billeh=True
    )
    nu.plot_delta_theta(dt_dir, network, ax=axs.flatten()[i])
    # plot the average weights as a function of delta theta.
    blk_ave = nu.block_ave_weights(
        dt_dir, new_edges["syn_weight"], np.arange(0, 181, 5)
    )
    nu.plot_block_ave_weights(*blk_ave, np.arange(0, 181, 5), ax=axs2.flatten()[i])

fig.suptitle("Billeh network")
fig2.suptitle("Billeh network")

# dt_ori, dt_dir = get_delta_theta("e4_to_e4.json", edges, nodes)
# plot_delta_theta(dt_dir, "e4_to_e4")

# %%
new_edges["syn_weight"]

# reproduce the code above...
ind = np.digitize(dt_dir, np.arange(0, 181, 5))
ave_weights = np.array([new_edges["syn_weight"][ind == i].mean() for i in range(36)])


# %% let's pick up all the connections from l4e to l4e.
# pick up edge type ids from the dataframe
edges["types"]["dynamics_params"].unique()
etype_ids = edges["types"].query("dynamics_params == 'e4_to_e4.json'")["edge_type_id"]
all_selected = np.isin(edges["edge_type_id"], etype_ids)
# all_selected.sum()
l4edges = nu.filter_by_truthtable(edges, all_selected)

# also, find out the core nodes.
# core_inds = np.where(nodes["core"])[0]


# out of the l4edges, find out the ones that are incoming to or outgoing from the core.
# core_incoming = np.isin(l4edges["target_id"], core_inds)
# core_outgoing = np.isin(l4edges["source_id"], core_inds)

# probably, I can directly work on all the core edges...
nodes["tuning_angle"]

# calculate two versions of delta theta (orientation difference and direction difference)
pre_theta = nodes["tuning_angle"][l4edges["source_id"]]
post_theta = nodes["tuning_angle"][l4edges["target_id"]]

#
delta_theta_ori = nu.angle_difference(pre_theta, post_theta, mode="orientation")
delta_theta_dir = nu.angle_difference(pre_theta, post_theta, mode="direction")

# %%
# plot the number of connections as a function of the delta theta.
# this can be simply a histogram of delta theta.
import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(delta_theta_ori, bins=np.arange(0, 91, 5))
# plot probability desity histogram
# plt.hist(delta_theta_dir, bins=np.arange(0, 181, 5), density=True)
sns.histplot(delta_theta_dir, bins=np.arange(0, 181, 5), stat="density")

# %% of those connections, determine the average weights as a function of delta theta.
delta_theta_dir.shape
weights = l4edges["syn_weight"]


# %% I kind of want to do some test to determine the speed of pysonata...
import sonata.circuit as sc

reload(nu)

# I want to compare two networks.
network = "core"
edges = nu.load_edges(network)
ids, counts = np.unique(edges["edge_type_id"], return_counts=True)

network2 = "core_like05"
edges2 = nu.load_edges(network2)
ids2, counts2 = np.unique(edges2["edge_type_id"], return_counts=True)

# %% let's limit the analysis to core cells.
nodes = nu.load_nodes(network)
nodes2 = nu.load_nodes(network2)

# pick out only neurons in the 200 µm core.
core_nodes = np.sqrt(nodes["x"] ** 2 + nodes["z"] ** 2) < 200
core_edges = np.isin(edges["source_id"], np.where(core_nodes)[0]) & np.isin(
    edges["target_id"], np.where(core_nodes)[0]
)
ids, counts_core = np.unique(edges["edge_type_id"][core_edges], return_counts=True)

core_nodes2 = np.sqrt(nodes2["x"] ** 2 + nodes2["z"] ** 2) < 200
core_edges2 = np.isin(edges2["source_id"], np.where(core_nodes2)[0]) & np.isin(
    edges2["target_id"], np.where(core_nodes2)[0]
)
ids2, counts_core2 = np.unique(edges2["edge_type_id"][core_edges2], return_counts=True)

# make a series for both counts
counts_core_ser = pd.Series(counts_core, index=ids)
counts_core2_ser = pd.Series(counts_core2, index=ids2)

# take the intersection of the two series
common_ids = counts_core_ser.index.intersection(counts_core2_ser.index)
counts_core_ser = counts_core_ser.loc[common_ids]
counts_core2_ser = counts_core2_ser.loc[common_ids]


# %% compare the two networks
# ok, the ids are actually common, so I can compare the counts.

# scatter plot of the counts to see which ones mainly change.
plt.figure()
plt.scatter(counts2, counts)
# make it log-log
plt.xscale("log")
plt.yscale("log")

# also plot the equal line
plt.plot([1, 1e6], [1, 1e6], "k--")
plt.xlabel("# edges with no Rossi (Q=0.5)")
plt.ylabel("# edges with Rossi (Q=0.5)")
plt.title("each dot is a unique edge type id")


# %%
plt.figure()
plt.scatter(counts_core2_ser, counts_core_ser)
plt.xscale("log")
plt.yscale("log")
plt.plot([1, 1e6], [1, 1e6], "k--")
plt.xlabel("# edges with no Rossi (Q=0.5) in the core")
plt.ylabel("# edges with Rossi (Q=0.5) in the core")
plt.title("each dot is a unique edge type id")

# %% histogram of the counts/counts2

plt.figure()
plt.hist(counts / counts2, bins=np.linspace(1, 5, 50))

np.mean(counts / counts2)

np.mean(counts_core_ser / counts_core2_ser)


# %% by the way, let's see if the distribution of the weights are different.
# I can simply plot the histogram of the weights.

plt.figure()
plt.hist(
    edges["syn_weight"],
    bins=np.linspace(-50, 50, 200),
    alpha=0.5,
    label="core",
    density=True,
)

# do the same for the other network
plt.hist(
    edges2["syn_weight"],
    bins=np.linspace(-50, 50, 200),
    alpha=0.5,
    label="core_like05",
    density=True,
)

# weight distribution is very similar.
