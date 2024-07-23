# script to calculate the fano factor of a spike train

# %%
import os
import pandas as pd
import network_utils as nu
import numpy as np
import matplotlib.pyplot as plt


def pop_fano(spikes, bin_sizes, t_start=0.2, t_end=2.0):
    # bin_size in seconds
    fanos = np.zeros(len(bin_sizes))
    for i, bin_size in enumerate(bin_sizes):
        sp_counts = np.histogram(spikes, bins=np.arange(t_start, t_end, bin_size))[0]
        fanos[i] = np.var(sp_counts) / np.mean(sp_counts)

    return fanos


def pop_fano_table(table, bin_sizes, n_stim):
    fanos = np.zeros((n_stim, len(bin_sizes)))
    for i, (stim_id, subtable) in enumerate(table.group_by("stimulus_presentation_id")):
        fanos[i, :] = pop_fano(
            subtable["time_since_stimulus_presentation_onset"].to_numpy(), bin_sizes
        )

    fanos_mean = fanos.mean(axis=0)
    fanos_std = fanos.std(axis=0)
    fanos_sem = fanos.std(axis=0) / np.sqrt(n_stim)

    fano_summary = {
        "mean": fanos_mean,
        "std": fanos_std,
        "sem": fanos_sem,
        "fanos": fanos,
        "bin_sizes": bin_sizes,
    }
    return fano_summary


# %%

network = "core"
output = "output_adjusted"
spike_file = f"{network}/{output}/spikes.csv"

spikes = pd.read_csv(spike_file, sep=" ")


# %% load the cells

nodes = nu.load_nodes(network, core_radius=200)
core_mask = nodes["core"]
core_id = nodes["node_id"][core_mask]

# cell type table
cell_naming = pd.read_csv(
    "base_props/cell_type_naming_scheme.csv", sep=" ", index_col=0
)


node_ei = nodes["types"]["ei"][nodes["node_type_id"][core_mask]]
node_id = nodes["node_id"][core_mask]
node_id_e = node_id[node_ei == "e"]

# sample_num = 68  # based on the average e cells in neuropixels
# random sampling of 68 neurons from node_id_e
# let's draw a sample number from a Gaussian distribution with mean 68 and std 10.

fanos = []
for i in range(30):
    sample_num = int(np.random.normal(68, 10))
    sampled_ids = np.random.choice(node_id_e, sample_num, replace=False)
    bin_sizes = np.logspace(-3, 0, 20)

    spikes_sample = spikes[spikes["node_ids"].isin(sampled_ids)]
    fano = pop_fano(
        spikes_sample["timestamps"].to_numpy() / 1000, bin_sizes, t_start=0.7, t_end=2.5
    )
    fanos.append(fano)


fanos = np.array(fanos)

# %%
fanos_mean = fanos.mean(axis=0)
fanos_std = fanos.std(axis=0)
fanos_sem = fanos.std(axis=0) / np.sqrt(30)

plt.errorbar(bin_sizes, fanos_mean, yerr=fanos_sem)
plt.xscale("log")
# %% show raster for some of the cells

sample_num = int(np.random.normal(68, 10))
sample_ids = np.random.choice(node_id_e, sample_num, replace=False)
spikes_sample = spikes[spikes["node_ids"].isin(sample_ids)]

# check the firing rates.
rates = spikes_sample.groupby("node_ids").size()
# sort by the firing rates
sorted_ids = rates.sort_values(ascending=False).index
# assign serial ID based on sorted_ids and make a scatter plot of the spikes
serial_id = {nid: i for i, nid in enumerate(sorted_ids)}
spikes_sample["serial_id"] = spikes_sample["node_ids"].map(serial_id)

plt.figure(figsize=(10, 1))
plt.scatter(
    spikes_sample["timestamps"] / 1000,
    spikes_sample["serial_id"],
    s=1,
    c="k",
    alpha=0.5,
)

# %%
