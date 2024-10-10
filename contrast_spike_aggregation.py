# %% process contrast files.

import network_utils as nu
import numpy as np
import matplotlib.pyplot as plt
import stimulus_trials as st
import spike_files as sf
from tqdm import tqdm
import argparse


# fast way to summarize spike rates from one file

# %% class is made so things should be easy now.
# TODO: later make this an argument to the script and make a workflow
basedir = "core"
parser = argparse.ArgumentParser(
    description="Summarize spike counts from contrast stimuli."
)
parser.add_argument("basedir", type=str, help="The base directory of the network.")
parser.add_argument("network_option", type=str, help="The network option.")
args = parser.parse_args()
basedir = args.basedir
network_option = args.network_option

# %%

nodes = nu.load_nodes(basedir)
n_nodes = nodes["node_id"].shape[0]

intervals = [[700, 2500]]


def get_spike_counts(filename, n_nodes, intervals):
    spike_hdf = sf.SpikeHDF(filename, n_nodes)
    spike_counts = [spike_hdf.get_spike_count(interval) for interval in intervals]
    spike_counts = np.stack(spike_counts)
    return spike_counts


contrast_stim = st.ContrastStimulus()
all_dirs = contrast_stim.get_all_result_paths(basedir, network_option)

print("Counting spikes from all conditions")
all_spikes = []
for dir_name in tqdm(all_dirs):
    filename = f"{dir_name}/spikes.h5"
    spike_counts = get_spike_counts(filename, n_nodes, intervals)
    all_spikes.append(spike_counts)

all_spikes = np.stack(all_spikes)
all_spikes = np.squeeze(all_spikes)

# %%
# reshape to the stimulus shapes
stim_shape = contrast_stim.get_shape()
# should be like (spont_len, (angles, contrasts, trials))
spont_len = stim_shape[0]
evoked_shape = stim_shape[1]

spont_spikes = all_spikes[:spont_len]
evoked_spikes = np.reshape(all_spikes[spont_len:], evoked_shape + (n_nodes,))

# all_spikes has the shape of
# (n_orientations, n_contrasts, n_trials, n_intervals, n_nodes)

# %%
# store it in a file
savename = f"{basedir}/contrasts_{network_option}/spike_counts.npz"
interval_length = np.diff(intervals[0])[0]
np.savez_compressed(
    savename,
    spont_spikes=spont_spikes,
    evoked_spikes=evoked_spikes,
    interval_length=interval_length,
)
# savename = f"{basedir}/contrasts/spike_counts.npy"
# np.save(savename, all_spikes)
print(f"Done. Saved all the spike counts in {savename}.")
