# %% class for analzing spike files

import numpy as np
import h5py
import network_utils as nu


class SpikeHDF:
    def __init__(self, filename, n_nodes):
        self.filename = filename
        self.n_nodes = n_nodes
        with h5py.File(filename, "r") as f:
            self.spikes = f["spikes/v1/node_ids"][:]
            self.times = f["spikes/v1/timestamps"][:]

    def get_spike_count(self, interval):
        spike_counts = np.zeros(self.n_nodes)
        valid_inds = (self.times > interval[0]) & (self.times < interval[1])
        bin_counts = np.bincount(self.spikes[valid_inds])
        spike_counts[: len(bin_counts)] = bin_counts

        return spike_counts
