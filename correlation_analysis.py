# %% depicting layer 6 correlation structure.
import network_utils as nu
import stimulus_trials as st

# reload st
from importlib import reload

reload(st)

base_dir = "core"
network = "checkpoint"

v1_nodes = nu.load_nodes_pl(base_dir, core_radius=200).collect().to_pandas()
ctdf = nu.get_cell_type_table()


# get trials
stimulus = st.DriftingGratingsStimulus()
paths = stimulus.get_all_result_paths(base_dir, network)


# %% let's do analysis for one of them.

import numpy as np
import h5py
from numba import jit

# read the data.


def read_spikes(path):
    f = h5py.File(f"{path}/spikes.h5", "r")
    node_ids = f["spikes/v1/node_ids"][:]
    timestamps = f["spikes/v1/timestamps"][:]
    f.close()
    output = {"node_ids": node_ids, "timestamps": timestamps}
    return output


spikes = read_spikes(paths[0])

import matplotlib.pyplot as plt

plt.plot(spikes["node_ids"])


# %%
@jit(nopython=True)
def format_spikes(node_ids, timestamps, n_nodes):
    spike_times = []
    for i in range(n_nodes):
        # store the sorted spike time for each neuron
        spike_times.append(timestamps[node_ids == i])
        spike_times[-1].sort()
    return spike_times


spikes_formatted = format_spikes(
    spikes["node_ids"], spikes["timestamps"], v1_nodes.shape[0]
)

spikes_formatted[0]

# %%
%%timeit
import numpy as np
from numba import njit, prange
from numba.typed import List


@njit
def format_spikes_sorted(node_ids, timestamps, n_nodes):
    """
    Reformats spike data by sorting the entire dataset by node_ids (primary key)
    and then sorting the timestamps within each node group (secondary key).

    Parameters
    ----------
    node_ids : 1D np.ndarray of int64
        Array of neuron IDs.
    timestamps : 1D np.ndarray of float64
        Array of spike times.
    n_nodes : int
        Total number of neurons (neurons with no spikes will yield an empty array).

    Returns
    -------
    typed.List of 1D np.ndarray
        A list where each element (index i) is a sorted array of spike times for neuron i.
    """
    # Sort indices by node_ids (primary sort)
    order = np.argsort(node_ids)
    sorted_node_ids = node_ids[order]
    sorted_timestamps = timestamps[order]
    n_spikes = sorted_node_ids.shape[0]

    # For each contiguous group of equal node_ids, sort the timestamps in that group.
    if n_spikes > 0:
        start = 0
        for i in range(1, n_spikes + 1):
            if i == n_spikes or sorted_node_ids[i] != sorted_node_ids[start]:
                # Copy the segment and sort it
                seg_length = i - start
                temp = np.empty(seg_length, dtype=timestamps.dtype)
                # Copy the segment into temp
                for k in range(seg_length):
                    temp[k] = sorted_timestamps[start + k]
                # Sort the segment (np.sort is supported in nopython mode)
                temp = np.sort(temp)
                # Write back the sorted segment.
                for k in range(seg_length):
                    sorted_timestamps[start + k] = temp[k]
                start = i

    # Count the number of spikes for each neuron.
    counts = np.zeros(n_nodes, dtype=np.int64)
    for i in range(n_spikes):
        # Assumes node IDs are in the range 0 to n_nodes-1.
        counts[sorted_node_ids[i]] += 1

    # Compute cumulative counts so that for neuron i, its spikes are in
    # sorted_timestamps[ cum_counts[i] : cum_counts[i] + counts[i] ]
    cum_counts = np.empty(n_nodes, dtype=np.int64)
    total = 0
    for i in range(n_nodes):
        cum_counts[i] = total
        total += counts[i]

    # Preallocate a numba.typed.List for output arrays.
    out = List()
    for i in range(n_nodes):
        out.append(np.empty(counts[i], dtype=timestamps.dtype))

    for i in range(n_nodes):
        start_idx = cum_counts[i]
        for j in range(counts[i]):
            out[i][j] = sorted_timestamps[start_idx + j]

    return out


spikes_formatted = format_spikes_sorted(
    spikes["node_ids"], spikes["timestamps"], v1_nodes.shape[0]
)



# %% let's do the correlation analysis.

# pick up all the excitatory cells from each layer.

cell_type = "L2/3_Exc"
v1_nodes["cell_type"] = ctdf.loc[v1_nodes["pop_name"]]["cell_type"].values

# pick up the core cells of the cell type.

type_df = v1_nodes.query(f"cell_type == '{cell_type}' and core == True").copy()


# for these neurons, evaluate the correlation of the activity as a function of the
# difference in the preferred orientation.

# for now let's use tuning angle in place for the preferred angle properly measured.
