# %% cortical amplification analysis


# import data
import network_utils as nu
import cortical_amplification_lib as cal
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
import h5py
from numba import njit
reload(nu)
reload(cal)

base_dir = "core_nll_0"

# %%
# edge_lf = nu.load_edges_pl(base_dir, src="v1", tgt="v1")
# edge_df = edge_lf.collect().to_pandas()

# suffix = "_bio_trained"
suffix = "_naive"

v1_edges = nu.load_edges(base_dir, src="v1", tgt="v1", appendix=suffix, separate_tau=True)
lgn_edges = nu.load_edges(base_dir, src="lgn", tgt="v1", appendix="", separate_tau=True)
bkg_edges = nu.load_edges(base_dir, src="bkg", tgt="v1", appendix=suffix, separate_tau=True)





# %%

# a function to construct double alpha based on the three parameters


# make double alpha function
@njit
def double_alpha_shape(tau_syn_fast, tau_syn_slow, amp_slow, delay, window_size=50, dt=0.25):
    # units are ms
    t = np.arange(0, window_size, dt)
    effective_t = np.maximum(t - delay, 0)
    alpha_fast = effective_t * np.exp(-effective_t / tau_syn_fast)
    alpha_slow = effective_t * np.exp(-effective_t / tau_syn_slow)
    return t, alpha_fast + alpha_slow * amp_slow


# an example
tau_syn = edges["types"]["tau_syn"][250]
t, trace = double_alpha_shape(tau_syn[0], tau_syn[1], tau_syn[2], 2)
plt.plot(t, trace)



# %% get nodes
v1_nodes = nu.load_nodes(base_dir, expand=True, core_radius=200)
lgn_nodes = nu.load_nodes(base_dir, expand=False, loc="lgn")
bkg_nodes = nu.load_nodes(base_dir, expand=False, loc="bkg")




# %% pick a L4 Exc cell.

# fix seed here.
np.random.seed(0)
l4e = nodes.query("cell_type == 'L4_Exc' and core == True")
sample_id = l4e.sample(1).index.values[0]
print(f"sample_id: {sample_id}")


# %% get spikes
reload(nu)
v1_spikes = nu.load_spike_dict(f"{base_dir}/output_bio_trained/spikes.h5")
v1_n_neu = len(v1_nodes["node_type_id"])
v1_spikes_list = [np.array(np.round(v1_spikes.get(i, np.array([])) / dt), dtype=np.int64) for i in range(v1_n_neu)]


lgn_spikes = nu.load_spike_dict(f"{base_dir}/filternet/spikes.h5", pop_name="lgn")
lgn_n_neu = len(lgn_nodes["node_type_id"])
lgn_spikes_list = [np.array(np.round(lgn_spikes.get(i, np.array([])) / dt), dtype=np.int64) for i in range(lgn_n_neu)]


bkg_spikes = nu.load_spike_dict(f"{base_dir}/bkg/bkg_spikes_250Hz_3s.h5", pop_name="bkg")
bkg_n_neu = len(bkg_nodes["node_type_id"])
bkg_spikes_list = [np.array(np.round(bkg_spikes.get(i, np.array([])) / dt), dtype=np.int64) for i in range(bkg_n_neu)]



# %%



# %%

@njit
def calculate_one_trace(spike_inds, shape, duration):
    trace = np.zeros(duration)
    for i in range(len(spike_inds)):
        if spike_inds[i] + len(shape) <= duration:
            trace[spike_inds[i]:spike_inds[i]+len(shape)] += shape
        else:
            trace[spike_inds[i]:] += shape[:duration - spike_inds[i]]
    return trace

# %%
%time trace = calculate_one_trace(spike_inds, shape, duration)

# %%
# @njit @njit makes it longer.
def calculate_pre_trace(pre_ids, syn_weights, n_syns, tau_syns, delays, spikes_list, duration):
    traces = np.zeros((len(pre_ids), duration))
    for i in range(len(pre_ids)):
        # if the pre_ids is not the key of the spikes, skip it.
        if len(spikes_list[pre_ids[i]]) == 0:
            continue
        
        spike_inds = spikes_list[pre_ids[i]]

        syn_weight = syn_weights[i]
        n_syn = n_syns[i]
        tau_syn = tau_syns[i]
        delay = delays[i]

        t, shape = double_alpha_shape(tau_syn[0], tau_syn[1], tau_syn[2], delay)
        shape = shape * syn_weight * n_syn
        traces[i] = calculate_one_trace(spike_inds, shape, duration)
    return traces

def calculate_pre_trace2(edges, target_id, spikes_list, duration):
    edge_mask = edges["target_id"] == target_id
    pre_ids = edges["source_id"][edge_mask]
    syn_weights = edges["syn_weight"][edge_mask]
    n_syns = edges["n_syns"][edge_mask]
    tau_syns = list(edges["types"]["tau_syn"][edges["edge_type_id"][edge_mask]])
    delays = list(edges["types"]["delay"][edges["edge_type_id"][edge_mask]])
    pre_trace = calculate_pre_trace(pre_ids, syn_weights, n_syns, tau_syns, delays, spikes_list, duration)
    return {"exc": np.maximum(pre_trace, 0).sum(axis=0), "inh": np.minimum(pre_trace, 0).sum(axis=0)}



# %%

sample_id = l4e.sample(1).index.values[0]
print(f"sample_id: {sample_id}")

pre_trace = calculate_pre_trace2(v1_edges, sample_id, v1_spikes_list, duration)
lgn_pre_trace = calculate_pre_trace2(lgn_edges, sample_id, lgn_spikes_list, duration)
bkg_pre_trace = calculate_pre_trace2(bkg_edges, sample_id, bkg_spikes_list, duration)


plt.plot(pre_trace["exc"])
plt.plot(lgn_pre_trace["exc"])
plt.plot(bkg_pre_trace["exc"])


# %% let's get traces from 100 random neurons in the l4 and have them averaged.


np.random.seed(0)
l4e = v1_nodes.query("cell_type == 'L4_Exc' and core == True")
l4e45 = l4e.query("tuning_angle > 40 and tuning_angle < 50")
sample_ids = l4e45.sample(100).index.values


pre_traces = np.mean([calculate_pre_trace2(v1_edges, sample_id, v1_spikes_list, duration)["exc"] for sample_id in sample_ids], axis=0)
lgn_pre_traces = np.mean([calculate_pre_trace2(lgn_edges, sample_id, lgn_spikes_list, duration)["exc"] for sample_id in sample_ids], axis=0)
bkg_pre_traces = np.mean([calculate_pre_trace2(bkg_edges, sample_id, bkg_spikes_list, duration)["exc"] for sample_id in sample_ids], axis=0)


# %%
t = np.linspace(0, duration * dt, len(pre_traces))
plt.plot(t, pre_traces)
plt.plot(t, lgn_pre_traces)
plt.plot(t, bkg_pre_traces)

plt.title("Average EPSC from 100 random L4 Exc cells (tuned to stim, naive)")
plt.legend(["V1", "LGN", "BKG"])


# %%
len(lgn_spikes_list)
len(bkg_spikes_list)

mask = bkg_edges["target_id"] == sample_id
pre_ids = bkg_edges["source_id"][mask]
syn_weights = bkg_edges["syn_weight"][mask]
n_syns = bkg_edges["n_syns"][mask]
tau_syns = list(bkg_edges["types"]["tau_syn"][bkg_edges["edge_type_id"][mask]])
delays = list(bkg_edges["types"]["delay"][bkg_edges["edge_type_id"][mask]])
pre_trace = calculate_pre_trace(pre_ids, syn_weights, n_syns, tau_syns, delays, bkg_spikes_list, duration)

bkg_spikes_list[pre_ids[0]]

bkg_spikes_list

np.sum([len(i) for i in bkg_spikes_list])
bkg_spikes_list




# %%
# %timeit pre_trace = calculate_pre_trace(pre_ids, syn_weights, n_syns, tau_syns, delays, spikes_list, duration)

# %%
plt.plot(np.maximum(pre_trace, 0).sum(axis=0))  # show only positive inputs





# plt.plot(trace)





