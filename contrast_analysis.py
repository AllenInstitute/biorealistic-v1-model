# %% using the spike counts analyze the contrast response
import numpy as np
import network_utils as nu
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import lazyscience as lz
import stimulus_trials as st
from importlib import reload

# load spike counts

basedir = "core"
spike_file_name = f"{basedir}/contrasts/spike_counts.npz"
f = np.load(spike_file_name)
spont_spikes = f["spont_spikes"]
evoked_spikes = f["evoked_spikes"]
interval_length = f["interval_length"]

# %%
reload(nu)
nodes = nu.load_nodes(basedir, core_radius=200, expand=True)

nodes_p = nodes[nodes["core"]]

# first, get the node_id of specific cell type.
# ids = nodes_c.filter(pl.col("cell_type") == "L2/3_Exc").select("node_id").to_numpy()
# type_df = nodes_p.query("cell_type == 'L6_SST' & core == True")

fig, axs = plt.subplots(5, 4, figsize=(20, 20))
axsf = axs.flatten()
count = 0
for cell_type, type_df in nodes_p.groupby("cell_type"):
    ax = axsf[count]
    count += 1
    # if count == 3:
    # break

    print(f"{cell_type}: {type_df.shape[0]}")
    ids_sorted = type_df.sort_values("tuning_angle").index.to_numpy()

    # % using these ids, plot heatmap of the spike rates
    compact_data = evoked_spikes[:, :, :, ids_sorted]
    av_spikes = np.mean(compact_data, axis=2)

    # reshape to 2D and plot
    av_spikes.T.shape
    plot_spikes = np.reshape(av_spikes.T, (av_spikes.T.shape[0], -1))
    sns.heatmap(plot_spikes, ax=ax, cmap="viridis")
    ax.set_title(cell_type)


plt.tight_layout()
fig.savefig(f"{basedir}/figures/contrast_response_heatmap.png", dpi=300)

# %% Let's also make a normalized version of this plot

fig, axs = plt.subplots(5, 4, figsize=(20, 20))
axsf = axs.flatten()
count = 0


# start with one figure
# fig, ax = plt.subplots(figsize=(10, 10))
for cell_type, type_df in nodes_p.groupby("cell_type"):
    ax = axsf[count]
    count += 1
    # type_df = nodes_p.query("cell_type == 'L4_Exc' & core == True")
    ids_sorted = type_df.sort_values("tuning_angle").index.to_numpy()

    # % using these ids, plot heatmap of the spike rates
    compact_data = evoked_spikes[:, :, :, ids_sorted]
    spont_data = spont_spikes[:, ids_sorted].mean(axis=0)
    av_spikes = np.mean(compact_data, axis=2)
    av_mean = av_spikes.mean(axis=(0, 1))

    norm_data = (av_spikes - spont_data) / (av_mean + spont_data)
    av_spikes = norm_data

    # reshape to 2D and plot
    av_spikes.T.shape
    plot_spikes = np.reshape(av_spikes.T, (av_spikes.T.shape[0], -1))
    sns.heatmap(plot_spikes, ax=ax, cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_title(cell_type)

plt.tight_layout()
fig.savefig(f"{basedir}/figures/contrast_response_heatmap_norm.png", dpi=300)

# %% OK. This yields nothing, so I'd need to extract significantly responding cells.
reload(lz)
qpsig = lz.quasi_poisson_sig_test_counts(
    evoked_spikes, interval_length, spont_spikes, interval_length
)

# fraction of cells that are significantly responding
frac_pos = (qpsig < 0.01).sum() / qpsig.size
frac_neg = ((1 - qpsig) < 0.01).sum() / qpsig.size

print(f"Fraction of cells with positive response: {frac_pos}")
print(f"Fraction of cells with negative response: {frac_neg}")


# %%
(qpsig < 0.01).sum() / qpsig.size
resp_count = evoked_spikes.sum(axis=(0, 1, 2))

sig_cells_pos = np.array(qpsig) < 0.01
sig_cells_neg = (1 - np.array(qpsig)) < 0.01


# %%

netdf = nodes_p.reset_index()


def binomprob(truth_table):
    prob = truth_table.sum() / len(truth_table)
    err = np.sqrt(prob * (1 - prob) / len(truth_table))
    return (prob, err)


def barfraction(sig_cells, signature, ax, invx=False):
    resultdict = {}
    for pop_name, ids in netdf.groupby("cell_type")["node_id"]:
        resultdict[pop_name] = binomprob(sig_cells[ids])

    values, errors = zip(*list(resultdict.values()))
    keys = list(resultdict.keys())

    xpos = range(len(values))
    if invx:
        ax.barh(xpos, values, xerr=errors, color="C0")
        ax.invert_xaxis()
        ax.set_xlim([1, 0])
    else:
        ax.barh(xpos, values, xerr=errors, color="C1")
        ax.set_xlim([0, 1])

    ax.set_yticks(xpos)
    ax.set_yticklabels(keys)
    if invx:
        ax.set_title(f"Fraction of cells responsive (orange: positive, blue: negative)")


fig, ax = plt.subplots(figsize=(6, 5))
ax.invert_yaxis()
barfraction(sig_cells_pos, "positive", ax)
ax2 = ax.twiny()
barfraction(sig_cells_neg, "negative", ax2, invx=True)
fig.tight_layout()

# make the first xtick font color orange, and second blue
for label in ax.get_xticklabels():
    label.set_color("C1")
for label in ax2.get_xticklabels():
    label.set_color("C0")

fig.savefig(f"{basedir}/figures/contrast_responsive_cells.pdf")


# %% Center of gravity analysis
