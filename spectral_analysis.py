# %% look at the spectra of the spiking activity evoked period

import numpy as np
import h5py

import pandas as pd
import matplotlib.pyplot as plt
import plotting_utils as pu
from numba import njit
from pandarallel import pandarallel

pandarallel.initialize()


# %%


# for each node_id (index) in spike_df, do a Fourier transform of the spike train
# and sum them up.
# %%


# @njit
def sparse_ft(timestamps, freqs):
    # a function to calculate the Fourier transform of a sparse spike train
    # timestamps are in ms, and freqs are Hz.

    # for eash timestamp, calculate the complex phase, and add them all.
    complex_phase = np.exp(2j * np.pi * freqs * timestamps[:, np.newaxis] / 1000)
    # spectra = np.abs(complex_phase.sum(axis=0))
    spectra = complex_phase.sum(axis=0)
    return spectra


# stamps = spike_df[spike_df.index == 48008]["timestamps"].to_numpy()
freqs = np.arange(3, 150, 0.25)
# freqs * stamps[:, np.newaxis]
# sparse_ft(stamps, freqs)

# %timeit sparse_ft(stamps, freqs)


def get_fts(config_file, freqs):

    # spike_df, _, _, _ = pu.make_figure_elements(config_file, 200, "tuning_angle", True)
    spike_df = pu.get_spikes(config_file, infer=True)

    spike_df["neuron_id"] = spike_df.index
    spike_df_neurons = spike_df.groupby("neuron_id").apply(
        lambda x: x["timestamps"].to_numpy()
    )

    fts = spike_df_neurons.parallel_apply(lambda x: sparse_ft(x, freqs))

    max_id = np.max(spike_df["neuron_id"])

    ave_fts = fts.mean(axis=0) / len(spike_df) * max_id
    return ave_fts


config_files = [
    # "core/output/config.json",
    # "core/output_adjusted/config_plain.json",
    "core_nll_1/output_checkpoint/config_checkpoint.json",
    "core_nll_1/output_plain/config_plain.json",
]


data = []

for config_file in config_files:
    ave_fts = get_fts(config_file, freqs)
    # aplly smoothing
    window = 100
    ave_fts_smooth = np.convolve(
        np.abs(ave_fts) ** 2, np.ones(window) / window, mode="same"
    )
    data.append(ave_fts_smooth)


# plt.plot(freqs, np.abs(ave_fts) ** 2)
plt.plot(freqs, ave_fts_smooth)
plt.ylim(bottom=0.0, top=0.001)
# plt.ylim(1.74, 1.8)

# %%


def nobox(ax):
    """remove top and right spines and ticks"""
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax  # remove top and right spines.


data_all = np.stack(data, axis=1)
# normalize the axis 0
data_all_norm = data_all / data_all.sum(axis=0)
plt.figure(figsize=(4, 3))
plt.plot(freqs, data_all_norm[:, [0, 2]])
plt.xlim(0, 120)
plt.ylim(0, 0.012)
# plt.yscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized power (a.u.)")
# plt.legend(["New model (double alpha)", "New model (single alpha)", "Billeh model"])
plt.legend(["New model", "Billeh model"])


nobox(plt.gca())


# %% replot

plt.plot(freqs, np.abs(ave_fts) ** 2)
plt.plot(freqs, ave_fts_smooth)
plt.ylim(bottom=0.0, top=0.0025)

ave_fts


# %% rasters
pu.plot_raster(config_files[2], sortby="tuning_angle", infer=True, radius=200)
# plt.xlim(1000, 1500)
