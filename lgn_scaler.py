# a script to produce scaled LGN inputs based on the given rates.

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

# %% pick up the lgn rate file from the base directory
basedir = "small"
rate_file = basedir + "/filternet/rates.h5"

# read the rate file
with h5py.File(rate_file, "r") as f:
    rates = f["firing_rates/lgn/firing_rates_Hz"][()]


# %%
# rates.shape  # this is a 3001 x 17400 matrix, for 3001 time points and 17400 LGN cells
# first 500 ms is the baseline, then 2500 ms is the stimulus
# we can pick up any time point of the baseline activity for scaling.
pick_time = 200
spont_rates = rates[pick_time, :]
spont_rates

dt = 0.1  # ms
# now, I want to generate a Poisson spike train for each LGN cell
# with the given rate.
duration = 5000  # ms
n_lgn = rates.shape[1]
n_steps = int(duration / dt) + 1  # number of time steps

# what's the average rate?
Ave_Rate = np.mean(spont_rates)

# I want to generate a Poisson spike train for each LGN cell, and extract the indices
# and store in a dataframe that contains 3 colums: timestamps, population, node_ids
# population is fixed to "lgn" for this.


def make_spike_df(spont_rates, duration, n_lgn, scaling):
    spikes = []
    for i in range(n_lgn):
        rate = spont_rates[i]
        n_spikes = np.random.poisson(rate * duration / 1000 * scaling)
        # spike_times = np.random.choice(np.arange(0, duration, dt), n_spikes)
        # it could be continuous, so I need to use np.random.uniform
        spike_times = np.random.uniform(0, duration, n_spikes)
        spikes.append(
            pd.DataFrame(
                {
                    "timestamps": spike_times,
                    "population": "lgn",
                    "node_ids": i,
                }
            )
        )

    # combine all the dataframes into one
    spikes = pd.concat(spikes, ignore_index=True)
    return spikes


def generate_hz_file(spont_rates, duration, n_lgn, Ave_Rate, freq):
    # based on the average rate, I want to generate a file that contains the
    # specified average frequency of the LGN input.
    scaling = freq / Ave_Rate
    spikes = make_spike_df(spont_rates, duration, n_lgn, scaling)
    # file format should be up to the first digit.
    spike_file = f"scaled_spont_lgn_5s/{freq:.1f}_Hz_spikes.csv"
    spikes.to_csv(spike_file, index=False, sep=" ")


# %% write down the spike file (for the original ratio)
spikes = make_spike_df(spont_rates, duration, n_lgn, 1.0)
lgn_spike_dir = "scaled_spont_lgn_5s"
# make the directory if it does not exist
pathlib.Path(lgn_spike_dir).mkdir(parents=True, exist_ok=True)
spike_file = lgn_spike_dir + "/original_ratio_spikes.csv"
spikes.to_csv(spike_file, index=False, sep=" ")

# %% generate the spike files for the specified frequencies up to 30 Hz.
for freq in np.arange(1.0, 30.0, 1.0):
    generate_hz_file(spont_rates, duration, n_lgn, Ave_Rate, freq)

# %% let's see the histogram of the baseline rates
# plt.hist(spont_rates, bins=100)
