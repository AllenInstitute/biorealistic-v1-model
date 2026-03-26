# %%
# study of the stablity. Let's look at the phase of the E and PV to see if
# there is interesting ocillations in the nework.
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotting_utils as pu
import scipy
import scipy.signal
import pathlib

# %% set up the directory where figures are saved.
pathlib.Path("lgn_scaling_figures").mkdir(parents=True, exist_ok=True)


# %%
# load up the data.


def read_spikes(config_file):
    config_js = pu.read_config(config_file)
    net = pu.form_network(config_js)
    spike_df = pu.get_spikes(config_js)

    v1df = net.nodes["v1"].to_dataframe()
    v1df = pu.pick_core(v1df, radius=100.0).copy()
    v1df["Cell Type"] = v1df["pop_name"].apply(pu.identify_cell_type)
    v1df["Sort Position"] = pu.determine_sort_position(v1df)

    # make the core spike_df
    spike_df = spike_df.loc[spike_df.index.isin(v1df.index)].copy()
    spike_df["Sorted ID"] = v1df["Sort Position"].loc[spike_df.index]
    spike_df["Cell Type"] = v1df["Cell Type"].loc[spike_df.index]
    spike_df["pop_name"] = v1df["pop_name"].loc[spike_df.index]
    return spike_df, v1df


# change the config file for yours.
config_file = "small/output_spont_lgn_5s/lgn_fr_20.0Hz/config_11.json"
# config_file = "small/output_spont_lgn_5s/lgn_fr_10.0Hz/config_1.json"
# config_file = "small/output_spont_lgn_5s/lgn_fr_4.0Hz/config_3.json"
spike_df, v1df = read_spikes(config_file)

# %% let's plot the firing rate as function of time for specific cell types.
# layer 4 cells

plot_cell_types = ["e4", "i4Pv", "i4Sst", "i4Vip"]
color_order = ["tab:red", "tab:blue", "tab:olive", "tab:purple", "tab:purple"]
cell_color = {ct: c for ct, c in zip(plot_cell_types, color_order)}
# plt.figure(figsize=(10, 5))
fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

# get the trace if the spike's pop_name contains the plot_cell_types.
# and plot the trace, for each one.
traces_raw = {}
traces_smooth = {}
for cell_type in plot_cell_types:
    binsize_fine = 0.25
    binsize = 1
    spikes = spike_df.loc[spike_df["pop_name"].str.contains(cell_type)].copy()
    # get the trace
    # trace['timestamps'] should contain the time of the spikes.
    # so, making a histogram of the timestamps should give us the firing rate.
    # time stamps are in ms
    nneu = v1df.loc[v1df["pop_name"].str.contains(cell_type)].shape[0]

    bins = np.arange(0, 3000, binsize)
    bins_fine = np.arange(0, 3000, binsize_fine)

    trace = np.histogram(spikes["timestamps"], bins=bins)[0] / nneu / binsize * 1000
    trace_fine = (
        np.histogram(spikes["timestamps"], bins=bins_fine)[0]
        / np.sqrt(nneu)
        / binsize_fine
        * 1000
    )
    traces_raw[cell_type] = trace
    # let's smooth the trace with gaussian filter
    trace_smooth = scipy.ndimage.gaussian_filter1d(trace_fine, sigma=5)
    traces_smooth[cell_type] = trace_smooth

    # plot the trace
    axs[0].plot(bins[:-1], trace, label=cell_type, color=cell_color[cell_type])
    axs[1].plot(
        bins_fine[:-1], trace_smooth, label=cell_type, color=cell_color[cell_type]
    )

axs[0].legend()
# zoom in in the first 100 ms
axs[0].set_xlim(1000, 1200)
axs[0].set_title("raw trace")
axs[1].set_title("smoothed trace (with finer bins)")
# plt.ylim(0, 200)
# save the figure
# plt.savefig("lgn_scaling_figures/smoothed_trace.png", dpi=300)
plt.savefig("lgn_scaling_figures/traces.png", dpi=300)

# %% let's look at the power spectrum of the traces.
for cell_type in plot_cell_types:
    trace = traces_raw[cell_type]
    # the power spectrum
    # f, Pxx_den = scipy.signal.periodogram(trace, fs=1000 / binsize)
    # plt.semilogy(f, Pxx_den, label=cell_type)
    # welch's method
    f, Pxx_den = scipy.signal.welch(trace, fs=1000 / binsize)
    # plt.semilogy(f, Pxx_den, label=cell_type)
    plt.plot(f, Pxx_den, label=cell_type, color=cell_color[cell_type])

# zoom in to the first 250 Hz
plt.xlim(0, 250)
# plt.ylim(0, 4000)
plt.legend()
plt.savefig("lgn_scaling_figures/power_spectrum.png", dpi=300)


# %% also the plot the phase relationship between the E PV SST cells.
# for some period of time, tracked by color
tct1 = "i4Pv"
tct2 = "i4Sst"
starttime = int(1000 / binsize)
endtime = int(1200 / binsize)
tr = slice(starttime, endtime)
settings = {
    "alpha": 1,
}
plt.plot(traces_smooth["e4"][tr], traces_smooth[tct1][tr], **settings)
plt.figure()
plt.plot(traces_smooth["e4"][tr], traces_smooth[tct2][tr], **settings)
plt.figure()
plt.plot(traces_smooth[tct1][tr], traces_smooth[tct2][tr], **settings)

# %% detect the peak times of each trace

peaks = {}
for cell_type in plot_cell_types:
    peaks[cell_type] = scipy.signal.find_peaks(traces_smooth[cell_type])[0]

# for each peak of the excitatory cell, find the time difference from the
# nearest peak of the other cell types.
# and plot the histogram of the time differences.

# and plot the histogram of the time differences.
for cell_type in plot_cell_types:
    if cell_type == "e4":
        continue
    if cell_type == "i4Vip":  # delete this condition if VIP has activity.
        continue
    print(cell_type)
    tdiffs = []
    for peak_t in peaks["e4"]:
        tdiff = np.abs(peaks[cell_type] - peak_t)
        # tdiffs.append(np.min(tdiff))
        # instead of the absolute diffs, I want to save the signed diffs.
        # avoid empty tdiff.
        if tdiff.size > 0:
            tdiffs.append(peaks[cell_type][np.argmin(tdiff)] - peak_t)
    # convert time difference to ms
    tdiffs = np.array(tdiffs) * binsize_fine
    plt.hist(tdiffs, label=cell_type, alpha=0.5, bins=np.linspace(-20, 20, 50))

plt.legend()
plt.savefig("lgn_scaling_figures/peak_time_difference.png", dpi=300)

# you can stop reading the code here.

###########################################

# %% load up all the config files in the lgn directory, and plot the
# overall firing rates of each cell type as a function of the LGN firing rate.
lgnrates = np.arange(1, 29, 1)
config_files = []
for lgnrate in lgnrates:
    config_files.extend(
        glob.glob(f"small/output_spont_lgn_5s/lgn_fr_{lgnrate}.0Hz/*.json")
    )

# in this blok, summarize the FR of each cell type into a list.
cellfr = {}
for config_file in config_files:
    spike_df, v1df = new_func(config_file)
    for cell_type in plot_cell_types:
        nneu = v1df.loc[v1df["pop_name"].str.contains(cell_type)].shape[0]
        fr = (
            spike_df.loc[spike_df["pop_name"].str.contains(cell_type)].shape[0]
            / (3000 / 1000)
            / nneu
        )
        # fr = spike_df.loc[spike_df['pop_name'].str.contains(cell_type)].shape[0] / (3000 / 1000)
        if cell_type not in cellfr:
            cellfr[cell_type] = [fr]
        else:
            cellfr[cell_type].append(fr)


# %% plot the firing rates.
plt.figure(figsize=(10, 5))
for cell_type in plot_cell_types:
    plt.plot(lgnrates, cellfr[cell_type], label=cell_type)

plt.legend()
plt.savefig("lgn_scaling_figures/firing_rate.png", dpi=300)

# %% show some rasters as well.
# make 3 config files (4hz, 10hz, 20hz) as a list
configs = [
    "small/output_spont_lgn_5s/lgn_fr_4.0Hz/config_3.json",
    "small/output_spont_lgn_5s/lgn_fr_10.0Hz/config_1.json",
    "small/output_spont_lgn_5s/lgn_fr_20.0Hz/config_11.json",
]

# for each config file, plot the raster plot and save it as a file.
for config_file in configs:
    plt.figure(figsize=(10, 6))
    pu.plot_raster(config_file, s=3, radius=100.0)
    plt.xlim(0, 1000)
    plt.savefig(
        f"lgn_scaling_figures/raster_{config_file.split('/')[-2].split('.')[0]}.png",
        dpi=300,
    )
