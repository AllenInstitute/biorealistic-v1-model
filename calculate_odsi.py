# %%
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as pylab
import os
import json
from sonata.circuit import File
import math
import sys
from pathlib import Path


def calculateFiringRate(gids, ts, numNrns, start_time=0.0, duration=2.0):
    # print("Calculating Firing Rate")
    start_time = start_time * 1000.0
    # end_time = gray_screen + 2000.0  # requires at least 2 seconds of stimulus
    end_time = start_time + duration * 1000.0

    gids = gids[np.where(np.logical_and(ts > start_time, ts <= end_time))[0]]

    gid_bins = np.arange(0 - 0.5, numNrns + 0.5, 1)

    hist, bins = np.histogram(gids, bins=gid_bins)

    mean_firing_rates = hist / duration

    # if gray_screen > 0:
    #     mean_firing_rates = hist / ((3000.0 - gray_screen) / 1000.0)
    # else:
    #     mean_firing_rates = hist / (3000.0 / 1000.0)
    return mean_firing_rates


def calculate_Rates_DF(numNrns, trials=10, angles=np.arange(0, 360, 45), set_name=None):
    Rates_DF = pd.DataFrame(
        index=range(numNrns * len(angles)),
        columns=[
            "DG_angle",
            "node_id",
            "Avg_rate(Hz)",
            "SD_rate(Hz)",
            "Spont_rate(Hz)",
        ],
    )

    for i, ori in enumerate(angles):
        # initialize with nans
        firingRatesTrials = np.empty((trials, numNrns))
        firingRatesTrials[:] = np.nan
        spontRatesTrials = np.empty((trials, numNrns))
        spontRatesTrials[:] = np.nan

        for trial in range(trials):
            spikes_file_name = (
                # f"{basedir}/8dir_10trials/angle{ori}_trial{trial}/spikes.csv"
                f"{set_name}/angle{ori}_trial{trial}/spikes.csv"
            )
            # spikes = np.loadtxt(spikes_file_name, unpack=True)
            # if the file is not found, skip it
            try:
                spikes = pd.read_csv(spikes_file_name, sep=" ")
            except FileNotFoundError:
                print(f"File not found: {spikes_file_name} .  Skipping...")
                # the value remains nan, and ignored in the mean calculation
                continue
            # ts, gids = spikes
            # print(spikes)
            ts = np.array(spikes["timestamps"])
            gids = np.array(spikes["node_ids"], dtype=int)
            # gids = gids.astype(int)

            firingRates = calculateFiringRate(gids, ts, numNrns, start_time=0.5)
            # firingRates = calculateFiringRate(
            #     gids, ts, numNrns, start_time=0.1, duration=0.5
            # )
            spontRates = calculateFiringRate(
                gids,
                ts,
                numNrns,
                start_time=0.1,
                duration=0.4,
                # gids,
                # ts,
                # numNrns,
                # start_time=0.0,
                # duration=0.1,
            )
            # print(spikes_file_name)

            firingRatesTrials[trial, :] = firingRates
            spontRatesTrials[trial, :] = spontRates

        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "DG_angle"] = ori
        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "node_id"] = np.arange(
            numNrns
        )
        # at one point, try nanmean and nanstd for remedying faining cluster computation
        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "Avg_rate(Hz)"] = np.nanmean(
            firingRatesTrials, axis=0
        )
        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "SD_rate(Hz)"] = np.nanstd(
            firingRatesTrials, axis=0
        )
        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "Spont_rate(Hz)"] = (
            np.nanmean(spontRatesTrials, axis=0)
        )

    return Rates_DF


def calculate_OSI_DSI_from_DF(rates_df, basedir):
    num_angles = 8

    num_neurons = len(rates_df) // num_angles
    all_rates = np.zeros((num_neurons, num_angles))
    all_spont = np.zeros((num_neurons, num_angles))

    angles = range(0, 360, 45)
    angle_counts = 0
    for angle, g in rates_df.groupby("DG_angle"):
        all_rates[:, angle_counts] = g["Avg_rate(Hz)"]
        all_spont[:, angle_counts] = g["Spont_rate(Hz)"]
        angle_counts += 1

    preferred_angle_ind = np.argmax(all_rates, axis=1)
    preferred_rates = all_rates[
        np.arange(len(preferred_angle_ind)), preferred_angle_ind
    ]

    phase_rad = np.array(angles) * math.pi / 180

    dsi = np.abs(
        (all_rates * np.exp(1j * phase_rad)).sum(axis=1) / all_rates.sum(axis=1)
    )
    osi = np.abs(
        (all_rates * np.exp(2j * phase_rad)).sum(axis=1) / all_rates.sum(axis=1)
    )

    osi_df = pd.DataFrame()
    osi_df["node_id"] = range(num_neurons)
    osi_df["DSI"] = dsi
    osi_df["OSI"] = osi
    osi_df["preferred_angle"] = np.array(angles)[preferred_angle_ind]
    osi_df["max_mean_rate(Hz)"] = preferred_rates
    osi_df["Avg_Rate(Hz)"] = np.mean(all_rates, axis=1)
    osi_df["Spont_Rate(Hz)"] = np.mean(all_spont, axis=1)
    return osi_df


if __name__ == "__main__":
    args = sys.argv
    basedir = args[1]
    network_option = args[2]
    network_dir = basedir + "/network"
    set_name = basedir + f"/8dir_10trials_{network_option}"
    metric_dir = basedir + "/metrics"
    # network_dir = args[1]
    # set_name = args[2]

    # Path(set_name).mkdir(exist_ok=True)
    Path(metric_dir).mkdir(exist_ok=True)
    trials = 10
    angles = np.arange(0, 360, 45)

    nodes_file_name = network_dir + "/v1_nodes.h5"
    type_file_name = network_dir + "/v1_node_types.csv"
    net = File(data_files=nodes_file_name, data_type_files=type_file_name)
    # nodes_DF = pd.read_csv(nodes_file_name, sep=" ")
    nodes_DF = net.nodes["v1"].to_dataframe()
    # there is additional file in there that use the subset of the node ids.
    # let's reset the index, and use them as the node ids.
    numNrns = len(nodes_DF)

    print("FR calculation started...")
    Rates_DF = calculate_Rates_DF(
        numNrns,
        trials=trials,
        angles=angles,
        set_name=set_name,
    )
    # drop the rows that is not included in node_ids
    if basedir == "tensorflow":
        node_ids = np.load(set_name + "/node_ids.npy")
        Rates_DF = Rates_DF.loc[Rates_DF["node_id"].isin(node_ids)]

    Rates_DF.to_csv(
        basedir + f"/metrics/Rates_DF_{network_option}.csv", sep=" ", index=False
    )
    print("Done Rates DF!")

    # Rates_DF = pd.read_csv(set_name + "/Rates_DF.csv", sep=" ", index_col=False)
    osi_df = calculate_OSI_DSI_from_DF(Rates_DF, basedir=basedir)
    if basedir == "tensorflow":
        # replace the node_ids with the node_ids
        osi_df["node_id"] = node_ids
    osi_df.to_csv(
        basedir + f"/metrics/OSI_DSI_DF_{network_option}.csv", sep=" ", index=False
    )
    print("Done with all!")
