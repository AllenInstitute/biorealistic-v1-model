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


def calculateFiringRate(gids, ts, numNrns, gray_screen=0.0):

    # print("Calculating Firing Rate")
    start_time = gray_screen
    end_time = gray_screen + 2000.0  # requires at least 2 seconds of stimulus

    gids = gids[np.where(np.logical_and(ts > start_time, ts <= end_time))[0]]

    gid_bins = np.arange(0 - 0.5, numNrns + 0.5, 1)

    hist, bins = np.histogram(gids, bins=gid_bins)
    
    mean_firing_rates = hist / 2.0

    # if gray_screen > 0:
    #     mean_firing_rates = hist / ((3000.0 - gray_screen) / 1000.0)
    # else:
    #     mean_firing_rates = hist / (3000.0 / 1000.0)
    return mean_firing_rates


def calculate_Rates_DF(numNrns, trials=10, angles=np.arange(0, 360, 45), basedir=None):

    Rates_DF = pd.DataFrame(
        index=range(numNrns * len(angles)),
        columns=["DG_angle", "node_id", "Avg_rate(Hz)", "SD_rate(Hz)"],
    )

    for i, ori in enumerate(angles):
        firingRatesTrials = np.zeros((trials, numNrns))

        for trial in range(trials):

            spikes_file_name = (
                f"{basedir}/8dir_10trials/angle{ori}_trial{trial}/spikes.csv"
            )
            # spikes = np.loadtxt(spikes_file_name, unpack=True)
            spikes = pd.read_csv(spikes_file_name, sep=" ")
            # ts, gids = spikes
            # print(spikes)
            ts = np.array(spikes["timestamps"])
            gids = np.array(spikes["node_ids"], dtype=int)
            # gids = gids.astype(int)

            firingRates = calculateFiringRate(gids, ts, numNrns, gray_screen=500.0)
            # print(spikes_file_name)

            firingRatesTrials[trial, :] = firingRates

        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "DG_angle"] = ori
        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "node_id"] = np.arange(
            numNrns
        )
        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "Avg_rate(Hz)"] = np.mean(
            firingRatesTrials, axis=0
        )
        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "SD_rate(Hz)"] = np.std(
            firingRatesTrials, axis=0
        )

    Rates_DF.to_csv(basedir + "/metrics/Rates_DF.csv", sep=" ", index=False)
    return Rates_DF


def calculate_OSI_DSI_from_DF(rates_df, basedir):
    num_angles = 8

    num_neurons = len(rates_df) // num_angles
    all_rates = np.zeros((num_neurons, num_angles))

    angles = range(0, 360, 45)
    angle_counts = 0
    for angle, g in rates_df.groupby("DG_angle"):
        all_rates[:, angle_counts] = g["Avg_rate(Hz)"]
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

    osi_df.to_csv(basedir + "/metrics/OSI_DSI_DF.csv", sep=" ", index=False)


if __name__ == "__main__":
    args = sys.argv
    basedir = args[1]
    network_dir = basedir + "/network"
    set_name = basedir + "/8dir_10trials"
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
    numNrns = len(nodes_DF)

    print("FR calculation started...")
    Rates_DF = calculate_Rates_DF(
        numNrns, trials=trials, angles=angles, basedir=basedir
    )
    print("Done Rates DF!")

    # Rates_DF = pd.read_csv(set_name + "/Rates_DF.csv", sep=" ", index_col=False)
    calculate_OSI_DSI_from_DF(Rates_DF, basedir=basedir)
    print("Done with all!")
