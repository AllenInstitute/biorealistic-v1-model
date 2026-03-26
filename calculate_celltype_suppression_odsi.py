#!/usr/bin/env python3
"""
Calculate OSI/DSI metrics for cell-type suppression experiments using the standard method.
"""

import numpy as np
import pandas as pd
import math
import h5py
from pathlib import Path

# Cell-type suppression experiments
EXPERIMENTS = {
    "pv_high_neg1000": "8dir_10trials_pv_high_neg1000",
    "pv_low_neg1000": "8dir_10trials_pv_low_neg1000",
    "sst_high_neg1000": "8dir_10trials_sst_high_neg1000",
    "sst_low_neg1000": "8dir_10trials_sst_low_neg1000",
    "vip_high_neg1000": "8dir_10trials_vip_high_neg1000",
    "vip_low_neg1000": "8dir_10trials_vip_low_neg1000",
}


def calculateFiringRate(gids, ts, numNrns, start_time=0.0, duration=2.0):
    """Calculate firing rate from spike data."""
    start_time = start_time * 1000.0
    end_time = start_time + duration * 1000.0

    gids = gids[np.where(np.logical_and(ts > start_time, ts <= end_time))[0]]
    gid_bins = np.arange(0 - 0.5, numNrns + 0.5, 1)
    hist, bins = np.histogram(gids, bins=gid_bins)
    mean_firing_rates = hist / duration

    return mean_firing_rates


def calculate_Rates_DF(numNrns, trials=10, angles=np.arange(0, 360, 45), set_name=None):
    """Calculate firing rates across all angles and trials."""
    Rates_DF = pd.DataFrame(
        index=range(numNrns * len(angles)),
        columns=[
            "DG_angle",
            "node_id",
            "Ave_Rate(Hz)",
            "SD_rate(Hz)",
            "Spont_rate(Hz)",
        ],
    )

    for i, ori in enumerate(angles):
        firingRatesTrials = np.empty((trials, numNrns))
        firingRatesTrials[:] = np.nan
        spontRatesTrials = np.empty((trials, numNrns))
        spontRatesTrials[:] = np.nan

        for trial in range(trials):
            spikes_file_name = f"{set_name}/angle{ori}_trial{trial}/spikes.csv"

            try:
                spikes = pd.read_csv(spikes_file_name, sep=" ")
            except FileNotFoundError:
                print(f"    File not found: {spikes_file_name}. Skipping...")
                continue

            ts = np.array(spikes["timestamps"])
            gids = np.array(spikes["node_ids"], dtype=int)

            firingRates = calculateFiringRate(gids, ts, numNrns, start_time=0.5)
            spontRates = calculateFiringRate(
                gids, ts, numNrns, start_time=0.1, duration=0.4
            )

            firingRatesTrials[trial, :] = firingRates
            spontRatesTrials[trial, :] = spontRates

        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "DG_angle"] = ori
        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "node_id"] = np.arange(numNrns)
        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "Ave_Rate(Hz)"] = np.nanmean(
            firingRatesTrials, axis=0
        )
        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "SD_rate(Hz)"] = np.nanstd(
            firingRatesTrials, axis=0
        )
        Rates_DF.loc[i * numNrns : (i + 1) * numNrns - 1, "Spont_rate(Hz)"] = np.nanmean(
            spontRatesTrials, axis=0
        )

    return Rates_DF


def calculate_OSI_DSI_from_DF(rates_df, basedir):
    """Calculate OSI and DSI from rates dataframe."""
    num_angles = 8
    num_neurons = len(rates_df) // num_angles
    all_rates = np.zeros((num_neurons, num_angles))
    all_spont = np.zeros((num_neurons, num_angles))

    angles = range(0, 360, 45)
    angle_counts = 0
    for angle, g in rates_df.groupby("DG_angle"):
        all_rates[:, angle_counts] = g["Ave_Rate(Hz)"]
        all_spont[:, angle_counts] = g["Spont_rate(Hz)"]
        angle_counts += 1

    preferred_angle_ind = np.argmax(all_rates, axis=1)
    preferred_rates = all_rates[np.arange(len(preferred_angle_ind)), preferred_angle_ind]

    phase_rad = np.array(list(angles)) * math.pi / 180

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
    osi_df["preferred_angle"] = np.array(list(angles))[preferred_angle_ind]
    osi_df["max_mean_rate(Hz)"] = preferred_rates
    osi_df["Ave_Rate(Hz)"] = np.mean(all_rates, axis=1)
    osi_df["Spont_Rate(Hz)"] = np.mean(all_spont, axis=1)
    return osi_df


def get_num_neurons(base_dir: Path):
    """Get total number of V1 neurons."""
    network_dir = base_dir / "network"
    with h5py.File(network_dir / "v1_nodes.h5", "r") as f:
        return len(f["nodes"]["v1"]["node_id"][:])


def main():
    print("=" * 80)
    print("CALCULATING OSI/DSI FOR CELL-TYPE SUPPRESSION EXPERIMENTS")
    print("=" * 80)

    networks = [f"core_nll_{i}" for i in range(10)]
    
    for net in networks:
        base_dir = Path(net)
        if not base_dir.exists():
            continue
            
        print(f"\n{'='*40}")
        print(f"PROCESSING NETWORK: {net}")
        print(f"{'='*40}")
        
        metrics_dir = base_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True, parents=True)
        
        num_neurons = get_num_neurons(base_dir)
        print(f"\nTotal V1 neurons: {num_neurons}")

        for exp_name, exp_dir in EXPERIMENTS.items():
            exp_path = base_dir / exp_dir
            if not exp_path.exists():
                print(f"\nWARNING: {exp_path} not found, skipping")
                continue

            # Check if metrics already exist
            osi_output = metrics_dir / f"OSI_DSI_DF_{exp_name}.csv"
            rates_output = metrics_dir / f"Rates_DF_{exp_name}.csv"
            
            if osi_output.exists() and rates_output.exists():
                print(f"\nSkipping {exp_name} in {net} - metrics already exist")
                continue

            print(f"\nProcessing {exp_name}...")
            print(f"  Directory: {exp_path}")

            # Calculate rates dataframe
            print("  Calculating firing rates...")
            rates_df = calculate_Rates_DF(
                numNrns=num_neurons,
                trials=10,
                angles=np.arange(0, 360, 45),
                set_name=str(exp_path)
            )

            # Calculate OSI/DSI
            print("  Calculating OSI/DSI...")
            osi_df = calculate_OSI_DSI_from_DF(rates_df, str(exp_path))

            # Save results
            osi_df.to_csv(osi_output, index=False, sep=" ")
            rates_df.to_csv(rates_output, index=False, sep=" ")

            print(f"  ✓ Saved OSI/DSI to {osi_output.name}")
            print(f"  ✓ Saved Rates to {rates_output.name}")

            # Print summary stats
            print(f"  Summary:")
            print(f"    Mean OSI: {osi_df['OSI'].mean():.4f}")
            print(f"    Mean DSI: {osi_df['DSI'].mean():.4f}")
            print(f"    Mean rate: {osi_df['Ave_Rate(Hz)'].mean():.2f} Hz")

    print("\n" + "=" * 80)
    print("DONE - Metrics saved to metrics/ directories")
    print("=" * 80)


if __name__ == "__main__":
    main()
