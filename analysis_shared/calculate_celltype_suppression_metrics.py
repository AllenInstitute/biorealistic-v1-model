#!/usr/bin/env python3
"""
Calculate OSI/DSI metrics for cell-type specific suppression experiments.
Compares PV, SST, VIP high/low suppression to baseline and full inhibitory suppression.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Base directories
BASE_DIR = Path("core_nll_0")
BASELINE_DIR = BASE_DIR / "8dir_10trials_plain"
NETWORK_DIR = BASE_DIR / "network"

# Experiment configurations
EXPERIMENTS = {
    "pv_high": "8dir_10trials_pv_high_neg1000",
    "pv_low": "8dir_10trials_pv_low_neg1000",
    "sst_high": "8dir_10trials_sst_high_neg1000",
    "sst_low": "8dir_10trials_sst_low_neg1000",
    "vip_high": "8dir_10trials_vip_high_neg1000",
    "vip_low": "8dir_10trials_vip_low_neg1000",
    "inh_high": "8dir_10trials_inh_high_neg1000",  # Full inhibitory suppression (high outgoing)
    "inh_low": "8dir_10trials_inh_low_neg1000",  # Full inhibitory suppression (low outgoing)
}

ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
N_TRIALS = 10
STIM_START = 1000.0  # ms
STIM_END = 2000.0  # ms
SPONT_START = 100.0  # ms (0.1 s)
SPONT_END = 500.0  # ms (0.5 s)


def load_node_info() -> pd.DataFrame:
    """Load V1 node information including cell types."""
    with h5py.File(NETWORK_DIR / "v1_nodes.h5", "r") as f:
        node_ids = f["nodes"]["v1"]["node_id"][:]
        node_type_ids = f["nodes"]["v1"]["node_type_id"][:]
        x = f["nodes"]["v1"]["0"]["x"][:]
        z = f["nodes"]["v1"]["0"]["z"][:]

    node_types_df = pd.read_csv(NETWORK_DIR / "v1_node_types.csv", delimiter=" ")

    df = pd.DataFrame(
        {"node_id": node_ids, "node_type_id": node_type_ids, "x": x, "z": z}
    )

    df = df.merge(node_types_df[["node_type_id", "pop_name", "ei"]], on="node_type_id")
    return df


def load_spikes(spike_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load spike times and node IDs from h5 file."""
    with h5py.File(spike_file, "r") as f:
        times = f["spikes"]["v1"]["timestamps"][:]
        node_ids = f["spikes"]["v1"]["node_ids"][:]
    return times, node_ids


def calculate_firing_rate(
    times: np.ndarray,
    node_ids: np.ndarray,
    target_nodes: np.ndarray,
    start_time: float,
    end_time: float,
) -> float:
    """Calculate mean firing rate for target nodes in time window."""
    mask = (times >= start_time) & (times < end_time)
    mask &= np.isin(node_ids, target_nodes)

    n_spikes = np.sum(mask)
    duration = (end_time - start_time) / 1000.0  # convert to seconds
    n_neurons = len(target_nodes)

    return n_spikes / (duration * n_neurons) if n_neurons > 0 else 0.0


def calculate_osi_dsi_for_cell(angle_responses: np.ndarray) -> Tuple[float, float]:
    """Calculate OSI and DSI from responses to 8 directions."""
    angles_rad = np.deg2rad(ANGLES)

    # DSI: vector sum at preferred orientation
    x_sum = np.sum(angle_responses * np.cos(angles_rad))
    y_sum = np.sum(angle_responses * np.sin(angles_rad))
    vector_sum = np.sqrt(x_sum**2 + y_sum**2)
    dsi = vector_sum / np.sum(angle_responses) if np.sum(angle_responses) > 0 else 0.0

    # OSI: vector sum at double the angle (orientation selectivity)
    x_sum_ori = np.sum(angle_responses * np.cos(2 * angles_rad))
    y_sum_ori = np.sum(angle_responses * np.sin(2 * angles_rad))
    vector_sum_ori = np.sqrt(x_sum_ori**2 + y_sum_ori**2)
    osi = (
        vector_sum_ori / np.sum(angle_responses) if np.sum(angle_responses) > 0 else 0.0
    )

    return osi, dsi


def calculate_metrics_for_experiment(
    exp_name: str, exp_dir: str, baseline_dir: str, node_info: pd.DataFrame
) -> pd.DataFrame:
    """Calculate metrics comparing experiment to baseline."""
    print(f"\nProcessing {exp_name}...")

    exp_path = BASE_DIR / exp_dir
    baseline_path = BASELINE_DIR

    # Get excitatory neurons only
    exc_nodes = node_info[node_info["ei"] == "e"]["node_id"].values

    results = []

    # Process each excitatory neuron
    for node_id in exc_nodes:
        # Baseline responses
        baseline_responses = []
        baseline_spont = []
        for angle in ANGLES:
            for trial in range(N_TRIALS):
                spike_file = baseline_path / f"angle{angle}_trial{trial}" / "spikes.h5"
                if spike_file.exists():
                    times, node_ids = load_spikes(spike_file)
                    fr = calculate_firing_rate(
                        times, node_ids, np.array([node_id]), STIM_START, STIM_END
                    )
                    baseline_responses.append(fr)
                    fr_spont = calculate_firing_rate(
                        times, node_ids, np.array([node_id]), SPONT_START, SPONT_END
                    )
                    baseline_spont.append(fr_spont)

        if len(baseline_responses) == 0:
            continue

        baseline_responses = np.array(baseline_responses).reshape(len(ANGLES), N_TRIALS)
        baseline_mean_per_angle = np.mean(baseline_responses, axis=1)
        baseline_mean_rate = np.mean(baseline_mean_per_angle)
        baseline_osi, baseline_dsi = calculate_osi_dsi_for_cell(baseline_mean_per_angle)
        baseline_spont_responses = np.array(baseline_spont).reshape(len(ANGLES), N_TRIALS)
        baseline_spont_mean_per_angle = np.mean(baseline_spont_responses, axis=1)
        baseline_spont_mean = np.mean(baseline_spont_mean_per_angle)

        # Experiment responses
        exp_responses = []
        exp_spont = []
        for angle in ANGLES:
            for trial in range(N_TRIALS):
                spike_file = exp_path / f"angle{angle}_trial{trial}" / "spikes.h5"
                if spike_file.exists():
                    times, node_ids = load_spikes(spike_file)
                    fr = calculate_firing_rate(
                        times, node_ids, np.array([node_id]), STIM_START, STIM_END
                    )
                    exp_responses.append(fr)
                    fr_spont = calculate_firing_rate(
                        times, node_ids, np.array([node_id]), SPONT_START, SPONT_END
                    )
                    exp_spont.append(fr_spont)

        if len(exp_responses) == 0:
            continue

        exp_responses = np.array(exp_responses).reshape(len(ANGLES), N_TRIALS)
        exp_mean_per_angle = np.mean(exp_responses, axis=1)
        exp_mean_rate = np.mean(exp_mean_per_angle)
        exp_osi, exp_dsi = calculate_osi_dsi_for_cell(exp_mean_per_angle)
        exp_spont_responses = np.array(exp_spont).reshape(len(ANGLES), N_TRIALS)
        exp_spont_mean_per_angle = np.mean(exp_spont_responses, axis=1)
        exp_spont_mean = np.mean(exp_spont_mean_per_angle)

        # Calculate deltas
        delta_rate = exp_mean_rate - baseline_mean_rate
        pct_delta_rate = (
            100 * delta_rate / baseline_mean_rate if baseline_mean_rate > 0 else 0
        )
        delta_spont = exp_spont_mean - baseline_spont_mean
        pct_delta_spont = (
            100 * delta_spont / baseline_spont_mean if baseline_spont_mean > 0 else 0
        )

        delta_osi = exp_osi - baseline_osi
        pct_delta_osi = 100 * delta_osi / baseline_osi if baseline_osi > 0 else 0

        delta_dsi = exp_dsi - baseline_dsi
        pct_delta_dsi = 100 * delta_dsi / baseline_dsi if baseline_dsi > 0 else 0

        # Get cell type info
        node_row = node_info[node_info["node_id"] == node_id].iloc[0]

        results.append(
            {
                "node_id": node_id,
                "pop_name": node_row["pop_name"],
                "experiment": exp_name,
                "baseline_rate": baseline_mean_rate,
                "exp_rate": exp_mean_rate,
                "baseline_spont": baseline_spont_mean,
                "exp_spont": exp_spont_mean,
                "delta_rate": delta_rate,
                "pct_delta_rate": pct_delta_rate,
                "delta_spont": delta_spont,
                "pct_delta_spont": pct_delta_spont,
                "baseline_osi": baseline_osi,
                "exp_osi": exp_osi,
                "delta_osi": delta_osi,
                "pct_delta_osi": pct_delta_osi,
                "baseline_dsi": baseline_dsi,
                "exp_dsi": exp_dsi,
                "delta_dsi": delta_dsi,
                "pct_delta_dsi": pct_delta_dsi,
            }
        )

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("CALCULATING CELL-TYPE SPECIFIC SUPPRESSION METRICS")
    print("=" * 80)

    # Load node information
    print("\nLoading node information...")
    node_info = load_node_info()
    print(f"Loaded {len(node_info)} nodes")
    print(f"Excitatory neurons: {np.sum(node_info['ei'] == 'e')}")

    # Calculate metrics for all experiments
    all_results = []
    for exp_name, exp_dir in EXPERIMENTS.items():
        if (BASE_DIR / exp_dir).exists():
            df = calculate_metrics_for_experiment(
                exp_name, exp_dir, BASELINE_DIR, node_info
            )
            all_results.append(df)
        else:
            print(f"WARNING: {exp_dir} not found, skipping")

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Save results
    output_file = "analysis_shared/celltype_suppression_metrics.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved metrics to {output_file}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for exp_name in EXPERIMENTS.keys():
        exp_data = combined_df[combined_df["experiment"] == exp_name]
        if len(exp_data) > 0:
            print(f"\n{exp_name.upper()}:")
            print(
                f"  Mean Δ rate: {exp_data['delta_rate'].mean():.2f} Hz ({exp_data['pct_delta_rate'].mean():.1f}%)"
            )
            print(
                f"  Mean Δ spont: {exp_data['delta_spont'].mean():.2f} Hz ({exp_data['pct_delta_spont'].mean():.1f}%)"
            )
            print(
                f"  Mean Δ OSI: {exp_data['delta_osi'].mean():.4f} ({exp_data['pct_delta_osi'].mean():.1f}%)"
            )
            print(
                f"  Mean Δ DSI: {exp_data['delta_dsi'].mean():.4f} ({exp_data['pct_delta_dsi'].mean():.1f}%)"
            )
            print(f"  N cells: {len(exp_data)}")


if __name__ == "__main__":
    main()
