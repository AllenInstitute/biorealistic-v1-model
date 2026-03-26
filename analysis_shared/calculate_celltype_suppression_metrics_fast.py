#!/usr/bin/env python3
"""
Calculate OSI/DSI metrics for cell-type specific suppression experiments - FAST VERSION.
Uses subset of neurons for speed.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Tuple

# Base directories
BASE_DIR = Path("core_nll_0")
BASELINE_DIR = BASE_DIR / "8dir_10trials"
NETWORK_DIR = BASE_DIR / "network"

# Experiment configurations
EXPERIMENTS = {
    "pv_high": "8dir_10trials_pv_high_neg1000",
    "pv_low": "8dir_10trials_pv_low_neg1000",
    "sst_high": "8dir_10trials_sst_high_neg1000",
    "sst_low": "8dir_10trials_sst_low_neg1000",
    "vip_high": "8dir_10trials_vip_high_neg1000",
    "vip_low": "8dir_10trials_vip_low_neg1000",
    "inh_high": "8dir_10trials_inh_high_neg1000",
    "inh_low": "8dir_10trials_inh_low_neg1000",
}

ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
N_TRIALS = 10
STIM_START = 1000.0
STIM_END = 2000.0

# Sample every Nth neuron for speed
SAMPLE_EVERY = 10


def load_node_info() -> pd.DataFrame:
    """Load V1 node information."""
    with h5py.File(NETWORK_DIR / "v1_nodes.h5", "r") as f:
        node_ids = f["nodes"]["v1"]["node_id"][:]
        node_type_ids = f["nodes"]["v1"]["node_type_id"][:]

    node_types_df = pd.read_csv(NETWORK_DIR / "v1_node_types.csv", delimiter=" ")

    df = pd.DataFrame({
        "node_id": node_ids,
        "node_type_id": node_type_ids,
    })

    df = df.merge(node_types_df[["node_type_id", "pop_name", "ei"]], on="node_type_id")
    return df


def calculate_responses_for_neuron(node_id: int, result_dir: Path) -> np.ndarray:
    """Calculate firing rate for all angles/trials for a single neuron."""
    responses = []

    for angle in ANGLES:
        for trial in range(N_TRIALS):
            spike_file = result_dir / f"angle{angle}_trial{trial}" / "spikes.h5"
            if not spike_file.exists():
                responses.append(0.0)
                continue

            with h5py.File(spike_file, "r") as f:
                times = f["spikes"]["v1"]["timestamps"][:]
                node_ids = f["spikes"]["v1"]["node_ids"][:]

            # Count spikes in stimulus window
            mask = (times >= STIM_START) & (times < STIM_END) & (node_ids == node_id)
            n_spikes = np.sum(mask)
            duration = (STIM_END - STIM_START) / 1000.0  # seconds
            fr = n_spikes / duration

            responses.append(fr)

    return np.array(responses).reshape(len(ANGLES), N_TRIALS)


def calculate_osi_dsi(angle_responses: np.ndarray) -> Tuple[float, float]:
    """Calculate OSI and DSI."""
    angles_rad = np.deg2rad(ANGLES)

    # DSI
    x_sum = np.sum(angle_responses * np.cos(angles_rad))
    y_sum = np.sum(angle_responses * np.sin(angles_rad))
    vector_sum = np.sqrt(x_sum**2 + y_sum**2)
    dsi = vector_sum / np.sum(angle_responses) if np.sum(angle_responses) > 0 else 0.0

    # OSI
    x_sum_ori = np.sum(angle_responses * np.cos(2 * angles_rad))
    y_sum_ori = np.sum(angle_responses * np.sin(2 * angles_rad))
    vector_sum_ori = np.sqrt(x_sum_ori**2 + y_sum_ori**2)
    osi = vector_sum_ori / np.sum(angle_responses) if np.sum(angle_responses) > 0 else 0.0

    return osi, dsi


def main():
    print("=" * 80)
    print("CALCULATING CELL-TYPE SUPPRESSION METRICS (FAST VERSION)")
    print("=" * 80)

    # Load nodes
    print("\nLoading node information...")
    node_info = load_node_info()
    exc_nodes = node_info[node_info["ei"] == "e"]["node_id"].values

    # Sample neurons
    sampled_exc_nodes = exc_nodes[::SAMPLE_EVERY]
    print(f"Processing {len(sampled_exc_nodes)} of {len(exc_nodes)} excitatory neurons (every {SAMPLE_EVERY}th)")

    # Calculate baseline responses for all sampled neurons
    print("\nCalculating baseline responses...")
    baseline_data = {}
    for i, node_id in enumerate(sampled_exc_nodes):
        if i % 100 == 0:
            print(f"  {i}/{len(sampled_exc_nodes)} neurons...")

        responses = calculate_responses_for_neuron(node_id, BASELINE_DIR)
        mean_per_angle = np.mean(responses, axis=1)
        mean_rate = np.mean(mean_per_angle)
        osi, dsi = calculate_osi_dsi(mean_per_angle)

        baseline_data[node_id] = {
            "mean_rate": mean_rate,
            "osi": osi,
            "dsi": dsi,
        }

    # Process each experiment
    all_results = []

    for exp_name, exp_dir in EXPERIMENTS.items():
        exp_path = BASE_DIR / exp_dir
        if not exp_path.exists():
            print(f"\nWARNING: {exp_dir} not found, skipping")
            continue

        print(f"\nProcessing {exp_name}...")

        for i, node_id in enumerate(sampled_exc_nodes):
            if i % 500 == 0:
                print(f"  {i}/{len(sampled_exc_nodes)} neurons...")

            # Get experiment responses
            responses = calculate_responses_for_neuron(node_id, exp_path)
            mean_per_angle = np.mean(responses, axis=1)
            mean_rate = np.mean(mean_per_angle)
            osi, dsi = calculate_osi_dsi(mean_per_angle)

            # Get baseline
            baseline = baseline_data[node_id]

            # Calculate deltas
            delta_rate = mean_rate - baseline["mean_rate"]
            pct_delta_rate = 100 * delta_rate / baseline["mean_rate"] if baseline["mean_rate"] > 0 else 0

            delta_osi = osi - baseline["osi"]
            pct_delta_osi = 100 * delta_osi / baseline["osi"] if baseline["osi"] > 0 else 0

            delta_dsi = dsi - baseline["dsi"]
            pct_delta_dsi = 100 * delta_dsi / baseline["dsi"] if baseline["dsi"] > 0 else 0

            # Get cell type
            node_row = node_info[node_info["node_id"] == node_id].iloc[0]

            all_results.append({
                "node_id": node_id,
                "pop_name": node_row["pop_name"],
                "experiment": exp_name,
                "baseline_rate": baseline["mean_rate"],
                "exp_rate": mean_rate,
                "delta_rate": delta_rate,
                "pct_delta_rate": pct_delta_rate,
                "baseline_osi": baseline["osi"],
                "exp_osi": osi,
                "delta_osi": delta_osi,
                "pct_delta_osi": pct_delta_osi,
                "baseline_dsi": baseline["dsi"],
                "exp_dsi": dsi,
                "delta_dsi": delta_dsi,
                "pct_delta_dsi": pct_delta_dsi,
            })

    # Save results
    df = pd.DataFrame(all_results)
    output_file = "analysis_shared/celltype_suppression_metrics.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved {len(df)} rows to {output_file}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for exp_name in EXPERIMENTS.keys():
        exp_data = df[df["experiment"] == exp_name]
        if len(exp_data) > 0:
            print(f"\n{exp_name.upper()}:")
            print(f"  Mean Δ rate: {exp_data['delta_rate'].mean():+.2f} Hz ({exp_data['pct_delta_rate'].mean():+.1f}%)")
            print(f"  Mean Δ OSI: {exp_data['delta_osi'].mean():+.4f} ({exp_data['pct_delta_osi'].mean():+.1f}%)")
            print(f"  Mean Δ DSI: {exp_data['delta_dsi'].mean():+.4f} ({exp_data['pct_delta_dsi'].mean():+.1f}%)")


if __name__ == "__main__":
    main()
