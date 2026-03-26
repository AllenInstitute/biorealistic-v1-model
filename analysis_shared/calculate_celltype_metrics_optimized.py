#!/usr/bin/env python3
"""
Calculate OSI/DSI metrics - OPTIMIZED VERSION.
Loads all spike data at once for efficiency.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict

BASE_DIR = Path("core_nll_0")
BASELINE_DIR = BASE_DIR / "8dir_10trials_plain"
NETWORK_DIR = BASE_DIR / "network"

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
SPONT_START = 100.0
SPONT_END = 500.0
SAMPLE_EVERY = 10


def load_all_responses(
    result_dir: Path, target_nodes: np.ndarray, max_node_id: int
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Load all responses for target nodes across all angles/trials."""
    print(f"  Loading spikes from {result_dir.name}...")

    # Initialize responses dict: node_id -> array of shape (n_angles, n_trials)
    stim_responses = {node_id: np.zeros((len(ANGLES), N_TRIALS)) for node_id in target_nodes}
    spont_responses = {node_id: np.zeros((len(ANGLES), N_TRIALS)) for node_id in target_nodes}

    for angle_idx, angle in enumerate(ANGLES):
        for trial in range(N_TRIALS):
            spike_file = result_dir / f"angle{angle}_trial{trial}" / "spikes.h5"
            if not spike_file.exists():
                continue

            with h5py.File(spike_file, "r") as f:
                times = f["spikes"]["v1"]["timestamps"][:]
                node_ids = f["spikes"]["v1"]["node_ids"][:]

            stim_mask = (times >= STIM_START) & (times < STIM_END)
            spont_mask = (times >= SPONT_START) & (times < SPONT_END)

            stim_counts = np.bincount(
                node_ids[stim_mask], minlength=max_node_id + 1
            )
            spont_counts = np.bincount(
                node_ids[spont_mask], minlength=max_node_id + 1
            )

            stim_duration = (STIM_END - STIM_START) / 1000.0
            spont_duration = (SPONT_END - SPONT_START) / 1000.0

            for node_id in target_nodes:
                stim_responses[node_id][angle_idx, trial] = (
                    stim_counts[node_id] / stim_duration
                )
                spont_responses[node_id][angle_idx, trial] = (
                    spont_counts[node_id] / spont_duration
                )

    return stim_responses, spont_responses


def calculate_osi_dsi(angle_responses: np.ndarray) -> Tuple[float, float]:
    """Calculate OSI and DSI from mean responses per angle."""
    angles_rad = np.deg2rad(ANGLES)

    # DSI
    x_sum = np.sum(angle_responses * np.cos(angles_rad))
    y_sum = np.sum(angle_responses * np.sin(angles_rad))
    dsi = (
        np.sqrt(x_sum**2 + y_sum**2) / np.sum(angle_responses)
        if np.sum(angle_responses) > 0
        else 0.0
    )

    # OSI
    x_sum_ori = np.sum(angle_responses * np.cos(2 * angles_rad))
    y_sum_ori = np.sum(angle_responses * np.sin(2 * angles_rad))
    osi = (
        np.sqrt(x_sum_ori**2 + y_sum_ori**2) / np.sum(angle_responses)
        if np.sum(angle_responses) > 0
        else 0.0
    )

    return osi, dsi


def main():
    print("=" * 80)
    print("CALCULATING CELL-TYPE SUPPRESSION METRICS (OPTIMIZED)")
    print("=" * 80)

    # Load nodes
    print("\nLoading node information...")
    with h5py.File(NETWORK_DIR / "v1_nodes.h5", "r") as f:
        node_ids = f["nodes"]["v1"]["node_id"][:]
        node_type_ids = f["nodes"]["v1"]["node_type_id"][:]

    node_types_df = pd.read_csv(NETWORK_DIR / "v1_node_types.csv", delimiter=" ")

    node_info = pd.DataFrame(
        {
            "node_id": node_ids,
            "node_type_id": node_type_ids,
        }
    )
    node_info = node_info.merge(
        node_types_df[["node_type_id", "pop_name", "ei"]], on="node_type_id"
    )

    # Sample excitatory neurons
    exc_nodes = node_info[node_info["ei"] == "e"]["node_id"].values
    sampled_nodes = exc_nodes[::SAMPLE_EVERY]
    print(f"Processing {len(sampled_nodes)} of {len(exc_nodes)} excitatory neurons")

    # Load baseline
    print("\nLoading baseline responses...")
    max_node_id = int(node_info["node_id"].max())

    baseline_stim, baseline_spont = load_all_responses(
        BASELINE_DIR, sampled_nodes, max_node_id
    )

    # Calculate baseline metrics
    print("Calculating baseline metrics...")
    baseline_metrics = {}
    for node_id in sampled_nodes:
        mean_per_angle = np.mean(baseline_stim[node_id], axis=1)
        osi, dsi = calculate_osi_dsi(mean_per_angle)
        baseline_metrics[node_id] = {
            "rate": np.mean(mean_per_angle),
            "spont": np.mean(np.mean(baseline_spont[node_id], axis=1)),
            "osi": osi,
            "dsi": dsi,
        }

    # Process experiments
    all_results = []

    for exp_name, exp_dir in EXPERIMENTS.items():
        exp_path = BASE_DIR / exp_dir
        if not exp_path.exists():
            print(f"\nWARNING: {exp_dir} not found, skipping")
            continue

        print(f"\nProcessing {exp_name}...")
        exp_stim, exp_spont = load_all_responses(exp_path, sampled_nodes, max_node_id)

        print("Calculating metrics...")
        for node_id in sampled_nodes:
            mean_per_angle = np.mean(exp_stim[node_id], axis=1)
            osi, dsi = calculate_osi_dsi(mean_per_angle)
            exp_rate = np.mean(mean_per_angle)
            exp_spont_mean = np.mean(np.mean(exp_spont[node_id], axis=1))

            baseline = baseline_metrics[node_id]

            delta_rate = exp_rate - baseline["rate"]
            pct_delta_rate = (
                100 * delta_rate / baseline["rate"] if baseline["rate"] > 0 else 0
            )

            delta_spont = exp_spont_mean - baseline["spont"]
            pct_delta_spont = (
                100 * delta_spont / baseline["spont"] if baseline["spont"] > 0 else 0
            )

            delta_osi = osi - baseline["osi"]
            pct_delta_osi = (
                100 * delta_osi / baseline["osi"] if baseline["osi"] > 0 else 0
            )

            delta_dsi = dsi - baseline["dsi"]
            pct_delta_dsi = (
                100 * delta_dsi / baseline["dsi"] if baseline["dsi"] > 0 else 0
            )

            node_row = node_info[node_info["node_id"] == node_id].iloc[0]

            all_results.append(
                {
                    "node_id": node_id,
                    "pop_name": node_row["pop_name"],
                    "experiment": exp_name,
                    "baseline_rate": baseline["rate"],
                    "exp_rate": exp_rate,
                    "baseline_spont": baseline["spont"],
                    "exp_spont": exp_spont_mean,
                    "delta_rate": delta_rate,
                    "pct_delta_rate": pct_delta_rate,
                    "delta_spont": delta_spont,
                    "pct_delta_spont": pct_delta_spont,
                    "baseline_osi": baseline["osi"],
                    "exp_osi": osi,
                    "delta_osi": delta_osi,
                    "pct_delta_osi": pct_delta_osi,
                    "baseline_dsi": baseline["dsi"],
                    "exp_dsi": dsi,
                    "delta_dsi": delta_dsi,
                    "pct_delta_dsi": pct_delta_dsi,
                }
            )

    # Save
    df = pd.DataFrame(all_results)
    output_file = "analysis_shared/celltype_suppression_metrics.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved {len(df)} rows to {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for exp_name in EXPERIMENTS.keys():
        exp_data = df[df["experiment"] == exp_name]
        if len(exp_data) > 0:
            print(f"\n{exp_name.upper()}:")
            print(
                f"  Mean Δ rate: {exp_data['delta_rate'].mean():+.2f} Hz ({exp_data['pct_delta_rate'].mean():+.1f}%)"
            )
            print(
                f"  Mean Δ OSI: {exp_data['delta_osi'].mean():+.4f} ({exp_data['pct_delta_osi'].mean():+.1f}%)"
            )
            print(
                f"  Mean Δ DSI: {exp_data['delta_dsi'].mean():+.4f} ({exp_data['pct_delta_dsi'].mean():+.1f}%)"
            )
            print(f"  N cells: {len(exp_data)}")


if __name__ == "__main__":
    main()
