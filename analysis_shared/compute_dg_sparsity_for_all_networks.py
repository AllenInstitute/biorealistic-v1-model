#!/usr/bin/env python3
"""Compute trial-averaged sparsity for drifting gratings for a specific network."""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def calculate_lifetime_sparsity(rates_stim: np.ndarray) -> float:
    """
    Calculate lifetime sparsity across stimuli for one trial.
    Formula: (1 - (mean^2 / mean_sq)) / (1 - 1/n)
    """
    if rates_stim.size == 0:
        return np.nan
    mean_rate = float(np.mean(rates_stim))
    if mean_rate == 0.0:
        return np.nan
    mean_sq = float(np.mean(rates_stim**2))
    n = rates_stim.size
    return (1.0 - (mean_rate**2) / mean_sq) / (1.0 - 1.0 / n)


def load_dg_spike_counts(
    base_dir: Path, network_name: str, node_ids: np.ndarray
) -> np.ndarray:
    """
    Load all drifting grating spike counts efficiently using bincount.
    Returns: array of shape (neurons, 10 trials, 8 orientations)
    """
    print(f"Loading spike data from {network_name} for {len(node_ids)} neurons...")

    data_dir = base_dir / f"8dir_10trials_{network_name}"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # First pass: find the global maximum node ID across all trials
    print("Finding global max node ID...")
    max_node_id = 0
    for trial in range(10):
        for angle in range(0, 360, 45):
            trial_dir = data_dir / f"angle{angle}_trial{trial}"
            spike_file = trial_dir / "spikes.h5"
            if spike_file.exists():
                try:
                    with h5py.File(spike_file, 'r') as f:
                        if 'spikes' in f and 'v1' in f['spikes'] and 'node_ids' in f['spikes']['v1']:
                            file_node_ids = f['spikes']['v1']['node_ids'][:]
                            if file_node_ids.size > 0:
                                max_node_id = max(max_node_id, int(np.max(file_node_ids)))
                except Exception as e:
                    print(f"Error checking {spike_file}: {e}")
                    pass

    print(f"Global max node ID: {max_node_id}")
    
    # Ensure we can index up to max(node_ids)
    req_max = int(np.max(node_ids))
    if max_node_id < req_max:
        max_node_id = req_max

    # Initialize storage: (neurons, trials, orientations)
    # We use max_node_id + 1 to directly index by node_id
    spike_counts_full = np.zeros((max_node_id + 1, 10, 8), dtype=np.float32)

    # Second pass: load spike counts
    idx = 0
    for trial in range(10):
        for ori_idx, angle in enumerate(range(0, 360, 45)):
            trial_dir = data_dir / f"angle{angle}_trial{trial}"
            spike_file = trial_dir / "spikes.h5"

            if not spike_file.exists():
                continue

            try:
                with h5py.File(spike_file, 'r') as f:
                    if 'spikes' in f and 'v1' in f['spikes'] and 'node_ids' in f['spikes']['v1']:
                        file_node_ids = f['spikes']['v1']['node_ids'][:]
                        
                        if file_node_ids.size > 0:
                            # Use bincount for fast counting
                            counts = np.bincount(file_node_ids, minlength=max_node_id + 1)
                            # Trim if bincount returns more than expected (shouldn't happen given minlength logic)
                            if counts.size > max_node_id + 1:
                                counts = counts[:max_node_id + 1]
                            spike_counts_full[:counts.size, trial, ori_idx] = counts
            except Exception as e:
                print(f"Error loading {spike_file}: {e}")
                continue

            idx += 1
            if idx % 10 == 0:
                print(f"  Loaded {idx}/80 trial files...")

    print(f"Loaded all {idx} trial files")

    # Extract only the neurons we care about
    spike_counts = spike_counts_full[node_ids, :, :]
    
    # DEBUG: Check spike counts for a few node IDs
    print(f"DEBUG: Spike counts shape: {spike_counts.shape}")
    print(f"DEBUG: Total spikes in extracted: {np.sum(spike_counts)}")
    
    # Check L4 example (approximate ID range check or just check non-zero rows)
    non_zero_neurons = np.sum(np.sum(spike_counts, axis=(1,2)) > 0)
    print(f"DEBUG: Neurons with >0 spikes: {non_zero_neurons} / {len(node_ids)}")
    
    return spike_counts


def calculate_trial_averaged_sparsities(spike_counts: np.ndarray) -> np.ndarray:
    """
    Calculate trial-averaged sparsity for all neurons.
    Input: spike_counts of shape (neurons, trials, orientations)
    Output: array of sparsity values for each neuron
    """
    n_neurons = spike_counts.shape[0]
    sparsities = np.full(n_neurons, np.nan)

    for i in range(n_neurons):
        per_trial_sparsity = []
        for trial in range(spike_counts.shape[1]):
            trial_rates = spike_counts[i, trial, :]
            ls = calculate_lifetime_sparsity(trial_rates)
            if not np.isnan(ls):
                per_trial_sparsity.append(ls)

        if len(per_trial_sparsity) > 0:
            sparsities[i] = np.mean(per_trial_sparsity)

    return sparsities


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DG sparsity for a network.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("core_nll_0"),
        help="Network base directory containing 8dir_10trials_<network> outputs (default: core_nll_0).",
    )
    parser.add_argument("--network", type=str, required=True, help="Network name (e.g., bio_trained, plain)")
    args = parser.parse_args()

    base_dir = args.base_dir
    metrics_dir = base_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    network = args.network
    print(f"Processing network: {network}")

    node_file = base_dir / "network" / "v1_nodes.h5"
    if not node_file.exists():
        raise FileNotFoundError(f"Missing node file: {node_file}")

    with h5py.File(node_file, "r") as f:
        node_ids = f["nodes"]["v1"]["node_id"][:].astype(np.int64)

    output_file = metrics_dir / f"dg_trial_averaged_sparsity_{network}.npy"
    
    # Compute sparsity
    try:
        spike_counts = load_dg_spike_counts(base_dir, network, node_ids)
        sparsities = calculate_trial_averaged_sparsities(spike_counts)
        
        # Save to cache
        np.save(output_file, sparsities)
        print(f"Saved sparsity cache to {output_file}")
        print(f"Calculated sparsity for {np.sum(~np.isnan(sparsities))} neurons")
        
    except Exception as e:
        print(f"Failed to compute sparsity: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

