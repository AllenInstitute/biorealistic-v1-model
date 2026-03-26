#!/usr/bin/env python3
"""Generate scatter plots of firing rate vs OSI/DSI/Image selectivity."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import glob


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


def load_all_dg_spike_counts(base_dir: str, node_ids: np.ndarray) -> np.ndarray:
    """
    Load all drifting grating spike counts efficiently using bincount.
    Returns: array of shape (neurons, 10 trials, 8 orientations)
    """
    print(f"Loading spike data from all trials for {len(node_ids)} neurons...")

    # First pass: find the global maximum node ID across all trials
    print("Finding global max node ID...")
    max_node_id = 0
    for trial in range(10):
        for angle in range(0, 360, 45):
            trial_dir = Path(base_dir) / f"8dir_10trials_bio_trained/angle{angle}_trial{trial}"
            spike_file = trial_dir / "spikes.h5"
            if spike_file.exists():
                try:
                    with h5py.File(spike_file, 'r') as f:
                        file_node_ids = f['spikes']['v1']['node_ids'][:]
                        max_node_id = max(max_node_id, int(np.max(file_node_ids)))
                except:
                    pass

    print(f"Global max node ID: {max_node_id}")

    # Initialize storage: (neurons, trials, orientations)
    spike_counts_full = np.zeros((max_node_id + 1, 10, 8), dtype=np.float32)

    # Second pass: load spike counts
    idx = 0
    for trial in range(10):
        for ori_idx, angle in enumerate(range(0, 360, 45)):
            trial_dir = Path(base_dir) / f"8dir_10trials_bio_trained/angle{angle}_trial{trial}"
            spike_file = trial_dir / "spikes.h5"

            if not spike_file.exists():
                continue

            try:
                with h5py.File(spike_file, 'r') as f:
                    spikes_data = f['spikes']['v1']
                    file_node_ids = spikes_data['node_ids'][:]

                    # Use bincount for fast counting - much faster than looping!
                    counts = np.bincount(file_node_ids, minlength=max_node_id + 1)
                    spike_counts_full[:, trial, ori_idx] = counts
            except Exception as e:
                print(f"Error loading {spike_file}: {e}")
                continue

            idx += 1
            if idx % 10 == 0:
                print(f"  Loaded {idx}/80 trial files...")

    print(f"Loaded all {idx} trial files")

    # Extract only the neurons we care about
    spike_counts = spike_counts_full[node_ids, :, :]
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


# Load data
DATA_PATH = Path("cell_categorization/core_nll_0_neuron_features.parquet")
df = pd.read_parquet(DATA_PATH)

print("Calculating trial-averaged sparsity for drifting gratings...")
# Check if cached file exists
cache_file = Path("core_nll_0/metrics/dg_trial_averaged_sparsity.npy")

if cache_file.exists():
    print(f"Loading cached sparsity from {cache_file}")
    df['dg_sparsity'] = np.load(cache_file)
else:
    print("Cache not found, computing sparsity (this may take a few minutes)...")
    # Load all spike counts at once
    spike_counts = load_all_dg_spike_counts("core_nll_0", df['node_id'].values)

    # Calculate sparsity for all neurons
    sparsities = calculate_trial_averaged_sparsities(spike_counts)
    df['dg_sparsity'] = sparsities

    # Save to cache
    np.save(cache_file, sparsities)
    print(f"Saved sparsity cache to {cache_file}")

print(f"Calculated sparsity for {df['dg_sparsity'].notna().sum()} neurons")

# Define coarse cell type mapping based on class column
def get_coarse_type(row):
    """Map cells to coarse types: Exc, PV, SST, VIP, Inh."""
    if row['ei'] == 'e':
        return 'Exc'
    else:
        # Get class2 information
        class2 = row.get('cell_class', '')
        if class2 == 'PV':
            return 'PV'
        elif class2 == 'SST':
            return 'SST'
        elif class2 == 'VIP':
            return 'VIP'
        else:
            return 'Inh'

df['coarse_type'] = df.apply(get_coarse_type, axis=1)

# Define colors for coarse types
COARSE_COLORS = {
    'Exc': '#D42A2A',  # Red from L2/3 Exc
    'PV': '#4C7F19',   # Green from PV
    'SST': '#197F7F',  # Cyan from SST
    'VIP': '#9932FF',  # Purple from VIP
    'Inh': '#787878',  # Gray for other inhibitory
}

# Filter to only valid data
df_valid = df.dropna(subset=['firing_rate', 'orientation_selectivity',
                              'dg_dsi', 'image_selectivity',
                              'dg_sparsity']).copy()

# Create figure for each metric
metrics = [
    ('orientation_selectivity', 'OSI'),
    ('dg_dsi', 'DSI'),
    ('image_selectivity', 'Image Selectivity'),
    ('dg_sparsity', 'DG Sparsity')
]

for metric_col, metric_label in metrics:
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot each coarse type
    for coarse_type in ['Exc', 'PV', 'SST', 'VIP', 'Inh']:
        subset = df_valid[df_valid['coarse_type'] == coarse_type]
        if len(subset) > 0:
            ax.scatter(
                subset[metric_col],
                subset['firing_rate'],
                c=COARSE_COLORS[coarse_type],
                label=coarse_type,
                alpha=0.5,
                s=3,
                edgecolors='none'
            )

    # Format axes
    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.01)

    plt.tight_layout()

    # Save figure
    output_name = f"firing_rate_vs_{metric_col}.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Saved {output_name}")
    plt.close()

print("Done!")
