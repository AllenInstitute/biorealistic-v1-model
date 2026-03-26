#!/usr/bin/env python3
"""Debug metrics calculation."""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path

BASE_DIR = Path("core_nll_0")
NETWORK_DIR = BASE_DIR / "network"

# Load node info
print("Loading nodes...")
with h5py.File(NETWORK_DIR / "v1_nodes.h5", "r") as f:
    node_ids = f["nodes"]["v1"]["node_id"][:]
    node_type_ids = f["nodes"]["v1"]["node_type_id"][:]

node_types_df = pd.read_csv(NETWORK_DIR / "v1_node_types.csv", delimiter=" ")

df = pd.DataFrame({
    "node_id": node_ids,
    "node_type_id": node_type_ids,
})

df = df.merge(node_types_df[["node_type_id", "pop_name", "ei"]], on="node_type_id")

print(f"Total nodes: {len(df)}")
print(f"Excitatory nodes: {np.sum(df['ei'] == 'e')}")

# Get first 10 exc nodes
exc_nodes = df[df["ei"] == "e"]["node_id"].values[:10]
print(f"\nFirst 10 exc nodes: {exc_nodes}")

# Try loading one spike file
spike_file = BASE_DIR / "8dir_10trials_pv_high_neg1000/angle0_trial0/spikes.h5"
print(f"\nLoading {spike_file}...")
with h5py.File(spike_file, "r") as f:
    times = f["spikes"]["v1"]["timestamps"][:]
    spike_node_ids = f["spikes"]["v1"]["node_ids"][:]

print(f"Total spikes: {len(times)}")
print(f"Time range: {times.min():.1f} - {times.max():.1f} ms")
print(f"Unique neurons spiking: {len(np.unique(spike_node_ids))}")

# Check if our exc nodes are in the spike data
for node_id in exc_nodes:
    mask = spike_node_ids == node_id
    n_spikes = np.sum(mask)
    print(f"Node {node_id}: {n_spikes} spikes")
