#!/usr/bin/env python3
"""Quick metrics calculation with heavy sampling."""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, Tuple

BASE_DIR = Path("core_nll_0")
NETWORK_DIR = BASE_DIR / "network"

# Only process these experiments for now
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
STIM_START, STIM_END = 1000.0, 2000.0
SAMPLE_EVERY = 50  # Much more aggressive sampling


def load_responses_batch(result_dir: Path, target_nodes: np.ndarray, max_trials_per_angle: int = 3) -> Dict:
    """Load responses - use only first few trials per angle for speed."""
    responses = {nid: np.zeros((len(ANGLES), max_trials_per_angle)) for nid in target_nodes}

    for ang_idx, angle in enumerate(ANGLES):
        for trial in range(max_trials_per_angle):
            spike_file = result_dir / f"angle{angle}_trial{trial}" / "spikes.h5"
            if not spike_file.exists():
                continue

            with h5py.File(spike_file, "r") as f:
                times = f["spikes"]["v1"]["timestamps"][:]
                node_ids = f["spikes"]["v1"]["node_ids"][:]

            mask = (times >= STIM_START) & (times < STIM_END)
            duration = (STIM_END - STIM_START) / 1000.0

            for nid in target_nodes:
                n_spikes = np.sum(mask & (node_ids == nid))
                responses[nid][ang_idx, trial] = n_spikes / duration

    return responses


def calc_osi_dsi(angle_resp: np.ndarray) -> Tuple[float, float]:
    """Calculate OSI/DSI."""
    angles_rad = np.deg2rad(ANGLES)
    total = np.sum(angle_resp)
    if total == 0:
        return 0.0, 0.0

    # DSI
    x_sum = np.sum(angle_resp * np.cos(angles_rad))
    y_sum = np.sum(angle_resp * np.sin(angles_rad))
    dsi = np.sqrt(x_sum**2 + y_sum**2) / total

    # OSI
    x_ori = np.sum(angle_resp * np.cos(2 * angles_rad))
    y_ori = np.sum(angle_resp * np.sin(2 * angles_rad))
    osi = np.sqrt(x_ori**2 + y_ori**2) / total

    return osi, dsi


print("=" * 80)
print("QUICK METRICS CALCULATION")
print("=" * 80)

# Load nodes
print("\nLoading nodes...")
with h5py.File(NETWORK_DIR / "v1_nodes.h5", "r") as f:
    all_node_ids = f["nodes"]["v1"]["node_id"][:]
    node_type_ids = f["nodes"]["v1"]["node_type_id"][:]

node_types = pd.read_csv(NETWORK_DIR / "v1_node_types.csv", delimiter=" ")
node_info = pd.DataFrame({"node_id": all_node_ids, "node_type_id": node_type_ids})
node_info = node_info.merge(node_types[["node_type_id", "pop_name", "ei"]], on="node_type_id")

exc_nodes = node_info[node_info["ei"] == "e"]["node_id"].values
sampled_nodes = exc_nodes[::SAMPLE_EVERY]
print(f"Processing {len(sampled_nodes)} neurons (every {SAMPLE_EVERY}th)")

# Baseline
print("\nBaseline...")
baseline_resp = load_responses_batch(BASE_DIR / "8dir_10trials", sampled_nodes)
baseline_metrics = {}
for nid in sampled_nodes:
    mean_per_ang = np.mean(baseline_resp[nid], axis=1)
    osi, dsi = calc_osi_dsi(mean_per_ang)
    baseline_metrics[nid] = {
        "rate": np.mean(mean_per_ang),
        "osi": osi,
        "dsi": dsi,
    }

# Process experiments
all_results = []
for exp_name, exp_dir in EXPERIMENTS.items():
    exp_path = BASE_DIR / exp_dir
    if not exp_path.exists():
        print(f"Skipping {exp_name} - not found")
        continue

    print(f"\n{exp_name}...")
    exp_resp = load_responses_batch(exp_path, sampled_nodes)

    for nid in sampled_nodes:
        mean_per_ang = np.mean(exp_resp[nid], axis=1)
        osi, dsi = calc_osi_dsi(mean_per_ang)
        rate = np.mean(mean_per_ang)

        b = baseline_metrics[nid]

        delta_rate = rate - b["rate"]
        pct_delta_rate = 100 * delta_rate / b["rate"] if b["rate"] > 0 else 0

        delta_osi = osi - b["osi"]
        pct_delta_osi = 100 * delta_osi / b["osi"] if b["osi"] > 0 else 0

        delta_dsi = dsi - b["dsi"]
        pct_delta_dsi = 100 * delta_dsi / b["dsi"] if b["dsi"] > 0 else 0

        node_row = node_info[node_info["node_id"] == nid].iloc[0]

        all_results.append({
            "node_id": nid,
            "pop_name": node_row["pop_name"],
            "experiment": exp_name,
            "baseline_rate": b["rate"],
            "exp_rate": rate,
            "delta_rate": delta_rate,
            "pct_delta_rate": pct_delta_rate,
            "baseline_osi": b["osi"],
            "exp_osi": osi,
            "delta_osi": delta_osi,
            "pct_delta_osi": pct_delta_osi,
            "baseline_dsi": b["dsi"],
            "exp_dsi": dsi,
            "delta_dsi": delta_dsi,
            "pct_delta_dsi": pct_delta_dsi,
        })

# Save
df = pd.DataFrame(all_results)
output = "analysis_shared/celltype_suppression_metrics.csv"
df.to_csv(output, index=False)
print(f"\n✓ Saved {len(df)} rows to {output}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for exp in EXPERIMENTS.keys():
    exp_data = df[df["experiment"] == exp]
    if len(exp_data) > 0:
        print(f"\n{exp.upper()}:")
        print(f"  Δ rate: {exp_data['delta_rate'].mean():+.2f} Hz ({exp_data['pct_delta_rate'].mean():+.1f}%)")
        print(f"  Δ OSI: {exp_data['delta_osi'].mean():+.4f} ({exp_data['pct_delta_osi'].mean():+.1f}%)")
        print(f"  Δ DSI: {exp_data['delta_dsi'].mean():+.4f} ({exp_data['pct_delta_dsi'].mean():+.1f}%)")
