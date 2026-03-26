#!/usr/bin/env python3
"""Generate scatter plots of firing rate vs OSI/DSI for Neuropixels data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def load_cell_type_colors():
    """Load cell type color mapping from the naming scheme."""
    color_file = Path("base_props/cell_type_naming_scheme.csv")
    df = pd.read_csv(color_file, sep=r'\s+', engine='python')

    # Create mapping from cell_type to hex color
    color_map = {}
    for _, row in df.iterrows():
        cell_type = row['cell_type']
        color = row['hex']
        color_map[cell_type] = color

    return color_map


def get_layer_specific_type_color(cell_type):
    """Get color for layer-specific cell types (e.g., L6 Exc, L2/3 PV)."""
    if pd.isna(cell_type):
        return '#787878'  # Gray for unknown

    # Handle both simplified and layer-specific types
    if 'Exc' in cell_type:
        if 'L2/3' in cell_type:
            return '#D42A2A'  # L2/3 Exc
        elif 'L4' in cell_type:
            return '#C93C3C'  # L4 Exc
        elif 'L5' in cell_type:
            return '#AF5746'  # L5 Exc
        elif 'L6' in cell_type:
            return '#AF5757'  # L6 Exc
        else:
            return '#D42A2A'  # Default Exc
    elif 'PV' in cell_type:
        if 'L2/3' in cell_type:
            return '#4C7F19'  # L2/3 PV
        elif 'L4' in cell_type:
            return '#4F7A24'  # L4 PV
        elif 'L5' in cell_type:
            return '#557A30'  # L5 PV
        elif 'L6' in cell_type:
            return '#5B7A3D'  # L6 PV
        else:
            return '#4C7F19'  # Default PV
    elif 'SST' in cell_type:
        if 'L2/3' in cell_type:
            return '#197F7F'  # L2/3 SST
        elif 'L4' in cell_type:
            return '#247A7A'  # L4 SST
        elif 'L5' in cell_type:
            return '#307A7A'  # L5 SST
        elif 'L6' in cell_type:
            return '#3D7A7A'  # L6 SST
        else:
            return '#197F7F'  # Default SST
    elif 'VIP' in cell_type:
        if 'L2/3' in cell_type:
            return '#9932FF'  # L2/3 VIP
        elif 'L4' in cell_type:
            return '#9444E4'  # L4 VIP
        elif 'L5' in cell_type:
            return '#9152CF'  # L5 VIP
        elif 'L6' in cell_type:
            return '#875AB4'  # L6 VIP
        else:
            return '#9932FF'  # Default VIP
    elif 'Htr3a' in cell_type or 'L1' in cell_type:
        return '#787878'  # L1 Inh
    else:
        return '#787878'  # Gray for other


# Load Neuropixels data
print("Loading Neuropixels data...")
DATA_PATH = Path("neuropixels/metrics/OSI_DSI_neuropixels_v4.csv")
df = pd.read_csv(DATA_PATH, sep=r'\s+', quotechar='"')

print(f"Loaded {len(df)} units")
print(f"Columns: {df.columns.tolist()}")

# Aggregate cell types to coarse categories (Exc, PV, SST, VIP)
# Following the convention in analysis_shared/grouping.py
def get_coarse_cell_type(cell_type):
    """
    Aggregate layer-specific types to coarse categories.
    - All Exc types (L2/3_Exc, L4_Exc, L5_Exc, L6_Exc) → Exc
    - All layer-specific PV (L2/3_PV, L4_PV, L5_PV, L6_PV) → PV
    - All layer-specific SST (L2/3_SST, L4_SST, L5_SST, L6_SST) → SST
    - All layer-specific VIP (L2/3_VIP, L4_VIP, L5_VIP, L6_VIP) → VIP
    - L1 Htr3a → VIP (as per convention)
    """
    if pd.isna(cell_type):
        return None
    if 'Exc' in cell_type:
        return 'Exc'
    elif 'PV' in cell_type:
        return 'PV'
    elif 'SST' in cell_type:
        return 'SST'
    elif 'VIP' in cell_type or 'Htr3a' in cell_type:
        return 'VIP'
    else:
        return None

df['coarse_type'] = df['cell_type'].apply(get_coarse_cell_type)

# Map coarse types to colors
coarse_colors = {
    'Exc': '#D42A2A',   # Red
    'PV': '#4C7F19',    # Green
    'SST': '#197F7F',   # Cyan
    'VIP': '#9932FF',   # Purple
}

df['color'] = df['coarse_type'].map(coarse_colors)

# Filter to only valid data (remove NaN values)
df_valid = df.dropna(subset=['Ave_Rate(Hz)', 'OSI', 'DSI', 'coarse_type']).copy()
print(f"Valid units with OSI/DSI data: {len(df_valid)}")

# Get unique cell types for legend
unique_types = ['Exc', 'PV', 'SST', 'VIP']
print(f"Cell types: {unique_types}")

# Create figure with 2 panels (OSI and DSI only)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

# Panel 1: Firing Rate vs OSI
ax = axes[0]
for cell_type in unique_types:
    subset = df_valid[df_valid['coarse_type'] == cell_type]
    if len(subset) > 0:
        color = coarse_colors[cell_type]
        ax.scatter(
            subset['OSI'],
            subset['Ave_Rate(Hz)'],
            c=color,
            label=cell_type,
            alpha=0.5,
            s=5,
            edgecolors='none'
        )

ax.set_xlabel('OSI', fontsize=12)
ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, fontsize=10, ncol=1, loc='upper right')

# Panel 2: Firing Rate vs DSI
ax = axes[1]
for cell_type in unique_types:
    subset = df_valid[df_valid['coarse_type'] == cell_type]
    if len(subset) > 0:
        color = coarse_colors[cell_type]
        ax.scatter(
            subset['DSI'],
            subset['Ave_Rate(Hz)'],
            c=color,
            label=cell_type,
            alpha=0.5,
            s=5,
            edgecolors='none'
        )

ax.set_xlabel('DSI', fontsize=12)
ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save figure
output_name = "neuropixels_firing_rate_vs_selectivity.png"
plt.savefig(output_name, dpi=300, bbox_inches='tight')
print(f"Saved {output_name}")
plt.close()

print("Done!")
