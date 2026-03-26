#!/usr/bin/env python3
"""Create annotated heatmap of granular connectivity patterns."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "core_nll_0"
OUTPUT_DIR = BASE_DIR / "figures" / "selectivity_outgoing"

INPUT_FILE = OUTPUT_DIR / "outgoing_weight_granular_summary.csv"


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    # Select columns for the heatmap
    # Main categories: exc/inh balance, then detailed inhibitory breakdown
    target_cols = [
        'exc_high', 'exc_low',
        'inh_high', 'inh_low',
        'inh_pv', 'inh_sst', 'inh_vip',
        'inh_high_pv', 'inh_high_sst', 'inh_high_vip',
        'inh_low_pv', 'inh_low_sst', 'inh_low_vip'
    ]

    # Create data matrix
    data = df[target_cols].values
    row_labels = df['group'].values
    col_labels = [
        'Exc\nHigh', 'Exc\nLow',
        'Inh\nHigh', 'Inh\nLow',
        'Inh\nPV', 'Inh\nSST', 'Inh\nVIP',
        'Inh High\nPV', 'Inh High\nSST', 'Inh High\nVIP',
        'Inh Low\nPV', 'Inh Low\nSST', 'Inh Low\nVIP'
    ]

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Main heatmap (all columns)
    ax1 = axes[0]
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=col_labels, yticklabels=row_labels,
                cbar_kws={'label': 'Weight Fraction'}, ax=ax1,
                linewidths=0.5, linecolor='gray')
    ax1.set_title('Granular Connectivity Patterns: Weight Fractions by Target Type',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Target Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Source Group', fontsize=12, fontweight='bold')

    # Add vertical separators to group columns
    for x in [2, 4, 7, 10]:
        ax1.axvline(x, color='black', linewidth=2)

    # Focused heatmap (just inhibitory targets)
    ax2 = axes[1]
    inh_cols = ['inh_high', 'inh_low', 'inh_pv', 'inh_sst', 'inh_vip',
                'inh_high_pv', 'inh_high_sst', 'inh_high_vip',
                'inh_low_pv', 'inh_low_sst', 'inh_low_vip']
    inh_col_labels = ['Inh\nHigh', 'Inh\nLow', 'All\nPV', 'All\nSST', 'All\nVIP',
                      'High\nPV', 'High\nSST', 'High\nVIP',
                      'Low\nPV', 'Low\nSST', 'Low\nVIP']

    inh_data = df[inh_cols].values
    sns.heatmap(inh_data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=inh_col_labels, yticklabels=row_labels,
                cbar_kws={'label': 'Weight Fraction'}, ax=ax2,
                linewidths=0.5, linecolor='gray')
    ax2.set_title('Inhibitory Target Breakdown (Detail View)',
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Inhibitory Target Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Source Group', fontsize=12, fontweight='bold')

    # Add vertical separators
    for x in [2, 5, 8]:
        ax2.axvline(x, color='black', linewidth=2)

    plt.tight_layout()

    # Save figure
    out_path = OUTPUT_DIR / 'granular_connectivity_heatmap.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {out_path}")

    out_path_svg = OUTPUT_DIR / 'granular_connectivity_heatmap.svg'
    plt.savefig(out_path_svg, bbox_inches='tight')
    print(f"Saved vector version to {out_path_svg}")

    # Create a third figure focused on the paradox comparison
    fig2, ax = plt.subplots(figsize=(10, 6))

    # Compare inh_high_core vs inh_low_core
    comparison_cols = ['inh_high', 'inh_low', 'inh_pv', 'inh_sst', 'inh_vip',
                       'inh_high_pv', 'inh_low_pv']
    comparison_labels = ['Total\nHigh Inh', 'Total\nLow Inh',
                        'All PV', 'All SST', 'All VIP',
                        'High-wt\nPV', 'Low-wt\nPV']

    # Get data for core groups only
    core_mask = df['group'].str.contains('core')
    core_data = df[core_mask][comparison_cols].values
    core_labels = df[core_mask]['group'].values

    sns.heatmap(core_data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=comparison_labels, yticklabels=core_labels,
                cbar_kws={'label': 'Weight Fraction'}, ax=ax,
                linewidths=1, linecolor='black', vmin=0, vmax=0.16)

    ax.set_title('Paradoxical Effect Explanation: Core Inhibitory Groups\n' +
                 'Low-weight inhibitory neurons preferentially target other inhibitory cells',
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_xlabel('Target Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Source Group', fontsize=12, fontweight='bold')

    # Add annotation for key finding
    ax.text(0.5, -0.25,
            'Key Finding: inh_low_core → high-weight inhibitory (10.6%) is 1.6× higher than inh_high_core (6.7%)\n' +
            'This disinhibition of high-weight inhibitory neurons explains the paradoxical suppression of excitatory activity',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    out_path2 = OUTPUT_DIR / 'paradox_explanation_heatmap.png'
    plt.savefig(out_path2, dpi=300, bbox_inches='tight')
    print(f"Saved paradox explanation heatmap to {out_path2}")

    out_path2_svg = OUTPUT_DIR / 'paradox_explanation_heatmap.svg'
    plt.savefig(out_path2_svg, bbox_inches='tight')
    print(f"Saved vector version to {out_path2_svg}")

    plt.show()


if __name__ == "__main__":
    main()
