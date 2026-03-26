#!/usr/bin/env python3
"""Compare all connections vs core-to-core connections."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "core_nll_0"
OUTPUT_DIR = BASE_DIR / "figures" / "selectivity_outgoing"

# Load both datasets
df_all = pd.read_csv(OUTPUT_DIR / "outgoing_weight_granular_summary.csv")
df_c2c = pd.read_csv(OUTPUT_DIR / "outgoing_weight_granular_core_to_core_summary.csv")


def main() -> None:
    # Focus on core source groups only
    core_groups = ["inh_high_core", "inh_low_core"]

    # Key columns to compare
    key_cols = ['inh_high', 'inh_low', 'inh_pv', 'inh_sst', 'inh_vip',
                'inh_high_pv', 'inh_high_sst', 'inh_high_vip',
                'inh_low_pv', 'inh_low_sst', 'inh_low_vip']

    col_labels = ['Inh\nHigh', 'Inh\nLow', 'All\nPV', 'All\nSST', 'All\nVIP',
                  'High\nPV', 'High\nSST', 'High\nVIP',
                  'Low\nPV', 'Low\nSST', 'Low\nVIP']

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # 1. All connections
    ax1 = axes[0]
    data_all = df_all[df_all['group'].isin(core_groups)][key_cols].values
    row_labels = df_all[df_all['group'].isin(core_groups)]['group'].values

    sns.heatmap(data_all, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=col_labels, yticklabels=row_labels,
                cbar_kws={'label': 'Weight Fraction'}, ax=ax1,
                linewidths=1, linecolor='black', vmin=0, vmax=0.16)

    ax1.set_title('All Connections (including to periphery)',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel('Target Type', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Source Group', fontsize=11, fontweight='bold')

    for x in [2, 5, 8]:
        ax1.axvline(x, color='black', linewidth=2)

    # 2. Core-to-core connections
    ax2 = axes[1]
    data_c2c = df_c2c[df_c2c['group'].isin(core_groups)][key_cols].values

    sns.heatmap(data_c2c, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=col_labels, yticklabels=row_labels,
                cbar_kws={'label': 'Weight Fraction'}, ax=ax2,
                linewidths=1, linecolor='black', vmin=0, vmax=0.16)

    ax2.set_title('Core-to-Core Connections Only (removes spatial bias)',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel('Target Type', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Source Group', fontsize=11, fontweight='bold')

    for x in [2, 5, 8]:
        ax2.axvline(x, color='black', linewidth=2)

    # 3. Difference (core-to-core minus all)
    ax3 = axes[2]
    diff_data = data_c2c - data_all

    sns.heatmap(diff_data, annot=True, fmt='+.3f', cmap='RdBu_r', center=0,
                xticklabels=col_labels, yticklabels=row_labels,
                cbar_kws={'label': 'Difference (C2C - All)'}, ax=ax3,
                linewidths=1, linecolor='black', vmin=-0.02, vmax=0.02)

    ax3.set_title('Difference: Core-to-Core minus All Connections\n' +
                  '(positive = stronger in core-to-core)',
                  fontsize=13, fontweight='bold', pad=15)
    ax3.set_xlabel('Target Type', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Source Group', fontsize=11, fontweight='bold')

    for x in [2, 5, 8]:
        ax3.axvline(x, color='black', linewidth=2)

    plt.tight_layout()

    # Save figure
    out_path = OUTPUT_DIR / 'core_to_core_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison to {out_path}")

    out_path_svg = OUTPUT_DIR / 'core_to_core_comparison.svg'
    plt.savefig(out_path_svg, bbox_inches='tight')
    print(f"Saved vector version to {out_path_svg}")

    # Print numerical comparison
    print("\n" + "="*80)
    print("QUANTITATIVE COMPARISON: Core-to-Core vs All Connections")
    print("="*80)

    for i, group in enumerate(row_labels):
        print(f"\n{group.upper()}:")
        print(f"  Total inhibitory targets:")
        inh_all = data_all[i, 0] + data_all[i, 1]  # inh_high + inh_low
        inh_c2c = data_c2c[i, 0] + data_c2c[i, 1]
        print(f"    All connections:        {inh_all:.3f}")
        print(f"    Core-to-core:           {inh_c2c:.3f}")
        print(f"    Difference:             {inh_c2c - inh_all:+.3f}")

        print(f"  High-weight inhibitory:")
        print(f"    All connections:        {data_all[i, 0]:.3f}")
        print(f"    Core-to-core:           {data_c2c[i, 0]:.3f}")
        print(f"    Difference:             {data_c2c[i, 0] - data_all[i, 0]:+.3f}")

        print(f"  PV targets:")
        pv_idx = key_cols.index('inh_pv')
        print(f"    All connections:        {data_all[i, pv_idx]:.3f}")
        print(f"    Core-to-core:           {data_c2c[i, pv_idx]:.3f}")
        print(f"    Difference:             {data_c2c[i, pv_idx] - data_all[i, pv_idx]:+.3f}")

    # Key finding
    print("\n" + "="*80)
    print("KEY FINDING:")
    print("="*80)
    inh_low_idx = list(row_labels).index('inh_low_core')
    inh_high_idx = list(row_labels).index('inh_high_core')

    inh_high_col = key_cols.index('inh_high')
    ratio_all = data_all[inh_low_idx, inh_high_col] / data_all[inh_high_idx, inh_high_col]
    ratio_c2c = data_c2c[inh_low_idx, inh_high_col] / data_c2c[inh_high_idx, inh_high_col]

    print(f"inh_low_core → high-weight inhibitory / inh_high_core → high-weight inhibitory:")
    print(f"  All connections:        {ratio_all:.2f}× higher")
    print(f"  Core-to-core:           {ratio_c2c:.2f}× higher")
    print(f"\nThe paradoxical effect is {'STRONGER' if ratio_c2c > ratio_all else 'WEAKER'} when considering only core-to-core connections.")
    print(f"This {'confirms' if ratio_c2c > ratio_all else 'suggests'} the effect is not due to spatial bias.")


if __name__ == "__main__":
    main()
