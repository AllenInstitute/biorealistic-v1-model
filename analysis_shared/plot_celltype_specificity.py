#!/usr/bin/env python3
"""Visualize cell-type specific connectivity patterns."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "core_nll_0"
OUTPUT_DIR = BASE_DIR / "figures" / "selectivity_outgoing"

# Load datasets
df_all = pd.read_csv(OUTPUT_DIR / "outgoing_weight_granular_summary.csv")
df_c2c = pd.read_csv(OUTPUT_DIR / "outgoing_weight_granular_core_to_core_summary.csv")


def main() -> None:
    core_groups = ['inh_high_core', 'inh_low_core']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cell type preferences (as % of total inhibitory)
    ax = axes[0, 0]
    cell_types = ['PV', 'SST', 'VIP']
    x = np.arange(len(cell_types))
    width = 0.35

    for i, group in enumerate(core_groups):
        row_all = df_all[df_all['group'] == group].iloc[0]
        inh_total = sum(row_all[f'inh_{ct.lower()}'] for ct in ['pv', 'sst', 'vip', 'htr3a'])
        pcts = [100 * row_all[f'inh_{ct.lower()}'] / inh_total for ct in cell_types]

        offset = width * (i - 0.5)
        label = 'Low-weight inh' if 'low' in group else 'High-weight inh'
        color = '#e74c3c' if 'low' in group else '#3498db'
        ax.bar(x + offset, pcts, width, label=label, color=color, alpha=0.8)

    ax.set_xlabel('Target Cell Type', fontweight='bold')
    ax.set_ylabel('% of Total Inhibitory Weight', fontweight='bold')
    ax.set_title('Cell Type Preference (All Connections)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cell_types)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Enrichment ratios (inh_low / inh_high)
    ax = axes[0, 1]
    metrics = ['All PV', 'All SST', 'All VIP', 'High-wt PV', 'High-wt SST', 'High-wt VIP']
    cols_all = ['inh_pv', 'inh_sst', 'inh_vip', 'inh_high_pv', 'inh_high_sst', 'inh_high_vip']

    low_all = df_all[df_all['group'] == 'inh_low_core'].iloc[0]
    high_all = df_all[df_all['group'] == 'inh_high_core'].iloc[0]
    low_c2c = df_c2c[df_c2c['group'] == 'inh_low_core'].iloc[0]
    high_c2c = df_c2c[df_c2c['group'] == 'inh_high_core'].iloc[0]

    ratios_all = [low_all[col] / high_all[col] if high_all[col] > 0 else 0 for col in cols_all]
    ratios_c2c = [low_c2c[col] / high_c2c[col] if high_c2c[col] > 0 else 0 for col in cols_all]

    x = np.arange(len(metrics))
    ax.bar(x - width/2, ratios_all, width, label='All connections', color='#95a5a6', alpha=0.8)
    ax.bar(x + width/2, ratios_c2c, width, label='Core-to-core', color='#2c3e50', alpha=0.8)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Target Type', fontweight='bold')
    ax.set_ylabel('Enrichment (Low/High ratio)', fontweight='bold')
    ax.set_title('Low-weight preference over High-weight\n(values >1 = low preferentially targets)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 3. High vs Low weight breakdown by cell type
    ax = axes[1, 0]
    cell_types = ['PV', 'SST', 'VIP']

    for i, group in enumerate(core_groups):
        row_c2c = df_c2c[df_c2c['group'] == group].iloc[0]
        high_vals = [row_c2c[f'inh_high_{ct.lower()}'] for ct in cell_types]
        low_vals = [row_c2c[f'inh_low_{ct.lower()}'] for ct in cell_types]

        x_offset = i * (len(cell_types) + 1)
        x_pos = np.arange(len(cell_types)) + x_offset

        label = 'Low-weight inh' if 'low' in group else 'High-weight inh'
        color_base = '#e74c3c' if 'low' in group else '#3498db'

        ax.bar(x_pos, high_vals, 0.4, label=f'{label} → High-wt targets' if i == 0 else '',
               color=color_base, alpha=0.9)
        ax.bar(x_pos, low_vals, 0.4, bottom=high_vals,
               label=f'{label} → Low-wt targets' if i == 0 else '',
               color=color_base, alpha=0.4)

    ax.set_ylabel('Weight Fraction', fontweight='bold')
    ax.set_title('Target Weight Level by Cell Type (Core-to-Core)', fontweight='bold')
    xtick_pos = [1, 5]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(['High-weight\nSource', 'Low-weight\nSource'])

    # Add cell type labels
    for i, ct in enumerate(cell_types):
        ax.text(i, -0.015, ct, ha='center', fontsize=9, style='italic')
        ax.text(i + 4, -0.015, ct, ha='center', fontsize=9, style='italic')

    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 4. SST enrichment analysis
    ax = axes[1, 1]

    # Plot SST targeting across groups
    groups_plot = ['inh_high_core', 'inh_low_core']
    labels_plot = ['High-wt\nSource', 'Low-wt\nSource']

    sst_all = [df_all[df_all['group'] == g].iloc[0]['inh_sst'] for g in groups_plot]
    sst_c2c = [df_c2c[df_c2c['group'] == g].iloc[0]['inh_sst'] for g in groups_plot]

    x = np.arange(len(groups_plot))
    ax.bar(x - width/2, sst_all, width, label='All connections', color='#e67e22', alpha=0.7)
    ax.bar(x + width/2, sst_c2c, width, label='Core-to-core', color='#d35400', alpha=0.9)

    # Add ratio annotations
    for i, (g, label) in enumerate(zip(groups_plot, labels_plot)):
        ratio_all = df_all[df_all['group'] == g].iloc[0]['inh_sst'] / df_all[df_all['group'] == 'inh_high_core'].iloc[0]['inh_sst']
        ratio_c2c = df_c2c[df_c2c['group'] == g].iloc[0]['inh_sst'] / df_c2c[df_c2c['group'] == 'inh_high_core'].iloc[0]['inh_sst']

        if 'low' in g:
            ax.text(i, max(sst_all[i], sst_c2c[i]) + 0.003,
                   f'{ratio_all:.2f}× (all)\n{ratio_c2c:.2f}× (c2c)',
                   ha='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_ylabel('SST Weight Fraction', fontweight='bold')
    ax.set_xlabel('Source Group', fontweight='bold')
    ax.set_title('SST Targeting: Strongest Preference in Low-Weight Sources\n(1.98× enrichment in core-to-core)',
                 fontweight='bold', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    out_path = OUTPUT_DIR / 'celltype_specificity_analysis.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {out_path}")

    out_path_svg = OUTPUT_DIR / 'celltype_specificity_analysis.svg'
    plt.savefig(out_path_svg, bbox_inches='tight')
    print(f"Saved to {out_path_svg}")


if __name__ == "__main__":
    main()
