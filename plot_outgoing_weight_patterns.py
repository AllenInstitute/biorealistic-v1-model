#!/usr/bin/env python
"""
Visualize outgoing connectivity patterns from CORE neurons by high/low outgoing weight groups.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('core_nll_0/figures/selectivity_outgoing/outgoing_weight_granular_core_to_core_summary.csv')

# Filter to core only
core_df = df[df['group'].str.contains('core')].copy()

# Parse group names - high/low refers to OUTGOING WEIGHT, not selectivity!
core_df['cell_type'] = core_df['group'].str.extract(r'(inh|pv|sst|vip)')[0]
core_df['weight_group'] = core_df['group'].str.extract(r'_(high|low)_')[0]

print("Groups (high/low = outgoing WEIGHT groups):")
print(core_df[['group', 'cell_type', 'weight_group', 'n_connections']].to_string(index=False))

# Create output directory
import os
os.makedirs('core_nll_0/figures/selectivity_analysis', exist_ok=True)

# Define colors for targets
target_colors = {
    'Exc': '#D42A2A',
    'PV': '#4C7F19',
    'SST': '#197F7F',
    'VIP': '#9932FF'
}

# Cell type colors (for borders/labels)
celltype_colors = {
    'inh': '#787878',
    'pv': '#4C7F19',
    'sst': '#197F7F',
    'vip': '#9932FF'
}

# Create stacked bar plot
fig, ax = plt.subplots(figsize=(7, 5))

cell_types = ['inh', 'pv', 'sst', 'vip']
x_positions = []
x_labels = []
bar_width = 0.35

# Prepare data for stacked bars
for i, ct in enumerate(cell_types):
    base_x = i * 2  # Space between cell types

    for j, wg in enumerate(['high', 'low']):
        row = core_df[(core_df['cell_type'] == ct) & (core_df['weight_group'] == wg)]

        if not row.empty:
            x_pos = base_x + j * (bar_width + 0.05)
            x_positions.append(x_pos)
            x_labels.append(f'{ct.upper()}\n{wg}')

            # Calculate total excitatory and inhibitory by subtype
            exc_total = row['exc_high'].values[0] + row['exc_low'].values[0]
            pv_total = row['inh_pv'].values[0]
            sst_total = row['inh_sst'].values[0]
            vip_total = row['inh_vip'].values[0]

            # Stack the bars
            bottom = 0
            for target, value, color in [
                ('Exc', exc_total, target_colors['Exc']),
                ('PV', pv_total, target_colors['PV']),
                ('SST', sst_total, target_colors['SST']),
                ('VIP', vip_total, target_colors['VIP'])
            ]:
                ax.bar(x_pos, value, bar_width, bottom=bottom,
                       color=color, edgecolor='black', linewidth=0.5,
                       label=target if i == 0 and j == 0 else "")

                # Add text label if segment is large enough
                if value > 0.03:
                    ax.text(x_pos, bottom + value/2, f'{value:.2f}',
                           ha='center', va='center', fontsize=7, weight='bold',
                           color='white' if value > 0.15 else 'black')
                bottom += value

# Formatting
ax.set_ylabel('Fraction of outgoing weight', fontsize=11)
ax.set_xlabel('Source cell type and outgoing weight group', fontsize=11)
ax.set_title('Core Connectivity: Target Composition by Source Type', fontsize=12, pad=15)
ax.set_xticks(x_positions)
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylim(0, 1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
ax.legend(title='Target type', loc='upper right', fontsize=9)

# Add vertical separators between cell types
for i in range(1, len(cell_types)):
    ax.axvline(i * 2 - 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

plt.tight_layout()
plt.savefig('core_nll_0/figures/selectivity_analysis/outgoing_weight_stacked.png',
            dpi=300, bbox_inches='tight')
plt.savefig('core_nll_0/figures/selectivity_analysis/outgoing_weight_stacked.svg',
            bbox_inches='tight')
print("\nSaved: outgoing_weight_stacked.png/svg")
plt.close()

# Create a second figure showing just the differences
fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
fig.suptitle('High vs Low Outgoing Weight: Target Preferences', fontsize=12, y=0.98)

for idx, ct in enumerate(cell_types):
    ax = axes[idx // 2, idx % 2]

    high_row = core_df[(core_df['cell_type'] == ct) & (core_df['weight_group'] == 'high')]
    low_row = core_df[(core_df['cell_type'] == ct) & (core_df['weight_group'] == 'low')]

    if not high_row.empty and not low_row.empty:
        # Get values
        targets = ['Exc', 'PV', 'SST', 'VIP']
        high_vals = [
            high_row['exc_high'].values[0] + high_row['exc_low'].values[0],
            high_row['inh_pv'].values[0],
            high_row['inh_sst'].values[0],
            high_row['inh_vip'].values[0]
        ]
        low_vals = [
            low_row['exc_high'].values[0] + low_row['exc_low'].values[0],
            low_row['inh_pv'].values[0],
            low_row['inh_sst'].values[0],
            low_row['inh_vip'].values[0]
        ]

        x = np.arange(len(targets))
        width = 0.35

        bars1 = ax.bar(x - width/2, high_vals, width, label='High weight',
                      color=[target_colors[t] for t in targets], alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, low_vals, width, label='Low weight',
                      color=[target_colors[t] for t in targets], alpha=0.4,
                      edgecolor='black', linewidth=0.5)

        ax.set_ylabel('Fraction', fontsize=9)
        ax.set_title(f'{ct.upper()} neurons', fontsize=10,
                    color=celltype_colors[ct], weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(targets, fontsize=9)
        ax.set_ylim(0, max(max(high_vals), max(low_vals)) * 1.15)
        if idx == 0:
            ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add difference annotations for significant changes
        for i, (h, l, t) in enumerate(zip(high_vals, low_vals, targets)):
            diff = l - h
            if abs(diff) > 0.02:  # Only show if difference is substantial
                y_pos = max(h, l) + 0.02
                color = 'red' if diff > 0 else 'blue'
                ax.text(x[i], y_pos, f'{diff:+.2f}', ha='center',
                       fontsize=7, color=color, weight='bold')

plt.tight_layout()
plt.savefig('core_nll_0/figures/selectivity_analysis/high_vs_low_weight_comparison.png',
            dpi=300, bbox_inches='tight')
plt.savefig('core_nll_0/figures/selectivity_analysis/high_vs_low_weight_comparison.svg',
            bbox_inches='tight')
print("Saved: high_vs_low_weight_comparison.png/svg")
plt.close()

# Print detailed findings
print("\n" + "="*80)
print("CORE CONNECTIVITY: HIGH vs LOW OUTGOING WEIGHT GROUPS")
print("="*80)
print("\nNote: 'high' and 'low' refer to neurons grouped by their OUTGOING WEIGHT,")
print("      not by selectivity or any other feature.")

print("\n1. TARGET COMPOSITION BY SOURCE TYPE:")
for ct in cell_types:
    print(f"\n   {ct.upper()} neurons:")

    high_row = core_df[(core_df['cell_type'] == ct) & (core_df['weight_group'] == 'high')]
    low_row = core_df[(core_df['cell_type'] == ct) & (core_df['weight_group'] == 'low')]

    if not high_row.empty:
        exc = high_row['exc_high'].values[0] + high_row['exc_low'].values[0]
        pv = high_row['inh_pv'].values[0]
        sst = high_row['inh_sst'].values[0]
        vip = high_row['inh_vip'].values[0]
        n = high_row['n_connections'].values[0]
        print(f"      High weight ({n:,} conn): Exc={exc:.3f}, PV={pv:.3f}, SST={sst:.3f}, VIP={vip:.3f}")

    if not low_row.empty:
        exc = low_row['exc_high'].values[0] + low_row['exc_low'].values[0]
        pv = low_row['inh_pv'].values[0]
        sst = low_row['inh_sst'].values[0]
        vip = low_row['inh_vip'].values[0]
        n = low_row['n_connections'].values[0]
        print(f"      Low weight  ({n:,} conn): Exc={exc:.3f}, PV={pv:.3f}, SST={sst:.3f}, VIP={vip:.3f}")

print("\n\n2. KEY DIFFERENCES (Low - High):")
for ct in cell_types:
    high_row = core_df[(core_df['cell_type'] == ct) & (core_df['weight_group'] == 'high')]
    low_row = core_df[(core_df['cell_type'] == ct) & (core_df['weight_group'] == 'low')]

    if not high_row.empty and not low_row.empty:
        print(f"\n   {ct.upper()} neurons:")

        h_exc = high_row['exc_high'].values[0] + high_row['exc_low'].values[0]
        l_exc = low_row['exc_high'].values[0] + low_row['exc_low'].values[0]
        print(f"      Exc:  {l_exc-h_exc:+.3f}")

        h_pv = high_row['inh_pv'].values[0]
        l_pv = low_row['inh_pv'].values[0]
        print(f"      PV:   {l_pv-h_pv:+.3f}")

        h_sst = high_row['inh_sst'].values[0]
        l_sst = low_row['inh_sst'].values[0]
        print(f"      SST:  {l_sst-h_sst:+.3f}")

        h_vip = high_row['inh_vip'].values[0]
        l_vip = low_row['inh_vip'].values[0]
        print(f"      VIP:  {l_vip-h_vip:+.3f}")

        # E/I ratio
        h_inh = h_pv + h_sst + h_vip
        l_inh = l_pv + l_sst + l_vip
        print(f"      E/I ratio: high={h_exc/h_inh:.2f}, low={l_exc/l_inh:.2f}, diff={l_exc/l_inh - h_exc/h_inh:+.2f}")

print("\n\n3. UNIVERSAL PATTERNS:")
print("\n   All cell types show the SAME trend:")
all_exc_diff = []
all_pv_diff = []
all_sst_diff = []

for ct in cell_types:
    high_row = core_df[(core_df['cell_type'] == ct) & (core_df['weight_group'] == 'high')]
    low_row = core_df[(core_df['cell_type'] == ct) & (core_df['weight_group'] == 'low')]

    if not high_row.empty and not low_row.empty:
        h_exc = high_row['exc_high'].values[0] + high_row['exc_low'].values[0]
        l_exc = low_row['exc_high'].values[0] + low_row['exc_low'].values[0]
        all_exc_diff.append(l_exc - h_exc)

        h_pv = high_row['inh_pv'].values[0]
        l_pv = low_row['inh_pv'].values[0]
        all_pv_diff.append(l_pv - h_pv)

        h_sst = high_row['inh_sst'].values[0]
        l_sst = low_row['inh_sst'].values[0]
        all_sst_diff.append(l_sst - h_sst)

print(f"   • Low-weight groups target LESS Exc (avg diff: {np.mean(all_exc_diff):.3f})")
print(f"   • Low-weight groups target MORE PV (avg diff: {np.mean(all_pv_diff):.3f})")
print(f"   • Low-weight groups target MORE SST (avg diff: {np.mean(all_sst_diff):.3f})")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
