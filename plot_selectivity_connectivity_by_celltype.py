#!/usr/bin/env python
"""
Visualize outgoing connectivity patterns from CORE neurons, organized by cell type and selectivity.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
df = pd.read_csv('core_nll_0/figures/selectivity_outgoing/outgoing_weight_granular_core_to_core_summary.csv')

# Filter to core only
core_df = df[df['group'].str.contains('core')].copy()

# Parse group names to extract cell type and selectivity
core_df['cell_type'] = core_df['group'].str.extract(r'(inh|pv|sst|vip)')[0]
core_df['selectivity'] = core_df['group'].str.extract(r'_(high|low)_')[0]

print("Available groups:")
print(core_df[['group', 'cell_type', 'selectivity', 'n_connections']].to_string(index=False))

# Create output directory
import os
os.makedirs('core_nll_0/figures/selectivity_analysis', exist_ok=True)

# Define colors
colors = {
    'inh': '#787878',
    'pv': '#4C7F19',
    'sst': '#197F7F',
    'vip': '#9932FF'
}

# Create comprehensive figure
fig = plt.figure(figsize=(8, 7))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.35)

# 1. E/I ratio by cell type and selectivity
ax1 = fig.add_subplot(gs[0, :])
cell_types = ['inh', 'pv', 'sst', 'vip']
x = np.arange(len(cell_types))
width = 0.35

ei_ratios_high = []
ei_ratios_low = []

for ct in cell_types:
    high_row = core_df[(core_df['cell_type'] == ct) & (core_df['selectivity'] == 'high')]
    low_row = core_df[(core_df['cell_type'] == ct) & (core_df['selectivity'] == 'low')]

    if not high_row.empty:
        e = high_row['exc_high'].values[0] + high_row['exc_low'].values[0]
        i = high_row['inh_high'].values[0] + high_row['inh_low'].values[0]
        ei_ratios_high.append(e/i if i > 0 else 0)
    else:
        ei_ratios_high.append(0)

    if not low_row.empty:
        e = low_row['exc_high'].values[0] + low_row['exc_low'].values[0]
        i = low_row['inh_high'].values[0] + low_row['inh_low'].values[0]
        ei_ratios_low.append(e/i if i > 0 else 0)
    else:
        ei_ratios_low.append(0)

bars1 = ax1.bar(x - width/2, ei_ratios_high, width, label='High selectivity',
                color=[colors[ct] for ct in cell_types], alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x + width/2, ei_ratios_low, width, label='Low selectivity',
                color=[colors[ct] for ct in cell_types], alpha=0.4, edgecolor='black', linewidth=0.5)

ax1.axhline(1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Balanced E/I')
ax1.set_ylabel('E/I ratio')
ax1.set_title('E/I Balance by Cell Type and Selectivity')
ax1.set_xticks(x)
ax1.set_xticklabels(['All Inh', 'PV', 'SST', 'VIP'])
ax1.legend(ncol=3, fontsize=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add value labels
for i, (h, l) in enumerate(zip(ei_ratios_high, ei_ratios_low)):
    if h > 0:
        ax1.text(x[i] - width/2, h + 0.1, f'{h:.1f}', ha='center', fontsize=7)
    if l > 0:
        ax1.text(x[i] + width/2, l + 0.1, f'{l:.1f}', ha='center', fontsize=7)

# 2. Inhibitory subtype targeting by source type
ax2 = fig.add_subplot(gs[1, 0])
subtypes = ['PV', 'SST', 'VIP']
x_sub = np.arange(len(subtypes))
width_sub = 0.15

for i, ct in enumerate(cell_types):
    high_row = core_df[(core_df['cell_type'] == ct) & (core_df['selectivity'] == 'high')]
    if not high_row.empty:
        vals = [high_row['inh_pv'].values[0], high_row['inh_sst'].values[0], high_row['inh_vip'].values[0]]
        ax2.bar(x_sub + i*width_sub - 1.5*width_sub, vals, width_sub,
                label=ct.upper() + ' high', color=colors[ct], alpha=0.7, edgecolor='black', linewidth=0.3)

ax2.set_ylabel('Fraction of outgoing weight')
ax2.set_title('Inhibitory Targeting (High Selectivity)')
ax2.set_xticks(x_sub)
ax2.set_xticklabels(subtypes)
ax2.legend(fontsize=7, ncol=2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 3. Inhibitory subtype targeting - low selectivity
ax3 = fig.add_subplot(gs[1, 1])
for i, ct in enumerate(cell_types):
    low_row = core_df[(core_df['cell_type'] == ct) & (core_df['selectivity'] == 'low')]
    if not low_row.empty:
        vals = [low_row['inh_pv'].values[0], low_row['inh_sst'].values[0], low_row['inh_vip'].values[0]]
        ax3.bar(x_sub + i*width_sub - 1.5*width_sub, vals, width_sub,
                label=ct.upper() + ' low', color=colors[ct], alpha=0.4, edgecolor='black', linewidth=0.3)

ax3.set_ylabel('Fraction of outgoing weight')
ax3.set_title('Inhibitory Targeting (Low Selectivity)')
ax3.set_xticks(x_sub)
ax3.set_xticklabels(subtypes)
ax3.legend(fontsize=7, ncol=2)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# 4. Heatmap of exc_high targeting
ax4 = fig.add_subplot(gs[2, 0])
exc_high_data = []
labels = []
for ct in cell_types:
    for sel in ['high', 'low']:
        row = core_df[(core_df['cell_type'] == ct) & (core_df['selectivity'] == sel)]
        if not row.empty:
            exc_high_data.append(row['exc_high'].values[0])
            labels.append(f'{ct.upper()}\n{sel}')

y_pos = np.arange(len(labels))
bar_colors = [colors[l.split('\n')[0].lower()] for l in labels]
bar_alphas = [0.7 if 'high' in l else 0.4 for l in labels]
# Apply alpha to colors manually
import matplotlib.colors as mcolors
bar_colors_with_alpha = [mcolors.to_rgba(c, alpha=a) for c, a in zip(bar_colors, bar_alphas)]
bars = ax4.barh(y_pos, exc_high_data, color=bar_colors_with_alpha, edgecolor='black', linewidth=0.5)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(labels, fontsize=8)
ax4.set_xlabel('Fraction to high-sel Exc')
ax4.set_title('Targeting High-Selectivity Exc')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

for i, v in enumerate(exc_high_data):
    ax4.text(v + 0.005, i, f'{v:.2f}', va='center', fontsize=7)

# 5. SST vs PV preference
ax5 = fig.add_subplot(gs[2, 1])
sst_vals = []
pv_vals = []
labels5 = []

for ct in cell_types:
    for sel in ['high', 'low']:
        row = core_df[(core_df['cell_type'] == ct) & (core_df['selectivity'] == sel)]
        if not row.empty:
            total_inh = row['inh_pv'].values[0] + row['inh_sst'].values[0] + row['inh_vip'].values[0]
            if total_inh > 0:
                pv_vals.append(row['inh_pv'].values[0] / total_inh)
                sst_vals.append(row['inh_sst'].values[0] / total_inh)
                labels5.append(f'{ct.upper()}\n{sel}')

y_pos = np.arange(len(labels5))
width_bar = 0.35

ax5.barh(y_pos - width_bar/2, pv_vals, width_bar, label='PV fraction',
         color='#4C7F19', alpha=0.7, edgecolor='black', linewidth=0.5)
ax5.barh(y_pos + width_bar/2, sst_vals, width_bar, label='SST fraction',
         color='#197F7F', alpha=0.7, edgecolor='black', linewidth=0.5)

ax5.set_yticks(y_pos)
ax5.set_yticklabels(labels5, fontsize=8)
ax5.set_xlabel('Fraction of Inh targeting')
ax5.set_title('PV vs SST Preference')
ax5.legend(fontsize=8)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

plt.suptitle('Core Connectivity by Cell Type × Selectivity', fontsize=13, y=0.995)
plt.savefig('core_nll_0/figures/selectivity_analysis/celltype_selectivity_analysis.png',
            dpi=300, bbox_inches='tight')
plt.savefig('core_nll_0/figures/selectivity_analysis/celltype_selectivity_analysis.svg',
            bbox_inches='tight')
print("\nSaved: celltype_selectivity_analysis.png/svg")
plt.close()

# Print detailed findings
print("\n" + "="*80)
print("CORE CONNECTIVITY: BY CELL TYPE AND SELECTIVITY")
print("="*80)

print("\n1. E/I RATIOS BY CELL TYPE:")
for ct in cell_types:
    high_row = core_df[(core_df['cell_type'] == ct) & (core_df['selectivity'] == 'high')]
    low_row = core_df[(core_df['cell_type'] == ct) & (core_df['selectivity'] == 'low')]

    if not high_row.empty:
        e_h = high_row['exc_high'].values[0] + high_row['exc_low'].values[0]
        i_h = high_row['inh_high'].values[0] + high_row['inh_low'].values[0]
        n_h = high_row['n_connections'].values[0]
        print(f"   {ct.upper():4s} high-sel: E/I = {e_h/i_h:.2f}  ({n_h:,} connections)")

    if not low_row.empty:
        e_l = low_row['exc_high'].values[0] + low_row['exc_low'].values[0]
        i_l = low_row['inh_high'].values[0] + low_row['inh_low'].values[0]
        n_l = low_row['n_connections'].values[0]
        print(f"   {ct.upper():4s} low-sel:  E/I = {e_l/i_l:.2f}  ({n_l:,} connections)")
    print()

print("\n2. KEY DIFFERENCES BETWEEN HIGH AND LOW SELECTIVITY (same cell type):")
for ct in cell_types:
    high_row = core_df[(core_df['cell_type'] == ct) & (core_df['selectivity'] == 'high')]
    low_row = core_df[(core_df['cell_type'] == ct) & (core_df['selectivity'] == 'low')]

    if not high_row.empty and not low_row.empty:
        print(f"\n   {ct.upper()} neurons:")

        # E/I difference
        e_h = high_row['exc_high'].values[0] + high_row['exc_low'].values[0]
        i_h = high_row['inh_high'].values[0] + high_row['inh_low'].values[0]
        e_l = low_row['exc_high'].values[0] + low_row['exc_low'].values[0]
        i_l = low_row['inh_high'].values[0] + low_row['inh_low'].values[0]

        print(f"      E/I: high={e_h/i_h:.2f}, low={e_l/i_l:.2f}, diff={e_h/i_h - e_l/i_l:.2f}")

        # SST preference
        h_sst = high_row['inh_sst'].values[0]
        l_sst = low_row['inh_sst'].values[0]
        print(f"      SST targeting: high={h_sst:.3f}, low={l_sst:.3f}, diff={l_sst-h_sst:.3f}")

        # PV preference
        h_pv = high_row['inh_pv'].values[0]
        l_pv = low_row['inh_pv'].values[0]
        print(f"      PV targeting:  high={h_pv:.3f}, low={l_pv:.3f}, diff={l_pv-h_pv:.3f}")

print("\n\n3. STRONGEST PATHWAYS (top 10):")
pathways = []
for _, row in core_df.iterrows():
    group = row['group'].replace('_core', '')
    pathways.append((f"{group} → exc_high", row['exc_high']))
    pathways.append((f"{group} → exc_low", row['exc_low']))
    pathways.append((f"{group} → PV", row['inh_pv']))
    pathways.append((f"{group} → SST", row['inh_sst']))
    pathways.append((f"{group} → VIP", row['inh_vip']))

pathways.sort(key=lambda x: x[1], reverse=True)
for i, (pathway, weight) in enumerate(pathways[:10]):
    print(f"   {i+1:2d}. {pathway:30s}: {weight:.3f}")

print("\n\n4. CELL TYPE SPECIFIC PATTERNS:")
print("\n   VIP neurons (disinhibition specialists):")
vip_high = core_df[(core_df['cell_type'] == 'vip') & (core_df['selectivity'] == 'high')].iloc[0]
vip_low = core_df[(core_df['cell_type'] == 'vip') & (core_df['selectivity'] == 'low')].iloc[0]
print(f"      VIP-high → SST: {vip_high['inh_sst']:.3f} (strongest SST targeting!)")
print(f"      VIP-low  → SST: {vip_low['inh_sst']:.3f}")

print("\n   SST neurons (dendrite-targeting):")
sst_high = core_df[(core_df['cell_type'] == 'sst') & (core_df['selectivity'] == 'high')].iloc[0]
sst_low = core_df[(core_df['cell_type'] == 'sst') & (core_df['selectivity'] == 'low')].iloc[0]
print(f"      SST-high → VIP: {sst_high['inh_vip']:.3f}")
print(f"      SST-low  → VIP: {sst_low['inh_vip']:.3f}")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
