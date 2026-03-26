#!/usr/bin/env python
"""
Visualize outgoing connectivity patterns based on selectivity grouping.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
summary_df = pd.read_csv('core_nll_0/figures/selectivity_outgoing/outgoing_weight_granular_summary.csv')
core_df = pd.read_csv('core_nll_0/figures/selectivity_outgoing/outgoing_weight_granular_core_to_core_summary.csv')

# Define color scheme for target groups
target_colors = {
    'exc_high': '#D42A2A',
    'exc_low': '#F9BBBB',
    'inh_high': '#197F7F',
    'inh_low': '#9FD4D4',
    'inh_pv': '#4C7F19',
    'inh_sst': '#197F7F',
    'inh_vip': '#9932FF',
    'inh_htr3a': '#787878'
}

# Create output directory
import os
os.makedirs('core_nll_0/figures/selectivity_analysis', exist_ok=True)

# 1. Compare connectivity fractions by source group
fig, axes = plt.subplots(2, 2, figsize=(7, 6))
fig.suptitle('Outgoing Connectivity by Selectivity Group', fontsize=12, y=0.98)

# Prepare data for plotting
exc_cols = ['exc_high', 'exc_low']
inh_cols = ['inh_high', 'inh_low']
pv_sst_vip = ['inh_pv', 'inh_sst', 'inh_vip']

source_groups = summary_df['group'].values

# Plot 1: Excitatory targets
ax = axes[0, 0]
x = np.arange(len(source_groups))
width = 0.35
ax.bar(x - width/2, summary_df['exc_high'], width, label='High selectivity', color='#D42A2A')
ax.bar(x + width/2, summary_df['exc_low'], width, label='Low selectivity', color='#F9BBBB')
ax.set_ylabel('Fraction of outgoing weight')
ax.set_title('To Excitatory Neurons')
ax.set_xticks(x)
ax.set_xticklabels(source_groups, rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Plot 2: Inhibitory targets
ax = axes[0, 1]
ax.bar(x - width/2, summary_df['inh_high'], width, label='High selectivity', color='#197F7F')
ax.bar(x + width/2, summary_df['inh_low'], width, label='Low selectivity', color='#9FD4D4')
ax.set_ylabel('Fraction of outgoing weight')
ax.set_title('To Inhibitory Neurons')
ax.set_xticks(x)
ax.set_xticklabels(source_groups, rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Plot 3: Inhibitory subtypes
ax = axes[1, 0]
ax.bar(x - width, summary_df['inh_pv'], width*0.8, label='PV', color='#4C7F19')
ax.bar(x, summary_df['inh_sst'], width*0.8, label='SST', color='#197F7F')
ax.bar(x + width, summary_df['inh_vip'], width*0.8, label='VIP', color='#9932FF')
ax.set_ylabel('Fraction of outgoing weight')
ax.set_title('To Inhibitory Subtypes')
ax.set_xticks(x)
ax.set_xticklabels(source_groups, rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=7, ncol=3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Plot 4: E/I ratio
ax = axes[1, 1]
exc_total = summary_df['exc_high'] + summary_df['exc_low']
inh_total = summary_df['inh_high'] + summary_df['inh_low']
ei_ratio = exc_total / inh_total
ax.bar(x, ei_ratio, color='gray', alpha=0.7)
ax.axhline(1, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('E/I ratio')
ax.set_title('Excitatory/Inhibitory Ratio')
ax.set_xticks(x)
ax.set_xticklabels(source_groups, rotation=45, ha='right', fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('core_nll_0/figures/selectivity_analysis/connectivity_by_selectivity.png', dpi=300, bbox_inches='tight')
plt.savefig('core_nll_0/figures/selectivity_analysis/connectivity_by_selectivity.svg', bbox_inches='tight')
print("Saved: connectivity_by_selectivity.png/svg")
plt.close()

# 2. Heatmap of connectivity patterns
fig, ax = plt.subplots(figsize=(6, 4))
# Select key columns for heatmap
heatmap_cols = ['exc_high', 'exc_low', 'inh_pv', 'inh_sst', 'inh_vip']
heatmap_data = summary_df[heatmap_cols].T
heatmap_data.columns = source_groups

sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
            cbar_kws={'label': 'Fraction of outgoing weight'},
            ax=ax)
ax.set_ylabel('Target group')
ax.set_xlabel('Source group')
ax.set_title('Connectivity Pattern Heatmap')
plt.tight_layout()
plt.savefig('core_nll_0/figures/selectivity_analysis/connectivity_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('core_nll_0/figures/selectivity_analysis/connectivity_heatmap.svg', bbox_inches='tight')
print("Saved: connectivity_heatmap.png/svg")
plt.close()

# 3. Core vs Periphery comparison
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

# Compare high selectivity groups
high_groups = summary_df[summary_df['group'].str.contains('high')]
ax = axes[0]
x = np.arange(2)
width = 0.25
core_high = high_groups[high_groups['group'].str.contains('core')][exc_cols + inh_cols].values[0]
periph_high = high_groups[high_groups['group'].str.contains('periphery')][exc_cols + inh_cols].values[0]

for i, (col, color) in enumerate(zip(exc_cols + inh_cols, ['#D42A2A', '#F9BBBB', '#197F7F', '#9FD4D4'])):
    ax.bar(x + i*width - 1.5*width, [core_high[i], periph_high[i]], width,
           label=col.replace('_', ' ').title(), color=color)

ax.set_ylabel('Fraction of outgoing weight')
ax.set_title('High Selectivity: Core vs Periphery')
ax.set_xticks(x)
ax.set_xticklabels(['Core', 'Periphery'])
ax.legend(fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Compare low selectivity groups
low_groups = summary_df[summary_df['group'].str.contains('low')]
ax = axes[1]
core_low = low_groups[low_groups['group'].str.contains('core')][exc_cols + inh_cols].values[0]
periph_low = low_groups[low_groups['group'].str.contains('periphery')][exc_cols + inh_cols].values[0]

for i, (col, color) in enumerate(zip(exc_cols + inh_cols, ['#D42A2A', '#F9BBBB', '#197F7F', '#9FD4D4'])):
    ax.bar(x + i*width - 1.5*width, [core_low[i], periph_low[i]], width,
           label=col.replace('_', ' ').title(), color=color)

ax.set_ylabel('Fraction of outgoing weight')
ax.set_title('Low Selectivity: Core vs Periphery')
ax.set_xticks(x)
ax.set_xticklabels(['Core', 'Periphery'])
ax.legend(fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('core_nll_0/figures/selectivity_analysis/core_vs_periphery.png', dpi=300, bbox_inches='tight')
plt.savefig('core_nll_0/figures/selectivity_analysis/core_vs_periphery.svg', bbox_inches='tight')
print("Saved: core_vs_periphery.png/svg")
plt.close()

# 4. Generate findings report
print("\n" + "="*60)
print("KEY FINDINGS FROM CONNECTIVITY ANALYSIS")
print("="*60)

print("\n1. SELECTIVITY-DEPENDENT E/I BALANCE:")
for _, row in summary_df.iterrows():
    exc_total = row['exc_high'] + row['exc_low']
    inh_total = row['inh_high'] + row['inh_low']
    print(f"   {row['group']:25s}: E/I ratio = {exc_total/inh_total:.3f}")

print("\n2. PREFERENCE FOR HIGH VS LOW SELECTIVITY TARGETS:")
for _, row in summary_df.iterrows():
    exc_pref = row['exc_high'] / (row['exc_high'] + row['exc_low']) if (row['exc_high'] + row['exc_low']) > 0 else 0
    inh_pref = row['inh_high'] / (row['inh_high'] + row['inh_low']) if (row['inh_high'] + row['inh_low']) > 0 else 0
    print(f"   {row['group']:25s}: Exc high = {exc_pref:.1%}, Inh high = {inh_pref:.1%}")

print("\n3. INHIBITORY SUBTYPE TARGETING:")
for _, row in summary_df.iterrows():
    inh_total = row['inh_pv'] + row['inh_sst'] + row['inh_vip']
    if inh_total > 0:
        print(f"   {row['group']:25s}: PV={row['inh_pv']/inh_total:.1%}, SST={row['inh_sst']/inh_total:.1%}, VIP={row['inh_vip']/inh_total:.1%}")

print("\n4. CORE VS PERIPHERY DIFFERENCES:")
high_core = summary_df[summary_df['group'] == 'inh_high_core'].iloc[0]
high_periph = summary_df[summary_df['group'] == 'inh_high_periphery'].iloc[0]
print(f"   High selectivity core to exc: {high_core['exc_high']+high_core['exc_low']:.3f}")
print(f"   High selectivity periphery to exc: {high_periph['exc_high']+high_periph['exc_low']:.3f}")
print(f"   Difference: {(high_core['exc_high']+high_core['exc_low'])-(high_periph['exc_high']+high_periph['exc_low']):.3f}")

print("\n" + "="*60)
print("Analysis complete! Figures saved to core_nll_0/figures/selectivity_analysis/")
print("="*60)
