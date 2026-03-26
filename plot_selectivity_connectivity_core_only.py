#!/usr/bin/env python
"""
Visualize outgoing connectivity patterns from CORE neurons only, based on selectivity grouping.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
core_df = pd.read_csv('core_nll_0/figures/selectivity_outgoing/outgoing_weight_granular_core_to_core_summary.csv')

# Create output directory
import os
os.makedirs('core_nll_0/figures/selectivity_analysis', exist_ok=True)

# Filter to core only
core_only = core_df[core_df['group'].str.contains('core')].copy()
core_only['selectivity'] = core_only['group'].str.extract(r'inh_(high|low)')[0]

print("Core groups:")
print(core_only[['group', 'selectivity']])

# Define color scheme
exc_high_color = '#D42A2A'
exc_low_color = '#F9BBBB'
inh_high_color = '#197F7F'
inh_low_color = '#9FD4D4'
pv_color = '#4C7F19'
sst_color = '#197F7F'
vip_color = '#9932FF'

# Create comprehensive figure
fig = plt.figure(figsize=(7.5, 7))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.35)

# 1. Excitatory vs Inhibitory targeting (by selectivity)
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(2)
width = 0.35

high_sel = core_only[core_only['selectivity'] == 'high'].iloc[0]
low_sel = core_only[core_only['selectivity'] == 'low'].iloc[0]

exc_high_vals = [high_sel['exc_high'] + high_sel['exc_low'],
                 low_sel['exc_high'] + low_sel['exc_low']]
inh_high_vals = [high_sel['inh_high'] + high_sel['inh_low'],
                 low_sel['inh_high'] + low_sel['inh_low']]

ax1.bar(x - width/2, exc_high_vals, width, label='To Exc', color='#C93C3C', alpha=0.8)
ax1.bar(x + width/2, inh_high_vals, width, label='To Inh', color='#197F7F', alpha=0.8)
ax1.set_ylabel('Fraction of outgoing weight')
ax1.set_title('E vs I Targeting from Core')
ax1.set_xticks(x)
ax1.set_xticklabels(['High selectivity\nsource', 'Low selectivity\nsource'])
ax1.legend()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add E/I ratio annotations
for i, (e, inh) in enumerate(zip(exc_high_vals, inh_high_vals)):
    ratio = e / inh
    ax1.text(x[i], max(e, inh) + 0.02, f'E/I={ratio:.2f}',
             ha='center', fontsize=8, style='italic')

# 2. High vs Low selectivity preference in targets
ax2 = fig.add_subplot(gs[0, 1])
categories = ['To Exc\n(from high)', 'To Exc\n(from low)', 'To Inh\n(from high)', 'To Inh\n(from low)']
x_pos = np.arange(len(categories))

high_exc_pref = high_sel['exc_high'] / (high_sel['exc_high'] + high_sel['exc_low'])
low_exc_pref = low_sel['exc_high'] / (low_sel['exc_high'] + low_sel['exc_low'])
high_inh_pref = high_sel['inh_high'] / (high_sel['inh_high'] + high_sel['inh_low'])
low_inh_pref = low_sel['inh_high'] / (low_sel['inh_high'] + low_sel['inh_low'])

values = [high_exc_pref, low_exc_pref, high_inh_pref, low_inh_pref]
colors = [exc_high_color, exc_high_color, inh_high_color, inh_high_color]

bars = ax2.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Equal preference')
ax2.set_ylabel('Fraction to high selectivity targets')
ax2.set_title('Selectivity Preference in Targets')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(categories, fontsize=8)
ax2.set_ylim(0, 1)
ax2.legend(fontsize=7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add percentage labels
for i, v in enumerate(values):
    ax2.text(i, v + 0.03, f'{v:.1%}', ha='center', fontsize=8, weight='bold')

# 3. Detailed excitatory targeting breakdown
ax3 = fig.add_subplot(gs[1, 0])
width = 0.35
x = np.arange(2)

ax3.bar(x - width/2, [high_sel['exc_high'], low_sel['exc_high']],
        width, label='To high-sel Exc', color=exc_high_color)
ax3.bar(x + width/2, [high_sel['exc_low'], low_sel['exc_low']],
        width, label='To low-sel Exc', color=exc_low_color)
ax3.set_ylabel('Fraction of outgoing weight')
ax3.set_title('Excitatory Target Selectivity')
ax3.set_xticks(x)
ax3.set_xticklabels(['High selectivity\nsource', 'Low selectivity\nsource'])
ax3.legend()
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# 4. Inhibitory subtype targeting
ax4 = fig.add_subplot(gs[1, 1])
subtypes = ['PV', 'SST', 'VIP']
high_subtype_vals = [high_sel['inh_pv'], high_sel['inh_sst'], high_sel['inh_vip']]
low_subtype_vals = [low_sel['inh_pv'], low_sel['inh_sst'], low_sel['inh_vip']]

x = np.arange(len(subtypes))
width = 0.35

ax4.bar(x - width/2, high_subtype_vals, width, label='High sel source',
        color=[pv_color, sst_color, vip_color], alpha=0.7, edgecolor='black', linewidth=0.5)
ax4.bar(x + width/2, low_subtype_vals, width, label='Low sel source',
        color=[pv_color, sst_color, vip_color], alpha=0.4, edgecolor='black', linewidth=0.5)
ax4.set_ylabel('Fraction of outgoing weight')
ax4.set_title('Inhibitory Subtype Targeting')
ax4.set_xticks(x)
ax4.set_xticklabels(subtypes)
ax4.legend()
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# 5. Inhibitory subtype targeting - SPLIT by target selectivity
ax5 = fig.add_subplot(gs[2, :])
subtypes_detailed = ['PV\n(high)', 'PV\n(low)', 'SST\n(high)', 'SST\n(low)', 'VIP\n(high)', 'VIP\n(low)']
x = np.arange(len(subtypes_detailed))
width = 0.35

high_detailed = [high_sel['inh_high_pv'], high_sel['inh_low_pv'],
                 high_sel['inh_high_sst'], high_sel['inh_low_sst'],
                 high_sel['inh_high_vip'], high_sel['inh_low_vip']]
low_detailed = [low_sel['inh_high_pv'], low_sel['inh_low_pv'],
                low_sel['inh_high_sst'], low_sel['inh_low_sst'],
                low_sel['inh_high_vip'], low_sel['inh_low_vip']]

colors_detailed = [pv_color, pv_color, sst_color, sst_color, vip_color, vip_color]
alphas = [0.9, 0.4, 0.9, 0.4, 0.9, 0.4]

# Plot for high selectivity source
for i, (val, color, alpha) in enumerate(zip(high_detailed, colors_detailed, alphas)):
    ax5.bar(x[i] - width/2, val, width, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)

# Plot for low selectivity source
for i, (val, color, alpha) in enumerate(zip(low_detailed, colors_detailed, alphas)):
    ax5.bar(x[i] + width/2, val, width, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)

# Add custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='High sel source'),
    Patch(facecolor='gray', alpha=0.3, edgecolor='black', label='Low sel source'),
    Patch(facecolor='white', edgecolor='white', label=''),
    Patch(facecolor=pv_color, alpha=0.9, label='High sel target'),
    Patch(facecolor=pv_color, alpha=0.4, label='Low sel target')
]
ax5.legend(handles=legend_elements, ncol=5, fontsize=8, loc='upper right')

ax5.set_ylabel('Fraction of outgoing weight')
ax5.set_title('Inhibitory Subtype × Target Selectivity')
ax5.set_xticks(x)
ax5.set_xticklabels(subtypes_detailed, fontsize=9)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

plt.suptitle('Core Neurons: Connectivity by Source Selectivity', fontsize=13, y=0.995)
plt.savefig('core_nll_0/figures/selectivity_analysis/core_connectivity_analysis.png',
            dpi=300, bbox_inches='tight')
plt.savefig('core_nll_0/figures/selectivity_analysis/core_connectivity_analysis.svg',
            bbox_inches='tight')
print("Saved: core_connectivity_analysis.png/svg")
plt.close()

# Generate detailed findings
print("\n" + "="*70)
print("CORE NEURONS: KEY CONNECTIVITY FINDINGS")
print("="*70)

print("\n1. E/I BALANCE BY SOURCE SELECTIVITY:")
high_e = high_sel['exc_high'] + high_sel['exc_low']
high_i = high_sel['inh_high'] + high_sel['inh_low']
low_e = low_sel['exc_high'] + low_sel['exc_low']
low_i = low_sel['inh_high'] + low_sel['inh_low']
print(f"   High selectivity sources: {high_e:.3f} to Exc, {high_i:.3f} to Inh")
print(f"   → E/I ratio = {high_e/high_i:.2f}")
print(f"   Low selectivity sources:  {low_e:.3f} to Exc, {low_i:.3f} to Inh")
print(f"   → E/I ratio = {low_e/low_i:.2f}")
print(f"   Difference: {(high_e/high_i) - (low_e/low_i):.2f}")

print("\n2. SELECTIVITY HOMOPHILY (preference for similar selectivity):")
print(f"   High-sel → High-sel Exc: {high_exc_pref:.1%}")
print(f"   Low-sel → High-sel Exc:  {low_exc_pref:.1%}")
print(f"   High-sel → High-sel Inh: {high_inh_pref:.1%}")
print(f"   Low-sel → High-sel Inh:  {low_inh_pref:.1%}")

print("\n3. INHIBITORY SUBTYPE PREFERENCES:")
print(f"   From HIGH selectivity sources:")
total_inh_high = high_sel['inh_pv'] + high_sel['inh_sst'] + high_sel['inh_vip']
print(f"      PV:  {high_sel['inh_pv']/total_inh_high:.1%} ({high_sel['inh_pv']:.3f} total weight)")
print(f"      SST: {high_sel['inh_sst']/total_inh_high:.1%} ({high_sel['inh_sst']:.3f} total weight)")
print(f"      VIP: {high_sel['inh_vip']/total_inh_high:.1%} ({high_sel['inh_vip']:.3f} total weight)")

print(f"\n   From LOW selectivity sources:")
total_inh_low = low_sel['inh_pv'] + low_sel['inh_sst'] + low_sel['inh_vip']
print(f"      PV:  {low_sel['inh_pv']/total_inh_low:.1%} ({low_sel['inh_pv']:.3f} total weight)")
print(f"      SST: {low_sel['inh_sst']/total_inh_low:.1%} ({low_sel['inh_sst']:.3f} total weight)")
print(f"      VIP: {low_sel['inh_vip']/total_inh_low:.1%} ({low_sel['inh_vip']:.3f} total weight)")

print("\n4. KEY ASYMMETRIES:")
# SST preference difference
sst_diff = (low_sel['inh_sst']/total_inh_low) - (high_sel['inh_sst']/total_inh_high)
print(f"   Low-sel sources target SST {sst_diff:.1%} MORE than high-sel sources")

# PV targeting to high vs low selectivity
pv_high_to_high = high_sel['inh_high_pv'] / high_sel['inh_pv']
pv_high_to_low = high_sel['inh_low_pv'] / high_sel['inh_pv']
print(f"   High-sel sources: {pv_high_to_high:.1%} of PV targeting → high-sel PV")
print(f"                     {pv_high_to_low:.1%} of PV targeting → low-sel PV")

pv_low_to_high = low_sel['inh_high_pv'] / low_sel['inh_pv']
pv_low_to_low = low_sel['inh_low_pv'] / low_sel['inh_pv']
print(f"   Low-sel sources:  {pv_low_to_high:.1%} of PV targeting → high-sel PV")
print(f"                     {pv_low_to_low:.1%} of PV targeting → low-sel PV")

print("\n5. STRONGEST INDIVIDUAL PATHWAYS:")
# Create pathway dictionary
pathways = {
    'High→Exc-high': high_sel['exc_high'],
    'High→Exc-low': high_sel['exc_low'],
    'High→PV': high_sel['inh_pv'],
    'High→SST': high_sel['inh_sst'],
    'High→VIP': high_sel['inh_vip'],
    'Low→Exc-high': low_sel['exc_high'],
    'Low→Exc-low': low_sel['exc_low'],
    'Low→PV': low_sel['inh_pv'],
    'Low→SST': low_sel['inh_sst'],
    'Low→VIP': low_sel['inh_vip'],
}
sorted_pathways = sorted(pathways.items(), key=lambda x: x[1], reverse=True)
for i, (pathway, weight) in enumerate(sorted_pathways[:6]):
    print(f"   {i+1}. {pathway:15s}: {weight:.3f}")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
