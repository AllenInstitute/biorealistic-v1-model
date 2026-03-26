#!/usr/bin/env python3
"""
Create scatter plots comparing 30-cell decoding accuracy with firing rates.
Compares bio_trained vs naive networks for evoked rate and net evoked rate.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
decoding_bio_path = "summary50/decoding_summary.csv"
decoding_naive_path = "summary50/decoding_summary_naive.csv"
firing_rate_path = "/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/digital_twin/firing_rate_analysis_complete/firing_rate_statistics_by_celltype.csv"
color_scheme_path = "../base_props/cell_type_naming_scheme.csv"

def load_color_scheme():
    """Load the official color scheme for cell types."""
    colors_df = pd.read_csv(color_scheme_path, sep=' ')
    color_map = dict(zip(colors_df['cell_type'], colors_df['hex']))
    return color_map

def load_and_process_data():
    """Load decoding and firing rate data."""
    # Load decoding data
    decoding_bio = pd.read_csv(decoding_bio_path)
    decoding_naive = pd.read_csv(decoding_naive_path)
    
    # Filter for 30-cell sample size and add network type
    decoding_bio_30 = decoding_bio[decoding_bio['sample_size'] == 30].copy()
    decoding_naive_30 = decoding_naive[decoding_naive['sample_size'] == 30].copy()
    
    decoding_bio_30['network_type'] = 'bio_trained'
    decoding_naive_30['network_type'] = 'naive'
    
    # Combine decoding data
    decoding_combined = pd.concat([decoding_bio_30, decoding_naive_30], ignore_index=True)
    
    # Load firing rate data
    firing_rates = pd.read_csv(firing_rate_path)
    
    # Calculate mean accuracy and firing rates per cell type and network type
    decoding_summary = decoding_combined.groupby(['cell_type', 'network_type'])['accuracy'].agg(['mean', 'std', 'sem']).reset_index()
    decoding_summary.columns = ['cell_type', 'network_type', 'accuracy_mean', 'accuracy_std', 'accuracy_sem']
    
    # Merge with firing rates
    merged_data = pd.merge(decoding_summary, firing_rates, 
                          left_on=['cell_type', 'network_type'], 
                          right_on=['cell_type', 'variant'])
    
    return merged_data

def create_scatter_plot(data, x_col, y_col, title_base, filename_base, color_map):
    """Create separate scatter plots for bio_trained and naive networks."""
    figures = {}
    
    for network_type in ['bio_trained', 'naive']:
        fig, ax = plt.subplots(figsize=(8, 5))
        subset = data[data['network_type'] == network_type]
        
        # Create scatter plot with cell type colors and labels
        for cell_type in subset['cell_type'].unique():
            cell_data = subset[subset['cell_type'] == cell_type]
            if len(cell_data) > 0:
                color = color_map.get(cell_type, '#666666')
                
                ax.scatter(cell_data[x_col], cell_data[y_col], 
                          color=color, marker='o', s=120, alpha=1.0,
                          label=cell_type, edgecolors='black', linewidth=0.8,
                          zorder=3)
                
                # Add error bars
                ax.errorbar(cell_data[x_col], cell_data[y_col], 
                           yerr=cell_data['accuracy_sem'], 
                           color=color, alpha=0.7, capsize=4, linewidth=1.5,
                           linestyle='none', zorder=2)
                
                # Store data for text labels (will add after plot setup)
                x_pos = cell_data[x_col].iloc[0]
                y_pos = cell_data[y_col].iloc[0]
                
                # Create a shorter label for better visibility
                short_label = cell_type.replace('L2/3_', '23').replace('L', '')
                if short_label.endswith('_Exc'):
                    short_label = short_label.replace('_Exc', 'E')
                elif short_label.endswith('_Inh'):
                    short_label = short_label.replace('_Inh', 'I')
                
                # Store for later annotation
                if not hasattr(ax, '_annotation_data'):
                    ax._annotation_data = []
                ax._annotation_data.append((x_pos, y_pos, short_label))
        
        # Customize plot
        x_label = x_col.replace('_', ' ').title() + ' (Hz)'
        if 'net_evoked_rate' in x_col:
            x_label = 'Evoked - Spontaneous Rate (Hz)'
        ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
        ax.set_ylabel('30-Cell Decoding Accuracy', fontsize=14, fontweight='bold')
        ax.set_title(f'{network_type.replace("_", "-").title()}', 
                    fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3, zorder=1)
        
        # Make tick labels larger and more visible
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', labelsize=10, width=1, length=4)
        
        # Add legend for all cell types, positioned outside the plot area
        legend_elements = []
        for cell_type in sorted(subset['cell_type'].unique()):
            color = color_map.get(cell_type, '#666666')
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                          markersize=8, label=cell_type, markeredgecolor='black',
                          markeredgewidth=0.8)
            )
        
        if legend_elements:
            legend = ax.legend(handles=legend_elements, loc='center left', fontsize=9, 
                             ncol=1, framealpha=0.95, edgecolor='black', 
                             bbox_to_anchor=(1.02, 0.5))
        
        # Add text labels for each cell type (now that axis limits are set)
        if hasattr(ax, '_annotation_data'):
            for x_pos, y_pos, short_label in ax._annotation_data:
                # Use fixed offset in points to avoid overlap
                ax.annotate(short_label, (x_pos, y_pos), 
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=9, fontweight='bold', color='black',
                           ha='left', va='bottom', zorder=4)
        
        # Add correlation info (positioned to avoid legend)
        if len(subset) > 1:
            corr = np.corrcoef(subset[x_col], subset[y_col])[0, 1]
            ax.text(0.98, 0.02, f'r = {corr:.3f}', 
                   transform=ax.transAxes, fontsize=14, fontweight='bold',
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                           alpha=0.9, edgecolor='black', linewidth=1))
        
        plt.tight_layout()
        filename = f'{filename_base}_{network_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        figures[network_type] = fig
        print(f"Saved: {filename}")
    
    return figures

def main():
    """Main function to create scatter plots."""
    # Load data
    print("Loading data...")
    data = load_and_process_data()
    color_map = load_color_scheme()
    
    print(f"Loaded data for {len(data)} cell type × network combinations")
    print(f"Cell types: {sorted(data['cell_type'].unique())}")
    print(f"Network types: {sorted(data['network_type'].unique())}")
    
    # Create scatter plots
    print("\nCreating scatter plots...")
    
    # Plot 1: 30-cell accuracy vs evoked rate
    fig1 = create_scatter_plot(
        data, 
        'evoked_rate_mean', 
        'accuracy_mean',
        '30-Cell Decoding Accuracy vs Evoked Firing Rate',
        'decoding_accuracy_vs_evoked_rate',
        color_map
    )
    
    # Plot 2: 30-cell accuracy vs evoked - spontaneous rate
    fig2 = create_scatter_plot(
        data, 
        'net_evoked_rate_mean', 
        'accuracy_mean',
        '30-Cell Decoding Accuracy vs Evoked - Spontaneous Firing Rate',
        'decoding_accuracy_vs_evoked_minus_spontaneous_rate',
        color_map
    )
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    
    for network_type in ['bio_trained', 'naive']:
        subset = data[data['network_type'] == network_type]
        print(f"\n{network_type.upper()}:")
        print(f"  Evoked rate range: {subset['evoked_rate_mean'].min():.2f} - {subset['evoked_rate_mean'].max():.2f} Hz")
        print(f"  Evoked - spontaneous rate range: {subset['net_evoked_rate_mean'].min():.2f} - {subset['net_evoked_rate_mean'].max():.2f} Hz")
        print(f"  Accuracy range: {subset['accuracy_mean'].min():.3f} - {subset['accuracy_mean'].max():.3f}")
        
        # Correlations
        evoked_corr = np.corrcoef(subset['evoked_rate_mean'], subset['accuracy_mean'])[0, 1]
        net_evoked_corr = np.corrcoef(subset['net_evoked_rate_mean'], subset['accuracy_mean'])[0, 1]
        print(f"  Correlation (accuracy vs evoked rate): {evoked_corr:.3f}")
        print(f"  Correlation (accuracy vs evoked - spontaneous rate): {net_evoked_corr:.3f}")
    
    # Save summary data
    summary_file = "decoding_vs_firing_rate_summary.csv"
    data.to_csv(summary_file, index=False)
    print(f"\nSummary data saved to: {summary_file}")
    
    print("\nScatter plots saved:")
    print("  - decoding_accuracy_vs_evoked_rate_bio_trained.png")
    print("  - decoding_accuracy_vs_evoked_rate_naive.png")
    print("  - decoding_accuracy_vs_evoked_minus_spontaneous_rate_bio_trained.png")
    print("  - decoding_accuracy_vs_evoked_minus_spontaneous_rate_naive.png")

if __name__ == "__main__":
    main() 