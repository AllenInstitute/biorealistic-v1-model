#!/usr/bin/env python3
"""
Aggregate and analyze L2/3-specific reciprocal connection results.
Focus on L2/3 Exc vs L2/3 PV connections for experimental comparison.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests


def load_l23_networks_data(base_dir, network_type):
    """Load data from all L2/3 network analyses"""
    all_data = []
    
    for i in range(10):
        network_dir = f"{base_dir}/core_nll_{i}_{network_type}"
        csv_file = f"{network_dir}/reciprocal_connections_L2_3_Exc_L2_3_PV_L2_3_SST_L2_3_VIP_{network_type}.csv"
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['network'] = f'core_nll_{i}'
            df['network_type'] = network_type
            
            # Calculate log-space correlation
            df['log_correlation'] = np.log(df['weight_a_to_b']) - np.log(df['weight_b_to_a'])
            
            all_data.append(df)
            print(f"Loaded {len(df)} connections from {csv_file}")
        else:
            print(f"Warning: {csv_file} not found")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def create_l23_pv_focused_plots(bio_data, naive_data, output_dir):
    """Create plots specifically focused on L2/3 Exc vs L2/3 PV connections"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for L2/3 Exc -> L2/3 PV connections
    bio_exc_pv = bio_data[
        (bio_data['source_type_a_to_b'] == 'L2/3_Exc') & 
        (bio_data['target_type_a_to_b'] == 'L2/3_PV')
    ]
    naive_exc_pv = naive_data[
        (naive_data['source_type_a_to_b'] == 'L2/3_Exc') & 
        (naive_data['target_type_a_to_b'] == 'L2/3_PV')
    ]
    
    # Filter for L2/3 PV -> L2/3 Exc connections
    bio_pv_exc = bio_data[
        (bio_data['source_type_a_to_b'] == 'L2/3_PV') & 
        (bio_data['target_type_a_to_b'] == 'L2/3_Exc')
    ]
    naive_pv_exc = naive_data[
        (naive_data['source_type_a_to_b'] == 'L2/3_PV') & 
        (naive_data['target_type_a_to_b'] == 'L2/3_Exc')
    ]
    
    print(f"L2/3 Exc -> L2/3 PV connections: Bio={len(bio_exc_pv)}, Naive={len(naive_exc_pv)}")
    print(f"L2/3 PV -> L2/3 Exc connections: Bio={len(bio_pv_exc)}, Naive={len(naive_pv_exc)}")
    
    # Create detailed scatter plots for L2/3 Exc <-> PV
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # L2/3 Exc -> PV (Bio-trained)
    ax = axes[0, 0]
    if len(bio_exc_pv) > 0:
        ax.scatter(np.log(bio_exc_pv['weight_a_to_b']), np.log(bio_exc_pv['weight_b_to_a']), 
                  alpha=0.6, s=1, color='blue')
        r_bio_exc_pv, p_bio_exc_pv = stats.pearsonr(np.log(bio_exc_pv['weight_a_to_b']), 
                                                    np.log(bio_exc_pv['weight_b_to_a']))
        ax.set_title(f'Bio-trained: L2/3 Exc → PV\nR = {r_bio_exc_pv:.3f}, p = {p_bio_exc_pv:.3e}')
    else:
        ax.set_title('Bio-trained: L2/3 Exc → PV\nNo data')
    ax.set_xlabel('Log(Weight A→B)')
    ax.set_ylabel('Log(Weight B→A)')
    ax.grid(True, alpha=0.3)
    
    # L2/3 Exc -> PV (Naive)
    ax = axes[0, 1]
    if len(naive_exc_pv) > 0:
        ax.scatter(np.log(naive_exc_pv['weight_a_to_b']), np.log(naive_exc_pv['weight_b_to_a']), 
                  alpha=0.6, s=1, color='red')
        r_naive_exc_pv, p_naive_exc_pv = stats.pearsonr(np.log(naive_exc_pv['weight_a_to_b']), 
                                                        np.log(naive_exc_pv['weight_b_to_a']))
        ax.set_title(f'Naive: L2/3 Exc → PV\nR = {r_naive_exc_pv:.3f}, p = {p_naive_exc_pv:.3e}')
    else:
        ax.set_title('Naive: L2/3 Exc → PV\nNo data')
    ax.set_xlabel('Log(Weight A→B)')
    ax.set_ylabel('Log(Weight B→A)')
    ax.grid(True, alpha=0.3)
    
    # L2/3 PV -> Exc (Bio-trained)
    ax = axes[1, 0]
    if len(bio_pv_exc) > 0:
        ax.scatter(np.log(bio_pv_exc['weight_a_to_b']), np.log(bio_pv_exc['weight_b_to_a']), 
                  alpha=0.6, s=1, color='blue')
        r_bio_pv_exc, p_bio_pv_exc = stats.pearsonr(np.log(bio_pv_exc['weight_a_to_b']), 
                                                    np.log(bio_pv_exc['weight_b_to_a']))
        ax.set_title(f'Bio-trained: L2/3 PV → Exc\nR = {r_bio_pv_exc:.3f}, p = {p_bio_pv_exc:.3e}')
    else:
        ax.set_title('Bio-trained: L2/3 PV → Exc\nNo data')
    ax.set_xlabel('Log(Weight A→B)')
    ax.set_ylabel('Log(Weight B→A)')
    ax.grid(True, alpha=0.3)
    
    # L2/3 PV -> Exc (Naive)
    ax = axes[1, 1]
    if len(naive_pv_exc) > 0:
        ax.scatter(np.log(naive_pv_exc['weight_a_to_b']), np.log(naive_pv_exc['weight_b_to_a']), 
                  alpha=0.6, s=1, color='red')
        r_naive_pv_exc, p_naive_pv_exc = stats.pearsonr(np.log(naive_pv_exc['weight_a_to_b']), 
                                                        np.log(naive_pv_exc['weight_b_to_a']))
        ax.set_title(f'Naive: L2/3 PV → Exc\nR = {r_naive_pv_exc:.3f}, p = {p_naive_pv_exc:.3e}')
    else:
        ax.set_title('Naive: L2/3 PV → Exc\nNo data')
    ax.set_xlabel('Log(Weight A→B)')
    ax.set_ylabel('Log(Weight B→A)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/L23_exc_pv_scatter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Log correlation distributions for Exc -> PV
    ax = axes[0, 0]
    if len(bio_exc_pv) > 0 and len(naive_exc_pv) > 0:
        ax.hist(bio_exc_pv['log_correlation'], bins=50, alpha=0.6, label='Bio-trained', 
                color='blue', density=True)
        ax.hist(naive_exc_pv['log_correlation'], bins=50, alpha=0.6, label='Naive', 
                color='red', density=True)
        ax.axvline(bio_exc_pv['log_correlation'].mean(), color='blue', linestyle='--', linewidth=2)
        ax.axvline(naive_exc_pv['log_correlation'].mean(), color='red', linestyle='--', linewidth=2)
        
        # Statistical test
        t_stat, p_val = stats.ttest_ind(bio_exc_pv['log_correlation'], naive_exc_pv['log_correlation'])
        ax.set_title(f'L2/3 Exc → PV Log Correlation\nt = {t_stat:.3f}, p = {p_val:.3e}')
    else:
        ax.set_title('L2/3 Exc → PV Log Correlation\nInsufficient data')
    ax.set_xlabel('Log Correlation')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log correlation distributions for PV -> Exc
    ax = axes[0, 1]
    if len(bio_pv_exc) > 0 and len(naive_pv_exc) > 0:
        ax.hist(bio_pv_exc['log_correlation'], bins=50, alpha=0.6, label='Bio-trained', 
                color='blue', density=True)
        ax.hist(naive_pv_exc['log_correlation'], bins=50, alpha=0.6, label='Naive', 
                color='red', density=True)
        ax.axvline(bio_pv_exc['log_correlation'].mean(), color='blue', linestyle='--', linewidth=2)
        ax.axvline(naive_pv_exc['log_correlation'].mean(), color='red', linestyle='--', linewidth=2)
        
        # Statistical test
        t_stat, p_val = stats.ttest_ind(bio_pv_exc['log_correlation'], naive_pv_exc['log_correlation'])
        ax.set_title(f'L2/3 PV → Exc Log Correlation\nt = {t_stat:.3f}, p = {p_val:.3e}')
    else:
        ax.set_title('L2/3 PV → Exc Log Correlation\nInsufficient data')
    ax.set_xlabel('Log Correlation')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Network consistency for Exc -> PV
    ax = axes[1, 0]
    if len(bio_exc_pv) > 0 and len(naive_exc_pv) > 0:
        bio_by_net = bio_exc_pv.groupby('network')['log_correlation'].mean()
        naive_by_net = naive_exc_pv.groupby('network')['log_correlation'].mean()
        
        networks = sorted(set(bio_by_net.index) & set(naive_by_net.index))
        bio_vals = [bio_by_net[net] for net in networks]
        naive_vals = [naive_by_net[net] for net in networks]
        
        x_pos = np.arange(len(networks))
        width = 0.35
        
        ax.bar(x_pos - width/2, bio_vals, width, label='Bio-trained', color='blue', alpha=0.7)
        ax.bar(x_pos + width/2, naive_vals, width, label='Naive', color='red', alpha=0.7)
        
        ax.set_xlabel('Network')
        ax.set_ylabel('Mean Log Correlation')
        ax.set_title('L2/3 Exc → PV by Network')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([net.replace('core_nll_', '') for net in networks])
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title('L2/3 Exc → PV by Network\nInsufficient data')
    
    # Network consistency for PV -> Exc
    ax = axes[1, 1]
    if len(bio_pv_exc) > 0 and len(naive_pv_exc) > 0:
        bio_by_net = bio_pv_exc.groupby('network')['log_correlation'].mean()
        naive_by_net = naive_pv_exc.groupby('network')['log_correlation'].mean()
        
        networks = sorted(set(bio_by_net.index) & set(naive_by_net.index))
        bio_vals = [bio_by_net[net] for net in networks]
        naive_vals = [naive_by_net[net] for net in networks]
        
        x_pos = np.arange(len(networks))
        width = 0.35
        
        ax.bar(x_pos - width/2, bio_vals, width, label='Bio-trained', color='blue', alpha=0.7)
        ax.bar(x_pos + width/2, naive_vals, width, label='Naive', color='red', alpha=0.7)
        
        ax.set_xlabel('Network')
        ax.set_ylabel('Mean Log Correlation')
        ax.set_title('L2/3 PV → Exc by Network')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([net.replace('core_nll_', '') for net in networks])
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title('L2/3 PV → Exc by Network\nInsufficient data')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/L23_exc_pv_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_l23_correlation_heatmap(bio_data, naive_data, output_file):
    """Create aggregated correlation heatmap for L2/3 connections"""
    
    # Define L2/3 cell types
    l23_types = ['L2/3_Exc', 'L2/3_PV', 'L2/3_SST', 'L2/3_VIP']
    
    # Initialize correlation matrices
    bio_corr_matrix = np.full((len(l23_types), len(l23_types)), np.nan)
    naive_corr_matrix = np.full((len(l23_types), len(l23_types)), np.nan)
    diff_matrix = np.full((len(l23_types), len(l23_types)), np.nan)
    
    # Calculate correlations for each cell type pair
    for i, source_type in enumerate(l23_types):
        for j, target_type in enumerate(l23_types):
            # Bio-trained
            bio_subset = bio_data[
                (bio_data['source_type_a_to_b'] == source_type) & 
                (bio_data['target_type_a_to_b'] == target_type)
            ]
            
            # Naive
            naive_subset = naive_data[
                (naive_data['source_type_a_to_b'] == source_type) & 
                (naive_data['target_type_a_to_b'] == target_type)
            ]
            
            print(f"  {source_type} → {target_type}: Bio={len(bio_subset)}, Naive={len(naive_subset)} connections")
            
            # Calculate correlations across all networks (require minimum 100 connections)
            if len(bio_subset) >= 100:
                try:
                    bio_corr, bio_p = stats.pearsonr(
                        np.log(bio_subset['weight_a_to_b']), 
                        np.log(bio_subset['weight_b_to_a'])
                    )
                    if not np.isnan(bio_corr):
                        bio_corr_matrix[i, j] = bio_corr
                except:
                    pass
            
            if len(naive_subset) >= 100:
                try:
                    naive_corr, naive_p = stats.pearsonr(
                        np.log(naive_subset['weight_a_to_b']), 
                        np.log(naive_subset['weight_b_to_a'])
                    )
                    if not np.isnan(naive_corr):
                        naive_corr_matrix[i, j] = naive_corr
                except:
                    pass
            
            # Calculate difference only if both are valid
            if not np.isnan(bio_corr_matrix[i, j]) and not np.isnan(naive_corr_matrix[i, j]):
                diff_matrix[i, j] = bio_corr_matrix[i, j] - naive_corr_matrix[i, j]
    
    # --- NEW: Move any lower-triangle values up to the upper triangle ---
    n_types = len(l23_types)
    for i in range(n_types):
        for j in range(i):  # j < i  => lower triangle
            # Bio-trained
            if np.isnan(bio_corr_matrix[j, i]) and not np.isnan(bio_corr_matrix[i, j]):
                bio_corr_matrix[j, i] = bio_corr_matrix[i, j]
            bio_corr_matrix[i, j] = np.nan  # blank out lower triangle
            # Naive
            if np.isnan(naive_corr_matrix[j, i]) and not np.isnan(naive_corr_matrix[i, j]):
                naive_corr_matrix[j, i] = naive_corr_matrix[i, j]
            naive_corr_matrix[i, j] = np.nan
            # Difference
            if (not np.isnan(diff_matrix[i, j])) and np.isnan(diff_matrix[j, i]):
                diff_matrix[j, i] = diff_matrix[i, j]
            diff_matrix[i, j] = np.nan
    # ---------------------------------------------------------------
    
    # Create the heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Bio-trained heatmap
    im1 = axes[0].imshow(bio_corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    axes[0].set_title('Bio-trained: Reciprocal Weight Correlations')
    axes[0].set_xticks(range(len(l23_types)))
    axes[0].set_yticks(range(len(l23_types)))
    axes[0].set_xticklabels(l23_types, rotation=45, ha='right')
    axes[0].set_yticklabels(l23_types)
    
    # Add correlation values as text
    for i in range(len(l23_types)):
        for j in range(len(l23_types)):
            if not np.isnan(bio_corr_matrix[i, j]):
                text_color = 'white' if abs(bio_corr_matrix[i, j]) > 0.5 else 'black'
                axes[0].text(j, i, f'{bio_corr_matrix[i, j]:.3f}', 
                           ha='center', va='center', color=text_color, fontsize=8, weight='bold')
            else:
                # Show "n.d." for no data
                axes[0].text(j, i, 'n.d.', ha='center', va='center', color='gray', fontsize=6)
    
    # Naive heatmap
    im2 = axes[1].imshow(naive_corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    axes[1].set_title('Naive: Reciprocal Weight Correlations')
    axes[1].set_xticks(range(len(l23_types)))
    axes[1].set_yticks(range(len(l23_types)))
    axes[1].set_xticklabels(l23_types, rotation=45, ha='right')
    axes[1].set_yticklabels(l23_types)
    
    # Add correlation values as text
    for i in range(len(l23_types)):
        for j in range(len(l23_types)):
            if not np.isnan(naive_corr_matrix[i, j]):
                text_color = 'white' if abs(naive_corr_matrix[i, j]) > 0.5 else 'black'
                axes[1].text(j, i, f'{naive_corr_matrix[i, j]:.3f}', 
                           ha='center', va='center', color=text_color, fontsize=8, weight='bold')
            else:
                # Show "n.d." for no data
                axes[1].text(j, i, 'n.d.', ha='center', va='center', color='gray', fontsize=6)
    
    # Difference heatmap
    # Determine color scale for difference
    diff_max = np.nanmax(np.abs(diff_matrix))
    if np.isnan(diff_max):
        diff_max = 0.5
    im3 = axes[2].imshow(diff_matrix, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max, aspect='equal')
    axes[2].set_title('Difference (Bio - Naive)')
    axes[2].set_xticks(range(len(l23_types)))
    axes[2].set_yticks(range(len(l23_types)))
    axes[2].set_xticklabels(l23_types, rotation=45, ha='right')
    axes[2].set_yticklabels(l23_types)
    
    # Add difference values as text
    for i in range(len(l23_types)):
        for j in range(len(l23_types)):
            if not np.isnan(diff_matrix[i, j]):
                text_color = 'white' if abs(diff_matrix[i, j]) > diff_max * 0.5 else 'black'
                axes[2].text(j, i, f'{diff_matrix[i, j]:.3f}', 
                           ha='center', va='center', color=text_color, fontsize=8, weight='bold')
            else:
                # Show "n.d." for no data
                axes[2].text(j, i, 'n.d.', ha='center', va='center', color='gray', fontsize=6)
    
    # Add colorbars
    plt.colorbar(im1, ax=axes[0], label='Correlation')
    plt.colorbar(im2, ax=axes[1], label='Correlation') 
    plt.colorbar(im3, ax=axes[2], label='Difference')
    
    # Set common labels and add grid
    for ax in axes:
        ax.set_xlabel('Target Cell Type')
        ax.set_ylabel('Source Cell Type')
        # Add grid lines to separate cells
        ax.set_xticks(np.arange(-0.5, len(l23_types), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(l23_types), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return bio_corr_matrix, naive_corr_matrix, diff_matrix, l23_types


def create_l23_summary_statistics(bio_data, naive_data, output_file):
    """Create summary statistics for L2/3 connections"""
    
    # Define L2/3 cell type pairs
    l23_pairs = [
        ('L2/3_Exc', 'L2/3_Exc'),
        ('L2/3_Exc', 'L2/3_PV'),
        ('L2/3_Exc', 'L2/3_SST'),
        ('L2/3_Exc', 'L2/3_VIP'),
        ('L2/3_PV', 'L2/3_Exc'),
        ('L2/3_PV', 'L2/3_PV'),
        ('L2/3_PV', 'L2/3_SST'),
        ('L2/3_PV', 'L2/3_VIP'),
        ('L2/3_SST', 'L2/3_Exc'),
        ('L2/3_SST', 'L2/3_PV'),
        ('L2/3_SST', 'L2/3_SST'),
        ('L2/3_SST', 'L2/3_VIP'),
        ('L2/3_VIP', 'L2/3_Exc'),
        ('L2/3_VIP', 'L2/3_PV'),
        ('L2/3_VIP', 'L2/3_SST'),
        ('L2/3_VIP', 'L2/3_VIP')
    ]
    
    results = []
    
    for source_type, target_type in l23_pairs:
        bio_subset = bio_data[
            (bio_data['source_type_a_to_b'] == source_type) & 
            (bio_data['target_type_a_to_b'] == target_type)
        ]
        naive_subset = naive_data[
            (naive_data['source_type_a_to_b'] == source_type) & 
            (naive_data['target_type_a_to_b'] == target_type)
        ]
        
        if len(bio_subset) > 0 and len(naive_subset) > 0:
            # Calculate statistics by network
            bio_by_network = bio_subset.groupby('network')['log_correlation'].mean()
            naive_by_network = naive_subset.groupby('network')['log_correlation'].mean()
            
            # Ensure we have the same networks
            common_networks = sorted(set(bio_by_network.index) & set(naive_by_network.index))
            
            if len(common_networks) >= 3:  # Need at least 3 networks for meaningful statistics
                bio_values = [bio_by_network[net] for net in common_networks]
                naive_values = [naive_by_network[net] for net in common_networks]
                
                # Perform paired t-test (since same networks)
                t_stat, p_value = stats.ttest_rel(bio_values, naive_values)
                
                # Calculate effect size (Cohen's d for paired samples)
                diff_values = np.array(bio_values) - np.array(naive_values)
                cohens_d = np.mean(diff_values) / np.std(diff_values, ddof=1)
                
                results.append({
                    'source_type': source_type,
                    'target_type': target_type,
                    'bio_mean': np.mean(bio_values),
                    'naive_mean': np.mean(naive_values),
                    'mean_difference': np.mean(bio_values) - np.mean(naive_values),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'n_networks': len(common_networks),
                    'bio_connections': len(bio_subset),
                    'naive_connections': len(naive_subset)
                })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # Apply multiple comparisons correction
        _, p_corrected, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_corrected'] = p_corrected
        
        # Sort by effect size
        results_df = results_df.sort_values('cohens_d', key=abs, ascending=False)
    
    results_df.to_csv(output_file, index=False)
    return results_df


def main():
    print("=== L2/3 Reciprocal Connection Analysis ===")
    
    # Load data
    print("Loading bio-trained L2/3 data...")
    bio_data = load_l23_networks_data("reciprocal_L23_detailed", "bio_trained")
    
    print("Loading naive L2/3 data...")
    naive_data = load_l23_networks_data("reciprocal_L23_detailed_naive", "naive")
    
    if len(bio_data) == 0 or len(naive_data) == 0:
        print("Error: No data loaded. Check that the analysis has been run.")
        return
    
    print(f"Loaded {len(bio_data)} bio-trained and {len(naive_data)} naive reciprocal L2/3 connections")
    
    # Create output directory
    output_dir = "reciprocal_L23_summary"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create L2/3 Exc vs PV focused plots
    print("Creating L2/3 Exc vs PV focused plots...")
    create_l23_pv_focused_plots(bio_data, naive_data, output_dir)
    
    # Create correlation heatmap
    print("Creating L2/3 correlation heatmap...")
    bio_matrix, naive_matrix, diff_matrix, l23_types = create_l23_correlation_heatmap(
        bio_data, naive_data, f"{output_dir}/L23_correlation_heatmap.png"
    )
    
    # Create summary statistics
    print("Creating L2/3 summary statistics...")
    results_df = create_l23_summary_statistics(bio_data, naive_data, f"{output_dir}/L23_comparison_statistics.csv")
    
    # Print correlation matrix results
    print("\n=== L2/3 RECIPROCAL WEIGHT CORRELATIONS ===")
    print("Bio-trained correlations:")
    for i, source in enumerate(l23_types):
        for j, target in enumerate(l23_types):
            if not np.isnan(bio_matrix[i, j]):
                print(f"  {source} → {target}: R = {bio_matrix[i, j]:.3f}")
    
    print("\nKey correlation differences (Bio - Naive):")
    # Find largest differences
    diff_indices = []
    for i in range(len(l23_types)):
        for j in range(len(l23_types)):
            if not np.isnan(diff_matrix[i, j]):
                diff_indices.append((abs(diff_matrix[i, j]), i, j, diff_matrix[i, j]))
    
    diff_indices.sort(reverse=True)
    for abs_diff, i, j, diff in diff_indices[:5]:  # Top 5 differences
        print(f"  {l23_types[i]} → {l23_types[j]}: Δ = {diff:+.3f} (Bio: {bio_matrix[i, j]:.3f}, Naive: {naive_matrix[i, j]:.3f})")
    
    # Print key results
    print("\n=== KEY L2/3 ASYMMETRY RESULTS ===")
    if len(results_df) > 0:
        # Focus on Exc-PV interactions
        exc_pv_results = results_df[
            ((results_df['source_type'] == 'L2/3_Exc') & (results_df['target_type'] == 'L2/3_PV')) |
            ((results_df['source_type'] == 'L2/3_PV') & (results_df['target_type'] == 'L2/3_Exc'))
        ]
        
        print("L2/3 Excitatory ↔ PV Interactions:")
        for _, row in exc_pv_results.iterrows():
            print(f"  {row['source_type']} → {row['target_type']}: "
                  f"Bio={row['bio_mean']:.3f}, Naive={row['naive_mean']:.3f}, "
                  f"Δ={row['mean_difference']:.3f}, d={row['cohens_d']:.3f}, "
                  f"p={row['p_value']:.3e}")
        
        print("\nAll L2/3 connections (sorted by effect size):")
        for _, row in results_df.head(10).iterrows():
            significance = "***" if row['p_corrected'] < 0.001 else "**" if row['p_corrected'] < 0.01 else "*" if row['p_corrected'] < 0.05 else ""
            print(f"  {row['source_type']} → {row['target_type']}: "
                  f"Bio={row['bio_mean']:.3f}, Naive={row['naive_mean']:.3f}, "
                  f"Δ={row['mean_difference']:.3f}, d={row['cohens_d']:.3f}, "
                  f"p={row['p_corrected']:.3e} {significance}")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    main() 