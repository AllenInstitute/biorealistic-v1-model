import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy import stats


def load_all_networks_data(base_dir, network_type):
    """Load correlation data from all networks and calculate reciprocal correlations"""
    pattern = f"{base_dir}/core_nll_*_{network_type}/reciprocal_connections_all_types_{network_type}.csv"
    all_files = glob.glob(pattern)
    
    if len(all_files) == 0:
        raise ValueError(f"No files found matching pattern: {pattern}")
    
    all_correlations = []
    
    for file_path in all_files:
        # Extract network name from path
        network_name = os.path.basename(os.path.dirname(file_path)).replace(f"_{network_type}", "")
        
        df = pd.read_csv(file_path)
        
        # Create simplified source and target types
        df['source_type'] = df['source_type_a_to_b'].copy()
        df['target_type'] = df['target_type_a_to_b'].copy()
        
        # Aggregate inhibitory cell types across layers
        inh_mapping = {}
        for layer in ["L1", "L2/3", "L4", "L5", "L6"]:
            for inh_type in ["PV", "SST", "VIP"]:
                inh_mapping[f"{layer}_{inh_type}"] = inh_type
        
        df['source_type'] = df['source_type'].replace(inh_mapping)
        df['target_type'] = df['target_type'].replace(inh_mapping)
        
        # Calculate correlations for each cell type pair
        grouped = df.groupby(['source_type', 'target_type'])
        
        for (source_type, target_type), group in grouped:
            if len(group) >= 10:  # Need enough data points for meaningful correlation
                # Calculate Pearson correlation between A→B and B→A log weights
                log_corr, p_value = stats.pearsonr(group['log_weight_a_to_b'], group['log_weight_b_to_a'])
                
                all_correlations.append({
                    'network': network_name,
                    'network_type': network_type,
                    'source_type': source_type,
                    'target_type': target_type,
                    'log_correlation': log_corr,
                    'p_value': p_value,
                    'n_connections': len(group)
                })
    
    return pd.DataFrame(all_correlations)


def create_aggregated_heatmap(bio_data, naive_data, output_file):
    """Create aggregated correlation heatmaps comparing bio_trained vs naive"""
    
    # Get unique cell type pairs
    all_pairs = set()
    for df in [bio_data, naive_data]:
        pairs = [(row['source_type'], row['target_type']) for _, row in df.iterrows()]
        all_pairs.update(pairs)
    
    all_source_types = sorted(list(set([pair[0] for pair in all_pairs])))
    all_target_types = sorted(list(set([pair[1] for pair in all_pairs])))
    
    # Calculate mean correlations for each network type
    bio_matrix = np.full((len(all_source_types), len(all_target_types)), np.nan)
    naive_matrix = np.full((len(all_source_types), len(all_target_types)), np.nan)
    bio_sem_matrix = np.full((len(all_source_types), len(all_target_types)), np.nan)
    naive_sem_matrix = np.full((len(all_source_types), len(all_target_types)), np.nan)
    
    def fill_matrix(data, matrix, sem_matrix):
        grouped = data.groupby(['source_type', 'target_type'])['log_correlation']
        means = grouped.mean()
        sems = grouped.sem()
        
        for (source, target), mean_corr in means.items():
            if source in all_source_types and target in all_target_types:
                i = all_source_types.index(source)
                j = all_target_types.index(target)
                matrix[i, j] = mean_corr
                sem_matrix[i, j] = sems[(source, target)]
    
    fill_matrix(bio_data, bio_matrix, bio_sem_matrix)
    fill_matrix(naive_data, naive_matrix, naive_sem_matrix)
    
    # Create difference matrix
    diff_matrix = bio_matrix - naive_matrix
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Common parameters for heatmaps
    vmin, vmax = -0.5, 0.5
    
    # Bio-trained heatmap
    im1 = axes[0].imshow(bio_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
    axes[0].set_title('Bio-trained Networks\n(Mean Log-space Correlation)', fontsize=12)
    axes[0].set_xticks(range(len(all_target_types)))
    axes[0].set_yticks(range(len(all_source_types)))
    axes[0].set_xticklabels(all_target_types, rotation=45, ha='right')
    axes[0].set_yticklabels(all_source_types)
    
    # Naive heatmap
    im2 = axes[1].imshow(naive_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
    axes[1].set_title('Naive Networks\n(Mean Log-space Correlation)', fontsize=12)
    axes[1].set_xticks(range(len(all_target_types)))
    axes[1].set_yticks(range(len(all_source_types)))
    axes[1].set_xticklabels(all_target_types, rotation=45, ha='right')
    axes[1].set_yticklabels([])  # No y-labels for middle plot
    
    # Difference heatmap
    diff_vmax = np.nanmax(np.abs(diff_matrix))
    im3 = axes[2].imshow(diff_matrix, cmap='RdBu_r', vmin=-diff_vmax, vmax=diff_vmax, aspect='equal')
    axes[2].set_title('Difference\n(Bio-trained - Naive)', fontsize=12)
    axes[2].set_xticks(range(len(all_target_types)))
    axes[2].set_yticks(range(len(all_source_types)))
    axes[2].set_xticklabels(all_target_types, rotation=45, ha='right')
    axes[2].set_yticklabels([])  # No y-labels for right plot
    
    # Add significance markers to bio and naive heatmaps
    for ax, data, matrix in [(axes[0], bio_data, bio_matrix), (axes[1], naive_data, naive_matrix)]:
        grouped = data.groupby(['source_type', 'target_type'])['log_correlation']
        for (source, target), group in grouped:
            if source in all_source_types and target in all_target_types:
                i = all_source_types.index(source)
                j = all_target_types.index(target)
                
                # Test if correlation is significantly different from 0
                _, p_value = stats.ttest_1samp(group.dropna(), 0)
                
                if p_value < 0.001:
                    marker = '***'
                elif p_value < 0.01:
                    marker = '**'
                elif p_value < 0.05:
                    marker = '*'
                else:
                    marker = ''
                
                if marker:
                    text_color = 'white' if abs(matrix[i, j]) > (vmax - vmin) * 0.3 else 'black'
                    ax.text(j, i, marker, ha='center', va='center', 
                           color=text_color, fontsize=10, weight='bold')
    
    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label('Log-space Correlation', rotation=270, labelpad=15)
    
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label('Log-space Correlation', rotation=270, labelpad=15)
    
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
    cbar3.set_label('Difference', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return bio_matrix, naive_matrix, diff_matrix, all_source_types, all_target_types


def create_summary_statistics(bio_data, naive_data, output_file):
    """Create summary statistics comparing bio_trained vs naive networks"""
    
    # Calculate basic statistics
    bio_stats = bio_data.groupby(['source_type', 'target_type'])['log_correlation'].agg([
        'count', 'mean', 'std', 'sem'
    ]).reset_index()
    bio_stats['network_type'] = 'bio_trained'
    
    naive_stats = naive_data.groupby(['source_type', 'target_type'])['log_correlation'].agg([
        'count', 'mean', 'std', 'sem'
    ]).reset_index()
    naive_stats['network_type'] = 'naive'
    
    # Combine statistics
    all_stats = pd.concat([bio_stats, naive_stats], ignore_index=True)
    
    # Perform statistical tests
    comparison_results = []
    
    grouped_bio = bio_data.groupby(['source_type', 'target_type'])['log_correlation']
    grouped_naive = naive_data.groupby(['source_type', 'target_type'])['log_correlation']
    
    for (source, target) in set(bio_data[['source_type', 'target_type']].itertuples(index=False, name=None)):
        bio_group = grouped_bio.get_group((source, target)) if (source, target) in grouped_bio.groups else pd.Series(dtype=float)
        naive_group = grouped_naive.get_group((source, target)) if (source, target) in grouped_naive.groups else pd.Series(dtype=float)
        
        if len(bio_group) > 0 and len(naive_group) > 0:
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(bio_group.dropna(), naive_group.dropna(), equal_var=False)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(bio_group) - 1) * bio_group.var() + 
                                 (len(naive_group) - 1) * naive_group.var()) / 
                                (len(bio_group) + len(naive_group) - 2))
            cohens_d = (bio_group.mean() - naive_group.mean()) / pooled_std if pooled_std > 0 else np.nan
            
            comparison_results.append({
                'source_type': source,
                'target_type': target,
                'bio_mean': bio_group.mean(),
                'naive_mean': naive_group.mean(),
                'mean_difference': bio_group.mean() - naive_group.mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'bio_n': len(bio_group),
                'naive_n': len(naive_group)
            })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    # Apply multiple testing correction
    if len(comparison_df) > 0:
        from statsmodels.stats.multitest import multipletests
        _, p_corrected, _, _ = multipletests(comparison_df['p_value'].dropna(), 
                                           method='bonferroni', alpha=0.05)
        comparison_df.loc[comparison_df['p_value'].notna(), 'p_corrected'] = p_corrected
    
    # Save results
    all_stats.to_csv(output_file.replace('.csv', '_detailed_stats.csv'), index=False)
    comparison_df.to_csv(output_file, index=False)
    
    return all_stats, comparison_df


def create_distribution_plots(bio_data, naive_data, output_file):
    """Create distribution plots comparing bio_trained vs naive"""
    
    # Select key cell type pairs for detailed visualization
    key_pairs = [
        ('L2/3_Exc', 'PV'),
        ('L2/3_Exc', 'SST'), 
        ('L2/3_Exc', 'L2/3_Exc'),
        ('PV', 'L2/3_Exc'),
        ('SST', 'L2/3_Exc'),
        ('L5_IT', 'L5_IT'),
        ('L6_Exc', 'L6_Exc')
    ]
    
    # Filter for available pairs
    available_pairs = []
    for source, target in key_pairs:
        bio_subset = bio_data[(bio_data['source_type'] == source) & (bio_data['target_type'] == target)]
        naive_subset = naive_data[(naive_data['source_type'] == source) & (naive_data['target_type'] == target)]
        if len(bio_subset) > 0 and len(naive_subset) > 0:
            available_pairs.append((source, target))
    
    if len(available_pairs) == 0:
        print("No common cell type pairs found between bio_trained and naive networks")
        return
    
    # Create subplot grid
    n_pairs = len(available_pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_pairs == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, (source, target) in enumerate(available_pairs):
        ax = axes[idx]
        
        bio_subset = bio_data[(bio_data['source_type'] == source) & (bio_data['target_type'] == target)]
        naive_subset = naive_data[(naive_data['source_type'] == source) & (naive_data['target_type'] == target)]
        
        # Create histograms
        bins = np.linspace(-1, 1, 31)
        ax.hist(bio_subset['log_correlation'], bins=bins, alpha=0.6, label='Bio-trained', 
                color='blue', density=True)
        ax.hist(naive_subset['log_correlation'], bins=bins, alpha=0.6, label='Naive', 
                color='red', density=True)
        
        # Add vertical lines for means
        ax.axvline(bio_subset['log_correlation'].mean(), color='blue', linestyle='--', linewidth=2)
        ax.axvline(naive_subset['log_correlation'].mean(), color='red', linestyle='--', linewidth=2)
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(bio_subset['log_correlation'], naive_subset['log_correlation'])
        
        ax.set_title(f'{source} → {target}\np = {p_value:.3f}', fontsize=10)
        ax.set_xlabel('Log-space Correlation')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(available_pairs), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_network_consistency_plot(bio_data, naive_data, output_file):
    """Create plots showing consistency across networks"""
    
    # Calculate correlation for each cell type pair in each network
    bio_by_network = bio_data.groupby(['network', 'source_type', 'target_type'])['log_correlation'].mean().reset_index()
    naive_by_network = naive_data.groupby(['network', 'source_type', 'target_type'])['log_correlation'].mean().reset_index()
    
    # Select a few key cell type pairs
    key_pairs = [('L2/3_Exc', 'PV'), ('L2/3_Exc', 'SST'), ('L2/3_Exc', 'L2/3_Exc')]
    
    fig, axes = plt.subplots(1, len(key_pairs), figsize=(5 * len(key_pairs), 5))
    if len(key_pairs) == 1:
        axes = [axes]
    
    for idx, (source, target) in enumerate(key_pairs):
        ax = axes[idx]
        
        bio_subset = bio_by_network[(bio_by_network['source_type'] == source) & 
                                   (bio_by_network['target_type'] == target)]
        naive_subset = naive_by_network[(naive_by_network['source_type'] == source) & 
                                       (naive_by_network['target_type'] == target)]
        
        if len(bio_subset) > 0 and len(naive_subset) > 0:
            # Box plots for each network type
            networks = sorted(bio_subset['network'].unique())
            bio_values = [bio_subset[bio_subset['network'] == net]['log_correlation'].values for net in networks]
            naive_values = [naive_subset[naive_subset['network'] == net]['log_correlation'].values for net in networks]
            
            positions_bio = np.arange(len(networks)) - 0.2
            positions_naive = np.arange(len(networks)) + 0.2
            
            bp1 = ax.boxplot(bio_values, positions=positions_bio, widths=0.3, 
                           patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.6))
            bp2 = ax.boxplot(naive_values, positions=positions_naive, widths=0.3,
                           patch_artist=True, boxprops=dict(facecolor='red', alpha=0.6))
            
            ax.set_xticks(range(len(networks)))
            ax.set_xticklabels([net.replace('core_nll_', '') for net in networks])
            ax.set_xlabel('Network')
            ax.set_ylabel('Log-space Correlation')
            ax.set_title(f'{source} → {target}')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='blue', alpha=0.6, label='Bio-trained'),
                             Patch(facecolor='red', alpha=0.6, label='Naive')]
            ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Load data from both network types
    print("Loading bio-trained network data...")
    bio_data = load_all_networks_data("reciprocal_all_networks", "bio_trained")
    
    print("Loading naive network data...")
    naive_data = load_all_networks_data("reciprocal_all_networks_naive", "naive")
    
    print(f"Loaded {len(bio_data)} bio-trained and {len(naive_data)} naive reciprocal connections")
    
    # Create output directory
    os.makedirs("reciprocal_summary_plots", exist_ok=True)
    
    # Create aggregated heatmap
    print("Creating aggregated correlation heatmaps...")
    bio_matrix, naive_matrix, diff_matrix, source_types, target_types = create_aggregated_heatmap(
        bio_data, naive_data, "reciprocal_summary_plots/aggregated_correlation_heatmap.png"
    )
    
    # Create summary statistics
    print("Calculating summary statistics...")
    detailed_stats, comparison_stats = create_summary_statistics(
        bio_data, naive_data, "reciprocal_summary_plots/network_comparison_statistics.csv"
    )
    
    # Create distribution plots
    print("Creating distribution plots...")
    create_distribution_plots(
        bio_data, naive_data, "reciprocal_summary_plots/correlation_distributions.png"
    )
    
    # Create network consistency plots
    print("Creating network consistency plots...")
    create_network_consistency_plot(
        bio_data, naive_data, "reciprocal_summary_plots/network_consistency.png"
    )
    
    # Print summary
    print("\n=== SUMMARY RESULTS ===")
    print(f"Total reciprocal connections analyzed:")
    print(f"  Bio-trained: {len(bio_data):,}")
    print(f"  Naive: {len(naive_data):,}")
    
    print(f"\nCell type pairs analyzed: {len(comparison_stats)}")
    
    print(f"\nMean log-space correlations:")
    print(f"  Bio-trained: {bio_data['log_correlation'].mean():.4f} ± {bio_data['log_correlation'].std():.4f}")
    print(f"  Naive: {naive_data['log_correlation'].mean():.4f} ± {naive_data['log_correlation'].std():.4f}")
    
    # Significant differences
    significant = comparison_stats[comparison_stats['p_corrected'] < 0.05] if 'p_corrected' in comparison_stats.columns else comparison_stats[comparison_stats['p_value'] < 0.05]
    print(f"\nSignificant differences (p < 0.05): {len(significant)}/{len(comparison_stats)}")
    
    if len(significant) > 0:
        print("\nTop 5 largest effect sizes:")
        top_effects = significant.nlargest(5, 'cohens_d')[['source_type', 'target_type', 'mean_difference', 'cohens_d', 'p_value']]
        for _, row in top_effects.iterrows():
            print(f"  {row['source_type']} → {row['target_type']}: Δ = {row['mean_difference']:.3f}, d = {row['cohens_d']:.3f}, p = {row['p_value']:.3f}")
    
    print(f"\nAll plots saved to: reciprocal_summary_plots/")


if __name__ == "__main__":
    main() 