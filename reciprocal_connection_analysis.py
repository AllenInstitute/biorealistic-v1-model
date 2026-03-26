#!/usr/bin/env python3
"""
Reciprocal Connection Analysis

This script analyzes reciprocally connected neuron pairs in the biorealistic model,
similar to the analysis in Znamenskiy et al. It finds pairs of neurons that are
connected in both directions (A->B and B->A) and analyzes the correlation between
their synaptic weights.

Usage:
    python reciprocal_connection_analysis.py base_dir [--cell-types L2/3] [--output-dir reciprocal_analysis]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import network_utils as nu
from response_correlation_calculations import calculate_edge_df_core
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def calculate_edge_df_core_with_network_type(base_dir, network_type):
    """
    Calculate edge_df_core with the correct network type.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing network data
    network_type : str
        Network type to analyze (e.g., 'bio_trained', 'naive', 'checkpoint')
    
    Returns
    -------
    pd.DataFrame
        Edge dataframe for core neurons with network-type-specific weights
    """
    # Map network types to appendix for edge files
    network_type_mapping = {
        'bio_trained': '_bio_trained',
        'naive': '_naive', 
        'checkpoint': '_checkpoint',
        'plain': ''  # No appendix for plain/base network
    }
    
    if network_type not in network_type_mapping:
        raise ValueError(f"Unknown network_type: {network_type}. Must be one of: {list(network_type_mapping.keys())}")
    
    appendix = network_type_mapping[network_type]
    
    # Load edges with the appropriate appendix
    edge_lf = nu.load_edges_pl(base_dir, appendix=appendix)  # polars lazy frame
    node_lf = nu.load_nodes_pl(base_dir, core_radius=200)  # polars lazy frame

    cores = node_lf.select("core").collect().to_series()
    source_ids = edge_lf.select("source_id").collect().to_series()
    target_ids = edge_lf.select("target_id").collect().to_series()

    source_in_core = cores[source_ids]
    target_in_core = cores[target_ids]
    both_in_core = source_in_core & target_in_core

    edge_df_core = edge_lf.filter(both_in_core).collect().to_pandas()
    print(f"Loaded {len(edge_df_core)} core edges for network type: {network_type}")
    
    return edge_df_core


def find_reciprocal_connections(edge_df):
    """
    Find reciprocally connected neuron pairs from edge dataframe.
    
    Parameters
    ----------
    edge_df : pd.DataFrame
        Edge dataframe with columns: source_id, target_id, syn_weight, etc.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with reciprocal connections containing:
        - node_a, node_b: the two connected nodes
        - weight_a_to_b: synaptic weight from A to B
        - weight_b_to_a: synaptic weight from B to A
        - source_type_a_to_b, target_type_a_to_b: cell types for A->B connection
        - source_type_b_to_a, target_type_b_to_a: cell types for B->A connection
    """
    print("Finding reciprocal connections...")
    
    # Create a lookup for faster matching
    # Make edge identifier: sort the two IDs so (1,2) and (2,1) have same key
    edge_df = edge_df.copy()
    edge_df['edge_key'] = edge_df.apply(
        lambda row: f"{min(row['source_id'], row['target_id'])}_{max(row['source_id'], row['target_id'])}", 
        axis=1
    )
    
    # Group by edge_key to find bidirectional connections
    edge_groups = edge_df.groupby('edge_key')
    
    reciprocal_connections = []
    
    for edge_key, group in edge_groups:
        if len(group) == 2:  # Exactly two connections between the same pair
            # Sort by source_id to ensure consistent ordering
            group_sorted = group.sort_values('source_id')
            conn1 = group_sorted.iloc[0]
            conn2 = group_sorted.iloc[1]
            
            # Verify this is truly reciprocal (A->B and B->A)
            if (conn1['source_id'] == conn2['target_id'] and 
                conn1['target_id'] == conn2['source_id']):
                
                # Take absolute values and add small epsilon to avoid log(0)
                weight_a_to_b_abs = abs(conn1['syn_weight'])
                weight_b_to_a_abs = abs(conn2['syn_weight'])
                
                # Skip connections with zero weights (shouldn't happen but safety check)
                if weight_a_to_b_abs == 0 or weight_b_to_a_abs == 0:
                    continue
                    
                reciprocal_connections.append({
                    'node_a': conn1['source_id'],
                    'node_b': conn1['target_id'],
                    'weight_a_to_b': weight_a_to_b_abs,
                    'weight_b_to_a': weight_b_to_a_abs,
                    'log_weight_a_to_b': np.log(weight_a_to_b_abs),
                    'log_weight_b_to_a': np.log(weight_b_to_a_abs),
                    'source_type_a_to_b': conn1.get('source_type', 'Unknown'),
                    'target_type_a_to_b': conn1.get('target_type', 'Unknown'),
                    'source_type_b_to_a': conn2.get('source_type', 'Unknown'),
                    'target_type_b_to_a': conn2.get('target_type', 'Unknown'),
                    # Additional useful information
                    'n_syns_a_to_b': conn1.get('n_syns', 1),
                    'n_syns_b_to_a': conn2.get('n_syns', 1),
                })
    
    reciprocal_df = pd.DataFrame(reciprocal_connections)
    print(f"Found {len(reciprocal_df)} reciprocal connections")
    
    return reciprocal_df


def filter_by_cell_types(reciprocal_df, cell_types_filter=None):
    """
    Filter reciprocal connections by cell types.
    
    Parameters
    ----------
    reciprocal_df : pd.DataFrame
        DataFrame of reciprocal connections
    cell_types_filter : list or None
        List of cell types to include (e.g., ['L2/3_Exc', 'L2/3_PV'])
        If None, include all cell types
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if cell_types_filter is None:
        return reciprocal_df
    
    # Filter connections where both directions involve the specified cell types
    mask = (
        reciprocal_df['source_type_a_to_b'].isin(cell_types_filter) &
        reciprocal_df['target_type_a_to_b'].isin(cell_types_filter) &
        reciprocal_df['source_type_b_to_a'].isin(cell_types_filter) &
        reciprocal_df['target_type_b_to_a'].isin(cell_types_filter)
    )
    
    filtered_df = reciprocal_df[mask].copy()
    print(f"Filtered to {len(filtered_df)} connections involving cell types: {cell_types_filter}")
    
    return filtered_df


def aggregate_inhibitory_cell_types(reciprocal_df):
    """
    Aggregate inhibitory cell types across layers (PV, SST, VIP) for simplified analysis.
    
    Parameters
    ----------
    reciprocal_df : pd.DataFrame
        DataFrame of reciprocal connections
    
    Returns
    -------
    pd.DataFrame
        DataFrame with aggregated inhibitory cell types
    """
    reciprocal_df = reciprocal_df.copy()
    
    # Map layer-specific inhibitory types to simplified types
    inh_mapping = {}
    for layer in ["L1", "L2/3", "L4", "L5", "L6"]:
        for inh_type in ["PV", "SST", "VIP"]:
            inh_mapping[f"{layer}_{inh_type}"] = inh_type
    
    # Apply the mapping to aggregate inhibitory types across layers
    reciprocal_df['source_type_a_to_b'] = reciprocal_df['source_type_a_to_b'].replace(inh_mapping)
    reciprocal_df['target_type_a_to_b'] = reciprocal_df['target_type_a_to_b'].replace(inh_mapping)
    reciprocal_df['source_type_b_to_a'] = reciprocal_df['source_type_b_to_a'].replace(inh_mapping)
    reciprocal_df['target_type_b_to_a'] = reciprocal_df['target_type_b_to_a'].replace(inh_mapping)
    
    return reciprocal_df


def create_cell_type_pair_column(reciprocal_df):
    """
    Create a column identifying the cell type pair (order-independent).
    
    Parameters
    ----------
    reciprocal_df : pd.DataFrame
        DataFrame of reciprocal connections
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'cell_type_pair' column
    """
    reciprocal_df = reciprocal_df.copy()
    
    # Create ordered pair names for consistent grouping
    reciprocal_df['cell_type_pair'] = reciprocal_df.apply(
        lambda row: '_'.join(sorted([row['source_type_a_to_b'], row['target_type_a_to_b']])),
        axis=1
    )
    
    return reciprocal_df


def plot_reciprocal_weight_scatter(reciprocal_df, output_file, title_suffix=""):
    """
    Create scatter plot of reciprocal connection weights.
    
    Parameters
    ----------
    reciprocal_df : pd.DataFrame
        DataFrame of reciprocal connections
    output_file : str
        Output file path
    title_suffix : str
        Additional text for plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create scatter plot in log space
    ax.scatter(reciprocal_df['weight_a_to_b'], reciprocal_df['weight_b_to_a'], 
               alpha=0.6, s=20)
    
    # Calculate correlation on log-transformed values
    log_weights_a = reciprocal_df['log_weight_a_to_b'].dropna()
    log_weights_b = reciprocal_df['log_weight_b_to_a'].dropna()
    
    if len(log_weights_a) > 1 and len(log_weights_b) > 1:
        # Ensure same indices for correlation calculation
        common_idx = log_weights_a.index.intersection(log_weights_b.index)
        if len(common_idx) > 1:
            corr_pearson, p_pearson = pearsonr(log_weights_a[common_idx], log_weights_b[common_idx])
            corr_spearman, p_spearman = spearmanr(log_weights_a[common_idx], log_weights_b[common_idx])
        else:
            corr_pearson = corr_spearman = p_pearson = p_spearman = np.nan
    else:
        corr_pearson = corr_spearman = p_pearson = p_spearman = np.nan
    
    # Add diagonal reference line
    weight_min = min(reciprocal_df['weight_a_to_b'].min(), reciprocal_df['weight_b_to_a'].min())
    weight_max = max(reciprocal_df['weight_a_to_b'].max(), reciprocal_df['weight_b_to_a'].max())
    ax.plot([weight_min, weight_max], [weight_min, weight_max], 'r--', alpha=0.5, label='y=x')
    
    # Set log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set labels and title
    ax.set_xlabel('|Weight A→B|')
    ax.set_ylabel('|Weight B→A|')
    ax.set_title(f'Reciprocal Connection Weights (Log Scale){title_suffix}\n'
                f'Log-space Pearson r={corr_pearson:.3f} (p={p_pearson:.3e})\n'
                f'Log-space Spearman ρ={corr_spearman:.3f} (p={p_spearman:.3e})\n'
                f'N={len(reciprocal_df)} pairs')
    
    # Add legend and grid
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Make axes equal for better comparison in log space
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return correlation statistics
    return {
        'n_pairs': len(reciprocal_df),
        'pearson_r': corr_pearson,
        'pearson_p': p_pearson,
        'spearman_rho': corr_spearman,
        'spearman_p': p_spearman
    }


def plot_reciprocal_weights_by_cell_type(reciprocal_df, output_file):
    """
    Create scatter plots grouped by cell type pairs.
    
    Parameters
    ----------
    reciprocal_df : pd.DataFrame
        DataFrame of reciprocal connections with 'cell_type_pair' column
    output_file : str
        Output file path
    """
    # Get unique cell type pairs
    cell_type_pairs = reciprocal_df['cell_type_pair'].unique()
    
    # Calculate grid dimensions
    n_pairs = len(cell_type_pairs)
    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    if n_pairs == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    correlation_stats = {}
    
    for i, cell_pair in enumerate(cell_type_pairs):
        ax = axes[i]
        subset = reciprocal_df[reciprocal_df['cell_type_pair'] == cell_pair]
        
        # Create scatter plot in log space
        ax.scatter(subset['weight_a_to_b'], subset['weight_b_to_a'], 
                  alpha=0.6, s=20)
        
        # Calculate correlation on log-transformed values
        log_weights_a = subset['log_weight_a_to_b'].dropna()
        log_weights_b = subset['log_weight_b_to_a'].dropna()
        
        if len(log_weights_a) > 1 and len(log_weights_b) > 1:
            common_idx = log_weights_a.index.intersection(log_weights_b.index)
            if len(common_idx) > 1:
                corr_pearson, p_pearson = pearsonr(log_weights_a[common_idx], log_weights_b[common_idx])
            else:
                corr_pearson, p_pearson = np.nan, np.nan
        else:
            corr_pearson, p_pearson = np.nan, np.nan
        
        correlation_stats[cell_pair] = {
            'n_pairs': len(subset),
            'pearson_r': corr_pearson,
            'pearson_p': p_pearson
        }
        
        # Add diagonal reference line
        if len(subset) > 0:
            weight_min = min(subset['weight_a_to_b'].min(), subset['weight_b_to_a'].min())
            weight_max = max(subset['weight_a_to_b'].max(), subset['weight_b_to_a'].max())
            ax.plot([weight_min, weight_max], [weight_min, weight_max], 'r--', alpha=0.5)
        
        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Set labels and title
        ax.set_xlabel('|Weight A→B|')
        ax.set_ylabel('|Weight B→A|')
        ax.set_title(f'{cell_pair}\nLog-space r={corr_pearson:.3f}, p={p_pearson:.3e}\nN={len(subset)}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for i in range(n_pairs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlation_stats


def plot_reciprocal_correlation_heatmap(reciprocal_df, output_file, title_suffix=""):
    """
    Create a heatmap showing reciprocal connection correlation coefficients.
    
    Parameters
    ----------
    reciprocal_df : pd.DataFrame
        DataFrame of reciprocal connections with 'cell_type_pair' column
    output_file : str
        Output file path
    title_suffix : str
        Additional text for plot title
    """
    # Get unique cell types from the data
    all_source_types = set(reciprocal_df['source_type_a_to_b'].unique())
    all_target_types = set(reciprocal_df['target_type_a_to_b'].unique())
    all_cell_types = sorted(list(all_source_types | all_target_types))
    
    # Initialize matrices
    n_types = len(all_cell_types)
    corr_matrix = np.full((n_types, n_types), np.nan)
    pval_matrix = np.full((n_types, n_types), np.nan)
    count_matrix = np.zeros((n_types, n_types), dtype=int)
    
    # Create cell type to index mapping
    type_to_idx = {cell_type: i for i, cell_type in enumerate(all_cell_types)}
    
    # Calculate correlations for each pair
    for _, row in reciprocal_df.iterrows():
        source_type = row['source_type_a_to_b']
        target_type = row['target_type_a_to_b']
        
        if source_type in type_to_idx and target_type in type_to_idx:
            i, j = type_to_idx[source_type], type_to_idx[target_type]
            count_matrix[i, j] += 1
    
    # Calculate correlations for pairs with sufficient data
    for i, source_type in enumerate(all_cell_types):
        for j, target_type in enumerate(all_cell_types):
            # Get data for this specific pair (both directions)
            mask1 = (reciprocal_df['source_type_a_to_b'] == source_type) & (reciprocal_df['target_type_a_to_b'] == target_type)
            mask2 = (reciprocal_df['source_type_a_to_b'] == target_type) & (reciprocal_df['target_type_a_to_b'] == source_type)
            subset = reciprocal_df[mask1 | mask2]
            
            if len(subset) >= 10:  # Minimum threshold for correlation
                log_weights_a = subset['log_weight_a_to_b'].dropna()
                log_weights_b = subset['log_weight_b_to_a'].dropna()
                
                if len(log_weights_a) > 1 and len(log_weights_b) > 1:
                    common_idx = log_weights_a.index.intersection(log_weights_b.index)
                    if len(common_idx) > 1:
                        corr_val, p_val = pearsonr(log_weights_a[common_idx], log_weights_b[common_idx])
                        corr_matrix[i, j] = corr_val
                        pval_matrix[i, j] = p_val
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use a diverging colormap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='equal')
    
    # Set ticks and labels
    ax.set_xticks(range(n_types))
    ax.set_yticks(range(n_types))
    ax.set_xticklabels(all_cell_types, rotation=45, ha='right')
    ax.set_yticklabels(all_cell_types)
    
    # Add text annotations with correlation values and significance
    for i in range(n_types):
        for j in range(n_types):
            corr_val = corr_matrix[i, j]
            p_val = pval_matrix[i, j]
            
            if not np.isnan(corr_val):
                # Determine significance stars
                significance_str = ""
                if not np.isnan(p_val):
                    if p_val < 0.001:
                        significance_str = "***"
                    elif p_val < 0.01:
                        significance_str = "**"
                    elif p_val < 0.05:
                        significance_str = "*"
                
                # Format the text
                text = f"{corr_val:.2f}{significance_str}"
                
                # Choose text color based on background
                text_color = 'white' if abs(corr_val) > 0.3 else 'black'
                
                ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=8, weight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log-space Correlation (r)', rotation=270, labelpad=15)
    
    ax.set_title(f'Reciprocal Connection Correlations{title_suffix}')
    ax.set_xlabel('Target Cell Type')
    ax.set_ylabel('Source Cell Type')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return summary statistics
    valid_correlations = corr_matrix[~np.isnan(corr_matrix)]
    return {
        'n_cell_types': n_types,
        'n_valid_correlations': len(valid_correlations),
        'mean_correlation': np.mean(valid_correlations) if len(valid_correlations) > 0 else np.nan,
        'max_correlation': np.max(valid_correlations) if len(valid_correlations) > 0 else np.nan,
        'min_correlation': np.min(valid_correlations) if len(valid_correlations) > 0 else np.nan
    }


def save_correlation_summary(correlation_stats, output_file):
    """
    Save correlation statistics to CSV file.
    
    Parameters
    ----------
    correlation_stats : dict
        Dictionary of correlation statistics
    output_file : str
        Output CSV file path
    """
    # Convert to DataFrame for easy saving
    stats_df = pd.DataFrame.from_dict(correlation_stats, orient='index')
    stats_df.index.name = 'cell_type_pair'
    stats_df.reset_index(inplace=True)
    
    # Sort by number of pairs (descending)
    stats_df = stats_df.sort_values('n_pairs', ascending=False)
    
    stats_df.to_csv(output_file, index=False)
    print(f"Correlation summary saved to {output_file}")


def process_single_network(base_dir, cell_types_filter=None, network_type="bio_trained", output_dir="reciprocal_analysis", aggregate_inhibitory=False, heatmap_only=False):
    """
    Process a single network to find and analyze reciprocal connections.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing network data
    cell_types_filter : list or None
        List of cell types to filter for
    network_type : str
        Network type to analyze (e.g., 'bio_trained', 'naive', 'checkpoint')
    output_dir : str
        Output directory for results
    aggregate_inhibitory : bool
        Whether to aggregate inhibitory cell types across layers
    heatmap_only : bool
        Whether to generate only heatmap plots (skip scatter plots)
    
    Returns
    -------
    dict
        Summary statistics for this network
    """
    print(f"Processing network: {base_dir} (type: {network_type})")
    
    # Create output directory
    network_output_dir = os.path.join(output_dir, f"{os.path.basename(base_dir)}_{network_type}")
    os.makedirs(network_output_dir, exist_ok=True)
    
    # Load edge data with appropriate network type
    edge_df_core = calculate_edge_df_core_with_network_type(base_dir, network_type)
    
    # Add cell type information
    ctdf = nu.get_cell_type_table()
    node_lf = nu.load_nodes_pl(base_dir, core_radius=200)
    node_df = node_lf.collect().to_pandas()
    
    edge_df_core["source_type"] = ctdf["cell_type"][
        node_df["pop_name"][edge_df_core["source_id"]]
    ].values
    edge_df_core["target_type"] = ctdf["cell_type"][
        node_df["pop_name"][edge_df_core["target_id"]]
    ].values
    
    # Find reciprocal connections
    reciprocal_df = find_reciprocal_connections(edge_df_core)
    
    if len(reciprocal_df) == 0:
        print(f"No reciprocal connections found in {base_dir}")
        return {'network': base_dir, 'n_reciprocal': 0}
    
    # Filter by cell types if specified
    if cell_types_filter:
        reciprocal_df = filter_by_cell_types(reciprocal_df, cell_types_filter)
        # Sanitize cell type names for filename (replace / with _)
        sanitized_types = [ct.replace('/', '_') for ct in cell_types_filter]
        filter_suffix = f"_{'_'.join(sanitized_types)}_{network_type}"
    else:
        filter_suffix = f"_all_types_{network_type}"
    
    if len(reciprocal_df) == 0:
        print(f"No reciprocal connections found after filtering in {base_dir}")
        return {'network': base_dir, 'n_reciprocal': 0}
    
    # Aggregate inhibitory cell types if requested
    if aggregate_inhibitory:
        reciprocal_df = aggregate_inhibitory_cell_types(reciprocal_df)
        print(f"Aggregated inhibitory cell types across layers")
    
    # Add cell type pair column
    reciprocal_df = create_cell_type_pair_column(reciprocal_df)
    
    # Save raw reciprocal connections data
    reciprocal_df.to_csv(
        os.path.join(network_output_dir, f"reciprocal_connections{filter_suffix}.csv"),
        index=False
    )
    
    # Create plots based on options
    if not heatmap_only:
        # Create overall scatter plot
        overall_stats = plot_reciprocal_weight_scatter(
            reciprocal_df,
            os.path.join(network_output_dir, f"reciprocal_weights_scatter{filter_suffix}.png"),
            title_suffix=f" ({os.path.basename(base_dir)} - {network_type})"
        )
        
        # Create scatter plots by cell type (only if not too many cell types)
        unique_types = len(set(reciprocal_df['source_type_a_to_b'].unique()) | set(reciprocal_df['target_type_a_to_b'].unique()))
        if unique_types <= 15:  # Limit to avoid huge figures
            cell_type_stats = plot_reciprocal_weights_by_cell_type(
                reciprocal_df,
                os.path.join(network_output_dir, f"reciprocal_weights_by_cell_type{filter_suffix}.png")
            )
        else:
            print(f"Skipping individual cell type plots (too many types: {unique_types})")
            cell_type_stats = {}
    else:
        # Skip scatter plots
        overall_stats = {'n_pairs': len(reciprocal_df), 'pearson_r': np.nan, 'pearson_p': np.nan, 'spearman_rho': np.nan, 'spearman_p': np.nan}
        cell_type_stats = {}
    
    # Always create heatmap
    heatmap_stats = plot_reciprocal_correlation_heatmap(
        reciprocal_df,
        os.path.join(network_output_dir, f"reciprocal_correlation_heatmap{filter_suffix}.png"),
        title_suffix=f" ({os.path.basename(base_dir)} - {network_type})"
    )
    
    # Save correlation summary (only if cell type stats are available)
    if cell_type_stats:
        save_correlation_summary(
            cell_type_stats,
            os.path.join(network_output_dir, f"correlation_summary{filter_suffix}.csv")
        )
    
    # Return summary for aggregation
    return {
        'network': base_dir,
        'network_type': network_type,
        'n_reciprocal': len(reciprocal_df),
        'overall_pearson_r': overall_stats['pearson_r'],
        'overall_pearson_p': overall_stats['pearson_p'],
        'overall_spearman_rho': overall_stats['spearman_rho'],
        'overall_spearman_p': overall_stats['spearman_p'],
        'cell_type_stats': cell_type_stats,
        'heatmap_stats': heatmap_stats
    }


def get_l23_cell_types():
    """Get L2/3 cell types for filtering"""
    ctdf = nu.get_cell_type_table()
    l23_types = ctdf[ctdf['layer'] == '2/3']['cell_type'].unique().tolist()
    return l23_types


def main():
    parser = argparse.ArgumentParser(
        description="Analyze reciprocally connected neuron pairs"
    )
    parser.add_argument(
        "base_dirs", 
        nargs="+", 
        help="Base directory(ies) containing network data"
    )
    parser.add_argument(
        "--cell-types", 
        nargs="*", 
        default=None,
        help="Cell types to include in analysis (default: all types). Use 'L2/3' for L2/3 types only."
    )
    parser.add_argument(
        "--output-dir", 
        default="reciprocal_analysis",
        help="Output directory for results"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Use parallel processing for multiple networks"
    )
    parser.add_argument(
        "--network-type", 
        default="bio_trained",
        choices=["bio_trained", "naive", "checkpoint", "plain"],
        help="Network type to analyze (default: bio_trained)"
    )
    parser.add_argument(
        "--aggregate-inhibitory", 
        action="store_true",
        help="Aggregate inhibitory cell types (PV, SST, VIP) across layers"
    )
    parser.add_argument(
        "--heatmap-only", 
        action="store_true",
        help="Generate only heatmap plots (skip individual scatter plots)"
    )
    
    args = parser.parse_args()
    
    # Handle special case for L2/3 filter
    if args.cell_types and len(args.cell_types) == 1 and args.cell_types[0] == "L2/3":
        cell_types_filter = get_l23_cell_types()
        print(f"Using L2/3 cell types: {cell_types_filter}")
    else:
        cell_types_filter = args.cell_types
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process networks
    if args.parallel and len(args.base_dirs) > 1:
        print("Using parallel processing...")
        with ProcessPoolExecutor(max_workers=min(len(args.base_dirs), multiprocessing.cpu_count())) as executor:
            futures = [
                executor.submit(process_single_network, base_dir, cell_types_filter, args.network_type, args.output_dir, args.aggregate_inhibitory, args.heatmap_only)
                for base_dir in args.base_dirs
            ]
            all_results = []
            for future in as_completed(futures):
                all_results.append(future.result())
    else:
        print("Using sequential processing...")
        all_results = []
        for base_dir in args.base_dirs:
            result = process_single_network(base_dir, cell_types_filter, args.network_type, args.output_dir, args.aggregate_inhibitory, args.heatmap_only)
            all_results.append(result)
    
    # Save aggregated summary
    summary_df = pd.DataFrame(all_results)
    if cell_types_filter:
        sanitized_types = [ct.replace('/', '_') for ct in cell_types_filter]
        filter_suffix = f"_{'_'.join(sanitized_types)}_{args.network_type}"
    else:
        filter_suffix = f"_all_types_{args.network_type}"
    summary_df.to_csv(
        os.path.join(args.output_dir, f"network_summary{filter_suffix}.csv"),
        index=False
    )
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    print(f"Summary: {len(all_results)} networks processed")
    print(f"Total reciprocal connections found: {summary_df['n_reciprocal'].sum()}")


if __name__ == "__main__":
    main() 