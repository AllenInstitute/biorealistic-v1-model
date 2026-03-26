#!/usr/bin/env python
"""
Compare log-normal parameters between PSP characterization data and network weights.
Analyze whether the actual network synaptic weights match the log-normal distribution
parameters derived from experimental PSP characterization.
"""
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import network_utils as nu
from pathlib import Path
from tqdm import tqdm

# Define directories and paths
NETWORK_DIR = "core_nll_0/network"
OUTPUT_DIR = "plots"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

def load_psp_characterization(verbose=True):
    """
    Load PSP characterization data from CSV file.
    
    Returns:
    --------
    pandas.DataFrame: Dataframe containing PSP characterization data
    """
    # Load PSP characterization file (using space delimiter)
    psp_filepath = "base_props/psp_characterization.csv"
    psp_df = pd.read_csv(psp_filepath, delimiter=' ', skipinitialspace=True)
    
    if verbose:
        # Print info about the logn_shape columns
        logn_shape_cols = [col for col in psp_df.columns if 'logn_shape' in col]
        print(f"Found logn_shape columns: {logn_shape_cols}")
        
        for col in logn_shape_cols:
            non_nan_count = psp_df[col].notna().sum()
            print(f"Column {col}: {non_nan_count} non-NaN values")
            
            # Show example values
            if non_nan_count > 0:
                example_values = psp_df[col].dropna().head(3)
                print(f"Example values from {col}:")
                for val in example_values:
                    print(f"  {val} (type: {type(val)})")
    
    return psp_df

def load_network_data(network_dir=NETWORK_DIR):
    """
    Load network data from h5 files.
    
    Parameters:
    -----------
    network_dir : str
        Directory containing network files
    
    Returns:
    --------
    tuple: (nodes_df, edges_df)
    """
    # Use network_utils to load nodes data - with expand=True to get DataFrame
    base_dir = network_dir.split('/network')[0]
    nodes_df = nu.load_nodes(base_dir, expand=True)
    print(f"Loaded {len(nodes_df)} nodes using network_utils")
    
    # Load edges data directly from HDF5
    edges_file = f"{network_dir}/v1_v1_edges.h5"
    with h5py.File(edges_file, 'r') as f:
        edges_data = {
            'source': np.array(f['edges/v1_to_v1/source_node_id']),
            'target': np.array(f['edges/v1_to_v1/target_node_id']),
            'weight': np.array(f['edges/v1_to_v1/0/syn_weight'])
        }
    edges_df = pd.DataFrame(edges_data)
    
    print(f"Loaded {len(edges_df)} edges")
    return nodes_df, edges_df

def map_nodes_to_cell_types(nodes_df):
    """
    Map nodes to cell types.
    
    Parameters:
    -----------
    nodes_df : pandas.DataFrame
        DataFrame containing node information from nu.load_nodes(expand=True)
    
    Returns:
    --------
    dict: Mapping from node_id to cell_type
    """
    # Create a dictionary mapping node_ids to cell_types
    # With expand=True, node_id becomes the index of the DataFrame
    node_to_cell_type = nodes_df['cell_type'].to_dict()
    
    print(f"Mapped {len(node_to_cell_type)} nodes to cell types")
    print(f"Example cell types: {list(set(node_to_cell_type.values()))[:5]}")
    return node_to_cell_type

def map_edges_to_cell_types(edges_df, node_to_cell_type):
    """
    Map edge source and target nodes to their respective cell types.
    
    Parameters:
    -----------
    edges_df : pandas.DataFrame
        DataFrame containing edge information
    node_to_cell_type : dict
        Dictionary mapping node_ids to cell_types
    
    Returns:
    --------
    pandas.DataFrame: DataFrame with additional source_cell_type and target_cell_type columns
    """
    # Map source and target to their cell types
    edges_df['source_cell_type'] = edges_df['source'].map(node_to_cell_type)
    edges_df['target_cell_type'] = edges_df['target'].map(node_to_cell_type)
    
    # Identify excitatory/inhibitory nature based on source cell type
    edges_df['is_excitatory'] = edges_df['source_cell_type'].str.contains('Exc')
    
    # Count how many edges were successfully mapped
    mapped_edges = edges_df[~edges_df['source_cell_type'].isna() & ~edges_df['target_cell_type'].isna()]
    print(f"Successfully mapped {len(mapped_edges)} of {len(edges_df)} edges to cell types")
    
    return edges_df

def calculate_log_std(edges_df, cell_types, verbose=True):
    """
    Calculate standard deviation in log space for network synaptic weights.
    
    Parameters:
    -----------
    edges_df : pandas.DataFrame
        DataFrame containing edge information with source_cell_type and target_cell_type columns
    cell_types : list
        List of cell type names
    verbose : bool
        Whether to print debug information
    
    Returns:
    --------
    pandas.DataFrame: Matrix of standard deviation values in log space
    """
    # Create empty DataFrame for log std values with cell types as index and columns
    log_std_df = pd.DataFrame(index=cell_types, columns=cell_types)
    
    # Function to calculate log std for a group of weights
    def calc_log_std(weights):
        # Filter out zero or negative weights
        positive_weights = weights[weights > 0]
        if len(positive_weights) > 0:
            # Calculate standard deviation in log space
            return np.std(np.log(positive_weights))
        else:
            return np.nan
    
    print("Calculating standard deviation in log space...")
    # Use tqdm for progress tracking
    cell_type_pairs = [(s, t) for s in cell_types for t in cell_types]
    for source_type, target_type in tqdm(cell_type_pairs, desc="Processing cell type pairs"):
        # Filter edges for this source-target pair
        mask = (edges_df['source_cell_type'] == source_type) & (edges_df['target_cell_type'] == target_type)
        edges_subset = edges_df[mask]
        
        if len(edges_subset) > 0:
            # Calculate log std
            log_std = calc_log_std(edges_subset['weight'].values)
            log_std_df.loc[source_type, target_type] = log_std
    
    # Report statistics
    non_nan_count = log_std_df.count().sum()
    total_cells = len(cell_types) * len(cell_types)
    fill_percentage = 100 * non_nan_count / total_cells
    print(f"\nFilled {non_nan_count} out of {total_cells} cells in the log_std matrix ({fill_percentage:.2f}%)")
    
    return log_std_df

def extract_logn_shape(psp_df, cell_types, verbose=True):
    """
    Extract log-normal shape parameters from PSP characterization data.
    
    Parameters:
    -----------
    psp_df : pandas.DataFrame
        DataFrame containing PSP characterization data
    cell_types : list
        List of cell type names
    verbose : bool
        Whether to print debug information
    
    Returns:
    --------
    pandas.DataFrame: Matrix of log-normal shape parameters
    """
    # Create matrix for log-normal shape values
    logn_shape_df = pd.DataFrame(index=cell_types, columns=cell_types)
    
    # Create mappings for PSP types to simplify later lookups
    psp_mapping = {
        # Excitatory cell mappings
        'l23_pyr': 'L2/3_Exc',
        'l4_pyr': 'L4_Exc',
        'l5_ET_pyr': 'L5_Exc',  # Map to L5_Exc
        'l5_IT_pyr': 'L5_Exc',  # Map to L5_Exc
        'l5_pyr': 'L5_Exc',
        'l6_pyr': 'L6_Exc',
        
        # Inhibitory cell mappings
        'pvalb': 'PV',  # Will be matched partially to L2/3_PV, L4_PV, etc.
        'sst': 'SST',   # Will be matched partially to L2/3_SST, L4_SST, etc.
        'vip': 'VIP',   # Will be matched partially to L2/3_VIP, L4_VIP, etc.
    }
    
    # Print PSP types for debugging
    psp_pre_types = sorted(psp_df['pre'].unique())
    psp_post_types = sorted(psp_df['post'].unique())
    if verbose:
        print(f"PSP pre types: {psp_pre_types}")
        print(f"PSP post types: {psp_post_types}\n")
    
    # Dictionary to store PSP shape values by connection type
    psp_logn_shape = {}
    
    # Process PSP data to get log-normal shape values for each connection type
    for pre_type in psp_pre_types:
        for post_type in psp_post_types:
            # Filter PSP data for the current pre-post type pair
            mask = (psp_df['pre'] == pre_type) & (psp_df['post'] == post_type)
            filtered_data = psp_df[mask]
            
            if not filtered_data.empty:
                # Extract log-normal shape parameter (use 90th percentile value if available, otherwise resting)
                if filtered_data['logn_shape_90th'].notna().any() and filtered_data['logn_shape_90th'].iloc[0] != '[]':
                    logn_shape_str = filtered_data['logn_shape_90th'].iloc[0]
                    if isinstance(logn_shape_str, str) and logn_shape_str != '[]':
                        logn_shape = float(logn_shape_str)
                        if verbose:
                            print(f"Connection {pre_type}->{post_type}: Using logn_shape_90th = {logn_shape}")
                    else:
                        if verbose:
                            print(f"Connection {pre_type}->{post_type}: Invalid logn_shape_90th value")
                        continue
                elif filtered_data['logn_shape_resting'].notna().any() and filtered_data['logn_shape_resting'].iloc[0] != '[]':
                    logn_shape_str = filtered_data['logn_shape_resting'].iloc[0]
                    if isinstance(logn_shape_str, str) and logn_shape_str != '[]':
                        logn_shape = float(logn_shape_str)
                        if verbose:
                            print(f"Connection {pre_type}->{post_type}: Using logn_shape_resting = {logn_shape}")
                    else:
                        if verbose:
                            print(f"Connection {pre_type}->{post_type}: Invalid logn_shape_resting value")
                        continue
                else:
                    if verbose:
                        print(f"Connection {pre_type}->{post_type}: No valid log-normal shape values found")
                    continue
                
                # Store the log-normal shape for this connection type
                psp_logn_shape[(pre_type, post_type)] = logn_shape
    
    # Fill the log-normal shape matrix for network cell types
    for source_type in cell_types:
        for target_type in cell_types:
            # Match the network cell type to PSP cell type using the mapping
            matched_pre_type = None
            matched_post_type = None
            
            for psp_type, net_marker in psp_mapping.items():
                # Check if this PSP type corresponds to the source network cell type
                if net_marker in source_type:
                    matched_pre_type = psp_type
                
                # Check if this PSP type corresponds to the target network cell type
                if net_marker in target_type:
                    matched_post_type = psp_type
            
            # If we found matches for both pre and post types
            if matched_pre_type and matched_post_type:
                # Try to get the specific log-normal shape value
                logn_shape = psp_logn_shape.get((matched_pre_type, matched_post_type))
                
                # If no specific value, try more generic options (using 'all_pyr' or 'all_int')
                if logn_shape is None:
                    # Determine if source is excitatory or inhibitory
                    if 'Exc' in source_type:
                        generic_pre = 'all_pyr'
                    else:
                        generic_pre = 'all_int'
                    
                    # Determine if target is excitatory or inhibitory
                    if 'Exc' in target_type:
                        generic_post = 'all_pyr'
                    else:
                        generic_post = 'all_int'
                    
                    # Try to get a more generic log-normal shape value
                    logn_shape = psp_logn_shape.get((generic_pre, matched_post_type)) or \
                                psp_logn_shape.get((matched_pre_type, generic_post)) or \
                                psp_logn_shape.get((generic_pre, generic_post))
                
                # Set the value in the matrix if we found a matching logn_shape
                if logn_shape is not None:
                    logn_shape_df.loc[source_type, target_type] = logn_shape
    
    # Report statistics on the filled matrix
    non_nan_count = logn_shape_df.count().sum()
    total_cells = len(cell_types) * len(cell_types)
    fill_percentage = 100 * non_nan_count / total_cells
    print(f"\nFilled {non_nan_count} out of {total_cells} cells in the logn_shape matrix ({fill_percentage:.2f}%)")
    
    return logn_shape_df

def create_heatmap(matrix, title, output_file=None, vmin=None, vmax=None, cmap='viridis'):
    """
    Plot a heatmap of a matrix.
    
    Parameters:
    -----------
    matrix : pandas.DataFrame
        Matrix to plot
    title : str
        Title for the plot
    output_file : str, optional
        Path to save the figure
    vmin, vmax : float, optional
        Minimum and maximum values for the colorbar
    cmap : str, optional
        Colormap to use
    """
    plt.figure(figsize=(12, 10))
    
    # Plot the heatmap
    ax = sns.heatmap(matrix.astype(float), cmap=cmap, annot=False, 
                     linewidths=0.5, vmin=vmin, vmax=vmax)
    
    # Add bold lines to separate layers
    plt.title(title, fontsize=16)
    plt.xlabel('Target Cell Type', fontsize=14)
    plt.ylabel('Source Cell Type', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=200)
        print(f"Saved visualization to {output_file}")
    
    return plt.gcf()

def main():
    """Main function to analyze log-normal parameters"""
    print("Loading data...")
    
    # Load PSP characterization data
    psp_df = load_psp_characterization()
    
    # Use network_utils to get cell types table
    ctdf = nu.get_cell_type_table()
    cell_types = ctdf['cell_type'].unique()
    print(f"Found {len(cell_types)} cell types: {cell_types[:5]}...")
    
    # Extract log-normal shape parameters from PSP characterization
    print("Extracting log-normal shape parameters...")
    logn_shape_df = extract_logn_shape(psp_df, cell_types)
    
    # Save to CSV for later reference
    logn_shape_df.to_csv(f"{OUTPUT_DIR}/logn_shape_from_psp.csv")
    
    # Load network data
    print("Loading network data...")
    nodes_df, edges_df = load_network_data()
    
    # Map nodes to cell types
    node_to_cell_type = map_nodes_to_cell_types(nodes_df)
    
    # Map edges to cell types
    print("Mapping edges to cell types...")
    edges_df = map_edges_to_cell_types(edges_df, node_to_cell_type)
    
    # Calculate standard deviation in log space
    print("Calculating standard deviation in log space...")
    log_std_df = calculate_log_std(edges_df, cell_types)
    
    # Save to CSV for later reference
    log_std_df.to_csv(f"{OUTPUT_DIR}/log_std_from_network.csv")
    
    # Calculate the difference between log-normal shape and log std
    print("Calculating differences...")
    # Only subtract for cells where both values exist
    common_indices = logn_shape_df.dropna(how='all').index.intersection(log_std_df.dropna(how='all').index)
    common_columns = logn_shape_df.dropna(how='all', axis=1).columns.intersection(log_std_df.dropna(how='all', axis=1).columns)
    
    # Initialize difference dataframe with NaN values
    diff_df = pd.DataFrame(index=cell_types, columns=cell_types)
    
    # Fill only where both values exist
    for source in common_indices:
        for target in common_columns:
            if pd.notna(logn_shape_df.loc[source, target]) and pd.notna(log_std_df.loc[source, target]):
                diff_df.loc[source, target] = logn_shape_df.loc[source, target] - log_std_df.loc[source, target]
    
    # Save to CSV for later reference
    diff_df.to_csv(f"{OUTPUT_DIR}/logn_difference.csv")
    difference_df = log_std_df - logn_shape_df
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Determine global min and max values for consistent color scaling
    global_min = min(logn_shape_df.min().min(), log_std_df.min().min())
    global_max = max(logn_shape_df.max().max(), log_std_df.max().max())
    
    create_heatmap(logn_shape_df, "Log-Normal Shape from PSP Characterization", f"{OUTPUT_DIR}/logn_shape_from_psp.png")
    create_heatmap(log_std_df, "Log Standard Deviation from Network Weights", f"{OUTPUT_DIR}/log_std_from_network.png")
    create_heatmap(diff_df, "Difference (PSP Shape - Network Log Std)", f"{OUTPUT_DIR}/logn_difference.png")
    
    print("Analysis completed. Results are in the plots directory.")

if __name__ == "__main__":
    main()
