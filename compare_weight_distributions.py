#!/usr/bin/env python
"""
Compare weight distributions between original network and trained uniform network.
Analyze whether the trained weights got closer to or farther from the original distribution.

Usage:
    python compare_weight_distributions.py [network_number]
    
    network_number: Integer 0-9 representing which network to analyze (default: 0)
"""
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import network_utils as nu
from pathlib import Path
import sys
import argparse

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Compare weight distributions between networks')
    parser.add_argument('network_number', type=int, nargs='?', default=0,
                        help='Network number to analyze (0-9)')
    return parser.parse_args()

args = parse_args()
network_number = args.network_number

# Validate network number
if network_number < 0 or network_number > 9:
    print(f"Error: Network number must be between 0 and 9, got {network_number}")
    sys.exit(1)

# Define directories and paths based on network number
ORIGINAL_DIR = f"core_nll_{network_number}"
TRAINED_DIR = f"core_nll_{network_number}_uniform"
ORIGINAL_EDGES_FILE = f"{ORIGINAL_DIR}/network/v1_v1_edges.h5"
TRAINED_EDGES_FILE = f"{TRAINED_DIR}/network/v1_v1_edges_noweightloss.h5"
OUTPUT_DIR = "weight_analysis"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print(f"Analyzing original network: {ORIGINAL_EDGES_FILE}")
print(f"Comparing with trained network: {TRAINED_EDGES_FILE}")

# Load cell types
ctdf = nu.get_cell_type_table()
cell_types = ctdf['cell_type'].unique()
print(f"Found {len(cell_types)} cell types")

# Load nodes information for both networks
orig_nodes = nu.load_nodes(ORIGINAL_DIR, expand=True)
trained_nodes = nu.load_nodes(TRAINED_DIR, expand=True)

# Load edges data
with h5py.File(ORIGINAL_EDGES_FILE, 'r') as f:
    orig_edges = {
        'source': np.array(f['edges/v1_to_v1/source_node_id']),
        'target': np.array(f['edges/v1_to_v1/target_node_id']),
        'weight': np.array(f['edges/v1_to_v1/0/syn_weight'])
    }
    orig_edges = pd.DataFrame(orig_edges)

with h5py.File(TRAINED_EDGES_FILE, 'r') as f:
    trained_edges = {
        'source': np.array(f['edges/v1_to_v1/source_node_id']),
        'target': np.array(f['edges/v1_to_v1/target_node_id']),
        'weight': np.array(f['edges/v1_to_v1/0/syn_weight'])
    }
    trained_edges = pd.DataFrame(trained_edges)

print(f"Original edges: {len(orig_edges)}")
print(f"Trained edges: {len(trained_edges)}")

# Map node IDs to cell types
def map_to_cell_types(edges, nodes):
    """Map edge source and target to their respective cell types"""
    # For load_nodes with expand=True, node_id is already the index
    # Create nodeid to cell_type mapping for faster lookups
    node_to_celltype = nodes['cell_type'].to_dict()
    
    # Map source and target nodes to cell types
    edges['source_cell_type'] = edges['source'].map(node_to_celltype)
    edges['target_cell_type'] = edges['target'].map(node_to_celltype)
    
    # Identify excitatory/inhibitory nature based on weight sign
    # Excitatory = positive weights, Inhibitory = negative weights
    edges['is_excitatory'] = edges['weight'] > 0
    
    return edges

orig_edges = map_to_cell_types(orig_edges, orig_nodes)
trained_edges = map_to_cell_types(trained_edges, trained_nodes)

# Calculate the uniform weights used as starting point
# These should be the average of original excitatory and inhibitory weights
# Using weight sign to determine excitatory/inhibitory connections
orig_exc_mean = orig_edges[orig_edges['weight'] > 0]['weight'].mean()
orig_inh_mean = orig_edges[orig_edges['weight'] < 0]['weight'].mean()

print(f"Original excitatory mean weight: {orig_exc_mean:.6f}")
print(f"Original inhibitory mean weight: {orig_inh_mean:.6f}")

# Function to calculate mean weights for each cell type pair
def calculate_pair_means(edges):
    """Calculate mean weights for each source-target cell type pair"""
    return edges.groupby(['source_cell_type', 'target_cell_type'])['weight'].mean().reset_index()

# Calculate mean weights for each cell type pair
orig_means = calculate_pair_means(orig_edges)
trained_means = calculate_pair_means(trained_edges)

# Merge the means for comparison
comparison = pd.merge(
    orig_means, 
    trained_means, 
    on=['source_cell_type', 'target_cell_type'],
    suffixes=('_orig', '_trained')
)

# Add the uniform starting weights based on source cell type
comparison['is_excitatory'] = comparison['source_cell_type'].str.contains('Exc')
comparison['weight_uniform'] = comparison['is_excitatory'].map({True: orig_exc_mean, False: orig_inh_mean})

# Calculate distances
comparison['dist_uniform_to_orig'] = np.abs(comparison['weight_orig'] - comparison['weight_uniform'])
comparison['dist_trained_to_orig'] = np.abs(comparison['weight_trained'] - comparison['weight_orig'])

# Determine if the trained weight got closer or farther compared to uniform
comparison['result'] = np.where(
    comparison['dist_trained_to_orig'] < comparison['dist_uniform_to_orig'],
    'Closer',
    np.where(
        comparison['dist_trained_to_orig'] > comparison['dist_uniform_to_orig'],
        'Farther',
        'Same'
    )
)

# Create a cell type x cell type result matrix
result_matrix = pd.DataFrame(
    index=cell_types,
    columns=cell_types
)

# Fill the matrix with results
for _, row in comparison.iterrows():
    source_type = row['source_cell_type']
    target_type = row['target_cell_type']
    result_matrix.loc[source_type, target_type] = row['result']

# Count results
closer_count = (comparison['result'] == 'Closer').sum()
farther_count = (comparison['result'] == 'Farther').sum()
same_count = (comparison['result'] == 'Same').sum()
total_count = closer_count + farther_count + same_count

# Print counts
print("\nOverall Results:")
print(f"Total cell type pairs analyzed: {total_count}")
print(f"Closer: {closer_count} ({100*closer_count/total_count:.1f}%)")
print(f"Farther: {farther_count} ({100*farther_count/total_count:.1f}%)")
print(f"Same: {same_count} ({100*same_count/total_count:.1f}%)")

def plot_weight_comparison_matrix(result_matrix, cell_types, output_file=None):
    """Create a visualization of the weight comparison results matrix.
    
    Parameters:
    -----------
    result_matrix : pandas.DataFrame
        DataFrame with source cell types as index and target cell types as columns.
        Values are 'Closer', 'Farther', or 'Same'.
    cell_types : list
        List of cell type names.
    output_file : str, optional
        Path to save the figure. If None, the figure is just displayed.    
    """
    # Create a new figure with the right size ratio for the cell grid
    plt.figure(figsize=(16, 14))
    
    # Identify layer boundaries based on cell type names
    # Assume cell types are named with layer prefix (e.g., L1_, L2/3_, etc.)
    layer_boundaries = []
    current_layer = None
    for i, cell_type in enumerate(cell_types):
        layer = cell_type.split('_')[0]  # Extract layer part (L1, L2/3, etc.)
        if layer != current_layer:
            if i > 0:  # Don't add boundary at the start
                layer_boundaries.append(i - 0.5)
            current_layer = layer
    
    # Create a grid of rectangles for the heatmap
    for i, row_idx in enumerate(result_matrix.index):
        for j, col_idx in enumerate(result_matrix.columns):
            val = result_matrix.loc[row_idx, col_idx]
            
            # Default color for 'No Connection'
            rect_color = [0.85, 0.85, 0.85]  # Light grey
            
            if pd.notna(val):
                if val == 'Closer':
                    rect_color = [0.3, 0.7, 0.3]  # Green
                elif val == 'Same':
                    rect_color = [0.2, 0.5, 0.8]  # Blue
                elif val == 'Farther':
                    rect_color = [0.9, 0.3, 0.3]  # Red
                
                # Add text label in the center of each cell
                plt.text(j, i, val,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8,
                        color='black',
                        fontweight='bold')
            
            # Create rectangle patch with white border
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor=rect_color, edgecolor='white', linewidth=0.5)
            plt.gca().add_patch(rect)
    
    # Add bold layer boundary lines
    for boundary in layer_boundaries:
        # Horizontal boundaries (across rows)
        plt.axhline(y=boundary, color='black', linewidth=2, alpha=0.7)
        # Vertical boundaries (across columns)
        plt.axvline(x=boundary, color='black', linewidth=2, alpha=0.7)
    
    plt.title('Did trained weights get closer to original distribution?', fontsize=16)
    plt.xlabel('Target Cell Type', fontsize=14)
    plt.ylabel('Source Cell Type', fontsize=14)
    
    # Add a legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='#4daf4a', alpha=0.7, label='Closer'),
        plt.Rectangle((0, 0), 1, 1, color='#377eb8', alpha=0.7, label='Same'),
        plt.Rectangle((0, 0), 1, 1, color='#e41a1c', alpha=0.7, label='Farther'),
        plt.Rectangle((0, 0), 1, 1, color='lightgrey', alpha=0.7, label='No Connection')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Set up the ticks and labels
    plt.xticks(np.arange(len(cell_types)), cell_types, rotation=90)
    plt.yticks(np.arange(len(cell_types)), cell_types)
    plt.xlim(-0.5, len(cell_types)-0.5)
    plt.ylim(-0.5, len(cell_types)-0.5)
    plt.gca().invert_yaxis()  # Invert y-axis to match DataFrame orientation
    
    # Save the figure if output_file is specified
    if output_file:
        plt.tight_layout()
        plt.savefig(output_file, dpi=200)
        print(f"Saved visualization to {output_file}")
    
    return plt.gcf()  # Return the figure for further modifications if needed

# Create a visualization of the weight comparison matrix
output_file = f"{OUTPUT_DIR}/weight_comparison_matrix_network_{network_number}.png"
plot_weight_comparison_matrix(result_matrix, cell_types, output_file)

# Save the CSV files for future reference
comparison.to_csv(f"{OUTPUT_DIR}/weight_comparison_details_network_{network_number}.csv", index=False)
result_matrix.to_csv(f"{OUTPUT_DIR}/weight_comparison_matrix_network_{network_number}.csv")

# Print the result matrix
print("\nResult Matrix (showing whether trained weights got closer to original):")
print(result_matrix)

print(f"\nAnalysis completed for network {network_number}")
print(f"Output files saved to directory: {OUTPUT_DIR}")
print(f"  - {output_file}")
print(f"  - {OUTPUT_DIR}/weight_comparison_details_network_{network_number}.csv")
print(f"  - {OUTPUT_DIR}/weight_comparison_matrix_network_{network_number}.csv")

if __name__ == "__main__":
    # If you want to recreate just the plot with different parameters, you can use:
    # plot_weight_comparison_matrix(result_matrix, cell_types, "new_visualization.png")
    pass
