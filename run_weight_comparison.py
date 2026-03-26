#!/usr/bin/env python
"""
Helper script to run weight distribution comparisons on multiple networks.
Allows running the analysis on all networks (0-9) or a specific subset.
When analyzing all networks, also creates an aggregated result showing how
often each cell type pair gets closer or farther across networks.
"""
import subprocess
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Run weight distribution comparison for multiple networks')
    parser.add_argument('--networks', type=str, default='all',
                        help='Network numbers to analyze, comma-separated (e.g., "0,1,5") or "all" for all networks')
    return parser.parse_args()

def create_aggregated_analysis(networks):
    """Create an aggregated analysis of results from multiple networks"""
    print("\nCreating aggregated analysis across networks...")
    
    # First, check if we have enough data
    valid_networks = []
    for network in networks:
        matrix_file = f"weight_analysis/weight_comparison_matrix_network_{network}.csv"
        if os.path.exists(matrix_file):
            valid_networks.append(network)
    
    if not valid_networks:
        print("No valid result matrices found. Cannot create aggregated analysis.")
        return
    
    print(f"Found valid results for {len(valid_networks)} networks")
    
    # Load the first matrix to get the cell types and dimensions
    first_matrix = pd.read_csv(f"weight_analysis/weight_comparison_matrix_network_{valid_networks[0]}.csv", index_col=0)
    cell_types = first_matrix.index.tolist()
    
    # Initialize counters for 'Closer', 'Farther', 'Same'
    closer_count = pd.DataFrame(0, index=cell_types, columns=cell_types)
    farther_count = pd.DataFrame(0, index=cell_types, columns=cell_types)
    same_count = pd.DataFrame(0, index=cell_types, columns=cell_types)
    total_count = pd.DataFrame(0, index=cell_types, columns=cell_types)
    
    # Count occurrences of each result across networks
    for network in valid_networks:
        matrix_file = f"weight_analysis/weight_comparison_matrix_network_{network}.csv"
        matrix = pd.read_csv(matrix_file, index_col=0)
        
        for source in cell_types:
            for target in cell_types:
                if pd.notna(matrix.loc[source, target]):
                    total_count.loc[source, target] += 1
                    if matrix.loc[source, target] == 'Closer':
                        closer_count.loc[source, target] += 1
                    elif matrix.loc[source, target] == 'Farther':
                        farther_count.loc[source, target] += 1
                    elif matrix.loc[source, target] == 'Same':
                        same_count.loc[source, target] += 1
    
    # Calculate percentages
    closer_percent = closer_count / total_count * 100
    closer_percent = closer_percent.fillna(0)
    
    # Create a visualization of the aggregated results
    plt.figure(figsize=(18, 16))
    
    # Identify layer boundaries based on cell type names
    layer_boundaries = []
    current_layer = None
    for i, cell_type in enumerate(cell_types):
        layer = cell_type.split('_')[0]  # Extract layer part (L1, L2/3, etc.)
        if layer != current_layer:
            if i > 0:  # Don't add boundary at the start
                layer_boundaries.append(i - 0.5)
            current_layer = layer
    
    # Create a grid of rectangles colored by percentage of 'Closer' results
    for i, row_idx in enumerate(closer_percent.index):
        for j, col_idx in enumerate(closer_percent.columns):
            count = total_count.loc[row_idx, col_idx]
            if count > 0:
                closer_pct = closer_percent.loc[row_idx, col_idx]
                
                # Calculate color based on percentage
                # Use a gradient: red for mostly 'Farther', green for mostly 'Closer'
                if closer_pct > 50:
                    # More 'Closer' than 'Farther'
                    intensity = (closer_pct - 50) / 50  # 0 to 1
                    rect_color = [0.9 - 0.6 * intensity, 0.9, 0.9 - 0.6 * intensity]  # White to Green
                else:
                    # More 'Farther' than 'Closer'
                    intensity = (50 - closer_pct) / 50  # 0 to 1
                    rect_color = [0.9, 0.9 - 0.6 * intensity, 0.9 - 0.6 * intensity]  # White to Red
                
                # Add text label showing counts
                plt.text(j, i, f"{closer_count.loc[row_idx, col_idx]:g}/{count:g}\n{closer_pct:.1f}%",
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8,
                        color='black',
                        fontweight='bold')
            else:
                rect_color = [0.85, 0.85, 0.85]  # Light grey for no data
            
            # Create rectangle patch
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor=rect_color, edgecolor='white', linewidth=0.5)
            plt.gca().add_patch(rect)
    
    # Add layer boundary lines
    for boundary in layer_boundaries:
        plt.axhline(y=boundary, color='black', linewidth=2, alpha=0.7)
        plt.axvline(x=boundary, color='black', linewidth=2, alpha=0.7)
    
    plt.title('Percentage of networks where trained weights got closer to original distribution', fontsize=16)
    plt.xlabel('Target Cell Type', fontsize=14)
    plt.ylabel('Source Cell Type', fontsize=14)
    
    # Add a legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='#2ca02c', alpha=0.7, label='100% Closer'),
        plt.Rectangle((0, 0), 1, 1, color='#d7ffd7', alpha=0.7, label='~60% Closer'),
        plt.Rectangle((0, 0), 1, 1, color='#f7f7f7', alpha=0.7, label='50% Closer/Farther'),
        plt.Rectangle((0, 0), 1, 1, color='#ffd7d7', alpha=0.7, label='~60% Farther'),
        plt.Rectangle((0, 0), 1, 1, color='#d62728', alpha=0.7, label='100% Farther'),
        plt.Rectangle((0, 0), 1, 1, color='lightgrey', alpha=0.7, label='No Data')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Set up the ticks and labels
    plt.xticks(np.arange(len(cell_types)), cell_types, rotation=90)
    plt.yticks(np.arange(len(cell_types)), cell_types)
    plt.xlim(-0.5, len(cell_types)-0.5)
    plt.ylim(-0.5, len(cell_types)-0.5)
    plt.gca().invert_yaxis()  # Invert y-axis to match DataFrame orientation
    
    # Save the aggregated visualization
    plt.tight_layout()
    output_file = "weight_analysis/weight_comparison_matrix_aggregated.png"
    plt.savefig(output_file, dpi=200)
    print(f"Saved aggregated visualization to {output_file}")
    
    # Save the count data
    closer_count.to_csv("weight_analysis/aggregated_closer_count.csv")
    farther_count.to_csv("weight_analysis/aggregated_farther_count.csv")
    total_count.to_csv("weight_analysis/aggregated_total_count.csv")
    closer_percent.to_csv("weight_analysis/aggregated_closer_percent.csv")
    
    # Print overall statistics
    total_cells = total_count.values.sum()
    total_closer = closer_count.values.sum()
    total_farther = farther_count.values.sum()
    total_same = same_count.values.sum()
    
    print("\nOverall Aggregated Statistics:")
    print(f"Total cell type pairs analyzed across all networks: {total_cells:g}")
    print(f"Total 'Closer' outcomes: {total_closer:g} ({100*total_closer/total_cells:.1f}%)")
    print(f"Total 'Farther' outcomes: {total_farther:g} ({100*total_farther/total_cells:.1f}%)")
    print(f"Total 'Same' outcomes: {total_same:g} ({100*total_same/total_cells:.1f}%)")

def main():
    args = parse_args()
    
    # Determine which networks to analyze
    if args.networks.lower() == 'all':
        networks = list(range(10))  # 0-9
        run_aggregation = True
    else:
        try:
            networks = [int(n) for n in args.networks.split(',')]
            # Validate network numbers
            for n in networks:
                if n < 0 or n > 9:
                    print(f"Error: Network number {n} is invalid. Must be between 0 and 9.")
                    return
            # Only run aggregation if more than one network is specified
            run_aggregation = len(networks) > 1
        except ValueError:
            print("Error: Invalid network list format. Use comma-separated integers (e.g., '0,1,5').")
            return
    
    # Create weight_analysis directory if it doesn't exist
    Path("weight_analysis").mkdir(exist_ok=True)
    
    # Run comparison for each network
    completed_networks = []
    for network in networks:
        # Check if both original and uniform directories exist
        original_dir = f"core_nll_{network}"
        uniform_dir = f"core_nll_{network}_uniform"
        
        if not os.path.exists(original_dir):
            print(f"Skipping network {network}: Original directory {original_dir} does not exist")
            continue
            
        if not os.path.exists(uniform_dir):
            print(f"Skipping network {network}: Uniform directory {uniform_dir} does not exist")
            continue
        
        # Check if edge files exist
        original_edges = f"{original_dir}/network/v1_v1_edges.h5"
        uniform_edges = f"{uniform_dir}/network/v1_v1_edges_noweightloss.h5"
        
        if not os.path.exists(original_edges):
            print(f"Skipping network {network}: Edge file {original_edges} not found")
            continue
            
        if not os.path.exists(uniform_edges):
            print(f"Skipping network {network}: Edge file {uniform_edges} not found")
            continue
        
        print(f"\n{'='*50}")
        print(f"Running weight comparison for network {network}")
        print(f"{'='*50}")
        
        try:
            # Run the comparison script for this network
            subprocess.run(['python', 'compare_weight_distributions.py', str(network)], check=True)
            completed_networks.append(network)
        except subprocess.CalledProcessError as e:
            print(f"Error running comparison for network {network}: {e}")
    
    print("\nNetwork comparisons completed!")
    print(f"Results are saved in the weight_analysis directory")
    
    # Create aggregated analysis if applicable
    if run_aggregation and completed_networks:
        create_aggregated_analysis(completed_networks)

if __name__ == "__main__":
    main()
