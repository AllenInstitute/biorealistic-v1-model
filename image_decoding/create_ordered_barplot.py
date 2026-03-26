#!/usr/bin/env python3
"""
Create bar plots for decoding performance with custom cell type ordering.
Orders by subtypes: Exc, PV, SST, VIP, Inh, with L5 ET/IT/NP between L4_Exc and L6_Exc.
"""

import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba


def get_cell_type_order():
    """Return custom cell type ordering: Exc, PV, SST, VIP, Inh subtypes."""
    # Define the desired order based on subtypes
    order = [
        # Excitatory cells by layer
        "L1_Inh",  # L1 only has inhibitory cells
        "L2/3_Exc",
        "L4_Exc", 
        "L5_ET",   # L5 excitatory subtypes between L4 and L6
        "L5_IT",
        "L5_NP", 
        "L6_Exc",
        # PV cells by layer
        "L2/3_PV",
        "L4_PV",
        "L5_PV",
        "L6_PV",
        # SST cells by layer
        "L2/3_SST",
        "L4_SST", 
        "L5_SST",
        "L6_SST",
        # VIP cells by layer
        "L2/3_VIP",
        "L4_VIP",
        "L5_VIP",
        "L6_VIP",
    ]
    return order


def load_color_scheme():
    """Load cell type colors from naming scheme."""
    colors_df = pd.read_csv("base_props/cell_type_naming_scheme.csv", sep=r'\s+')
    color_map = dict(zip(colors_df["cell_type"], colors_df["hex"]))
    return color_map, colors_df


def get_subtype_colors(colors_df):
    """Get representative colors for each cell subtype."""
    # Group by cell class and get representative colors
    subtype_colors = {}
    
    # Excitatory cells - use reddish tones
    exc_colors = colors_df[colors_df['class'] == 'Exc']['hex'].tolist()
    exc_rgb = [to_rgba(c, alpha=0.15) for c in exc_colors]
    subtype_colors['Exc'] = np.mean([c[:3] for c in exc_rgb], axis=0).tolist() + [0.15]
    
    # Inhibitory L1 - use gray tones
    inh_colors = colors_df[colors_df['class'] == 'Inh']['hex'].tolist()
    inh_rgb = [to_rgba(c, alpha=0.15) for c in inh_colors]
    subtype_colors['Inh'] = np.mean([c[:3] for c in inh_rgb], axis=0).tolist() + [0.15]
    
    # PV cells - use green tones
    pv_colors = colors_df[colors_df['class'] == 'PV']['hex'].tolist()
    pv_rgb = [to_rgba(c, alpha=0.15) for c in pv_colors]
    subtype_colors['PV'] = np.mean([c[:3] for c in pv_rgb], axis=0).tolist() + [0.15]
    
    # SST cells - use cyan tones
    sst_colors = colors_df[colors_df['class'] == 'SST']['hex'].tolist()
    sst_rgb = [to_rgba(c, alpha=0.15) for c in sst_colors]
    subtype_colors['SST'] = np.mean([c[:3] for c in sst_rgb], axis=0).tolist() + [0.15]
    
    # VIP cells - use purple tones
    vip_colors = colors_df[colors_df['class'] == 'VIP']['hex'].tolist()
    vip_rgb = [to_rgba(c, alpha=0.15) for c in vip_colors]
    subtype_colors['VIP'] = np.mean([c[:3] for c in vip_rgb], axis=0).tolist() + [0.15]
    
    return subtype_colors


def get_cell_subtype(cell_type):
    """Get the subtype (Exc, PV, SST, VIP, Inh) for a given cell type."""
    if cell_type == 'L1_Inh':
        return 'Inh'
    elif 'Exc' in cell_type or cell_type in ['L5_ET', 'L5_IT', 'L5_NP']:
        return 'Exc'
    elif 'PV' in cell_type:
        return 'PV'
    elif 'SST' in cell_type:
        return 'SST'
    elif 'VIP' in cell_type:
        return 'VIP'
    else:
        return 'Other'


def add_background_shading(ax, cell_types, subtype_colors):
    """Add faint background shading for each cell subtype."""
    # Get the x-axis positions
    x_positions = range(len(cell_types))
    
    # Group consecutive cell types by subtype
    current_subtype = None
    start_pos = 0
    
    for i, cell_type in enumerate(cell_types):
        subtype = get_cell_subtype(cell_type)
        
        if subtype != current_subtype:
            # End the previous group if it exists
            if current_subtype is not None:
                end_pos = i - 0.5
                color = subtype_colors.get(current_subtype, [0.9, 0.9, 0.9, 0.1])
                ax.axvspan(start_pos - 0.5, end_pos, facecolor=color, zorder=0)
            
            # Start new group
            current_subtype = subtype
            start_pos = i
    
    # Handle the last group
    if current_subtype is not None:
        end_pos = len(cell_types) - 0.5
        color = subtype_colors.get(current_subtype, [0.9, 0.9, 0.9, 0.1])
        ax.axvspan(start_pos - 0.5, end_pos, facecolor=color, zorder=0)


def create_ordered_barplot(csv_path: Path, output_path: Path, sample_sizes: list = None):
    """Create bar plot with custom cell type ordering."""
    if sample_sizes is None:
        sample_sizes = [10, 30, 50]
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter for desired sample sizes
    df_plot = df[df["sample_size"].isin(sample_sizes)].copy()
    
    # Get custom ordering
    cell_type_order = get_cell_type_order()
    
    # Filter to only include cell types present in data and in our desired order
    available_types = set(df_plot["cell_type"].unique())
    filtered_order = [ct for ct in cell_type_order if ct in available_types]
    
    # Load colors
    color_map, colors_df = load_color_scheme()
    subtype_colors = get_subtype_colors(colors_df)
    
    # Create the plot
    plt.figure(figsize=(14, 6))
    
    # Create barplot with custom order and subtle colors
    ax = sns.barplot(
        data=df_plot, 
        x="cell_type", 
        y="accuracy", 
        hue="sample_size", 
        order=filtered_order,
        errorbar="se",
        palette="Set2"  # More subtle color palette
    )
    
    # Add background shading for each cell subtype
    add_background_shading(ax, filtered_order, subtype_colors)
    
    # Customize plot
    ax.set_xlabel("Cell Type", fontsize=12)
    ax.set_ylabel("Decoding Accuracy", fontsize=12)
    ax.set_title("Image Decoding Accuracy by Cell Type", fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ordered bar plot to {output_path}")
    print(f"Cell type order used: {filtered_order}")


def main():
    parser = argparse.ArgumentParser(description="Create bar plots with custom cell type ordering")
    parser.add_argument("--bio_csv", type=Path, 
                        default=Path("image_decoding/summary50/decoding_summary.csv"),
                        help="Bio-trained decoding summary CSV")
    parser.add_argument("--naive_csv", type=Path, 
                        default=Path("image_decoding/summary50/decoding_summary_naive.csv"),
                        help="Naive decoding summary CSV")
    parser.add_argument("--sample_sizes", type=int, nargs="*", default=[10, 30, 50],
                        help="Sample sizes to include in plot")
    parser.add_argument("--output_dir", type=Path, 
                        default=Path("image_decoding/summary50"),
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots for both bio-trained and naive if files exist
    if args.bio_csv.exists():
        output_path = args.output_dir / "accuracy_barplot_bio_trained_ordered.png"
        create_ordered_barplot(args.bio_csv, output_path, args.sample_sizes)
    else:
        print(f"Bio-trained CSV not found: {args.bio_csv}")
    
    if args.naive_csv.exists():
        output_path = args.output_dir / "accuracy_barplot_naive_ordered.png"
        create_ordered_barplot(args.naive_csv, output_path, args.sample_sizes)
    else:
        print(f"Naive CSV not found: {args.naive_csv}")


if __name__ == "__main__":
    main() 