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


def get_subtype_colors():
    """Get representative colors for each cell subtype."""
    colors_df = pd.read_csv("base_props/cell_type_naming_scheme.csv", sep=r'\s+')
    
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


def load_summary(path: Path, label: str) -> pd.DataFrame:
    """Load decoding summary CSV and append a column identifying network type."""
    df = pd.read_csv(path)
    df["network_type"] = label
    return df


def main():
    parser = argparse.ArgumentParser(description="Plot bio_trained vs naive decoding accuracies.")
    parser.add_argument("--bio_csv", type=Path, default=Path("image_decoding/summary50/decoding_summary.csv"),
                        help="CSV file for bio_trained (default: summary50/decoding_summary.csv)")
    parser.add_argument("--naive_csv", type=Path, default=Path("image_decoding/summary50/decoding_summary_naive.csv"),
                        help="CSV file for naive (default: summary50/decoding_summary_naive.csv)")
    parser.add_argument("--sample_size", type=int, default=50, help="Sample size to plot (default 50)")
    parser.add_argument("--out_png", type=Path, default=Path("image_decoding/summary50/decoding_comparison_boxplot.png"),
                        help="Output PNG path")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    bio_df = load_summary(args.bio_csv, "bio_trained")
    naive_df = load_summary(args.naive_csv, "naive")
    df_all = pd.concat([bio_df, naive_df], ignore_index=True)

    # Filter by desired sample_size ---------------------------------------
    df_plot = df_all[df_all["sample_size"] == args.sample_size].copy()
    if df_plot.empty:
        raise ValueError(f"No rows with sample_size == {args.sample_size}. Available: {df_all['sample_size'].unique()}")

    # Get custom ordering
    cell_type_order = get_cell_type_order()
    available_types = set(df_plot["cell_type"].unique())
    filtered_order = [ct for ct in cell_type_order if ct in available_types]
    
    # Get subtype colors for background shading
    subtype_colors = get_subtype_colors()

    # ---------------------------------------------------------------------
    plt.figure(figsize=(14, 5))
    ax = sns.boxplot(data=df_plot, x="cell_type", y="accuracy", hue="network_type", order=filtered_order, palette="Set2")
    
    # Add background shading for each cell subtype
    add_background_shading(ax, filtered_order, subtype_colors)
    
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Decoding accuracy")
    ax.set_title(f"Image decoding accuracy (sample_size={args.sample_size})")
    plt.xticks(rotation=90)
    plt.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=300)
    plt.close()
    print(f"Saved boxplot to {args.out_png}")


if __name__ == "__main__":
    main() 