from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aggregate_boxplots_odsi import discover_and_aggregate
from aggregate_similarity_odsi import add_l5_exc_combined, compute_similarity
from image_decoding.plot_utils import (
    cell_type_order,
    dataset_palette,
    draw_metric_boxplot_with_similarity_heatmap,
)

def main():
    parser = argparse.ArgumentParser(description="Tune Fig 3 firing rate boxplot + heatmap font sizes.")
    parser.add_argument("--fontsize", type=float, default=12, help="Base font size for the plot")
    parser.add_argument("--ylabel-fontsize", type=float, default=None, help="Font size for Y axis labels")
    parser.add_argument("--xtick-fontsize", type=float, default=None, help="Font size for X tick labels")
    parser.add_argument("--annot-fontsize", type=float, default=None, help="Font size for heatmap annotations")
    parser.add_argument("--output", type=Path, default=Path("figures/paper/figure3/firing_rate_similarity_tuned.png"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--radius", type=float, default=200.0)
    args = parser.parse_args()

    # Default sub-font sizes to base fontsize if not provided
    ylabel_fs = args.ylabel_fontsize if args.ylabel_fontsize is not None else args.fontsize
    xtick_fs = args.xtick_fontsize if args.xtick_fontsize is not None else args.fontsize
    annot_fs = args.annot_fontsize if args.annot_fontsize is not None else args.fontsize

    include_variants = {
        "bio_trained": "Trained",
        "plain": "Untrained",
    }

    print("Aggregating data...")
    df = discover_and_aggregate(
        args.root.resolve(),
        core_radius=args.radius,
        include_variants=include_variants,
    )
    
    if df.empty:
        print("No aggregated data found.")
        return

    metric = "Rate at preferred direction (Hz)"
    print(f"Computing similarity for {metric}...")
    
    # Standard Figure 3 processing: add L5_Exc combined
    df_metric = add_l5_exc_combined(df, metric)
    
    # Compute similarity (KS-based, same as original)
    # The original order was Neuropixels, Trained, Untrained.
    # We want Untrained, Trained, Neuropixels.
    mat_sim, ks_per_type, x_order, _pal_unused = compute_similarity(
        df_metric,
        metric=metric,
        log_transform=True,
        log_eps=1e-3,
        include_naive=False,
    )
    
    if mat_sim.empty:
        print(f"No similarity data for {metric}")
        return

    # Define desired order: Untrained, Trained
    # Neuropixels is the reference for similarity, so it's handled separately in draw function
    datasets_heatmap = ["Untrained", "Trained"]
    datasets_boxplot = ["Untrained", "Trained", "Neuropixels"]

    present_cell_types = df_metric["cell_type"].unique().tolist()
    cell_types = [ct for ct in cell_type_order() if ct in present_cell_types]

    pal = dataset_palette()
    
    # Apply global style overrides via rcParams for the boxplot part
    style_overrides = {
        "font.family": "Arial",
        "font.size": args.fontsize,
        "axes.labelsize": ylabel_fs,
        "xtick.labelsize": xtick_fs,
        "ytick.labelsize": args.fontsize,
        "legend.fontsize": args.fontsize,
    }

    print(f"Generating plot with fontsize knobs: base={args.fontsize}, ylabel={ylabel_fs}, xtick={xtick_fs}, annot={annot_fs}")
    
    # Call the original drawing function with overrides
    draw_metric_boxplot_with_similarity_heatmap(
        df_metric,
        metric,
        mat_sim,
        datasets_boxplot=datasets_boxplot,
        datasets_heatmap=datasets_heatmap,
        palette=pal,
        cell_types=cell_types,
        out_path=args.output,
        style_overrides=style_overrides,
        heatmap_xtick_fontsize=xtick_fs,
        heat_annot_fontsize=annot_fs,
        boxplot_ylabel_fontsize=ylabel_fs,
    )
    
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

