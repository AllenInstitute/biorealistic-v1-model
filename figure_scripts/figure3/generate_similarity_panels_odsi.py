from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 3: boxplots (raw metric) with similarity heatmap underneath")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--radius", type=float, default=200.0)
    parser.add_argument("--outdir", type=Path, default=Path("figures/paper/figure3"))
    parser.add_argument("--log_eps", type=float, default=1e-3)
    parser.add_argument("--e_only", action="store_true")
    parser.add_argument("--figsize-w", type=float, default=7.5)
    parser.add_argument("--figsize-h", type=float, default=4.0)
    parser.add_argument("--box-height-ratio", type=float, default=2.8)
    parser.add_argument("--heatmap-height-ratio", type=float, default=0.4)
    parser.add_argument("--cbar-width-ratio", type=float, default=1.2)
    parser.add_argument("--hspace", type=float, default=0.005)
    parser.add_argument("--tight-layout-pad", type=float, default=0.08)
    parser.add_argument("--bottom", type=float, default=0.25)
    parser.add_argument("--fontsize", type=float, default=11)
    parser.add_argument("--ylabel-fontsize", type=float, default=11.5)
    parser.add_argument("--xtick-fontsize", type=float, default=12)
    parser.add_argument("--annot-fontsize", type=float, default=9)
    args = parser.parse_args()

    df = discover_and_aggregate(
        args.root.resolve(),
        core_radius=args.radius,
        include_variants={"bio_trained": "Trained", "plain": "Untrained"},
    )
    if df.empty:
        print("No aggregated data found.")
        return

    if args.e_only and "ei" in df.columns:
        df = df[df["ei"] == "e"]

    if "Spont_Rate(Hz)" in df.columns and "Spontaneous rate (Hz)" not in df.columns:
        df["Spontaneous rate (Hz)"] = df["Spont_Rate(Hz)"]

    metrics = [
        ("Spontaneous rate (Hz)", True, "spontaneous_rate_similarity_combined.png"),
        ("Rate at preferred direction (Hz)", True, "firing_rate_similarity_combined.png"),
        ("OSI", False, "osi_similarity_combined.png"),
        ("DSI", False, "dsi_similarity_combined.png"),
    ]

    pal = dataset_palette()
    args.outdir.mkdir(parents=True, exist_ok=True)

    style_overrides = {
        "font.family": "Arial",
        "font.size": args.fontsize,
        "axes.labelsize": args.ylabel_fontsize,
        "xtick.labelsize": args.xtick_fontsize,
        "ytick.labelsize": args.fontsize,
        "legend.fontsize": args.fontsize,
    }

    for metric, do_log, fname in metrics:
        df_metric = add_l5_exc_combined(df, metric)
        mat_sim, ks_per_type, x_order, _pal_unused = compute_similarity(
            df_metric,
            metric=metric,
            log_transform=do_log,
            log_eps=args.log_eps,
            include_naive=False,
        )
        if mat_sim.empty:
            print(f"No data for metric {metric}")
            continue

        present_cell_types = df_metric["cell_type"].unique().tolist()
        cell_types = [ct for ct in cell_type_order() if ct in present_cell_types]
        datasets_heatmap = ["Untrained", "Trained"]
        datasets_boxplot = ["Untrained", "Trained", "Neuropixels"]

        # Calculate and print median similarity scores
        print(f"\nMedian Similarity Scores for {metric}:")
        for ds in datasets_heatmap:
            if ds in mat_sim.columns:
                median_sim = mat_sim[ds].median()
                print(f"  {ds}: {median_sim:.4f}")

        draw_metric_boxplot_with_similarity_heatmap(
            df_metric,
            metric,
            mat_sim,
            datasets_boxplot=datasets_boxplot,
            datasets_heatmap=datasets_heatmap,
            palette=pal,
            cell_types=cell_types,
            out_path=args.outdir / fname,
            figsize=(float(args.figsize_w), float(args.figsize_h)),
            height_ratios=[float(args.box_height_ratio), float(args.heatmap_height_ratio)],
            width_ratios=[24, float(args.cbar_width_ratio)],
            hspace=float(args.hspace),
            tight_layout_pad=float(args.tight_layout_pad),
            bottom=float(args.bottom),
            heatmap_xtick_fontsize=args.xtick_fontsize,
            heat_annot_fontsize=args.annot_fontsize,
            boxplot_ylabel_fontsize=args.ylabel_fontsize,
            style_overrides=style_overrides,
        )
        print(f"Saved {fname}")


if __name__ == "__main__":
    main()
