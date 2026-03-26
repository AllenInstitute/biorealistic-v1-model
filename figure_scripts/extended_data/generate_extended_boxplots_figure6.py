"""Extended data figure for Figure 6: boxplots and similarity scores for
spontaneous rate, evoked (average) rate, and DSI.

Produces three output types matching Fig. 6 style:
  1. Multi-metric aggregate boxplots (plot_boxplots)
  2. Per-metric boxplot + similarity heatmap (draw_metric_boxplot_with_similarity_heatmap)
  3. Summary similarity score boxplots (draw_similarity_summary_boxplot_multi_metric)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aggregate_boxplots_odsi import (
    discover_and_aggregate,
    plot_boxplots,
)
from aggregate_similarity_odsi import add_l5_exc_combined, compute_similarity
from image_decoding.plot_utils import (
    cell_type_order,
    dataset_palette,
    draw_metric_boxplot_with_similarity_heatmap,
    draw_similarity_summary_boxplot_multi_metric,
)


# ---------------------------------------------------------------------------
# Column renaming helpers
# ---------------------------------------------------------------------------

_RENAME_MAP = {
    "Spont_Rate(Hz)": "Spont. rate (Hz)",
    "Ave_Rate(Hz)": "Evoked rate (Hz)",
}


def _rename_extra_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=_RENAME_MAP)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extended data for Figure 6: boxplots + similarity scores for "
            "spont. rate, evoked rate, and DSI "
            "(Syn. weight distr. constrained vs unconstrained)"
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--radius", type=float, default=200.0)
    parser.add_argument("--outdir", type=Path, default=Path("figures/paper/extended_data/figure6"))
    parser.add_argument("--log_eps", type=float, default=1e-3)
    parser.add_argument("--e_only", action="store_true")
    parser.add_argument("--figsize-w", type=float, default=5.2)
    parser.add_argument("--figsize-h", type=float, default=2.1)
    parser.add_argument("--panel-w", type=float, default=7.5)
    parser.add_argument("--panel-h", type=float, default=4.0)
    parser.add_argument("--box-height-ratio", type=float, default=2.8)
    parser.add_argument("--heatmap-height-ratio", type=float, default=0.4)
    parser.add_argument("--cbar-width-ratio", type=float, default=1.2)
    parser.add_argument("--hspace", type=float, default=0.005)
    parser.add_argument("--tight-layout-pad", type=float, default=0.08)
    parser.add_argument("--bottom", type=float, default=0.25)
    parser.add_argument("--fontsize", type=float, default=10.5)
    parser.add_argument("--ylabel-fontsize", type=float, default=11.0)
    parser.add_argument("--xtick-fontsize", type=float, default=11.0)
    parser.add_argument("--annot-fontsize", type=float, default=8.5)
    args = parser.parse_args()

    label_constrained = "Syn. weight distr. constrained"
    label_unconstrained = "Syn. weight distr. unconstrained"
    heatmap_label_map = {
        label_constrained: "constrained",
        label_unconstrained: "unconstrained",
    }

    df = discover_and_aggregate(
        args.root.resolve(),
        core_radius=args.radius,
        include_variants={"bio_trained": label_constrained, "naive": label_unconstrained},
    )
    if df.empty:
        print("No aggregated data found.")
        return

    # Rename extra metric columns to human-readable labels
    df = _rename_extra_cols(df)

    if args.e_only and "ei" in df.columns:
        df = df[df["ei"] == "e"]

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Boxplots
    # ------------------------------------------------------------------
    boxplot_metrics = [
        "Spont. rate (Hz)",
        "Evoked rate (Hz)",
        "DSI",
    ]
    plot_boxplots(
        df,
        metrics=boxplot_metrics,
        output=args.outdir / "extended_boxplots_spont_evoked_dsi.pdf",
        e_only=False,
    )
    print("Saved extended_boxplots_spont_evoked_dsi.pdf")

    # ------------------------------------------------------------------
    # 2. Per-metric boxplot + similarity heatmap (same style as Fig. 6)
    # ------------------------------------------------------------------
    # (metric_col, log_transform, output_filename)
    panel_metrics = [
        ("Spont. rate (Hz)", True, "extended_spont_rate_similarity_combined.pdf"),
        ("Evoked rate (Hz)", True, "extended_evoked_rate_similarity_combined.pdf"),
        ("DSI", False, "extended_dsi_similarity_combined.pdf"),
    ]

    pal = dataset_palette()
    style_overrides = {
        "font.family": "Arial",
        "font.size": args.fontsize,
        "axes.labelsize": args.ylabel_fontsize,
        "xtick.labelsize": args.xtick_fontsize,
        "ytick.labelsize": args.fontsize,
        "legend.fontsize": args.fontsize,
    }

    for metric_col, do_log, fname in panel_metrics:
        if metric_col not in df.columns:
            print(f"Column '{metric_col}' not found, skipping.")
            continue
        df_metric = add_l5_exc_combined(df, metric_col)
        mat_sim, _ks_per_type, _x_order, _pal_unused = compute_similarity(
            df_metric,
            metric=metric_col,
            log_transform=do_log,
            log_eps=args.log_eps,
            include_naive=True,
            dataset_order_override=[label_constrained, label_unconstrained],
        )
        if mat_sim.empty:
            print(f"No similarity data for {metric_col}")
            continue

        present_cell_types = df_metric["cell_type"].unique().tolist()
        cell_types = [ct for ct in cell_type_order() if ct in present_cell_types]
        datasets_heatmap = [label_constrained, label_unconstrained]
        datasets_boxplot = [label_constrained, label_unconstrained, "Neuropixels"]

        print(f"\nMedian Similarity Scores for {metric_col}:")
        for ds in datasets_heatmap:
            if ds in mat_sim.columns:
                print(f"  {ds}: {mat_sim[ds].median():.4f}")

        draw_metric_boxplot_with_similarity_heatmap(
            df_metric,
            metric_col,
            mat_sim,
            datasets_boxplot=datasets_boxplot,
            datasets_heatmap=datasets_heatmap,
            palette=pal,
            cell_types=cell_types,
            out_path=args.outdir / fname,
            figsize=(float(args.panel_w), float(args.panel_h)),
            height_ratios=[float(args.box_height_ratio), float(args.heatmap_height_ratio)],
            width_ratios=[24, float(args.cbar_width_ratio)],
            hspace=float(args.hspace),
            tight_layout_pad=float(args.tight_layout_pad),
            bottom=float(args.bottom),
            heatmap_dataset_label_map=heatmap_label_map,
            heatmap_xtick_fontsize=args.xtick_fontsize,
            heat_annot_fontsize=args.annot_fontsize,
            boxplot_ylabel_fontsize=args.ylabel_fontsize,
            style_overrides=style_overrides,
        )
        print(f"Saved {fname}")

    # ------------------------------------------------------------------
    # 3. Similarity summary boxplots (same style as Fig. 6)
    # ------------------------------------------------------------------
    # (metric_label, metric_col, log_transform)
    sim_metrics = [
        ("Spont. rate", "Spont. rate (Hz)", True),
        ("Evoked rate", "Evoked rate (Hz)", True),
        ("DSI", "DSI", False),
    ]

    rows = []
    for metric_label, metric_col, do_log in sim_metrics:
        if metric_col not in df.columns:
            print(f"Column '{metric_col}' not found, skipping.")
            continue
        df_metric = add_l5_exc_combined(df, metric_col)
        _mat_sim, ks_per_type, _x_order, _pal_unused = compute_similarity(
            df_metric,
            metric=metric_col,
            log_transform=do_log,
            log_eps=args.log_eps,
            include_naive=True,
            dataset_order_override=[label_constrained, label_unconstrained],
        )
        if ks_per_type.empty:
            continue
        ks_per_type = ks_per_type.copy()
        ks_per_type["metric"] = metric_label
        rows.append(ks_per_type)

    if not rows:
        print("No similarity data to plot.")
        return

    combined = pd.concat(rows, ignore_index=True)
    out_sim = args.outdir / "extended_similarity_spont_evoked_dsi.pdf"
    draw_similarity_summary_boxplot_multi_metric(
        combined,
        metric_order=[m[0] for m in sim_metrics],
        dataset_order=[label_constrained, label_unconstrained],
        palette=pal,
        out_path=out_sim,
        figsize=(float(args.figsize_w), float(args.figsize_h)),
    )
    print(f"Saved {out_sim.name}")


if __name__ == "__main__":
    main()
