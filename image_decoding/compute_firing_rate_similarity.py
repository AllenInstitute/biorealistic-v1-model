import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from image_decoding.plot_utils import (
    cell_type_order,
    dataset_order,
    dataset_palette,
    draw_metric_boxplot_with_similarity_heatmap,
)


def main():
    parser = argparse.ArgumentParser(description="Compute firing-rate distribution similarity vs Neuropixels per cell type")
    parser.add_argument("--np_cached_root", type=Path, default=Path("image_decoding/neuropixels/cached_rates"))
    parser.add_argument("--core_root", type=Path, default=Path("."))
    parser.add_argument("--networks", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--exclude_naive", action="store_true")
    parser.add_argument("--exclude_adjusted", action="store_true")
    parser.add_argument("--log_eps", type=float, default=1e-3, help="epsilon added before log10 to stabilize")
    parser.add_argument(
        "--boxplot_ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional y-axis limits for the firing-rate boxplot (e.g. 0 60)",
    )
    parser.add_argument("--fig_width_scale", type=float, default=1.0, help="scale factor for figure width (default 1.0)")
    parser.add_argument("--out_combined", type=Path, default=Path("image_decoding/summary/firing_rate_similarity_combined.png"))
    parser.add_argument("--fontsize", type=float, default=11)
    parser.add_argument("--ylabel-fontsize", type=float, default=11.5)
    parser.add_argument("--xtick-fontsize", type=float, default=12)
    parser.add_argument("--annot-fontsize", type=float, default=9)
    parser.add_argument("--bottom", type=float, default=0.25)
    args = parser.parse_args()

    # Reuse loading utilities from the boxplot script to ensure consistency
    from image_decoding.plot_firing_rate_boxplot import (
        load_np_unit_rates,
        load_model_unit_rates,
        add_l5_exc_combined,
    )

    df_np = load_np_unit_rates(args.np_cached_root)
    df_bio = load_model_unit_rates(args.core_root, args.networks, "bio_trained")
    df_naive = load_model_unit_rates(args.core_root, args.networks, "naive")
    df_plain = load_model_unit_rates(args.core_root, args.networks, "plain")
    df_adjusted = load_model_unit_rates(args.core_root, args.networks, "adjusted")

    data = pd.concat([df_np, df_bio, df_naive, df_plain, df_adjusted], ignore_index=True)
    data = add_l5_exc_combined(data)
    data = data.dropna(subset=["dataset", "cell_type", "firing_rate"]).copy()

    if args.exclude_naive:
        data = data[data["dataset"] != "Naive"]
    if args.exclude_adjusted:
        data = data[data["dataset"] != "Adjusted"]

    # Only compare types present in Neuropixels, except that we also want to *display*
    # L5 subtypes (IT/ET/NP) while comparing them to the Neuropixels L5_Exc aggregate.
    np_cell_types = set(data.loc[data["dataset"] == "Neuropixels", "cell_type"].unique())
    l5_subtypes = {"L5_ET", "L5_IT", "L5_NP"}
    extra_compare = set()
    if "L5_Exc" in np_cell_types:
        extra_compare = l5_subtypes & set(data["cell_type"].unique())
    compare_cell_types = np_cell_types | extra_compare
    data = data[data["cell_type"].isin(compare_cell_types)].copy()

    # Prepare log-transformed rates for robustness
    def to_log_rates(x: np.ndarray) -> np.ndarray:
        return np.log10(np.asarray(x, dtype=float) + float(args.log_eps))

    # Enforce a consistent dataset order
    desired_order = ["Trained", "Untrained", "Adjusted", "Naive"]
    # Backward compatible if existing cached outputs are labeled Bio-trained
    if "Trained" not in data["dataset"].unique().tolist() and "Bio-trained" in data["dataset"].unique().tolist():
        desired_order = ["Bio-trained", "Untrained", "Adjusted", "Naive"]
    datasets = [d for d in desired_order if d in data["dataset"].unique().tolist()]
    # Compute only KS-based similarity (1 - KS)
    from scipy.stats import ks_2samp as _ks

    rows: List[Dict] = []
    ordered_cts = sorted(compare_cell_types, key=lambda c: cell_type_order().index(c) if c in cell_type_order() else 1e9)
    for ct in ordered_cts:
        np_ct = ct
        if ct in extra_compare:
            np_ct = "L5_Exc"
        np_vals = data[(data["dataset"] == "Neuropixels") & (data["cell_type"] == np_ct)]["firing_rate"].to_numpy()
        if np_vals.size == 0:
            continue
        np_log = to_log_rates(np_vals)
        for ds in datasets:
            ds_vals = data[(data["dataset"] == ds) & (data["cell_type"] == ct)]["firing_rate"].to_numpy()
            if ds_vals.size == 0:
                continue
            ds_log = to_log_rates(ds_vals)
            dist = float(_ks(np_log, ds_log).statistic)
            sim = 1.0 - dist
            rows.append({
                "cell_type": ct,
                "dataset": ds,
                "similarity": sim,
            })

    combined = pd.DataFrame(rows)

    if not combined.empty:
        mat_sim = combined.pivot(index="cell_type", columns="dataset", values="similarity")
        ordered_rows = [ct for ct in cell_type_order() if ct in mat_sim.index]
        mat_sim = mat_sim.loc[ordered_rows]

        pal = dataset_palette()
        width = 7.5 * float(args.fig_width_scale)
        height = 4.0

        style_overrides = {
            "font.family": "Arial",
            "font.size": args.fontsize,
            "axes.labelsize": args.ylabel_fontsize,
            "xtick.labelsize": args.xtick_fontsize,
            "ytick.labelsize": args.fontsize,
            "legend.fontsize": args.fontsize,
        }

        data_plot = data.rename(columns={"firing_rate": "Firing rate (Hz)"}).copy()
        present_cell_types = data_plot["cell_type"].unique().tolist()
        cell_types = [ct for ct in cell_type_order() if ct in present_cell_types]

        datasets_heatmap = dataset_order(include_naive=not args.exclude_naive, present=combined["dataset"].unique().tolist())
        # Reorder: Untrained, Trained
        datasets_heatmap = [d for d in ["Untrained", "Trained"] if d in datasets_heatmap]
        datasets_boxplot = datasets_heatmap + ["Neuropixels"]

        # Calculate and print median similarity scores
        print(f"\nMedian Similarity Scores for Firing rate (Hz):")
        for ds in datasets_heatmap:
            if ds in mat_sim.columns:
                median_sim = mat_sim[ds].median()
                print(f"  {ds}: {median_sim:.4f}")

        draw_metric_boxplot_with_similarity_heatmap(
            data_plot,
            "Firing rate (Hz)",
            mat_sim,
            datasets_boxplot=datasets_boxplot,
            datasets_heatmap=datasets_heatmap,
            palette=pal,
            cell_types=cell_types,
            out_path=args.out_combined,
            figsize=(width, height),
            boxplot_ylim=tuple(args.boxplot_ylim) if args.boxplot_ylim is not None else None,
            bottom=args.bottom,
            heatmap_xtick_fontsize=args.xtick_fontsize,
            heat_annot_fontsize=args.annot_fontsize,
            boxplot_ylabel_fontsize=args.ylabel_fontsize,
            style_overrides=style_overrides,
        )


if __name__ == "__main__":
    main()


