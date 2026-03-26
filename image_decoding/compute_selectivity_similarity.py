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


def _value_col(df: pd.DataFrame) -> str:
    return "image_selectivity" if "image_selectivity" in df.columns else "lifetime_sparsity"


def add_l5_exc_combined(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    mask = df["cell_type"].isin(["L5_ET", "L5_IT", "L5_NP"]) & df[value_col].notna()
    add = df.loc[mask, ["dataset", value_col]].copy()
    add["cell_type"] = "L5_Exc"
    return pd.concat([df, add], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Similarity (1-KS) for image selectivity distributions; outputs combined figure")
    parser.add_argument("--np_by_unit", type=Path, default=Path("image_decoding/neuropixels/summary/sparsity_neuropixels_by_unit.csv"))
    parser.add_argument("--model_by_unit", type=Path, default=Path("image_decoding/summary/sparsity_model_by_unit.csv"))
    parser.add_argument("--exclude_naive", action="store_true")
    parser.add_argument("--exclude_adjusted", action="store_true")
    parser.add_argument("--out_combined", type=Path, default=Path("image_decoding/summary/selectivity_similarity_combined.png"))
    parser.add_argument("--fig_width_scale", type=float, default=1.0, help="scale factor for figure width (default 1.0)")
    parser.add_argument("--fontsize", type=float, default=11)
    parser.add_argument("--ylabel-fontsize", type=float, default=11.5)
    parser.add_argument("--xtick-fontsize", type=float, default=12)
    parser.add_argument("--annot-fontsize", type=float, default=9)
    parser.add_argument("--bottom", type=float, default=0.25)
    args = parser.parse_args()

    # Load tables
    df_np = pd.read_csv(args.np_by_unit)
    df_model = pd.read_csv(args.model_by_unit)

    # Standardize value column
    val_np = _value_col(df_np)
    val_model = _value_col(df_model)
    df_np = df_np.rename(columns={val_np: "image_selectivity"})
    df_model = df_model.rename(columns={val_model: "image_selectivity"})

    # Filter model types of interest
    df_model = df_model[df_model["network_type"].isin(["bio_trained", "naive", "plain", "adjusted"])].copy()

    # Dataset labels
    df_np["dataset"] = "Neuropixels"
    df_model["dataset"] = df_model["network_type"].replace({
        "bio_trained": "Trained",
        "naive": "Naive",
        "plain": "Untrained",
        "adjusted": "Adjusted",
    })

    # Map cell type variants
    df_np["cell_type"] = df_np["cell_type"].replace({"L1_Htr3a": "L1_Inh"})

    # Keep only needed columns
    keep_cols = ["dataset", "cell_type", "image_selectivity"]
    df_np = df_np[keep_cols].copy()
    df_model = df_model[keep_cols + ["network_type"]].copy()

    # Add combined L5_Exc
    df_np = add_l5_exc_combined(df_np, "image_selectivity")
    df_model = add_l5_exc_combined(df_model, "image_selectivity")

    data = pd.concat([df_np, df_model], ignore_index=True)
    data = data.dropna(subset=["dataset", "cell_type", "image_selectivity"]).copy()
    if args.exclude_naive:
        data = data[data["dataset"] != "Naive"]
    if args.exclude_adjusted:
        data = data[data["dataset"] != "Adjusted"]

    # Only compare types present in Neuropixels
    np_cell_types = set(data.loc[data["dataset"] == "Neuropixels", "cell_type"].unique())
    l5_subtypes = {"L5_ET", "L5_IT", "L5_NP"}
    extra_compare = set()
    if "L5_Exc" in np_cell_types:
        extra_compare = l5_subtypes & set(data["cell_type"].unique())
    compare_cell_types = np_cell_types | extra_compare
    data = data[data["cell_type"].isin(compare_cell_types)].copy()

    present_ds = data["dataset"].unique().tolist()
    x_order = dataset_order(include_naive=not args.exclude_naive, present=present_ds)
    pal = dataset_palette()
    datasets = [d for d in x_order if d != "Neuropixels"]

    # Compute 1 - KS per cell type
    from scipy.stats import ks_2samp as _ks

    rows: List[Dict] = []
    ordered_cts = sorted(compare_cell_types, key=lambda c: cell_type_order().index(c) if c in cell_type_order() else 1e9)
    for ct in ordered_cts:
        np_ct = ct
        if ct in extra_compare:
            np_ct = "L5_Exc"
        np_vals = data[(data["dataset"] == "Neuropixels") & (data["cell_type"] == np_ct)]["image_selectivity"].to_numpy()
        if np_vals.size == 0:
            continue
        for ds in datasets:
            ds_vals = data[(data["dataset"] == ds) & (data["cell_type"] == ct)]["image_selectivity"].to_numpy()
            if ds_vals.size == 0:
                continue
            dist = float(_ks(np_vals, ds_vals).statistic)
            sim = 1.0 - dist
            rows.append({"cell_type": ct, "dataset": ds, "similarity": sim})

    combined = pd.DataFrame(rows)
    if combined.empty:
        print("No data to plot.")
        return

    # Pivot to heatmap matrix and enforce ordering; draw via combined boxplot+heatmap style
    mat_sim = combined.pivot(index="cell_type", columns="dataset", values="similarity")
    ordered_rows = [ct for ct in cell_type_order() if ct in mat_sim.index]
    mat_sim = mat_sim.loc[ordered_rows]

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

    data_plot = data.rename(columns={"image_selectivity": "Image selectivity"}).copy()
    present_cell_types = data_plot["cell_type"].unique().tolist()
    cell_types = [ct for ct in cell_type_order() if ct in present_cell_types]

    datasets_heatmap = [d for d in x_order if d in mat_sim.columns]
    datasets_heatmap = [d for d in ["Untrained", "Trained"] if d in datasets_heatmap]
    datasets_boxplot = datasets_heatmap + ["Neuropixels"]

    # Calculate and print median similarity scores
    print(f"\nMedian Similarity Scores for Image selectivity:")
    for ds in datasets_heatmap:
        if ds in mat_sim.columns:
            median_sim = mat_sim[ds].median()
            print(f"  {ds}: {median_sim:.4f}")

    draw_metric_boxplot_with_similarity_heatmap(
        data_plot,
        "Image selectivity",
        mat_sim,
        datasets_boxplot=datasets_boxplot,
        datasets_heatmap=datasets_heatmap,
        palette=pal,
        cell_types=cell_types,
        out_path=args.out_combined,
        figsize=(width, height),
        boxplot_yscale="log",
        boxplot_ylim=(5e-5, 1.05),
        bottom=args.bottom,
        heatmap_xtick_fontsize=args.xtick_fontsize,
        heat_annot_fontsize=args.annot_fontsize,
        boxplot_ylabel_fontsize=args.ylabel_fontsize,
        style_overrides=style_overrides,
    )


if __name__ == "__main__":
    main()


