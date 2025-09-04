import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from image_decoding.plot_utils import (
    dataset_palette,
    dataset_order,
    cell_type_order,
    load_cell_type_colors,
    get_subtype_colors_from_scheme,
    add_background_shading,
    set_horizontal_legend,
)


def load_colors() -> Dict[str, str]:
    df = pd.read_csv("base_props/cell_type_naming_scheme.csv", sep=r"\s+")
    return dict(zip(df["cell_type"], df["hex"]))


def _value_col(df: pd.DataFrame) -> str:
    return "image_selectivity" if "image_selectivity" in df.columns else "lifetime_sparsity"


def prepare_long_table(np_path: Path, model_path: Path) -> pd.DataFrame:
    df_np = pd.read_csv(np_path)
    df_model = pd.read_csv(model_path)

    # Standardize columns
    val_np = _value_col(df_np)
    val_model = _value_col(df_model)
    df_np = df_np.rename(columns={val_np: "image_selectivity"})
    df_model = df_model.rename(columns={val_model: "image_selectivity"})

    # Map names
    df_np["cell_type"] = df_np["cell_type"].replace({"L1_Htr3a": "L1_Inh"})
    df_model = df_model[df_model["network_type"].isin(["bio_trained", "naive", "plain", "adjusted"])].copy()

    # Dataset label
    df_np["dataset"] = "Neuropixels"
    df_model["dataset"] = df_model["network_type"].replace({
        "bio_trained": "Bio-trained",
        "naive": "Naive",
        "plain": "Plain",
        "adjusted": "Adjusted",
    })

    # Keep only necessary columns
    keep_cols = ["dataset", "cell_type", "image_selectivity"]
    df_np = df_np[keep_cols].copy()
    df_model = df_model[keep_cols + ["network_type"]].copy()

    # Add combined L5_Exc for all datasets (duplicate rows with remapped cell_type)
    def add_l5_exc_combined(df: pd.DataFrame) -> pd.DataFrame:
        mask = df["cell_type"].isin(["L5_ET", "L5_IT", "L5_NP"]) & df["image_selectivity"].notna()
        add = df.loc[mask, ["dataset", "image_selectivity"]].copy()
        add["cell_type"] = "L5_Exc"
        return pd.concat([df, add], ignore_index=True)

    df_np = add_l5_exc_combined(df_np)
    df_model = add_l5_exc_combined(df_model)

    all_df = pd.concat([df_np, df_model], ignore_index=True)
    return all_df.dropna(subset=["cell_type", "image_selectivity"]) 


def main():
    parser = argparse.ArgumentParser(description="Box plots of image selectivity per cell type and dataset")
    # Defaults use legacy filenames for backward compatibility
    parser.add_argument("--np_by_unit", type=Path, default=Path("image_decoding/neuropixels/summary/sparsity_neuropixels_by_unit.csv"))
    parser.add_argument("--model_by_unit", type=Path, default=Path("image_decoding/summary/sparsity_model_by_unit.csv"))
    parser.add_argument("--out", type=Path, default=Path("image_decoding/summary/selectivity_boxplot.png"))
    parser.add_argument("--exclude_naive", action="store_true", help="Exclude Naive dataset from the plot")
    args = parser.parse_args()

    data = prepare_long_table(args.np_by_unit, args.model_by_unit)
    order = [ct for ct in cell_type_order() if ct in data["cell_type"].unique()]

    # Palette and hue order
    pal = dataset_palette()
    desired = ["Neuropixels", "Bio-trained", "Naive", "Plain", "Adjusted"]
    if args.exclude_naive:
        desired = [d for d in desired if d != "Naive"]
    hue_order = [d for d in desired if d in data["dataset"].unique().tolist()]
    hue_palette = {k: pal[k] for k in hue_order}

    plt.figure(figsize=(8.0, 3.5))
    ax = plt.gca()

    # Background shading by subtype blocks
    colors_df = load_cell_type_colors()
    subtype_bg = get_subtype_colors_from_scheme(colors_df)
    add_background_shading(ax, order, subtype_bg)

    sns.boxplot(
        data=data,
        x="cell_type",
        y="image_selectivity",
        order=order,
        hue="dataset",
        hue_order=hue_order,
        palette=hue_palette,
        showcaps=True,
        fliersize=1.5,
        width=0.7,
        boxprops={"edgecolor": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )

    ax.set_ylabel("Image selectivity")
    ax.set_xlabel("")
    ax.set_yscale("log")
    ax.set_ylim(bottom=5e-5, top=1.05)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=90, ha="right")
    sns.despine(ax=ax)
    set_horizontal_legend(ax)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    try:
        plt.savefig(args.out.with_suffix(".svg"))
    except Exception:
        pass
    plt.close()


if __name__ == "__main__":
    main()

