import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_colors() -> Dict[str, str]:
    df = pd.read_csv("base_props/cell_type_naming_scheme.csv", sep=r"\s+")
    return dict(zip(df["cell_type"], df["hex"]))


def cell_type_order() -> List[str]:
    return [
        "L2/3_Exc",
        "L4_Exc",
        "L5_Exc",
        "L5_ET",
        "L5_IT",
        "L5_NP",
        "L6_Exc",
        "L2/3_PV",
        "L4_PV",
        "L5_PV",
        "L6_PV",
        "L2/3_SST",
        "L4_SST",
        "L5_SST",
        "L6_SST",
        "L2/3_VIP",
        "L4_VIP",
        "L5_VIP",
        "L6_VIP",
        "L1_Inh",
    ]


def _value_col(df: pd.DataFrame) -> str:
    return "image_selectivity" if "image_selectivity" in df.columns else "lifetime_sparsity"


def get_cell_subtype(cell_type: str) -> str:
    if cell_type == "L1_Inh":
        return "Inh"
    if "Exc" in cell_type or cell_type in {"L5_ET", "L5_IT", "L5_NP"}:
        return "Exc"
    if "PV" in cell_type:
        return "PV"
    if "SST" in cell_type:
        return "SST"
    if "VIP" in cell_type:
        return "VIP"
    return "Other"


def get_subtype_colors_from_scheme(colors_df: pd.DataFrame) -> Dict[str, tuple]:
    # Build faint background colors by averaging class colors
    def mean_rgba(hex_list: List[str]) -> tuple:
        if len(hex_list) == 0:
            return (0.9, 0.9, 0.9, 0.12)
        import matplotlib.colors as mcolors

        rgbs = [mcolors.to_rgba(h, alpha=0.12) for h in hex_list]
        arr = np.array([c[:3] for c in rgbs])
        mean_rgb = arr.mean(axis=0)
        return (mean_rgb[0], mean_rgb[1], mean_rgb[2], 0.12)

    subtype_colors: Dict[str, tuple] = {}
    for subtype, cls in [("Exc", "Exc"), ("PV", "PV"), ("SST", "SST"), ("VIP", "VIP"), ("Inh", "Inh")]:
        hexes = colors_df[colors_df["class"] == cls]["hex"].tolist()
        subtype_colors[subtype] = mean_rgba(hexes)
    return subtype_colors


def add_background_shading(ax, cell_types: List[str], subtype_colors: Dict[str, tuple]):
    current = None
    start = 0
    for i, ct in enumerate(cell_types + [None]):
        subtype = get_cell_subtype(ct) if ct is not None else None
        if subtype != current:
            if current is not None:
                end = i - 0.5
                color = subtype_colors.get(current, (0.92, 0.92, 0.92, 0.12))
                ax.axvspan(start - 0.5, end, facecolor=color, zorder=0)
            current = subtype
            start = i


def prepare_long_table(np_path: Path, model_path: Path) -> pd.DataFrame:
    df_np = pd.read_csv(np_path)
    df_model = pd.read_csv(model_path)

    # Standardize value column
    df_np = df_np.rename(columns={_value_col(df_np): "image_selectivity"})
    df_model = df_model.rename(columns={_value_col(df_model): "image_selectivity"})

    # Map names and datasets
    df_np["cell_type"] = df_np["cell_type"].replace({"L1_Htr3a": "L1_Inh"})
    df_np["dataset"] = "Neuropixels"
    df_model = df_model[df_model.get("network_type").isin(["bio_trained"])].copy()
    df_model["dataset"] = "Bio-trained"

    keep_cols = ["dataset", "cell_type", "image_selectivity"]
    df_np = df_np[keep_cols].copy()
    df_model = df_model[keep_cols].copy()

    # Add combined L5_Exc alongside subtypes for both datasets
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
    parser = argparse.ArgumentParser(description="Violin plots of image selectivity (Neuropixels vs Bio-trained)")
    parser.add_argument("--np_by_unit", type=Path, default=Path("image_decoding/neuropixels/summary/sparsity_neuropixels_by_unit.csv"))
    parser.add_argument("--model_by_unit", type=Path, default=Path("image_decoding/summary/sparsity_model_by_unit.csv"))
    parser.add_argument("--out", type=Path, default=Path("image_decoding/summary/selectivity_violin.png"))
    args = parser.parse_args()

    data = prepare_long_table(args.np_by_unit, args.model_by_unit)
    order = [ct for ct in cell_type_order() if ct in data["cell_type"].unique()]

    # Hue colors (match other figures)
    hue_order = ["Neuropixels", "Bio-trained"]
    hue_palette = {
        "Neuropixels": "#7f7f7f",  # gray
        "Bio-trained": "#2ca02c",  # green
    }

    plt.figure(figsize=(8.0, 3.5))
    ax = plt.gca()
    sns.despine(ax=ax)

    # Background shading by subtype blocks
    colors_df = pd.read_csv("base_props/cell_type_naming_scheme.csv", sep=r"\s+")
    subtype_bg = get_subtype_colors_from_scheme(colors_df)
    add_background_shading(ax, order, subtype_bg)

    sns.violinplot(
        data=data,
        x="cell_type",
        y="image_selectivity",
        order=order,
        hue="dataset",
        hue_order=hue_order,
        palette=hue_palette,
        split=True,
        cut=0,
        inner="quartile",
        linewidth=0.6,
        # Make appearance robust across environments
        scale="width",
        scale_hue=False,
        width=0.8,
    )

    # Axis styling
    ax.set_ylabel("Image selectivity")
    ax.set_xlabel("")
    try:
        y_top = float(np.nanpercentile(data["image_selectivity"], 95)) + 0.10
    except Exception:
        y_top = 0.3
    ax.set_ylim(bottom=0, top=min(0.4, max(0.12, y_top)))
    plt.xticks(rotation=90, ha="right")
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.17))

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


