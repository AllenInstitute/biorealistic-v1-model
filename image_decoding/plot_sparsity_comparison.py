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
        "L1_Inh",
        "L2/3_Exc",
        "L4_Exc",
        "L5_Exc",  # show both combined and subtypes
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
    ]


def map_np_cell_type(ct: str) -> str:
    if ct == "L1_Htr3a":
        return "L1_Inh"
    return ct


def map_model_cell_type(ct: str) -> str:
    # Keep model L5 excitatory subtypes separate
    return ct


def _value_col(df: pd.DataFrame) -> str:
    return "image_selectivity" if "image_selectivity" in df.columns else "lifetime_sparsity"


def collapse_l5_exc(df_units: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    val_col = _value_col(df_units)
    mask = df_units["cell_type"].isin(["L5_ET", "L5_IT", "L5_NP"]) & df_units[val_col].notna()
    sub = df_units.loc[mask, val_col].to_numpy()
    if sub.size == 0:
        return pd.DataFrame(columns=["dataset", "cell_type", "mean", "std", "n_units", "sem"])  # empty
    mean = float(np.mean(sub))
    std = float(np.std(sub, ddof=1)) if sub.size > 1 else 0.0
    n = int(sub.size)
    sem = std / np.sqrt(max(n, 1))
    return pd.DataFrame(
        [{
            "dataset": dataset_name,
            "cell_type": "L5_Exc",
            "mean": mean,
            "std": std,
            "n_units": n,
            "sem": sem,
        }]
    )


def aggregate_by_unit(df_units: pd.DataFrame, dataset_name: str, type_mapper) -> pd.DataFrame:
    d = df_units.copy()
    val_col = _value_col(d)
    d["cell_type"] = d["cell_type"].map(type_mapper)
    d = d.dropna(subset=["cell_type", val_col])  # ensure valid
    d["dataset"] = dataset_name
    grouped = d.groupby(["dataset", "cell_type"])  # type: ignore[list-item]
    out = (
        grouped[val_col]
        .agg([("mean", "mean"), ("std", "std"), ("n_units", "count")])
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["n_units"].clip(lower=1))
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot lifetime sparsity comparison: Neuropixels vs model")
    parser.add_argument(
        "--np_by_unit",
        type=Path,
        default=Path("image_decoding/neuropixels/summary/sparsity_neuropixels_by_unit.csv"),
    )
    parser.add_argument(
        "--model_by_unit",
        type=Path,
        default=Path("image_decoding/summary/sparsity_model_by_unit.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("image_decoding/summary/sparsity_neuropixels_vs_model.png"),
    )
    args = parser.parse_args()

    # Load inputs
    df_np = pd.read_csv(args.np_by_unit)
    df_model = pd.read_csv(args.model_by_unit)

    # Restrict model to two datasets and remap
    df_model = df_model[df_model["network_type"].isin(["bio_trained", "naive"])].copy()
    df_model["cell_type"] = df_model["cell_type"].map(map_model_cell_type)

    # Build aggregated frames
    agg_np = aggregate_by_unit(df_np, "Neuropixels", map_np_cell_type)
    # add Neuropixels combined L5_Exc to reflect absence of subtypes
    agg_np = pd.concat([agg_np, collapse_l5_exc(df_np.rename(columns={"lifetime_sparsity": "lifetime_sparsity"}), "Neuropixels")], ignore_index=True)
    agg_bio = aggregate_by_unit(df_model[df_model["network_type"] == "bio_trained"], "Bio-trained", lambda x: x)
    agg_naive = aggregate_by_unit(df_model[df_model["network_type"] == "naive"], "Naive", lambda x: x)
    # For model datasets, include both subtypes and combined L5_Exc
    comb_bio = collapse_l5_exc(df_model[df_model["network_type"] == "bio_trained"].rename(columns={"lifetime_sparsity": "lifetime_sparsity"}), "Bio-trained")
    comb_naive = collapse_l5_exc(df_model[df_model["network_type"] == "naive"].rename(columns={"lifetime_sparsity": "lifetime_sparsity"}), "Naive")
    df_plot = pd.concat([agg_np, agg_bio, agg_naive, comb_bio, comb_naive], ignore_index=True)

    # Order and colors
    order = [ct for ct in cell_type_order() if ct in df_plot["cell_type"].unique()]
    # Background subtype shading
    colors_df = pd.read_csv("base_props/cell_type_naming_scheme.csv", sep=r"\s+")
    def mean_rgba(hex_list):
        if len(hex_list) == 0:
            return (0.9, 0.9, 0.9, 0.12)
        import matplotlib.colors as mcolors
        rgbs = [mcolors.to_rgba(h, alpha=0.12) for h in hex_list]
        arr = np.array([c[:3] for c in rgbs])
        m = arr.mean(axis=0)
        return (m[0], m[1], m[2], 0.12)
    subtype_bg = {
        'Exc': mean_rgba(colors_df[colors_df['class']=='Exc']['hex'].tolist()),
        'PV': mean_rgba(colors_df[colors_df['class']=='PV']['hex'].tolist()),
        'SST': mean_rgba(colors_df[colors_df['class']=='SST']['hex'].tolist()),
        'VIP': mean_rgba(colors_df[colors_df['class']=='VIP']['hex'].tolist()),
        'Inh': mean_rgba(colors_df[colors_df['class']=='Inh']['hex'].tolist()),
    }
    def subtype(ct):
        if ct == 'L1_Inh':
            return 'Inh'
        if ct in {'L5_ET','L5_IT','L5_NP'} or 'Exc' in ct:
            return 'Exc'
        for k in ['PV','SST','VIP']:
            if k in ct:
                return k
        return 'Other'

    # Plot settings
    plt.figure(figsize=(7.5, 3.5))
    ax = plt.gca()
    sns.despine(ax=ax)

    # Background shading by subtype blocks
    cur=None; start=0
    for i, ct in enumerate(order + [None]):
        st = subtype(ct) if ct is not None else None
        if st != cur:
            if cur is not None:
                ax.axvspan(start-0.5, i-0.5, facecolor=subtype_bg.get(cur,(0.92,0.92,0.92,0.12)), zorder=0)
            cur=st; start=i

    # Build grouped bars manually with dataset-colored bars
    datasets = ["Neuropixels", "Bio-trained", "Naive"]
    # No hatch patterns; color alone distinguishes datasets
    hatch_map = {"Neuropixels": "", "Bio-trained": "", "Naive": ""}
    hue_palette = {"Neuropixels": "#7f7f7f", "Bio-trained": "#2ca02c", "Naive": "#d62728"}
    offset = np.linspace(-0.25, 0.25, num=len(datasets))
    width = 0.25

    # Prepare data for quick lookup
    key = df_plot.set_index(["dataset", "cell_type"]).to_dict(orient="index")

    x = np.arange(len(order))
    for di, dataset in enumerate(datasets):
        means = []
        sems = []
        for ct in order:
            rec = key.get((dataset, ct))
            if rec is None:
                means.append(np.nan)
                sems.append(0.0)
            else:
                means.append(rec["mean"])  # type: ignore[index]
                sems.append(rec["sem"])   # type: ignore[index]
        bars = ax.bar(
            x + offset[di],
            means,
            width=width,
            color=hue_palette[dataset],
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch_map[dataset],
            label=dataset,
        )
        # Error bars
        ax.errorbar(
            x + offset[di],
            means,
            yerr=sems,
            fmt="none",
            ecolor="black",
            elinewidth=0.6,
            capsize=2,
            capthick=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, ha="right")
    ax.set_ylabel("Image selectivity")
    # Dynamic ceiling: max(mean + sem) with small headroom, capped to 0.4
    try:
        y_top = float(np.nanmax(df_plot["mean"] + df_plot["sem"])) + 0.02
    except Exception:
        y_top = 0.3
    ax.set_ylim(bottom=0, top=min(0.4, max(0.12, y_top)))
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.15))
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


