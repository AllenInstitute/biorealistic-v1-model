import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import colorsys
from image_decoding.plot_utils import dataset_palette, dataset_order, cell_type_order, draw_combined_similarity


def _value_col(df: pd.DataFrame) -> str:
    return "image_selectivity" if "image_selectivity" in df.columns else "lifetime_sparsity"


def add_l5_exc_combined(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    mask = df["cell_type"].isin(["L5_ET", "L5_IT", "L5_NP"]) & df[value_col].notna()
    add = df.loc[mask, ["dataset", value_col]].copy()
    add["cell_type"] = "L5_Exc"
    return pd.concat([df, add], ignore_index=True)


def cell_type_order() -> List[str]:
    return [
        # Excitatory
        "L2/3_Exc",
        "L4_Exc",
        "L5_Exc",
        "L5_ET",
        "L5_IT",
        "L5_NP",
        "L6_Exc",
        # PV
        "L2/3_PV",
        "L4_PV",
        "L5_PV",
        "L6_PV",
        # SST
        "L2/3_SST",
        "L4_SST",
        "L5_SST",
        "L6_SST",
        # VIP
        "L2/3_VIP",
        "L4_VIP",
        "L5_VIP",
        "L6_VIP",
        # L1 at the end
        "L1_Inh",
    ]


def hue_deg_to_hex(hue_deg: float, s: float = 0.75, v: float = 0.85) -> str:
    h = (hue_deg % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def main():
    parser = argparse.ArgumentParser(description="Similarity (1-KS) for image selectivity distributions; outputs combined figure")
    parser.add_argument("--np_by_unit", type=Path, default=Path("image_decoding/neuropixels/summary/sparsity_neuropixels_by_unit.csv"))
    parser.add_argument("--model_by_unit", type=Path, default=Path("image_decoding/summary/sparsity_model_by_unit.csv"))
    parser.add_argument("--exclude_naive", action="store_true")
    parser.add_argument("--out_combined", type=Path, default=Path("image_decoding/summary/selectivity_similarity_combined.png"))
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
        "bio_trained": "Bio-trained",
        "naive": "Naive",
        "plain": "Plain",
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

    # Only compare types present in Neuropixels
    np_cell_types = set(data.loc[data["dataset"] == "Neuropixels", "cell_type"].unique())
    data = data[data["cell_type"].isin(np_cell_types)].copy()

    # Dataset order
    desired_order = ["Bio-trained", "Plain", "Adjusted", "Naive"]
    datasets = [d for d in desired_order if d in data["dataset"].unique()]
    x_order = [d for d in ["Bio-trained", "Plain", "Adjusted", "Naive"] if d in datasets]

    # Palette (match firing rate)
    palette = {
        "Bio-trained": hue_deg_to_hex(135),
        "Plain": hue_deg_to_hex(45),
        "Adjusted": hue_deg_to_hex(225),
        "Naive": hue_deg_to_hex(315),
    }

    # Compute 1 - KS per cell type
    from scipy.stats import ks_2samp as _ks

    rows: List[Dict] = []
    for ct in sorted(np_cell_types, key=lambda c: cell_type_order().index(c) if c in cell_type_order() else 1e9):
        np_vals = data[(data["dataset"] == "Neuropixels") & (data["cell_type"] == ct)]["image_selectivity"].to_numpy()
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

    # Pivot to heatmap matrix and enforce ordering; draw via shared utility
    mat_sim = combined.pivot(index="cell_type", columns="dataset", values="similarity")
    ordered_rows = [ct for ct in cell_type_order() if ct in mat_sim.index]
    mat_sim = mat_sim.loc[ordered_rows]
    mat_sim = mat_sim[[d for d in x_order if d in mat_sim.columns]]
    draw_combined_similarity(mat_sim, combined, x_order, palette, args.out_combined)


if __name__ == "__main__":
    main()


