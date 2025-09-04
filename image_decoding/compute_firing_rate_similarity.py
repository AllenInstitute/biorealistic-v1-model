import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import colorsys
from image_decoding.plot_utils import dataset_palette, dataset_order, cell_type_order, draw_combined_similarity


def main():
    parser = argparse.ArgumentParser(description="Compute firing-rate distribution similarity vs Neuropixels per cell type")
    parser.add_argument("--np_cached_root", type=Path, default=Path("image_decoding/neuropixels/cached_rates"))
    parser.add_argument("--core_root", type=Path, default=Path("."))
    parser.add_argument("--networks", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--exclude_naive", action="store_true")
    parser.add_argument("--log_eps", type=float, default=1e-3, help="epsilon added before log10 to stabilize")
    parser.add_argument("--out_combined", type=Path, default=Path("image_decoding/summary/firing_rate_similarity_combined.png"))
    args = parser.parse_args()

    # Reuse loading utilities from the boxplot script to ensure consistency
    from image_decoding.plot_firing_rate_boxplot import (
        load_np_unit_rates,
        load_model_unit_rates,
        add_l5_exc_combined,
        cell_type_order,
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

    # Only compare cell types present in Neuropixels
    np_cell_types = set(data.loc[data["dataset"] == "Neuropixels", "cell_type"].unique())
    data = data[data["cell_type"].isin(np_cell_types)].copy()

    # Prepare log-transformed rates for robustness
    def to_log_rates(x: np.ndarray) -> np.ndarray:
        return np.log10(np.asarray(x, dtype=float) + float(args.log_eps))

    # Enforce a consistent dataset order
    desired_order = ["Bio-trained", "Plain", "Adjusted", "Naive"]
    datasets = [d for d in desired_order if d in data["dataset"].unique()]
    # Compute only KS-based similarity (1 - KS)
    from scipy.stats import ks_2samp as _ks

    rows: List[Dict] = []
    for ct in sorted(np_cell_types, key=lambda c: cell_type_order().index(c) if c in cell_type_order() else 1e9):
        np_vals = data[(data["dataset"] == "Neuropixels") & (data["cell_type"] == ct)]["firing_rate"].to_numpy()
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
        ks_per_type = combined.copy()
        x_order = dataset_order(include_naive=False, present=ks_per_type["dataset"].unique().tolist())
        mat_sim = combined.pivot(index="cell_type", columns="dataset", values="similarity")
        ordered_rows = [ct for ct in cell_type_order() if ct in mat_sim.index]
        mat_sim = mat_sim.loc[ordered_rows]
        mat_sim = mat_sim[[d for d in x_order if d in mat_sim.columns]]
        pal = dataset_palette()
        draw_combined_similarity(mat_sim, ks_per_type, x_order, pal, args.out_combined)


if __name__ == "__main__":
    main()


