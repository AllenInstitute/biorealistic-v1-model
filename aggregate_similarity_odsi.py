import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from image_decoding.plot_utils import (
    dataset_palette,
    dataset_order,
    cell_type_order as id_cell_type_order,
    draw_combined_similarity,
)

# Reuse the aggregator used by the boxplot script
from aggregate_boxplots_odsi import discover_and_aggregate


def add_l5_exc_combined(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    mask = df["cell_type"].isin(["L5_ET", "L5_IT", "L5_NP"]) & df[metric].notna()
    if not mask.any():
        return df
    add = df.loc[mask, ["dataset", metric]].copy()
    add["cell_type"] = "L5_Exc"
    return pd.concat([df, add], ignore_index=True)


def compute_similarity(
    df: pd.DataFrame,
    metric: str,
    log_transform: bool = False,
    log_eps: float = 1e-3,
    include_naive: bool = False,
    dataset_order_override: Optional[List[str]] = None,
    reference_dataset: str = "Neuropixels",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, str]]:
    # Filter relevant columns and rows
    df = df.dropna(subset=["dataset", "cell_type", metric]).copy()

    # Only compare types present in Neuropixels, except that we also want to *display*
    # L5 subtypes (IT/ET/NP) while comparing them to the Neuropixels L5_Exc aggregate.
    ref_cell_types = set(df.loc[df["dataset"] == reference_dataset, "cell_type"].unique())
    l5_subtypes = {"L5_ET", "L5_IT", "L5_NP"}
    extra_compare = set()
    if "L5_Exc" in ref_cell_types:
        present_types = set(df["cell_type"].unique())
        extra_compare = l5_subtypes & present_types

    compare_cell_types = ref_cell_types | extra_compare
    df = df[df["cell_type"].isin(compare_cell_types)].copy()

    # Dataset order and palette
    present = df["dataset"].unique().tolist()
    if dataset_order_override is not None:
        x_order = [d for d in dataset_order_override if d in present and d != "Neuropixels"]
    else:
        x_order = dataset_order(include_naive=include_naive, present=present)
    pal = dataset_palette()

    # Helper to transform values
    def transform(vals: np.ndarray) -> np.ndarray:
        if not log_transform:
            return np.asarray(vals, dtype=float)
        return np.log10(np.asarray(vals, dtype=float) + float(log_eps))

    from scipy.stats import ks_2samp as _ks

    rows: List[Dict] = []
    # Stable cell type ordering
    ct_order = id_cell_type_order()
    ordered_cts = sorted(compare_cell_types, key=lambda c: ct_order.index(c) if c in ct_order else 1e9)

    for ct in ordered_cts:
        np_ct = ct
        if ct in extra_compare:
            np_ct = "L5_Exc"

        ref_vals = df[(df["dataset"] == reference_dataset) & (df["cell_type"] == np_ct)][metric].to_numpy()
        if ref_vals.size == 0:
            continue
        ref_tx = transform(ref_vals)
        for ds in x_order:
            if ds == reference_dataset:
                continue
            ds_vals = df[(df["dataset"] == ds) & (df["cell_type"] == ct)][metric].to_numpy()
            if ds_vals.size == 0:
                continue
            ds_tx = transform(ds_vals)
            dist = float(_ks(ref_tx, ds_tx).statistic)
            sim = 1.0 - dist
            rows.append({"cell_type": ct, "dataset": ds, "similarity": sim})

        if reference_dataset in x_order:
            rows.append({"cell_type": ct, "dataset": reference_dataset, "similarity": 1.0})

    combined = pd.DataFrame(rows)
    if combined.empty:
        return combined, combined, x_order, pal

    # Pivot to heatmap matrix and enforce ordering
    mat_sim = combined.pivot(index="cell_type", columns="dataset", values="similarity")
    row_order = [ct for ct in ct_order if ct in mat_sim.index]
    mat_sim = mat_sim.loc[row_order]
    mat_sim = mat_sim[[d for d in x_order if d in mat_sim.columns]]

    return mat_sim, combined, x_order, pal


def main():
    parser = argparse.ArgumentParser(description="Compute distribution similarity (1-KS) vs Neuropixels for Rate/OSI/DSI")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--radius", type=float, default=200.0)
    parser.add_argument("--outdir", type=Path, default=Path("figures/paper"))
    parser.add_argument("--log_eps", type=float, default=1e-3)
    parser.add_argument("--e_only", action="store_true")
    args = parser.parse_args()

    # Load aggregated data (Neuropixels + Bio-trained + Untrained)
    df = discover_and_aggregate(args.root.resolve(), core_radius=args.radius, include_variants={"bio_trained": "Bio-trained", "plain": "Untrained"})
    if df.empty:
        print("No aggregated data found.")
        return

    # Optionally restrict to excitatory
    if args.e_only and "ei" in df.columns:
        df = df[df["ei"] == "e"]

    # Metrics: firing rate (pref dir), OSI, DSI
    metrics = [
        ("Rate at preferred direction (Hz)", True, "firing_rate_similarity_combined.png"),
        ("OSI", False, "osi_similarity_combined.png"),
        ("DSI", False, "dsi_similarity_combined.png"),
    ]

    args.outdir.mkdir(parents=True, exist_ok=True)
    for metric, do_log, fname in metrics:
        # Add combined L5_Exc before computing similarity
        df_metric = add_l5_exc_combined(df, metric)
        mat_sim, ks_per_type, x_order, pal = compute_similarity(
            df_metric,
            metric=metric,
            log_transform=do_log,
            log_eps=args.log_eps,
            include_naive=False,
        )
        if mat_sim.empty:
            print(f"No data for metric {metric}")
            continue
        # Shrink width by ~15%
        width = 3.6 * 0.85
        draw_combined_similarity(mat_sim, ks_per_type, x_order, pal, args.outdir / fname, figsize=(width, 6.0))
        print(f"Saved {fname}")


if __name__ == "__main__":
    main()


