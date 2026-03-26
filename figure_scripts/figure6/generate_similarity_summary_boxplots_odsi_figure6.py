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

from aggregate_boxplots_odsi import discover_and_aggregate
from aggregate_similarity_odsi import add_l5_exc_combined, compute_similarity
from image_decoding.plot_utils import dataset_palette, draw_similarity_summary_boxplot_multi_metric


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Figure 6: combined summary similarity-score boxplots (metric on x-axis) "
            "for drifting gratings; Syn. weight distr. constrained vs Syn. weight distr. unconstrained"
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--radius", type=float, default=200.0)
    parser.add_argument("--outdir", type=Path, default=Path("figures/paper/figure6"))
    parser.add_argument("--log_eps", type=float, default=1e-3)
    parser.add_argument("--e_only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figsize-w", type=float, default=5.2)
    parser.add_argument("--figsize-h", type=float, default=2.1)
    args = parser.parse_args()

    label_from_data = "Syn. weight distr. constrained"
    label_free_change = "Syn. weight distr. unconstrained"

    df = discover_and_aggregate(
        args.root.resolve(),
        core_radius=args.radius,
        include_variants={"bio_trained": label_from_data, "naive": label_free_change},
    )
    if df.empty:
        print("No aggregated data found.")
        return

    if args.e_only and "ei" in df.columns:
        df = df[df["ei"] == "e"]

    metrics = [
        ("FR", "Rate at preferred direction (Hz)", True),
        ("OSI", "OSI", False),
        ("DSI", "DSI", False),
        ("Stimulus selectivity", "Stimulus selectivity", False),
    ]

    pal = dataset_palette()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.output if args.output is not None else (args.outdir / "dg_similarity_summary_metrics.png")

    rows = []
    for metric_label, metric_col, do_log in metrics:
        if metric_col == "Stimulus selectivity":
            # Load DG trial-averaged sparsity from per-network caches (no Neuropixels equivalent)
            parts = []
            for metrics_dir in sorted((args.root.resolve()).glob("core_nll_*/metrics")):
                base_dir = metrics_dir.parent
                for variant_key, plot_label in {"bio_trained": label_from_data, "naive": label_free_change}.items():
                    cache = metrics_dir / f"dg_trial_averaged_sparsity_{variant_key}.npy"
                    if not cache.exists() and variant_key == "bio_trained":
                        cache = metrics_dir / "dg_trial_averaged_sparsity.npy"
                    if not cache.exists():
                        continue

                    try:
                        import h5py
                        import network_utils as nu
                        from plotting_utils import pick_core

                        nodes_df = nu.load_nodes(str(base_dir), core_radius=args.radius, expand=True)
                        if "Cell Type" in nodes_df.columns:
                            nodes_df.rename(columns={"Cell Type": "cell_type"}, inplace=True)
                        nodes_df["cell_type"] = nodes_df["cell_type"].astype(str).str.replace(" ", "_", regex=False)
                        nodes_df["cell_type"] = nodes_df["cell_type"].str.replace("L1_Htr3a", "L1_Inh", regex=False)
                        nodes_df = pick_core(nodes_df, radius=args.radius)

                        node_file = base_dir / "network" / "v1_nodes.h5"
                        if not node_file.exists():
                            continue
                        with h5py.File(node_file, "r") as f:
                            node_ids_all = f["nodes"]["v1"]["node_id"][:].astype(np.int64)
                        sparsity_all = np.load(cache)
                        if len(sparsity_all) != len(node_ids_all):
                            continue

                        sp = pd.Series(sparsity_all, index=node_ids_all, name="Stimulus selectivity")
                        sp.index.name = "node_id"
                        tmp = nodes_df[["node_id", "cell_type"]].merge(
                            sp.reset_index(), on="node_id", how="left"
                        )
                        tmp["dataset"] = plot_label
                        parts.append(tmp[["dataset", "cell_type", "Stimulus selectivity"]])
                    except Exception:
                        continue

            if not parts:
                continue
            df_metric = pd.concat(parts, ignore_index=True)
            df_metric = add_l5_exc_combined(
                df_metric.rename(columns={"Stimulus selectivity": "_m"}), "_m"
            ).rename(columns={"_m": "Stimulus selectivity"})
            _mat_sim, ks_per_type, _x_order, _pal_unused = compute_similarity(
                df_metric,
                metric=metric_col,
                log_transform=do_log,
                log_eps=args.log_eps,
                include_naive=True,
                dataset_order_override=[label_from_data, label_free_change],
                reference_dataset=label_from_data,
            )
        else:
            df_metric = add_l5_exc_combined(df, metric_col)
            _mat_sim, ks_per_type, _x_order, _pal_unused = compute_similarity(
                df_metric,
                metric=metric_col,
                log_transform=do_log,
                log_eps=args.log_eps,
                include_naive=True,
                dataset_order_override=[label_from_data, label_free_change],
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
    draw_similarity_summary_boxplot_multi_metric(
        combined,
        metric_order=[m[0] for m in metrics],
        dataset_order=[label_from_data, label_free_change],
        palette=pal,
        out_path=out_path,
        figsize=(float(args.figsize_w), float(args.figsize_h)),
    )
    print(f"Saved {out_path.name}")


if __name__ == "__main__":
    main()
