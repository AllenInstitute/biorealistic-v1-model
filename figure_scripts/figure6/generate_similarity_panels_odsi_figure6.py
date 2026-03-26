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
from image_decoding.plot_utils import cell_type_order, dataset_palette, draw_metric_boxplot_with_similarity_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Figure 6: drifting grating boxplots (raw metric) with similarity heatmap underneath; "
            "Syn. weight distr. constrained vs Syn. weight distr. unconstrained"
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--radius", type=float, default=200.0)
    parser.add_argument("--outdir", type=Path, default=Path("figures/paper/figure6"))
    parser.add_argument("--log_eps", type=float, default=1e-3)
    parser.add_argument("--e_only", action="store_true")
    parser.add_argument("--figsize-w", type=float, default=7.5)
    parser.add_argument("--figsize-h", type=float, default=4.0)
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

    label_from_data = "Syn. weight distr. constrained"
    label_free_change = "Syn. weight distr. unconstrained"
    heatmap_label_map = {
        label_from_data: "constrained",
        label_free_change: "unconstrained",
    }

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
        ("Rate at preferred direction (Hz)", True, "dg_firing_rate_similarity_combined.png"),
        ("OSI", False, "dg_osi_similarity_combined.png"),
        ("DSI", False, "dg_dsi_similarity_combined.png"),
    ]

    pal = dataset_palette()
    args.outdir.mkdir(parents=True, exist_ok=True)

    style_overrides = {
        "font.family": "Arial",
        "font.size": args.fontsize,
        "axes.labelsize": args.ylabel_fontsize,
        "xtick.labelsize": args.xtick_fontsize,
        "ytick.labelsize": args.fontsize,
        "legend.fontsize": args.fontsize,
    }

    for metric, do_log, fname in metrics:
        df_metric = add_l5_exc_combined(df, metric)
        mat_sim, _ks_per_type, x_order, _pal_unused = compute_similarity(
            df_metric,
            metric=metric,
            log_transform=do_log,
            log_eps=args.log_eps,
            include_naive=True,
            dataset_order_override=[label_from_data, label_free_change],
        )
        if mat_sim.empty:
            print(f"No data for metric {metric}")
            continue

        present_cell_types = df_metric["cell_type"].unique().tolist()
        cell_types = [ct for ct in cell_type_order() if ct in present_cell_types]
        datasets_heatmap = [label_from_data, label_free_change]
        datasets_boxplot = [label_from_data, label_free_change, "Neuropixels"]

        # Calculate and print median similarity scores
        print(f"\nMedian Similarity Scores for {metric}:")
        for ds in datasets_heatmap:
            if ds in mat_sim.columns:
                median_sim = mat_sim[ds].median()
                print(f"  {ds}: {median_sim:.4f}")

        draw_metric_boxplot_with_similarity_heatmap(
            df_metric,
            metric,
            mat_sim,
            datasets_boxplot=datasets_boxplot,
            datasets_heatmap=datasets_heatmap,
            palette=pal,
            cell_types=cell_types,
            out_path=args.outdir / fname,
            figsize=(float(args.figsize_w), float(args.figsize_h)),
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

    # ---------------------------------------------------------------------
    # Stimulus selectivity: drifting-gratings trial-averaged sparsity
    # ---------------------------------------------------------------------
    # Neuropixels does not have DG sparsity in the OSI/DSI CSVs, so we compute
    # similarity using "from data" as the reference distribution.
    parts: list[pd.DataFrame] = []
    missing_naive: list[str] = []
    missing_from_data: list[str] = []
    for metrics_dir in sorted((args.root.resolve()).glob("core_nll_*/metrics")):
        base_dir = metrics_dir.parent
        for variant_key, plot_label in {"bio_trained": label_from_data, "naive": label_free_change}.items():
            cache = metrics_dir / f"dg_trial_averaged_sparsity_{variant_key}.npy"
            if not cache.exists() and variant_key == "bio_trained":
                # Backwards-compat fallback
                cache = metrics_dir / "dg_trial_averaged_sparsity.npy"
            if not cache.exists():
                if variant_key == "naive":
                    missing_naive.append(str(base_dir))
                if variant_key == "bio_trained":
                    missing_from_data.append(str(base_dir))
                continue

            try:
                import network_utils as nu
                from plotting_utils import pick_core

                nodes_df = nu.load_nodes(str(base_dir), core_radius=args.radius, expand=True)
                if "node_id" not in nodes_df.columns and nodes_df.index.name == "node_id":
                    nodes_df = nodes_df.reset_index()
                if "Cell Type" in nodes_df.columns:
                    nodes_df.rename(columns={"Cell Type": "cell_type"}, inplace=True)
                nodes_df["cell_type"] = nodes_df["cell_type"].astype(str).str.replace(" ", "_", regex=False)
                nodes_df["cell_type"] = nodes_df["cell_type"].str.replace("L1_Htr3a", "L1_Inh", regex=False)
                nodes_df = pick_core(nodes_df, radius=args.radius)

                # Use the network node_id ordering to align with the cached array.
                node_file = base_dir / "network" / "v1_nodes.h5"
                if not node_file.exists():
                    continue
                import h5py

                with h5py.File(node_file, "r") as f:
                    node_ids_all = f["nodes"]["v1"]["node_id"][:].astype(np.int64)
                sparsity_all = np.load(cache)
                if len(sparsity_all) != len(node_ids_all):
                    print(
                        f"Stimulus selectivity: skipping {base_dir.name} {variant_key} due to length mismatch "
                        f"(cache={len(sparsity_all)} vs nodes={len(node_ids_all)}). Consider recomputing with "
                        f"analysis_shared/compute_dg_sparsity_for_all_networks.py"
                    )
                    continue

                sp = pd.Series(sparsity_all, index=node_ids_all, name="Stimulus selectivity")
                sp.index.name = "node_id"

                if "node_id" not in nodes_df.columns:
                    raise KeyError("node_id column missing after load_nodes")

                tmp = nodes_df[["node_id", "cell_type"]].merge(sp.reset_index(), on="node_id", how="left")
                tmp["dataset"] = plot_label
                parts.append(tmp[["dataset", "cell_type", "Stimulus selectivity"]])
            except Exception as e:
                print(f"Stimulus selectivity: failed to load {base_dir.name} {variant_key}: {e}")
                continue

    if parts:
        df_sel = pd.concat(parts, ignore_index=True)
        present_ds = set(df_sel["dataset"].unique().tolist())
        if (label_from_data not in present_ds) or (label_free_change not in present_ds):
            print(
                "Stimulus selectivity: missing one or both datasets; skipping plot. "
                f"present={sorted(present_ds)}"
            )
        else:
            # Add combined L5_Exc for display
            df_sel = add_l5_exc_combined(df_sel.rename(columns={"Stimulus selectivity": "_m"}), "_m").rename(
                columns={"_m": "Stimulus selectivity"}
            )

            mat_sim, _ks_per_type, x_order, _pal_unused = compute_similarity(
                df_sel,
                metric="Stimulus selectivity",
                log_transform=False,
                include_naive=True,
                dataset_order_override=[label_from_data, label_free_change],
                reference_dataset=label_from_data,
            )

            present_cell_types = df_sel["cell_type"].unique().tolist()
            cell_types = [ct for ct in cell_type_order() if ct in present_cell_types]
            datasets_heatmap = [label_from_data, label_free_change]
            datasets_boxplot = [label_from_data, label_free_change] # Neuropixels not available for DG sparsity

            # Calculate and print median similarity scores
            print(f"\nMedian Similarity Scores for Stimulus selectivity:")
            for ds in datasets_heatmap:
                if ds in mat_sim.columns:
                    median_sim = mat_sim[ds].median()
                    print(f"  {ds}: {median_sim:.4f}")

            draw_metric_boxplot_with_similarity_heatmap(
                df_sel,
                "Stimulus selectivity",
                mat_sim,
                datasets_boxplot=datasets_boxplot,
                datasets_heatmap=datasets_heatmap,
                palette=pal,
                cell_types=cell_types,
                out_path=args.outdir / "dg_stimulus_selectivity_similarity_combined.png",
                figsize=(float(args.figsize_w), float(args.figsize_h)),
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
            print("Saved dg_stimulus_selectivity_similarity_combined.png")
    else:
        print("No DG sparsity caches found; skipping stimulus selectivity plot")

    if missing_naive:
        missing_naive = sorted(set(missing_naive))
        print("Missing naive DG sparsity caches for:")
        for bd in missing_naive:
            print("  ", bd)
        print(
            "To compute them, run (from repo root):\n"
            "  conda run -n new_v1 python analysis_shared/compute_dg_sparsity_for_all_networks.py --base-dir core_nll_0 --network naive\n"
            "(repeat for each core_nll_*)"
        )

    if missing_from_data:
        missing_from_data = sorted(set(missing_from_data))
        print("Missing bio_trained DG sparsity caches for:")
        for bd in missing_from_data:
            print("  ", bd)
        print(
            "To compute them, run (from repo root):\n"
            "  conda run -n new_v1 python analysis_shared/compute_dg_sparsity_for_all_networks.py --base-dir core_nll_0 --network bio_trained\n"
            "(repeat for each core_nll_*)"
        )


if __name__ == "__main__":
    main()
