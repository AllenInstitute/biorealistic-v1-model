#!/usr/bin/env python
"""
Generate publication-ready synaptic weight distributions for targeted cell types.
Outputs default to `figures/paper/figure6/` (Figure 6 panels).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import network_utils as nu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot synaptic weight distributions for specific cell types."
    )
    parser.add_argument(
        "--network-dir",
        type=Path,
        default=Path("core_nll_0"),
        help="Directory that contains the `network/` folder.",
    )
    parser.add_argument(
        "--cell-types",
        nargs="+",
        default=["L2/3_Exc"],
        help=(
            "Cell types or pop_name entries to include (e.g., L2/3_Exc or e23Cux2). "
            "Multiple entries are allowed."
        ),
    )
    parser.add_argument(
        "--direction",
        choices=["incoming", "outgoing"],
        default="outgoing",
        help="Whether to plot incoming or outgoing weights.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/paper/figure6"),
        help="Directory to save the figures.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Apply logarithmic scaling to the x-axis (only positive weights).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=110,
        help="Dots per inch for the saved PNG (lower resolution to match paper).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["plain"],
        choices=["plain", "bio_trained", "naive"],
        help="Network variants to plot (plain = pre-training, bio_trained, naive).",
    )
    return parser.parse_args()


def resolve_cell_types(requested: list[str], nodes: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    ctdf = nu.get_cell_type_table()
    color_map: dict[str, str] = {}
    for cell_type, color in zip(ctdf["cell_type"], ctdf["hex"]):
        color_map.setdefault(cell_type, color)

    available = set(nodes["cell_type"].unique())
    resolved = []
    missing = []
    for entry in requested:
        normalized = entry.replace(" ", "_")
        if normalized in available:
            resolved.append(normalized)
            continue

        if entry in ctdf.index:
            cell_type = ctdf.loc[entry, "cell_type"]
            if isinstance(cell_type, pd.Series):
                cell_type = cell_type.iloc[0]
            resolved.append(cell_type)
            continue

        missing.append(entry)

    if missing:
        raise ValueError(f"Could not resolve cell types: {missing}")

    # Harmonize with boxplot high-cohort red to match figures
    cohort_high_red = "#c73635"
    for ct in resolved:
        color_map[ct] = cohort_high_red

    return resolved, color_map


def gather_weights(
    edges: dict[str, np.ndarray],
    nodes: pd.DataFrame,
    target_cell_types: list[str],
    direction: str,
) -> dict[str, np.ndarray]:
    key = "target_id" if direction == "incoming" else "source_id"
    weights_by_type: dict[str, np.ndarray] = {}
    for cell_type in target_cell_types:
        node_subset = nodes.loc[nodes["cell_type"] == cell_type]
        if node_subset.empty:
            weights_by_type[cell_type] = np.array([])
            continue

        mask = np.isin(edges[key], node_subset.index.values)
        weights_by_type[cell_type] = edges["syn_weight"][mask]
    return weights_by_type


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading nodes from {args.network_dir} ...")
    load_start = perf_counter()
    nodes = nu.load_nodes(str(args.network_dir), expand=True)
    node_load_time = perf_counter() - load_start
    resolved_types, color_map = resolve_cell_types(args.cell_types, nodes)

    # Preload weights for all variants so we can enforce a shared x-range
    variant_payloads = []
    for variant in args.variants:
        appendix = "" if variant == "plain" else f"_{variant}"
        print(f"Loading edges ({variant}) ...")
        edge_start = perf_counter()
        edges = nu.load_edges(
            str(args.network_dir), src="v1", tgt="v1", appendix=appendix
        )
        edge_load_time = perf_counter() - edge_start
        weights_by_type = gather_weights(edges, nodes, resolved_types, args.direction)
        variant_payloads.append((variant, weights_by_type, edge_load_time))

    # Compute a common x-range across all variants (use positive weights if log-scale)
    pooled_weights = []
    for _, weights_by_type, _ in variant_payloads:
        for w in weights_by_type.values():
            if args.log_scale:
                w = w[w > 0]
            pooled_weights.append(w)
    pooled_weights = np.concatenate(pooled_weights) if pooled_weights else np.array([])
    if pooled_weights.size == 0:
        raise RuntimeError("No weights available to plot across requested variants.")
    x_min, x_max = pooled_weights.min(), pooled_weights.max()
    # Clamp to a publication-friendly range if log scale is used
    if args.log_scale:
        x_min = max(x_min, 3e-1)
        x_max = min(x_max, 4e2)

    print(
        f"Loaded nodes in {node_load_time:.2f}s; "
        f"resolved {len(resolved_types)} cell type(s). Shared x-range: [{x_min:.3e}, {x_max:.3e}]"
    )

    uniform_ylim = None

    for variant, weights_by_type, edge_load_time in variant_payloads:
        fig, ax = plt.subplots(figsize=(2.0, 1.5))
        # ax.set_title(
        #     f"{variant.replace('_', ' ').title()}",
        #     fontsize=11,
        # )
        ax.set_xlabel("Synaptic weight (pA)")
        ax.set_ylabel("Density")
        if args.log_scale:
            ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)

        plotted = False
        for cell_type, weights in weights_by_type.items():
            if weights.size == 0:
                print(f"Warning: {cell_type} has no sampled connections.")
                continue
            plot_weights = weights
            if args.log_scale:
                plot_weights = weights[weights > 0]
                if plot_weights.size == 0:
                    print(f"Warning: {cell_type} has no positive weights for log scale.")
                    continue
            sns.kdeplot(
                plot_weights,
                ax=ax,
                bw_method="scott",
                fill=False,
                label=None,
                color=color_map.get(cell_type, "#222222"),
                linewidth=1.25,
            )
            plotted = True

        if not plotted:
            raise RuntimeError("No weights were plotted; check the requested cell types.")

        sns.despine(ax=ax)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        safe_types = [ct.replace("/", "_").replace(" ", "_") for ct in resolved_types]
        log_suffix = "_log" if args.log_scale else ""
        variant_suffix = variant if variant != "plain" else "plain"
        base_name = (
            f"syn_weight_distribution_{args.direction}_{'_'.join(safe_types)}"
            f"{log_suffix}_{variant_suffix}"
        )
        png_path = args.output_dir / f"{base_name}.png"
        pdf_path = args.output_dir / f"{base_name}.pdf"
        plt.savefig(png_path, dpi=args.dpi)
        plt.savefig(pdf_path)
        if variant == "naive":
            uniform_ylim = ax.get_ylim()

        plt.close(fig)

        total_edges = sum(len(w) for w in weights_by_type.values())
        print(
            f"Saved distribution ({total_edges:,} synapses) to {png_path.name}; "
            f"edges loaded in {edge_load_time:.2f}s"
        )

    # Add a uniform reference line plot at 10.2 pA using the same axes ranges
    if uniform_ylim is not None:
        fig, ax = plt.subplots(figsize=(2.0, 1.5))
        if args.log_scale:
            ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(uniform_ylim)
        ax.axvline(
            10.2,
            color=color_map.get(resolved_types[0], "#222222"),
            linewidth=1.5,
        )
        ax.set_xlabel("Synaptic weight (pA)")
        ax.set_ylabel("Density")
        sns.despine(ax=ax)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        safe_types = [ct.replace("/", "_").replace(" ", "_") for ct in resolved_types]
        log_suffix = "_log" if args.log_scale else ""
        base_name = (
            f"syn_weight_distribution_{args.direction}_{'_'.join(safe_types)}"
            f"{log_suffix}_naive_uniform_ref"
        )
        png_path = args.output_dir / f"{base_name}.png"
        pdf_path = args.output_dir / f"{base_name}.pdf"
        plt.savefig(png_path, dpi=args.dpi)
        plt.savefig(pdf_path)
        plt.close(fig)
        print(f"Saved uniform reference line plot to {png_path.name}")


if __name__ == "__main__":
    main()

