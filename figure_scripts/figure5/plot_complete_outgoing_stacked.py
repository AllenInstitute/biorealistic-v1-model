#!/usr/bin/env python3
"""Visualize complete outgoing connectivity patterns with stacked bars that add to 1.0."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked outgoing connectivity fractions for high/low cohorts."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix that matches outgoing_weight_complete_targets{suffix}.csv",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Explicit path to the complete targets CSV. Overrides --tag.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/paper/figure5",
        help="Directory to store the generated figures (defaults to figure 5 panels).",
    )
    return parser.parse_args()


def load_core_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    core_df = df[df["group"].str.contains("_core")].copy()
    core_df["cell_type"] = core_df["group"].str.extract(r"(exc|inh|pv|sst|vip)")[0]
    core_df["weight_group"] = core_df["group"].str.extract(r"_(high|mid|low)_")[0]
    return core_df


def load_core_dataframe_aggregated(tag: str) -> pd.DataFrame:
    suffix = f"_{tag}" if tag else ""
    base_dirs = [Path(f"core_nll_{i}") for i in range(10) if Path(f"core_nll_{i}").exists()]
    paths: list[Path] = []
    for base_dir in base_dirs:
        p = (
            base_dir
            / "figures"
            / "selectivity_outgoing"
            / f"outgoing_weight_complete_targets{suffix}.csv"
        )
        if p.exists():
            paths.append(p)

    if not paths:
        fallback = Path(
            f"core_nll_0/figures/selectivity_outgoing/outgoing_weight_complete_targets{suffix}.csv"
        )
        if fallback.exists():
            return load_core_dataframe(str(fallback))
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for p in paths:
        df = load_core_dataframe(str(p))
        if df.empty:
            continue
        df["_source"] = p.parent.parent.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    full = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        c
        for c in full.columns
        if c not in {"group", "cell_type", "weight_group", "_source"}
    ]
    agg = full.groupby(["group", "cell_type", "weight_group"], as_index=False)[
        numeric_cols
    ].mean(numeric_only=True)
    return agg


def build_stacked_plot(core_df: pd.DataFrame, output_dir: str, suffix: str) -> None:
    target_colors = {
        "Exc": "#D42A2A",
        "PV": "#4C7F19",
        "SST": "#197F7F",
        "VIP": "#9932FF",
        "Htr3a": "#787878",
    }
    cell_types: Sequence[str] = ["exc", "inh", "pv", "sst", "vip"]

    # Use low-mid-high ordering consistently across panels
    weight_groups: Sequence[str] = ["low", "mid", "high"]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    x_positions = []
    x_labels = []
    bar_width = 0.25
    group_spacing = len(weight_groups) * (bar_width + 0.05) + 0.25

    for i, ct in enumerate(cell_types):
        base_x = i * group_spacing
        for j, wg in enumerate(weight_groups):
            row = core_df[
                (core_df["cell_type"] == ct) & (core_df["weight_group"] == wg)
            ]
            if row.empty:
                continue

            x_pos = base_x + j * (bar_width + 0.05)
            x_positions.append(x_pos)
            x_labels.append(f"{ct.upper()}\n{wg}")

            fractions = [
                ("Exc", row["exc_total"].values[0]),
                ("PV", row["inh_pv"].values[0]),
                ("SST", row["inh_sst"].values[0]),
                ("VIP", row["inh_vip"].values[0]),
                ("Htr3a", row["inh_htr3a"].values[0]),
            ]

            bottom = 0.0
            for idx, (target, value) in enumerate(fractions):
                ax.bar(
                    x_pos,
                    value,
                    bar_width,
                    bottom=bottom,
                    color=target_colors[target],
                    edgecolor="black",
                    linewidth=0.5,
                    label=target if (i == 0 and j == 0 and idx == 0) else "",
                )

                if value > 0.05:
                    text_color = "white" if value > 0.15 else "black"
                    ax.text(
                        x_pos,
                        bottom + value / 2,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        weight="bold",
                        color=text_color,
                    )
                elif value > 0.01:
                    ax.text(
                        x_pos,
                        bottom + value / 2,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black",
                    )
                bottom += value

    ax.set_ylabel("Fraction of outgoing weight", fontsize=11)
    ax.set_xlabel("Source cell type and outgoing weight group", fontsize=11)
    ax.set_title(
        "Core-to-Core Connectivity: Complete Target Composition", fontsize=12, pad=15
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=target_colors[label], edgecolor="black", label=label)
        for label in ["Exc", "PV", "SST", "VIP", "Htr3a"]
    ]
    ax.legend(
        handles=handles,
        title="Target type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
    )

    for i in range(1, len(cell_types)):
        ax.axvline(
            i * group_spacing - 0.15,
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.3,
        )

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"complete_outgoing_stacked{suffix}.png")
    pdf_path = os.path.join(output_dir, f"complete_outgoing_stacked{suffix}.pdf")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"\nSaved: {png_path} / {pdf_path}")
    plt.close(fig)


def build_comparison_plot(core_df: pd.DataFrame, output_dir: str, suffix: str) -> None:
    target_colors = {
        "Exc": "#D42A2A",
        "PV": "#4C7F19",
        "SST": "#197F7F",
        "VIP": "#9932FF",
        "Htr3a": "#787878",
    }
    weight_groups: Sequence[str] = ["low", "mid", "high"]
    cell_types: Sequence[str] = ["exc", "inh", "pv", "sst", "vip"]
    n_cols = 3
    n_rows = int(np.ceil(len(cell_types) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5, 3.2 * n_rows))
    axes = np.atleast_2d(axes)
    fig.suptitle(
        "High vs Low Outgoing Weight: Complete Target Preferences", fontsize=12, y=0.98
    )

    targets = ["Exc", "PV", "SST", "VIP", "Htr3a"]
    alpha_map = {"high": 0.9, "mid": 0.6, "low": 0.3}

    for idx, ct in enumerate(cell_types):
        ax = axes[idx // n_cols, idx % n_cols]
        available_rows = {
            wg: core_df[(core_df["cell_type"] == ct) & (core_df["weight_group"] == wg)]
            for wg in weight_groups
        }
        if all(r.empty for r in available_rows.values()):
            ax.set_visible(False)
            continue

        x = np.arange(len(targets))
        width = 0.22
        for w_idx, weight in enumerate(weight_groups):
            row = available_rows[weight]
            if row.empty:
                continue
            vals = [
                row["exc_total"].values[0],
                row["inh_pv"].values[0],
                row["inh_sst"].values[0],
                row["inh_vip"].values[0],
                row["inh_htr3a"].values[0],
            ]
            offset = (w_idx - (len(weight_groups) - 1) / 2) * (width + 0.02)
            ax.bar(
                x + offset,
                vals,
                width,
                color=[target_colors[t] for t in targets],
                alpha=alpha_map[weight],
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_ylabel("Fraction", fontsize=9)
        ax.set_title(f"{ct.upper()} neurons", fontsize=10, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(targets, fontsize=9)
        max_val = max(
            max(
                row["exc_total"].values[0]
                for row in available_rows.values()
                if not row.empty
            ),
            max(
                (
                    row["inh_pv"] + row["inh_sst"] + row["inh_vip"] + row["inh_htr3a"]
                ).values[0]
                for row in available_rows.values()
                if not row.empty
            ),
        )
        ax.set_ylim(0, max_val * 1.2)
        if idx == 0:
            from matplotlib.patches import Patch

            handles = [
                Patch(
                    facecolor="gray",
                    edgecolor="black",
                    alpha=alpha_map[weight],
                    label=f"{weight.title()} weight",
                )
                for weight in weight_groups
            ]
            ax.legend(
                handles=handles, fontsize=8, title="Weight cohort", loc="upper right"
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(cell_types), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"complete_high_vs_low_comparison{suffix}.png")
    pdf_path = os.path.join(output_dir, f"complete_high_vs_low_comparison{suffix}.pdf")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path} / {pdf_path}")
    plt.close(fig)


def print_summary(core_df: pd.DataFrame, tag: str) -> None:
    cell_types: Sequence[str] = ["exc", "inh", "pv", "sst", "vip"]
    weight_groups: Sequence[str] = ["high", "mid", "low"]
    print("\n" + "=" * 80)
    print(f"COMPLETE CONNECTIVITY ANALYSIS: HIGH/MID/LOW OUTGOING WEIGHT (tag='{tag}')")
    print("=" * 80)
    print("\n1. TARGET COMPOSITION (fractions sum to 1.0):")

    for ct in cell_types:
        print(f"\n   {ct.upper()} neurons:")
        for weight in weight_groups:
            row = core_df[
                (core_df["cell_type"] == ct) & (core_df["weight_group"] == weight)
            ]
            if row.empty:
                continue
            values = row.iloc[0]
            total = (
                values["exc_total"]
                + values["inh_pv"]
                + values["inh_sst"]
                + values["inh_vip"]
                + values["inh_htr3a"]
            )
            print(
                f"      {weight.title():<4} weight: Exc={values['exc_total']:.3f}, PV={values['inh_pv']:.3f}, "
                f"SST={values['inh_sst']:.3f}, VIP={values['inh_vip']:.3f}, "
                f"Htr3a={values['inh_htr3a']:.3f} (sum={total:.3f})"
            )


def main() -> None:
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    if args.input:
        print(f"Loading data from {args.input}")
        core_df = load_core_dataframe(args.input)
    else:
        print(
            f"Loading aggregated data from core_nll_0..9 outgoing_weight_complete_targets{suffix}.csv"
        )
        core_df = load_core_dataframe_aggregated(args.tag or "")

    print("Groups (high/mid/low outgoing weight cohorts):")
    print(core_df[["group", "cell_type", "weight_group"]].to_string(index=False))

    build_stacked_plot(core_df, args.output_dir, suffix)
    build_comparison_plot(core_df, args.output_dir, suffix)
    print_summary(core_df, args.tag)


if __name__ == "__main__":
    main()
