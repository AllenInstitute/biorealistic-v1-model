#!/usr/bin/env python3
"""Stacked bar plots of target fractions for synapsecount-defined cohorts (Figure 5 degree-style).

Input defaults to:
  core_nll_0/figures/selectivity_outgoing/outgoing_synapsecount_complete_targets{suffix}.csv

This is a synapse-count analog of `plot_target_fraction_figure5.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analysis_shared.style import apply_pub_style

OUTPUT_DIR = PROJECT_ROOT / "figures" / "paper" / "figure5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NETWORKS = [
    ("", "Bio-trained"),
    ("_plain", "Untrained"),
]

TARGET_COLORS = {
    "Exc": "#D42A2A",
    "PV": "#4C7F19",
    "SST": "#197F7F",
    "VIP": "#9932FF",
}

# =============================================================================
# Plot settings (edit these for consistent sizing across all panels in this file)
# =============================================================================
COMBINED_FIGSIZE = (5.0, 3.0)
COMBINED_BAR_WIDTH = 0.18
COMBINED_BAR_SPACING = 0.02
COMBINED_BLOCK_SPACING = 0.08
# Label hierarchy (closest to axis → furthest): cohort (xticks), training, cell type
COMBINED_COHORT_TICK_FS = 10
COMBINED_COHORT_TICK_PAD = 2
COMBINED_TRAINING_LABEL_Y = -0.14
COMBINED_TRAINING_LABEL_FS = 10
COMBINED_CELLTYPE_LABEL_Y = -0.28
COMBINED_CELLTYPE_LABEL_FS = 11

# Match weight-panel aesthetics
Y_LABEL_FS = 9  # smaller than weight panel to keep left margin tidy
Y_TICK_FS = 9  # match y-tick font to weight panel (avoid apply_pub_style shrink)
LEGEND_FS = 9
LEGEND_TITLE_FS = 9
ANNOT_FS = 7
ANNOT_MIN_FRAC = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot target fraction stacked bars for synapsecount-defined cohorts (Fig.5)."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional suffix (e.g. '_plain') to run a single specific network.",
    )
    return parser.parse_args()


def load_core_dataframe(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        print(f"Warning: File not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    core_df = df[df["group"].str.contains("_core")].copy()
    core_df["cell_type"] = core_df["group"].str.extract(r"(exc|inh|pv|sst|vip)")[0]
    core_df["weight_group"] = core_df["group"].str.extract(r"_(high|mid|low)_")[0]
    return core_df


def build_target_fraction_plot(core_df: pd.DataFrame, suffix: str) -> None:
    cell_types: Sequence[str] = ["exc", "inh"]
    weight_groups: Sequence[str] = ["low", "mid", "high"]
    targets: Sequence[str] = ["Exc", "PV", "SST", "VIP"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x_positions = []
    x_labels = []
    bar_width = 0.25
    group_spacing = len(weight_groups) * (bar_width + 0.05) + 0.35

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
            ]

            bottom = 0.0
            for idx, (target, value) in enumerate(fractions):
                ax.bar(
                    x_pos,
                    value,
                    bar_width,
                    bottom=bottom,
                    color=TARGET_COLORS[target],
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
                bottom += value

    ax.set_ylabel("Fraction of outgoing synapse count", fontsize=11)
    ax.set_xlabel("Source cell type and outgoing cohort", fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Patch(facecolor=TARGET_COLORS[label], edgecolor="black", label=label)
        for label in targets
    ]
    ax.legend(
        handles=handles,
        title="Target type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=LEGEND_FS,
        title_fontsize=LEGEND_TITLE_FS,
    )

    tag = "bio_trained" if suffix == "" else "untrained"
    png_path = OUTPUT_DIR / f"target_fraction_exc_inh_synapsecount_{tag}.png"
    pdf_path = OUTPUT_DIR / f"target_fraction_exc_inh_synapsecount_{tag}.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path} and PDF")
    plt.close(fig)


def build_combined_training_plot(bio_df: pd.DataFrame, plain_df: pd.DataFrame) -> None:
    weight_groups: Sequence[str] = ["low", "mid", "high"]
    stack_order = ["VIP", "SST", "PV", "Exc"]

    combined = []
    for df, training in ((plain_df, "Untrained"), (bio_df, "Trained")):
        tmp = df.copy()
        tmp["training"] = training
        combined.append(tmp)
    combined_df = pd.concat(combined, ignore_index=True)

    fig, ax = plt.subplots(figsize=COMBINED_FIGSIZE)
    bar_width = COMBINED_BAR_WIDTH
    bar_spacing = COMBINED_BAR_SPACING
    block_spacing = COMBINED_BLOCK_SPACING
    training_label_y = COMBINED_TRAINING_LABEL_Y
    label_y = COMBINED_CELLTYPE_LABEL_Y

    x_positions = []
    xtick_labels = []
    # Reorder to (cell type, training, cohort): EXC then INH, with Untrained/Trained inside each.
    pair_order = [
        ("Untrained", "exc"),
        ("Trained", "exc"),
        ("Untrained", "inh"),
        ("Trained", "inh"),
    ]

    for p_idx, (training, ct) in enumerate(pair_order):
        base_x = p_idx * (
            len(weight_groups) * (bar_width + bar_spacing) + block_spacing
        )
        for j, wg in enumerate(weight_groups):
            row = combined_df[
                (combined_df["training"] == training)
                & (combined_df["weight_group"] == wg)
                & (combined_df["cell_type"] == ct)
            ]
            if row.empty:
                continue
            x_pos = base_x + j * (bar_width + bar_spacing)
            x_positions.append(x_pos)
            xtick_labels.append(f"{wg.title()}")

            values = {
                "Exc": row["exc_total"].values[0],
                "PV": row["inh_pv"].values[0],
                "SST": row["inh_sst"].values[0],
                "VIP": row["inh_vip"].values[0],
            }
            bottom = 0.0
            for target in stack_order:
                val = values[target]
                ax.bar(
                    x_pos,
                    val,
                    bar_width,
                    bottom=bottom,
                    color=TARGET_COLORS[target],
                    edgecolor="black",
                    linewidth=0.5,
                )
                # Match the weight-panel: annotate sizeable fractions inside bars
                if val > ANNOT_MIN_FRAC:
                    y_pos = bottom + val / 2
                    # Keep a small manual nudge to avoid legend overlap (same as weight panel)
                    if training == "Trained" and ct == "inh" and target == "Exc":
                        y_pos = 0.40
                    ax.text(
                        x_pos,
                        y_pos,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=ANNOT_FS,
                        fontweight="bold",
                        color="white",
                    )
                bottom += val

    ax.set_ylabel("Fraction of outgoing synapse count", fontsize=Y_LABEL_FS)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=Y_TICK_FS)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels, fontsize=COMBINED_COHORT_TICK_FS)
    ax.tick_params(axis="x", pad=COMBINED_COHORT_TICK_PAD)

    ticks = ax.get_xticks()
    if len(ticks) >= 6:
        block_size = len(weight_groups)
        # Repeat training labels under each cell type block
        exc_untrained_center = np.mean(ticks[0:block_size])
        exc_trained_center = np.mean(ticks[block_size : block_size * 2])
        inh_untrained_center = np.mean(ticks[block_size * 2 : block_size * 3])
        inh_trained_center = np.mean(ticks[block_size * 3 : block_size * 4])

        for x, lbl in (
            (exc_untrained_center, "Untrained"),
            (exc_trained_center, "Trained"),
            (inh_untrained_center, "Untrained"),
            (inh_trained_center, "Trained"),
        ):
            ax.text(
                x,
                training_label_y,
                lbl,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=COMBINED_TRAINING_LABEL_FS,
            )

        exc_center = np.mean(ticks[0 : block_size * 2])
        inh_center = np.mean(ticks[block_size * 2 : block_size * 4])
        ax.text(
            exc_center,
            label_y,
            "EXC",
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=COMBINED_CELLTYPE_LABEL_FS,
        )
        ax.text(
            inh_center,
            label_y,
            "INH",
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=COMBINED_CELLTYPE_LABEL_FS,
        )

    handles = [
        Patch(facecolor=TARGET_COLORS[label], edgecolor="black", label=label)
        for label in ["Exc", "PV", "SST", "VIP"]
    ]
    ax.legend(
        handles=handles,
        title="Target type",
        loc="upper right",
        fontsize=LEGEND_FS,
        title_fontsize=LEGEND_TITLE_FS,
    )

    png_path = (
        OUTPUT_DIR / "target_fraction_exc_inh_synapsecount_combined_celltype_first.png"
    )
    pdf_path = (
        OUTPUT_DIR / "target_fraction_exc_inh_synapsecount_combined_celltype_first.pdf"
    )
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path} and PDF")
    plt.close(fig)


def main() -> None:
    apply_pub_style()
    args = parse_args()

    to_run = []
    if args.tag is not None:
        label = "Bio-trained" if args.tag == "" else "Untrained"
        to_run.append((args.tag, label))
    else:
        to_run = NETWORKS

    loaded = {}
    for suffix, label in to_run:
        input_path = f"core_nll_0/figures/selectivity_outgoing/outgoing_synapsecount_complete_targets{suffix}.csv"
        print(f"\nProcessing {label} ({input_path})...")
        core_df = load_core_dataframe(input_path)
        if core_df.empty:
            continue
        loaded[suffix] = core_df
        build_target_fraction_plot(core_df, suffix)

    if "" in loaded and "_plain" in loaded:
        build_combined_training_plot(loaded[""], loaded["_plain"])


if __name__ == "__main__":
    main()
