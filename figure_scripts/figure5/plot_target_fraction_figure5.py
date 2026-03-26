#!/usr/bin/env python3
"""Bar plot of target fractions (Exc vs Inh source, granular targets) for outgoing weight cohorts (Figure 5)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "figures" / "paper" / "figure5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIRS = [
    PROJECT_ROOT / f"core_nll_{i}" for i in range(10) if (PROJECT_ROOT / f"core_nll_{i}").exists()
]

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
COMBINED_WEIGHT_TICK_FS = 10
COMBINED_EXTRA_BOTTOM_PAD = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot target fraction bar plots for Exc/Inh sources with granular targets (Figure 5)."
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


def load_core_dataframe_aggregated(suffix: str) -> pd.DataFrame:
    if not BASE_DIRS:
        return pd.DataFrame()

    paths: list[Path] = []
    for base_dir in BASE_DIRS:
        p = (
            base_dir
            / "figures"
            / "selectivity_outgoing"
            / f"outgoing_weight_complete_targets{suffix}.csv"
        )
        if p.exists():
            paths.append(p)

    if not paths:
        fallback = (
            PROJECT_ROOT
            / "core_nll_0"
            / "figures"
            / "selectivity_outgoing"
            / f"outgoing_weight_complete_targets{suffix}.csv"
        )
        return load_core_dataframe(str(fallback))

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


def build_target_fraction_plot(
    core_df: pd.DataFrame, suffix: str, network_label: str
) -> None:
    """Create bar plot showing Exc/PV/SST/VIP target fractions for Exc/Inh sources."""
    cell_types: Sequence[str] = ["exc", "inh"]
    # Ordering: lowest unit = weight group (low→mid→high), then cell type (Exc, Inh), then training (Untrained, Trained)
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

    ax.set_ylabel("Fraction of outgoing weight", fontsize=11)
    ax.set_xlabel("Source cell type and outgoing weight group", fontsize=11)
    # Title removed as requested
    # ax.set_title(
    #     f"Target Composition ({network_label})", fontsize=12, pad=15
    # )
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
        fontsize=9,
    )

    if len(cell_types) > 1:
        ax.axvline(
            group_spacing - 0.175,
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.3,
        )

    tag = "bio_trained" if suffix == "" else "untrained"
    png_path = OUTPUT_DIR / f"target_fraction_exc_inh_{tag}.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.close(fig)


def build_comparison_plot(
    core_df: pd.DataFrame, suffix: str, network_label: str
) -> None:
    """Create comparison plot showing high/mid/low side by side for Exc/Inh sources with granular targets."""
    weight_groups: Sequence[str] = ["high", "mid", "low"]
    cell_types: Sequence[str] = ["exc", "inh"]
    targets: Sequence[str] = ["Exc", "PV", "SST", "VIP"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    # Title removed as requested
    # fig.suptitle(
    #     f"Target Preferences by Weight Cohort ({network_label})", fontsize=12, y=0.98
    # )

    alpha_map = {"high": 0.9, "mid": 0.6, "low": 0.3}

    for idx, ct in enumerate(cell_types):
        ax = axes[idx]
        available_rows = {
            wg: core_df[(core_df["cell_type"] == ct) & (core_df["weight_group"] == wg)]
            for wg in weight_groups
        }
        if all(r.empty for r in available_rows.values()):
            ax.set_visible(False)
            continue

        x = np.arange(len(targets))
        width = 0.22

        # Determine max value for y-limit scaling
        max_val = 0.0

        for w_idx, weight in enumerate(weight_groups):
            row = available_rows[weight]
            if row.empty:
                continue

            vals = [
                row["exc_total"].values[0],
                row["inh_pv"].values[0],
                row["inh_sst"].values[0],
                row["inh_vip"].values[0],
            ]
            max_val = max(max_val, max(vals))

            offset = (w_idx - (len(weight_groups) - 1) / 2) * (width + 0.02)
            ax.bar(
                x + offset,
                vals,
                width,
                color=[TARGET_COLORS[t] for t in targets],
                alpha=alpha_map[weight],
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_ylabel("Fraction", fontsize=9)
        # Subplot title removed as requested
        # ax.set_title(f"{ct.upper()} neurons", fontsize=10, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(targets, fontsize=9)
        ax.set_ylim(0, max_val * 1.2)

        if idx == 0:
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

    tag = "bio_trained" if suffix == "" else "untrained"
    png_path = OUTPUT_DIR / f"target_fraction_comparison_exc_inh_{tag}.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {png_path}")
    plt.close(fig)


def build_combined_training_plot(bio_df: pd.DataFrame, plain_df: pd.DataFrame) -> None:
    """Single panel combining untrained and trained with boxplot-style labels."""
    weight_groups: Sequence[str] = ["low", "mid", "high"]
    targets: Sequence[str] = ["Exc", "PV", "SST", "VIP"]
    # Stack order so Exc ends up on top of the stack
    stack_order = ["VIP", "SST", "PV", "Exc"]

    # Build a combined dataframe with training status
    combined = []
    for df, training in ((plain_df, "Untrained"), (bio_df, "Trained")):
        tmp = df.copy()
        tmp["training"] = training
        combined.append(tmp)
    combined_df = pd.concat(combined, ignore_index=True)

    fig, ax = plt.subplots(figsize=COMBINED_FIGSIZE)
    # Layout knobs (user-adjustable) — centralized above
    bar_width = COMBINED_BAR_WIDTH
    bar_spacing = COMBINED_BAR_SPACING
    block_spacing = COMBINED_BLOCK_SPACING
    training_label_y = COMBINED_TRAINING_LABEL_Y
    label_y = COMBINED_CELLTYPE_LABEL_Y

    x_positions = []
    xtick_labels = []

    # Reorder to (cell type, training, cohort): EXC → (Untrained, Trained) → (Low, Mid, High),
    # then INH → (Untrained, Trained) → (Low, Mid, High).
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
            # Only show weight-group label; cell-type will be added as shared labels below
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
                    label=(
                        target
                        if (not x_positions[:-1] and target == stack_order[-1])
                        else ""
                    ),
                )
                if val > 0.05:
                    y_pos = bottom + val / 2
                    # Lower trained-inh annotations slightly to avoid legend overlap
                    if training == "Trained" and ct == "inh" and target == "Exc":
                        y_pos = 0.40
                    ax.text(
                        x_pos,
                        y_pos,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color="white",
                    )
                bottom += val

    ax.set_ylabel("Fraction of outgoing weight", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels, fontsize=COMBINED_COHORT_TICK_FS)
    ax.tick_params(axis="x", pad=COMBINED_COHORT_TICK_PAD)

    # Add repeated training labels beneath each cell-type block (Untrained/Trained repeated for EXC and INH)
    ticks = ax.get_xticks()
    if len(ticks) >= 6:
        block_size = len(weight_groups)
        y_text = training_label_y
        # EXC block
        exc_untrained_center = np.mean(ticks[0:block_size])
        exc_trained_center = np.mean(ticks[block_size : block_size * 2])
        # INH block
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
                y_text,
                lbl,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=COMBINED_TRAINING_LABEL_FS,
            )

    # Add shared EXC/INH labels centered over their 6-bar blocks
    if len(ticks) >= 6:
        block_size = len(weight_groups)
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
        for label in targets
    ]
    ax.legend(handles=handles, title="Target type", loc="upper right", fontsize=9)

    png_path = OUTPUT_DIR / "target_fraction_exc_inh_combined_celltype_first.png"
    pdf_path = OUTPUT_DIR / "target_fraction_exc_inh_combined_celltype_first.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path} and PDF")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.tag is not None:
        suffix, label = args.tag, args.tag.replace("_", "").title()
        df = load_core_dataframe(
            f"core_nll_0/figures/selectivity_outgoing/outgoing_weight_complete_targets{suffix}.csv"
        )
        if df.empty:
            print("No data found")
            return
        build_target_fraction_plot(df, suffix, label)
        build_comparison_plot(df, suffix, label)
        return

    bio_df = load_core_dataframe_aggregated("")
    plain_df = load_core_dataframe_aggregated("_plain")

    # Individual panels
    if not bio_df.empty:
        build_target_fraction_plot(bio_df, "", "Bio-trained")
        build_comparison_plot(bio_df, "", "Bio-trained")

    if not plain_df.empty:
        build_target_fraction_plot(plain_df, "_plain", "Untrained")
        build_comparison_plot(plain_df, "_plain", "Untrained")

    # Combined trained vs untrained panel if both available
    if not bio_df.empty and not plain_df.empty:
        build_combined_training_plot(bio_df, plain_df)


if __name__ == "__main__":
    main()
