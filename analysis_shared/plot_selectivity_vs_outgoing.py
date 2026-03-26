#!/usr/bin/env python3
"""Static plots summarising selectivity vs total outgoing synaptic weight."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = (
    PROJECT_ROOT / "cell_categorization" / "core_nll_0_neuron_features.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "figures" / "selectivity_outgoing"


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run cell_categorization/build_neuron_umap.py first."
        )
    df = pd.read_parquet(FEATURES_PATH)
    df = df.rename(columns={"outgoing_weight_sum_abs": "outgoing_weight_sum_abs"})
    required = {
        "cell_type",
        "image_selectivity",
        "outgoing_weight_sum",
        "outgoing_weight_sum_abs",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing expected columns in features parquet: {sorted(missing)}"
        )
    df = df.dropna(subset=["image_selectivity", "outgoing_weight_sum"])
    return df


def plot_scatter_grid(df: pd.DataFrame) -> None:
    cell_types = sorted(df["cell_type"].unique())
    n_cells = len(cell_types)
    n_cols = 4
    n_rows = ceil(n_cells / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3.2, n_rows * 3.2),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    axes_flat = axes.flat if isinstance(axes, np.ndarray) else (axes,)

    for ax, cell_type in zip(axes_flat, cell_types):
        subset = df[df["cell_type"] == cell_type]
        if subset.empty:
            ax.axis("off")
            continue
        x = subset["outgoing_weight_sum_abs"]
        y = subset["image_selectivity"]
        ax.scatter(x, y, s=8, alpha=0.6)
        if len(subset) >= 2:
            r = float(np.corrcoef(x, y)[0, 1])
        else:
            r = np.nan
        ax.set_title(f"{cell_type}\nR={r:.2f} (n={len(subset)})", fontsize=9)
        ax.set_xlabel("|Total outgoing weight|")
        ax.set_ylabel("Image selectivity")

    for ax in axes_flat[n_cells:]:
        ax.axis("off")

    fig.suptitle("Selectivity vs total outgoing weight (per cell type)", fontsize=14)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "scatter_selectivity_vs_outgoing.png", dpi=200)
    plt.close(fig)


def plot_distribution_facets(df: pd.DataFrame) -> None:
    cell_types = sorted(df["cell_type"].unique())
    palette = sns.color_palette("tab20", n_colors=max(len(cell_types), 3))
    color_map = {ct: palette[i % len(palette)] for i, ct in enumerate(cell_types)}

    def facet_plot(data: pd.DataFrame, color=None, **kwargs):  # type: ignore[override]
        ct = data["cell_type"].iloc[0]
        sns.histplot(
            data=data,
            x="outgoing_weight_sum_abs",
            bins=40,
            color=color_map.get(ct, "#333333"),
            kde=False,
            stat="density",
            **kwargs,
        )

    g = sns.FacetGrid(
        df,
        col="cell_type",
        col_wrap=4,
        sharex=False,
        sharey=False,
        height=2.6,
    )
    g.map_dataframe(facet_plot)
    g.set_axis_labels("|Total outgoing weight|", "Density")
    g.set_titles(col_template="{col_name}")
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle("Distribution of total outgoing weights by cell type", fontsize=14)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(OUTPUT_DIR / "hist_outgoing_weight_by_type.png", dpi=200)
    plt.close(g.fig)


def main() -> None:
    df = load_features()
    plot_scatter_grid(df)
    plot_distribution_facets(df)


if __name__ == "__main__":
    main()
