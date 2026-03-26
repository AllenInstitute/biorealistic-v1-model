#!/usr/bin/env python3
"""Plot connectivity fractions for PV/SST/VIP high/low source groups (core/periphery).

Outputs:
- figures/selectivity_outgoing/subtyped_source_connectivity.png
- figures/selectivity_outgoing/subtyped_source_connectivity.svg

Figure: 2x2 panels (Core vs Periphery x E/I vs Inh subtypes), stacked bars per source group.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "core_nll_0"
OUTPUT_DIR = BASE_DIR / "figures" / "selectivity_outgoing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_ALL = OUTPUT_DIR / "outgoing_weight_granular_summary.csv"
SUMMARY_C2C = OUTPUT_DIR / "outgoing_weight_granular_core_to_core_summary.csv"
CELLTYPE_COLOR_CSV = PROJECT_ROOT / "base_props" / "cell_type_naming_scheme.csv"


def _load_colors() -> Dict[str, str]:
    df = pd.read_csv(CELLTYPE_COLOR_CSV, sep=r"\s+")
    # Map short cell_type (pv/sst/vip) to hex
    # The CSV uses lowercase in `cell_type` and provides `hex`.
    color_map = {}
    for key in ("pv", "sst", "vip"):
        rows = df[df["cell_type"].str.lower() == key]
        if not rows.empty:
            color_map[key] = rows.iloc[0]["hex"]
    # Fallbacks to ensure consistent palette
    color_map.setdefault("pv", "#4C7F19")
    color_map.setdefault("sst", "#197F7F")
    color_map.setdefault("vip", "#9932FF")
    color_map.setdefault("exc", "#C93C3C")
    color_map.setdefault("inh", "#787878")
    return color_map


def _order_source_groups(groups: List[str]) -> List[str]:
    order_priority = [
        "pv_high_core",
        "pv_low_core",
        "sst_high_core",
        "sst_low_core",
        "vip_high_core",
        "vip_low_core",
        "inh_high_core",
        "inh_low_core",
        "pv_high_periphery",
        "pv_low_periphery",
        "sst_high_periphery",
        "sst_low_periphery",
        "vip_high_periphery",
        "vip_low_periphery",
        "inh_high_periphery",
        "inh_low_periphery",
    ]
    # keep only those present
    present = [g for g in order_priority if g in groups]
    # append any others at the end
    tail = [g for g in groups if g not in present]
    return present + tail


def _subset_by_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    if region == "core":
        mask = df["group"].str.contains("_core")
    elif region == "periphery":
        mask = df["group"].str.contains("_periphery")
    else:
        mask = pd.Series(False, index=df.index)
    return df[mask].copy()


def _plot_panel_stacked_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    groups: List[str],
    stacks: List[str],
    colors: Dict[str, str],
    title: str,
    ylabel: str,
) -> None:
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))

    for key in stacks:
        values = [float(df.loc[df["group"] == g, key].values[0]) if (df["group"] == g).any() else 0.0 for g in groups]
        ax.bar(x, values, bottom=bottom, color=colors.get(key, "#888"), label=key.upper())
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_core", "").replace("_periphery", "") for g in groups], rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3, linestyle=":")


def main() -> None:
    if not SUMMARY_ALL.exists():
        raise FileNotFoundError(f"Missing {SUMMARY_ALL}")
    df_all = pd.read_csv(SUMMARY_ALL)

    # Core-to-core summary has an extra n_connections column; align columns.
    if SUMMARY_C2C.exists():
        df_c2c = pd.read_csv(SUMMARY_C2C)
        if "n_connections" in df_c2c.columns:
            df_c2c = df_c2c.drop(columns=["n_connections"])  # fractions only
    else:
        df_c2c = None

    colors = _load_colors()
    # map for stacks
    stack_colors = {
        "exc_high": colors["exc"],
        "exc_low": "#EFBFBF",
        "inh_high": colors["inh"],
        "inh_low": "#CECECE",
        "inh_pv": colors["pv"],
        "inh_sst": colors["sst"],
        "inh_vip": colors["vip"],
    }

    # Determine ordering of groups present in df_all
    groups = _order_source_groups(df_all["group"].tolist())
    df_all = df_all.set_index("group").reindex(groups).reset_index()
    if df_c2c is not None:
        df_c2c = df_c2c.set_index("group").reindex(groups).reset_index()

    # Panels: Core/Periphery x E/I vs Inh subtypes
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), dpi=150)

    # Core region
    df_core = _subset_by_region(df_all, "core")
    core_groups = [g for g in groups if g.endswith("_core")]
    _plot_panel_stacked_bars(
        axes[0, 0],
        df_core,
        core_groups,
        stacks=["exc_high", "exc_low", "inh_high", "inh_low"],
        colors=stack_colors,
        title="Core: E/I fractions",
        ylabel="Weight fraction",
    )
    _plot_panel_stacked_bars(
        axes[1, 0],
        df_core,
        core_groups,
        stacks=["inh_pv", "inh_sst", "inh_vip"],
        colors=stack_colors,
        title="Core: inhibitory subtypes",
        ylabel="Weight fraction",
    )

    # Periphery region
    df_periph = _subset_by_region(df_all, "periphery")
    periph_groups = [g for g in groups if g.endswith("_periphery")]
    _plot_panel_stacked_bars(
        axes[0, 1],
        df_periph,
        periph_groups,
        stacks=["exc_high", "exc_low", "inh_high", "inh_low"],
        colors=stack_colors,
        title="Periphery: E/I fractions",
        ylabel="Weight fraction",
    )
    _plot_panel_stacked_bars(
        axes[1, 1],
        df_periph,
        periph_groups,
        stacks=["inh_pv", "inh_sst", "inh_vip"],
        colors=stack_colors,
        title="Periphery: inhibitory subtypes",
        ylabel="Weight fraction",
    )

    # Common aesthetics
    handles, labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = OUTPUT_DIR / "subtyped_source_connectivity.png"
    svg_path = OUTPUT_DIR / "subtyped_source_connectivity.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Saved {png_path} and {svg_path}")


if __name__ == "__main__":
    main()







