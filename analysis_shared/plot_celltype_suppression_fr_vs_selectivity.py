#!/usr/bin/env python3
"""Percent change scatter plots for cell-type specific suppression experiments."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

BASE_DIR = PROJECT_ROOT / "core_nll_0"
METRIC_DIR = BASE_DIR / "metrics"
OUTPUT_DIR = BASE_DIR / "figures" / "perturbation_fr_selectivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = PROJECT_ROOT / "cell_categorization" / "core_nll_0_neuron_features.parquet"

FRACTION_EPS = 1e-6
CORE_RADIUS = 200.0
METRICS = ("Ave_Rate(Hz)", "OSI", "DSI")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display: str
    metric_file: str
    target_nodes_file: Path


BASELINE_SPEC = DatasetSpec(
    key="bio_trained",
    display="Bio-trained",
    metric_file="OSI_DSI_DF_bio_trained.csv",
    target_nodes_file=Path(),
)

CELLTYPE_SUPPRESSION: Sequence[DatasetSpec] = (
    DatasetSpec(
        key="pv_high_neg1000",
        display="PV High",
        metric_file="OSI_DSI_DF_pv_high_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "pv_high_outgoing_nodes.json",
    ),
    DatasetSpec(
        key="pv_low_neg1000",
        display="PV Low",
        metric_file="OSI_DSI_DF_pv_low_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "pv_low_outgoing_nodes.json",
    ),
    DatasetSpec(
        key="sst_high_neg1000",
        display="SST High",
        metric_file="OSI_DSI_DF_sst_high_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "sst_high_outgoing_nodes.json",
    ),
    DatasetSpec(
        key="sst_low_neg1000",
        display="SST Low",
        metric_file="OSI_DSI_DF_sst_low_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "sst_low_outgoing_nodes.json",
    ),
    DatasetSpec(
        key="vip_high_neg1000",
        display="VIP High",
        metric_file="OSI_DSI_DF_vip_high_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "vip_high_outgoing_nodes.json",
    ),
    DatasetSpec(
        key="vip_low_neg1000",
        display="VIP Low",
        metric_file="OSI_DSI_DF_vip_low_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "vip_low_outgoing_nodes.json",
    ),
)


def _format_sigfig(value: float, sig: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    if value == 0:
        return "0"
    return f"{value:.{sig}g}"


def _load_features() -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH)
    cols = [col for col in ("node_id", "cell_type", "radius") if col in features.columns]
    if "radius" not in cols:
        raise ValueError("Neuron features parquet must contain 'radius' column")
    features = features[cols].drop_duplicates(subset="node_id").set_index("node_id")
    return features


def _build_cell_type_order(present_types: Iterable[str]) -> list[str]:
    from image_decoding.plot_utils import cell_type_order

    base = cell_type_order()
    present_set = set(present_types)

    reordered: list[str] = []
    l5_subtypes = ["L5_ET", "L5_IT", "L5_NP"]

    for ct in base:
        if ct == "L5_Exc":
            if ct in present_set and ct not in reordered:
                reordered.append(ct)
            for subtype in l5_subtypes:
                if subtype in present_set and subtype not in reordered:
                    reordered.append(subtype)
        elif ct in l5_subtypes:
            continue
        elif ct in present_set and ct not in reordered:
            reordered.append(ct)

    for ct in present_types:
        if ct not in reordered:
            reordered.append(ct)
    return reordered


def _load_node_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    ids = data.get("node_id", [])
    return {int(x) for x in ids}


def _load_means(
    metric_file: str,
    features: pd.DataFrame,
    *,
    exclude_ids: set[int] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(METRIC_DIR / metric_file, sep=" ")
    df = df.rename(columns={"max_mean_rate(Hz)": "Rate at preferred direction (Hz)"})

    if exclude_ids:
        df = df[~df["node_id"].isin(exclude_ids)]

    merged = df.merge(
        features,
        how="left",
        left_on="node_id",
        right_index=True,
        validate="many_to_one",
    )
    merged = merged.dropna(subset=["cell_type", "radius"])
    merged = merged[merged["radius"] <= CORE_RADIUS]

    merged["cell_type"] = merged["cell_type"].str.replace(" ", "_", regex=False)
    merged["cell_type"] = merged["cell_type"].str.replace("L1_Htr3a", "L1_Inh", regex=False)

    grouped = merged.groupby("cell_type")[list(METRICS)].mean(numeric_only=True)

    # Add aggregated L5_Exc
    l5_subset = merged[merged["cell_type"].isin({"L5_ET", "L5_IT", "L5_NP"})]
    if not l5_subset.empty:
        grouped.loc["L5_Exc", list(METRICS)] = l5_subset[list(METRICS)].mean(numeric_only=True)

    return grouped


def _compute_percent_frame(
    baseline_means: pd.DataFrame,
    experiment_means: pd.DataFrame,
    order: Sequence[str],
    key: str,
    display: str,
) -> pd.DataFrame:
    merged = baseline_means.join(
        experiment_means,
        how="inner",
        lsuffix="_baseline",
        rsuffix="_exp",
    )
    if merged.empty:
        return pd.DataFrame()

    result = merged.copy()
    for metric in METRICS:
        base_col = f"{metric}_baseline"
        exp_col = f"{metric}_exp"
        result[f"{metric}_pct"] = np.where(
            np.abs(result[base_col]) > FRACTION_EPS,
            (result[exp_col] - result[base_col]) / result[base_col] * 100.0,
            np.nan,
        )
        result[metric] = result[exp_col]

    columns_to_keep = [
        "cell_type",
        *METRICS,
        *(f"{metric}_pct" for metric in METRICS),
    ]
    result = result.reset_index()[columns_to_keep]
    result["experiment"] = key
    result["display"] = display

    result["cell_type"] = pd.Categorical(result["cell_type"], categories=order, ordered=True)
    result.sort_values(["cell_type", "experiment"], inplace=True)
    result = result.reset_index(drop=True)
    return result


def _plot_scatter_panel(
    ax: plt.Axes,
    pct_frames: dict[str, pd.DataFrame],
    title: str,
    measure: str = "DSI",
    ignore_types: set[str] | None = None,
    xlim: tuple[float, float] = (-50, 150),
    ylim: tuple[float, float] = (-50, 50),
) -> None:
    """Plot a single scatter panel showing % change for one cell type."""

    colors = {
        "pv_high_neg1000": "C3",
        "pv_low_neg1000": "C3",
        "sst_high_neg1000": "C0",
        "sst_low_neg1000": "C0",
        "vip_high_neg1000": "C2",
        "vip_low_neg1000": "C2",
    }

    markers = {
        "pv_high_neg1000": "o",
        "pv_low_neg1000": "x",
        "sst_high_neg1000": "o",
        "sst_low_neg1000": "x",
        "vip_high_neg1000": "o",
        "vip_low_neg1000": "x",
    }

    # Extract the cell type from title
    cell_type = title.split("(")[0].strip().replace(" ", "_")

    # First draw lines from origin to each point (lower zorder)
    for key, frame in pct_frames.items():
        if frame.empty:
            continue

        ct_data = frame[frame["cell_type"] == cell_type]
        if ct_data.empty:
            continue

        x = ct_data["Ave_Rate(Hz)_pct"].iloc[0]
        y = ct_data[f"{measure}_pct"].iloc[0]

        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        ax.plot([0, x], [0, y], c=colors.get(key, "gray"), alpha=0.2, linewidth=0.5, zorder=1)

    # Then draw scatter points on top (higher zorder)
    for key, frame in pct_frames.items():
        if frame.empty:
            continue

        ct_data = frame[frame["cell_type"] == cell_type]
        if ct_data.empty:
            continue

        x = ct_data["Ave_Rate(Hz)_pct"].iloc[0]
        y = ct_data[f"{measure}_pct"].iloc[0]

        if not (np.isfinite(x) and np.isfinite(y)):
            continue

        ax.scatter(x, y, c=colors.get(key, "gray"), marker=markers.get(key, "o"),
                  s=40, alpha=0.8, edgecolors="k", linewidth=0.6, zorder=3)

    ax.axhline(0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("% Δ rate", fontsize=7)
    ax.set_ylabel(f"% Δ {measure}", fontsize=7)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Keep aspect ratio equal
    if xlim[1] != xlim[0] and ylim[1] != ylim[0]:
        ax.set_aspect("equal", adjustable="box")


def main() -> None:
    print("Loading neuron features...")
    features = _load_features()

    print("Loading baseline metrics...")
    baseline_means = _load_means(
        BASELINE_SPEC.metric_file,
        features,
        exclude_ids=None,
    )

    # Get cell type order
    all_types = set(baseline_means.index)
    for spec in CELLTYPE_SUPPRESSION:
        exp_means = _load_means(spec.metric_file, features, exclude_ids=None)
        all_types.update(exp_means.index)

    order = _build_cell_type_order(all_types)

    print("Computing percent changes...")
    pct_frames = {}
    for spec in CELLTYPE_SUPPRESSION:
        target_ids = _load_node_ids(spec.target_nodes_file)
        exp_means = _load_means(
            spec.metric_file,
            features,
            exclude_ids=target_ids,  # Exclude targeted neurons
        )
        pct_frame = _compute_percent_frame(
            baseline_means, exp_means, order, spec.key, spec.display
        )
        pct_frames[spec.key] = pct_frame
        print(f"  {spec.display}: {len(pct_frame)} cell types")

    # Get baseline FR and selectivity for titles
    baseline_stats = {}
    for ct in order:
        if ct in baseline_means.index:
            baseline_stats[ct] = {
                "FR": baseline_means.loc[ct, "Ave_Rate(Hz)"],
                "DSI": baseline_means.loc[ct, "DSI"],
                "OSI": baseline_means.loc[ct, "OSI"],
            }

    # Compute axis limits excluding L5_NP
    print("Computing axis limits (excluding L5_NP)...")
    all_x_vals = []
    all_y_dsi_vals = []
    all_y_osi_vals = []

    for key, frame in pct_frames.items():
        # Exclude L5_NP
        non_np = frame[frame["cell_type"] != "L5_NP"]

        x_vals = non_np["Ave_Rate(Hz)_pct"].values
        dsi_vals = non_np["DSI_pct"].values
        osi_vals = non_np["OSI_pct"].values

        all_x_vals.extend(x_vals[np.isfinite(x_vals)])
        all_y_dsi_vals.extend(dsi_vals[np.isfinite(dsi_vals)])
        all_y_osi_vals.extend(osi_vals[np.isfinite(osi_vals)])

    # Add some margin
    if all_x_vals:
        x_min, x_max = np.min(all_x_vals), np.max(all_x_vals)
        x_range = x_max - x_min
        xlim = (x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    else:
        xlim = (-50, 150)

    if all_y_dsi_vals:
        y_min, y_max = np.min(all_y_dsi_vals), np.max(all_y_dsi_vals)
        y_range = y_max - y_min
        ylim_dsi = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    else:
        ylim_dsi = (-50, 50)

    if all_y_osi_vals:
        y_min, y_max = np.min(all_y_osi_vals), np.max(all_y_osi_vals)
        y_range = y_max - y_min
        ylim_osi = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    else:
        ylim_osi = (-50, 50)

    print(f"  X limits: {xlim}")
    print(f"  DSI Y limits: {ylim_dsi}")
    print(f"  OSI Y limits: {ylim_osi}")

    # Create figure - DSI
    print("Creating DSI figure...")
    n_types = len(order)
    n_cols = 5
    n_rows = ceil(n_types / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_types == 1 else axes

    for idx, ct in enumerate(order):
        if ct in baseline_stats:
            stats = baseline_stats[ct]
            title = f"{ct.replace('_', ' ')} (FR={stats['FR']:.1f} Hz, DSI={stats['DSI']:.2f})"
        else:
            title = ct.replace("_", " ")

        _plot_scatter_panel(axes[idx], pct_frames, title, measure="DSI", xlim=xlim, ylim=ylim_dsi)

    # Hide unused panels
    for idx in range(n_types, len(axes)):
        axes[idx].set_visible(False)

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C3", markersize=8, label="PV High", markeredgecolor="k"),
        Line2D([0], [0], marker="x", color="C3", markerfacecolor="C3", markersize=8, label="PV Low"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", markersize=8, label="SST High", markeredgecolor="k"),
        Line2D([0], [0], marker="x", color="C0", markerfacecolor="C0", markersize=8, label="SST Low"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C2", markersize=8, label="VIP High", markeredgecolor="k"),
        Line2D([0], [0], marker="x", color="C2", markerfacecolor="C2", markersize=8, label="VIP Low"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=6)

    plt.suptitle("Cell-type suppression: FR change vs DSI change (non-targeted core)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    output_dsi = OUTPUT_DIR / "celltype_suppression_non_targeted_fr_vs_selectivity_dsi_pct.png"
    fig.savefig(output_dsi, dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_dsi}")
    fig.savefig(output_dsi.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    # Create figure - OSI
    print("Creating OSI figure...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_types == 1 else axes

    for idx, ct in enumerate(order):
        if ct in baseline_stats:
            stats = baseline_stats[ct]
            title = f"{ct.replace('_', ' ')} (FR={stats['FR']:.1f} Hz, OSI={stats['OSI']:.2f})"
        else:
            title = ct.replace("_", " ")

        _plot_scatter_panel(axes[idx], pct_frames, title, measure="OSI", xlim=xlim, ylim=ylim_osi)

    for idx in range(n_types, len(axes)):
        axes[idx].set_visible(False)

    fig.legend(handles=legend_elements, loc="upper right", fontsize=8, ncol=6)

    plt.suptitle("Cell-type suppression: FR change vs OSI change (non-targeted core)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    output_osi = OUTPUT_DIR / "celltype_suppression_non_targeted_fr_vs_selectivity_osi_pct.png"
    fig.savefig(output_osi, dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_osi}")
    fig.savefig(output_osi.with_suffix(".svg"), bbox_inches="tight")
    plt.close()

    print(f"\n✓ All figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
