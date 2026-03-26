#!/usr/bin/env python3
"""Percent change scatter plots (non-targeted core neurons) for outgoing-weight perturbations."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

BASE_DIR = PROJECT_ROOT / "core_nll_0"
METRIC_DIR = BASE_DIR / "metrics"
OUTPUT_DIR = BASE_DIR / "figures" / "perturbation_fr_selectivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = (
    PROJECT_ROOT / "cell_categorization" / "core_nll_0_neuron_features.parquet"
)

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
    target_nodes_file=Path(),  # unused
)

EXC_OUTGOING: Sequence[DatasetSpec] = (
    DatasetSpec(
        key="exc_high_neg1000",
        display="High outgoing",
        metric_file="OSI_DSI_DF_exc_high_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "high_outgoing_exc_nodes.json",
    ),
    DatasetSpec(
        key="exc_low_neg1000",
        display="Low outgoing",
        metric_file="OSI_DSI_DF_exc_low_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "low_outgoing_exc_nodes.json",
    ),
)

INH_OUTGOING: Sequence[DatasetSpec] = (
    DatasetSpec(
        key="inh_high_neg1000",
        display="High outgoing",
        metric_file="OSI_DSI_DF_inh_high_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "high_outgoing_inh_nodes.json",
    ),
    DatasetSpec(
        key="inh_low_neg1000",
        display="Low outgoing",
        metric_file="OSI_DSI_DF_inh_low_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "low_outgoing_inh_nodes.json",
    ),
)

EXC_OUTGOING_CORE: Sequence[DatasetSpec] = (
    DatasetSpec(
        key="exc_high_core_neg1000",
        display="High outgoing core",
        metric_file="OSI_DSI_DF_exc_high_core_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "high_outgoing_exc_core_nodes.json",
    ),
    DatasetSpec(
        key="exc_low_core_neg1000",
        display="Low outgoing core",
        metric_file="OSI_DSI_DF_exc_low_core_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "low_outgoing_exc_core_nodes.json",
    ),
)

INH_OUTGOING_CORE: Sequence[DatasetSpec] = (
    DatasetSpec(
        key="inh_high_core_neg1000",
        display="High outgoing core",
        metric_file="OSI_DSI_DF_inh_high_core_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "high_outgoing_inh_core_nodes.json",
    ),
    DatasetSpec(
        key="inh_low_core_neg1000",
        display="Low outgoing core",
        metric_file="OSI_DSI_DF_inh_low_core_neg1000.csv",
        target_nodes_file=BASE_DIR / "node_sets" / "low_outgoing_inh_core_nodes.json",
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
    cols = [
        col for col in ("node_id", "cell_type", "radius") if col in features.columns
    ]
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
    merged["cell_type"] = merged["cell_type"].str.replace(
        "L1_Htr3a", "L1_Inh", regex=False
    )

    grouped = merged.groupby("cell_type")[list(METRICS)].mean(numeric_only=True)

    l5_subset = merged[merged["cell_type"].isin({"L5_ET", "L5_IT", "L5_NP"})]
    if not l5_subset.empty:
        grouped.loc["L5_Exc", list(METRICS)] = l5_subset[list(METRICS)].mean(
            numeric_only=True
        )

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
        return pd.DataFrame(
            columns=[
                "cell_type",
                "experiment",
                "display",
                *METRICS,
                *(f"{metric}_pct" for metric in METRICS),
            ]
        )

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

    result["cell_type"] = pd.Categorical(
        result["cell_type"], categories=order, ordered=True
    )
    result.sort_values(["cell_type", "experiment"], inplace=True)
    result = result.reset_index(drop=True)
    return result


def _collect_percent_values(
    exp_df: pd.DataFrame,
    measure: str,
    ignore_types: set[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if exp_df is None or exp_df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    df = exp_df
    if ignore_types:
        df = df[~df["cell_type"].isin(ignore_types)]

    if df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    x = df["Ave_Rate(Hz)_pct"].to_numpy(dtype=float)
    y_col = f"{measure}_pct"
    if y_col not in df:
        return np.array([], dtype=float), np.array([], dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _collect_rate_percent_values(
    exp_df: pd.DataFrame,
    ignore_types: set[str] | None = None,
) -> np.ndarray:
    if exp_df is None or exp_df.empty:
        return np.array([], dtype=float)

    df = exp_df
    if ignore_types:
        df = df[~df["cell_type"].isin(ignore_types)]

    if df.empty or "Ave_Rate(Hz)_pct" not in df:
        return np.array([], dtype=float)

    arr = df["Ave_Rate(Hz)_pct"].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def _ensure_zero_in_range(limits: Tuple[float, float]) -> Tuple[float, float]:
    low, high = limits
    if low > 0:
        low = 0.0
    if high < 0:
        high = 0.0
    return low, high


def _axis_limits(
    values: Sequence[float], default: Tuple[float, float]
) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default

    v_min = float(arr.min())
    v_max = float(arr.max())

    if v_min == v_max:
        span = abs(v_min) if abs(v_min) > 0 else 1.0
        pad = 0.1 * span
        v_min -= pad
        v_max += pad
    else:
        pad = 0.05 * (v_max - v_min)
        v_min -= pad
        v_max += pad

    return v_min, v_max


def _plot_group(
    title: str,
    baseline_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    order: Sequence[str],
    output_path: Path,
    measure: str,
    measure_label: str,
) -> None:
    if not order:
        print(f"No baseline data available for {title}")
        return
    if exp_df.empty:
        print(f"No data available for {title}")
        return

    n_types = len(order)
    cols = 5
    rows = ceil(n_types / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes_flat = axes.flatten()

    experiments = list(exp_df["experiment"].unique())
    palette = {
        key: color
        for key, color in zip(
            experiments, sns.color_palette("colorblind", len(experiments))
        )
    }
    display_lookup = (
        exp_df.drop_duplicates("experiment")
        .set_index("experiment")["display"]
        .to_dict()
    )
    baseline_color = "#4d4d4d"

    base_lookup = baseline_df.set_index("cell_type")
    exp_lookup = exp_df.set_index(["cell_type", "experiment"])

    handles: list[Line2D] = []
    labels: list[str] = []
    legend_added: set[str] = set()

    axes_info: list[tuple[plt.Axes, int]] = []

    for idx, cell_type in enumerate(order):
        ax = axes_flat[idx]
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)

        if cell_type not in base_lookup.index:
            ax.axis("off")
            continue

        base_row = base_lookup.loc[cell_type]
        base_fr = float(base_row["Ave_Rate(Hz)"])
        base_measure = float(base_row[measure])

        pretty = cell_type.replace("_", " ")
        ax.set_title(
            f"{pretty} (FR={_format_sigfig(base_fr)} Hz, {measure_label}={_format_sigfig(base_measure)})",
            fontsize=9,
        )

        ax.scatter(0.0, 0.0, marker="x", s=35, color=baseline_color, linewidths=1.4)
        cell_has_data = False

        for experiment in experiments:
            key = (cell_type, experiment)
            if key not in exp_lookup.index:
                continue
            row = exp_lookup.loc[key]
            x_val = float(row.get("Ave_Rate(Hz)_pct", np.nan))
            y_val = float(row.get(f"{measure}_pct", np.nan))
            if not np.isfinite(x_val) or not np.isfinite(y_val):
                continue
            ax.scatter(
                x_val,
                y_val,
                s=40,
                color=palette[experiment],
                edgecolor="black",
                linewidths=0.6,
            )
            ax.plot(
                [0.0, x_val],
                [0.0, y_val],
                color=palette[experiment],
                linewidth=1.0,
                alpha=0.7,
            )
            cell_has_data = True
            if experiment not in legend_added:
                handles.append(
                    Line2D(
                        [],
                        [],
                        marker="o",
                        linestyle="",
                        color=palette[experiment],
                        markeredgecolor="black",
                        markeredgewidth=0.6,
                        label=display_lookup.get(experiment, experiment),
                    )
                )
                legend_added.add(experiment)

        if cell_has_data and "baseline" not in legend_added:
            handles.insert(
                0,
                Line2D(
                    [],
                    [],
                    marker="x",
                    linestyle="",
                    color=baseline_color,
                    markeredgewidth=1.4,
                    label="Baseline",
                ),
            )
            legend_added.add("baseline")

        axes_info.append((ax, idx))

    for extra_ax in axes_flat[n_types:]:
        extra_ax.axis("off")

    rate_vals = _collect_rate_percent_values(exp_df, ignore_types={"L5_NP"})
    x_limits = _ensure_zero_in_range(
        _axis_limits(rate_vals, default=(-20.0, 20.0))
        if rate_vals.size
        else (-20.0, 20.0)
    )
    _, y_vals = _collect_percent_values(exp_df, measure, ignore_types={"L5_NP"})
    y_limits = _ensure_zero_in_range(
        _axis_limits(y_vals, default=(-20.0, 20.0)) if y_vals.size else (-20.0, 20.0)
    )

    for ax, idx in axes_info:
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        if x_limits[1] != x_limits[0] and y_limits[1] != y_limits[0]:
            ax.set_aspect("equal", adjustable="box")
        if idx % cols == 0:
            ax.set_ylabel(f"% Δ {measure_label}")
        else:
            ax.set_ylabel("")
        if idx // cols == rows - 1:
            ax.set_xlabel("% Δ rate")
        else:
            ax.set_xlabel("")

    fig.suptitle(
        f"{title} — {measure_label} (% change, non-targeted core)", fontsize=14
    )
    if handles:
        fig.legend(
            handles,
            [h.get_label() for h in handles],
            loc="upper center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, 0.98),
        )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
    fig.savefig(output_path, dpi=200)
    try:
        fig.savefig(output_path.with_suffix(".svg"))
    except Exception:
        pass
    plt.close(fig)


def main() -> None:
    features = _load_features()
    baseline_means_global = _load_means(
        BASELINE_SPEC.metric_file, features, exclude_ids=None
    )
    order = _build_cell_type_order(baseline_means_global.index)
    baseline_df = baseline_means_global.reset_index()

    def process_group(specs: Sequence[DatasetSpec], prefix: str) -> None:
        frames: List[pd.DataFrame] = []
        for spec in specs:
            target_ids = _load_node_ids(spec.target_nodes_file)
            base_filtered = _load_means(
                BASELINE_SPEC.metric_file, features, exclude_ids=target_ids
            )
            exp_means = _load_means(spec.metric_file, features, exclude_ids=target_ids)
            frame = _compute_percent_frame(
                base_filtered, exp_means, order, spec.key, spec.display
            )
            frames.append(frame)

        if not frames:
            print(f"No data for group {prefix}")
            return
        combined = pd.concat(frames, ignore_index=True)
        for measure, label, suffix in (
            ("OSI", "OSI", "osi_pct"),
            ("DSI", "DSI", "dsi_pct"),
        ):
            _plot_group(
                title=f"{prefix.capitalize()} outgoing perturbations",
                baseline_df=baseline_df,
                exp_df=combined,
                order=order,
                output_path=OUTPUT_DIR
                / f"{prefix}_outgoing_non_targeted_fr_vs_selectivity_{suffix}.png",
                measure=measure,
                measure_label=label,
            )

    process_group(EXC_OUTGOING, "exc")
    process_group(INH_OUTGOING, "inh")
    process_group(EXC_OUTGOING_CORE, "exc_core")
    process_group(INH_OUTGOING_CORE, "inh_core")


if __name__ == "__main__":
    main()
