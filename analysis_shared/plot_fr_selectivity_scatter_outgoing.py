#!/usr/bin/env python3
"""Percent change scatter plots for outgoing-weight perturbations."""

from __future__ import annotations

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

from image_decoding.plot_utils import cell_type_order


BASE_DIR = PROJECT_ROOT / "core_nll_0"
METRIC_DIR = BASE_DIR / "metrics"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "paper" / "figure5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = (
    PROJECT_ROOT / "cell_categorization" / "core_nll_0_neuron_features.parquet"
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display: str
    metric_file: str


BASELINE_SPEC = DatasetSpec(
    key="bio_trained",
    display="Bio-trained",
    metric_file="OSI_DSI_DF_bio_trained.csv",
)

EXC_OUTGOING: Sequence[DatasetSpec] = (
    DatasetSpec("exc_high_neg1000", "High outgoing", "OSI_DSI_DF_exc_high_neg1000.csv"),
    DatasetSpec("exc_low_neg1000", "Low outgoing", "OSI_DSI_DF_exc_low_neg1000.csv"),
)

INH_OUTGOING: Sequence[DatasetSpec] = (
    DatasetSpec("inh_high_neg1000", "High outgoing", "OSI_DSI_DF_inh_high_neg1000.csv"),
    DatasetSpec("inh_low_neg1000", "Low outgoing", "OSI_DSI_DF_inh_low_neg1000.csv"),
)

EXC_OUTGOING_CORE: Sequence[DatasetSpec] = (
    DatasetSpec(
        "exc_high_core_neg1000",
        "High outgoing core",
        "OSI_DSI_DF_exc_high_core_neg1000.csv",
    ),
    DatasetSpec(
        "exc_low_core_neg1000",
        "Low outgoing core",
        "OSI_DSI_DF_exc_low_core_neg1000.csv",
    ),
)

INH_OUTGOING_CORE: Sequence[DatasetSpec] = (
    DatasetSpec(
        "inh_high_core_neg1000",
        "High outgoing core",
        "OSI_DSI_DF_inh_high_core_neg1000.csv",
    ),
    DatasetSpec(
        "inh_low_core_neg1000",
        "Low outgoing core",
        "OSI_DSI_DF_inh_low_core_neg1000.csv",
    ),
)

PLOT_SPECS = (
    ("OSI", "OSI", (None, None), "osi_pct"),
    ("DSI", "DSI", (None, None), "dsi_pct"),
)

METRICS = ("Ave_Rate(Hz)", "OSI", "DSI")


def _format_sigfig(value: float, sig: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    if value == 0:
        return "0"
    return f"{value:.{sig}g}"


def _load_features() -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH)
    features = features.drop_duplicates(subset="node_id")
    return features[["node_id", "cell_type"]]


def _build_cell_type_order(present_types: Iterable[str]) -> list[str]:
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


def _load_dataset(spec: DatasetSpec, features: pd.DataFrame) -> pd.DataFrame:
    metric_path = METRIC_DIR / spec.metric_file
    if not metric_path.exists():
        raise FileNotFoundError(metric_path)
    df = pd.read_csv(metric_path, sep=" ")
    df = df.rename(columns={"max_mean_rate(Hz)": "Rate at preferred direction (Hz)"})
    df = df.merge(features, how="left", on="node_id", validate="many_to_one")
    df = df.dropna(subset=["cell_type"])
    df["cell_type"] = df["cell_type"].str.replace(" ", "_", regex=False)
    df["cell_type"] = df["cell_type"].str.replace("L1_Htr3a", "L1_Inh", regex=False)
    return df


def _cell_type_means(spec: DatasetSpec, features: pd.DataFrame) -> pd.DataFrame:
    df = _load_dataset(spec, features)
    grouped = df.groupby("cell_type")[list(METRICS)].mean(numeric_only=True)
    l5_subset = df[df["cell_type"].isin({"L5_ET", "L5_IT", "L5_NP"})]
    if not l5_subset.empty:
        grouped.loc["L5_Exc", list(METRICS)] = l5_subset[list(METRICS)].mean(
            numeric_only=True
        )
    return grouped


def _prepare_group_frames(
    baseline_means: pd.DataFrame,
    features: pd.DataFrame,
    experiments: Sequence[DatasetSpec],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    records: list[dict[str, float | str]] = []
    all_types: set[str] = set(baseline_means.index)

    for spec in experiments:
        try:
            means = _cell_type_means(spec, features)
        except FileNotFoundError:
            continue
        all_types.update(means.index)
        for cell_type, row in means.iterrows():
            records.append(
                {
                    "cell_type": cell_type,
                    "experiment": spec.key,
                    "display": spec.display,
                    "Ave_Rate(Hz)": row["Ave_Rate(Hz)"],
                    "OSI": row["OSI"],
                    "DSI": row.get("DSI"),
                }
            )

    exp_df = pd.DataFrame.from_records(records)
    if not exp_df.empty:
        exp_df = exp_df.dropna(subset=list(METRICS), how="all")

    present_types = [ct for ct in baseline_means.index if ct in all_types]
    present_types.extend(ct for ct in all_types if ct not in present_types)
    order = _build_cell_type_order(present_types)

    baseline_df = baseline_means.reset_index()
    baseline_df = baseline_df[baseline_df["cell_type"].isin(order)]
    baseline_df = baseline_df.dropna(subset=list(METRICS), how="all")
    baseline_df["cell_type"] = pd.Categorical(
        baseline_df["cell_type"], categories=order, ordered=True
    )
    baseline_df.sort_values("cell_type", inplace=True)
    baseline_df["cell_type"] = baseline_df["cell_type"].cat.remove_unused_categories()
    order = list(baseline_df["cell_type"].cat.categories)

    for metric in METRICS:
        baseline_df[f"{metric}_pct"] = 0.0

    if baseline_df.empty:
        return baseline_df, exp_df, []

    if not exp_df.empty:
        exp_df = exp_df[exp_df["cell_type"].isin(order)]
        exp_df["cell_type"] = pd.Categorical(
            exp_df["cell_type"], categories=order, ordered=True
        )
        exp_df.sort_values(["cell_type", "experiment"], inplace=True)

        base_lookup = baseline_df.set_index("cell_type")
        for metric in METRICS:
            base_vals = base_lookup[metric].reindex(exp_df["cell_type"]).to_numpy()
            exp_vals = exp_df[metric].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = np.where(
                    np.abs(base_vals) > 1e-9,
                    (exp_vals - base_vals) / base_vals * 100.0,
                    np.nan,
                )
            exp_df[f"{metric}_pct"] = pct

    return baseline_df, exp_df, order


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
    values: Sequence[float], *, default: Tuple[float, float]
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
    all_x_vals: list[float] = []
    all_y_vals: list[float] = []

    fr_pct_col = "Ave_Rate(Hz)_pct"
    measure_pct_col = f"{measure}_pct"

    for idx, cell_type in enumerate(order):
        ax = axes_flat[idx]
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)

        if cell_type not in base_lookup.index:
            ax.axis("off")
            continue

        base_row = base_lookup.loc[cell_type]
        base_fr = float(base_row["Ave_Rate(Hz)"])
        base_measure = float(base_row[measure])
        if not np.isfinite(base_fr) or not np.isfinite(base_measure):
            ax.axis("off")
            continue

        pretty = cell_type.replace("_", " ")
        ax.set_title(
            f"{pretty} (FR={_format_sigfig(base_fr)} Hz, {measure_label}={_format_sigfig(base_measure)})",
            fontsize=9,
        )

        ax.scatter(0.0, 0.0, marker="x", s=35, color=baseline_color, linewidths=1.4)
        x_values = [0.0]
        y_values = [0.0]
        cell_has_data = False

        for experiment in experiments:
            key = (cell_type, experiment)
            if key not in exp_lookup.index:
                continue
            row = exp_lookup.loc[key]
            x_val = float(row.get(fr_pct_col, np.nan))
            y_val = float(row.get(measure_pct_col, np.nan))
            if not np.isfinite(x_val) or not np.isfinite(y_val):
                continue
            ax.scatter(
                x_val,
                y_val,
                s=40,
                color=palette[experiment],
                edgecolor="black",
                linewidths=0.6,
                label=display_lookup.get(experiment, experiment),
            )
            ax.plot(
                [0.0, x_val],
                [0.0, y_val],
                color=palette[experiment],
                linewidth=1.0,
                alpha=0.7,
            )
            cell_has_data = True
            x_values.append(x_val)
            y_values.append(y_val)
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
                    )
                )
                labels.append(display_lookup.get(experiment, experiment))
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
                ),
            )
            labels.insert(0, "Baseline")
            legend_added.add("baseline")

        axes_info.append((ax, idx))
        all_x_vals.extend(x_values)
        all_y_vals.extend(y_values)

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

    fig.suptitle(f"{title} — {measure_label} (% change)", fontsize=14)
    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.98),
        )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    try:
        fig.savefig(output_path.with_suffix(".svg"))
    except Exception:
        pass
    plt.close(fig)


def main() -> None:
    features = _load_features()
    baseline_means = _cell_type_means(BASELINE_SPEC, features)

    group_specs = [
        ("exc_outgoing", "Excitatory outgoing perturbations", EXC_OUTGOING),
        ("inh_outgoing", "Inhibitory outgoing perturbations", INH_OUTGOING),
        (
            "exc_outgoing_core",
            "Excitatory outgoing perturbations (core)",
            EXC_OUTGOING_CORE,
        ),
        (
            "inh_outgoing_core",
            "Inhibitory outgoing perturbations (core)",
            INH_OUTGOING_CORE,
        ),
    ]

    prepared_groups: list[tuple[str, str, pd.DataFrame, pd.DataFrame, List[str]]] = []
    for key, title, specs in group_specs:
        base_df, exp_df, order = _prepare_group_frames(baseline_means, features, specs)
        prepared_groups.append((key, title, base_df, exp_df, order))

    for measure, label, _bounds, suffix in PLOT_SPECS:
        for key, title, base_df, exp_df, order in prepared_groups:
            if base_df.empty or exp_df.empty or not order:
                continue
            _plot_group(
                title=title,
                baseline_df=base_df,
                exp_df=exp_df,
                order=order,
                output_path=OUTPUT_DIR / f"{key}_fr_vs_selectivity_{suffix}.png",
                measure=measure,
                measure_label=label,
            )


if __name__ == "__main__":
    main()
