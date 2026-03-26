#!/usr/bin/env python3
"""Plot percent change in firing rate vs percent change in image selectivity per cell type."""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analysis_shared.plot_fr_selectivity_scatter import build_cell_type_order

BASE_DIR = PROJECT_ROOT / "core_nll_0"
CACHE_DIR_TEMPLATE = "cached_rates_{scenario}"
SELECTIVITY_CSV = PROJECT_ROOT / "image_decoding" / "summary" / "selectivity_model_by_type.csv"
OUTPUT_DIR = BASE_DIR / "figures" / "perturbation_fr_selectivity"

INH_SCENARIOS = OrderedDict(
    {
        "inh_selective_pos100": "Selective +100 pA",
        "inh_selective_neg100": "Selective -100 pA",
        "inh_nonselective_matched_pos100": "Non-selective +100 pA",
        "inh_nonselective_matched_neg100": "Non-selective -100 pA",
    }
)

EXC_SCENARIOS = OrderedDict(
    {
        "exc_selective_matched_pos100": "Selective +100 pA",
        "exc_selective_matched_neg100": "Selective -100 pA",
        "exc_nonselective_pos100": "Non-selective +100 pA",
        "exc_nonselective_neg100": "Non-selective -100 pA",
    }
)

CLASS_COLORS = {
    "Exc": "#1f77b4",
    "PV": "#d62728",
    "SST": "#2ca02c",
    "VIP": "#9467bd",
    "Inh": "#7f7f7f",
    "L1": "#bcbd22",
}


def _cell_class(cell_type: str) -> str:
    if cell_type.startswith("L1"):
        return "L1"
    if "Exc" in cell_type or cell_type in {"L5_ET", "L5_IT", "L5_NP", "L5_Exc"}:
        return "Exc"
    if "PV" in cell_type:
        return "PV"
    if "SST" in cell_type:
        return "SST"
    if "VIP" in cell_type:
        return "VIP"
    return "Inh"


def load_mean_rates(scenario: str) -> pd.Series:
    cache_dir = BASE_DIR / CACHE_DIR_TEMPLATE.format(scenario=scenario)
    rates = np.load(cache_dir / "rates_core.npy")  # reps x cells x images
    meta = pd.read_parquet(cache_dir / "meta_core.parquet")
    mean_rate_per_cell = rates.mean(axis=(0, 2))
    df = pd.DataFrame({"cell_type": meta["cell_type"], "mean_rate": mean_rate_per_cell})
    series = df.groupby("cell_type", as_index=True)["mean_rate"].mean()
    return series


def load_selectivity_means() -> pd.DataFrame:
    df = pd.read_csv(SELECTIVITY_CSV)
    return df.pivot(index="cell_type", columns="network_type", values="mean_image_selectivity")


def compute_percent_delta(
    scenarios: Mapping[str, str],
    rate_table: Mapping[str, pd.Series],
    selectivity_table: pd.DataFrame,
    baseline_key: str = "bio_trained",
) -> pd.DataFrame:
    baseline_rates = rate_table[baseline_key]
    baseline_sel = selectivity_table[baseline_key]

    rows = []
    for scenario, label in scenarios.items():
        rate_series = rate_table[scenario]
        sel_series = selectivity_table[scenario]

        shared = baseline_rates.index.intersection(rate_series.index).intersection(baseline_sel.dropna().index)
        if sel_series is not None:
            shared = shared.intersection(sel_series.dropna().index)
        if shared.empty:
            continue

        rate_pct = 100.0 * (rate_series.loc[shared] - baseline_rates.loc[shared]) / baseline_rates.loc[shared]
        sel_pct = 100.0 * (sel_series.loc[shared] - baseline_sel.loc[shared]) / baseline_sel.loc[shared]

        df = pd.DataFrame(
            {
                "scenario": scenario,
                "scenario_label": label,
                "cell_type": shared,
                "delta_rate_pct": rate_pct.values,
                "delta_selectivity_pct": sel_pct.values,
                "baseline_rate": baseline_rates.loc[shared].values,
                "baseline_selectivity": baseline_sel.loc[shared].values,
            }
        )
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=[
            "scenario",
            "scenario_label",
            "cell_type",
            "delta_rate_pct",
            "delta_selectivity_pct",
            "baseline_rate",
            "baseline_selectivity",
        ])
    return pd.concat(rows, ignore_index=True)


def plot_panels(title: str, deltas: pd.DataFrame, order: Sequence[str], out_path: Path) -> None:
    if deltas.empty:
        return

    cols = 5
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0), sharex=True, sharey=True)
    axes = axes.flatten()

    mask = (deltas["cell_type"] != "L5_NP") & np.isfinite(deltas["delta_rate_pct"]) & np.isfinite(deltas["delta_selectivity_pct"])
    data_for_limits = deltas.loc[mask] if mask.any() else deltas
    max_abs = np.nanmax(np.abs(np.concatenate([
        data_for_limits["delta_rate_pct"].to_numpy(),
        data_for_limits["delta_selectivity_pct"].to_numpy(),
    ])))
    limit = 5.0 if not np.isfinite(max_abs) or max_abs == 0 else max_abs * 1.05

    scenarios = deltas["scenario_label"].unique()
    palette = dict(zip(scenarios, sns.color_palette("colorblind", len(scenarios))))

    for idx, cell_type in enumerate(order):
        ax = axes[idx]
        sub = deltas[deltas["cell_type"] == cell_type]
        if sub.empty:
            ax.axis("off")
            continue

        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
        ax.scatter(0, 0, marker="x", color="#444444", s=35)

        for scenario_label, group in sub.groupby("scenario_label"):
            color = palette[scenario_label]
            ax.scatter(
                group["delta_rate_pct"],
                group["delta_selectivity_pct"],
                s=40,
                color=color,
                edgecolor="black",
                linewidths=0.5,
                label=scenario_label,
            )

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        base_rate = sub.iloc[0]["baseline_rate"]
        base_sel = sub.iloc[0]["baseline_selectivity"]
        title = f"{cell_type.replace('_', ' ')} (FR={base_rate:.2f} Hz, Sel={base_sel:.2f})"
        ax.set_title(title, fontsize=10)

        if idx % cols == 0:
            ax.set_ylabel("% Δ Image Selectivity")
        else:
            ax.set_ylabel("")
        if idx // cols == rows - 1:
            ax.set_xlabel("% Δ Mean Rate")
        else:
            ax.set_xlabel("")

    for ax in axes[len(order):]:
        ax.axis("off")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=palette[label], markeredgecolor="black", markeredgewidth=0.5, label=label)
        for label in scenarios
    ]
    handles.insert(0, plt.Line2D([], [], marker="x", linestyle="", color="#444444", label="Baseline"))
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.98))

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    selectivity_table = load_selectivity_means()
    rate_table = {scenario: load_mean_rates(scenario) for scenario in ["bio_trained", *INH_SCENARIOS.keys(), *EXC_SCENARIOS.keys()]}

    all_cell_types = rate_table["bio_trained"].index
    order = build_cell_type_order(all_cell_types)

    inh_deltas = compute_percent_delta(INH_SCENARIOS, rate_table, selectivity_table)
    plot_panels(
        "Inhibitory perturbations",
        inh_deltas,
        order,
        OUTPUT_DIR / "inh_delta_fr_vs_selectivity_pct.png",
    )

    exc_deltas = compute_percent_delta(EXC_SCENARIOS, rate_table, selectivity_table)
    plot_panels(
        "Excitatory perturbations",
        exc_deltas,
        order,
        OUTPUT_DIR / "exc_delta_fr_vs_selectivity_pct.png",
    )


if __name__ == "__main__":
    main()
