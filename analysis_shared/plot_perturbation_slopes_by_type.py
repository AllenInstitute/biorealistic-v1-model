#!/usr/bin/env python3
"""Plot perturbation deltas with experiment type on the x-axis."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analysis_shared.osi_boxplot_utils import DatasetSpec, load_osi_dataset
from image_decoding.plot_utils import cell_type_order

BASE_DIR = PROJECT_ROOT / "core_nll_0"
OUTPUT_PATH = BASE_DIR / "figures" / "perturbation_slopes" / "matched_deltas_by_type.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

BASELINE_SPEC = DatasetSpec(
    label="Bio-trained Model",
    basedir=BASE_DIR,
    metric_file="OSI_DSI_DF_bio_trained.csv",
)

# Mapping: group -> (display label, DatasetSpec, amplitude pA)
INH_EXPERIMENTS: Dict[str, Tuple[str, DatasetSpec, int]] = {
    "Selective +100": (
        "Selective +100",
        DatasetSpec(
            label="Inh-selective pos 100",
            basedir=BASE_DIR,
            metric_file="OSI_DSI_DF_inh_selective_pos100.csv",
        ),
        +100,
    ),
    "Selective -100": (
        "Selective -100",
        DatasetSpec(
            label="Inh-selective neg 100",
            basedir=BASE_DIR,
            metric_file="OSI_DSI_DF_inh_selective_neg100.csv",
        ),
        -100,
    ),
    "Non-selective +100": (
        "Non-selective +100",
        DatasetSpec(
            label="Inh-nonselective matched pos 100",
            basedir=BASE_DIR,
            metric_file="OSI_DSI_DF_inh_nonselective_matched_pos100.csv",
        ),
        +100,
    ),
    "Non-selective -100": (
        "Non-selective -100",
        DatasetSpec(
            label="Inh-nonselective matched neg 100",
            basedir=BASE_DIR,
            metric_file="OSI_DSI_DF_inh_nonselective_matched_neg100.csv",
        ),
        -100,
    ),
}

EXC_EXPERIMENTS: Dict[str, Tuple[str, DatasetSpec, int]] = {
    "Selective +100": (
        "Selective +100",
        DatasetSpec(
            label="Exc-selective matched pos 100",
            basedir=BASE_DIR,
            metric_file="OSI_DSI_DF_exc_selective_matched_pos100.csv",
        ),
        +100,
    ),
    "Selective -100": (
        "Selective -100",
        DatasetSpec(
            label="Exc-selective matched neg 100",
            basedir=BASE_DIR,
            metric_file="OSI_DSI_DF_exc_selective_matched_neg100.csv",
        ),
        -100,
    ),
    "Non-selective +100": (
        "Non-selective +100",
        DatasetSpec(
            label="Exc-nonselective pos 100",
            basedir=BASE_DIR,
            metric_file="OSI_DSI_DF_exc_nonselective_pos100.csv",
        ),
        +100,
    ),
    "Non-selective -100": (
        "Non-selective -100",
        DatasetSpec(
            label="Exc-nonselective neg 100",
            basedir=BASE_DIR,
            metric_file="OSI_DSI_DF_exc_nonselective_neg100.csv",
        ),
        -100,
    ),
}

MEASURES = [
    "Spont_Rate(Hz)",
    "Ave_Rate(Hz)",
    "Rate at preferred direction (Hz)",
    "OSI",
    "DSI",
]

MEASURE_TITLES = {
    "Spont_Rate(Hz)": "Spont rate",
    "Ave_Rate(Hz)": "Mean rate",
    "Rate at preferred direction (Hz)": "Pref rate",
    "OSI": "OSI",
    "DSI": "DSI",
}


def _mean_by_cell_type(spec: DatasetSpec) -> pd.DataFrame:
    df = load_osi_dataset(spec)
    return df.groupby("cell_type").mean(numeric_only=True)


def _compute_deltas(
    baseline: pd.DataFrame,
    experiments: Dict[str, Tuple[str, DatasetSpec, int]],
) -> pd.DataFrame:
    order = cell_type_order()
    rows = []
    for key, (_, spec, amp) in experiments.items():
        try:
            exp_means = _mean_by_cell_type(spec)
        except FileNotFoundError:
            continue
        shared = baseline.index.intersection(exp_means.index)
        for cell_type in shared:
            for measure in MEASURES:
                base_val = baseline.loc[cell_type, measure]
                exp_val = exp_means.loc[cell_type, measure]
                rows.append(
                    {
                        "experiment": key,
                        "cell_type": cell_type,
                        "measure": measure,
                        "delta": exp_val - base_val,
                        "amplitude": amp,
                    }
                )
    df = pd.DataFrame(rows)
    df["cell_type"] = pd.Categorical(df["cell_type"], categories=order, ordered=True)
    df.sort_values(["cell_type", "experiment"], inplace=True)
    return df


def _pivot_matrix(df: pd.DataFrame, measure: str, experiments: Sequence[str]) -> pd.DataFrame:
    sub = df[df["measure"] == measure]
    pivot = sub.pivot(index="cell_type", columns="experiment", values="delta")
    pivot = pivot.loc[~pivot.index.isna()]
    pivot = pivot[[exp for exp in experiments if exp in pivot.columns]]
    pivot = pivot.dropna(how="all")
    return pivot


def plot_slopes() -> None:
    baseline_means = _mean_by_cell_type(BASELINE_SPEC)

    inh_df = _compute_deltas(baseline_means, INH_EXPERIMENTS)
    exc_df = _compute_deltas(baseline_means, EXC_EXPERIMENTS)

    inh_order = list(INH_EXPERIMENTS.keys())
    exc_order = list(EXC_EXPERIMENTS.keys())

    n_measures = len(MEASURES)
    fig, axes = plt.subplots(2, n_measures, figsize=(2.6 * n_measures, 6.5), squeeze=False)

    for col, measure in enumerate(MEASURES):
        inh_matrix = _pivot_matrix(inh_df, measure, inh_order)
        exc_matrix = _pivot_matrix(exc_df, measure, exc_order)

        inh_vmax = np.nanmax(np.abs(inh_matrix.values)) if not inh_matrix.empty else 1.0
        exc_vmax = np.nanmax(np.abs(exc_matrix.values)) if not exc_matrix.empty else 1.0
        inh_vmax = inh_vmax if inh_vmax > 0 else 1.0
        exc_vmax = exc_vmax if exc_vmax > 0 else 1.0

        ax_inh = axes[0, col]
        if inh_matrix.empty:
            ax_inh.axis("off")
            ax_inh.set_title("Inhibitory\n(no data)")
        else:
            hm = sns.heatmap(
                inh_matrix,
                ax=ax_inh,
                cmap="coolwarm",
                center=0.0,
                vmin=-inh_vmax,
                vmax=inh_vmax,
                linewidths=0.5,
                linecolor="white",
                cbar=True,
                cbar_kws={"label": "Δ value" if col == n_measures - 1 else None},
            )
            if col < n_measures - 1 and hm.collections:
                hm.collections[-1].colorbar.remove()
            ax_inh.set_title(MEASURE_TITLES[measure])
            ax_inh.set_xlabel("")
            ax_inh.set_ylabel("Inhibitory" if col == 0 else "")
            ax_inh.tick_params(axis="x", rotation=35)

        ax_exc = axes[1, col]
        if exc_matrix.empty:
            ax_exc.axis("off")
            ax_exc.set_title("Excitatory\n(no data)")
        else:
            hm = sns.heatmap(
                exc_matrix,
                ax=ax_exc,
                cmap="coolwarm",
                center=0.0,
                vmin=-exc_vmax,
                vmax=exc_vmax,
                linewidths=0.5,
                linecolor="white",
                cbar=True,
                cbar_kws={"label": "Δ value" if col == n_measures - 1 else None},
            )
            if col < n_measures - 1 and hm.collections:
                hm.collections[-1].colorbar.remove()
            ax_exc.set_title("")
            ax_exc.set_xlabel("Experiment")
            ax_exc.set_ylabel("Excitatory" if col == 0 else "")
            ax_exc.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_slopes()
