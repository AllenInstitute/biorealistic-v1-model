#!/usr/bin/env python3
"""Visualize matched perturbation slopes grouped by measure."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from image_decoding.plot_utils import cell_type_order

SLOPE_PATH = Path("core_nll_0/metrics/perturbation_matched_deltas.csv")
OUTPUT_DIR = Path("core_nll_0/figures/perturbation_slopes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MEASURE_LABELS = {
    "Spont_Rate(Hz)": "Spont rate",
    "Ave_Rate(Hz)": "Mean rate",
    "Rate at preferred direction (Hz)": "Preferred rate",
    "OSI": "OSI",
    "DSI": "DSI",
}

CONDITIONS = {
    "inh_nonselective_matched": "Inhibitory non-selective (matched)",
    "exc_selective_matched": "Excitatory selective (matched)",
}

AMPLITUDE_ORDER = [100, -100]


def _prepare_matrix(df: pd.DataFrame, condition: str, measure: str) -> pd.DataFrame:
    subset = df[(df["condition"] == condition) & (df["measure"] == measure)].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["slope_per_pA"] = subset["diff"] / subset["amplitude"]

    order = cell_type_order()
    subset["cell_type"] = pd.Categorical(subset["cell_type"], categories=order, ordered=True)
    subset = subset.dropna(subset=["cell_type"])

    pivot = subset.pivot(index="cell_type", columns="amplitude", values="slope_per_pA")
    pivot = pivot.reindex(order).dropna(how="all")
    pivot = pivot[[c for c in AMPLITUDE_ORDER if c in pivot.columns]]
    # Rename columns to show sign
    pivot.rename(columns={amp: f"{amp:+}" for amp in pivot.columns}, inplace=True)
    return pivot


def plot_measure_panels(summary_df: pd.DataFrame, out_path: Path) -> None:
    measures = list(MEASURE_LABELS.keys())
    fig, axes = plt.subplots(len(measures), len(CONDITIONS), figsize=(10, 2.2 * len(measures)), squeeze=False)

    for row, measure in enumerate(measures):
        for col, (cond_key, cond_label) in enumerate(CONDITIONS.items()):
            ax = axes[row, col]
            matrix = _prepare_matrix(summary_df, cond_key, measure)
            if matrix.empty:
                ax.axis('off')
                ax.set_title(f"{cond_label}\n(no data)")
                continue
            vmax = np.nanmax(np.abs(matrix.values))
            vmax = vmax if vmax > 0 else 1.0
            hm = sns.heatmap(
                matrix,
                ax=ax,
                cmap='coolwarm',
                center=0.0,
                vmin=-vmax,
                vmax=vmax,
                linewidths=0.5,
                linecolor='white',
                cbar=True,
                cbar_kws={'label': 'Slope (Δ per pA)'},
            )
            # attach measure label on the leftmost column
            if col == 0:
                ax.set_ylabel(MEASURE_LABELS[measure])
            else:
                ax.set_ylabel('')
            ax.set_xlabel('Amplitude (pA)')
            ax.set_title(cond_label)
            ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    summary_df = pd.read_csv(SLOPE_PATH)
    plot_measure_panels(summary_df, OUTPUT_DIR / 'matched_slopes_by_measure.png')


if __name__ == '__main__':
    main()
