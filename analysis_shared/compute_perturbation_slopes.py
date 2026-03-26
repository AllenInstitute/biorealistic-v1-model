#!/usr/bin/env python3
"""Compute perturbation deltas and slopes for matched clamp experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from analysis_shared.osi_boxplot_utils import DatasetSpec, load_osi_dataset, build_cell_type_order

BASE_DIR = Path("core_nll_0")
METRICS_DIR = BASE_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MEASURES = [
    "Spont_Rate(Hz)",
    "Ave_Rate(Hz)",
    "Rate at preferred direction (Hz)",
    "OSI",
    "DSI",
]

BASELINE_SPEC = DatasetSpec(
    label="Bio-trained Model",
    basedir=BASE_DIR,
    metric_file="OSI_DSI_DF_bio_trained.csv",
    radius=200.0,
)

MATCHED_CASES: Dict[str, Dict[int, DatasetSpec]] = {
    "inh_nonselective_matched": {
        +100: DatasetSpec(
            label="Inh-nonselective matched pos 100",
            basedir=BASE_DIR,
            metric_file="OSI_DSI_DF_inh_nonselective_matched_pos100.csv",
            radius=200.0,
        ),
        -100: DatasetSpec(
            label="Inh-nonselective matched neg 100",
            basedir=BASE_DIR,
            metric_file="OSI_DSI_DF_inh_nonselective_matched_neg100.csv",
            radius=200.0,
        ),
    },
    "exc_selective_matched": {
        +100: DatasetSpec(
            label="Exc-selective matched pos 100",
            basedir=BASE_DIR,
            metric_file="OSI_DSI_DF_exc_selective_matched_pos100.csv",
            radius=200.0,
        ),
        -100: DatasetSpec(
            label="Exc-selective matched neg 100",
            basedir=BASE_DIR,
            metric_file="OSI_DSI_DF_exc_selective_matched_neg100.csv",
            radius=200.0,
        ),
    },
}


@dataclass
class SummaryRow:
    condition: str
    amplitude: int
    cell_type: str
    measure: str
    mean_value: float
    baseline_value: float
    diff: float
    rel_diff: float | None


@dataclass
class SlopeRow:
    condition: str
    cell_type: str
    measure: str
    slope_per_pA: float
    intercept: float


def _to_str_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cell_type"] = df["cell_type"].astype(str)
    return df


def compute_deltas() -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline_df = _to_str_index(load_osi_dataset(BASELINE_SPEC))
    order = build_cell_type_order(baseline_df["cell_type"].unique())
    baseline_means = baseline_df.groupby("cell_type").mean(numeric_only=True)

    summary_rows: List[SummaryRow] = []
    slope_rows: List[SlopeRow] = []

    for condition, spec_map in MATCHED_CASES.items():
        dataset_means: Dict[int, pd.DataFrame] = {}
        for amp, spec in spec_map.items():
            try:
                df = _to_str_index(load_osi_dataset(spec))
            except FileNotFoundError:
                continue
            df = df[df["cell_type"].isin(order)]
            dataset_means[amp] = df.groupby("cell_type").mean(numeric_only=True)
            for cell_type in order:
                if cell_type not in dataset_means[amp].index:
                    continue
                for measure in MEASURES:
                    if measure not in dataset_means[amp].columns:
                        continue
                    mean_val = dataset_means[amp].loc[cell_type, measure]
                    baseline_val = baseline_means.loc[cell_type, measure]
                    diff = mean_val - baseline_val
                    rel_diff = diff / baseline_val if baseline_val != 0 else None
                    summary_rows.append(
                        SummaryRow(
                            condition=condition,
                            amplitude=amp,
                            cell_type=cell_type,
                            measure=measure,
                            mean_value=mean_val,
                            baseline_value=baseline_val,
                            diff=diff,
                            rel_diff=rel_diff,
                        )
                    )

        if +100 not in dataset_means or -100 not in dataset_means:
            continue
        for cell_type in order:
            if cell_type not in dataset_means[+100].index or cell_type not in dataset_means[-100].index:
                continue
            for measure in MEASURES:
                if measure not in dataset_means[+100].columns or measure not in dataset_means[-100].columns:
                    continue
                val_pos = dataset_means[+100].loc[cell_type, measure]
                val_neg = dataset_means[-100].loc[cell_type, measure]
                slope = (val_pos - val_neg) / 200.0
                # slope per pA, intercept at 0 pA using average of two points
                intercept = (val_pos + val_neg) / 2.0
                slope_rows.append(
                    SlopeRow(
                        condition=condition,
                        cell_type=cell_type,
                        measure=measure,
                        slope_per_pA=slope,
                        intercept=intercept,
                    )
                )

    summary_df = pd.DataFrame([s.__dict__ for s in summary_rows])
    slopes_df = pd.DataFrame([s.__dict__ for s in slope_rows])
    return summary_df, slopes_df


def main() -> None:
    summary_df, slopes_df = compute_deltas()

    summary_path = METRICS_DIR / "perturbation_matched_deltas.csv"
    slopes_path = METRICS_DIR / "perturbation_matched_slopes.csv"

    summary_df.to_csv(summary_path, index=False)
    slopes_df.to_csv(slopes_path, index=False)

    print(f"Saved summary deltas to {summary_path}")
    print(f"Saved slopes to {slopes_path}")


if __name__ == "__main__":
    main()
