#!/usr/bin/env python3
"""Plot delta firing rate vs delta image selectivity per cell type."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
import sys
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import sys
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

FEATURES_PATH = PROJECT_ROOT / "cell_categorization" / "core_nll_0_neuron_features.parquet"

@dataclass(frozen=True)
class DatasetSpec:
    label: str
    basedir: Path | str
    metric_file: str

    def metrics_path(self) -> Path:
        base = Path(self.basedir)
        return base / "metrics" / self.metric_file

BASE_DIR = PROJECT_ROOT / "core_nll_0"
BASELINE_SPEC = DatasetSpec(
    label="Bio-trained Model",
    basedir=BASE_DIR,
    metric_file="OSI_DSI_DF_bio_trained.csv",
)

INH_SPECS: Mapping[str, DatasetSpec] = {
    "Selective +100": DatasetSpec("Inh-selective pos 100", BASE_DIR, "OSI_DSI_DF_inh_selective_pos100.csv"),
    "Selective -100": DatasetSpec("Inh-selective neg 100", BASE_DIR, "OSI_DSI_DF_inh_selective_neg100.csv"),
    "Non-selective +100": DatasetSpec("Inh-nonselective matched pos 100", BASE_DIR, "OSI_DSI_DF_inh_nonselective_matched_pos100.csv"),
    "Non-selective -100": DatasetSpec("Inh-nonselective matched neg 100", BASE_DIR, "OSI_DSI_DF_inh_nonselective_matched_neg100.csv"),
}

EXC_SPECS: Mapping[str, DatasetSpec] = {
    "Selective +100": DatasetSpec("Exc-selective matched pos 100", BASE_DIR, "OSI_DSI_DF_exc_selective_matched_pos100.csv"),
    "Selective -100": DatasetSpec("Exc-selective matched neg 100", BASE_DIR, "OSI_DSI_DF_exc_selective_matched_neg100.csv"),
    "Non-selective +100": DatasetSpec("Exc-nonselective pos 100", BASE_DIR, "OSI_DSI_DF_exc_nonselective_pos100.csv"),
    "Non-selective -100": DatasetSpec("Exc-nonselective neg 100", BASE_DIR, "OSI_DSI_DF_exc_nonselective_neg100.csv"),
}

METRICS = ("Ave_Rate(Hz)", "OSI")


def _load_features() -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH)
    features = features.drop_duplicates(subset="node_id")
    return features[["node_id", "cell_type"]]


def _load_dataset(spec: DatasetSpec, features: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(spec.metrics_path(), sep=" ")
    df = df.rename(columns={"max_mean_rate(Hz)": "Rate at preferred direction (Hz)"})
    df = df.merge(features, on="node_id", how="left", validate="many_to_one")
    df = df.dropna(subset=["cell_type"])
    df["cell_type"] = df["cell_type"].str.replace(" ", "_", regex=False)
    df["cell_type"] = df["cell_type"].str.replace("L1_Htr3a", "L1_Inh", regex=False)
    return df


def _cell_type_means(spec: DatasetSpec, features: pd.DataFrame) -> pd.DataFrame:
    df = _load_dataset(spec, features)
    grouped = df.groupby("cell_type")[list(METRICS)].mean(numeric_only=True)
    return grouped


def build_cell_type_order(present_types: Iterable[str]) -> list[str]:
    from analysis_shared.plot_fr_selectivity_scatter import build_cell_type_order as _base_order
    return _base_order(present_types)


def _compute_deltas(
    experiments: Mapping[str, DatasetSpec],
    baseline_means: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for label, spec in experiments.items():
        exp_means = _cell_type_means(spec, features)
        shared = baseline_means.index.intersection(exp_means.index)
        for cell_type in shared:
            rows.append(
                {
                    "scenario": label,
                    "cell_type": cell_type,
                    "delta_rate": exp_means.loc[cell_type, "Ave_Rate(Hz)"] - baseline_means.loc[cell_type, "Ave_Rate(Hz)"],
                    "delta_selectivity": exp_means.loc[cell_type, "OSI"] - baseline_means.loc[cell_type, "OSI"],
                    "baseline_rate": baseline_means.loc[cell_type, "Ave_Rate(Hz)"],
                    "baseline_selectivity": baseline_means.loc[cell_type, "OSI"],
                }
            )
    df = pd.DataFrame(rows)
    return df


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


CLASS_COLORS = {
    "Exc": "#1f77b4",
    "PV": "#d62728",
    "SST": "#2ca02c",
    "VIP": "#9467bd",
    "Inh": "#7f7f7f",
    "L1": "#bcbd22",
}


def plot_group(
    title: str,
    deltas: pd.DataFrame,
    order: Sequence[str],
    output: Path,
) -> None:
    if deltas.empty:
        return

    cols = 5
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0), sharex=True, sharey=True)
    axes = axes.flatten()

    all_x = deltas["delta_rate"].to_numpy()
    all_y = deltas["delta_selectivity"].to_numpy()
    if all_x.size == 0 or all_y.size == 0:
        return
    limit = max(np.max(np.abs(all_x)), np.max(np.abs(all_y)))
    if limit == 0:
        limit = 0.1
    limit *= 1.05

    scenario_palette = dict(zip(sorted(deltas["scenario"].unique()), sns.color_palette("colorblind", deltas["scenario"].nunique())))

    for idx, cell_type in enumerate(order):
        ax = axes[idx]
        subset = deltas[deltas["cell_type"] == cell_type]
        if subset.empty:
            ax.axis("off")
            continue
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.5, linestyle="--")

        base_row = subset.iloc[0]
        label = cell_type.replace("_", " ")
        ax.set_title(label, fontsize=10)

        ax.scatter(0, 0, marker="x", color="#444444", s=35)

        for scenario, group in subset.groupby("scenario"):
            cls = _cell_class(cell_type)
            color = scenario_palette[scenario]
            ax.scatter(group["delta_rate"], group["delta_selectivity"], s=40, color=color, edgecolor="black", linewidths=0.5, label=scenario)

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")

        if idx % cols == 0:
            ax.set_ylabel("Δ OSI")
        else:
            ax.set_ylabel("")
        if idx // cols == rows - 1:
            ax.set_xlabel("Δ Mean Rate (Hz)")
        else:
            ax.set_xlabel("")

    for ax in axes[len(order):]:
        ax.axis("off")

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, markeredgecolor="black", markeredgewidth=0.5, label=scenario)
        for scenario, color in scenario_palette.items()
    ]
    handles.insert(0, plt.Line2D([], [], marker="x", linestyle="", color="#444444", label="Baseline"))
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.98))

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    features = _load_features()
    baseline_means = _cell_type_means(BASELINE_SPEC, features)
    order = build_cell_type_order(baseline_means.index)

    inh_deltas = _compute_deltas(INH_SPECS, baseline_means, features)
    inh_output = BASE_DIR / "figures" / "perturbation_fr_selectivity" / "inh_delta_fr_vs_selectivity.png"
    plot_group("Inhibitory perturbations", inh_deltas, order, inh_output)

    exc_deltas = _compute_deltas(EXC_SPECS, baseline_means, features)
    exc_output = BASE_DIR / "figures" / "perturbation_fr_selectivity" / "exc_delta_fr_vs_selectivity.png"
    plot_group("Excitatory perturbations", exc_deltas, order, exc_output)


if __name__ == "__main__":
    main()
