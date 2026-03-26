#!/usr/bin/env python3
"""Box plots of core firing-rate/selectivity metrics split by outgoing cohorts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "core_nll_0"
METRICS_DIR = BASE_DIR / "metrics"
OUTPUT_DIR = BASE_DIR / "figures" / "selectivity_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = (
    PROJECT_ROOT / "cell_categorization" / "core_nll_0_neuron_features.parquet"
)
NETWORKS: List[Tuple[str, str, str]] = [
    ("bio_trained", "Bio-trained", "OSI_DSI_DF_bio_trained.csv"),
    ("plain", "Plain", "OSI_DSI_DF_plain.csv"),
]
CORE_RADIUS = 200.0

CELL_CLASSES: Dict[str, Dict[str, Iterable[str]]] = {
    "Exc": {
        "files": {
            "high": "high_outgoing_exc_core_nodes.json",
            "low": "low_outgoing_exc_core_nodes.json",
        },
        "matches": ("Exc", "IT", "ET", "NP"),
    },
    "PV": {
        "files": {
            "high": "pv_high_outgoing_core_nodes.json",
            "low": "pv_low_outgoing_core_nodes.json",
        },
        "matches": ("PV", "Pvalb"),
    },
    "SST": {
        "files": {
            "high": "sst_high_outgoing_core_nodes.json",
            "low": "sst_low_outgoing_core_nodes.json",
        },
        "matches": ("SST", "Sst"),
    },
    "VIP": {
        "files": {
            "high": "vip_high_outgoing_core_nodes.json",
            "low": "vip_low_outgoing_core_nodes.json",
        },
        "matches": ("VIP", "Vip"),
    },
}

METRICS = [
    ("rate", "Average firing rate (Hz)", "core_rate_by_cohort"),
    ("OSI", "Orientation selectivity (OSI)", "core_osi_by_cohort"),
    ("DSI", "Direction selectivity (DSI)", "core_dsi_by_cohort"),
    ("dg_sparsity", "DG trial-averaged sparsity", "core_dg_sparsity_by_cohort"),
    ("image_selectivity", "Image selectivity", "core_image_selectivity_by_cohort"),
]


def load_features() -> pd.DataFrame:
    features = pd.read_parquet(FEATURES_PATH).set_index("node_id")
    return features[["cell_type", "radius", "image_selectivity"]]


def load_sparsity(features: pd.DataFrame) -> pd.Series:
    cache = METRICS_DIR / "dg_trial_averaged_sparsity.npy"
    if cache.exists():
        sparsity = np.load(cache)
        if len(sparsity) == len(features):
            return pd.Series(sparsity, index=features.index, name="dg_sparsity")
    raise FileNotFoundError(
        "DG sparsity cache not found or length mismatch: "
        f"expected len={len(features)}"
    )


def load_node_ids(filename: str) -> set[int]:
    path = BASE_DIR / "node_sets" / filename
    if not path.exists():
        return set()
    df = pd.read_json(path)
    return set(map(int, df["node_id"]))


def matches_any(text: str, tokens: Iterable[str]) -> bool:
    return any(token in text for token in tokens)


def build_cohort_table(features: pd.DataFrame) -> pd.DataFrame:
    core_ids = set(features.index[features["radius"] <= CORE_RADIUS])
    rows: List[Tuple[int, str, str]] = []

    for label, info in CELL_CLASSES.items():
        tokens = info["matches"]
        mask = (
            features["cell_type"].fillna("").apply(lambda txt: matches_any(txt, tokens))
        )
        ids = set(map(int, features.index[mask])) & core_ids
        if not ids:
            continue

        high_ids = load_node_ids(info["files"]["high"])
        low_ids = load_node_ids(info["files"]["low"])

        for node_id in ids:
            if node_id in high_ids:
                cohort = "high"
            elif node_id in low_ids:
                cohort = "low"
            else:
                cohort = "mid"
            rows.append((node_id, label, cohort))

    cohort_df = pd.DataFrame(rows, columns=["node_id", "cell_class", "cohort"])

    # Aggregate inhibitory cohorts (PV/SST/VIP combined)
    inh_rows = cohort_df[cohort_df["cell_class"].isin(["PV", "SST", "VIP"])]
    if not inh_rows.empty:
        inh_added = inh_rows.copy()
        inh_added["cell_class"] = "Inh"
        cohort_df = pd.concat([cohort_df, inh_added], ignore_index=True)

    return cohort_df


def load_metrics_table(metrics_file: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_file, sep=" ")
    df = df.rename(
        columns={
            "Ave_Rate(Hz)": "rate",
            "max_mean_rate(Hz)": "max_rate",
        }
    )
    return df


def make_boxplot(
    df: pd.DataFrame, title: str, metric_col: str, metric_label: str, tag: str
) -> None:
    sns.set(style="whitegrid", context="talk")
    order = [
        c for c in ["Exc", "PV", "SST", "VIP", "Inh"] if c in df["cell_class"].unique()
    ]
    if not order:
        print(f"No data for {title}; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="cell_class",
        y=metric_col,
        hue="cohort",
        order=order,
        hue_order=["low", "mid", "high"],
        showfliers=False,
        ax=ax,
    )
    ax.set_xlabel("Cell class")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.legend(title="Outgoing cohort", frameon=True)
    plt.tight_layout()
    out_png = OUTPUT_DIR / f"{tag}.png"
    out_svg = OUTPUT_DIR / f"{tag}.svg"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"Saved {title} to {out_png}")


def main() -> None:
    features = load_features()
    features["dg_sparsity"] = load_sparsity(features)
    cohorts = build_cohort_table(features)

    feature_metrics = features[["image_selectivity", "dg_sparsity"]].reset_index()

    for key, label, metrics_fname in NETWORKS:
        metrics_df = load_metrics_table(METRICS_DIR / metrics_fname)
        merged = cohorts.merge(metrics_df, on="node_id", how="inner")
        merged = merged.merge(feature_metrics, on="node_id", how="left")

        if merged.empty:
            print(f"{label}: no matching neurons; skipping")
            continue

        print(
            f"{label}: {len(merged)} core neurons. cohort counts:\n"
            f"{merged.groupby(['cell_class', 'cohort']).size()}"
        )

        for metric_col, metric_label, filename_prefix in METRICS:
            if metric_col not in merged.columns:
                print(f"{label}: metric '{metric_col}' missing; skipping")
                continue
            plot_df = merged.dropna(subset=[metric_col]).copy()
            make_boxplot(
                plot_df,
                f"Core {metric_label} ({label})",
                metric_col,
                metric_label,
                f"{filename_prefix}_{key}",
            )


if __name__ == "__main__":
    main()
