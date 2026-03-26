#!/usr/bin/env python3
"""Heatmap of percent changes for outgoing-weight silencing (Figure 5)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_decoding.plot_utils import cell_type_order
from network_utils import load_nodes

OUTPUT_DIR = PROJECT_ROOT / "figures" / "paper" / "figure5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORE_RADIUS_UM = 200.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Heatmap of percent changes for outgoing-weight silencing (Figure 5)."
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "If provided, generate heatmaps for a single network directory (e.g. core_nll_0). "
            "If omitted, aggregate across core_nll_0..9 (default behavior)."
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to store the generated figures (default: figures/paper/figure5).",
    )
    ap.add_argument(
        "--baseline-exclude-targets",
        action="store_true",
        help=(
            "Compute baseline cell-type means after excluding the targeted cohort (matched non-targeted baseline). "
            "This is now the default; this flag is kept for backward compatibility."
        ),
    )
    ap.add_argument(
        "--baseline-include-targets",
        action="store_true",
        help=(
            "Legacy behavior: compute baseline cell-type means on all core neurons, including the targeted cohort."
        ),
    )
    ap.add_argument(
        "--include-targets",
        action="store_true",
        help=(
            "If set, include targeted neurons when computing perturbation and baseline cell-type means. "
            "This reproduces legacy core_nll_0 heatmaps that were computed on all neurons."
        ),
    )
    return ap.parse_args()


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display: str
    metric_file: str
    target_nodes_file: str | None = None


BASELINE_SPEC = DatasetSpec(
    key="bio_trained",
    display="Bio-trained",
    metric_file="OSI_DSI_DF_bio_trained.csv",
    target_nodes_file=None,
)

# Combine all silencing experiments of interest
PERTURBATIONS: Sequence[DatasetSpec] = (
    DatasetSpec(
        "pv_low_neg1000",
        "PV Low\nSilencing",
        "OSI_DSI_DF_pv_low_neg1000.csv",
        target_nodes_file="pv_low_outgoing_core_nodes.json",
    ),
    DatasetSpec(
        "pv_high_neg1000",
        "PV High\nSilencing",
        "OSI_DSI_DF_pv_high_neg1000.csv",
        target_nodes_file="pv_high_outgoing_core_nodes.json",
    ),
    DatasetSpec(
        "sst_low_neg1000",
        "SST Low\nSilencing",
        "OSI_DSI_DF_sst_low_neg1000.csv",
        target_nodes_file="sst_low_outgoing_core_nodes.json",
    ),
    DatasetSpec(
        "sst_high_neg1000",
        "SST High\nSilencing",
        "OSI_DSI_DF_sst_high_neg1000.csv",
        target_nodes_file="sst_high_outgoing_core_nodes.json",
    ),
    DatasetSpec(
        "vip_low_neg1000",
        "VIP Low\nSilencing",
        "OSI_DSI_DF_vip_low_neg1000.csv",
        target_nodes_file="vip_low_outgoing_core_nodes.json",
    ),
    DatasetSpec(
        "vip_high_neg1000",
        "VIP High\nSilencing",
        "OSI_DSI_DF_vip_high_neg1000.csv",
        target_nodes_file="vip_high_outgoing_core_nodes.json",
    ),
)

METRICS = ("Ave_Rate(Hz)", "OSI", "DSI", "dg_sparsity")
METRIC_LABELS = {
    "Ave_Rate(Hz)": "Firing Rate",
    "OSI": "OSI",
    "DSI": "DSI",
    "dg_sparsity": "DG Selectivity",
}


def _discover_base_dirs() -> list[Path]:
    base_dirs: list[Path] = []
    for i in range(10):
        d = PROJECT_ROOT / f"core_nll_{i}"
        if d.exists():
            base_dirs.append(d)
    return base_dirs


def _load_features(base_dir: Path) -> pd.DataFrame:
    features_path = (
        PROJECT_ROOT / "cell_categorization" / f"{base_dir.name}_neuron_features.parquet"
    )
    if features_path.exists():
        features = pd.read_parquet(features_path)
        cols = [c for c in ("node_id", "cell_type", "radius") if c in features.columns]
        if set(cols) == {"node_id", "cell_type", "radius"}:
            features = features[cols].drop_duplicates(subset="node_id")
            return features

    nodes_df = load_nodes(str(base_dir), loc="v1", expand=True)
    if "cell_type" not in nodes_df.columns:
        raise KeyError(f"cell_type not found in expanded nodes for {base_dir}")
    nodes_df = nodes_df.copy()
    if "x" in nodes_df.columns and "z" in nodes_df.columns:
        # Match legacy core definition used in core_nll_0_neuron_features.parquet:
        # radius = sqrt(x^2 + z^2)
        nodes_df["radius"] = np.sqrt(nodes_df["x"] ** 2 + nodes_df["z"] ** 2)
    else:
        nodes_df["radius"] = np.nan

    features = nodes_df.reset_index()[["node_id", "cell_type", "radius"]]
    features = features.drop_duplicates(subset="node_id")
    return features


def _load_target_node_ids(base_dir: Path, filename: str | None) -> set[int]:
    if not filename:
        return set()
    path = base_dir / "node_sets" / filename
    if not path.exists():
        return set()

    try:
        data = json.loads(path.read_text())
        ids = data.get("node_id", [])
        return {int(x) for x in ids}
    except Exception:
        pass

    try:
        df = pd.read_json(path)
        if "node_id" in df.columns:
            return {int(x) for x in df["node_id"].tolist()}
    except Exception:
        pass

    return set()


def _load_node_ids(base_dir: Path) -> np.ndarray:
    node_file = base_dir / "network" / "v1_nodes.h5"
    if not node_file.exists():
        raise FileNotFoundError(f"Missing node file: {node_file}")
    with h5py.File(node_file, "r") as f:
        return f["nodes"]["v1"]["node_id"][:].astype(np.int64)


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


def _load_dataset(
    spec: DatasetSpec,
    base_dir: Path,
    features: pd.DataFrame,
    node_ids: np.ndarray,
    *,
    exclude_node_ids: set[int] | None = None,
) -> pd.DataFrame:
    metric_dir = base_dir / "metrics"
    metric_path = metric_dir / spec.metric_file
    if not metric_path.exists():
        raise FileNotFoundError(metric_path)
    df = pd.read_csv(metric_path, sep=" ")
    df = df.rename(columns={"max_mean_rate(Hz)": "Rate at preferred direction (Hz)"})
    df = df.merge(features, how="left", on="node_id", validate="many_to_one")

    if exclude_node_ids:
        df = df[~df["node_id"].isin(exclude_node_ids)]

    df = df.dropna(subset=["cell_type", "radius"])
    df = df[df["radius"] <= CORE_RADIUS_UM]

    # Load DG Sparsity if available
    sparsity_path = metric_dir / f"dg_trial_averaged_sparsity_{spec.key}.npy"
    if sparsity_path.exists():
        sparsity_arr = np.load(sparsity_path)
        if len(sparsity_arr) == len(node_ids):
            s_series = pd.Series(sparsity_arr, index=node_ids, name="dg_sparsity")
            df = df.merge(s_series, left_on="node_id", right_index=True, how="left")
        else:
            print(
                f"Warning: Sparsity length mismatch for {base_dir.name}/{spec.key}: {len(sparsity_arr)} vs {len(node_ids)}"
            )

    df["cell_type"] = df["cell_type"].str.replace(" ", "_", regex=False)
    df["cell_type"] = df["cell_type"].str.replace("L1_Htr3a", "L1_Inh", regex=False)
    return df


def _cell_type_means(
    spec: DatasetSpec,
    base_dir: Path,
    *,
    features: pd.DataFrame,
    node_ids: np.ndarray,
    exclude_node_ids: set[int] | None = None,
) -> pd.DataFrame:
    df = _load_dataset(
        spec,
        base_dir,
        features,
        node_ids,
        exclude_node_ids=exclude_node_ids,
    )

    available_metrics = [m for m in METRICS if m in df.columns]
    grouped = df.groupby("cell_type")[available_metrics].mean(numeric_only=True)

    l5_subset = df[df["cell_type"].isin({"L5_ET", "L5_IT", "L5_NP"})]
    if not l5_subset.empty:
        grouped.loc["L5_Exc", available_metrics] = l5_subset[available_metrics].mean(
            numeric_only=True
        )
    return grouped


def plot_heatmap(
    data: pd.DataFrame, metric: str, title: str, output_path: Path
) -> None:
    if data.empty:
        print(f"No data for {title}")
        return

    # data index: cell_type, columns: perturbation display name
    # Compressed height as requested
    plt.figure(figsize=(6.0, 5.2))

    # Center colormap at 0 using vcenter=0 if possible, or symmetric vmin/vmax
    # Handle NaN/Inf in data before calculating max
    safe_data = data.replace([np.inf, -np.inf], np.nan)
    min_val = safe_data.min().min()
    max_val = safe_data.max().max()

    if pd.isna(min_val) or pd.isna(max_val):
        print(f"Skipping {metric} heatmap due to all NaN/Inf values")
        return

    max_abs = max(abs(min_val), abs(max_val))
    # Cap at reasonable percentage to show detail
    v_limit = min(max_abs, 80.0)
    if v_limit == 0:
        v_limit = 1.0

    ax = sns.heatmap(
        data,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "% Change"},
        vmin=-v_limit,
        vmax=v_limit,
    )

    ax.set_title(title)
    ax.set_ylabel("Observed Cell Type")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.savefig(output_path.with_suffix(".pdf"))
    plt.close()
    print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_exclude_targets = (not args.baseline_include_targets) and (not args.include_targets)

    if args.base_dir is not None:
        base_dirs = [args.base_dir]
    else:
        base_dirs = _discover_base_dirs()
    if not base_dirs:
        raise RuntimeError("No core_nll_* directories found.")

    all_types: set[str] = set()

    pct_changes: dict[str, dict[str, list[pd.Series]]] = {
        metric: {spec.display: [] for spec in PERTURBATIONS} for metric in METRICS
    }

    used_networks = 0
    for base_dir in base_dirs:
        try:
            features = _load_features(base_dir)
            node_ids = _load_node_ids(base_dir)
        except FileNotFoundError as e:
            print(f"Skipping {base_dir.name}: {e}")
            continue

        try:
            baseline_means_all = _cell_type_means(
                BASELINE_SPEC,
                base_dir,
                features=features,
                node_ids=node_ids,
                exclude_node_ids=None,
            )
        except FileNotFoundError:
            print(f"Skipping {base_dir.name}: baseline file not found")
            continue

        used_networks += 1
        all_types.update(baseline_means_all.index)

        for spec in PERTURBATIONS:
            target_ids = _load_target_node_ids(base_dir, spec.target_nodes_file)
            baseline_means = baseline_means_all
            if baseline_exclude_targets:
                baseline_means = _cell_type_means(
                    BASELINE_SPEC,
                    base_dir,
                    features=features,
                    node_ids=node_ids,
                    exclude_node_ids=target_ids,
                )
            try:
                exclude_ids = None if args.include_targets else target_ids
                exp_means = _cell_type_means(
                    spec,
                    base_dir,
                    features=features,
                    node_ids=node_ids,
                    exclude_node_ids=exclude_ids,
                )
            except FileNotFoundError:
                print(f"Skipping {base_dir.name}/{spec.display}: file not found")
                continue

            all_types.update(exp_means.index)

            for metric in METRICS:
                if (
                    metric not in baseline_means.columns
                    or metric not in exp_means.columns
                ):
                    continue

                common_types = baseline_means.index.intersection(exp_means.index)
                base_vals = baseline_means.loc[common_types, metric]
                exp_vals = exp_means.loc[common_types, metric]

                with np.errstate(divide="ignore", invalid="ignore"):
                    pct_change = (exp_vals - base_vals) / base_vals * 100.0

                pct_change = pct_change.replace([np.inf, -np.inf], np.nan)
                pct_change.name = base_dir.name
                pct_changes[metric][spec.display].append(pct_change)

    if used_networks == 0:
        raise RuntimeError("No networks contained baseline metrics; nothing to plot.")

    perturbation_results: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        metric_df = pd.DataFrame(index=sorted(all_types))
        for display in [spec.display for spec in PERTURBATIONS]:
            series_list = pct_changes[metric][display]
            if not series_list:
                continue
            stacked = pd.concat(series_list, axis=1)
            metric_df[display] = stacked.mean(axis=1, skipna=True)
        perturbation_results[metric] = metric_df

    # 3. Reorder rows (cell types)
    final_order = _build_cell_type_order(all_types)

    for metric in METRICS:
        df = perturbation_results[metric]
        if df.empty:
            continue

        # Reindex to sorted order and existing columns
        df = df.reindex(index=[ct for ct in final_order if ct in df.index])

        # Filter rows that are all NaN
        df = df.dropna(how="all")

        metric_name = METRIC_LABELS.get(metric, metric)
        title = f"{metric_name} Change (%)"
        plot_heatmap(
            df,
            metric,
            title,
            OUTPUT_DIR / f"celltype_silencing_heatmap_{metric}.png",
        )


if __name__ == "__main__":
    main()
