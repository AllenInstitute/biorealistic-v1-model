#!/usr/bin/env python3
"""Box plots of core firing-rate/selectivity metrics split by outgoing cohorts (Exc/Inh only for Figure 5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# Plot settings (edit these for consistent sizing across all panels in this file)
# =============================================================================
FIGSIZE_SINGLE = (5, 4)
FIGSIZE_COMBINED = (10, 4)
FR_YMAX = 40

# Single-network boxplots
SINGLE_XLABEL_FS = 18
SINGLE_YLABEL_FS = 18

# Combined panels (both group-order modes)
COMBINED_YLABEL_FS = 18
COMBINED_XTICK_FS = 18
COMBINED_XTICK_PAD = 0

# Combined, celltype-first ordering: repeated "Untrained/Trained" labels + group labels
CELLTYPE_FIRST_TICK_FS = 17
CELLTYPE_FIRST_GROUP_LABEL_FS = 18
CELLTYPE_FIRST_GROUP_LABEL_Y = -0.20

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "figures" / "paper" / "extended_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NETWORKS: List[Tuple[str, str, str]] = [
    ("bio_trained", "Bio-trained", "OSI_DSI_DF_bio_trained.csv"),
    ("plain", "Plain", "OSI_DSI_DF_plain.csv"),
]
CORE_RADIUS = 200.0

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from network_utils import load_nodes

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
    ("OSI", "OSI", "core_osi_by_cohort"),
    ("DSI", "DSI", "core_dsi_by_cohort"),
    ("dg_sparsity", "DG trial-averaged sparsity", "core_dg_sparsity_by_cohort"),
    ("image_selectivity", "Image selectivity", "core_image_selectivity_by_cohort"),
]

TRAINING_LABELS = {
    "bio_trained": "trained",
    "plain": "untrained",
}

# Cohort palette (keep consistent across histogram/rate/sparsity panels)
COHORT_COLORS = {
    "low": "#2b7bba",  # blue
    "mid": "#888888",  # gray
    "high": "#c73635",  # red
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fig.5 boxplots for rates/selectivity split by outgoing cohorts."
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "If provided, generate plots for a single network directory (e.g. core_nll_0). "
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
        "--cohort-metric",
        choices=["weight", "synapsecount"],
        default="weight",
        help="Which cohort-definition node sets to use (default: weight).",
    )
    ap.add_argument(
        "--combined-only",
        action="store_true",
        help="Only generate the combined bio-vs-plain panels (skip per-network panels).",
    )
    ap.add_argument(
        "--group-order",
        choices=["training_first", "celltype_first"],
        default="training_first",
        help="Combined-panel x-axis hierarchy. training_first=(training, cell type, cohort); "
        "celltype_first=(cell type, training, cohort).",
    )
    ap.add_argument(
        "--hide-outliers",
        action="store_true",
        help="Hide outliers (fliers) on boxplots. By default, outliers are shown.",
    )
    return ap.parse_args()


def _cohort_filename(original: str, cohort_metric: str) -> str:
    """Map weight-based node-set filenames to synapsecount-based equivalents."""
    if cohort_metric == "weight":
        return original
    # synapsecount naming scheme introduced by create_highlow_outgoing_synapsecount_nodesets.py
    if original.startswith(("pv_", "sst_", "vip_")):
        # e.g. pv_high_outgoing_core_nodes.json -> pv_high_outgoing_synapsecount_core_nodes.json
        return original.replace("_outgoing_", "_outgoing_synapsecount_", 1)
    # e.g. high_outgoing_exc_core_nodes.json -> high_outgoing_synapsecount_exc_core_nodes.json
    return original.replace("_outgoing_", "_outgoing_synapsecount_", 1)


def _discover_base_dirs() -> List[Path]:
    base_dirs: List[Path] = []
    for i in range(10):
        d = PROJECT_ROOT / f"core_nll_{i}"
        if d.exists():
            base_dirs.append(d)
    return base_dirs


def _load_node_ids(base_dir: Path) -> np.ndarray:
    node_file = base_dir / "network" / "v1_nodes.h5"
    with h5py.File(node_file, "r") as f:
        return f["nodes"]["v1"]["node_id"][:].astype(np.int64)


def load_features(base_dir: Path) -> pd.DataFrame:
    nodes_df = load_nodes(str(base_dir), loc="v1", expand=True)
    nodes_df = nodes_df.copy()
    center_x = float(np.median(nodes_df["x"]))
    center_z = float(np.median(nodes_df["z"]))
    nodes_df["radius"] = np.sqrt(
        (nodes_df["x"] - center_x) ** 2 + (nodes_df["z"] - center_z) ** 2
    )
    features = nodes_df[["cell_type", "radius"]].copy()

    features_path = PROJECT_ROOT / "cell_categorization" / f"{base_dir.name}_neuron_features.parquet"
    if features_path.exists():
        try:
            extra = pd.read_parquet(features_path).set_index("node_id")
            if "image_selectivity" in extra.columns:
                features = features.join(extra[["image_selectivity"]], how="left")
        except Exception:
            pass

    if "image_selectivity" not in features.columns:
        features["image_selectivity"] = np.nan

    return features


def load_sparsity(base_dir: Path, node_ids: np.ndarray, network: str) -> pd.Series:
    """Load sparsity data for a specific network."""
    metrics_dir = base_dir / "metrics"
    cache = metrics_dir / f"dg_trial_averaged_sparsity_{network}.npy"
    if not cache.exists() and network == "bio_trained":
        cache = metrics_dir / "dg_trial_averaged_sparsity.npy"

    if cache.exists():
        sparsity = np.load(cache)
        if len(sparsity) == len(node_ids):
            series = pd.Series(sparsity, index=node_ids, name="dg_sparsity")
            series.index.name = "node_id"
            return series

    print(f"Warning: DG sparsity cache not found for {network} in {base_dir} or length mismatch.")
    series = pd.Series(np.nan, index=node_ids, name="dg_sparsity")
    series.index.name = "node_id"
    return series


def load_node_ids(base_dir: Path, filename: str) -> set[int]:
    path = base_dir / "node_sets" / filename
    if not path.exists():
        return set()
    df = pd.read_json(path)
    return set(map(int, df["node_id"]))


def matches_any(text: str, tokens: Iterable[str]) -> bool:
    return any(token in text for token in tokens)


def build_cohort_table(base_dir: Path, features: pd.DataFrame) -> pd.DataFrame:
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

        high_ids = load_node_ids(base_dir, info["files"]["high"])
        low_ids = load_node_ids(base_dir, info["files"]["low"])

        for node_id in ids:
            if node_id in high_ids:
                cohort = "high"
            elif node_id in low_ids:
                cohort = "low"
            else:
                cohort = "mid"
            rows.append((node_id, label, cohort))

    cohort_df = pd.DataFrame(rows, columns=["node_id", "cell_type", "cohort"])

    # Filter to only our target classes
    cohort_df = cohort_df[cohort_df["cell_type"].isin(["PV", "SST", "VIP"])]

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
    # Only Exc and Inh
    order = [c for c in ["PV", "SST", "VIP"] if c in df["cell_type"].unique()]
    if not order:
        print(f"No data for {title}; skipping plot")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    sns.boxplot(
        data=df,
        x="cell_class",
        y=metric_col,
        hue="cohort",
        order=order,
        hue_order=["low", "mid", "high"],
        palette=COHORT_COLORS,
        showfliers=True,
        ax=ax,
    )
    ax.set_xlabel("Cell class", fontsize=SINGLE_XLABEL_FS)
    ax.set_ylabel(metric_label, fontsize=SINGLE_YLABEL_FS)
    if metric_col == "rate":
        ax.set_ylim(0, FR_YMAX)
    # Title removed as requested
    # ax.set_title(title)
    # Remove legend as requested (labels are on top/handled elsewhere)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_png = OUTPUT_DIR / f"{tag}.png"
    out_pdf = OUTPUT_DIR / f"{tag}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved {title} to {out_png}")


def _lighten_color(color, amount: float = 0.35):
    """Lighten the given color by mixing it with white."""
    r, g, b, *rest = mcolors.to_rgba(color)
    r = 1 - amount * (1 - r)
    g = 1 - amount * (1 - g)
    b = 1 - amount * (1 - b)
    if rest:
        return (r, g, b, rest[0])
    return (r, g, b)


def make_combined_boxplot(
    df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    filename_prefix: str,
    *,
    out_suffix: str = "",
    showfliers: bool = True,
    group_order: str = "training_first",
    ylim: Tuple[float, float] | None = None,
) -> None:
    sns.set(style="whitegrid", context="talk")

    if group_order == "celltype_first":
        order_all = [
            "PV (untrained)",
            "PV (trained)",
            "SST (untrained)",
            "SST (trained)",
            "VIP (untrained)",
            "VIP (trained)",
        ]
    else:
        order_all = [
            "PV (untrained)",
            "SST (untrained)",
            "VIP (untrained)",
            "PV (trained)",
            "SST (trained)",
            "VIP (trained)",
        ]
    
    present_order = [c for c in order_all if c in df["cell_type_training"].unique()]
    if not present_order:
        print(f"No data for combined {metric_col}; skipping plot")
        return

    hue_order = ["low", "mid", "high"]
    palette = {k: COHORT_COLORS[k] for k in hue_order}

    fig, ax = plt.subplots(figsize=FIGSIZE_COMBINED)
    sns.boxplot(
        data=df,
        x="cell_type_training",
        y=metric_col,
        hue="cohort",
        order=present_order,
        hue_order=hue_order,
        palette=palette,
        showfliers=showfliers,
        ax=ax,
    )

    # Lighten untrained boxes to make them a bit fainter
    boxes_per_group = len(hue_order)
    for idx, artist in enumerate(ax.artists):
        group_idx = idx // boxes_per_group
        if group_idx < len(present_order) and "untrained" in present_order[group_idx]:
            artist.set_facecolor(_lighten_color(artist.get_facecolor(), amount=0.45))
            artist.set_alpha(0.8)

    ax.set_xlabel("")
    ax.set_ylabel(metric_label, fontsize=COMBINED_YLABEL_FS)
    if ylim:
        ax.set_ylim(ylim)
    
    ticks = list(range(len(present_order)))
    ax.set_xticks(ticks)
    if group_order == "celltype_first":
        ax.set_xticklabels(
            ["Untrained", "Trained", "Untrained", "Trained", "Untrained", "Trained"][: len(ticks)],
            fontsize=CELLTYPE_FIRST_TICK_FS,
        )
        if len(ticks) >= 6:
            y_ct = CELLTYPE_FIRST_GROUP_LABEL_Y
            ct_fs = CELLTYPE_FIRST_GROUP_LABEL_FS
            ax.text(0.5, y_ct, "PV", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=ct_fs)
            ax.text(2.5, y_ct, "SST", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=ct_fs)
            ax.text(4.5, y_ct, "VIP", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=ct_fs)
            ax.tick_params(axis="x", pad=COMBINED_XTICK_PAD)
    else:
        ax.set_xticklabels(
            [lbl.split(" ")[0] for lbl in present_order], fontsize=COMBINED_XTICK_FS
        )
        if len(ticks) >= 4:
            untrained_center = (ticks[0] + ticks[1]) / 2
            trained_center = (ticks[-2] + ticks[-1]) / 2
            y_text = -0.08
            ax.text(
                untrained_center,
                y_text,
                "Untrained",
                ha="center",
                va="center",
                transform=ax.get_xaxis_transform(),
                fontsize=16,
            )
            ax.text(
                trained_center,
                y_text,
                "Trained",
                ha="center",
                va="center",
                transform=ax.get_xaxis_transform(),
                fontsize=16,
            )
            ax.tick_params(axis="x", pad=18)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out_png = OUTPUT_DIR / f"{filename_prefix}_bio_vs_plain{out_suffix}.png"
    out_pdf = OUTPUT_DIR / f"{filename_prefix}_bio_vs_plain{out_suffix}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved combined {metric_col} plot to {out_png}")


def main() -> None:
    args = parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.base_dir is not None:
        base_dirs = [args.base_dir]
    else:
        base_dirs = _discover_base_dirs()
    if not base_dirs:
        raise RuntimeError("No core_nll_* directories found.")

    # Rebuild CELL_CLASSES mapping with the requested cohort metric filenames
    global CELL_CLASSES
    CELL_CLASSES = {
        k: {
            **v,
            "files": {
                wk: _cohort_filename(wv, args.cohort_metric)
                for wk, wv in v["files"].items()
            },
        }
        for k, v in CELL_CLASSES.items()
    }

    combined_parts: Dict[str, List[pd.DataFrame]] = {
        m[0]: [] for m in METRICS
    }

    for key, label, metrics_fname in NETWORKS:
        parts: List[pd.DataFrame] = []
        for base_dir in base_dirs:
            metrics_path = base_dir / "metrics" / metrics_fname
            if not metrics_path.exists():
                continue
            features = load_features(base_dir)
            cohorts = build_cohort_table(base_dir, features)
            if cohorts.empty:
                continue

            metrics_df = load_metrics_table(metrics_path)
            node_ids = _load_node_ids(base_dir)
            sparsity_series = load_sparsity(base_dir, node_ids, key)
            sparsity_df = sparsity_series.reset_index()
            feature_metrics_base = features[["image_selectivity"]].reset_index()

            merged = cohorts.merge(metrics_df, on="node_id", how="inner")
            merged = merged.merge(feature_metrics_base, on="node_id", how="left")
            merged = merged.merge(sparsity_df, on="node_id", how="left")
            if merged.empty:
                continue
            merged["_base_dir"] = base_dir.name
            parts.append(merged)

        if not parts:
            print(f"{label}: no matching neurons; skipping")
            continue

        merged = pd.concat(parts, ignore_index=True)
        print(
            f"{label}: {len(merged)} core neurons (aggregated). cohort counts:\n"
            f"{merged.groupby(['cell_type', 'cohort']).size()}"
        )

        for metric_col, metric_label, filename_prefix in METRICS:
            if metric_col not in merged.columns:
                print(f"{label}: metric '{metric_col}' missing; skipping")
                continue

            plot_df = merged.dropna(subset=[metric_col]).copy()
            if plot_df.empty:
                print(f"{label}: no valid data for {metric_col}")
                continue
            if not args.combined_only:
                make_boxplot(
                    plot_df,
                    f"Core {metric_label} ({label})",
                    metric_col,
                    metric_label,
                    f"{filename_prefix}_{key}"
                    + ("" if args.cohort_metric == "weight" else "_synapsecount"),
                )

            training_status = TRAINING_LABELS.get(key, key)
            combined = plot_df.copy()
            combined["training_status"] = training_status
            combined["cell_type_training"] = combined["cell_type"].apply(
                lambda c: f"{c} ({training_status})"
            )
            combined_parts[metric_col].append(combined)

    showfliers = not args.hide_outliers

    out_suffix = "" if args.cohort_metric == "weight" else "_synapsecount"
    if not showfliers:
        out_suffix = out_suffix + "_no_outliers"
    out_suffix = out_suffix + (
        "_celltype_first" if args.group_order == "celltype_first" else ""
    )

    for metric_col, metric_label, filename_prefix in METRICS:
        parts = combined_parts.get(metric_col, [])
        if not parts:
            continue
        
        combined_df = pd.concat(parts, ignore_index=True)
        
        ylim = None
        if metric_col == "rate":
            ylim = (0, FR_YMAX)
        elif metric_col in ["OSI", "DSI", "dg_sparsity", "image_selectivity"]:
            ylim = (0, 1)

        make_combined_boxplot(
            combined_df,
            metric_col,
            metric_label,
            filename_prefix,
            out_suffix=out_suffix,
            showfliers=showfliers,
            group_order=args.group_order,
            ylim=ylim,
        )


if __name__ == "__main__":
    main()
