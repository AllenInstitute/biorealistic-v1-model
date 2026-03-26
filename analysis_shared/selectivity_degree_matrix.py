#!/usr/bin/env python3
"""Correlation between image selectivity and source-specific in-degrees."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from aggregate_correlation_plot import process_network_data
from analysis_shared.grouping import simplify_inh
from analysis_shared.style import apply_pub_style, trim_spines

TYPE_ORDER: List[str] = [
    "L2/3_Exc",
    "L4_Exc",
    "L5_IT",
    "L5_ET",
    "L5_NP",
    "L6_Exc",
    "PV",
    "SST",
    "VIP",
    "L1_Inh",
]


def parse_network_id(base_dir: Path) -> int:
    digits = "".join(ch for ch in base_dir.name if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse network id from {base_dir}")
    return int(digits)


def normalize_type(cell_type: str | None) -> str | None:
    if cell_type is None or pd.isna(cell_type):
        return None
    ct = str(cell_type)
    if ct in TYPE_ORDER:
        return ct
    if ct.endswith("_Exc"):
        head = ct.split("_")[0]
        label = f"{head}_Exc"
        if label in TYPE_ORDER:
            return label
    if ct == "L1_Inh":
        return ct
    simplified = simplify_inh(ct)
    if simplified in TYPE_ORDER:
        return simplified
    return None


def load_selectivity(path: Path, sel_type: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["network_type"] == sel_type].copy()
    df["network"] = df["network"].astype(int)
    df["node_id"] = df["node_id"].astype(int)
    df["cell_type"] = df["cell_type"].map(normalize_type)
    df = df[df["cell_type"].isin(TYPE_ORDER)].copy()
    return df[["network", "node_id", "cell_type", "image_selectivity"]]


def gather_in_degrees(bases: list[Path], network_type: str) -> pd.DataFrame:
    records = []
    for base in bases:
        net_id = parse_network_id(base)
        edges = process_network_data((str(base), network_type))
        edges = edges[["source_id", "target_id", "source_type", "target_type"]].copy()
        edges["source_type"] = edges["source_type"].map(normalize_type)
        edges["target_type"] = edges["target_type"].map(normalize_type)
        edges = edges.dropna(subset=["source_type", "target_type"])
        edges = edges[
            edges["source_type"].isin(TYPE_ORDER)
            & edges["target_type"].isin(TYPE_ORDER)
        ]
        grp = (
            edges.groupby(["target_id", "target_type", "source_type"])
            .size()
            .rename("count")
            .reset_index()
        )
        grp["network"] = net_id
        records.append(grp)
    if not records:
        raise RuntimeError("No degree records gathered")
    df = pd.concat(records, ignore_index=True)
    pivot = df.pivot_table(
        index=["network", "target_id", "target_type"],
        columns="source_type",
        values="count",
        fill_value=0,
    )
    pivot = pivot.reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"target_id": "node_id", "target_type": "node_type"})
    return pivot


def gather_out_degrees(bases: list[Path], network_type: str) -> pd.DataFrame:
    records = []
    for base in bases:
        net_id = parse_network_id(base)
        edges = process_network_data((str(base), network_type))
        edges = edges[["source_id", "target_id", "source_type", "target_type"]].copy()
        edges["source_type"] = edges["source_type"].map(normalize_type)
        edges["target_type"] = edges["target_type"].map(normalize_type)
        edges = edges.dropna(subset=["source_type", "target_type"])
        edges = edges[
            edges["source_type"].isin(TYPE_ORDER)
            & edges["target_type"].isin(TYPE_ORDER)
        ]
        grp = (
            edges.groupby(["source_id", "source_type", "target_type"])
            .size()
            .rename("count")
            .reset_index()
        )
        grp["network"] = net_id
        records.append(grp)
    if not records:
        raise RuntimeError("No degree records gathered")
    df = pd.concat(records, ignore_index=True)
    pivot = df.pivot_table(
        index=["network", "source_id", "source_type"],
        columns="target_type",
        values="count",
        fill_value=0,
    )
    pivot = pivot.reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"source_id": "node_id", "source_type": "node_type"})
    return pivot


def compute_node_correlations(
    merged: pd.DataFrame,
    *,
    min_cells: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    partner_types = [c for c in TYPE_ORDER if c in merged.columns]
    node_types = [t for t in TYPE_ORDER if t in merged["node_type"].unique()]
    corr = pd.DataFrame(index=node_types, columns=partner_types, dtype=float)
    counts = pd.DataFrame(index=node_types, columns=partner_types, dtype=float)
    for node in node_types:
        sub = merged[merged["node_type"] == node]
        prop = sub["image_selectivity"].to_numpy(dtype=float)
        for partner in partner_types:
            deg = sub[partner].to_numpy(dtype=float)
            mask = np.isfinite(prop) & np.isfinite(deg)
            n = int(mask.sum())
            counts.loc[node, partner] = n
            if n < min_cells or n < 3:
                corr.loc[node, partner] = np.nan
                continue
            y = deg[mask]
            if np.allclose(y, y[0]):
                corr.loc[node, partner] = np.nan
                continue
            r = float(np.corrcoef(prop[mask], y)[0, 1])
            corr.loc[node, partner] = r
    return corr, counts


def compute_edge_correlations(
    bases: list[Path],
    network_type: str,
    selectivity_map: dict[int, pd.Series],
    *,
    selectivity_role: str,
    degree_node_role: str,
    degree_mode: str,
    min_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_pairs = len(TYPE_ORDER) * len(TYPE_ORDER)
    sum_x = np.zeros(n_pairs, dtype=np.float64)
    sum_y = np.zeros(n_pairs, dtype=np.float64)
    sum_x2 = np.zeros(n_pairs, dtype=np.float64)
    sum_y2 = np.zeros(n_pairs, dtype=np.float64)
    sum_xy = np.zeros(n_pairs, dtype=np.float64)
    counts = np.zeros(n_pairs, dtype=np.float64)

    for base in bases:
        net_id = parse_network_id(base)
        sel_series = selectivity_map.get(net_id)
        if sel_series is None or sel_series.empty:
            continue

        edges = process_network_data((str(base), network_type))
        cols = ["source_id", "target_id", "source_type", "target_type"]
        edges = edges[cols].copy()
        edges["source_type"] = edges["source_type"].map(normalize_type)
        edges["target_type"] = edges["target_type"].map(normalize_type)
        edges = edges.dropna(subset=["source_type", "target_type"])
        if edges.empty:
            continue

        src_codes = pd.Categorical(edges["source_type"], categories=TYPE_ORDER).codes
        tgt_codes = pd.Categorical(edges["target_type"], categories=TYPE_ORDER).codes
        valid = (src_codes >= 0) & (tgt_codes >= 0)
        if not valid.any():
            continue
        edges = edges.loc[valid].reset_index(drop=True)
        src_codes = src_codes[valid]
        tgt_codes = tgt_codes[valid]

        source_ids = edges["source_id"].to_numpy(dtype=int)
        target_ids = edges["target_id"].to_numpy(dtype=int)

        sel_ids = source_ids if selectivity_role == "source" else target_ids
        sel_vals = sel_series.reindex(sel_ids).to_numpy(dtype=float)

        in_counts = edges.groupby("target_id").size().astype(float)
        out_counts = edges.groupby("source_id").size().astype(float)

        if degree_node_role == "source":
            deg_ids = source_ids
            deg_series = in_counts if degree_mode == "incoming" else out_counts
        else:
            deg_ids = target_ids
            deg_series = in_counts if degree_mode == "incoming" else out_counts

        deg_vals = deg_series.reindex(deg_ids, fill_value=0.0).to_numpy(dtype=float)

        mask = np.isfinite(sel_vals) & np.isfinite(deg_vals)
        if not mask.any():
            continue

        pairs = src_codes[mask] * len(TYPE_ORDER) + tgt_codes[mask]
        x = sel_vals[mask]
        y = deg_vals[mask]

        counts += np.bincount(pairs, minlength=n_pairs)
        sum_x += np.bincount(pairs, weights=x, minlength=n_pairs)
        sum_y += np.bincount(pairs, weights=y, minlength=n_pairs)
        sum_x2 += np.bincount(pairs, weights=x * x, minlength=n_pairs)
        sum_y2 += np.bincount(pairs, weights=y * y, minlength=n_pairs)
        sum_xy += np.bincount(pairs, weights=x * y, minlength=n_pairs)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = sum_x / counts
        mean_y = sum_y / counts
        var_x = sum_x2 - counts * mean_x * mean_x
        var_y = sum_y2 - counts * mean_y * mean_y
        cov = sum_xy - counts * mean_x * mean_y

    corr = np.full(n_pairs, np.nan, dtype=float)
    valid = (counts >= max(min_edges, 2)) & (var_x > 0) & (var_y > 0)
    corr[valid] = cov[valid] / np.sqrt(var_x[valid] * var_y[valid])

    count_matrix = counts.reshape(len(TYPE_ORDER), len(TYPE_ORDER))
    corr_matrix = corr.reshape(len(TYPE_ORDER), len(TYPE_ORDER))

    corr_df = pd.DataFrame(corr_matrix, index=TYPE_ORDER, columns=TYPE_ORDER)
    count_df = pd.DataFrame(count_matrix, index=TYPE_ORDER, columns=TYPE_ORDER)
    return corr_df, count_df


def plot_matrix(matrix: pd.DataFrame, title: str, out_path: Path) -> None:
    apply_pub_style()
    data = matrix.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.1
    fig_w = 5.2
    fig_h = 4.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Partner cell type")
    ax.set_ylabel("Selectivity cell type")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Pearson r")
    ax.set_title(title, fontsize=10)
    trim_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image selectivity vs in-degree matrix"
    )
    parser.add_argument(
        "--bases", nargs="*", help="Base directories (default core_nll_*)"
    )
    parser.add_argument("--network-type", default="bio_trained")
    parser.add_argument(
        "--selectivity",
        type=Path,
        default=Path("image_decoding/summary/sparsity_model_by_unit.csv"),
    )
    parser.add_argument(
        "--selectivity-type",
        help="network_type label to use for selectivity file (default: match network-type)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-cells", type=int, default=50)
    parser.add_argument(
        "--mode",
        choices=["incoming", "outgoing"],
        default="incoming",
    )
    parser.add_argument(
        "--selectivity-role",
        choices=["target", "source"],
        default="target",
    )
    parser.add_argument(
        "--degree-node-role",
        choices=["target", "source"],
        help="Node (source/target) whose degree is measured (default: same as selectivity role)",
    )
    args = parser.parse_args()

    bases = (
        [Path(b) for b in args.bases]
        if args.bases
        else sorted(Path(".").glob("core_nll_*"))
    )
    bases = [b for b in bases if b.is_dir()]
    if not bases:
        raise RuntimeError("No bases found")

    sel_type = args.selectivity_type or args.network_type
    selectivity = load_selectivity(args.selectivity, sel_type)
    selectivity_map = {
        net: grp.set_index("node_id")["image_selectivity"]
        for net, grp in selectivity.groupby("network")
    }

    degree_node_role = args.degree_node_role or args.selectivity_role

    if degree_node_role == args.selectivity_role:
        if args.mode == "incoming":
            degrees = gather_in_degrees(bases, args.network_type)
        else:
            degrees = gather_out_degrees(bases, args.network_type)

        merged = degrees.merge(
            selectivity,
            left_on=["network", "node_id"],
            right_on=["network", "node_id"],
            how="inner",
        )
        merged["node_type"] = merged["cell_type"].where(
            merged["cell_type"].isin(TYPE_ORDER), merged["node_type"]
        )
        merged = merged[merged["node_type"].isin(TYPE_ORDER)].copy()
        merged = merged.drop(columns=["cell_type"], errors="ignore")
        corr_df, counts_df = compute_node_correlations(
            merged,
            min_cells=args.min_cells,
        )
    else:
        corr_df, counts_df = compute_edge_correlations(
            bases,
            args.network_type,
            selectivity_map,
            selectivity_role=args.selectivity_role,
            degree_node_role=degree_node_role,
            degree_mode=args.mode,
            min_edges=args.min_cells,
        )

    degree_label = "in-degree" if args.mode == "incoming" else "out-degree"
    title = f"{args.selectivity_role.capitalize()} selectivity vs {degree_node_role} {degree_label}"

    out_csv = args.out
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(out_csv)
    counts_df.to_csv(out_csv.with_name(out_csv.stem + "_counts.csv"))
    plot_matrix(corr_df, title, out_csv.with_suffix(".png"))


if __name__ == "__main__":
    main()
