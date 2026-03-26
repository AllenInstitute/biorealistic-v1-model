#!/usr/bin/env python3
"""Build cached pairwise statistics for weight-property correlations."""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import network_utils as nu

from analysis_shared.weight_property_survey import PropertyLoader, compute_degrees

NETWORK_TYPES = ["bio_trained", "plain", "naive"]
PROPERTY_NAMES = [
    "image_selectivity",
    "orientation_selectivity",
    "firing_rate",
    "oracle_score",
    "in_degree",
    "out_degree",
]
ORIENTATIONS = ["outgoing", "incoming"]


@dataclass
class PairAccumulator:
    types: List[str]

    def __post_init__(self) -> None:
        self.type_to_idx = {t: i for i, t in enumerate(self.types)}
        n = len(self.types)
        shape = (n, n)
        self.count = np.zeros(shape, dtype=np.float64)
        self.sum_w = np.zeros(shape, dtype=np.float64)
        self.sum_w2 = np.zeros(shape, dtype=np.float64)
        self.sum_p = np.zeros(shape, dtype=np.float64)
        self.sum_p2 = np.zeros(shape, dtype=np.float64)
        self.sum_wp = np.zeros(shape, dtype=np.float64)

    def update(
        self,
        weights: np.ndarray,
        props: np.ndarray,
        src_idx: np.ndarray,
        tgt_idx: np.ndarray,
    ) -> None:
        mask = np.isfinite(weights) & np.isfinite(props)
        if not np.any(mask):
            return
        w = weights[mask]
        p = props[mask]
        src = src_idx[mask]
        tgt = tgt_idx[mask]
        key = src * len(self.types) + tgt
        size = len(self.types) ** 2
        self.count += np.bincount(key, weights=np.ones_like(w), minlength=size).reshape(
            len(self.types), len(self.types)
        )
        self.sum_w += np.bincount(key, weights=w, minlength=size).reshape(
            len(self.types), len(self.types)
        )
        self.sum_w2 += np.bincount(key, weights=w * w, minlength=size).reshape(
            len(self.types), len(self.types)
        )
        self.sum_p += np.bincount(key, weights=p, minlength=size).reshape(
            len(self.types), len(self.types)
        )
        self.sum_p2 += np.bincount(key, weights=p * p, minlength=size).reshape(
            len(self.types), len(self.types)
        )
        self.sum_wp += np.bincount(key, weights=w * p, minlength=size).reshape(
            len(self.types), len(self.types)
        )


@dataclass
class PropertySpec:
    name: str
    label: str


PROPERTY_SPECS = [
    PropertySpec("image_selectivity", "Image selectivity"),
    PropertySpec("orientation_selectivity", "DG orientation selectivity"),
    PropertySpec("firing_rate", "Mean firing rate"),
    PropertySpec("oracle_score", "Oracle correlation"),
    PropertySpec("in_degree", "In-degree"),
    PropertySpec("out_degree", "Out-degree"),
    PropertySpec("natural_image_evoked_rate", "Natural-image evoked rate"),
    PropertySpec("dg_spont_rate", "DG spontaneous rate"),
    PropertySpec("dg_evoked_rate", "DG evoked rate"),
    PropertySpec("dg_mean_rate", "DG mean rate"),
    PropertySpec("dg_peak_rate", "DG peak rate"),
    PropertySpec("dg_dsi", "DG direction selectivity"),
]


def discover_bases(root: Path, pattern: str) -> List[Path]:
    matches = sorted(root.glob(pattern))
    return [p for p in matches if p.is_dir()]


def _appendix_for_network(network_type: str) -> str:
    if network_type == "bio_trained":
        return "_bio_trained"
    if network_type == "naive":
        return "_naive"
    return ""


def load_typed_edges(
    base_dir: Path, network_type: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    appendix = _appendix_for_network(network_type)
    edge_lf = nu.load_edges_pl(str(base_dir), appendix=appendix)
    node_lf = nu.load_nodes_pl(str(base_dir), core_radius=200)

    cores = node_lf.select("core").collect().to_series()
    src_ids = edge_lf.select("source_id").collect().to_series()
    tgt_ids = edge_lf.select("target_id").collect().to_series()
    both_core = cores[src_ids] & cores[tgt_ids]

    edges = edge_lf.filter(both_core).collect().to_pandas()

    node_df = node_lf.collect().to_pandas().set_index("node_id")
    ctdf = nu.get_cell_type_table()["cell_type"]
    source_pop = node_df.loc[edges["source_id"].to_numpy(), "pop_name"].to_numpy()
    target_pop = node_df.loc[edges["target_id"].to_numpy(), "pop_name"].to_numpy()
    edges["source_type"] = ctdf.loc[source_pop].to_numpy()
    edges["target_type"] = ctdf.loc[target_pop].to_numpy()
    return edges, node_df


def collect_types(bases: Iterable[Path], network_type: str) -> List[str]:
    types: set[str] = set()
    for base in bases:
        edges, node_df = load_typed_edges(base, network_type)
        types.update(edges["source_type"].astype(str).unique())
        types.update(edges["target_type"].astype(str).unique())
    return sorted(types)


def update_property(
    acc: PairAccumulator,
    prop_name: str,
    prop_series: pd.Series | pd.DataFrame,
    orientation: str,
    edges: pd.DataFrame,
) -> None:
    weights = np.abs(edges["syn_weight"].to_numpy(dtype=float))
    if orientation == "outgoing":
        nodes = edges["source_id"].to_numpy(dtype=int)
        types_src = edges["source_type"].tolist()
        types_tgt = edges["target_type"].tolist()
    else:
        nodes = edges["target_id"].to_numpy(dtype=int)
        types_src = edges["target_type"].tolist()
        types_tgt = edges["source_type"].tolist()

    if isinstance(prop_series, pd.DataFrame):
        series = prop_series.iloc[:, 0]
    else:
        series = prop_series

    props = series.reindex(nodes).to_numpy(dtype=float)
    acc.update(
        weights,
        props,
        types_src if orientation == "outgoing" else types_src,
        types_tgt if orientation == "outgoing" else types_tgt,
    )


def accumulate_stats(
    bases: List[Path],
    network_type: str,
    loader: PropertyLoader,
    types: List[str],
    *,
    min_edges: int,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    n = len(types)
    type_to_idx = {t: i for i, t in enumerate(types)}

    def new_accumulator() -> Dict[str, PairAccumulator]:
        return {orient: PairAccumulator(types) for orient in ORIENTATIONS}

    stats: Dict[str, Dict[str, Dict[str, PairAccumulator]]] = {
        spec.name: {
            orient: {"edge": PairAccumulator(types), "total": PairAccumulator(types)}
            for orient in ORIENTATIONS
        }
        for spec in PROPERTY_SPECS
    }

    for base in bases:
        net_id = int("".join(ch for ch in base.name if ch.isdigit()))
        edges, node_df = load_typed_edges(base, network_type)
        edges = edges[
            [
                "source_id",
                "target_id",
                "source_type",
                "target_type",
                "syn_weight",
            ]
        ].copy()
        edges = edges[
            edges["source_type"].isin(types) & edges["target_type"].isin(types)
        ]
        if edges.empty:
            continue
        weights = np.abs(edges["syn_weight"].to_numpy(dtype=float))
        source_ids = edges["source_id"].to_numpy(dtype=int)
        target_ids = edges["target_id"].to_numpy(dtype=int)
        src_idx = edges["source_type"].map(type_to_idx).to_numpy(dtype=np.int64)
        tgt_idx = edges["target_type"].map(type_to_idx).to_numpy(dtype=np.int64)

        node_ids = node_df.index.to_numpy()
        selectivity = loader.selectivity_series(network_type, net_id)
        osi_df = loader.osi_df(str(base), network_type, net_id)
        oracle = loader.oracle_series(str(base), network_type, net_id)
        natural_rate = loader.natural_image_rate(
            str(base), network_type, net_id, node_ids
        )
        dg_metrics = loader.dg_rates(str(base), network_type, net_id)
        dsi_series = loader.dsi_series(str(base), network_type, net_id)
        in_deg, out_deg = compute_degrees(edges)

        prop_map = {
            "image_selectivity": selectivity,
            "orientation_selectivity": (
                osi_df.get("OSI") if osi_df is not None else None
            ),
            "firing_rate": osi_df.get("Ave_Rate(Hz)") if osi_df is not None else None,
            "oracle_score": oracle,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "natural_image_evoked_rate": natural_rate,
            "dg_spont_rate": dg_metrics.get("dg_spont_rate"),
            "dg_evoked_rate": dg_metrics.get("dg_evoked_rate"),
            "dg_mean_rate": dg_metrics.get("dg_mean_rate"),
            "dg_peak_rate": dg_metrics.get("dg_peak_rate"),
            "dg_dsi": dsi_series,
        }

        df = edges.copy()
        df["abs_w"] = weights
        df["src_idx"] = src_idx
        df["tgt_idx"] = tgt_idx

        incoming_totals = df.groupby(
            ["target_id", "tgt_idx", "src_idx"], as_index=False
        )["abs_w"].sum()
        outgoing_totals = df.groupby(
            ["source_id", "src_idx", "tgt_idx"], as_index=False
        )["abs_w"].sum()

        for prop_name, series in prop_map.items():
            if series is None or series.empty:
                continue
            series = series.astype(float)
            for orient in ORIENTATIONS:
                if orient == "outgoing":
                    props = series.reindex(source_ids).to_numpy(dtype=float)
                    src_indices = src_idx
                    tgt_indices = tgt_idx
                else:
                    props = series.reindex(target_ids).to_numpy(dtype=float)
                    src_indices = src_idx
                    tgt_indices = tgt_idx
                stats[prop_name][orient]["edge"].update(
                    weights,
                    props,
                    src_indices,
                    tgt_indices,
                )

                if orient == "incoming" and not incoming_totals.empty:
                    props_total = series.reindex(
                        incoming_totals["target_id"].to_numpy()
                    ).to_numpy(dtype=float)
                    stats[prop_name][orient]["total"].update(
                        incoming_totals["abs_w"].to_numpy(dtype=float),
                        props_total,
                        incoming_totals["src_idx"].to_numpy(dtype=np.int64),
                        incoming_totals["tgt_idx"].to_numpy(dtype=np.int64),
                    )
                if orient == "outgoing" and not outgoing_totals.empty:
                    props_total = series.reindex(
                        outgoing_totals["source_id"].to_numpy()
                    ).to_numpy(dtype=float)
                    stats[prop_name][orient]["total"].update(
                        outgoing_totals["abs_w"].to_numpy(dtype=float),
                        props_total,
                        outgoing_totals["src_idx"].to_numpy(dtype=np.int64),
                        outgoing_totals["tgt_idx"].to_numpy(dtype=np.int64),
                    )

    output: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for spec in PROPERTY_SPECS:
        prop_stats: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        for orient in ORIENTATIONS:
            metric_stats: Dict[str, Dict[str, np.ndarray]] = {}
            for metric, acc in stats[spec.name][orient].items():
                entry = {
                    "count": acc.count,
                    "sum_w": acc.sum_w,
                    "sum_w2": acc.sum_w2,
                    "sum_p": acc.sum_p,
                    "sum_p2": acc.sum_p2,
                    "sum_wp": acc.sum_wp,
                }
                metric_stats[metric] = entry
            prop_stats[orient] = metric_stats
        output[spec.name] = prop_stats
    return output


def save_stats(
    out_dir: Path,
    network_type: str,
    types: List[str],
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"weight_property_stats_{network_type}.pkl"
    payload = {
        "types": types,
        "properties": stats,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[cache] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute weight-property statistics for Dash explorer"
    )
    parser.add_argument(
        "--network-type",
        choices=NETWORK_TYPES,
        nargs="*",
        default=NETWORK_TYPES,
        help="Network types to process (default: all)",
    )
    parser.add_argument(
        "--core-root",
        default=Path("."),
        type=Path,
        help="Root directory containing core_nll_* directories",
    )
    parser.add_argument(
        "--selectivity",
        default=Path("image_decoding/summary/sparsity_model_by_unit.csv"),
        type=Path,
        help="CSV file with selectivity data",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("survey"),
        type=Path,
        help="Output directory for cached stats",
    )
    args = parser.parse_args()

    loader = PropertyLoader(args.selectivity)

    for nt in args.network_type:
        bases = discover_bases(args.core_root, "core_nll_*")
        if not bases:
            raise RuntimeError(f"No bases found under {args.core_root}")
        types = collect_types(bases, nt)
        print(f"[info] {nt}: {len(types)} base types")
        stats = accumulate_stats(bases, nt, loader, types, min_edges=0)
        save_stats(args.out_dir, nt, types, stats)


if __name__ == "__main__":
    main()
