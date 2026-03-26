#!/usr/bin/env python3
"""Utilities for constructing per-neuron feature tables."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import network_utils as nu
from analysis_shared.type_aggregation import (
    class_part,
    layer_part,
    map_inhibitory,
    map_l5,
)
from analysis_shared.weight_property_survey import (
    PropertyLoader,
    parse_network_id,
    prepare_edge_table,
)

NUMERIC_FILL_VALUE = 0.0


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata describing a computed feature column."""

    name: str
    dtype: str
    description: str


CORE_FEATURE_SPECS: Sequence[FeatureSpec] = (
    FeatureSpec("image_selectivity", "float", "Image selectivity score"),
    FeatureSpec("orientation_selectivity", "float", "Drifting grating OSI"),
    FeatureSpec("firing_rate", "float", "Mean firing rate (Hz)"),
    FeatureSpec("oracle_score", "float", "Oracle correlation"),
    FeatureSpec("natural_image_evoked_rate", "float", "Imagenet evoked rate (Hz)"),
    FeatureSpec("dg_spont_rate", "float", "DG spontaneous rate (Hz)"),
    FeatureSpec("dg_mean_rate", "float", "DG mean rate (Hz)"),
    FeatureSpec("dg_evoked_rate", "float", "DG evoked rate (Hz)"),
    FeatureSpec("dg_peak_rate", "float", "DG peak rate (Hz)"),
    FeatureSpec("dg_dsi", "float", "DG direction selectivity index"),
    FeatureSpec(
        "activity_selectivity_score",
        "float",
        "Composite selectivity z-score",
    ),
    FeatureSpec("activity_rate_score", "float", "Composite firing-rate z-score"),
    FeatureSpec(
        "activity_discriminant",
        "float",
        "Selectivity minus rate z-score",
    ),
)

EDGE_METRICS: Sequence[str] = (
    "degree",
    "weight_sum",
    "weight_sum_abs",
    "weight_mean",
    "weight_mean_abs",
    "weight_std",
)

SELECTIVITY_METRICS: Sequence[str] = (
    "image_selectivity",
    "orientation_selectivity",
    "dg_dsi",
)

RATE_METRICS: Sequence[str] = (
    "natural_image_evoked_rate",
    "dg_spont_rate",
    "dg_evoked_rate",
    "dg_peak_rate",
)


def _safe_layer(label: str | float | None) -> str:
    if isinstance(label, str) and label:
        return layer_part(label)
    return "Unknown"


def _safe_class(label: str | float | None) -> str:
    if isinstance(label, str) and label:
        return class_part(label)
    return "Unknown"


def _map_series(
    series: pd.Series, mapping: Mapping[str, str], fallback: str = "Unknown"
) -> pd.Series:
    mapped = series.map(mapping)
    mapped = mapped.fillna(fallback)
    return mapped.astype(str)


def _column_or_default(
    df: pd.DataFrame, column: str, default: float | str = np.nan
) -> pd.Series:
    if column in df:
        return df[column]
    return pd.Series(default, index=df.index)


def _assign_feature(df: pd.DataFrame, name: str, series: pd.Series | None) -> None:
    if series is None:
        return
    aligned = series.reindex(df.index)
    df[name] = aligned.astype(float)


def _select_numeric_columns(
    df: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame | None:
    available = [col for col in columns if col in df.columns]
    if not available:
        return None
    subset = df[available].astype(float)
    subset = subset.replace([np.inf, -np.inf], np.nan)
    return subset


def _zscore_columns(data: pd.DataFrame) -> pd.DataFrame:
    centered = data.sub(data.mean(axis=0), axis=1)
    std = data.std(axis=0, ddof=0).replace(0, np.nan)
    return centered.divide(std, axis=1)


def _composite_score(data: pd.DataFrame) -> pd.Series:
    if data is None or data.empty:
        return pd.Series(np.nan, index=data.index if data is not None else [])
    zscores = _zscore_columns(data)
    score = zscores.mean(axis=1, skipna=True)
    return score


def _compute_activity_profiles(features: pd.DataFrame) -> None:
    selectivity_data = _select_numeric_columns(features, SELECTIVITY_METRICS)
    rate_data = _select_numeric_columns(features, RATE_METRICS)
    if rate_data is not None:
        rate_data = rate_data.clip(lower=0)
        rate_data = rate_data.transform(np.log1p)

    selectivity_score = _composite_score(selectivity_data).reindex(features.index)
    rate_score = _composite_score(rate_data).reindex(features.index)

    features["activity_selectivity_score"] = selectivity_score
    features["activity_rate_score"] = rate_score
    features["activity_discriminant"] = selectivity_score - rate_score

    labels = pd.Series("Unknown", index=features.index, dtype=object)
    if selectivity_score.notna().any():
        labels.loc[selectivity_score > 0] = "selective"
        labels.loc[selectivity_score <= 0] = "non_selective"
    features["activity_label"] = labels


def _compute_edge_metrics(edges: pd.DataFrame, key: str) -> pd.DataFrame:
    if edges.empty:
        columns = [f"{key}_{metric}" for metric in EDGE_METRICS]
        return pd.DataFrame(columns=columns, dtype=float)

    signed_group = edges.groupby(key)["syn_weight"]
    abs_weights = edges["syn_weight"].abs()

    metrics = pd.DataFrame(index=signed_group.count().index)
    metrics[f"{key}_degree"] = signed_group.count().astype(float)
    metrics[f"{key}_weight_sum"] = signed_group.sum().astype(float)
    metrics[f"{key}_weight_mean"] = signed_group.mean().astype(float)
    metrics[f"{key}_weight_std"] = signed_group.std().astype(float)

    abs_sum = abs_weights.groupby(edges[key]).sum().astype(float)
    abs_mean = abs_weights.groupby(edges[key]).mean().astype(float)
    metrics[f"{key}_weight_sum_abs"] = abs_sum
    metrics[f"{key}_weight_mean_abs"] = abs_mean

    return metrics


def collect_neuron_features(
    base_dir: str | Path,
    network_type: str,
    selectivity_path: Path,
    *,
    core_only: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame of per-neuron features for the specified network."""
    base_dir = str(base_dir)
    t_start = time.perf_counter()
    loader = PropertyLoader(Path(selectivity_path))
    net_id = parse_network_id(base_dir)

    t_edges = time.perf_counter()
    edges, node_df = prepare_edge_table(base_dir, network_type)
    if edges.empty:
        raise ValueError(f"No edges available for {base_dir} ({network_type}).")
    edge_elapsed = time.perf_counter() - t_edges

    nodes = node_df.copy()
    all_node_ids = nodes.index.to_numpy()
    if core_only and "core" in nodes.columns:
        nodes = nodes[nodes["core"]]
    if nodes.empty:
        raise ValueError("No neurons selected after core filter.")

    features = pd.DataFrame(index=nodes.index.copy())
    features.index.name = "node_id"
    features["network_id"] = net_id
    features["base_dir"] = Path(base_dir).name
    features["network_type"] = network_type
    features["pop_name"] = nodes["pop_name"].astype(str)
    features["population"] = _column_or_default(nodes, "population", "v1").astype(str)
    features["ei"] = _column_or_default(nodes, "ei", "?").astype(str)
    features["model_type"] = _column_or_default(nodes, "model_type", "unknown").astype(
        str
    )
    features["tuning_angle"] = _column_or_default(nodes, "tuning_angle")
    features["x"] = _column_or_default(nodes, "x")
    features["z"] = _column_or_default(nodes, "z")
    features["radius"] = np.sqrt(features["x"] ** 2 + features["z"] ** 2)

    # Cell-type labels and aggregations
    ctdf = nu.get_cell_type_table()["cell_type"]
    features["cell_type"] = _map_series(nodes["pop_name"], ctdf)
    features["cell_type_l5_agg"] = features["cell_type"].apply(
        lambda label: map_l5(label, "aggregate")
    )
    features["cell_type_inh_family"] = features["cell_type"].apply(
        lambda label: map_inhibitory(label, "family")
    )
    features["cell_layer"] = features["cell_type"].apply(_safe_layer)
    features["cell_class"] = features["cell_type"].apply(_safe_class)
    features["is_inhibitory"] = (
        features["ei"].isin(["i", "I", "inh", "Inh"]).astype(int)
    )

    # Load scalar properties
    selectivity = loader.selectivity_series(network_type, net_id)
    osi_df = loader.osi_df(base_dir, network_type, net_id)
    oracle = loader.oracle_series(base_dir, network_type, net_id)
    natural_rate = loader.natural_image_rate(
        base_dir, network_type, net_id, all_node_ids
    )
    dg_metrics = loader.dg_rates(base_dir, network_type, net_id)
    dsi_series = loader.dsi_series(base_dir, network_type, net_id)

    _assign_feature(features, "image_selectivity", selectivity)
    if osi_df is not None:
        _assign_feature(features, "orientation_selectivity", osi_df.get("OSI"))
        _assign_feature(features, "firing_rate", osi_df.get("Ave_Rate(Hz)"))
    _assign_feature(features, "oracle_score", oracle)
    _assign_feature(features, "natural_image_evoked_rate", natural_rate)
    for name, series in dg_metrics.items():
        _assign_feature(features, name, series)
    _assign_feature(features, "dg_dsi", dsi_series)

    # Edge-derived metrics
    edges = edges.copy()
    edges["syn_weight"] = edges["syn_weight"].astype(float)
    in_metrics = _compute_edge_metrics(edges, "target_id")
    out_metrics = _compute_edge_metrics(edges, "source_id")
    in_metrics.index.name = "node_id"
    out_metrics.index.name = "node_id"

    features = features.join(
        in_metrics.rename(columns=lambda c: c.replace("target_id_", "incoming_")),
        how="left",
    )
    features = features.join(
        out_metrics.rename(columns=lambda c: c.replace("source_id_", "outgoing_")),
        how="left",
    )

    _compute_activity_profiles(features)

    features = features.astype(
        {col: float for col in features.columns if features[col].dtype == "float64"}
    )

    total_elapsed = time.perf_counter() - t_start
    features.attrs["timing"] = {
        "edges": edge_elapsed,
        "total": total_elapsed,
    }
    return features


def prepare_numeric_matrix(
    df: pd.DataFrame,
    exclude: Iterable[str] | None = None,
    include: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return numeric feature subset with NaNs replaced by column medians."""
    numeric = df.select_dtypes(include=[np.number]).copy()
    if include is not None:
        include = [col for col in include if col in numeric.columns]
        if not include:
            raise ValueError("No requested numeric columns found in dataframe.")
        numeric = numeric.loc[:, include]
    if exclude:
        numeric = numeric.drop(columns=list(exclude), errors="ignore")
    numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    medians = numeric.median(axis=0, skipna=True)
    numeric = numeric.fillna(medians)
    numeric = numeric.astype(float)
    return numeric
