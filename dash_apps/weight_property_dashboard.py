#!/usr/bin/env python3
"""Interactive explorer for weight vs cell-property correlations."""

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import dash
from dash import Dash, Input, Output, dash_table, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from analysis_shared.type_aggregation import aggregate_square_sum, build_type_mapping

SURVEY_DIR = Path(__file__).resolve().parent.parent / "survey"
CACHE_TEMPLATE = "weight_property_stats_{network}.pkl"

NETWORK_OPTIONS = [
    {"label": "Bio-trained", "value": "bio_trained"},
    {"label": "Plain", "value": "plain"},
    {"label": "Naive", "value": "naive"},
]

PROPERTY_OPTIONS = [
    {"label": "Image selectivity", "value": "image_selectivity"},
    {"label": "DG orientation selectivity", "value": "orientation_selectivity"},
    {"label": "Mean firing rate", "value": "firing_rate"},
    {"label": "Oracle correlation", "value": "oracle_score"},
    {"label": "Total in-degree", "value": "in_degree"},
    {"label": "Total out-degree", "value": "out_degree"},
    {"label": "Natural-image evoked rate", "value": "natural_image_evoked_rate"},
    {"label": "DG spontaneous rate", "value": "dg_spont_rate"},
    {"label": "DG evoked rate", "value": "dg_evoked_rate"},
    {"label": "DG mean rate", "value": "dg_mean_rate"},
    {"label": "DG peak rate", "value": "dg_peak_rate"},
    {"label": "DG direction selectivity", "value": "dg_dsi"},
]

ORIENTATION_OPTIONS = [
    {"label": "Outgoing weights (source properties)", "value": "outgoing"},
    {"label": "Incoming weights (target properties)", "value": "incoming"},
]

WEIGHT_MODE_OPTIONS = [
    {"label": "Per-connection weights", "value": "edge"},
    {"label": "Total weight per neuron", "value": "total"},
]

L5_OPTIONS = [
    {"label": "Split L5 subtypes (IT/ET/NP)", "value": "split"},
    {"label": "Aggregate L5 excitatory", "value": "aggregate"},
]

INH_OPTIONS = [
    {"label": "Layer-specific inhibitory types", "value": "layer"},
    {"label": "Aggregate PV/SST/VIP families", "value": "family"},
]

SORT_OPTIONS = [
    {"label": "Layer-first order", "value": "layer"},
    {"label": "Class-first order", "value": "class"},
]

DEFAULT_MIN_COUNT = 200


@dataclass
class CachedStats:
    types: List[str]
    stats: Dict[str, Dict[str, Dict[str, np.ndarray]]]


def load_cache(network: str) -> CachedStats:
    path = SURVEY_DIR / CACHE_TEMPLATE.format(network=network)
    if not path.exists():
        raise FileNotFoundError(f"Missing cache for {network}: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    types = payload["types"]
    stats = payload["properties"]
    return CachedStats(types=types, stats=stats)


def aggregate_arrays(
    entry: Dict[str, np.ndarray], indices: np.ndarray, out_size: int
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for name, arr in entry.items():
        result[name] = aggregate_square_sum(arr, indices, out_size)
    return result


def compute_correlation(entry: Dict[str, np.ndarray], min_count: int) -> np.ndarray:
    count = entry["count"]
    sum_w = entry["sum_w"]
    sum_w2 = entry["sum_w2"]
    sum_p = entry["sum_p"]
    sum_p2 = entry["sum_p2"]
    sum_wp = entry["sum_wp"]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_w = sum_w / count
        mean_p = sum_p / count
        var_w = sum_w2 - count * mean_w * mean_w
        var_p = sum_p2 - count * mean_p * mean_p
        cov = sum_wp - count * mean_w * mean_p

    corr = np.full_like(count, np.nan, dtype=float)
    mask = (count >= min_count) & (var_w > 0) & (var_p > 0)
    corr[mask] = cov[mask] / np.sqrt(var_w[mask] * var_p[mask])
    corr = np.clip(corr, -1.0, 1.0)
    return corr


def build_heatmap(
    cache: CachedStats,
    property_name: str,
    orientation: str,
    weight_mode: str,
    l5_mode: str,
    inh_mode: str,
    sort_mode: str,
    min_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mapping = build_type_mapping(cache.types, l5_mode, inh_mode, sort_mode)
    stats_entry = cache.stats[property_name][orientation][weight_mode]
    aggregated = aggregate_arrays(stats_entry, mapping.indices, len(mapping.labels))
    corr = compute_correlation(aggregated, min_count)
    corr_df = pd.DataFrame(corr, index=mapping.labels, columns=mapping.labels)
    counts_df = pd.DataFrame(
        aggregated["count"], index=mapping.labels, columns=mapping.labels
    )
    return corr_df, counts_df


def summarize_matrix(df: pd.DataFrame, counts: pd.DataFrame) -> List[dict]:
    values = df.stack(dropna=False)
    if values.empty:
        return []
    max_idx = values.idxmax()
    min_idx = values.idxmin()
    summary = [
        {
            "stat": "Max r",
            "value": f"{values.max():.3f}",
            "selectivity": max_idx[0],
            "partner": max_idx[1],
            "pairs": int(counts.loc[max_idx]),
        },
        {
            "stat": "Min r",
            "value": f"{values.min():.3f}",
            "selectivity": min_idx[0],
            "partner": min_idx[1],
            "pairs": int(counts.loc[min_idx]),
        },
        {
            "stat": "Mean |r|",
            "value": f"{np.nanmean(np.abs(values.values)):.3f}",
            "selectivity": "-",
            "partner": "-",
            "pairs": int(counts.values.sum()),
        },
    ]
    return summary


def make_figure(df: pd.DataFrame, orientation: str) -> px.imshow:
    vmax = np.nanmax(np.abs(df.values))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.1
    fig = px.imshow(
        df,
        color_continuous_scale="RdBu_r",
        zmin=-vmax,
        zmax=vmax,
        aspect="auto",
    )
    fig.update_layout(
        height=680,
        margin=dict(l=60, r=60, t=60, b=60),
        coloraxis_colorbar=dict(title="Pearson r"),
    )
    fig.update_xaxes(side="top", title="Target cell type")
    fig.update_yaxes(title="Source cell type")
    if orientation == "outgoing":
        fig.update_yaxes(title="Source cell type (property)")
    else:
        fig.update_xaxes(side="top", title="Target cell type (property)")
    return fig


def create_app() -> Dash:
    caches = {opt["value"]: load_cache(opt["value"]) for opt in NETWORK_OPTIONS}

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("Weight vs Property Correlation Explorer"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Network"),
                            dcc.Dropdown(
                                id="network",
                                options=NETWORK_OPTIONS,
                                value="bio_trained",
                                clearable=False,
                            ),
                        ],
                        className="control",
                    ),
                    html.Div(
                        [
                            html.Label("Property"),
                            dcc.Dropdown(
                                id="property",
                                options=PROPERTY_OPTIONS,
                                value="image_selectivity",
                                clearable=False,
                            ),
                        ],
                        className="control",
                    ),
                    html.Div(
                        [
                            html.Label("Weights orientation"),
                            dcc.Dropdown(
                                id="orientation",
                                options=ORIENTATION_OPTIONS,
                                value="outgoing",
                                clearable=False,
                            ),
                        ],
                        className="control",
                    ),
                    html.Div(
                        [
                            html.Label("Weight metric"),
                            dcc.Dropdown(
                                id="weight-mode",
                                options=WEIGHT_MODE_OPTIONS,
                                value="edge",
                                clearable=False,
                            ),
                        ],
                        className="control",
                    ),
                    html.Div(
                        [
                            html.Label("L5 granularity"),
                            dcc.Dropdown(
                                id="l5-mode",
                                options=L5_OPTIONS,
                                value="split",
                                clearable=False,
                            ),
                        ],
                        className="control",
                    ),
                    html.Div(
                        [
                            html.Label("Inhibitory granularity"),
                            dcc.Dropdown(
                                id="inh-mode",
                                options=INH_OPTIONS,
                                value="layer",
                                clearable=False,
                            ),
                        ],
                        className="control",
                    ),
                    html.Div(
                        [
                            html.Label("Sort order"),
                            dcc.Dropdown(
                                id="sort-mode",
                                options=SORT_OPTIONS,
                                value="layer",
                                clearable=False,
                            ),
                        ],
                        className="control",
                    ),
                    html.Div(
                        [
                            html.Label("Min edges per pair"),
                            dcc.Input(
                                id="min-count",
                                type="number",
                                value=DEFAULT_MIN_COUNT,
                                min=0,
                                step=50,
                            ),
                        ],
                        className="control",
                    ),
                ],
                className="controls",
            ),
            dcc.Graph(id="weight-heatmap"),
            html.H4("Summary"),
            dash_table.DataTable(
                id="summary-table",
                columns=[
                    {"name": "Statistic", "id": "stat"},
                    {"name": "Value", "id": "value"},
                    {"name": "Selectivity type", "id": "selectivity"},
                    {"name": "Partner type", "id": "partner"},
                    {"name": "Edges", "id": "pairs"},
                ],
                data=[],
                style_table={"maxWidth": "600px"},
            ),
        ]
    )

    @app.callback(
        Output("weight-heatmap", "figure"),
        Output("summary-table", "data"),
        Input("network", "value"),
        Input("property", "value"),
        Input("orientation", "value"),
        Input("weight-mode", "value"),
        Input("l5-mode", "value"),
        Input("inh-mode", "value"),
        Input("sort-mode", "value"),
        Input("min-count", "value"),
    )
    def update_figure(
        network: str,
        prop: str,
        orientation: str,
        weight_mode: str,
        l5_mode: str,
        inh_mode: str,
        sort_mode: str,
        min_count: int,
    ):
        cache = caches[network]
        corr_df, count_df = build_heatmap(
            cache,
            prop,
            orientation,
            weight_mode,
            l5_mode,
            inh_mode,
            sort_mode,
            int(min_count) if min_count is not None else DEFAULT_MIN_COUNT,
        )
        fig = make_figure(corr_df, orientation)
        summary = summarize_matrix(corr_df, count_df)
        return fig, summary

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the weight-property Dash explorer"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()

    app = create_app()
    if args.no_run:
        return
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
