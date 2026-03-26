#!/usr/bin/env python3
"""Dash app for exploring selectivity–degree correlation matrices."""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import dash
from dash import Dash, Input, Output, dash_table, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from analysis_shared.type_aggregation import (
    aggregate_square_sum,
    aggregate_square_weighted_mean,
    build_type_mapping,
)

SURVEY_DIR = BASE_DIR / "survey"

NETWORK_OPTIONS = [
    {"label": "Bio-trained", "value": "bio_trained"},
    {"label": "Plain", "value": "plain"},
    {"label": "Naive", "value": "naive"},
]

SELECTIVITY_OPTIONS = [
    {"label": "Target selectivity", "value": "target"},
    {"label": "Source selectivity", "value": "source"},
]

DEGREE_NODE_OPTIONS = [
    {"label": "Target degree", "value": "target"},
    {"label": "Source degree", "value": "source"},
]

MODE_OPTIONS = [
    {"label": "Incoming (in-degree)", "value": "incoming"},
    {"label": "Outgoing (out-degree)", "value": "outgoing"},
]

L5_OPTIONS = [
    {"label": "Split L5 subtypes (IT/ET/NP)", "value": "split"},
    {"label": "Aggregate L5 excitatory", "value": "aggregate"},
]

INH_OPTIONS = [
    {"label": "Layer-specific inhibitory types", "value": "layer"},
    {"label": "Aggregate PV/SST/VIP families", "value": "family"},
]

DEFAULT_SORT_MODE = "class"

TYPE_ORDER: Tuple[str, ...] = (
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
)


def _network_suffix(network: str) -> List[str]:
    if network == "bio_trained":
        return ["_bio_trained", "_bio", ""]
    return [f"_{network}"]


def _matrix_base_name(selectivity_role: str, degree_role: str, mode: str) -> str:
    deg_label = "in_degree" if mode == "incoming" else "out_degree"
    return f"selectivity_{selectivity_role}_vs_{degree_role}_{deg_label}"


def load_matrix(
    network: str, selectivity_role: str, degree_role: str, mode: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (correlation_df, count_df)."""
    base = _matrix_base_name(selectivity_role, degree_role, mode)
    candidates = [
        SURVEY_DIR / f"{base}{suffix}.csv" for suffix in _network_suffix(network)
    ]
    for path in candidates:
        if path.exists():
            corr = pd.read_csv(path, index_col=0)
            counts_path = path.with_name(path.stem + "_counts.csv")
            corr = corr.reindex(index=TYPE_ORDER, columns=TYPE_ORDER)
            if counts_path.exists():
                counts = (
                    pd.read_csv(counts_path, index_col=0)
                    .reindex(index=TYPE_ORDER, columns=TYPE_ORDER)
                    .fillna(0.0)
                )
            else:
                counts = pd.DataFrame(
                    np.zeros_like(corr.values, dtype=float),
                    index=corr.index,
                    columns=corr.columns,
                )
            return corr, counts
    raise FileNotFoundError(
        f"No matrix found for parameters: {network=}, {selectivity_role=}, {degree_role=}, {mode=}"
    )


def summarize_matrix(df: pd.DataFrame) -> List[dict]:
    """Compute headline stats for display in table."""
    flattened = df.stack(dropna=False).dropna()
    if flattened.empty:
        return []
    top_pos = flattened.idxmax()
    top_neg = flattened.idxmin()
    stats = [
        {
            "stat": "Max r",
            "value": f"{flattened.max():.3f}",
            "source": top_pos[0],
            "target": top_pos[1],
        },
        {
            "stat": "Min r",
            "value": f"{flattened.min():.3f}",
            "source": top_neg[0],
            "target": top_neg[1],
        },
        {
            "stat": "|r| mean",
            "value": f"{np.nanmean(np.abs(flattened.values)):.3f}",
            "source": "-",
            "target": "-",
        },
    ]
    return stats


def make_heatmap(
    df: pd.DataFrame,
    selectivity_role: str,
    degree_role: str,
) -> dash.no_update:
    vmax = np.nanmax(np.abs(df.values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 0.1
    fig = px.imshow(
        df,
        color_continuous_scale="RdBu_r",
        zmin=-vmax,
        zmax=vmax,
        aspect="auto",
    )
    fig.update_layout(
        height=640,
        margin=dict(l=60, r=60, t=60, b=60),
        coloraxis_colorbar=dict(title="Pearson r"),
    )
    fig.update_xaxes(side="top", title="Target cell type")
    fig.update_yaxes(title="Source cell type")
    if degree_role == "target":
        fig.update_xaxes(side="top", title="Target cell type (property)")
    if selectivity_role == "target":
        fig.update_yaxes(title="Target cell type (property)")
    return fig


def aggregate_views(
    corr: pd.DataFrame,
    counts: pd.DataFrame,
    l5_mode: str,
    inh_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mapping = build_type_mapping(TYPE_ORDER, l5_mode, inh_mode, DEFAULT_SORT_MODE)
    corr_values = corr.to_numpy(dtype=float)
    count_values = counts.to_numpy(dtype=float)
    agg_corr, _ = aggregate_square_weighted_mean(
        corr_values, count_values, mapping.indices, len(mapping.labels)
    )
    agg_counts = aggregate_square_sum(
        count_values, mapping.indices, len(mapping.labels)
    )
    corr_df = pd.DataFrame(agg_corr, index=mapping.labels, columns=mapping.labels)
    counts_df = pd.DataFrame(agg_counts, index=mapping.labels, columns=mapping.labels)
    return corr_df, counts_df


def create_app() -> Dash:
    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.H2("Selectivity–Degree Correlation Explorer"),
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
                        className="control-block",
                    ),
                    html.Div(
                        [
                            html.Label("Selectivity role"),
                            dcc.Dropdown(
                                id="selectivity-role",
                                options=SELECTIVITY_OPTIONS,
                                value="target",
                                clearable=False,
                            ),
                        ],
                        className="control-block",
                    ),
                    html.Div(
                        [
                            html.Label("Degree node"),
                            dcc.Dropdown(
                                id="degree-node",
                                options=DEGREE_NODE_OPTIONS,
                                value="target",
                                clearable=False,
                            ),
                        ],
                        className="control-block",
                    ),
                    html.Div(
                        [
                            html.Label("Degree mode"),
                            dcc.Dropdown(
                                id="mode",
                                options=MODE_OPTIONS,
                                value="incoming",
                                clearable=False,
                            ),
                        ],
                        className="control-block",
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
                        className="control-block",
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
                        className="control-block",
                    ),
                ],
                className="control-row",
            ),
            dcc.Graph(id="heatmap"),
            html.H4("Summary"),
            dash_table.DataTable(
                id="summary-table",
                columns=[
                    {"name": "Stat", "id": "stat"},
                    {"name": "Value", "id": "value"},
                    {"name": "Source type", "id": "source"},
                    {"name": "Target type", "id": "target"},
                ],
                data=[],
                style_table={"maxWidth": "520px"},
            ),
        ]
    )

    @app.callback(
        Output("heatmap", "figure"),
        Output("summary-table", "data"),
        Input("network", "value"),
        Input("selectivity-role", "value"),
        Input("degree-node", "value"),
        Input("mode", "value"),
        Input("l5-mode", "value"),
        Input("inh-mode", "value"),
    )
    def update_heatmap(
        network: str,
        sel_role: str,
        degree_role: str,
        mode: str,
        l5_mode: str,
        inh_mode: str,
    ):
        corr, counts = load_matrix(network, sel_role, degree_role, mode)
        corr_view, _ = aggregate_views(corr, counts, l5_mode, inh_mode)
        fig = make_heatmap(corr_view, sel_role, degree_role)
        summary = summarize_matrix(corr_view)
        return fig, summary

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the selectivity–degree Dash app")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to serve on (default: 8050)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Initialize the app without starting the server (useful for smoke tests)",
    )
    args = parser.parse_args()

    app = create_app()
    if args.no_run:
        return

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
