#!/usr/bin/env python3
"""Dash app for exploring selectivity vs outgoing connectivity by cell type."""

from __future__ import annotations

from itertools import cycle
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import dash
from dash import Dash, Input, Output, dcc, html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

FEATURES_PATH = (
    PROJECT_ROOT / "cell_categorization" / "core_nll_0_neuron_features.parquet"
)
if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"{FEATURES_PATH} not found. Run cell_categorization/build_neuron_umap.py first."
    )

DF = pd.read_parquet(FEATURES_PATH)
DF = DF[DF["outgoing_weight_sum"].notna()].copy()
DF["outgoing_weight_sum_abs"] = DF["outgoing_weight_sum_abs"].astype(float)
DF["outgoing_weight_sum"] = DF["outgoing_weight_sum"].astype(float)
DF["image_selectivity"] = DF["image_selectivity"].astype(float)

CELL_TYPES = sorted(DF["cell_type"].dropna().unique())


def assign_colors(values: Iterable[str]) -> Dict[str, str]:
    palette: List[str] = go.Figure().layout.colorway or [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    color_iter = cycle(palette)
    mapping: Dict[str, str] = {}
    for val in sorted(values):
        mapping[val] = mapping.get(val) or next(color_iter)
    return mapping


COLOR_MAP = assign_colors(CELL_TYPES)

DEFAULT_SELECTION = [ct for ct in CELL_TYPES if ct.startswith("L2/3")] or CELL_TYPES[:5]

WEIGHT_OPTIONS = [
    {"label": "Signed total weight", "value": "outgoing_weight_sum"},
    {"label": "Absolute total weight", "value": "outgoing_weight_sum_abs"},
]


def build_figure(data: pd.DataFrame, metric: str, title_suffix: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [None, {"type": "histogram"}],
            [{"type": "histogram"}, {"type": "scatter"}],
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        column_widths=[0.25, 0.75],
        row_heights=[0.25, 0.75],
        horizontal_spacing=0.04,
        vertical_spacing=0.04,
    )

    legend_added = set()
    for cell_type in sorted(data["cell_type"].unique()):
        subset = data[data["cell_type"] == cell_type]
        if subset.empty:
            continue
        color = COLOR_MAP.get(cell_type, "#555555")
        showlegend = cell_type not in legend_added

        fig.add_trace(
            go.Histogram(
                x=subset[metric],
                name=cell_type,
                marker_color=color,
                opacity=0.8,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Histogram(
                y=subset["image_selectivity"],
                name=cell_type,
                marker_color=color,
                opacity=0.8,
                showlegend=False,
                orientation="h",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=subset[metric],
                y=subset["image_selectivity"],
                mode="markers",
                marker=dict(color=color, size=6, opacity=0.7),
                name=cell_type,
                showlegend=showlegend,
                hovertemplate=(
                    "Cell type: %{customdata[0]}<br>Outgoing: %{x:.2f}<br>"
                    "Selectivity: %{y:.3f}<br>Node id: %{customdata[1]}<extra></extra>"
                ),
                customdata=np.column_stack((subset["cell_type"], subset["node_id"])),
            ),
            row=2,
            col=2,
        )

        legend_added.add(cell_type)

    fig.update_layout(
        barmode="stack",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=60, r=20, t=60, b=60),
    )

    fig.update_xaxes(title_text="Image selectivity", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Outgoing weight", row=2, col=2)
    fig.update_yaxes(title_text="Image selectivity", row=2, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=1)

    fig.update_layout(title=title_suffix)
    return fig


app = Dash(__name__)

app.layout = html.Div(
    [
        html.H2("Selectivity vs outgoing connectivity"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Cell types"),
                        dcc.Dropdown(
                            id="cell-type-selector",
                            options=[{"label": ct, "value": ct} for ct in CELL_TYPES],
                            value=DEFAULT_SELECTION,
                            multi=True,
                            placeholder="Select one or more cell types",
                        ),
                    ],
                    style={"width": "420px"},
                ),
                html.Div(
                    [
                        html.Label("Weight metric"),
                        dcc.RadioItems(
                            id="weight-metric",
                            options=WEIGHT_OPTIONS,
                            value="outgoing_weight_sum_abs",
                            inline=True,
                        ),
                    ]
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "columnGap": "40px",
                "rowGap": "16px",
                "alignItems": "flex-end",
            },
        ),
        html.Br(),
        html.Div(id="correlation-summary", style={"marginBottom": "12px"}),
        dcc.Graph(id="selectivity-outgoing-graph", style={"height": "720px"}),
        html.Div(
            "Scatter shows per-neuron image selectivity versus total outgoing synaptic weight. "
            "Marginal histograms reveal stacked distributions across the selected cell types.",
            style={"marginTop": "8px", "color": "#555"},
        ),
    ]
)


@app.callback(
    Output("selectivity-outgoing-graph", "figure"),
    Output("correlation-summary", "children"),
    Input("cell-type-selector", "value"),
    Input("weight-metric", "value"),
)
def update_graph(selected_types: List[str] | None, metric: str):
    if not selected_types:
        filtered = DF
    else:
        filtered = DF[DF["cell_type"].isin(selected_types)]

    title_suffix = "Selectivity vs outgoing weight"
    if not filtered.empty and len(filtered) >= 2:
        x = filtered[metric]
        y = filtered["image_selectivity"]
        r = float(np.corrcoef(x, y)[0, 1])
        summary = (
            f"Pearson r = {r:.3f} (n = {len(filtered)}) "
            f"using metric '{metric.replace('_', ' ')}'."
        )
    else:
        summary = "Not enough data to compute correlation."

    fig = build_figure(filtered, metric, title_suffix)
    fig.add_annotation(
        text=summary,
        xref="paper",
        yref="paper",
        x=0,
        y=1.12,
        showarrow=False,
        font=dict(size=12),
    )
    return fig, summary


if __name__ == "__main__":
    app.run(debug=True)
