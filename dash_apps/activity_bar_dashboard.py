#!/usr/bin/env python3
"""Dash app for exploring activity clusters across neuron categories."""
from __future__ import annotations

import os
import socket
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import dash
from dash import Dash, Input, Output, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

DATA_PATH = BASE_DIR / "cell_categorization/core_nll_0_neuron_features.parquet"
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run cell_categorization/build_neuron_umap.py to generate it."
    )

DF = pd.read_parquet(DATA_PATH)
DF["activity_label"] = DF["activity_label"].fillna("Unknown")

GROUP_OPTIONS: Sequence[dict[str, str]] = (
    {"label": "Cell type", "value": "cell_type"},
    {"label": "Cell type (L5 aggregated)", "value": "cell_type_l5_agg"},
    {"label": "Cell type (PV/SST/VIP families)", "value": "cell_type_inh_family"},
    {"label": "Cell class", "value": "cell_class"},
    {"label": "Cell layer", "value": "cell_layer"},
    {"label": "Population name", "value": "pop_name"},
    {"label": "Exc/Inh flag", "value": "inhibitory_label"},
    {"label": "Model type", "value": "model_type"},
)

ACTIVITY_COLORS: Dict[str, str] = {
    "selective": "#17becf",
    "non_selective": "#e377c2",
    "Unknown": "#666666",
}

METRIC_OPTIONS: Sequence[dict[str, str]] = (
    {"label": "Incoming degree", "value": "incoming_degree"},
    {"label": "Outgoing degree", "value": "outgoing_degree"},
    {"label": "Incoming weight sum", "value": "incoming_weight_sum"},
    {"label": "Outgoing weight sum", "value": "outgoing_weight_sum"},
    {"label": "Incoming weight |sum|", "value": "incoming_weight_sum_abs"},
    {"label": "Outgoing weight |sum|", "value": "outgoing_weight_sum_abs"},
    {"label": "Incoming weight mean", "value": "incoming_weight_mean"},
    {"label": "Outgoing weight mean", "value": "outgoing_weight_mean"},
    {"label": "Incoming weight mean |abs|", "value": "incoming_weight_mean_abs"},
    {"label": "Outgoing weight mean |abs|", "value": "outgoing_weight_mean_abs"},
)

STAT_OPTIONS: Sequence[dict[str, str]] = (
    {"label": "Mean", "value": "mean"},
    {"label": "Median", "value": "median"},
)

ORDER_OPTIONS: Sequence[dict[str, str]] = (
    {"label": "Original", "value": "original"},
    {"label": "Fraction selective", "value": "fraction_selective"},
    {"label": "Fraction non-selective", "value": "fraction_non_selective"},
    {"label": "Metric (selective)", "value": "metric_selective"},
    {"label": "Metric (non-selective)", "value": "metric_non_selective"},
)

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H2("Activity Cluster Summary"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Grouping"),
                        dcc.Dropdown(
                            id="group-column",
                            options=GROUP_OPTIONS,
                            value="cell_type",
                            clearable=False,
                        ),
                    ],
                    style={"width": "260px"},
                ),
                html.Div(
                    [
                        html.Label("Metric"),
                        dcc.Dropdown(
                            id="metric-column",
                            options=METRIC_OPTIONS,
                            value="incoming_degree",
                            clearable=False,
                        ),
                    ],
                    style={"width": "260px"},
                ),
                html.Div(
                    [
                        html.Label("Statistic"),
                        dcc.RadioItems(
                            id="metric-stat",
                            options=STAT_OPTIONS,
                            value="mean",
                            inline=True,
                        ),
                    ],
                    style={"width": "240px"},
                ),
                html.Div(
                    [
                        html.Label("Category order"),
                        dcc.Dropdown(
                            id="order-mode",
                            options=ORDER_OPTIONS,
                            value="fraction_selective",
                            clearable=False,
                        ),
                    ],
                    style={"width": "260px"},
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "20px",
                "marginBottom": "20px",
            },
        ),
        html.Div(
            [
                dcc.Graph(id="fraction-graph", config={"displaylogo": False}),
                dcc.Graph(id="metric-graph", config={"displaylogo": False}),
            ],
            style={"display": "grid", "gap": "16px"},
        ),
        html.Div(id="summary-text", style={"marginTop": "12px"}),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "12px"},
)


def _apply_order(
    df: pd.DataFrame,
    order_mode: str,
    group_col: str,
    fractions: pd.DataFrame,
    metric: pd.DataFrame,
) -> pd.DataFrame:
    if order_mode == "original":
        return df
    if order_mode == "fraction_selective":
        key = fractions.get(("selective", "fraction"), pd.Series())
    elif order_mode == "fraction_non_selective":
        key = fractions.get(("non_selective", "fraction"), pd.Series())
    elif order_mode == "metric_selective":
        key = metric.get(("selective", "value"), pd.Series())
    elif order_mode == "metric_non_selective":
        key = metric.get(("non_selective", "value"), pd.Series())
    else:
        return df
    key = key.reindex(df[group_col]).fillna(0.0)
    df = df.assign(_order_key=key.values)
    df = df.sort_values("_order_key", ascending=False)
    return df.drop(columns="_order_key")


@app.callback(
    Output("fraction-graph", "figure"),
    Output("metric-graph", "figure"),
    Output("summary-text", "children"),
    Input("group-column", "value"),
    Input("metric-column", "value"),
    Input("metric-stat", "value"),
    Input("order-mode", "value"),
)
def update_figures(
    group_col: str,
    metric_col: str,
    stat: str,
    order_mode: str,
):
    if group_col not in DF.columns:
        raise dash.exceptions.PreventUpdate
    grouped = DF.groupby([group_col, "activity_label"], dropna=False)
    counts = grouped.size().rename("count").reset_index()
    totals = counts.groupby(group_col)["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals

    metric_series = DF[metric_col]
    if stat == "mean":
        metric_vals = grouped[metric_col].mean().rename("value").reset_index()
    else:
        metric_vals = grouped[metric_col].median().rename("value").reset_index()

    pivot_fraction = counts.pivot(
        index=group_col, columns="activity_label", values="fraction"
    )
    pivot_metric = metric_vals.pivot(
        index=group_col, columns="activity_label", values="value"
    )

    order_df = pd.DataFrame({group_col: pivot_fraction.index})
    order_df = _apply_order(
        order_df, order_mode, group_col, pivot_fraction, pivot_metric
    )
    ordered = order_df[group_col].tolist()

    counts[group_col] = pd.Categorical(
        counts[group_col], categories=ordered, ordered=True
    )
    metric_vals[group_col] = pd.Categorical(
        metric_vals[group_col], categories=ordered, ordered=True
    )

    frac_fig = px.bar(
        counts,
        x=group_col,
        y="fraction",
        color="activity_label",
        color_discrete_map=ACTIVITY_COLORS,
        category_orders={group_col: ordered},
        barmode="stack",
        text=(
            counts["fraction"].map(lambda v: f"{v:.2f}") if len(ordered) <= 30 else None
        ),
    )
    frac_fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=40, b=60),
        xaxis_title=group_col.replace("_", " ").title(),
        yaxis_title="Fraction of neurons",
        legend_title="Activity label",
    )
    frac_fig.update_yaxes(range=[0, 1], tickformat=".0%")

    metric_fig = px.bar(
        metric_vals,
        x=group_col,
        y="value",
        color="activity_label",
        color_discrete_map=ACTIVITY_COLORS,
        category_orders={group_col: ordered},
        barmode="group",
    )
    metric_fig.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=40, b=60),
        xaxis_title=group_col.replace("_", " ").title(),
        yaxis_title=f"{stat.title()} {metric_col.replace('_', ' ')}",
        legend_title="Activity label",
    )

    summary = (
        f"Total neurons: {len(DF):,}. Selective: {(DF['activity_label'] == 'selective').sum():,}"
        f", Non-selective: {(DF['activity_label'] == 'non_selective').sum():,}."
    )
    return frac_fig, metric_fig, summary


if __name__ == "__main__":

    def _pick_port(base: int = 8050, attempts: int = 100) -> int:
        for port in range(base, base + attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("0.0.0.0", port))
                except OSError:
                    continue
                return port
        raise RuntimeError("Unable to find free port in range")

    port_str = os.environ.get("PORT")
    port = int(port_str) if port_str else _pick_port()
    app.run(debug=True, host="0.0.0.0", port=port)
