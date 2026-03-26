#!/usr/bin/env python3
"""Dash app for exploring neuron-level UMAP embeddings."""
from __future__ import annotations

import os
import socket
from itertools import cycle
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence

import dash
from dash import Dash, Input, Output, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from analysis_shared.neuron_features import CORE_FEATURE_SPECS

DATA_PATH = BASE_DIR / "cell_categorization/core_nll_0_neuron_features.parquet"
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run cell_categorization/build_neuron_umap.py to generate it."
    )

DF = pd.read_parquet(DATA_PATH)
DF["node_id"] = DF["node_id"].astype(int)
DF["inhibitory_label"] = DF["is_inhibitory"].map({0: "Exc", 1: "Inh"})

QUALITATIVE_SEQUENCE: List[str] = (
    px.colors.qualitative.Bold
    + px.colors.qualitative.Safe
    + px.colors.qualitative.Dark24
    + px.colors.qualitative.Light24
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Alphabet
)
UNKNOWN_COLOR = "#666666"


def _assign_category_colors(
    values: Iterable[str], base: Dict[str, str] | None = None
) -> Dict[str, str]:
    mapping: Dict[str, str] = {} if base is None else dict(base)
    assigned = set(mapping.values())
    palette = [color for color in QUALITATIVE_SEQUENCE if color not in assigned]
    if not palette:
        palette = list(QUALITATIVE_SEQUENCE)
    color_iter = cycle(palette)
    for val in sorted(values):
        if not isinstance(val, str) or not val or val in mapping:
            continue
        mapping[val] = next(color_iter)
    return mapping


def _build_category_color_maps(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    category_map: Dict[str, Dict[str, str]] = {}

    cell_type_values = df["cell_type"].dropna().unique()
    cell_type_map = _assign_category_colors(cell_type_values)
    category_map["cell_type"] = cell_type_map

    l5_values = df["cell_type_l5_agg"].dropna().unique()
    l5_base = {val: cell_type_map[val] for val in l5_values if val in cell_type_map}
    category_map["cell_type_l5_agg"] = _assign_category_colors(l5_values, l5_base)

    inh_values = df["cell_type_inh_family"].dropna().unique()
    inh_base = {val: cell_type_map[val] for val in inh_values if val in cell_type_map}
    category_map["cell_type_inh_family"] = _assign_category_colors(inh_values, inh_base)

    class_values = df["cell_class"].dropna().unique()
    category_map["cell_class"] = _assign_category_colors(class_values)

    layer_values = sorted(df["cell_layer"].dropna().unique())
    layer_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    layer_map = {
        layer: layer_palette[i % len(layer_palette)]
        for i, layer in enumerate(layer_values)
    }
    category_map["cell_layer"] = layer_map

    category_map["inhibitory_label"] = {"Exc": "#1f77b4", "Inh": "#ff7f0e"}

    category_map["activity_label"] = {
        "selective": "#17becf",
        "non_selective": "#e377c2",
    }

    for cmap in category_map.values():
        cmap.setdefault("Unknown", UNKNOWN_COLOR)

    return category_map


CATEGORY_COLOR_MAP = _build_category_color_maps(DF)

NUMERIC_EXCLUDE = {"network_id", "umap_x", "umap_y", "is_inhibitory"}
NUMERIC_COLUMNS = [
    col
    for col in DF.select_dtypes(include=[np.number]).columns
    if col not in NUMERIC_EXCLUDE
]

CORE_LABELS = {spec.name: spec.description for spec in CORE_FEATURE_SPECS}

NUMERIC_OPTIONS: List[dict[str, str]] = []
for name, label in CORE_LABELS.items():
    if name in NUMERIC_COLUMNS:
        NUMERIC_OPTIONS.append({"label": label, "value": name})
        NUMERIC_COLUMNS.remove(name)

for col in sorted(NUMERIC_COLUMNS):
    pretty = col.replace("_", " ").title()
    NUMERIC_OPTIONS.append({"label": pretty, "value": col})

CATEGORICAL_OPTIONS: Sequence[dict[str, str]] = (
    {"label": "Cell type", "value": "cell_type"},
    {"label": "Cell type (L5 aggregated)", "value": "cell_type_l5_agg"},
    {"label": "Cell type (PV/SST/VIP families)", "value": "cell_type_inh_family"},
    {"label": "Cell class", "value": "cell_class"},
    {"label": "Cell layer", "value": "cell_layer"},
    {"label": "Population name", "value": "pop_name"},
    {"label": "Exc/Inh flag", "value": "inhibitory_label"},
    {"label": "Model type", "value": "model_type"},
    {"label": "Activity cluster", "value": "activity_label"},
)

CELL_CLASS_OPTIONS = [
    {"label": cls, "value": cls} for cls in sorted(DF["cell_class"].dropna().unique())
]

HOVER_COLUMNS = [
    "node_id",
    "cell_type",
    "cell_class",
    "cell_layer",
    "pop_name",
    "image_selectivity",
    "firing_rate",
    "orientation_selectivity",
    "oracle_score",
    "incoming_degree",
    "outgoing_degree",
    "incoming_weight_mean_abs",
    "outgoing_weight_mean_abs",
]

HOVER_DATA = {col: True for col in HOVER_COLUMNS if col in DF.columns}

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H2("Neuron Feature UMAP Explorer"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Color mode"),
                        dcc.RadioItems(
                            id="color-mode",
                            options=[
                                {"label": "Numeric feature", "value": "numeric"},
                                {"label": "Categorical label", "value": "categorical"},
                            ],
                            value="numeric",
                            inline=True,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Numeric feature"),
                        dcc.Dropdown(
                            id="numeric-feature",
                            options=NUMERIC_OPTIONS,
                            value=NUMERIC_OPTIONS[0]["value"],
                            clearable=False,
                        ),
                    ],
                    id="numeric-feature-container",
                    style={"width": "320px"},
                ),
                html.Div(
                    [
                        html.Label("Categorical feature"),
                        dcc.Dropdown(
                            id="categorical-feature",
                            options=CATEGORICAL_OPTIONS,
                            value="cell_type",
                            clearable=False,
                        ),
                    ],
                    id="categorical-feature-container",
                    style={"width": "320px", "display": "none"},
                ),
                html.Div(
                    [
                        html.Label("Cell classes"),
                        dcc.Dropdown(
                            id="cell-class-filter",
                            options=CELL_CLASS_OPTIONS,
                            value=[opt["value"] for opt in CELL_CLASS_OPTIONS],
                            multi=True,
                        ),
                    ],
                    style={"width": "280px"},
                ),
                html.Div(
                    [
                        html.Label("Plot mode"),
                        dcc.RadioItems(
                            id="plot-mode",
                            options=[
                                {"label": "UMAP", "value": "umap"},
                                {
                                    "label": "Selectivity vs rate",
                                    "value": "scatter",
                                },
                            ],
                            value="umap",
                            inline=True,
                        ),
                    ],
                    style={"width": "240px"},
                ),
                html.Div(
                    [
                        html.Label("Point size"),
                        dcc.Slider(
                            id="point-size",
                            min=3,
                            max=12,
                            value=6,
                            step=1,
                            marks={i: str(i) for i in range(3, 13, 2)},
                        ),
                    ],
                    style={"width": "240px"},
                ),
                html.Div(
                    [
                        html.Label("Opacity"),
                        dcc.Slider(
                            id="point-opacity",
                            min=0.2,
                            max=1.0,
                            value=0.8,
                            step=0.1,
                            marks={0.2: "0.2", 0.5: "0.5", 0.8: "0.8", 1.0: "1.0"},
                        ),
                    ],
                    style={"width": "240px"},
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "20px",
                "marginBottom": "20px",
            },
        ),
        dcc.Graph(id="umap-graph", config={"displaylogo": False}),
        html.Div(id="stats-text", style={"marginTop": "12px"}),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "12px"},
)


@app.callback(
    Output("numeric-feature-container", "style"),
    Output("categorical-feature-container", "style"),
    Input("color-mode", "value"),
)
def toggle_dropdowns(color_mode: str):
    num_style = {"width": "320px"}
    cat_style = {"width": "320px"}
    if color_mode == "numeric":
        cat_style["display"] = "none"
    else:
        num_style["display"] = "none"
    return num_style, cat_style


@app.callback(
    Output("umap-graph", "figure"),
    Output("stats-text", "children"),
    Input("color-mode", "value"),
    Input("numeric-feature", "value"),
    Input("categorical-feature", "value"),
    Input("cell-class-filter", "value"),
    Input("plot-mode", "value"),
    Input("point-size", "value"),
    Input("point-opacity", "value"),
)
def update_graph(
    color_mode: str,
    numeric_feature: str,
    categorical_feature: str,
    cell_class_filter: Sequence[str],
    plot_mode: str,
    point_size: int,
    point_opacity: float,
):
    df = DF.copy()
    if cell_class_filter:
        df = df[df["cell_class"].isin(cell_class_filter)]
    count = len(df)

    if count == 0:
        return px.scatter(), "No neurons match the current filters."

    if plot_mode == "scatter":
        required = df[["activity_selectivity_score", "activity_rate_score"]].dropna()
        if required.empty:
            return px.scatter(), "Activity score data unavailable."
        plot_df = df.loc[required.index].copy()
        x_col = "activity_selectivity_score"
        y_col = "activity_rate_score"
        x_label = "Selectivity z-score"
        y_label = "Firing-rate z-score"

        if color_mode == "numeric":
            feature = numeric_feature
            label = next(
                (opt["label"] for opt in NUMERIC_OPTIONS if opt["value"] == feature),
                feature,
            )
            color_values = plot_df[feature].astype(float)
            color_values = color_values.replace([np.inf, -np.inf], np.nan)
            median = color_values.median(skipna=True)
            color_values = color_values.fillna(median if np.isfinite(median) else 0.0)
            plot_df["color_value"] = color_values
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color="color_value",
                color_continuous_scale="Viridis",
                render_mode="webgl",
                hover_data=HOVER_DATA,
            )
            fig.update_coloraxes(colorbar_title=label)
        else:
            feature = categorical_feature
            label = next(
                (
                    opt["label"]
                    for opt in CATEGORICAL_OPTIONS
                    if opt["value"] == feature
                ),
                feature,
            )
            categories = plot_df[feature].fillna("Unknown").astype(str)
            plot_df[feature] = categories
            unique_values = sorted(categories.unique())
            color_map = CATEGORY_COLOR_MAP.get(feature, {})
            color_discrete_map = {
                cat: color_map[cat]
                for cat in unique_values
                if cat in color_map and isinstance(color_map[cat], str)
            }
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color=feature,
                render_mode="webgl",
                hover_data=HOVER_DATA,
                category_orders={feature: unique_values},
                color_discrete_map=color_discrete_map if color_discrete_map else None,
            )
            fig.update_layout(legend_title=label)

        fig.update_traces(
            marker=dict(size=point_size, opacity=point_opacity, line=dict(width=0))
        )
        fig.update_layout(
            height=720,
            margin=dict(l=20, r=20, t=40, b=40),
            xaxis_title=x_label,
            yaxis_title=y_label,
        )
        return fig, f"Displaying {len(plot_df):,} neurons."

    if color_mode == "numeric":
        feature = numeric_feature
        label = next(
            (opt["label"] for opt in NUMERIC_OPTIONS if opt["value"] == feature),
            feature,
        )
        plot_df = df.copy()
        color_values = plot_df[feature].astype(float)
        if not np.isfinite(color_values).any():
            color_values = np.zeros_like(color_values)
        else:
            median = np.nanmedian(color_values)
            if not np.isfinite(median):
                median = 0.0
            color_values = np.where(np.isfinite(color_values), color_values, median)
        plot_df["color_value"] = color_values
        fig = px.scatter(
            plot_df,
            x="umap_x",
            y="umap_y",
            color="color_value",
            color_continuous_scale="Viridis",
            render_mode="webgl",
            hover_data=HOVER_DATA,
        )
        fig.update_coloraxes(colorbar_title=label)
    else:
        feature = categorical_feature
        label = next(
            (opt["label"] for opt in CATEGORICAL_OPTIONS if opt["value"] == feature),
            feature,
        )
        plot_df = df.copy()
        categories = plot_df[feature].fillna("Unknown").astype(str)
        plot_df[feature] = categories
        unique_values = sorted(categories.unique())
        color_map = CATEGORY_COLOR_MAP.get(feature, {})
        color_discrete_map = {
            cat: color_map[cat]
            for cat in unique_values
            if cat in color_map and isinstance(color_map[cat], str)
        }
        category_orders = {feature: unique_values} if len(unique_values) <= 60 else None
        fig = px.scatter(
            plot_df,
            x="umap_x",
            y="umap_y",
            color=feature,
            render_mode="webgl",
            hover_data=HOVER_DATA,
            category_orders=category_orders,
            color_discrete_map=color_discrete_map if color_discrete_map else None,
        )
        fig.update_layout(legend_title=label)

    fig.update_traces(
        marker=dict(size=point_size, opacity=point_opacity, line=dict(width=0))
    )
    fig.update_layout(
        height=720,
        margin=dict(l=20, r=20, t=40, b=40),
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    stats = f"Displaying {count:,} neurons."
    return fig, stats


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
