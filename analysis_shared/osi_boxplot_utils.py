"""Utilities for loading OSI/DSI datasets and generating box plot panels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

from sonata.circuit import File

import utils
from plotting_utils import pick_core
from image_decoding.plot_utils import cell_type_order


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    basedir: Path | str
    metric_file: str
    radius: float = 200.0

    def metrics_path(self) -> Path:
        base = Path(self.basedir)
        return base / "metrics" / self.metric_file


def _load_core_dataframe(basedir: Path, metric_path: Path, radius: float) -> pd.DataFrame:
    net = File(basedir / "network" / "v1_nodes.h5", basedir / "network" / "v1_node_types.csv")
    v1df = net.nodes["v1"].to_dataframe()
    osi_df = pd.read_csv(metric_path, sep=" ")
    df = v1df.merge(osi_df)
    df["cell_type"] = df["pop_name"].map(utils.cell_type_df["cell_type_old"])
    df["ei"] = df["pop_name"].map(utils.cell_type_df["ei"])
    df = pick_core(df, radius=radius)
    return df


def load_osi_dataset(spec: DatasetSpec) -> pd.DataFrame:
    """Load a dataset described by *spec* and attach metadata columns."""
    metric_path = spec.metrics_path()
    if not metric_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metric_path}")

    basedir = Path(spec.basedir)
    if basedir.name == "neuropixels":
        df = pd.read_csv(metric_path, sep=" ")
    else:
        df = _load_core_dataframe(basedir, metric_path, spec.radius)

    df = df.copy()
    df.rename(
        columns={"max_mean_rate(Hz)": "Rate at preferred direction (Hz)"}, inplace=True
    )
    df["cell_type"] = df["cell_type"].str.replace(" ", "_", regex=False)
    df["cell_type"] = df["cell_type"].str.replace("L1_Htr3a", "L1_Inh", regex=False)

    nonresponding = df["Rate at preferred direction (Hz)"] < 0.5
    df.loc[nonresponding, ["OSI", "DSI"]] = np.nan

    df["data_type"] = spec.label
    return df


def load_osi_datasets(specs: Sequence[DatasetSpec]) -> pd.DataFrame:
    frames = []
    for spec in specs:
        try:
            frames.append(load_osi_dataset(spec))
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")
    if not frames:
        raise ValueError("No datasets could be loaded.")
    df = pd.concat(frames, ignore_index=True)

    order = build_cell_type_order(df["cell_type"].unique())
    df["cell_type"] = pd.Categorical(df["cell_type"], categories=order, ordered=True)
    df = df.sort_values("cell_type")
    return df


def build_cell_type_order(present_types: Iterable[str]) -> list[str]:
    base_order = cell_type_order()
    present_set = set(present_types)

    reordered: list[str] = []
    l5_subtypes = ["L5_ET", "L5_IT", "L5_NP"]

    for ct in base_order:
        if ct == "L5_Exc":
            for subtype in l5_subtypes:
                if subtype in present_set and subtype not in reordered:
                    reordered.append(subtype)
            if ct in present_set:
                reordered.append(ct)
        elif ct in l5_subtypes:
            # handled above when encountering L5_Exc
            continue
        elif ct in present_set:
            reordered.append(ct)

    # Append any remaining types not covered by base order
    for ct in present_types:
        if ct not in reordered:
            reordered.append(ct)

    return reordered


def _get_layer_borders(ticklabels: Sequence[plt.Text]) -> list[float]:
    borders = [-0.5]
    prev_layer = None
    for tk in ticklabels:
        label = tk.get_text()
        if not label:
            continue
        layer_token = label.split("_")[0]
        if prev_layer is None:
            prev_layer = layer_token
        elif layer_token != prev_layer:
            borders.append(tk.get_position()[0] - 0.5)
            prev_layer = layer_token
    if ticklabels:
        borders.append(ticklabels[-1].get_position()[0] + 0.5)
    return borders


def _shade_layers(ax: plt.Axes, borders: Sequence[float], ylim: tuple[float, float]) -> None:
    for i in range(0, len(borders) - 1, 2):
        left, right = borders[i], borders[i + 1]
        if right <= left:
            continue
        rect = Rectangle((left, ylim[0]), right - left, ylim[1] - ylim[0], alpha=0.08, color="k", zorder=-10)
        ax.add_patch(rect)


def _plot_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, ylim: tuple[float, float], palette: Mapping[str, str] | None) -> None:
    sns.boxplot(
        x="cell_type",
        y=metric,
        hue="data_type",
        data=df,
        ax=ax,
        width=0.7,
        palette=palette,
    )
    ax.tick_params(axis="x", labelrotation=45)
    if ax.get_legend() is not None:
        ax.legend(loc="upper right")
    ax.set_ylim(*ylim)
    ax.set_xlabel("")
    borders = _get_layer_borders(ax.get_xticklabels())
    _shade_layers(ax, borders, ylim)


def _plot_scatter(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, palette: Mapping[str, str] | None) -> None:
    sns.scatterplot(x=x, y=y, hue="data_type", data=df, ax=ax, s=5, palette=palette)
    ax.legend(loc="upper right")
    ax.set_xscale("log")


def plot_box_grid(
    df: pd.DataFrame,
    output_path: Path | str,
    palette: Mapping[str, str] | None = None,
    include_scatter: bool = True,
) -> None:
    """Render the standard box/violin grid and save it to *output_path*."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if include_scatter:
        fig, axs = plt.subplots(4, 2, figsize=(24, 12))
        _plot_panel(axs[0, 0], df, "Spont_Rate(Hz)", (0, 50), palette)
        _plot_panel(axs[0, 1], df, "Ave_Rate(Hz)", (0, 50), palette)
        _plot_panel(axs[1, 0], df, "Rate at preferred direction (Hz)", (0, 50), palette)
        _plot_panel(axs[2, 0], df, "DSI", (0, 1), palette)
        _plot_panel(axs[2, 1], df, "OSI", (0, 1), palette)
        _plot_scatter(
            axs[3, 0], df, "Rate at preferred direction (Hz)", "DSI", palette
        )
        _plot_scatter(
            axs[3, 1], df, "Rate at preferred direction (Hz)", "OSI", palette
        )
    else:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        _plot_panel(axs[0, 0], df, "Spont_Rate(Hz)", (0, 50), palette)
        _plot_panel(axs[0, 1], df, "Ave_Rate(Hz)", (0, 50), palette)
        _plot_panel(axs[1, 0], df, "DSI", (0, 1), palette)
        _plot_panel(axs[1, 1], df, "OSI", (0, 1), palette)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
