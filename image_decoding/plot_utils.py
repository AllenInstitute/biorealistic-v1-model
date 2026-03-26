from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import colorsys
import numpy as np

from analysis_shared.celltype_labels import abbrev_cell_types
from analysis_shared.style import apply_pub_style, trim_spines


def hue_deg_to_hex(hue_deg: float, s: float = 0.75, v: float = 0.85) -> str:
    h = (hue_deg % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def dataset_palette() -> Dict[str, str]:
    pal = {
        "Neuropixels": "#7f7f7f",
        "Bio-trained": hue_deg_to_hex(135),
        "Untrained": hue_deg_to_hex(45),
        "Adjusted": hue_deg_to_hex(225),
        "Naive": hue_deg_to_hex(315),
    }
    pal["trained"] = pal["Bio-trained"]
    pal["Trained"] = pal["Bio-trained"]
    pal["Syn. weight distr. constrained"] = pal["Bio-trained"]
    pal["Syn. weight distr. unconstrained"] = pal["Naive"]
    return pal


def dataset_order(include_naive: bool, present: List[str]) -> List[str]:
    if "Trained" in present:
        trained_label = "Trained"
    elif "trained" in present:
        trained_label = "trained"
    else:
        trained_label = "Bio-trained"
    desired = [trained_label, "Untrained", "Adjusted", "Naive"]
    if not include_naive:
        desired = [d for d in desired if d != "Naive"]
    return [d for d in desired if d in present]


def cell_type_order() -> List[str]:
    return [
        # Excitatory
        "L2/3_Exc",
        "L4_Exc",
        "L5_Exc",
        "L5_ET",
        "L5_IT",
        "L5_NP",
        "L6_Exc",
        # L1 inhibitory first
        "L1_Inh",
        # PV
        "L2/3_PV",
        "L4_PV",
        "L5_PV",
        "L6_PV",
        # SST
        "L2/3_SST",
        "L4_SST",
        "L5_SST",
        "L6_SST",
        # VIP
        "L2/3_VIP",
        "L4_VIP",
        "L5_VIP",
        "L6_VIP",
    ]


def draw_combined_similarity(
    mat_sim: pd.DataFrame,
    ks_per_type: pd.DataFrame,
    x_order: List[str],
    palette: Dict[str, str],
    out_path: Path,
    figsize: Tuple[float, float] = (3.6, 6.0),
    height_ratios: List[float] = [0.6, 2.4],
    width_ratios: List[float] = [24, 1.5],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=height_ratios, width_ratios=width_ratios)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    # Top: boxplot of per-type similarity by dataset
    sns.boxplot(
        data=ks_per_type,
        x="dataset",
        y="similarity",
        order=x_order,
        hue="dataset",
        hue_order=x_order,
        palette={k: palette[k] for k in x_order},
        dodge=False,
        width=0.6,
        showcaps=True,
        fliersize=1.5,
        boxprops={"edgecolor": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        ax=ax_top,
    )
    ax_top.set_ylabel("Similarity score")
    ax_top.set_xlabel("")
    ax_top.set_ylim(0.0, 1.0)
    ax_top.set_xticklabels([])
    ax_top.tick_params(axis="x", which="both", length=0)
    if ax_top.get_legend() is not None:
        ax_top.legend_.remove()
    sns.despine(ax=ax_top)

    # Bottom: heatmap
    sns.heatmap(
        mat_sim,
        annot=True,
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        cbar=True,
        cbar_ax=cax,
        cbar_kws={"label": "Similarity score"},
        ax=ax_bot,
    )
    ax_bot.set_xlabel("Dataset vs Neuropixels")
    ax_bot.set_ylabel("Cell type")
    plt.setp(ax_bot.get_xticklabels(), rotation=35, ha="right")

    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    try:
        plt.savefig(out_path.with_suffix(".svg"))
    except Exception:
        pass
    plt.close()


def draw_metric_boxplot_with_similarity_heatmap(
    df: pd.DataFrame,
    metric: str,
    mat_sim: pd.DataFrame,
    *,
    datasets_boxplot: List[str],
    datasets_heatmap: List[str],
    palette: Dict[str, str],
    cell_types: List[str],
    out_path: Path,
    figsize: Tuple[float, float] = (7.5, 4.0),
    height_ratios: List[float] = [2.8, 0.4],
    width_ratios: List[float] = [24, 1.2],
    hspace: float = 0.005,
    wspace: float = 0.02,
    tight_layout_pad: float = 0.08,
    bottom: float = 0.25,
    left: float = 0.15,
    heatmap_xtick_fontsize: float = 7.0,
    heat_annot_fontsize: Optional[float] = None,
    heatmap_xtick_pad: float = 1.0,
    heatmap_dataset_label_map: Optional[Dict[str, str]] = None,
    mask_similarity_cell_types: List[str] = ["L5_ET", "L5_IT", "L5_NP"],
    boxplot_yscale: str | None = None,
    boxplot_ylim: Tuple[float, float] | None = None,
    boxplot_ylabel_fontsize: Optional[float] = None,
    legend_bbox_to_anchor: Tuple[float, float] = (0, 1.12),
    style_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    apply_pub_style()
    # Default to Arial if not specified in style_overrides
    if style_overrides is None:
        style_overrides = {}
    if "font.family" not in style_overrides:
        style_overrides["font.family"] = "Arial"
    
    plt.rcParams.update(style_overrides)

    df = df.dropna(subset=["dataset", "cell_type", metric]).copy()

    df = df[df["dataset"].isin(datasets_boxplot)].copy()
    df = df[df["cell_type"].isin(cell_types)].copy()

    heat = mat_sim.copy()
    if not heat.empty:
        heat = heat.reindex(index=cell_types)
        heat = heat.reindex(columns=datasets_heatmap)

    heat_t = heat.T

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=hspace,
        wspace=wspace,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    subtype_colors = get_subtype_colors_from_scheme(load_cell_type_colors())
    add_background_shading(ax_top, cell_types, subtype_colors)

    sns.boxplot(
        data=df,
        x="cell_type",
        y=metric,
        hue="dataset",
        order=cell_types,
        hue_order=datasets_boxplot,
        palette={k: palette[k] for k in datasets_boxplot if k in palette},
        ax=ax_top,
        width=0.7,
        fliersize=1.0,
        boxprops={"edgecolor": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )
    ax_top.set_xlabel("")
    ax_top.set_ylabel(metric)
    if boxplot_ylabel_fontsize is not None:
        ax_top.yaxis.label.set_size(boxplot_ylabel_fontsize)
    ax_top.tick_params(axis="x", labelbottom=False)
    trim_spines(ax_top)
    set_horizontal_legend(ax_top, bbox_to_anchor=legend_bbox_to_anchor)

    if boxplot_yscale is not None:
        ax_top.set_yscale(boxplot_yscale)

    # Consolidate Y-limit and offset logic
    if boxplot_ylim is not None:
        ymin, ymax = float(boxplot_ylim[0]), float(boxplot_ylim[1])
    else:
        ymin, ymax = ax_top.get_ylim()
        # Default for bounded metrics
        if metric in ("OSI", "DSI", "Image selectivity", "Stimulus selectivity"):
            ymin, ymax = 0.0, 1.05

    # Apply standard zero-offsets to prevent overlap with heatmap labels
    if metric in ("Firing rate (Hz)", "Rate at preferred direction (Hz)", "Spontaneous rate (Hz)"):
        ymin = min(ymin, -2.0)
    elif metric in ("OSI", "DSI", "Image selectivity", "Stimulus selectivity"):
        ymin = min(ymin, -0.02)
        
    ax_top.set_ylim(ymin, ymax)

    heat_t_plot = heat_t.copy()
    annot = heat_t.copy()
    annot = annot.applymap(lambda x: "" if pd.isna(x) else f"{float(x):.2f}")
    for ct in mask_similarity_cell_types:
        if ct in heat_t_plot.columns:
            heat_t_plot[ct] = np.nan
        if ct in annot.columns:
            annot[ct] = ""

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    
    annot_kws = {}
    if heat_annot_fontsize is not None:
        annot_kws["fontsize"] = heat_annot_fontsize

    sns.heatmap(
        heat_t_plot,
        annot=annot,
        fmt="",
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
        cbar=True,
        cbar_ax=cax,
        cbar_kws={"label": "Similarity score"},
        ax=ax_bot,
        annot_kws=annot_kws,
    )
    if heatmap_dataset_label_map is not None:
        yticklabels = [
            heatmap_dataset_label_map.get(str(lbl), str(lbl)) for lbl in heat_t_plot.index.tolist()
        ]
    else:
        yticklabels = [str(lbl) for lbl in heat_t_plot.index.tolist()]
    
    yticks = np.arange(len(yticklabels), dtype=float) + 0.5
    ax_bot.set_yticks(yticks)
    ax_bot.set_yticklabels(yticklabels, rotation=0)
    
    ax_bot.set_xlabel("")
    ax_bot.set_ylabel("")
    xlabels = [str(c) for c in heat_t_plot.columns.tolist()]
    if xlabels:
        xticks = np.arange(len(xlabels), dtype=float) + 0.5
        ax_bot.set_xticks(xticks)
        ax_bot.set_xticklabels(xlabels, rotation=90)
    ax_bot.tick_params(axis="x", pad=heatmap_xtick_pad)
    plt.setp(ax_bot.get_xticklabels(), fontsize=heatmap_xtick_fontsize)
    trim_spines(ax_bot)

    fig.tight_layout(pad=tight_layout_pad)
    fig.subplots_adjust(bottom=bottom, left=left)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def draw_similarity_summary_boxplot_multi_metric(
    df: pd.DataFrame,
    *,
    metric_order: List[str],
    dataset_order: List[str],
    palette: Dict[str, str],
    out_path: Path,
    figsize: Tuple[float, float] = (4.8, 1.9),
    style_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    apply_pub_style()
    if style_overrides is not None:
        plt.rcParams.update(style_overrides)

    data = df.dropna(subset=["metric", "dataset", "similarity"]).copy()
    data = data[data["metric"].isin(metric_order) & data["dataset"].isin(dataset_order)].copy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.boxplot(
        data=data,
        x="metric",
        y="similarity",
        hue="dataset",
        order=metric_order,
        hue_order=dataset_order,
        palette={k: palette[k] for k in dataset_order if k in palette},
        width=0.65,
        fliersize=1.2,
        boxprops={"edgecolor": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Similarity score")
    ax.set_ylim(0.0, 1.0)
    set_horizontal_legend(ax)
    trim_spines(ax)
    fig.tight_layout(pad=0.15)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def draw_similarity_summary_boxplot(
    ks_per_type: pd.DataFrame,
    *,
    datasets: List[str],
    palette: Dict[str, str],
    out_path: Path,
    figsize: Tuple[float, float] = (2.9, 1.8),
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    apply_pub_style()

    data = ks_per_type.dropna(subset=["dataset", "similarity"]).copy()
    data = data[data["dataset"].isin(datasets)].copy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.boxplot(
        data=data,
        x="dataset",
        y="similarity",
        order=datasets,
        hue="dataset",
        hue_order=datasets,
        palette={k: palette[k] for k in datasets if k in palette},
        dodge=False,
        width=0.6,
        fliersize=1.5,
        boxprops={"edgecolor": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Similarity score")
    ax.set_ylim(0.0, 1.0)
    if ax.get_legend() is not None:
        ax.legend_.remove()
    trim_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


# Background shading utilities shared by boxplots/decoding ---------------------

def load_cell_type_colors() -> pd.DataFrame:
    return pd.read_csv("base_props/cell_type_naming_scheme.csv", sep=r"\s+")


def get_cell_subtype(cell_type: str) -> str:
    if cell_type == "L1_Inh":
        return "Inh"
    if "Exc" in cell_type or cell_type in {"L5_ET", "L5_IT", "L5_NP"}:
        return "Exc"
    if "PV" in cell_type:
        return "PV"
    if "SST" in cell_type:
        return "SST"
    if "VIP" in cell_type:
        return "VIP"
    return "Other"


def get_subtype_colors_from_scheme(colors_df: pd.DataFrame) -> Dict[str, tuple]:
    def mean_rgba(hex_list: List[str]) -> tuple:
        if len(hex_list) == 0:
            return (0.9, 0.9, 0.9, 0.12)
        import matplotlib.colors as mcolors

        rgbs = [mcolors.to_rgba(h, alpha=0.12) for h in hex_list]
        arr = np.array([c[:3] for c in rgbs])
        mean_rgb = arr.mean(axis=0)
        return (mean_rgb[0], mean_rgb[1], mean_rgb[2], 0.12)

    subtype_colors: Dict[str, tuple] = {}
    for subtype, cls in [("Exc", "Exc"), ("PV", "PV"), ("SST", "SST"), ("VIP", "VIP"), ("Inh", "Inh")]:
        hexes = colors_df[colors_df["class"] == cls]["hex"].tolist()
        subtype_colors[subtype] = mean_rgba(hexes)
    return subtype_colors


def add_background_shading(ax, cell_types: List[str], subtype_colors: Dict[str, tuple]):
    current = None
    start = 0
    for i, ct in enumerate(cell_types + [None]):
        subtype = get_cell_subtype(ct) if ct is not None else None
        if subtype != current:
            if current is not None:
                end = i - 0.5
                color = subtype_colors.get(current, (0.92, 0.92, 0.92, 0.12))
                ax.axvspan(start - 0.5, end, facecolor=color, zorder=0)
            current = subtype
            start = i


def set_horizontal_legend(ax, bbox_to_anchor=(0, 1.17)):
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    dedup = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if not dedup:
        return
    handles_d, labels_d = zip(*dedup)
    ax.legend(
        handles_d,
        labels_d,
        frameon=False,
        ncol=len(labels_d),
        loc="upper left",
        bbox_to_anchor=bbox_to_anchor,
        handlelength=1.4,
        columnspacing=1.0,
        borderaxespad=0.0,
        title=None,
    )


