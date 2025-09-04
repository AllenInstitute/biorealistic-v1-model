from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import colorsys
import numpy as np


def hue_deg_to_hex(hue_deg: float, s: float = 0.75, v: float = 0.85) -> str:
    h = (hue_deg % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def dataset_palette() -> Dict[str, str]:
    return {
        "Neuropixels": "#7f7f7f",
        "Bio-trained": hue_deg_to_hex(135),
        "Plain": hue_deg_to_hex(45),
        "Adjusted": hue_deg_to_hex(225),
        "Naive": hue_deg_to_hex(315),
    }


def dataset_order(include_naive: bool, present: List[str]) -> List[str]:
    desired = ["Bio-trained", "Plain", "Adjusted", "Naive"]
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
        # L1 at the end
        "L1_Inh",
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


def set_horizontal_legend(ax):
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
        bbox_to_anchor=(0, 1.17),
        handlelength=1.4,
        columnspacing=1.0,
        borderaxespad=0.0,
        title=None,
    )


