import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plotting_utils import pick_core
import network_utils as nu


def find_core_runs(root: Path) -> List[Path]:
    return sorted(p for p in root.glob("core_nll_*/metrics") if p.is_dir())


def load_model_metrics(base_metrics: Path, variant: str, core_radius: float) -> pd.DataFrame:
    base_dir = base_metrics.parent
    # Load network nodes and map pop_name → cell_type and ei
    try:
        nodes_df = nu.load_nodes(str(base_dir), core_radius=core_radius, expand=True)
        # ensure required columns
        if "Cell Type" in nodes_df.columns:
            nodes_df.rename(columns={"Cell Type": "cell_type"}, inplace=True)
        if "ei" not in nodes_df.columns and "ei" in nu.get_cell_type_table(tgt=["ei"]).columns:
            ctdf = nu.get_cell_type_table()
            nodes_df = nodes_df.merge(ctdf[["ei"]], left_on="pop_name", right_index=True, how="left")
    except Exception:
        # Fallback minimal mapping
        net = nu.load_nodes(str(base_dir), core_radius=core_radius, expand=False)
        types = net["types"]
        nodes_df = pd.DataFrame({
            "node_id": net["node_id"],
            "x": net.get("x"),
            "z": net.get("z"),
            "pop_name": types.loc[net["node_type_id"], "pop_name"].values,
        })
        ctdf = nu.get_cell_type_table()
        nodes_df = nodes_df.merge(ctdf[["cell_type", "ei"]], left_on="pop_name", right_index=True, how="left")

    # Pick core
    nodes_df = pick_core(nodes_df, radius=core_radius)

    # Load metrics
    metrics_csv = base_metrics / f"OSI_DSI_DF_{variant}.csv"
    if not metrics_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(metrics_csv, sep=" ")

    # Merge and standardize columns
    if "node_id" not in df.columns:
        return pd.DataFrame()
    merged = nodes_df.merge(df, on="node_id", how="inner")
    merged.rename(columns={"max_mean_rate(Hz)": "Rate at preferred direction (Hz)"}, inplace=True)

    # Clean and standardize cell_type labels
    merged["cell_type"] = merged["cell_type"].str.replace(" ", "_", regex=False)
    merged["cell_type"] = merged["cell_type"].str.replace("L1_Htr3a", "L1_Inh", regex=False)

    # Exclude low responders for OSI/DSI
    if "Rate at preferred direction (Hz)" in merged.columns:
        nonresponding = merged["Rate at preferred direction (Hz)"] < 0.5
        merged.loc[nonresponding, "OSI"] = np.nan
        merged.loc[nonresponding, "DSI"] = np.nan

    merged["dataset"] = variant
    return merged


def load_neuropixels_metrics(np_metrics_csv: Path, label: str) -> pd.DataFrame:
    if not np_metrics_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(np_metrics_csv, sep=" ")
    # Ensure consistent columns
    df.rename(columns={"max_mean_rate(Hz)": "Rate at preferred direction (Hz)"}, inplace=True)
    # cell_type may contain spaces like "L2/3 PV"; standardize to underscores
    if "cell_type" in df.columns:
        df["cell_type"] = df["cell_type"].astype(str).str.replace(" ", "_", regex=False)
        df["cell_type"] = df["cell_type"].str.replace("L1_Htr3a", "L1_Inh", regex=False)
    df["dataset"] = label
    return df


def discover_and_aggregate(root: Path, core_radius: float, include_variants: Dict[str, str]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    # Neuropixels
    np_csv = root / "neuropixels/metrics/OSI_DSI_neuropixels_v4.csv"
    np_df = load_neuropixels_metrics(np_csv, label="Neuropixels")
    if not np_df.empty:
        rows.append(np_df)

    # Model runs
    for metrics_dir in find_core_runs(root):
        for variant_key, plot_label in include_variants.items():
            df = load_model_metrics(metrics_dir, variant=variant_key, core_radius=core_radius)
            if not df.empty:
                df = df.copy()
                df["dataset"] = plot_label
                rows.append(df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cell_type_order() -> List[str]:
    # Match the canonical order used in image_decoding/plot_firing_rate_boxplot.py
    return [
        "L2/3_Exc",
        "L4_Exc",
        "L5_Exc",
        "L5_ET",
        "L5_IT",
        "L5_NP",
        "L6_Exc",
        "L1_Inh",
        "L2/3_PV",
        "L4_PV",
        "L5_PV",
        "L6_PV",
        "L2/3_SST",
        "L4_SST",
        "L5_SST",
        "L6_SST",
        "L2/3_VIP",
        "L4_VIP",
        "L5_VIP",
        "L6_VIP",
    ]


def _get_cell_subtype(cell_type: str) -> str:
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


def _get_subtype_bg_colors() -> Dict[str, tuple]:
    # Build light RGBA background colors per subtype from naming scheme
    colors_df = pd.read_csv("base_props/cell_type_naming_scheme.csv", sep=r"\s+")
    import matplotlib.colors as mcolors

    def mean_rgba(hex_list: List[str]) -> tuple:
        if len(hex_list) == 0:
            return (0.9, 0.9, 0.9, 0.12)
        rgbs = [mcolors.to_rgba(h, alpha=0.12) for h in hex_list]
        arr = np.array([c[:3] for c in rgbs])
        mean_rgb = arr.mean(axis=0)
        return (mean_rgb[0], mean_rgb[1], mean_rgb[2], 0.12)

    subtype_colors: Dict[str, tuple] = {}
    for subtype, cls in [("Exc", "Exc"), ("PV", "PV"), ("SST", "SST"), ("VIP", "VIP"), ("Inh", "Inh")]:
        hexes = colors_df[colors_df["class"] == cls]["hex"].tolist()
        subtype_colors[subtype] = mean_rgba(hexes)
    return subtype_colors


def plot_boxplots(df: pd.DataFrame, metrics: List[str], output: Path, e_only: bool = False) -> None:
    if df.empty:
        print("No data to plot.")
        return

    # Optionally restrict to excitatory types
    if e_only and "ei" in df.columns:
        df = df[df["ei"] == "e"]

    # Add combined L5_Exc from L5 subtypes for all metrics
    def add_l5_exc_combined_generic(data: pd.DataFrame, metric_columns: List[str]) -> pd.DataFrame:
        mask_sub = data["cell_type"].isin(["L5_ET", "L5_IT", "L5_NP"]) 
        if not mask_sub.any():
            return data
        add = data.loc[mask_sub, [c for c in ["dataset", "cell_type", "ei"] if c in data.columns] + metric_columns].copy()
        add["cell_type"] = "L5_Exc"
        return pd.concat([data, add], ignore_index=True)

    df = add_l5_exc_combined_generic(df, metrics)

    order_all = cell_type_order()
    # Restrict order to those present in df to keep alignment
    present = [ct for ct in order_all if ct in df.get("cell_type", pd.Series(dtype=str)).unique().tolist()]
    if "cell_type" in df.columns:
        df = df[df["cell_type"].isin(present)]

    # Palette per dataset, not per cell type
    # Match colors used in image_decoding/plot_firing_rate_boxplot.py
    import colorsys

    def hue_deg_to_hex(hue_deg: float, s: float = 0.75, v: float = 0.85) -> str:
        h = (hue_deg % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    dataset_palette = {
        "Neuropixels": "#7f7f7f",
        "Bio-trained": hue_deg_to_hex(135),  # green-cyan
        "trained": hue_deg_to_hex(135),
        "Trained": hue_deg_to_hex(135),
        "Untrained": hue_deg_to_hex(45),     # golden/yellow
        "Syn. weight distr. constrained": hue_deg_to_hex(135),
        "Syn. weight distr. unconstrained": hue_deg_to_hex(315),
    }

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(8.0, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sns.despine(ax=ax)
        # Background shading by subtype blocks
        subtype_bg = _get_subtype_bg_colors()
        current = None
        start = 0
        for i, ct in enumerate(present + [None]):
            subtype = _get_cell_subtype(ct) if ct is not None else None
            if subtype != current:
                if current is not None:
                    end = i - 0.5
                    color = subtype_bg.get(current, (0.92, 0.92, 0.92, 0.12))
                    ax.axvspan(start - 0.5, end, facecolor=color, zorder=0)
                current = subtype
                start = i

        sns.boxplot(
            data=df,
            x="cell_type",
            y=metric,
            hue="dataset",
            order=present,
            palette=dataset_palette,
            ax=ax,
            width=0.7,
            fliersize=1.0,
            boxprops={"edgecolor": "black", "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
        )
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_xlabel("")

        # Set sensible y-limits per metric
        if metric in ("OSI", "DSI"):
            ax.set_ylim(0.0, 1.0)
        # Legend handled once at the top using first axis
        ax.legend_.remove() if ax.get_legend() is not None else None

    output.parent.mkdir(parents=True, exist_ok=True)
    # Build a single horizontal legend at the top based on the first axis' handles
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate
    seen = set()
    dedup = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if dedup:
        handles_d, labels_d = zip(*dedup)
        axes[0].legend(
            handles_d,
            labels_d,
            frameon=False,
            ncol=len(labels_d),
            loc="upper left",
            bbox_to_anchor=(0, 1.18),
            handlelength=1.4,
            columnspacing=1.0,
            borderaxespad=0.0,
            title=None,
        )
    fig.savefig(output, dpi=300)
    try:
        fig.savefig(output.with_suffix(".pdf"))
    except Exception:
        pass
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Aggregate OSI/DSI and rate boxplots across core_nll_* runs and Neuropixels.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root directory")
    parser.add_argument("--radius", type=float, default=200.0, help="Core radius in microns")
    parser.add_argument("--output", type=Path, default=Path("figures/boxplots/odsi_aggregate.png"), help="Output figure path")
    parser.add_argument("--metrics", nargs="*", default=["Rate at preferred direction (Hz)", "OSI", "DSI"], help="Metrics to plot")
    parser.add_argument("--e_only", action="store_true", help="Plot only excitatory cell types")
    args = parser.parse_args()

    # Map from file variant names to desired legend labels
    include_variants = {
        "bio_trained": "Bio-trained",
        "plain": "Untrained",
    }

    df = discover_and_aggregate(args.root.resolve(), core_radius=args.radius, include_variants=include_variants)
    if df.empty:
        print("No data found for plotting.")
        return

    plot_boxplots(df, metrics=args.metrics, output=args.output.resolve(), e_only=args.e_only)
    print(f"Saved plot to {args.output.resolve()}")


if __name__ == "__main__":
    main()


