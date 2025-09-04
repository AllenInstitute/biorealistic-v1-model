import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_colors() -> Dict[str, str]:
    df = pd.read_csv("base_props/cell_type_naming_scheme.csv", sep=r"\s+")
    return dict(zip(df["cell_type"], df["hex"]))


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
        # L1
        "L1_Inh",
    ]


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


def load_np_unit_rates(np_cached_root: Path) -> pd.DataFrame:
    rows = []
    for sess_dir in sorted([p for p in np_cached_root.iterdir() if p.is_dir()]):
        try:
            rates = np.load(sess_dir / "rates_core.npy", mmap_mode="r")  # (R, I, C)
            meta = pd.read_csv(sess_dir / "meta_core.csv")
        except FileNotFoundError:
            continue
        # mean rate per unit across reps and images
        unit_mean = rates.mean(axis=(0, 1))  # (C,)
        # Guard against rare length mismatches
        n = int(min(len(unit_mean), len(meta)))
        if n == 0:
            continue
        df = pd.DataFrame({
            "dataset": "Neuropixels",
            "cell_type": meta.loc[: n - 1, "cell_type"].replace({"L1_Htr3a": "L1_Inh"}).to_numpy(),
            "firing_rate": unit_mean[:n],
        })
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["dataset","cell_type","firing_rate"]) 


def load_model_unit_rates(core_root: Path, networks: List[int], network_type: str) -> pd.DataFrame:
    frames = []
    for net in networks:
        base = core_root / f"core_nll_{net}" / f"cached_rates_{network_type}"
        if not base.exists():
            continue
        try:
            rates = np.load(base / "rates_core.npy", mmap_mode="r")  # (R, C, I)
            meta = pd.read_parquet(base / "meta_core.parquet")
        except Exception:
            continue
        unit_mean = rates.mean(axis=(0, 2))  # (C,)
        # Map network_type to display dataset label
        label_map = {
            "bio_trained": "Bio-trained",
            "naive": "Naive",
            "plain": "Plain",
            "adjusted": "Adjusted",
        }
        df = pd.DataFrame({
            "dataset": label_map.get(network_type, network_type),
            "cell_type": meta["cell_type"],
            "firing_rate": unit_mean,
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["dataset","cell_type","firing_rate"]) 


def add_l5_exc_combined(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["cell_type"].isin(["L5_ET", "L5_IT", "L5_NP"]) & df["firing_rate"].notna()
    add = df.loc[mask, ["dataset", "firing_rate"]].copy()
    add["cell_type"] = "L5_Exc"
    return pd.concat([df, add], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Box plots of firing rates (Neuropixels vs model variants)")
    parser.add_argument("--out", type=Path, default=Path("image_decoding/summary/firing_rate_boxplot.png"))
    parser.add_argument("--np_cached_root", type=Path, default=Path("image_decoding/neuropixels/cached_rates"))
    parser.add_argument("--core_root", type=Path, default=Path("."))
    parser.add_argument("--networks", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--exclude_naive", action="store_true", help="Exclude Naive dataset from the plot")
    args = parser.parse_args()

    # Load per-unit mean firing rates
    df_np = load_np_unit_rates(args.np_cached_root)
    df_bio = load_model_unit_rates(args.core_root, args.networks, "bio_trained")
    df_naive = load_model_unit_rates(args.core_root, args.networks, "naive")
    df_plain = load_model_unit_rates(args.core_root, args.networks, "plain")
    df_adjusted = load_model_unit_rates(args.core_root, args.networks, "adjusted")

    data = pd.concat([df_np, df_bio, df_naive, df_plain, df_adjusted], ignore_index=True)
    if args.exclude_naive:
        data = data[data["dataset"] != "Naive"]
    # Add combined L5_Exc alongside subtypes
    data = add_l5_exc_combined(data)
    data = data.dropna(subset=["cell_type", "firing_rate"]) 

    order = [ct for ct in cell_type_order() if ct in data["cell_type"].unique()]

    # Hue colors: keep Neuropixels gray, set model hues to specified angles
    import colorsys

    def hue_deg_to_hex(hue_deg: float, s: float = 0.75, v: float = 0.85) -> str:
        h = (hue_deg % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    palette_all = {
        "Neuropixels": "#7f7f7f",
        "Bio-trained": hue_deg_to_hex(135),  # green-cyan
        "Naive": hue_deg_to_hex(315),        # magenta-purple (reserved for future use)
        "Plain": hue_deg_to_hex(45),         # golden/yellow
        "Adjusted": hue_deg_to_hex(225),     # blue-purple
    }
    # Determine hue order dynamically based on presence and exclusion
    desired_order = ["Neuropixels", "Bio-trained", "Naive", "Plain", "Adjusted"]
    if args.exclude_naive:
        desired_order = [d for d in desired_order if d != "Naive"]
    present = [d for d in desired_order if d in data["dataset"].unique().tolist()]
    hue_order = present
    hue_palette = {k: palette_all[k] for k in hue_order}

    plt.figure(figsize=(8.0, 3.5))
    ax = plt.gca()
    sns.despine(ax=ax)

    # Background shading by subtype blocks
    colors_df = pd.read_csv("base_props/cell_type_naming_scheme.csv", sep=r"\s+")
    subtype_bg = get_subtype_colors_from_scheme(colors_df)
    add_background_shading(ax, order, subtype_bg)

    sns.boxplot(
        data=data,
        x="cell_type",
        y="firing_rate",
        order=order,
        hue="dataset",
        hue_order=hue_order,
        palette=hue_palette,
        showcaps=True,
        fliersize=1.5,
        width=0.7,
        boxprops={"edgecolor": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )

    ax.set_ylabel("Firing rate (Hz)")
    ax.set_xlabel("")
    # Dynamic ceiling: 95th percentile + headroom, cap to avoid extreme outliers dominating
    # try:
    #     y_top = float(np.nanpercentile(data["firing_rate"], 95)) + 50.0
    # except Exception:
    #     y_top = 10.0
    y_top = 60
    ax.set_ylim(bottom=0, top=min(100.0, max(5.0, y_top)))
    plt.xticks(rotation=90, ha="right")
    # Force horizontal legend: dedupe labels and set ncol to number of datasets
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    dedup = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if dedup:
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

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    try:
        plt.savefig(args.out.with_suffix(".svg"))
    except Exception:
        pass
    plt.close()


if __name__ == "__main__":
    main()


