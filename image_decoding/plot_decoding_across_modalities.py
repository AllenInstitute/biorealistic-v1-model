import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from image_decoding.plot_utils import (
    cell_type_order,
    dataset_palette,
    dataset_order,
    load_cell_type_colors,
    get_subtype_colors_from_scheme,
    add_background_shading,
    set_horizontal_legend,
)


NETWORK_LABELS: Dict[str, str] = {
    "bio_trained": "Bio-trained",
    "naive": "Naive",
    "plain": "Plain",
    "adjusted": "Adjusted",
}


def load_single(summary_dir: Path, network_type: str) -> pd.DataFrame:
    csv_path = summary_dir / f"decoding_summary_{network_type}_by_type.csv"
    df = pd.read_csv(csv_path)
    df = df[df["sample_size"] == 30].copy()
    df["dataset"] = NETWORK_LABELS.get(network_type, network_type)
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare decoding across modalities for sample size 30")
    parser.add_argument("--summary_dir", type=Path, default=Path("image_decoding/summary50"))
    parser.add_argument(
        "--network_types",
        type=str,
        nargs="*",
        default=["bio_trained", "naive", "plain", "adjusted"],
    )
    parser.add_argument("--exclude_naive", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("image_decoding/summary50/accuracy_barplot_across_modalities_n30.png"))
    args = parser.parse_args()

    frames: List[pd.DataFrame] = []
    for nt in args.network_types:
        csv_path = args.summary_dir / f"decoding_summary_{nt}_by_type.csv"
        if not csv_path.exists():
            continue
        df_nt = load_single(args.summary_dir, nt)
        if df_nt.empty:
            continue
        frames.append(df_nt)
    if not frames:
        print("No decoding CSVs found.")
        return
    data = pd.concat(frames, ignore_index=True)

    # Filter out Naive if requested
    if args.exclude_naive:
        data = data[data["dataset"] != "Naive"]

    order = [ct for ct in cell_type_order() if ct in data["cell_type"].unique()]

    # Hue setup
    pal = dataset_palette()
    present = data["dataset"].unique().tolist()
    hue_order = dataset_order(include_naive=not args.exclude_naive, present=present)
    hue_palette = {k: pal[k] for k in hue_order}

    plt.figure(figsize=(10.0, 3.6))
    ax = plt.gca()

    # Background shading
    colors_df = load_cell_type_colors()
    subtype_bg = get_subtype_colors_from_scheme(colors_df)
    add_background_shading(ax, order, subtype_bg)

    sns.barplot(
        data=data,
        x="cell_type",
        y="accuracy",
        order=order,
        hue="dataset",
        hue_order=hue_order,
        palette=hue_palette,
        errorbar="se",
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Decoding accuracy (n=30)")
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=90, ha="right")
    sns.despine(ax=ax)
    set_horizontal_legend(ax)

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


