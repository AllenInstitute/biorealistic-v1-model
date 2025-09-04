import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from image_decoding.plot_utils import (
    cell_type_order,
    load_cell_type_colors,
    get_subtype_colors_from_scheme,
    add_background_shading,
)


NETWORK_LABELS: Dict[str, str] = {
    "bio_trained": "Bio-trained",
    "naive": "Naive",
    "plain": "Plain",
    "adjusted": "Adjusted",
}


def plot_one(df: pd.DataFrame, network_type: str, outdir: Path) -> None:
    order_all = cell_type_order()
    available_types = df["cell_type"].unique().tolist()
    order = [ct for ct in order_all if ct in available_types]

    plt.figure(figsize=(10.0, 3.6))
    ax = plt.gca()

    # Background shading by subtype blocks
    colors_df = load_cell_type_colors()
    subtype_bg = get_subtype_colors_from_scheme(colors_df)
    add_background_shading(ax, order, subtype_bg)

    sns.barplot(
        data=df,
        x="cell_type",
        y="accuracy",
        hue="sample_size",
        errorbar="se",
        order=order,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Decoding accuracy")
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=90, ha="right")
    sns.despine(ax=ax)
    # Horizontal legend
    legend_cols = len(sorted(df["sample_size"].unique()))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.20), ncol=legend_cols, title="Sample size")
    ax.set_title(NETWORK_LABELS.get(network_type, network_type))

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"accuracy_barplot_{network_type}_by_type.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    try:
        plt.savefig(out.with_suffix(".svg"))
    except Exception:
        pass
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot decoding accuracy barplots from cached CSVs")
    parser.add_argument("--summary_dir", type=Path, default=Path("image_decoding/summary50"))
    parser.add_argument("--network_types", type=str, nargs="*", default=["bio_trained", "naive", "plain", "adjusted"])
    args = parser.parse_args()

    for nt in args.network_types:
        csv_path = args.summary_dir / f"decoding_summary_{nt}_by_type.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        plot_one(df, nt, args.summary_dir)


if __name__ == "__main__":
    main()


