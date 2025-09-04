import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_colors() -> Dict[str, str]:
    df = pd.read_csv("base_props/cell_type_naming_scheme.csv", sep=r"\s+")
    return dict(zip(df["cell_type"], df["hex"]))


def _value_col(df: pd.DataFrame) -> str:
    return "image_selectivity" if "image_selectivity" in df.columns else "lifetime_sparsity"


def aggregate_selectivity(selectivity_units_path: Path) -> pd.DataFrame:
    df = pd.read_csv(selectivity_units_path)
    val_col = _value_col(df)
    grp = df.groupby(["network_type", "cell_type"])  # type: ignore[list-item]
    agg = (
        grp[val_col]
        .agg([("selectivity_mean", "mean"), ("selectivity_std", "std"), ("n_units", "count")])
        .reset_index()
    )
    agg["selectivity_sem"] = agg["selectivity_std"] / np.sqrt(agg["n_units"].clip(lower=1))
    return agg


def aggregate_decoding(decoding_full_path: Path, sample_size: int = 30, n_reps: int = 50) -> pd.DataFrame:
    df = pd.read_csv(decoding_full_path)
    df = df[(df["sample_size"] == sample_size) & (df["n_reps"] == n_reps)].copy()
    grp = df.groupby(["network_type", "cell_type"])  # type: ignore[list-item]
    agg = (
        grp["accuracy"].agg([("accuracy_mean", "mean"), ("accuracy_std", "std"), ("n_entries", "count")]).reset_index()
    )
    agg["accuracy_sem"] = agg["accuracy_std"] / np.sqrt(agg["n_entries"].clip(lower=1))
    return agg


def make_scatter(df_merged: pd.DataFrame, network_type: str, out_path: Path, colors: Dict[str, str]):
    sub = df_merged[df_merged["network_type"] == network_type].copy()
    if sub.empty:
        return
    plt.figure(figsize=(4.0, 3.5))
    ax = plt.gca()
    sns.despine(ax=ax)

    for _, row in sub.iterrows():
        ct = row["cell_type"]
        c = colors.get(ct, "#777777")
        ax.errorbar(
            row["selectivity_mean"],
            row["accuracy_mean"],
            yerr=row["accuracy_sem"],
            fmt="o",
            ms=6,
            mec="black",
            mfc=c,
            mew=0.6,
            ecolor="black",
            elinewidth=0.6,
            capsize=2,
            capthick=0.6,
            alpha=0.95,
        )

    # Add text labels with consistent policy: L23-Exc, L4-PV, L5-ET, L1-Inh
    def _format_label(name: str) -> str:
        label = name.replace("L2/3", "L23")
        if label in ("L5_ET", "L5_IT", "L5_NP"):
            return label.replace("_", "-")
        if label == "L1_Inh":
            return "L1-Inh"
        if "_" in label:
            layer, cls = label.split("_", 1)
            return f"{layer}-{cls}"
        return label

    for _, row in sub.iterrows():
        x = row["selectivity_mean"]
        y = row["accuracy_mean"]
        ax.annotate(
            _format_label(row["cell_type"]),
            (x, y),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=8,
            color="black",
            ha="left",
            va="bottom",
            zorder=5,
        )

    ax.set_xlabel("Image selectivity (trial-averaged)")
    ax.set_ylabel("Decoding accuracy (30 cells, 50 reps)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    try:
        plt.savefig(out_path.with_suffix(".svg"))
    except Exception:
        pass
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Scatter: image selectivity vs decoding accuracy per cell type")
    parser.add_argument("--selectivity_units", type=Path, default=Path("image_decoding/summary/sparsity_model_by_unit.csv"))
    parser.add_argument("--decoding_full", type=Path, default=Path("image_decoding/summary_full/decoding_summary_full.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("image_decoding/summary"))
    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--n_reps", type=int, default=50)
    args = parser.parse_args()

    sel = aggregate_selectivity(args.selectivity_units)
    dec = aggregate_decoding(args.decoding_full, sample_size=args.sample_size, n_reps=args.n_reps)
    merged = pd.merge(sel, dec, on=["network_type", "cell_type"], how="inner")

    colors = load_colors()
    make_scatter(merged, "bio_trained", args.outdir / "selectivity_vs_decoding_bio_trained.png", colors)
    make_scatter(merged, "naive", args.outdir / "selectivity_vs_decoding_naive.png", colors)


if __name__ == "__main__":
    main()


