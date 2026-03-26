"""Compare firing-rate changes between baseline and perturbation for targeted vs non-targeted populations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

FEATURES_PATH = Path("cell_categorization/core_nll_0_neuron_features.parquet")


def load_per_node_rates(sim_dir: Path) -> pd.DataFrame:
    metrics_file = sim_dir / "metrics" / "per_node_rates.parquet"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_file}")
    df = pd.read_parquet(metrics_file)
    return df.groupby("node_id", as_index=False)["rate_hz"].mean()


def summarize_groups(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    grouped = df.groupby(list(group_cols))
    return (
        grouped["delta_hz"]
        .agg(["mean", "median", "std", "count"])
        .rename(
            columns={
                "mean": "mean_delta_hz",
                "median": "median_delta_hz",
                "std": "std_delta_hz",
                "count": "n_cells",
            }
        )
        .reset_index()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize firing rate changes for targeted and non-targeted populations."
    )
    parser.add_argument(
        "baseline_dir",
        type=Path,
        help="Directory with baseline outputs (expects metrics/per_node_rates.parquet)",
    )
    parser.add_argument(
        "perturb_dir",
        type=Path,
        help="Directory with perturbation outputs (expects metrics/per_node_rates.parquet)",
    )
    parser.add_argument(
        "--group-col",
        default="cell_type",
        help="Column in the neuron feature table to aggregate over (default: cell_type)",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURES_PATH,
        help="Path to neuron feature parquet (default: cell_categorization/core_nll_0_neuron_features.parquet)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory to save summary CSVs (default: perturb_dir/metrics)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config JSON describing targeted node ids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline = load_per_node_rates(args.baseline_dir)
    perturb = load_per_node_rates(args.perturb_dir)

    features = pd.read_parquet(args.features)
    features = features.drop_duplicates(subset="node_id").set_index("node_id")

    merged = baseline.merge(
        perturb,
        on="node_id",
        how="inner",
        suffixes=("_baseline", "_perturb"),
    )
    merged = merged.merge(features, left_on="node_id", right_index=True, how="left")
    merged["delta_hz"] = merged["rate_hz_perturb"] - merged["rate_hz_baseline"]

    if args.config is not None:
        cfg = json.loads(args.config.read_text())
        try:
            node_list = cfg["inputs"]["InhSelectiveClamp"]["node_set"]["node_id"]
        except KeyError as exc:
            raise ValueError(
                f"Could not read targeted node ids from {args.config}"
            ) from exc
        target_ids = {int(nid) for nid in node_list}
        merged["is_target"] = merged["node_id"].isin(target_ids)
    else:
        merged["is_target"] = merged["activity_label"].eq("selective") & merged[
            "is_inhibitory"
        ].eq(True)

    group_col = args.group_col
    if group_col not in merged.columns:
        available = ", ".join(sorted(merged.columns))
        raise ValueError(
            f"Column '{group_col}' not found. Available columns: {available}"
        )

    outdir = args.outdir or (args.perturb_dir / "metrics")
    outdir.mkdir(parents=True, exist_ok=True)

    target_df = merged.loc[merged["is_target"]].copy()
    nontarget_df = merged.loc[~merged["is_target"]].copy()

    if target_df.empty:
        print("No targeted neurons found; nothing to summarize.")
    else:
        target_summary = summarize_groups(target_df, [group_col])
        overall_target = target_df["delta_hz"].agg(
            mean_delta_hz="mean",
            median_delta_hz="median",
            std_delta_hz="std",
            n_cells="count",
        )
        print("Targeted neurons (overall):")
        print(overall_target.to_string())
        print()
        print("Targeted neurons (grouped):")
        print(target_summary.head())
        target_summary.to_csv(
            outdir / "target_population_delta_by_group.csv", index=False
        )

    if nontarget_df.empty:
        print("No non-target neurons found; nothing to summarize.")
    else:
        nontarget_summary = summarize_groups(nontarget_df, [group_col])
        overall_non = nontarget_df["delta_hz"].agg(
            mean_delta_hz="mean",
            median_delta_hz="median",
            std_delta_hz="std",
            n_cells="count",
        )
        print("Non-target neurons (overall):")
        print(overall_non.to_string())
        print()
        print("Non-target neurons (grouped):")
        print(nontarget_summary.head())
        nontarget_summary.to_csv(
            outdir / "nontarget_population_delta_by_group.csv", index=False
        )


if __name__ == "__main__":
    main()
