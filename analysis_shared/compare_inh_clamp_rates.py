from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

FEATURES_PATH = Path("cell_categorization/core_nll_0_neuron_features.parquet")


def load_per_node_rates(sim_dir: Path) -> pd.DataFrame:
    metrics_file = sim_dir / "metrics" / "per_node_rates.parquet"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Expected metrics file missing: {metrics_file}")
    df = pd.read_parquet(metrics_file)
    return df.groupby("node_id", as_index=False)["rate_hz"].mean()


def load_target_node_ids(
    features_path: Path, config_path: Path | None = None
) -> pd.Index:
    if config_path is not None:
        cfg = json.loads(config_path.read_text())
        try:
            ids = cfg["inputs"]["InhSelectiveClamp"]["node_set"]["node_id"]
        except KeyError as exc:
            raise ValueError(
                f"Could not read targeted node ids from {config_path}"
            ) from exc
        return pd.Index([int(i) for i in ids])

    features = pd.read_parquet(features_path)
    targets = features.loc[
        (features["activity_label"] == "selective") & (features["is_inhibitory"]),
        "node_id",
    ].astype(int)
    return targets


def compare_rates(
    baseline_dir: Path, perturb_dir: Path, target_ids: pd.Index
) -> pd.DataFrame:
    baseline_rates = load_per_node_rates(baseline_dir)
    perturb_rates = load_per_node_rates(perturb_dir)

    merged = (
        baseline_rates.merge(
            perturb_rates,
            on="node_id",
            suffixes=("_baseline", "_perturb"),
            how="inner",
        )
        .loc[lambda df: df["node_id"].isin(target_ids)]
        .assign(delta_hz=lambda df: df["rate_hz_perturb"] - df["rate_hz_baseline"])
        .sort_values("delta_hz", ascending=False)
        .reset_index(drop=True)
    )
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs perturbation firing rates for targeted neurons."
    )
    parser.add_argument(
        "baseline_dir",
        type=Path,
        help="Directory containing baseline run outputs (expects metrics/per_node_rates.parquet)",
    )
    parser.add_argument(
        "perturb_dir",
        type=Path,
        help="Directory containing perturbation run outputs (expects metrics/per_node_rates.parquet)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config JSON describing targeted node ids for the perturbation.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path to save the per-neuron comparison table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_ids = load_target_node_ids(FEATURES_PATH, args.config)
    comparison = compare_rates(args.baseline_dir, args.perturb_dir, target_ids)

    summary = comparison["delta_hz"].agg(["mean", "median", "std", "count"])

    print("Targeted neuron firing-rate changes (perturb - baseline):")
    print(summary.to_string())
    print()
    print(comparison.head())

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.out_csv, index=False)
        print(f"Saved detailed table to {args.out_csv}")


if __name__ == "__main__":
    main()
