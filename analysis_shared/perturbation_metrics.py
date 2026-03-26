"""Utilities for summarizing perturbation simulation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate firing-rate metrics from perturbation runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more BMTK output directories containing spikes.csv.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("cell_categorization/core_nll_0_neuron_features.parquet"),
        help="Parquet file with neuron annotations (default: cell_categorization/core_nll_0_neuron_features.parquet).",
    )
    parser.add_argument(
        "--duration-ms",
        type=float,
        default=None,
        help="Simulation duration in milliseconds. If omitted, inferred from config_*.json in each run.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory to save aggregated metric tables (default: parent of the first run dir).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config JSON describing the targeted node ids.",
    )
    return parser.parse_args()


def infer_duration(run_dir: Path) -> float | None:
    for cfg_path in sorted(run_dir.glob("config*.json")):
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            continue
        duration = cfg.get("run", {}).get("duration")
        if duration is not None:
            return float(duration)
    return None


def load_spike_rates(
    run_dir: Path, duration_ms: float, node_index: pd.Index
) -> pd.DataFrame:
    spikes_path = run_dir / "spikes.csv"
    if not spikes_path.exists():
        raise FileNotFoundError(f"Missing spikes.csv in {run_dir}")

    spikes = pd.read_csv(spikes_path, sep=" ")
    if spikes.empty:
        counts = pd.Series(dtype="int64")
    else:
        counts = (
            spikes.loc[spikes["population"] == "v1", "node_ids"]
            .value_counts()
            .rename_axis("node_id")
        )

    duration_s = duration_ms / 1000.0
    rates = counts / duration_s
    rates = rates.reindex(node_index, fill_value=0.0)
    rates.name = "rate_hz"
    df = rates.reset_index()
    df["run_id"] = run_dir.name
    return df


def summarize_by_cell_type(
    merged: pd.DataFrame, group_cols: Iterable[str]
) -> pd.DataFrame:
    grouped = merged.groupby(list(group_cols))
    return (
        grouped["rate_hz"]
        .agg(["mean", "median", "std", "count"])
        .rename(
            columns={
                "mean": "mean_rate_hz",
                "median": "median_rate_hz",
                "std": "std_rate_hz",
                "count": "n_cells",
            }
        )
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    features = pd.read_parquet(args.features)
    features = features.drop_duplicates(subset="node_id").set_index("node_id")

    target_ids: set[int] | None = None
    if args.config is not None:
        cfg = json.loads(args.config.read_text())
        try:
            node_list = cfg["inputs"]["InhSelectiveClamp"]["node_set"]["node_id"]
        except KeyError as exc:
            raise ValueError(
                f"Could not read targeted node ids from {args.config}"
            ) from exc
        target_ids = {int(nid) for nid in node_list}

    outdir = args.outdir or args.run_dirs[0].parent
    outdir.mkdir(parents=True, exist_ok=True)

    per_run_tables = []
    per_node_tables = []
    for run_dir in args.run_dirs:
        run_duration = args.duration_ms or infer_duration(run_dir)
        if run_duration is None:
            raise ValueError(
                f"Could not infer duration for {run_dir}. Provide --duration-ms explicitly."
            )
        rates = load_spike_rates(run_dir, run_duration, features.index)
        per_node_tables.append(rates)
        merged = rates.merge(
            features,
            left_on="node_id",
            right_index=True,
            how="left",
            validate="many_to_one",
        )
        if target_ids is not None:
            merged["is_target"] = merged["node_id"].isin(target_ids)
        else:
            merged["is_target"] = merged["activity_label"].eq("selective") & merged[
                "is_inhibitory"
            ].eq(True)
        merged["run_id"] = run_dir.name
        per_run_tables.append(merged)

    per_node = pd.concat(per_node_tables, ignore_index=True)
    per_node.to_parquet(outdir / "per_node_rates.parquet", index=False)

    merged_all = pd.concat(per_run_tables, ignore_index=True)
    per_run_cell_type = summarize_by_cell_type(merged_all, ["run_id", "cell_type"])
    per_run_cell_type.to_csv(outdir / "cell_type_metrics_by_run.csv", index=False)

    summary_cell_type = summarize_by_cell_type(merged_all, ["cell_type"])
    summary_cell_type.to_csv(outdir / "cell_type_metrics_summary.csv", index=False)

    target_sel = merged_all.loc[merged_all["is_target"]]
    if not target_sel.empty:
        target_summary = summarize_by_cell_type(target_sel, ["run_id"])
        target_summary.to_csv(outdir / "target_metrics_by_run.csv", index=False)
        target_summary.to_csv(outdir / "selective_metrics_by_run.csv", index=False)
        target_summary.to_csv(outdir / "inh_selective_metrics_by_run.csv", index=False)

        overall_stats = target_sel["rate_hz"].agg(
            mean_rate_hz="mean",
            median_rate_hz="median",
            std_rate_hz="std",
        )
        summary_df = overall_stats.to_frame().T
        summary_df["n_cells"] = len(target_sel)
        summary_df["n_runs"] = target_sel["run_id"].nunique()
        summary_df.to_csv(outdir / "target_metrics_summary.csv", index=False)
        summary_df.to_csv(outdir / "selective_metrics_summary.csv", index=False)
        summary_df.to_csv(outdir / "inh_selective_metrics_summary.csv", index=False)
    else:
        (outdir / "target_metrics_by_run.csv").write_text(
            "run_id,mean_rate_hz,median_rate_hz,std_rate_hz,n_cells\n"
        )
        (outdir / "selective_metrics_by_run.csv").write_text(
            "run_id,mean_rate_hz,median_rate_hz,std_rate_hz,n_cells\n"
        )
        (outdir / "target_metrics_summary.csv").write_text(
            "mean_rate_hz,median_rate_hz,std_rate_hz,n_cells\n"
        )
        (outdir / "selective_metrics_summary.csv").write_text(
            "mean_rate_hz,median_rate_hz,std_rate_hz,n_cells\n"
        )
        (outdir / "inh_selective_metrics_by_run.csv").write_text(
            "run_id,mean_rate_hz,median_rate_hz,std_rate_hz,n_cells\n"
        )
        (outdir / "inh_selective_metrics_summary.csv").write_text(
            "mean_rate_hz,median_rate_hz,std_rate_hz,n_cells,n_runs\n"
        )


if __name__ == "__main__":
    main()
