from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aggregate_boxplots_odsi import discover_and_aggregate
from aggregate_similarity_odsi import add_l5_exc_combined, compute_similarity
from image_decoding.plot_utils import dataset_order, dataset_palette, draw_similarity_summary_boxplot_multi_metric


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 3: combined summary similarity-score boxplots (metric on x-axis)")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--radius", type=float, default=200.0)
    parser.add_argument("--outdir", type=Path, default=Path("figures/paper/figure3"))
    parser.add_argument("--log_eps", type=float, default=1e-3)
    parser.add_argument("--e_only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figsize-w", type=float, default=5.2)
    parser.add_argument("--figsize-h", type=float, default=2.1)
    args = parser.parse_args()

    df = discover_and_aggregate(
        args.root.resolve(),
        core_radius=args.radius,
        include_variants={"bio_trained": "Trained", "plain": "Untrained"},
    )
    if df.empty:
        print("No aggregated data found.")
        return

    if args.e_only and "ei" in df.columns:
        df = df[df["ei"] == "e"]

    metrics = [
        ("FR", "Rate at preferred direction (Hz)", True),
        ("OSI", "OSI", False),
        ("DSI", "DSI", False),
    ]

    pal = dataset_palette()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.output if args.output is not None else (args.outdir / "similarity_summary_metrics.png")

    rows = []
    present_datasets = set()
    for metric_label, metric_col, do_log in metrics:
        df_metric = add_l5_exc_combined(df, metric_col)
        _mat_sim, ks_per_type, _x_order, _pal_unused = compute_similarity(
            df_metric,
            metric=metric_col,
            log_transform=do_log,
            log_eps=args.log_eps,
            include_naive=False,
        )
        if ks_per_type.empty:
            continue
        ks_per_type = ks_per_type.copy()
        ks_per_type["metric"] = metric_label
        rows.append(ks_per_type)
        present_datasets.update(ks_per_type["dataset"].unique().tolist())

    if not rows:
        print("No similarity data to plot.")
        return

    combined = pd.concat(rows, ignore_index=True)
    ds_order = dataset_order(include_naive=False, present=sorted(present_datasets))
    draw_similarity_summary_boxplot_multi_metric(
        combined,
        metric_order=[m[0] for m in metrics],
        dataset_order=ds_order,
        palette=pal,
        out_path=out_path,
        figsize=(float(args.figsize_w), float(args.figsize_h)),
    )
    print(f"Saved {out_path.name}")


if __name__ == "__main__":
    main()
