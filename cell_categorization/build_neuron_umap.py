#!/usr/bin/env python3
"""Build a neuron-level feature table and compute a UMAP embedding."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from analysis_shared.neuron_features import (
    RATE_METRICS,
    collect_neuron_features,
    prepare_numeric_matrix,
)

DEFAULT_SELECTIVITY = Path("image_decoding/summary/sparsity_model_by_unit.csv")
DEFAULT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = DEFAULT_DIR / "core_nll_0_neuron_features.parquet"
ACTIVITY_FEATURES: tuple[str, ...] = (
    "image_selectivity",
    "orientation_selectivity",
    "dg_dsi",
    "natural_image_evoked_rate",
    "dg_spont_rate",
    "dg_evoked_rate",
    "dg_peak_rate",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute neuron UMAP embedding.")
    parser.add_argument(
        "--base-dir", default="core_nll_0", help="Network base directory"
    )
    parser.add_argument(
        "--network-type",
        default="bio_trained",
        help="Network suffix (bio_trained, naive, plain, ...)",
    )
    parser.add_argument(
        "--selectivity-path",
        type=Path,
        default=DEFAULT_SELECTIVITY,
        help="Path to sparsity/selectivity CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output Parquet path for neuron feature table",
    )
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.15)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=["network_id"],
        help="Numeric columns to exclude before scaling",
    )
    parser.add_argument(
        "--feature-set",
        choices=["activity", "all"],
        default="activity",
        help=(
            "Feature subset to use for UMAP: 'activity' uses DG/Image metrics only; "
            "'all' uses every numeric column."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.perf_counter()
    features = collect_neuron_features(
        args.base_dir, args.network_type, args.selectivity_path
    )
    t_collect = time.perf_counter()

    include_columns = ACTIVITY_FEATURES if args.feature_set == "activity" else None
    numeric = prepare_numeric_matrix(
        features, exclude=args.exclude_columns, include=include_columns
    )
    numeric = numeric.copy()
    for col in RATE_METRICS:
        if col in numeric.columns:
            numeric[col] = np.log1p(np.clip(numeric[col], a_min=0.0, a_max=None))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric.to_numpy(dtype=np.float64))
    t_scale = time.perf_counter()

    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )
    embedding = reducer.fit_transform(scaled)
    t_umap = time.perf_counter()

    features = features.copy()
    features["umap_x"] = embedding[:, 0]
    features["umap_y"] = embedding[:, 1]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    features.reset_index().to_parquet(args.out, index=False)

    meta = {
        "base_dir": args.base_dir,
        "network_type": args.network_type,
        "selectivity_path": str(args.selectivity_path),
        "out_path": str(args.out),
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": args.metric,
        "random_state": args.random_state,
        "feature_set": args.feature_set,
        "numeric_columns": list(numeric.columns),
        "timing_seconds": {
            "collect": t_collect - t0,
            "scale": t_scale - t_collect,
            "umap": t_umap - t_scale,
            "total": t_umap - t0,
        },
    }
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[info] Features saved to {args.out}")
    print(f"[info] Metadata saved to {meta_path}")
    timing = meta["timing_seconds"]
    print(
        f"[timing] collect={timing['collect']:.2f}s scale={timing['scale']:.2f}s "
        f"umap={timing['umap']:.2f}s total={timing['total']:.2f}s"
    )


if __name__ == "__main__":
    main()
