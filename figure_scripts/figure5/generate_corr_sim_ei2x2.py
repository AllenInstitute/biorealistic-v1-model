#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import List

# Ensure repository root is on PYTHONPATH when run from any working directory
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_shared.corr import compute_corr_ei2x2_cache, plot_corr_ei2x2_from_cache


def discover_bases(default_max: int = 10) -> List[str]:
    bases = []
    for i in range(default_max):
        d = f"core_nll_{i}"
        if os.path.isdir(d):
            bases.append(d)
    return bases


def main():
    ap = argparse.ArgumentParser(
        description="Generate 2x2 E/I Corr matrix with caching (simulation only)."
    )
    ap.add_argument(
        "--bases",
        nargs="*",
        default=None,
        help="Base directories (default: detected core_nll_* present)",
    )
    ap.add_argument(
        "--network-type",
        default="bio_trained",
        choices=["bio_trained", "naive"],
        help="Network type",
    )
    ap.add_argument(
        "--out",
        default="figures/paper/corr_sim_ei2x2_cached.png",
        help="Output PNG path",
    )
    ap.add_argument(
        "--bin-size", type=float, default=0.05, help="Bin size for response correlation"
    )
    ap.add_argument(
        "--x-min", type=float, default=-0.2, help="Min x for response correlation range"
    )
    ap.add_argument(
        "--x-max", type=float, default=0.5, help="Max x for response correlation range"
    )
    ap.add_argument(
        "--no-simplify-inh",
        action="store_true",
        help="Do not simplify inhibitory to PV/SST/VIP",
    )
    ap.add_argument(
        "--no-aggregate-l5",
        action="store_true",
        help="Do not aggregate L5 IT/ET/NP to L5_Exc",
    )
    ap.add_argument(
        "--max-per-pair",
        type=int,
        default=None,
        help="Optional max connections per pair for subsampling",
    )
    ap.add_argument(
        "--pair-limits-csv", default=None, help="Optional CSV with per-pair limits"
    )
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed")
    ap.add_argument(
        "--cache-dir",
        default="figures/paper/cache",
        help="Directory to store cache pickles",
    )
    ap.add_argument(
        "--cache-name", default="corr_sim_ei2x2.pkl", help="Cache file name"
    )
    ap.add_argument(
        "--replot-only",
        action="store_true",
        help="Load cache and re-render without recomputing",
    )
    ap.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cache freshness and recompute",
    )
    args = ap.parse_args()

    bases = args.bases if args.bases else discover_bases()
    if not bases:
        raise SystemExit(
            "No base directories found. Provide --bases or create core_nll_*."
        )

    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, args.cache_name)

    cache = None
    if (not args.force_recompute) and os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = None

    if cache is None and not args.replot_only:
        cache = compute_corr_ei2x2_cache(
            bases,
            args.network_type,
            simplify_inh=(not args.no_simplify_inh),
            aggregate_l5_types=(not args.no_aggregate_l5),
            bin_size=args.bin_size,
            x_min=args.x_min,
            x_max=args.x_max,
            max_per_pair=args.max_per_pair,
            pair_limits_csv=args.pair_limits_csv,
            sample_seed=args.seed,
        )
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    if cache is None:
        raise SystemExit("No cache available to plot. Run without --replot-only first.")

    plot_corr_ei2x2_from_cache(cache, args.out)
    print(f"saved: {args.out}\ncache: {cache_path}")


if __name__ == "__main__":
    main()
