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

from analysis_shared.pd_effect_size import (
    compute_effect_size_matrix,
    plot_effect_size_heatmaps,
)


def discover_bases(default_max: int = 10) -> List[str]:
    bases = []
    for i in range(default_max):
        d = f"core_nll_{i}"
        if os.path.isdir(d):
            bases.append(d)
    return bases


def cache_is_fresh(cache_path: str, script_path: str) -> bool:
    if not os.path.isfile(cache_path):
        return False
    try:
        return os.path.getmtime(cache_path) >= os.path.getmtime(script_path)
    except Exception:
        return True


def run_once(
    bases: List[str],
    network_type: str,
    *,
    simplify_inh: bool,
    aggregate_l5: bool,
    cache_dir: str,
    out_dir: str,
    vmax_percentile: float,
    force_recompute: bool,
    replot_only: bool,
    suffix: str,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    cache_path = os.path.join(
        cache_dir, f"pd_effect_{network_type}{'_' + suffix if suffix else ''}.pkl"
    )
    script_path = os.path.abspath(__file__)

    effect_data = None

    if replot_only and not os.path.isfile(cache_path):
        print(
            f"[warn] replot-only requested but cache missing: {cache_path}",
            file=sys.stderr,
        )

    if (not force_recompute) and cache_is_fresh(cache_path, script_path):
        try:
            with open(cache_path, "rb") as f:
                effect_data = pickle.load(f)
            print(f"[cache] Loaded fresh cache: {cache_path}")
        except Exception as e:
            print(
                f"[cache] Failed to load cache ({e}); will recompute.", file=sys.stderr
            )
            effect_data = None

    if effect_data is None and not replot_only:
        print(
            f"[compute] Computing effect sizes for {network_type} from bases: {bases}\n"
            f"         simplify_inh={simplify_inh} aggregate_l5={aggregate_l5}"
        )
        effect_data = compute_effect_size_matrix(
            bases,
            network_type,
            simplify_inh=simplify_inh,
            aggregate_l5_types=aggregate_l5,
            cache_path=cache_path,
        )
        print(f"[cache] Saved cache: {cache_path}")

    if effect_data is None:
        print(f"[error] No effect data to plot for {network_type}.", file=sys.stderr)
        return

    types = effect_data.get("types", [])
    print(f"[diag] {network_type} types={len(types)} example={types[:8]}")

    sub = f"{network_type}{'_' + suffix if suffix else ''}"
    target_out = os.path.join(out_dir, sub)
    plot_effect_size_heatmaps(effect_data, target_out, vmax_percentile=vmax_percentile)
    print(f"[plot] Saved heatmaps under: {target_out}")


def main():
    p = argparse.ArgumentParser(
        description="Generate PD effect size heatmaps (a/|c|, b/|c|) with caching."
    )
    p.add_argument(
        "--bases",
        nargs="*",
        default=None,
        help="Base directories (default: detected core_nll_* present)",
    )
    p.add_argument(
        "--network-type", choices=["bio_trained", "naive", "both"], default="both"
    )
    p.add_argument(
        "--cache-dir", default="figures/effect_size", help="Directory for cache pickles"
    )
    p.add_argument(
        "--out-dir", default="figures/effect_size", help="Directory to write figures"
    )
    p.add_argument(
        "--vmax-percentile",
        type=float,
        default=95.0,
        help="Percentile for robust color scaling (abs)",
    )
    p.add_argument(
        "--no-simplify-inh",
        action="store_true",
        help="Do not simplify inhibitory to PV/SST/VIP",
    )
    p.add_argument(
        "--aggregate-l5", action="store_true", help="Aggregate L5 subtypes into L5_Exc"
    )
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cache freshness and recompute",
    )
    p.add_argument(
        "--replot-only", action="store_true", help="Skip compute; load cache and replot"
    )
    p.add_argument(
        "--suffix",
        default="",
        help="Suffix to append to cache/figure names (e.g., inhagg or inhsplit)",
    )
    args = p.parse_args()

    bases = args.bases if args.bases else discover_bases()
    if not bases:
        print(
            "[error] No base directories found. Provide --bases or create core_nll_*.",
            file=sys.stderr,
        )
        sys.exit(1)

    simplify_inh = not args.no_simplify_inh
    todo = (
        [args.network_type]
        if args.network_type in ("bio_trained", "naive")
        else ["bio_trained", "naive"]
    )
    for nt in todo:
        run_once(
            bases,
            nt,
            simplify_inh=simplify_inh,
            aggregate_l5=args.aggregate_l5,
            cache_dir=args.cache_dir,
            out_dir=args.out_dir,
            vmax_percentile=args.vmax_percentile,
            force_recompute=args.force_recompute,
            replot_only=args.replot_only,
            suffix=args.suffix,
        )


if __name__ == "__main__":
    main()
