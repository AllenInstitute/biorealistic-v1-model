#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os

from analysis_shared.corr import corr_exc_matrix_plot
from analysis_shared.pd import pd_exc_matrix_plot
from analysis_shared.pd_mc import compute_pd_mc_pvalues, plot_pd_mc_histograms


def main():
    p = argparse.ArgumentParser(description="Unified analysis CLI for correlation and preferred-direction plots.")
    p.add_argument("base_dirs", nargs="+", help="One or more network base directories (e.g., core_nll_0)")
    p.add_argument("network_type", help="Network type: bio_trained or naive")
    p.add_argument("out_png", help="Output PNG path")
    p.add_argument("--mode", choices=["corr", "pd", "pd_mc"], default="corr", help="Which analysis to run")
    p.add_argument("--aggregate-l5", action="store_true", help="Aggregate L5 subtypes into L5_Exc")
    # Reduced options
    p.add_argument("--max-per-pair", type=int, default=None, help="Uniform cap for connections per source-target pair")
    p.add_argument("--pair-limits-csv", type=str, default=None, help="CSV with columns: source,target,connections")
    p.add_argument("--sample-seed", type=int, default=0, help="Sampling seed for reduced mode")
    # PD Monte Carlo
    p.add_argument("--mc-resamples", type=int, default=0, help="If mode=pd_mc, number of resamples (e.g., 100)")
    p.add_argument("--mc-connections", type=int, default=None, help="If mode=pd_mc, uniform connections per draw (overridden by per-pair CSV if given)")
    args = p.parse_args()

    if args.mode == "corr":
        corr_exc_matrix_plot(
            args.base_dirs,
            args.network_type,
            args.out_png,
            aggregate_l5_types=args.aggregate_l5,
            max_per_pair=args.max_per_pair,
            pair_limits_csv=args.pair_limits_csv,
            sample_seed=args.sample_seed,
        )
    elif args.mode == "pd":
        pd_exc_matrix_plot(
            args.base_dirs,
            args.network_type,
            args.out_png,
            aggregate_l5_types=args.aggregate_l5,
            max_per_pair=args.max_per_pair,
            pair_limits_csv=args.pair_limits_csv,
            sample_seed=args.sample_seed,
        )
    else:
        mc = compute_pd_mc_pvalues(
            args.base_dirs,
            args.network_type,
            aggregate_l5_types=args.aggregate_l5,
            resamples=args.mc_resamples,
            connections_per_draw=args.mc_connections,
            seed=args.sample_seed,
            pair_limits_csv=args.pair_limits_csv,
        )
        plot_pd_mc_histograms(mc, aggregate_l5_types=args.aggregate_l5, out_png=args.out_png)

    print(f"Saved: {args.out_png}")


if __name__ == "__main__":
    main()
