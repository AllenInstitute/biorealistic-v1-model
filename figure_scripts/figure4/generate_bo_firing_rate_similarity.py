from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Figure 4: BO natural-images firing-rate similarity (combined boxplot + heatmap)"
    )
    parser.add_argument("--np_cached_root", type=Path, default=Path("image_decoding/neuropixels/cached_rates"))
    parser.add_argument("--core_root", type=Path, default=Path("."))
    parser.add_argument("--networks", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--exclude_naive", action="store_true", default=True)
    parser.add_argument("--exclude_adjusted", action="store_true", default=True)
    parser.add_argument("--log_eps", type=float, default=1e-3)
    parser.add_argument("--fig_width_scale", type=float, default=1.0)
    parser.add_argument(
        "--boxplot_ylim",
        type=float,
        nargs=2,
        default=(0.0, 60.0),
        metavar=("YMIN", "YMAX"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/paper/figure4/bo_firing_rate_similarity_combined.png"),
    )
    args = parser.parse_args()

    import image_decoding.compute_firing_rate_similarity as mod

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            old_argv[0],
            "--np_cached_root",
            str(args.np_cached_root),
            "--core_root",
            str(args.core_root),
            "--boxplot_ylim",
            str(args.boxplot_ylim[0]),
            str(args.boxplot_ylim[1]),
            "--out_combined",
            str(args.out),
            "--log_eps",
            str(args.log_eps),
            "--fig_width_scale",
            str(args.fig_width_scale),
        ]
        if args.exclude_naive:
            sys.argv.append("--exclude_naive")
        if args.exclude_adjusted:
            sys.argv.append("--exclude_adjusted")
        if args.networks:
            sys.argv.append("--networks")
            sys.argv.extend([str(n) for n in args.networks])

        args.out.parent.mkdir(parents=True, exist_ok=True)
        mod.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
