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
        description="Figure 4: BO natural-images selectivity/sparsity similarity (combined boxplot + heatmap)"
    )
    parser.add_argument(
        "--np_by_unit",
        type=Path,
        default=Path("image_decoding/neuropixels/summary/sparsity_neuropixels_by_unit.csv"),
    )
    parser.add_argument(
        "--model_by_unit",
        type=Path,
        default=Path("image_decoding/summary/sparsity_model_by_unit.csv"),
    )
    parser.add_argument("--exclude_naive", action="store_true", default=True)
    parser.add_argument("--exclude_adjusted", action="store_true", default=True)
    parser.add_argument("--fig_width_scale", type=float, default=1.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/paper/figure4/bo_selectivity_similarity_combined.png"),
    )
    args = parser.parse_args()

    import image_decoding.compute_selectivity_similarity as mod

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            old_argv[0],
            "--np_by_unit",
            str(args.np_by_unit),
            "--model_by_unit",
            str(args.model_by_unit),
            "--out_combined",
            str(args.out),
            "--fig_width_scale",
            str(args.fig_width_scale),
        ]
        if args.exclude_naive:
            sys.argv.append("--exclude_naive")
        if args.exclude_adjusted:
            sys.argv.append("--exclude_adjusted")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        mod.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
