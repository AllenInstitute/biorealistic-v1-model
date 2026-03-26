#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Ensure repository root is on PYTHONPATH when run from any working directory
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_shared.pd_effect_size import compute_effect_size_matrix, plot_effect_size_heatmaps


def _detect_default_bases(max_n: int = 10) -> list[str]:
    return [f"core_nll_{i}" for i in range(max_n) if os.path.isdir(f"core_nll_{i}")]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate PD fit heatmaps (a/c, b/c, -log10 p(a), -log10 p(b)) "
            "for bio_trained and naive, using Fig5-style settings."
        )
    )
    ap.add_argument("--bases", nargs="*", default=None)
    ap.add_argument(
        "--out-dir",
        default="figures/paper/figure6",
        help="Output directory for PNG/PDF heatmaps.",
    )
    ap.add_argument(
        "--cache-dir",
        default="figures/paper/figure6/cache_pd_fit_heatmaps",
        help="Cache directory for computed effect-size matrices.",
    )
    ap.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cache and recompute.",
    )
    ap.add_argument(
        "--vmax-percentile",
        type=float,
        default=95.0,
        help="Percentile for color scaling (abs for a/c and b/c; value for -log10 p).",
    )
    args = ap.parse_args()

    bases = args.bases if args.bases else _detect_default_bases()
    if not bases:
        raise SystemExit("No base directories found (expected core_nll_*).")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    for network_type in ["bio_trained", "naive"]:
        cache_path = os.path.join(
            args.cache_dir,
            f"pd_effect_size_{network_type}_inhWithinLayer_splitE5_omitNP.pkl",
        )

        effect = None
        if (not args.force_recompute) and os.path.isfile(cache_path):
            try:
                import pickle

                with open(cache_path, "rb") as f:
                    effect = pickle.load(f)
            except Exception:
                effect = None

        if effect is None:
            effect = compute_effect_size_matrix(
                bases,
                network_type,
                simplify_inh=True,
                inh_respective_layer=True,
                aggregate_l5_types=False,
                omit_np=True,
            )
            try:
                import pickle

                with open(cache_path, "wb") as f:
                    pickle.dump(effect, f)
            except Exception:
                pass

        plot_effect_size_heatmaps(
            effect,
            args.out_dir,
            vmax_percentile=float(args.vmax_percentile),
            label_style="full",
            also_pdf=True,
            file_prefix=f"{network_type}_pd_",
        )

    print(f"saved PD fit heatmaps under {args.out_dir}")


if __name__ == "__main__":
    main()
