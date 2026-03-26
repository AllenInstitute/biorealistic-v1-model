from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aggregate_boxplots_odsi import discover_and_aggregate, plot_boxplots


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 3: OSI/DSI + rate aggregate boxplots")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--radius", type=float, default=200.0)
    parser.add_argument("--output", type=Path, default=Path("figures/paper/figure3/odsi_aggregate.png"))
    parser.add_argument("--metrics", nargs="*", default=["Rate at preferred direction (Hz)", "OSI", "DSI"])
    parser.add_argument("--e_only", action="store_true")
    args = parser.parse_args()

    include_variants = {
        "bio_trained": "Trained",
        "plain": "Untrained",
    }

    df = discover_and_aggregate(args.root.resolve(), core_radius=args.radius, include_variants=include_variants)
    if df.empty:
        print("No data found for plotting.")
        return

    plot_boxplots(df, metrics=args.metrics, output=args.output.resolve(), e_only=args.e_only)
    print(f"Saved plot to {args.output.resolve()}")


if __name__ == "__main__":
    main()
