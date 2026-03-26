#!/usr/bin/env python
"""
Create a summary file with complete target fractions (not split by high/low).
For stacked bar plots that should add to 1.0 (or close to it for core-to-core).
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summary of outgoing target fractions for stacked plots."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("core_nll_0"),
        help="Base directory containing figures/selectivity_outgoing (default: core_nll_0).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix aligning with outgoing_weight_granular_core_to_core_full*.csv",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Explicit path to granular core-to-core full CSV. Overrides --tag.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Explicit path for the summary CSV. Overrides default naming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    base_dir = args.base_dir

    input_path = (
        args.input
        or str(
            base_dir
            / "figures"
            / "selectivity_outgoing"
            / f"outgoing_weight_granular_core_to_core_full{suffix}.csv"
        )
    )
    output_path = (
        args.output
        or str(
            base_dir
            / "figures"
            / "selectivity_outgoing"
            / f"outgoing_weight_complete_targets{suffix}.csv"
        )
    )

    df = pd.read_csv(input_path)

    # Create summary with complete categories
    summary_rows = []

    for _, row in df.iterrows():
        inh_pv = row["inh_pv_fraction"]
        inh_sst = row["inh_sst_fraction"]
        inh_vip = row["inh_vip_fraction"]
        inh_htr3a = row["inh_htr3a_fraction"]

        exc_high_w = row["exc_high_weight"]
        exc_low_w = row["exc_low_weight"]
        total_w = row["total_weight"]

        inh_total_w = (
            row["inh_pv_weight"]
            + row["inh_sst_weight"]
            + row["inh_vip_weight"]
            + row["inh_htr3a_weight"]
        )

        exc_total_w = total_w - inh_total_w
        exc_total_frac = exc_total_w / total_w if total_w > 0 else 0

        exc_other_w = exc_total_w - exc_high_w - exc_low_w
        exc_other_frac = exc_other_w / total_w if total_w > 0 else 0

        summary_rows.append(
            {
                "group": row["source_group"],
                "n_connections": row["total_weight"],
                "exc_total": exc_total_frac,
                "exc_high": row["exc_high_fraction"],
                "exc_low": row["exc_low_fraction"],
                "exc_other": exc_other_frac,
                "inh_pv": inh_pv,
                "inh_sst": inh_sst,
                "inh_vip": inh_vip,
                "inh_htr3a": inh_htr3a,
                "total_check": exc_total_frac + inh_pv + inh_sst + inh_vip + inh_htr3a,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    print("\nTotal fractions (should be ~1.0 for complete categorization):")
    for _, row in summary_df.iterrows():
        if "_core" in row["group"]:
            print(f"{row['group']:20s}: {row['total_check']:.3f}")

    print("\nExcitatory breakdown:")
    for _, row in summary_df.iterrows():
        if "_core" in row["group"]:
            exc_high_pct = (
                row["exc_high"] / row["exc_total"] * 100 if row["exc_total"] > 0 else 0
            )
            exc_low_pct = (
                row["exc_low"] / row["exc_total"] * 100 if row["exc_total"] > 0 else 0
            )
            exc_other_pct = (
                row["exc_other"] / row["exc_total"] * 100 if row["exc_total"] > 0 else 0
            )
            print(
                f"{row['group']:20s}: Total={row['exc_total']:.3f} (high={exc_high_pct:.0f}%, low={exc_low_pct:.0f}%, other={exc_other_pct:.0f}%)"
            )


if __name__ == "__main__":
    main()
