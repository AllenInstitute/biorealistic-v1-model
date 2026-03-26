#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Ensure repository root is on PYTHONPATH when run from any working directory
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_shared.em_compare import load_em_corr_pickle, em_corr_matrix_plot
from analysis_shared.corr_mc import plot_corr_violin_with_em
from analysis_shared.style import apply_pub_style
from analysis_shared.grouping import apply_inh_simplification, aggregate_l5


def compute_corr_effect_size(
    bases,
    network_type: str,
    cache_path: str,
    *,
    simplify_inh: bool = False,
    aggregate_l5_types: bool = False,
) -> dict:
    # Load sim corr per pair and compute signed slope/intercept (no abs)
    from aggregate_correlation_plot import process_network_data
    import pandas as pd

    dfs = []
    for bd in bases:
        df = process_network_data((bd, network_type))
        dfs.append(
            df[
                ["source_type", "target_type", "Response Correlation", "syn_weight"]
            ].copy()
        )
    df = pd.concat(dfs, ignore_index=True)
    if simplify_inh:
        df = apply_inh_simplification(df)
    if aggregate_l5_types:
        df = aggregate_l5(df)

    # Cell-type-first ordering: Exc first, then inhibitory families.
    present_types = sorted(
        list(
            set(df["source_type"].astype(str)).union(set(df["target_type"].astype(str)))
        )
    )

    # Excitatory order: choose based on aggregation flag
    exc_order_agg = ["L2/3_Exc", "L4_Exc", "L5_Exc", "L6_Exc"]
    exc_order_split = ["L2/3_Exc", "L4_Exc", "L5_IT", "L5_ET", "L5_NP", "L6_Exc"]
    exc_order = exc_order_agg if aggregate_l5_types else exc_order_split
    exc_types = [t for t in exc_order if t in present_types]

    # Inhibitory order
    if simplify_inh:
        inh_types = [t for t in ["PV", "SST", "VIP", "L1_Inh"] if t in present_types]
    else:
        layers = ["L2/3", "L4", "L5", "L6"]
        inh_types = []
        for fam in ["PV", "SST", "VIP"]:
            for lyr in layers:
                lab = f"{lyr}_{fam}"
                if lab in present_types:
                    inh_types.append(lab)
        if "L1_Inh" in present_types:
            inh_types.append("L1_Inh")

    types = exc_types + [t for t in inh_types if t not in exc_types]
    type_to_idx = {t: i for i, t in enumerate(types)}
    n = len(types)
    A = np.full((n, n), np.nan)
    B = np.full((n, n), np.nan)

    from scipy.stats import linregress

    for s in types:
        for t in types:
            sub = df[(df["source_type"] == s) & (df["target_type"] == t)][
                ["Response Correlation", "syn_weight"]
            ].dropna()
            if len(sub) < 10:
                continue
            slope, intercept, _r, _p, _se = linregress(
                sub["Response Correlation"], sub["syn_weight"]
            )
            denom = intercept if intercept != 0 and np.isfinite(intercept) else np.nan
            i = type_to_idx[s]
            j = type_to_idx[t]
            if np.isfinite(denom):
                A[i, j] = slope / denom
                B[i, j] = 0.0  # placeholder for second panel consistency

    # types already ordered as desired (Exc first, then PV/SST/VIP/L1_Inh)
    out = {"types": types, "slope_over_abs_intercept": A, "placeholder": B}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out


def plot_corr_effect_size(
    effect_data: dict,
    out_dir: str,
    vmax_percentile: float = 95.0,
    vmax_abs: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    from analysis_shared.style import trim_spines

    apply_pub_style()
    types = effect_data["types"]
    A = effect_data["slope_over_abs_intercept"]
    vals = A[np.isfinite(A)]
    if vmax_abs is not None:
        vmax = float(vmax_abs)
    else:
        vmax = np.percentile(np.abs(vals), vmax_percentile) if vals.size else 1.0
    vmin, vmax = -vmax, vmax
    n = len(types)

    # Match PD effect-size sizing (compact matrices)
    fig_w = min(12, max(1, n * 0.23))
    fig_h = fig_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    def _simplify(lbl: str) -> str:
        mapping = {
            "L2/3_Exc": "E23",
            "L4_Exc": "E4",
            "L5_Exc": "E5",
            "L6_Exc": "E6",
            "L5_IT": "E5IT",
            "L5_ET": "E5ET",
            "L5_NP": "E5NP",
            "L1_Inh": "L1",
        }
        return mapping.get(lbl, lbl.replace("_Exc", "").replace("_", ""))

    im = ax.imshow(
        A, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest"
    )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([_simplify(x) for x in types], rotation=90, fontsize=6)
    ax.set_yticklabels([_simplify(x) for x in types], fontsize=6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("Effect size slope/intercept", fontsize=9)
    trim_spines(ax)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "corr_effect_slope_over_intercept.png"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Generate corr EM + MC violin + effect size plots"
    )
    ap.add_argument("--bases", nargs="*", default=None)
    ap.add_argument("--pair-limits-csv", default=None)
    ap.add_argument("--resamples", type=int, default=100)
    ap.add_argument(
        "--plot-cap",
        type=int,
        default=None,
        help="Only plot pairs with CSV limit <= this value",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="figures/corr_compare")
    ap.add_argument("--out-dir", default="figures/corr_compare")
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument(
        "--focus-types",
        nargs="*",
        default=None,
        help="Override focus types (e.g., L2/3_Exc L4_Exc L5_IT L5_ET L5_NP L6_Exc)",
    )
    ap.add_argument(
        "--suffix",
        default="",
        help="Suffix to append to output filenames (e.g., agg or l5split)",
    )
    ap.add_argument(
        "--vmax-abs",
        type=float,
        default=None,
        help="Optional fixed abs max for color scale (e.g., 2.0)",
    )
    ap.add_argument(
        "--only-effect-size",
        action="store_true",
        help="Only regenerate effect-size plots from cache",
    )
    args = ap.parse_args()

    bases = (
        args.bases
        if args.bases
        else [f"core_nll_{i}" for i in range(10) if os.path.isdir(f"core_nll_{i}")]
    )
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.only_effect_size:
        # 1) EM corr matrices (both aggregated and split L5)
        em_agg = load_em_corr_pickle(
            os.path.join("analysis_shared", "corr_vs_weight_minnie_250828.pkl"),
            split_l5=False,
        )
        em_split = load_em_corr_pickle(
            os.path.join("analysis_shared", "corr_vs_weight_minnie_250828.pkl"),
            split_l5=True,
        )
        print(f"[em] agg pairs: {len(em_agg)}, split-L5 pairs: {len(em_split)}")
        em_corr_matrix_plot(
            em_agg, os.path.join(args.out_dir, "em_corr_matrix_agg.png"), split_l5=False
        )
        em_corr_matrix_plot(
            em_split,
            os.path.join(args.out_dir, "em_corr_matrix_l5split.png"),
            split_l5=True,
        )

        # 2) MC corr violin with EM overlay (bio_trained and naive)
        focus_agg = ["L2/3_Exc", "L4_Exc", "L5_Exc", "L6_Exc"]
        focus_split = ["L2/3_Exc", "L4_Exc", "L5_IT", "L5_ET", "L5_NP", "L6_Exc"]

        def run_base_mc(
            bases,
            nt: str,
            focus_types: list[str],
            aggregate_l5_flag: bool,
            runs: int,
            seed: int,
            pair_limits_csv: str | None,
            plot_cap: int | None,
        ):
            """Run the same MC as corr_compare; returns mapping (pair->pvalue array)."""
            from aggregate_correlation_plot import (
                process_network_data,
                monte_carlo_pvalue_matrix,
            )
            from analysis_shared.sampling import read_pair_limits_csv
            import pandas as pd

            dfs = []
            for bd in bases:
                df = process_network_data((bd, nt))
                dfs.append(
                    df[
                        [
                            "source_type",
                            "target_type",
                            "Response Correlation",
                            "syn_weight",
                        ]
                    ].copy()
                )
            edge_df = pd.concat(dfs, ignore_index=True)
            limits = read_pair_limits_csv(pair_limits_csv) if pair_limits_csv else None
            pmap, _exc_types = monte_carlo_pvalue_matrix(
                edge_df,
                aggregate_l5=aggregate_l5_flag,
                runs=runs,
                max_per_pair=None,
                pair_limits=limits,
                base_seed=seed,
            )
            out = {}
            for s in focus_types:
                for t in focus_types:
                    arr = pmap.get((s, t), [])
                    if not arr:
                        continue
                    if plot_cap is not None:
                        if limits is None:
                            continue
                        if (s, t) not in limits:
                            continue
                        if limits[(s, t)] < plot_cap:
                            continue
                    out[(s, t)] = np.array(arr, dtype=float)
            return out

        for nt in ("bio_trained", "naive"):
            print(f"[mc-base] {nt} aggregated focus: {focus_agg}")
            mc_agg = run_base_mc(
                bases,
                nt,
                focus_agg,
                True,
                args.resamples,
                args.seed,
                args.pair_limits_csv,
                args.plot_cap,
            )
            from scipy.stats import linregress

            em_pvals_agg: dict[tuple[str, str], float] = {}
            for (s, t), df in em_agg.items():
                x = df["x"].to_numpy()
                y = df["y"].to_numpy()
                if len(x) >= 10:
                    _slope, _intercept, _r, p, _se = linregress(x, y)
                    em_pvals_agg[(s, t)] = float(p)
            plot_corr_violin_with_em(
                mc_agg,
                em_pvals_agg,
                os.path.join(args.out_dir, f"corr_violin_{nt}_agg.png"),
                focus_types=focus_agg,
            )

            print(f"[mc-base] {nt} split-L5 focus: {focus_split}")
            mc_split = run_base_mc(
                bases,
                nt,
                focus_split,
                False,
                args.resamples,
                args.seed,
                args.pair_limits_csv,
                args.plot_cap,
            )
            em_pvals_split: dict[tuple[str, str], float] = {}
            for (s, t), df in em_split.items():
                x = df["x"].to_numpy()
                y = df["y"].to_numpy()
                if len(x) >= 10:
                    _slope, _intercept, _r, p, _se = linregress(x, y)
                    em_pvals_split[(s, t)] = float(p)
            plot_corr_violin_with_em(
                mc_split,
                em_pvals_split,
                os.path.join(args.out_dir, f"corr_violin_{nt}_l5split.png"),
                focus_types=focus_split,
            )

    # 3) Corr effect size heatmap for simulation: generate both inhibitory-split and inhibitory-aggregated
    for nt in ("bio_trained", "naive"):
        cache_split = os.path.join(args.cache_dir, f"corr_effect_{nt}_inhsplit.pkl")
        if (not args.force_recompute) and os.path.isfile(cache_split):
            try:
                with open(cache_split, "rb") as f:
                    eff_split = pickle.load(f)
            except Exception:
                eff_split = compute_corr_effect_size(
                    bases, nt, cache_split, simplify_inh=False
                )
        else:
            eff_split = compute_corr_effect_size(
                bases, nt, cache_split, simplify_inh=False
            )
        plot_corr_effect_size(
            eff_split,
            os.path.join(args.out_dir, f"{nt}_inhsplit"),
            vmax_abs=args.vmax_abs,
        )

        cache_agg = os.path.join(args.cache_dir, f"corr_effect_{nt}_inhagg.pkl")
        if (not args.force_recompute) and os.path.isfile(cache_agg):
            try:
                with open(cache_agg, "rb") as f:
                    eff_agg = pickle.load(f)
            except Exception:
                eff_agg = compute_corr_effect_size(
                    bases, nt, cache_agg, simplify_inh=True, aggregate_l5_types=False
                )
        else:
            eff_agg = compute_corr_effect_size(
                bases, nt, cache_agg, simplify_inh=True, aggregate_l5_types=False
            )
        vmax_for_agg = args.vmax_abs if args.vmax_abs is not None else 2.0
        plot_corr_effect_size(
            eff_agg, os.path.join(args.out_dir, f"{nt}_inhagg"), vmax_abs=vmax_for_agg
        )

        cache_agg_l5 = os.path.join(
            args.cache_dir, f"corr_effect_{nt}_inhagg_l5agg.pkl"
        )
        if (not args.force_recompute) and os.path.isfile(cache_agg_l5):
            try:
                with open(cache_agg_l5, "rb") as f:
                    eff_agg_l5 = pickle.load(f)
            except Exception:
                eff_agg_l5 = compute_corr_effect_size(
                    bases, nt, cache_agg_l5, simplify_inh=True, aggregate_l5_types=True
                )
        else:
            eff_agg_l5 = compute_corr_effect_size(
                bases, nt, cache_agg_l5, simplify_inh=True, aggregate_l5_types=True
            )
        vmax_for_agg_l5 = args.vmax_abs if args.vmax_abs is not None else 2.0
        plot_corr_effect_size(
            eff_agg_l5,
            os.path.join(args.out_dir, f"{nt}_inhagg_l5agg"),
            vmax_abs=vmax_for_agg_l5,
        )

    print("done")


if __name__ == "__main__":
    main()
