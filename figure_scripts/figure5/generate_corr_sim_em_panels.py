#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repository root is on PYTHONPATH when run from any working directory
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_shared.em_compare import load_em_corr_pickle
from analysis_shared.grouping import aggregate_l5
from analysis_shared.stats import bin_mean_sem, ols_slope_p
from analysis_shared.style import apply_pub_style, trim_spines


def _load_sim_corr_all(base_dirs, network_type: str) -> pd.DataFrame:
    from aggregate_correlation_plot import process_network_data

    dfs = []
    for bd in base_dirs:
        df = process_network_data((bd, network_type))
        dfs.append(
            df[["source_type", "target_type", "Response Correlation", "syn_weight"]]
        )
    df = pd.concat(dfs, ignore_index=True)
    df = aggregate_l5(df)
    df = df[
        df["source_type"].str.contains("Exc") & df["target_type"].str.contains("Exc")
    ]
    return df


def _filter_pair(
    df_all: pd.DataFrame, sources: list[str], targets: list[str]
) -> pd.DataFrame:
    sub = df_all[
        df_all["source_type"].isin(sources) & df_all["target_type"].isin(targets)
    ]
    return sub[["Response Correlation", "syn_weight"]].dropna()


def _em_corr_group(
    em_data: dict, src_group: list[str], tgt_group: list[str]
) -> pd.DataFrame:
    parts = []
    for s in src_group:
        for t in tgt_group:
            key = (s, t)
            if key in em_data:
                df = em_data[key]
                if not df.empty:
                    parts.append(
                        df[["x", "y"]].rename(
                            columns={"x": "Response Correlation", "y": "syn_weight"}
                        )
                    )
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame(columns=["Response Correlation", "syn_weight"])


def _plot_hist_fit(
    ax,
    df: pd.DataFrame,
    *,
    bin_size: float = 0.05,
    x_min: float = -0.2,
    x_max: float = 0.5,
    show_fit: bool = True,
    annotate_p: bool = False,
    show_xlabel: bool = False,
):
    stats = {
        "N": 0,
        "slope": np.nan,
        "intercept": np.nan,
        "p": np.nan,
        "slope_over_abs_intercept": np.nan,
    }
    if df.empty or len(df) < 2:
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.set_xlim(x_min, x_max)
    else:
        xx = df["Response Correlation"].to_numpy(dtype=float)
        yy = df["syn_weight"].to_numpy(dtype=float)
        mask = (xx >= x_min) & (xx <= x_max)
        xx = xx[mask]
        yy = yy[mask]
        stats["N"] = int(len(xx))
        bins = np.arange(x_min, x_max + bin_size, bin_size)
        centers, means, sems = bin_mean_sem(xx, yy, bins)
        ax.bar(centers, means, width=bin_size, color="#b3b3b3", edgecolor="none")
        ax.errorbar(
            centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2
        )
        text = [f"N={len(xx)}"]
        if show_fit and len(xx) >= 10:
            res = ols_slope_p(xx, yy)
            stats.update(
                {
                    "slope": float(res.slope),
                    "intercept": float(res.intercept),
                    "p": float(res.p_value),
                }
            )
            denom = (
                abs(res.intercept)
                if np.isfinite(res.intercept) and res.intercept != 0
                else np.nan
            )
            if np.isfinite(denom):
                stats["slope_over_abs_intercept"] = float(res.slope / denom)
            line_x = np.array([x_min, x_max])
            ax.plot(
                line_x,
                res.intercept + res.slope * line_x,
                color="crimson",
                linewidth=1.2,
            )
            text.append(f"m/|b|={stats['slope_over_abs_intercept']:.2f}")
            if annotate_p and np.isfinite(res.p_value):
                text.append(f"p={res.p_value:.1e}")
        ax.text(
            0.5,
            0.98,
            "\n".join(text),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="center",
        )
        ax.set_xlim(x_min, x_max)
    trim_spines(ax)
    if show_xlabel:
        ax.set_xlabel("Response correlation", fontsize=8)
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Generate 2x3 Corr panels: simulation vs EM (bar + OLS fit)"
    )
    ap.add_argument("--bases", nargs="*", default=None)
    ap.add_argument("--network-type", default="bio_trained")
    ap.add_argument("--out", default="figures/paper/corr_sim_em_panels.png")
    ap.add_argument("--bin-size", type=float, default=0.05)
    args = ap.parse_args()

    apply_pub_style()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    bases = (
        args.bases
        if args.bases
        else [f"core_nll_{i}" for i in range(10) if os.path.isdir(f"core_nll_{i}")]
    )

    pairs = [
        (
            ["L2/3_Exc", "L4_Exc", "L5_Exc", "L6_Exc"],
            ["L2/3_Exc", "L4_Exc", "L5_Exc", "L6_Exc"],
            "All Exc → All Exc",
        ),
        (["L4_Exc"], ["L4_Exc"], "L4 Exc → L4 Exc"),
        (["L4_Exc"], ["L5_Exc"], "L4 Exc → L5 Exc"),
    ]

    sim_all = _load_sim_corr_all(bases, args.network_type)
    em = load_em_corr_pickle(
        os.path.join("analysis_shared", "corr_vs_weight_minnie_250828.pkl"),
        split_l5=False,
    )

    fig, axes = plt.subplots(3, 2, figsize=(4.2, 4.8), constrained_layout=True)
    axes[0, 0].set_title("Simulation", fontsize=10)
    axes[0, 1].set_title("Experimental", fontsize=10)

    rows = []
    for i, (src_grp, tgt_grp, label) in enumerate(pairs):
        sim_df = _filter_pair(sim_all, src_grp, tgt_grp)
        sim_stats = _plot_hist_fit(
            axes[i, 0],
            sim_df,
            bin_size=args.bin_size,
            show_fit=True,
            annotate_p=False,
            show_xlabel=(i == 2),
        )
        axes[i, 0].set_ylabel("Weight", fontsize=8)
        rows.append({"dataset": "simulation", "pair": label, **sim_stats})

        allowed = {"L2/3", "L4", "L5"}
        src_em = [
            s.replace("_Exc", "") for s in src_grp if s.replace("_Exc", "") in allowed
        ]
        tgt_em = [
            t.replace("_Exc", "") for t in tgt_grp if t.replace("_Exc", "") in allowed
        ]
        em_df = _em_corr_group(em, src_em, tgt_em)
        em_stats = _plot_hist_fit(
            axes[i, 1],
            em_df,
            bin_size=args.bin_size,
            show_fit=True,
            annotate_p=True,
            show_xlabel=(i == 2),
        )
        rows.append({"dataset": "experimental", "pair": label, **em_stats})

    for i, (_, _, label) in enumerate(pairs):
        y = 1.0 - (i + 0.5) / 3.0
        fig.text(-0.02, y, label, va="center", ha="left", fontsize=8, rotation=90)

    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"saved: {args.out}")

    csv_out = os.path.splitext(args.out)[0] + "_effect_sizes.csv"
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"saved: {csv_out}")


if __name__ == "__main__":
    main()
