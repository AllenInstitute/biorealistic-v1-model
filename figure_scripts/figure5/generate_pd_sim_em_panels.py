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

from analysis_shared.em_compare import load_em_pd_pickle
from analysis_shared.grouping import aggregate_l5
from analysis_shared.io import load_edges_with_pref_dir, load_edges_with_computed_pref_dir
from analysis_shared.stats import bin_mean_sem, fit_cosine_series_deg
from analysis_shared.style import apply_pub_style, trim_spines


def _load_sim_df_all(base_dirs, network_type: str, loader=None) -> pd.DataFrame:
    if loader is None:
        loader = load_edges_with_pref_dir
    from aggregate_correlation_plot import process_network_data

    dfs = []
    for bd in base_dirs:
        e = loader(bd, network_type)
        typed = process_network_data((bd, network_type))[
            ["source_id", "target_id", "source_type", "target_type"]
        ]
        e = e.merge(typed, on=["source_id", "target_id"], how="left").dropna(
            subset=["source_type", "target_type"]
        )
        dfs.append(e[["pref_dir_diff_deg", "syn_weight", "source_type", "target_type"]])
    df = pd.concat(dfs, ignore_index=True)
    df = aggregate_l5(df)
    df = df[
        df["source_type"].str.contains("Exc") & df["target_type"].str.contains("Exc")
    ]
    return df


def _filter_pair(
    df_all: pd.DataFrame, source_group: list[str], target_group: list[str]
) -> pd.DataFrame:
    sub = df_all[
        df_all["source_type"].isin(source_group)
        & df_all["target_type"].isin(target_group)
    ]
    return sub[["pref_dir_diff_deg", "syn_weight"]].dropna()


def _em_pd_group(
    em_data: dict, source_group: list[str], target_group: list[str]
) -> pd.DataFrame:
    parts = []
    for s in source_group:
        for t in target_group:
            key = (s, t)
            if key in em_data:
                df = em_data[key]
                if not df.empty:
                    parts.append(
                        df[["x", "y"]].rename(
                            columns={"x": "pref_dir_diff_deg", "y": "syn_weight"}
                        )
                    )
    if parts:
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame(columns=["pref_dir_diff_deg", "syn_weight"])


def _plot_hist_fit(
    ax,
    df: pd.DataFrame,
    *,
    bin_step: float = 20.0,
    show_fit: bool = True,
    annotate_pvalues: bool = False,
    show_xlabel: bool = False,
):
    x_min, x_max = 0.0, 180.0
    bins = np.arange(x_min, x_max + bin_step, bin_step)
    stats = {
        "N": 0,
        "a": np.nan,
        "b": np.nan,
        "c": np.nan,
        "p_a": np.nan,
        "p_b": np.nan,
        "a_over_abs_c": np.nan,
        "b_over_abs_c": np.nan,
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
        xx = df["pref_dir_diff_deg"].to_numpy(dtype=float)
        yy = df["syn_weight"].to_numpy(dtype=float)
        stats["N"] = int(len(xx))
        centers, means, sems = bin_mean_sem(xx, yy, bins)
        ax.bar(centers, means, width=bin_step, color="#aec7e8", edgecolor="none")
        ax.errorbar(
            centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2
        )
        text_lines = [f"N={len(xx)}"]
        if show_fit and len(xx) >= 10:
            xs = np.linspace(x_min, x_max, 361)
            fit = fit_cosine_series_deg(xx, yy)
            stats.update(
                {
                    "a": float(fit.a),
                    "b": float(fit.b),
                    "c": float(fit.c),
                    "p_a": float(fit.p_a),
                    "p_b": float(fit.p_b),
                }
            )
            denom = abs(fit.c) if np.isfinite(fit.c) and fit.c != 0 else np.nan
            if np.isfinite(denom):
                stats["a_over_abs_c"] = float(fit.a / denom)
                stats["b_over_abs_c"] = float(fit.b / denom)
            ys = (
                fit.a * np.cos(np.radians(xs))
                + fit.b * np.cos(2 * np.radians(xs))
                + fit.c
            )
            ax.plot(xs, ys, color="crimson", linewidth=1.2)
            text_lines.append(f"a/|c|={stats['a_over_abs_c']:.2f}")
            text_lines.append(f"b/|c|={stats['b_over_abs_c']:.2f}")
            if annotate_pvalues:
                text_lines.append(f"p(a)={fit.p_a:.1e}")
                text_lines.append(f"p(b)={fit.p_b:.1e}")
        ax.text(
            0.5,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="center",
        )
        ax.set_xlim(x_min, x_max)
    trim_spines(ax)
    if show_xlabel:
        ax.set_xlabel("ΔPD (deg)", fontsize=8)
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Generate 2x3 PD panels: simulation vs EM (hist + fit)"
    )
    ap.add_argument("--bases", nargs="*", default=None)
    ap.add_argument("--network-type", default="bio_trained")
    ap.add_argument("--out", default="figures/paper/pd_sim_em_panels.png")
    ap.add_argument("--bin-step", type=float, default=20.0)
    ap.add_argument("--no-computed-pd", action="store_false", dest="use_computed_pd",
                    help="Revert to structural tuning_angle instead of response-derived PD")
    ap.add_argument("--min-fr", type=float, default=1.0,
                    help="Min max_mean_rate(Hz) threshold for response-derived PD")
    ap.set_defaults(use_computed_pd=True)
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
        (["L2/3_Exc"], ["L5_Exc"], "L2/3 Exc → L5 Exc"),
        (["L5_Exc"], ["L5_Exc"], "L5 Exc → L5 Exc"),
    ]

    loader = None
    if getattr(args, 'use_computed_pd', True):
        from functools import partial
        loader = partial(load_edges_with_computed_pref_dir, min_fr=args.min_fr)
    sim_all = _load_sim_df_all(bases, args.network_type, loader=loader)
    em = load_em_pd_pickle(
        os.path.join(
            "analysis_shared", "delta_pref_direction_vs_weight_v1dd_250827.pkl"
        )
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
            bin_step=args.bin_step,
            show_fit=True,
            annotate_pvalues=False,
            show_xlabel=(i == 2),
        )
        axes[i, 0].set_ylabel("Weight", fontsize=8)
        rows.append({"dataset": "simulation", "pair": label, **sim_stats})

        src_em = [
            s.replace("_Exc", "")
            for s in src_grp
            if s in ("L2/3_Exc", "L4_Exc", "L5_Exc")
        ]
        tgt_em = [
            t.replace("_Exc", "")
            for t in tgt_grp
            if t in ("L2/3_Exc", "L4_Exc", "L5_Exc")
        ]
        em_df = _em_pd_group(em, src_em, tgt_em)
        em_stats = _plot_hist_fit(
            axes[i, 1],
            em_df,
            bin_step=args.bin_step,
            show_fit=True,
            annotate_pvalues=True,
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
