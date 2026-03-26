#!/usr/bin/env python3
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_shared.style import apply_pub_style, trim_spines
from analysis_shared.io import load_edges_with_pref_dir
from analysis_shared.grouping import aggregate_l5, apply_inh_simplification
from analysis_shared.stats import bin_mean_sem, fit_cosine_series_deg, ols_slope_p
from analysis_shared.pd import compute_pd_full_matrix_cache
from analysis_shared.corr import compute_corr_full_matrix_cache


def _load_pd_edges(bases: list[str], network_type: str, aggregate_l5_types: bool = True, simplify_inh: bool = True) -> pd.DataFrame:
    dfs = []
    for bd in bases:
        e = load_edges_with_pref_dir(bd, network_type)
        try:
            from aggregate_correlation_plot import process_network_data
            typed = process_network_data((bd, network_type))
            typed = typed[["source_id", "target_id", "source_type", "target_type"]]
            e = e.merge(typed, on=["source_id", "target_id"], how="left")
        except Exception:
            pass
        dfs.append(e)
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["source_type","target_type","pref_dir_diff_deg","syn_weight"]).copy()
    if simplify_inh:
        df = apply_inh_simplification(df)
    if aggregate_l5_types:
        df = aggregate_l5(df)
    return df


def _load_corr_edges(bases: list[str], network_type: str, aggregate_l5_types: bool = True, simplify_inh: bool = True) -> pd.DataFrame:
    from aggregate_correlation_plot import process_network_data
    dfs = []
    for bd in bases:
        df = process_network_data((bd, network_type))
        dfs.append(df[["source_type","target_type","Response Correlation","syn_weight"]].copy())
    df = pd.concat(dfs, ignore_index=True)
    if simplify_inh:
        df = apply_inh_simplification(df)
    if aggregate_l5_types:
        df = aggregate_l5(df)
    df = df.dropna(subset=["source_type","target_type","Response Correlation","syn_weight"]).copy()
    return df


def _simplify(lbl: str) -> str:
    mapping = {"L2/3_Exc": "E23", "L4_Exc": "E4", "L5_Exc": "E5", "L6_Exc": "E6", "L5_IT": "E5IT", "L5_ET": "E5ET", "L5_NP": "E5NP", "L1_Inh": "L1"}
    return mapping.get(lbl, lbl.replace("_Exc", "").replace("_", ""))


def _format_n_millions(n: int | float | None) -> str:
    if n is None:
        return "N=—"
    m = float(n) / 1e6
    if m >= 10.0:
        s = f"{m:.1f}M"
    else:
        s = f"{m:.2f}M"
    return f"N={s}"


def render_histogram_with_alpha(ax, centers, means, color, alpha):
    # Revert width hack
    # Use step fill to avoid internal seams
    if len(centers) > 1:
        w = centers[1] - centers[0]
        edges = np.concatenate([centers - w/2, [centers[-1] + w/2]])
        # stairs was added in mpl 3.4
        if hasattr(ax, "stairs"):
            ax.stairs(means, edges, fill=True, color=color, alpha=alpha, baseline=0)
        else:
            # Manual step fill
            x = np.zeros(2 * len(means) + 2)
            y = np.zeros(2 * len(means) + 2)
            x[0] = edges[0]; y[0] = 0
            x[1:-1:2] = edges[:-1]
            x[2:-1:2] = edges[1:]
            y[1:-1:2] = means
            y[2:-1:2] = means
            x[-1] = edges[-1]; y[-1] = 0
            ax.fill(x, y, facecolor=color, alpha=alpha, edgecolor="none")
    else:
        # Fallback for single bin
        ax.bar(centers, means, width=0.05, color=color, alpha=alpha, edgecolor="none")


def _load_pd_cache_for_nt(bases: list[str], nt: str) -> dict:
    cache_dir = os.path.join("figures", "paper", "cache_side_by_side")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"pd_sim_full_matrix_{nt}.pkl")
    cache = None
    if os.path.isfile(cache_path):
        try:
            import pickle
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = None
    if cache is None:
        cache = compute_pd_full_matrix_cache(
            bases,
            nt,
            simplify_inh=True,
            aggregate_l5_types=True,
            bin_step=20.0,
            max_per_pair=None,
            pair_limits_csv=None,
            sample_seed=0,
        )
        try:
            import pickle
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
        except Exception:
            pass
    return cache


def plot_pd_side_by_side(bases: list[str], out_png: str) -> None:
    apply_pub_style()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    # Half-column width figure; shrink vertical by ~30%
    fig, axes = plt.subplots(1, 2, figsize=(3.0, 1.4))
    pairs = [("bio_trained", axes[0]), ("naive", axes[1])]
    s, t = "L2/3_Exc", "L2/3_Exc"
    for nt, ax in pairs:
        cache = _load_pd_cache_for_nt(bases, nt)
        entry = cache.get("pairs", {}).get((s, t))
        centers = cache.get("centers") if cache is not None else None
        if (not entry) or (centers is None):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center"); trim_spines(ax); continue
        means = entry["means"]; sems = entry["sems"]; fit = entry["fit"]
        n_conn = int(entry.get("N", 0))
        x_min, x_max = 0.0, 180.0
        xs = np.linspace(x_min, x_max, 361)
        ys = fit["a"] * np.cos(np.radians(xs)) + fit["b"] * np.cos(2 * np.radians(xs)) + fit["c"]
        ax.bar(centers, means, width=20.0, color="#aec7e8", edgecolor="none")
        ax.errorbar(centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2)
        ax.plot(xs, ys, color="crimson", linewidth=1.2)
        # Effect-size text (a/c, b/c) matching final panel style (no p-values for sim)
        denom = fit.get("c", np.nan)
        a_over_c = (fit.get("a", np.nan) / denom) if (np.isfinite(denom) and denom != 0) else np.nan
        b_over_c = (fit.get("b", np.nan) / denom) if (np.isfinite(denom) and denom != 0) else np.nan
        overall_mean = float(np.nanmean(means)) if means.size else 0.0
        text_y = 0.08 if overall_mean >= 0 else 0.92
        ax.text(0.03, text_y, f"a/c={a_over_c:.2f}\nb/c={b_over_c:.2f}", transform=ax.transAxes, fontsize=6, va=("bottom" if overall_mean>=0 else "top"))
        ax.text(0.97, 0.08 if overall_mean>=0 else 0.92, _format_n_millions(n_conn), transform=ax.transAxes, fontsize=6, ha="right", va=("bottom" if overall_mean>=0 else "top"))
        ax.set_title(f"{_simplify(s)}→{_simplify(t)}  ({'bio' if nt=='bio_trained' else 'naive'})", fontsize=7, pad=1.0)
        ax.set_xlabel(r'$\delta$ Pref. Dir. (deg)', fontsize=7, labelpad=1.0)
        if nt == "bio_trained":
            ax.set_ylabel("Weight (pA)", fontsize=7)
        ax.set_xticks([0, 90, 180])
        ax.set_xlim(0.0, 180.0)
        ax.tick_params(axis="both", labelsize=6, pad=1)
        trim_spines(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _load_corr_cache_for_nt(bases: list[str], nt: str, aggregate_l5_types: bool = True) -> dict:
    cache_dir = os.path.join("figures", "paper", "cache_side_by_side")
    os.makedirs(cache_dir, exist_ok=True)
    cache_suffix = "l5agg" if aggregate_l5_types else "l5split"
    cache_path = os.path.join(cache_dir, f"corr_sim_full_matrix_{nt}_{cache_suffix}.pkl")
    # Check for backward compatibility with old cache files (without suffix)
    old_cache_path = os.path.join(cache_dir, f"corr_sim_full_matrix_{nt}.pkl")
    cache = None
    if os.path.isfile(cache_path):
        try:
            import pickle
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = None
    # Fallback to old cache if new one doesn't exist and aggregate_l5_types=True (old cache was aggregated)
    elif aggregate_l5_types and os.path.isfile(old_cache_path):
        try:
            import pickle
            with open(old_cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = None
    if cache is None:
        cache = compute_corr_full_matrix_cache(
            bases,
            nt,
            simplify_inh=True,
            aggregate_l5_types=aggregate_l5_types,
            bin_size=0.05,
            x_min=-0.5,
            x_max=0.5,
            max_per_pair=None,
            pair_limits_csv=None,
            sample_seed=0,
        )
        try:
            import pickle
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
        except Exception:
            pass
    return cache


def plot_corr_side_by_side(bases: list[str], out_png: str, *, force_recompute: bool = False) -> None:
    apply_pub_style()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    # Three rows (pairs) x two columns (bio, naive), half-column width overall; shrink vertical by ~30%
    fig, axes = plt.subplots(3, 2, figsize=(3.0, 3.4))
    pairs = [
        ("L2/3_Exc", "L2/3_Exc"),
        ("PV", "L6_Exc"),
        ("L6_Exc", "L6_Exc"),
    ]
    for i, (s, t) in enumerate(pairs):
        for j, nt in enumerate(["bio_trained", "naive"]):
            ax = axes[i, j]
            # If force_recompute is requested, delete old cache file to ensure new x-range is computed
            cache_dir = os.path.join("figures", "paper", "cache_side_by_side")
            cache_path = os.path.join(cache_dir, f"corr_sim_full_matrix_{nt}_l5agg.pkl")
            if force_recompute and os.path.isfile(cache_path):
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
            cache = _load_corr_cache_for_nt(bases, nt, aggregate_l5_types=True)
            entry = cache.get("pairs", {}).get((s, t)) if cache is not None else None
            centers = cache.get("centers") if cache is not None else None
            if (not entry) or (centers is None):
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                trim_spines(ax); continue
            means = entry["means"]; sems = entry["sems"]; res = entry["ols"]
            n_conn = int(entry.get("N", 0))
            x_min, x_max = -0.5, 0.5
            line_x = np.array([x_min, x_max])
            # bar_color = "#f4b6c2" if nt == "bio_trained" else "#b3b3b3"
            bar_color = "#f4b6c2"
            render_histogram_with_alpha(ax, centers, means, bar_color, 0.35)
            ax.errorbar(centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2)
            ax.plot(line_x, res["intercept"] + res["slope"] * line_x, color="crimson", linewidth=1.2)
            # Effect-size text (m/c) and N, no p-values for sim to match style
            slope = float(res.get("slope", np.nan)); intercept = float(res.get("intercept", np.nan))
            ratio = (slope / intercept) if (np.isfinite(slope) and np.isfinite(intercept) and intercept != 0.0) else np.nan
            overall_mean = float(np.nanmean(means)) if means.size else 0.0
            text_y = 0.08 if overall_mean >= 0 else 0.92
            ax.text(0.03, text_y, (f"m/c={ratio:.3f}" if np.isfinite(ratio) else f"m={slope:.3f}"), transform=ax.transAxes, fontsize=6, va=("bottom" if overall_mean>=0 else "top"))
            ax.text(0.97, 0.08 if overall_mean>=0 else 0.92, _format_n_millions(n_conn), transform=ax.transAxes, fontsize=6, ha="right", va=("bottom" if overall_mean>=0 else "top"))
            ax.set_title(f"{_simplify(s)}→{_simplify(t)}  ({'bio' if nt=='bio_trained' else 'no constraints'})", fontsize=7, pad=1.0)
            ax.set_xlabel("Response correlation", fontsize=7, labelpad=1.0)
            if j == 0:
                ax.set_ylabel("Weight (pA)", fontsize=7)
            ax.set_xlim(x_min, x_max)
            ax.set_xticks([-0.2, 0.0, 0.2, 0.4])
            ax.tick_params(axis="both", labelsize=6, pad=1)
            trim_spines(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_corr_side_by_side_new(bases: list[str], out_png: str, *, force_recompute: bool = False) -> None:
    """New version with updated pairs: E23→E5ET, PV→E4, E5IT→E23"""
    apply_pub_style()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    # Three rows (pairs) x two columns (bio, naive), half-column width overall; shrink vertical by ~30%
    fig, axes = plt.subplots(3, 2, figsize=(3.0, 3.4))
    pairs = [
        ("L2/3_Exc", "L5_ET"),
        ("PV", "L4_Exc"),
        ("L5_IT", "L2/3_Exc"),
    ]
    for i, (s, t) in enumerate(pairs):
        for j, nt in enumerate(["bio_trained", "naive"]):
            ax = axes[i, j]
            # If force_recompute is requested, delete old cache file to ensure new x-range is computed
            cache_dir = os.path.join("figures", "paper", "cache_side_by_side")
            cache_path = os.path.join(cache_dir, f"corr_sim_full_matrix_{nt}_l5split.pkl")
            if force_recompute and os.path.isfile(cache_path):
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
            # Use aggregate_l5_types=False to access L5_ET and L5_IT separately
            cache = _load_corr_cache_for_nt(bases, nt, aggregate_l5_types=False)
            entry = cache.get("pairs", {}).get((s, t)) if cache is not None else None
            centers = cache.get("centers") if cache is not None else None
            if (not entry) or (centers is None):
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                trim_spines(ax); continue
            means = entry["means"]; sems = entry["sems"]; res = entry["ols"]
            n_conn = int(entry.get("N", 0))
            x_min, x_max = -0.5, 0.5
            line_x = np.array([x_min, x_max])
            bar_color = "#f4b6c2"
            render_histogram_with_alpha(ax, centers, means, bar_color, 0.35)
            ax.errorbar(centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2)
            ax.plot(line_x, res["intercept"] + res["slope"] * line_x, color="crimson", linewidth=1.2)
            # Effect-size text (m/c) and N, no p-values for sim to match style
            slope = float(res.get("slope", np.nan)); intercept = float(res.get("intercept", np.nan))
            ratio = (slope / intercept) if (np.isfinite(slope) and np.isfinite(intercept) and intercept != 0.0) else np.nan
            overall_mean = float(np.nanmean(means)) if means.size else 0.0
            text_y = 0.08 if overall_mean >= 0 else 0.92
            ax.text(0.03, text_y, (f"m/c={ratio:.3f}" if np.isfinite(ratio) else f"m={slope:.3f}"), transform=ax.transAxes, fontsize=6, va=("bottom" if overall_mean>=0 else "top"))
            ax.text(0.97, 0.08 if overall_mean>=0 else 0.92, _format_n_millions(n_conn), transform=ax.transAxes, fontsize=6, ha="right", va=("bottom" if overall_mean>=0 else "top"))
            ax.set_title(f"{_simplify(s)}→{_simplify(t)}  ({'bio' if nt=='bio_trained' else 'no constraints'})", fontsize=7, pad=1.0)
            ax.set_xlabel("Response correlation", fontsize=7, labelpad=1.0)
            if j == 0:
                ax.set_ylabel("Weight (pA)", fontsize=7)
            ax.set_xlim(x_min, x_max)
            ax.set_xticks([-0.2, 0.0, 0.2, 0.4])
            ax.tick_params(axis="both", labelsize=6, pad=1)
            trim_spines(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_corr_side_by_side_extras(bases: list[str], out_png: str, *, force_recompute: bool = False) -> None:
    """Specific plot for extra examples: E23->E6, SST->E23, E6->E23"""
    apply_pub_style()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    # Three rows (pairs) x two columns (bio, naive)
    fig, axes = plt.subplots(3, 2, figsize=(3.0, 3.4))
    pairs = [
        ("L2/3_Exc", "L6_Exc"),
        ("L6_Exc", "SST"),
        ("SST", "L2/3_Exc"),
    ]
    for i, (s, t) in enumerate(pairs):
        for j, nt in enumerate(["bio_trained", "naive"]):
            ax = axes[i, j]
            # If force_recompute is requested, delete old cache file to ensure new x-range is computed
            cache_dir = os.path.join("figures", "paper", "cache_side_by_side")
            cache_path = os.path.join(cache_dir, f"corr_sim_full_matrix_{nt}_l5split.pkl")
            if force_recompute and os.path.isfile(cache_path):
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
            # Use aggregate_l5_types=False
            cache = _load_corr_cache_for_nt(bases, nt, aggregate_l5_types=False)
            entry = cache.get("pairs", {}).get((s, t)) if cache is not None else None
            centers = cache.get("centers") if cache is not None else None
            if (not entry) or (centers is None):
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                trim_spines(ax); continue
            means = entry["means"]; sems = entry["sems"]; res = entry["ols"]
            n_conn = int(entry.get("N", 0))
            x_min, x_max = -1.0, 1.0
            line_x = np.array([x_min, x_max])
            bar_color = "#f4b6c2"
            render_histogram_with_alpha(ax, centers, means, bar_color, 0.35)
            ax.errorbar(centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2)
            ax.plot(line_x, res["intercept"] + res["slope"] * line_x, color="crimson", linewidth=1.2)
            # Effect-size text (m/c) and N
            slope = float(res.get("slope", np.nan)); intercept = float(res.get("intercept", np.nan))
            ratio = (slope / intercept) if (np.isfinite(slope) and np.isfinite(intercept) and intercept != 0.0) else np.nan
            overall_mean = float(np.nanmean(means)) if means.size else 0.0
            text_y = 0.08 if overall_mean >= 0 else 0.92
            ax.text(0.03, text_y, (f"m/c={ratio:.3f}" if np.isfinite(ratio) else f"m={slope:.3f}"), transform=ax.transAxes, fontsize=6, va=("bottom" if overall_mean>=0 else "top"))
            ax.text(0.97, 0.08 if overall_mean>=0 else 0.92, _format_n_millions(n_conn), transform=ax.transAxes, fontsize=6, ha="right", va=("bottom" if overall_mean>=0 else "top"))
            ax.set_title(f"{_simplify(s)}→{_simplify(t)}  ({'bio' if nt=='bio_trained' else 'no constraints'})", fontsize=7, pad=1.0)
            ax.set_xlabel("Response correlation", fontsize=7, labelpad=1.0)
            if j == 0:
                ax.set_ylabel("Weight (pA)", fontsize=7)
            ax.set_xlim(x_min, x_max)
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.tick_params(axis="both", labelsize=6, pad=1)
            trim_spines(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def discover_bases(default_max: int = 10) -> list[str]:
    bases = []
    for i in range(default_max):
        d = f"core_nll_{i}"
        if os.path.isdir(d):
            bases.append(d)
    return bases


def main():
    ap = argparse.ArgumentParser(description="Generate side-by-side PD and Corr histograms for selected pairs (bio vs naive)")
    ap.add_argument("--out-dir", default="figures/paper", help="Output directory")
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    bases = discover_bases()
    if not bases:
        raise SystemExit("No base directories found.")

    os.makedirs(args.out_dir, exist_ok=True)
    plot_pd_side_by_side(bases, os.path.join(args.out_dir, "pd_side_by_side_E23_E23.png"))
    plot_corr_side_by_side(bases, os.path.join(args.out_dir, "corr_side_by_side_examples.png"), force_recompute=args.force_recompute)
    plot_corr_side_by_side_new(bases, os.path.join(args.out_dir, "corr_side_by_side_examples_new.png"), force_recompute=args.force_recompute)
    plot_corr_side_by_side_extras(bases, os.path.join(args.out_dir, "corr_side_by_side_extras.png"), force_recompute=args.force_recompute)
    print("saved side-by-side figures under", args.out_dir)


if __name__ == "__main__":
    main()


