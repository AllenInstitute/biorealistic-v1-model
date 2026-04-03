#!/usr/bin/env python3
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis_shared.style import apply_pub_style, trim_spines
from analysis_shared.celltype_labels import abbrev_cell_type
from analysis_shared.io import load_edges_with_pref_dir, load_edges_with_computed_pref_dir
from analysis_shared.grouping import aggregate_l5, apply_inh_simplification
from analysis_shared.stats import bin_mean_sem, fit_cosine_series_deg, ols_slope_p
from analysis_shared.pd import compute_pd_full_matrix_cache
from analysis_shared.corr import compute_corr_full_matrix_cache

from aggregate_correlation_plot import process_network_data


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
    return abbrev_cell_type(lbl)


def _format_n_millions(n: int | float | None) -> str:
    if n is None:
        return "N=—"
    m = float(n) / 1e6
    return f"N={m:.2f}M"


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


def _domain_mean_from_bin_means(centers: np.ndarray, means: np.ndarray, lo: float, hi: float) -> float:
    centers = np.asarray(centers)
    means = np.asarray(means)
    mask = (centers >= lo) & (centers <= hi) & np.isfinite(centers) & np.isfinite(means)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(means[mask]))


def _binstats_min_count(
    x: np.ndarray, y: np.ndarray, bins: np.ndarray, *, min_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    centers = (bins[:-1] + bins[1:]) / 2.0
    n_bins = len(centers)
    if x.size == 0:
        nan = np.full(n_bins, np.nan, dtype=float)
        return centers, nan, nan, np.zeros(n_bins, dtype=int)

    idx = np.digitize(x, bins, right=False) - 1
    ok = (idx >= 0) & (idx < n_bins)
    idx = idx[ok]
    y = y[ok]

    counts = np.bincount(idx, minlength=n_bins).astype(int)
    sumy = np.bincount(idx, weights=y, minlength=n_bins).astype(float)
    sumy2 = np.bincount(idx, weights=y * y, minlength=n_bins).astype(float)

    means = np.full(n_bins, np.nan, dtype=float)
    sems = np.full(n_bins, np.nan, dtype=float)

    good = counts > 0
    means[good] = sumy[good] / counts[good]

    n2 = counts >= 2
    if np.any(n2):
        var = (sumy2[n2] - (sumy[n2] * sumy[n2]) / counts[n2]) / (counts[n2] - 1)
        var = np.maximum(var, 0.0)
        sems[n2] = np.sqrt(var) / np.sqrt(counts[n2])

    bad = counts < int(min_count)
    means[bad] = np.nan
    sems[bad] = np.nan
    return centers, means, sems, counts


def _render_corr_hist_rangeavg(
    ax: plt.Axes,
    *,
    centers: np.ndarray,
    means: np.ndarray,
    sems: np.ndarray | None,
    n_conn: int | None,
    title: str | None = None,
    xlim: tuple[float, float] = (-1.0, 1.0),
    xticks: list[float] | None = None,
    bar_color: str = "#f4b6c2",
    bar_alpha: float = 0.35,
    delta_label_pos: str | None = None,
) -> None:
    ax.bar(
        centers,
        means,
        width=(centers[1] - centers[0]) if centers.size > 1 else 0.05,
        color=bar_color,
        edgecolor="none",
        alpha=bar_alpha,
    )
    if sems is not None:
        ax.errorbar(
            centers,
            means,
            yerr=sems,
            fmt="none",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )

    x_min, x_max = xlim
    anti_mean = _domain_mean_from_bin_means(centers, means, -1.0, -0.5)
    none_mean = _domain_mean_from_bin_means(centers, means, -0.25, 0.25)
    like_mean = _domain_mean_from_bin_means(centers, means, 0.5, 1.0)

    if np.isfinite(none_mean) and none_mean != 0:
        delta_anti = (
            (none_mean - anti_mean) / none_mean if np.isfinite(anti_mean) else np.nan
        )
        delta_like = (
            (like_mean - none_mean) / none_mean if np.isfinite(like_mean) else np.nan
        )
    else:
        delta_anti = np.nan
        delta_like = np.nan

    text_lines = []
    if np.isfinite(delta_anti):
        text_lines.append(f"Δanti={delta_anti:.2f}")
    if np.isfinite(delta_like):
        text_lines.append(f"Δlike={delta_like:.2f}")
    overall_mean = float(np.nanmean(means)) if means.size else 0.0
    if text_lines:
        if delta_label_pos == "bottom":
            ty = 0.08
            tva = "bottom"
        elif delta_label_pos == "top":
            ty = 0.92
            tva = "top"
        else:
            ty = 0.08 if overall_mean >= 0 else 0.92
            tva = "bottom" if overall_mean >= 0 else "top"

        ax.text(
            0.03,
            ty,
            "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=6,
            va=tva,
        )

    seg_color = "#4C78A8"
    domain_segments = [
        (-1.0, -0.5, anti_mean),
        (-0.25, 0.25, none_mean),
        (0.5, 1.0, like_mean),
    ]
    for xmin, xmax, yseg in domain_segments:
        if not np.isfinite(yseg):
            continue
        ax.hlines(
            y=yseg,
            xmin=max(xmin, x_min),
            xmax=min(xmax, x_max),
            color=seg_color,
            linewidth=1.2,
            alpha=0.85,
        )

    x_pts = np.array([-0.75, 0.0, 0.75])
    y_pts = np.array([anti_mean, none_mean, like_mean], dtype=float)
    if np.all(np.isfinite(y_pts)):
        ax.plot(
            x_pts,
            y_pts,
            color=seg_color,
            linewidth=1.0,
            marker="o",
            markersize=2.5,
            alpha=0.95,
            zorder=3,
        )
    if n_conn is not None:
        ax.text(
            0.97,
            0.08 if overall_mean >= 0 else 0.92,
            _format_n_millions(n_conn),
            transform=ax.transAxes,
            fontsize=6,
            ha="right",
            va="bottom" if overall_mean >= 0 else "top",
        )
    ax.set_xlim(x_min, x_max)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelsize=6, pad=1)
    ax.tick_params(axis="y", labelsize=6, pad=1)
    if title is not None:
        ax.set_title(title, fontsize=7, pad=1.5)

    if x_min <= 0 <= x_max:
        ax.axvline(0.0, color="#999999", linewidth=0.7, alpha=0.6, zorder=0)
    y0, y1 = ax.get_ylim()
    if y0 <= 0 <= y1:
        ax.axhline(0.0, color="#aaaaaa", linewidth=0.7, alpha=0.7, zorder=0)
    trim_spines(ax)


def _load_pd_cache_for_nt(bases: list[str], nt: str, loader=None, force_recompute: bool = False) -> dict:
    cache_dir = os.path.join("figures", "paper", "cache_side_by_side")
    os.makedirs(cache_dir, exist_ok=True)
    suffix = "_computedpd" if loader is not None else ""
    cache_path = os.path.join(cache_dir, f"pd_sim_full_matrix_{nt}{suffix}.pkl")
    cache = None
    if (not force_recompute) and os.path.isfile(cache_path):
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
            loader=loader,
        )
        try:
            import pickle
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)
        except Exception:
            pass
    return cache


def _is_layer_agnostic_inh(lbl: str) -> bool:
    return lbl in {"PV", "SST", "VIP", "L1_Inh"}


def _load_corr_selected_pairs_cache_for_nt(
    bases: list[str],
    nt: str,
    example_pairs: list[tuple[str, str]],
    *,
    cache_dir: str,
    force_recompute: bool,
    bin_size: float = 0.05,
    min_bin_count: int = 5,
) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f"corr_side_by_side_examples_{nt}_bin{bin_size:.3f}_min{min_bin_count}.pkl",
    )
    if (not force_recompute) and os.path.isfile(cache_path):
        try:
            import pickle

            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    bins = np.arange(-1.0, 1.0 + bin_size, bin_size)
    parts_by_pair: dict[tuple[str, str], list[pd.DataFrame]] = {
        pair: [] for pair in example_pairs
    }
    for bd in bases:
        df = process_network_data((bd, nt))
        df = df[["source_type", "target_type", "Response Correlation", "syn_weight"]]
        df_simpl = apply_inh_simplification(df)
        for s, t in example_pairs:
            view = df_simpl if (_is_layer_agnostic_inh(s) or _is_layer_agnostic_inh(t)) else df
            sub = view[(view["source_type"] == s) & (view["target_type"] == t)]
            if not sub.empty:
                parts_by_pair[(s, t)].append(sub)

    pairs_out: dict[tuple[str, str], dict | None] = {}
    for s, t in example_pairs:
        parts = parts_by_pair.get((s, t), [])
        if not parts:
            pairs_out[(s, t)] = None
            continue
        sub = pd.concat(parts, ignore_index=True).dropna(
            subset=["Response Correlation", "syn_weight"]
        )
        if sub.empty:
            pairs_out[(s, t)] = None
            continue
        x = sub["Response Correlation"].to_numpy()
        y = sub["syn_weight"].to_numpy()
        m = (x >= -1.0) & (x <= 1.0) & np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 3:
            pairs_out[(s, t)] = None
            continue
        cts, means, sems, counts = _binstats_min_count(x, y, bins, min_count=min_bin_count)
        pairs_out[(s, t)] = {
            "centers": cts,
            "means": means,
            "sems": sems,
            "counts": counts,
            "N": int(x.size),
        }

    cache = {
        "x_min": -1.0,
        "x_max": 1.0,
        "bin_size": float(bin_size),
        "min_bin_count": int(min_bin_count),
        "pairs": pairs_out,
    }
    try:
        import pickle

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass
    return cache


def _load_corr_cache_for_nt(bases: list[str], nt: str, aggregate_l5_types: bool = True) -> dict:
    cache_dir = os.path.join("figures", "paper", "cache_side_by_side")
    os.makedirs(cache_dir, exist_ok=True)
    cache_suffix = "l5agg" if aggregate_l5_types else "l5split"
    cache_path = os.path.join(cache_dir, f"corr_sim_full_matrix_{nt}_{cache_suffix}.pkl")
    old_cache_path = os.path.join(cache_dir, f"corr_sim_full_matrix_{nt}.pkl")
    cache = None
    if os.path.isfile(cache_path):
        try:
            import pickle

            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = None
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


def plot_pd_side_by_side(bases: list[str], out_png: str, loader=None, force_recompute: bool = False) -> None:
    apply_pub_style()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    # Half-column width figure; shrink vertical by ~30%
    fig, axes = plt.subplots(1, 2, figsize=(3.0, 1.4))
    pairs = [("bio_trained", axes[0]), ("naive", axes[1])]
    s, t = "L2/3_Exc", "L2/3_Exc"
    for nt, ax in pairs:
        cache = _load_pd_cache_for_nt(bases, nt, loader=loader, force_recompute=force_recompute)
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
        ax.set_title(f"{s}→{t}", fontsize=7, pad=1.0)
        ax.set_xlabel(r'$\delta$ Pref. dir. (deg)', fontsize=7, labelpad=1.0)
        if nt == "bio_trained":
            ax.set_ylabel("Weight (pA)", fontsize=7)
        ax.set_xticks([0, 90, 180])
        ax.set_xlim(0.0, 180.0)
        ax.tick_params(axis="both", labelsize=6, pad=1)
        trim_spines(ax)
    # Sync Y-axis range across the pair
    ylims = [ax.get_ylim() for ax in axes]
    y_lo = min(yl[0] for yl in ylims)
    y_hi = max(yl[1] for yl in ylims)
    for ax in axes:
        ax.set_ylim(y_lo, y_hi)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    if out_png.lower().endswith(".png"):
        out_pdf = out_png[:-4] + ".pdf"
        fig.savefig(out_pdf)
    plt.close(fig)


def plot_corr_side_by_side(bases: list[str], out_png: str, *, force_recompute: bool = False) -> None:
    apply_pub_style()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    # Three rows (pairs) x two columns (bio, naive), half-column width overall; shrink vertical by ~30%
    fig, axes = plt.subplots(3, 2, figsize=(3.0, 3.4))
    try:
        import matplotlib as mpl
        from matplotlib import font_manager as fm

        _ = fm.findfont("Arial", fallback_to_default=False)
        mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"]})
    except Exception:
        pass

    example_pairs = [
        ("L4_Exc", "L4_Exc"),
        ("L2/3_PV", "L2/3_Exc"),
        ("PV", "PV"),
    ]
    xticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    cache_dir = os.path.join(os.path.dirname(out_png), "cache_side_by_side")
    caches = {
        nt: _load_corr_selected_pairs_cache_for_nt(
            bases,
            nt,
            example_pairs,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            bin_size=0.05,
            min_bin_count=5,
        )
        for nt in ["bio_trained", "naive"]
    }
    for i, (s, t) in enumerate(example_pairs):
        for j, nt in enumerate(["bio_trained", "naive"]):
            ax = axes[i, j]
            cache = caches.get(nt)
            entry = cache.get("pairs", {}).get((s, t)) if cache is not None else None
            if not entry:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                trim_spines(ax)
                continue
            title = f"{s}→{t}"
            delta_label_pos = None
            if nt == "naive" and s == "L2/3_PV" and t == "L2/3_Exc":
                delta_label_pos = "bottom"

            _render_corr_hist_rangeavg(
                ax,
                centers=entry["centers"],
                means=entry["means"],
                sems=entry["sems"],
                n_conn=entry.get("N"),
                title=title,
                xlim=(-1.0, 1.0),
                xticks=xticks,
                delta_label_pos=delta_label_pos,
            )
            if j == 0:
                ax.set_ylabel("Weight (pA)", fontsize=7)
            if i == len(example_pairs) - 1:
                ax.set_xlabel("Response Corr.", fontsize=7, labelpad=1.0)
        # Sync Y-axis range for this pair row
        row_axes = [axes[i, j] for j in range(2)]
        ylims = [ax.get_ylim() for ax in row_axes]
        y_lo = min(yl[0] for yl in ylims)
        y_hi = max(yl[1] for yl in ylims)
        for ax in row_axes:
            ax.set_ylim(y_lo, y_hi)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    if out_png.lower().endswith(".png"):
        out_pdf = out_png[:-4] + ".pdf"
        fig.savefig(out_pdf)
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
            ax.set_title(f"{s}→{t}", fontsize=7, pad=1.0)
            ax.set_xlabel("Response Corr.", fontsize=7, labelpad=1.0)
            if j == 0:
                ax.set_ylabel("Weight (pA)", fontsize=7)
            ax.set_xlim(x_min, x_max)
            ax.set_xticks([-0.2, 0.0, 0.2, 0.4])
            ax.tick_params(axis="both", labelsize=6, pad=1)
            trim_spines(ax)
        # Sync Y-axis range for this pair row
        row_axes = [axes[i, j] for j in range(2)]
        ylims = [ax.get_ylim() for ax in row_axes]
        y_lo = min(yl[0] for yl in ylims)
        y_hi = max(yl[1] for yl in ylims)
        for ax in row_axes:
            ax.set_ylim(y_lo, y_hi)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    if out_png.lower().endswith(".png"):
        out_pdf = out_png[:-4] + ".pdf"
        fig.savefig(out_pdf)
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
            ax.set_title(f"{s}→{t}", fontsize=7, pad=1.0)
            ax.set_xlabel("Response Corr.", fontsize=7, labelpad=1.0)
            if j == 0:
                ax.set_ylabel("Weight (pA)", fontsize=7)
            ax.set_xlim(x_min, x_max)
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.tick_params(axis="both", labelsize=6, pad=1)
            trim_spines(ax)
        # Sync Y-axis range for this pair row
        row_axes = [axes[i, j] for j in range(2)]
        ylims = [ax.get_ylim() for ax in row_axes]
        y_lo = min(yl[0] for yl in ylims)
        y_hi = max(yl[1] for yl in ylims)
        for ax in row_axes:
            ax.set_ylim(y_lo, y_hi)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    if out_png.lower().endswith(".png"):
        out_pdf = out_png[:-4] + ".pdf"
        fig.savefig(out_pdf)
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
    ap.add_argument("--out-dir", default="figures/paper/figure6", help="Output directory")
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument("--no-computed-pd", action="store_false", dest="use_computed_pd",
                    help="Revert to structural tuning_angle instead of response-derived PD")
    ap.add_argument("--min-fr", type=float, default=1.0,
                    help="Min max_mean_rate(Hz) threshold for response-derived PD")
    ap.set_defaults(use_computed_pd=True)
    args = ap.parse_args()

    bases = discover_bases()
    if not bases:
        raise SystemExit("No base directories found.")

    from functools import partial
    loader = partial(load_edges_with_computed_pref_dir, min_fr=args.min_fr) if args.use_computed_pd else None

    os.makedirs(args.out_dir, exist_ok=True)
    plot_pd_side_by_side(bases, os.path.join(args.out_dir, "pd_side_by_side_E23_E23.png"),
                         loader=loader, force_recompute=args.force_recompute)
    plot_corr_side_by_side(bases, os.path.join(args.out_dir, "corr_side_by_side_examples.png"), force_recompute=args.force_recompute)
    plot_corr_side_by_side_new(bases, os.path.join(args.out_dir, "corr_side_by_side_examples_new.png"), force_recompute=args.force_recompute)
    plot_corr_side_by_side_extras(bases, os.path.join(args.out_dir, "corr_side_by_side_extras.png"), force_recompute=args.force_recompute)
    print("saved side-by-side figures under", args.out_dir)


if __name__ == "__main__":
    main()


