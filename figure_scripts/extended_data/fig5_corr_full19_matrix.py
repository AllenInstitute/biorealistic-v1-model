#!/usr/bin/env python3
"""
Extended data for Fig. 5 – Response-correlation like-to-like analysis.
Generates two full-page publication figures:
  1. 19×19 histogram matrix  (weight vs. response correlation, per cell-type pair)
  2. Companion heatmaps      (Δlike and Δanti, side-by-side on one page)

Usage (from repo root, conda env new_v1):
  python figure_scripts/extended_data/fig5_corr_full19_matrix.py
  python figure_scripts/extended_data/fig5_corr_full19_matrix.py --heatmap-only
  python figure_scripts/extended_data/fig5_corr_full19_matrix.py --matrix-only
  python figure_scripts/extended_data/fig5_corr_full19_matrix.py --network-type naive
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis_shared.celltype_labels import abbrev_cell_type, abbrev_cell_types
from analysis_shared.grouping import apply_inh_simplification
from analysis_shared.style import apply_pub_style, trim_spines
from aggregate_correlation_plot import process_network_data

# ---------------------------------------------------------------------------
# Cell-type ordering (full 19 types, matching AGENTS.md convention)
# ---------------------------------------------------------------------------
EXC_FULL = ["L2/3_Exc", "L4_Exc", "L5_IT", "L5_ET", "L5_NP", "L6_Exc"]
INH_FULL = [
    "L1_Inh",
    "L2/3_PV", "L4_PV", "L5_PV", "L6_PV",
    "L2/3_SST", "L4_SST", "L5_SST", "L6_SST",
    "L2/3_VIP", "L4_VIP", "L5_VIP", "L6_VIP",
]
ALL_19 = EXC_FULL + INH_FULL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_font():
    try:
        from matplotlib import font_manager as fm
        fm.findfont("Arial", fallback_to_default=False)
        mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"]})
    except Exception:
        pass


def _detect_bases(max_n: int = 10) -> list[str]:
    return [f"core_nll_{i}" for i in range(max_n) if os.path.isdir(f"core_nll_{i}")]


def _binstats(
    x: np.ndarray, y: np.ndarray, bins: np.ndarray, *, min_count: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (centers, means, sems, counts). Bins with count < min_count → NaN."""
    centers = (bins[:-1] + bins[1:]) / 2.0
    n_bins = len(centers)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        nan = np.full(n_bins, np.nan)
        return centers, nan, nan, np.zeros(n_bins, int)
    idx = np.digitize(x, bins, right=False) - 1
    valid = (idx >= 0) & (idx < n_bins)
    idx, y2 = idx[valid], y[valid]
    counts = np.bincount(idx, minlength=n_bins).astype(int)
    sumy = np.bincount(idx, weights=y2, minlength=n_bins)
    sumy2 = np.bincount(idx, weights=y2 * y2, minlength=n_bins)
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)
    g = counts > 0
    means[g] = sumy[g] / counts[g]
    g2 = counts >= 2
    if g2.any():
        var = (sumy2[g2] - sumy[g2] ** 2 / counts[g2]) / (counts[g2] - 1)
        sems[g2] = np.sqrt(np.maximum(var, 0)) / np.sqrt(counts[g2])
    bad = counts < min_count
    means[bad] = np.nan
    sems[bad] = np.nan
    return centers, means, sems, counts


def _domain_mean(centers: np.ndarray, means: np.ndarray, lo: float, hi: float) -> float:
    mask = (centers >= lo) & (centers <= hi) & np.isfinite(centers) & np.isfinite(means)
    return float(np.mean(means[mask])) if mask.any() else np.nan


def _compute_deltas(
    centers: np.ndarray, means: np.ndarray
) -> tuple[float, float]:
    anti = _domain_mean(centers, means, -1.0, -0.5)
    none_ = _domain_mean(centers, means, -0.25, 0.25)
    like = _domain_mean(centers, means, 0.5, 1.0)
    if not (np.isfinite(none_) and none_ != 0):
        return np.nan, np.nan
    d_anti = (none_ - anti) / none_ if np.isfinite(anti) else np.nan
    d_like = (like - none_) / none_ if np.isfinite(like) else np.nan
    return float(d_anti), float(d_like)


# ---------------------------------------------------------------------------
# Data loading / caching
# ---------------------------------------------------------------------------

def _load_data(bases: list[str], network_type: str) -> pd.DataFrame:
    parts = []
    for bd in bases:
        df = process_network_data((bd, network_type))
        parts.append(df[["source_type", "target_type", "Response Correlation", "syn_weight"]])
    return pd.concat(parts, ignore_index=True)


def _build_cache(
    bases: list[str],
    network_type: str,
    types: list[str],
    bins: np.ndarray,
    min_count: int,
) -> dict:
    t0 = time.time()
    df = _load_data(bases, network_type)
    df = df[df["source_type"].isin(types) & df["target_type"].isin(types)].copy()
    x_min, x_max = bins[0], bins[-1]

    pairs: dict = {}
    for s in types:
        for t in types:
            sub = df[(df["source_type"] == s) & (df["target_type"] == t)].dropna(
                subset=["Response Correlation", "syn_weight"]
            )
            x = sub["Response Correlation"].to_numpy()
            y = sub["syn_weight"].to_numpy()
            mask = (x >= x_min) & (x <= x_max) & np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if x.size < 3:
                pairs[(s, t)] = None
                continue
            centers, means, sems, counts = _binstats(x, y, bins, min_count=min_count)
            d_anti, d_like = _compute_deltas(centers, means)
            pairs[(s, t)] = {
                "N": int(x.size),
                "centers": centers,
                "means": means,
                "sems": sems,
                "counts": counts,
                "d_anti": d_anti,
                "d_like": d_like,
            }
    print(f"  Data loaded and binned in {time.time()-t0:.1f}s")
    return {"types": types, "bins": bins, "pairs": pairs}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

BAR_COLOR = "#f4b6c2"
SEG_COLOR = "#4C78A8"


def _render_hist_panel(
    ax: plt.Axes,
    entry: dict | None,
    *,
    x_min: float,
    x_max: float,
    bins: np.ndarray,
    show_xlabel: bool,
    show_ylabel: str | None,
    title: str | None,
    tiny_font: int = 5,
    label_font: int = 6,
) -> None:
    bin_size = float(bins[1] - bins[0])
    from matplotlib.ticker import MaxNLocator
    ax.set_xlim(x_min, x_max)
    # Minimal x-ticks to avoid clutter in a 19×19 grid
    ax.set_xticks([-1, 0, 1])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, prune="both"))
    ax.tick_params(axis="both", labelsize=tiny_font, length=2, pad=1)

    if entry is None or entry["means"] is None or not np.any(np.isfinite(entry["means"])):
        ax.text(0.5, 0.5, "—", ha="center", va="center",
                fontsize=tiny_font, transform=ax.transAxes, color="#aaaaaa")
        trim_spines(ax)
        if show_xlabel:
            ax.set_xlabel("Corr.", fontsize=label_font, labelpad=1)
        if show_ylabel:
            ax.set_ylabel(show_ylabel, fontsize=label_font, labelpad=2)
        if title:
            ax.set_title(title, fontsize=label_font, pad=1)
        return

    centers = entry["centers"]
    means = entry["means"]
    sems = entry["sems"]
    finite = np.isfinite(means)

    ax.bar(centers[finite], means[finite], width=bin_size,
           color=BAR_COLOR, edgecolor="none", alpha=0.85)
    ax.errorbar(centers[finite], means[finite], yerr=sems[finite],
                fmt="none", ecolor="#555555", elinewidth=0.6, capsize=1.5)

    # y-limits anchored at 0
    top = np.nanmax(means + np.where(np.isfinite(sems), sems, 0)) if finite.any() else 0.05
    bot = np.nanmin(means - np.where(np.isfinite(sems), sems, 0)) if finite.any() else -0.05
    span = max(top - bot, 1e-6)
    pad = 0.08 * span
    y_lo = 0.0 if bot >= 0 else bot - pad
    y_hi = 0.0 if top <= 0 else top + pad
    ax.set_ylim(y_lo, y_hi)

    # Vertical and horizontal reference lines
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.5, zorder=0)
    if y_lo <= 0 <= y_hi:
        ax.axhline(0.0, color="#bbbbbb", linewidth=0.5, zorder=0)

    # Domain-mean segments (anti / none / like)
    d_anti, d_like = entry["d_anti"], entry["d_like"]
    anti_m = _domain_mean(centers, means, -1.0, -0.5)
    none_m = _domain_mean(centers, means, -0.25, 0.25)
    like_m = _domain_mean(centers, means, 0.5, 1.0)
    for xlo, xhi, ym in [(-1.0, -0.5, anti_m), (-0.25, 0.25, none_m), (0.5, 1.0, like_m)]:
        if np.isfinite(ym) and y_lo <= ym <= y_hi:
            ax.hlines(ym, max(xlo, x_min), min(xhi, x_max),
                      colors=SEG_COLOR, linewidth=1.6, alpha=0.4, zorder=1)
    pts = [(xm, ym) for xm, ym in [(-0.75, anti_m), (0.0, none_m), (0.75, like_m)]
           if np.isfinite(ym)]
    if len(pts) >= 2:
        xs_p, ys_p = zip(*pts)
        ax.plot(xs_p, ys_p, color=SEG_COLOR, linewidth=0.8, alpha=0.4, zorder=1)

    # Δ annotation (very small, top-left)
    lines = []
    if np.isfinite(d_anti):
        lines.append(f"Δa={d_anti:.2f}")
    if np.isfinite(d_like):
        lines.append(f"Δl={d_like:.2f}")
    if lines:
        ax.text(0.04, 0.97, "\n".join(lines), transform=ax.transAxes,
                fontsize=tiny_font - 0.5, va="top", color="#222222", linespacing=1.1)

    trim_spines(ax)
    if show_xlabel:
        ax.set_xlabel("Corr.", fontsize=label_font, labelpad=1)
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel(show_ylabel, fontsize=label_font, labelpad=2)
    if title:
        ax.set_title(title, fontsize=label_font, pad=1)


# ---------------------------------------------------------------------------
# Figure 1: 19×19 histogram matrix
# ---------------------------------------------------------------------------

def plot_matrix(cache: dict, out_pdf: str) -> None:
    apply_pub_style()
    _setup_font()

    types = cache["types"]
    bins = cache["bins"]
    pairs = cache["pairs"]
    n = len(types)
    labels = abbrev_cell_types(types)

    # Panel size: ~0.52" each → ~10" for 19 types; add margins
    panel_in = 0.52
    fig_w = n * panel_in + 1.2
    fig_h = n * panel_in + 1.0
    fig, axes = plt.subplots(n, n, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)

    # Explicit margins so Source/Target labels can be placed precisely
    left_margin = 0.07
    right_margin = 0.99
    bottom_margin = 0.06
    top_margin = 0.96
    fig.subplots_adjust(
        left=left_margin, right=right_margin,
        bottom=bottom_margin, top=top_margin,
        hspace=0.08, wspace=0.22,
    )

    x_min, x_max = float(bins[0]), float(bins[-1])

    for i, s in enumerate(types):
        for j, t in enumerate(types):
            ax = axes[i, j]
            entry = pairs.get((s, t))
            _render_hist_panel(
                ax, entry,
                x_min=x_min, x_max=x_max, bins=bins,
                show_xlabel=(i == n - 1),
                show_ylabel=labels[i] if j == 0 else None,
                title=labels[j] if i == 0 else None,
            )

    # Source/Target labels anchored to the subplot area edges
    mid_x = (left_margin + right_margin) / 2
    mid_y = (bottom_margin + top_margin) / 2
    fig.text(mid_x, top_margin + 0.01, "Target", ha="center", va="bottom",
             fontsize=7, fontweight="bold")
    fig.text(left_margin - 0.05, mid_y, "Source", va="center", ha="right",
             rotation=90, fontsize=7, fontweight="bold")

    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved matrix: {out_pdf}")


# ---------------------------------------------------------------------------
# Figure 2: Δlike + Δanti heatmaps (side-by-side, one page)
# ---------------------------------------------------------------------------

def plot_heatmaps(cache: dict, out_pdf: str) -> None:
    apply_pub_style()
    _setup_font()

    types = cache["types"]
    pairs = cache["pairs"]
    n = len(types)
    labels = abbrev_cell_types(types)

    anti_mat = np.full((n, n), np.nan)
    like_mat = np.full((n, n), np.nan)
    for i, s in enumerate(types):
        for j, t in enumerate(types):
            e = pairs.get((s, t))
            if e is not None:
                anti_mat[i, j] = e["d_anti"]
                like_mat[i, j] = e["d_like"]

    # Shared color scale
    all_vals = np.concatenate([anti_mat[np.isfinite(anti_mat)],
                                like_mat[np.isfinite(like_mat)]])
    vmax = float(np.nanmax(np.abs(all_vals))) if all_vals.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#e8e8e8")

    # Side-by-side: two square heatmaps on one page (~1/3 of original cell size)
    cell_in = 0.13          # inches per cell
    hm_size = n * cell_in   # ~2.5" for 19 types
    cbar_w = 0.12
    fig_w = 2 * hm_size + cbar_w + 1.4
    fig_h = hm_size + 0.9

    fig = plt.figure(figsize=(fig_w, fig_h))

    from matplotlib.gridspec import GridSpec
    # 3 columns: Δlike | Δanti | colorbar
    gs = GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[hm_size, hm_size, cbar_w * 0.35],
        wspace=0.08,
        left=0.14, right=0.97,
        top=0.88, bottom=0.18,
    )

    tick_fs = 7

    def _draw_hm(ax, mat, title, show_ylabel=True):
        im = ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap=cmap,
                       interpolation="nearest", aspect="equal")
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=tick_fs, ha="center")
        if show_ylabel:
            ax.set_yticklabels(labels, fontsize=tick_fs)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Target", fontsize=8, labelpad=3)
        if show_ylabel:
            ax.set_ylabel("Source", fontsize=8, labelpad=3)
        ax.set_title(title, fontsize=9, pad=4)
        # NA diagonal markers
        na = ~np.isfinite(mat)
        for ii, jj in zip(*np.where(na)):
            ax.plot([jj - 0.5, jj + 0.5], [ii + 0.5, ii - 0.5],
                    color="#999999", linewidth=0.4, alpha=0.7)
        # Grid lines between cells
        for k in range(n + 1):
            ax.axhline(k - 0.5, color="white", linewidth=0.3, zorder=2)
            ax.axvline(k - 0.5, color="white", linewidth=0.3, zorder=2)
        trim_spines(ax)
        return im

    ax_like = fig.add_subplot(gs[0, 0])
    _draw_hm(ax_like, like_mat, "Δlike  (corr-based)", show_ylabel=True)

    ax_anti = fig.add_subplot(gs[0, 1])
    im_anti = _draw_hm(ax_anti, anti_mat, "Δanti  (corr-based)", show_ylabel=False)

    ax_cbar = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im_anti, cax=ax_cbar)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Δ (norm.)", fontsize=7, labelpad=2)

    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmaps: {out_pdf}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extended data Fig.5 – Corr like-to-like full 19-type matrix + heatmaps."
    )
    ap.add_argument("--bases", nargs="*", default=None)
    ap.add_argument("--network-type", default="bio_trained",
                    choices=["bio_trained", "naive"])
    ap.add_argument("--bin-size", type=float, default=0.1)
    ap.add_argument("--min-bin-count", type=int, default=5)
    ap.add_argument("--out-dir", default="figures/paper/extended_data/fig5_corr_full19",
                    help="Output directory for PDFs")
    ap.add_argument("--cache-dir", default="figures/paper/extended_data/fig5_corr_full19/cache")
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument("--matrix-only", action="store_true")
    ap.add_argument("--heatmap-only", action="store_true")
    args = ap.parse_args()

    bases = args.bases or _detect_bases()
    if not bases:
        raise SystemExit("No core_nll_* directories found.")

    nt = args.network_type
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, f"corr_full19_{nt}.pkl")

    bins = np.arange(-1.0, 1.0 + args.bin_size, args.bin_size)

    cache = None
    if not args.force_recompute and os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"Loaded cache: {cache_path}")
        except Exception:
            cache = None

    if cache is None:
        print("Building cache …")
        cache = _build_cache(bases, nt, ALL_19, bins, args.min_bin_count)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"Cache saved: {cache_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.heatmap_only:
        plot_matrix(cache, os.path.join(args.out_dir, f"corr_full19_matrix_{nt}.pdf"))

    if not args.matrix_only:
        plot_heatmaps(cache, os.path.join(args.out_dir, f"corr_full19_heatmaps_{nt}.pdf"))


if __name__ == "__main__":
    main()
