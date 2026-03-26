#!/usr/bin/env python3
"""
Extended data for Fig. 5 – Preferred-direction like-to-like analysis.
Generates two full-page publication figures:
  1. 19×19 histogram matrix  (weight vs. ΔPD, per cell-type pair)
  2. Companion heatmaps      (cosine-fit amplitude a and b, side-by-side on one page)

Usage (from repo root, conda env new_v1):
  python figure_scripts/extended_data/fig5_pd_full19_matrix.py
  python figure_scripts/extended_data/fig5_pd_full19_matrix.py --heatmap-only
  python figure_scripts/extended_data/fig5_pd_full19_matrix.py --matrix-only
  python figure_scripts/extended_data/fig5_pd_full19_matrix.py --network-type naive
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
from analysis_shared.io import load_edges_with_pref_dir
from analysis_shared.stats import bin_mean_sem, fit_cosine_series_deg
from analysis_shared.style import apply_pub_style, trim_spines
from aggregate_correlation_plot import process_network_data

# ---------------------------------------------------------------------------
# Cell-type ordering (full 19 types)
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


# ---------------------------------------------------------------------------
# Data loading / caching
# ---------------------------------------------------------------------------

def _load_pd_data(bases: list[str], network_type: str) -> pd.DataFrame:
    parts = []
    for bd in bases:
        e = load_edges_with_pref_dir(bd, network_type)
        try:
            typed = process_network_data((bd, network_type))
            typed = typed[["source_id", "target_id", "source_type", "target_type"]]
            e = e.merge(typed, on=["source_id", "target_id"], how="left")
        except Exception:
            pass
        parts.append(e)
    return pd.concat(parts, ignore_index=True)


def _build_cache(
    bases: list[str],
    network_type: str,
    types: list[str],
    bin_step: float,
) -> dict:
    t0 = time.time()
    df = _load_pd_data(bases, network_type)
    df = df.dropna(subset=["source_type", "target_type"])
    df = df[df["source_type"].isin(types) & df["target_type"].isin(types)].copy()

    x_min, x_max = 0.0, 180.0
    bins = np.arange(x_min, x_max + bin_step, bin_step)

    pairs: dict = {}
    for s in types:
        for t in types:
            sub = df[(df["source_type"] == s) & (df["target_type"] == t)][
                ["pref_dir_diff_deg", "syn_weight"]
            ].dropna()
            if sub.empty or len(sub) < 2:
                pairs[(s, t)] = None
                continue
            xx = sub["pref_dir_diff_deg"].to_numpy()
            yy = sub["syn_weight"].to_numpy()
            centers, means, sems = bin_mean_sem(xx, yy, bins)
            fit = fit_cosine_series_deg(xx, yy)
            pairs[(s, t)] = {
                "N": int(len(xx)),
                "centers": centers,
                "means": means,
                "sems": sems,
                "fit": {
                    "a": fit.a, "b": fit.b, "c": fit.c,
                    "p_a": fit.p_a, "p_b": fit.p_b,
                },
            }

    print(f"  PD data loaded and binned in {time.time()-t0:.1f}s")
    return {
        "types": types,
        "bin_step": bin_step,
        "x_min": x_min,
        "x_max": x_max,
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

BAR_COLOR = "#aec7e8"


def _render_pd_panel(
    ax: plt.Axes,
    entry: dict | None,
    *,
    x_min: float,
    x_max: float,
    bin_step: float,
    show_xlabel: bool,
    show_ylabel: str | None,
    title: str | None,
    tiny_font: int = 5,
    label_font: int = 6,
) -> None:
    from matplotlib.ticker import MaxNLocator
    ax.set_xlim(x_min, x_max)
    ax.set_xticks([0, 90, 180])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2, prune="both"))
    ax.tick_params(axis="both", labelsize=tiny_font, length=2, pad=1)

    if entry is None or not np.any(np.isfinite(entry["means"])):
        ax.text(0.5, 0.5, "—", ha="center", va="center",
                fontsize=tiny_font, transform=ax.transAxes, color="#aaaaaa")
        trim_spines(ax)
        if show_xlabel:
            ax.set_xlabel("ΔPD (°)", fontsize=label_font, labelpad=1)
        if show_ylabel:
            ax.set_ylabel(show_ylabel, fontsize=label_font, labelpad=2)
        if title:
            ax.set_title(title, fontsize=label_font, pad=1)
        return

    centers = entry["centers"]
    means = entry["means"]
    sems = entry["sems"]
    fit = entry["fit"]

    ax.bar(centers, means, width=bin_step, color=BAR_COLOR, edgecolor="none")
    ax.errorbar(centers, means, yerr=sems,
                fmt="none", ecolor="#555555", elinewidth=0.6, capsize=1.5)

    # Cosine-series fit overlay
    xs = np.linspace(x_min, x_max, 361)
    ys = (fit["a"] * np.cos(np.radians(xs))
          + fit["b"] * np.cos(2 * np.radians(xs))
          + fit["c"])
    ax.plot(xs, ys, color="crimson", linewidth=0.9)

    # Annotation: a and p_a only (space is tight)
    p_a = fit["p_a"]
    sig = ("***" if p_a < 1e-3 else "**" if p_a < 1e-2 else "*" if p_a < 0.05 else "")
    ax.text(0.04, 0.97, f"a={fit['a']:.2f}{sig}",
            transform=ax.transAxes, fontsize=tiny_font - 0.5, va="top", color="#222222")

    trim_spines(ax)
    if show_xlabel:
        ax.set_xlabel("ΔPD (°)", fontsize=label_font, labelpad=1)
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
    pairs = cache["pairs"]
    bin_step = float(cache["bin_step"])
    x_min = float(cache["x_min"])
    x_max = float(cache["x_max"])
    n = len(types)
    labels = abbrev_cell_types(types)

    panel_in = 0.52
    fig_w = n * panel_in + 1.2
    fig_h = n * panel_in + 1.0
    fig, axes = plt.subplots(n, n, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)

    left_margin = 0.07
    right_margin = 0.99
    bottom_margin = 0.06
    top_margin = 0.96
    fig.subplots_adjust(
        left=left_margin, right=right_margin,
        bottom=bottom_margin, top=top_margin,
        hspace=0.08, wspace=0.22,
    )

    for i, s in enumerate(types):
        for j, t in enumerate(types):
            ax = axes[i, j]
            entry = pairs.get((s, t))
            _render_pd_panel(
                ax, entry,
                x_min=x_min, x_max=x_max, bin_step=bin_step,
                show_xlabel=(i == n - 1),
                show_ylabel=labels[i] if j == 0 else None,
                title=labels[j] if i == 0 else None,
            )

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
# Figure 2: Cosine-fit amplitude heatmaps (a and b, side-by-side)
# ---------------------------------------------------------------------------

def plot_heatmaps(cache: dict, out_pdf: str) -> None:
    apply_pub_style()
    _setup_font()

    types = cache["types"]
    pairs = cache["pairs"]
    n = len(types)
    labels = abbrev_cell_types(types)

    a_mat = np.full((n, n), np.nan)
    b_mat = np.full((n, n), np.nan)
    for i, s in enumerate(types):
        for j, t in enumerate(types):
            e = pairs.get((s, t))
            if e is not None:
                c = e["fit"]["c"]
                if np.isfinite(c) and c != 0:
                    a_mat[i, j] = e["fit"]["a"] / c
                    b_mat[i, j] = e["fit"]["b"] / c

    # Shared symmetric color scale
    all_vals = np.concatenate([a_mat[np.isfinite(a_mat)], b_mat[np.isfinite(b_mat)]])
    vmax = float(np.nanmax(np.abs(all_vals))) if all_vals.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#e8e8e8")

    cell_in = 0.13
    hm_size = n * cell_in
    cbar_w = 0.12
    fig_w = 2 * hm_size + cbar_w + 1.4
    fig_h = hm_size + 0.9

    fig = plt.figure(figsize=(fig_w, fig_h))

    from matplotlib.gridspec import GridSpec
    # 3 columns: a | b | colorbar
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
        na = ~np.isfinite(mat)
        for ii, jj in zip(*np.where(na)):
            ax.plot([jj - 0.5, jj + 0.5], [ii + 0.5, ii - 0.5],
                    color="#999999", linewidth=0.4, alpha=0.7)
        for k in range(n + 1):
            ax.axhline(k - 0.5, color="white", linewidth=0.3, zorder=2)
            ax.axvline(k - 0.5, color="white", linewidth=0.3, zorder=2)
        trim_spines(ax)
        return im

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_hm(ax_a, a_mat, "a/c  (PD-based)", show_ylabel=True)

    ax_b = fig.add_subplot(gs[0, 1])
    im_b = _draw_hm(ax_b, b_mat, "b/c  (PD-based)", show_ylabel=False)

    ax_cbar = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im_b, cax=ax_cbar)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("norm. amplitude", fontsize=7, labelpad=2)

    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmaps: {out_pdf}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extended data Fig.5 – PD like-to-like full 19-type matrix + heatmaps."
    )
    ap.add_argument("--bases", nargs="*", default=None)
    ap.add_argument("--network-type", default="bio_trained",
                    choices=["bio_trained", "naive"])
    ap.add_argument("--bin-step", type=float, default=20.0)
    ap.add_argument("--out-dir", default="figures/paper/extended_data/fig5_pd_full19",
                    help="Output directory for PDFs")
    ap.add_argument("--cache-dir", default="figures/paper/extended_data/fig5_pd_full19/cache")
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument("--matrix-only", action="store_true")
    ap.add_argument("--heatmap-only", action="store_true")
    args = ap.parse_args()

    bases = args.bases or _detect_bases()
    if not bases:
        raise SystemExit("No core_nll_* directories found.")

    nt = args.network_type
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, f"pd_full19_{nt}.pkl")

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
        cache = _build_cache(bases, nt, ALL_19, args.bin_step)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"Cache saved: {cache_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.heatmap_only:
        plot_matrix(cache, os.path.join(args.out_dir, f"pd_full19_matrix_{nt}.pdf"))

    if not args.matrix_only:
        plot_heatmaps(cache, os.path.join(args.out_dir, f"pd_full19_heatmaps_{nt}.pdf"))


if __name__ == "__main__":
    main()
