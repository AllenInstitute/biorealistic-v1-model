#!/usr/bin/env python3
"""
Generate a simplified response-correlation panel for Fig.5:
 - 2x3: selected example pairs (user-specified)
 - Right column: two heatmaps (inh-aggregated) for Δlike and Δanti

Uses cached data from the existing corr_final workflow when available to avoid
re-computation.
"""
from __future__ import annotations

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis_shared.style import apply_pub_style, trim_spines
from analysis_shared.celltype_labels import abbrev_cell_type, abbrev_cell_types
from analysis_shared.grouping import apply_inh_simplification
from aggregate_correlation_plot import process_network_data


def _domain_mean_from_bin_means(
    centers: np.ndarray, means: np.ndarray, lo: float, hi: float
) -> float:
    centers = np.asarray(centers)
    means = np.asarray(means)
    mask = (centers >= lo) & (centers <= hi) & np.isfinite(centers) & np.isfinite(means)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(means[mask]))


def _binstats_min_count(
    x: np.ndarray, y: np.ndarray, bins: np.ndarray, *, min_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast binning via bincount: returns (centers, means, sems, counts)."""
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

    # Bin assignment: idx in [0, n_bins-1]
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

    # Sample SEM: std(ddof=1)/sqrt(n). Only defined for n>=2.
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
    n_fmt_millions: bool = True,
    delta_top_left: bool = False,
) -> None:
    """Bar + range-average summaries (anti/none/like) with Δ annotations (no fit line)."""
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

    # Two-step averaging: mean over bins, then mean over bin-means within each correlation range
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
        if delta_top_left:
            ax.text(
                0.03,
                0.95,
                "\n".join(text_lines),
                transform=ax.transAxes,
                fontsize=7,
                va="top",
            )
        else:
            ax.text(
                0.03,
                0.08 if overall_mean >= 0 else 0.92,
                "\n".join(text_lines),
                transform=ax.transAxes,
                fontsize=7,
                va="bottom" if overall_mean >= 0 else "top",
            )

    # Correlation-domain markers (anti / none / like): horizontal segments at the domain mean weight
    seg_color = "#4C78A8"
    domain_segments = [
        ("anti", -1.0, -0.5, anti_mean),
        ("none", -0.25, 0.25, none_mean),
        ("like", 0.5, 1.0, like_mean),
    ]
    for _, xmin, xmax, yseg in domain_segments:
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

    # Connect range averages with a simple polyline (anti → none → like)
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
        if n_fmt_millions:
            millions = n_conn / 1e6
            n_str = f"N={millions:.2f}M"
        else:
            n_str = f"N={n_conn:,}"
        ax.text(
            0.97,
            0.08 if overall_mean >= 0 else 0.92,
            n_str,
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            va="bottom" if overall_mean >= 0 else "top",
        )
    ax.set_xlim(x_min, x_max)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelsize=7, pad=1)
    ax.tick_params(axis="y", labelsize=7, pad=1)
    if title is not None:
        ax.set_title(title, fontsize=8, pad=1.5)

    # Visual anchors (match newer Fig5-style): x=0 and y=0
    if x_min <= 0 <= x_max:
        ax.axvline(0.0, color="#999999", linewidth=0.7, alpha=0.6, zorder=0)
    y0, y1 = ax.get_ylim()
    if y0 <= 0 <= y1:
        ax.axhline(0.0, color="#aaaaaa", linewidth=0.7, alpha=0.7, zorder=0)
    trim_spines(ax)


def _default_heatmap_paths(network_type: str) -> tuple[str, str]:
    # These are the currently-used Fig5-style inh-aggregated delta heatmaps.
    # We keep them as defaults but allow overriding via CLI flags.
    base = "aggregated_response_correlation"
    like = os.path.join(
        base,
        f"{network_type}_withinlayer_deltas_heatmap_inhAgg_noNP_binmeanrangeavg_min5_antiFlipped_delta_like_heatmap.png",
    )
    anti = os.path.join(
        base,
        f"{network_type}_withinlayer_deltas_heatmap_inhAgg_noNP_binmeanrangeavg_min5_antiFlipped_delta_anti_heatmap.png",
    )
    return like, anti


def _crop_whitespace(img: np.ndarray, *, pad: int = 2) -> np.ndarray:
    """Crop near-white borders to make embedded heatmaps larger."""
    arr = np.asarray(img)
    if arr.ndim < 2:
        return arr

    # Handle grayscale, RGB, RGBA
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
        alpha = None
    else:
        rgb = arr[..., :3]
        alpha = arr[..., 3] if (arr.shape[-1] >= 4) else None

    # "Non-white" heuristic: any RGB channel below 0.97 (for float images)
    rgbf = rgb.astype(float)
    if rgbf.max() > 1.0:
        rgbf = rgbf / 255.0
    nonwhite = np.any(rgbf < 0.97, axis=-1)
    if alpha is not None:
        af = alpha.astype(float)
        if af.max() > 1.0:
            af = af / 255.0
        nonwhite = nonwhite & (af > 0.05)

    ys, xs = np.where(nonwhite)
    if ys.size == 0 or xs.size == 0:
        return arr

    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, arr.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, arr.shape[1])
    return arr[y0:y1, x0:x1]


def _draw_heatmap_image(
    ax: plt.Axes,
    img_path: str,
    title: str,
    *,
    crop_pad: int,
    aspect: str,
    title_fontsize: int,
    title_pad: float,
) -> None:
    if not os.path.isfile(img_path):
        ax.text(
            0.5,
            0.5,
            f"Missing heatmap:\n{img_path}",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
        )
        ax.axis("off")
        return
    img = mpimg.imread(img_path)
    img = _crop_whitespace(img, pad=int(crop_pad))
    ax.imshow(img, interpolation="nearest", aspect=aspect)
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    ax.axis("off")


def _overlay_source_target_labels(ax: plt.Axes) -> None:
    ax.text(
        -0.08,
        1.02,
        "Source",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        -0.12,
        0.95,
        "Target",
        transform=ax.transAxes,
        ha="right",
        va="top",
        rotation=90,
        fontsize=7,
        fontweight="bold",
        clip_on=False,
    )


def _compute_delta_from_bin_means(
    centers: np.ndarray, means: np.ndarray
) -> tuple[float, float]:
    anti_mean = _domain_mean_from_bin_means(centers, means, -1.0, -0.5)
    none_mean = _domain_mean_from_bin_means(centers, means, -0.25, 0.25)
    like_mean = _domain_mean_from_bin_means(centers, means, 0.5, 1.0)
    if not np.isfinite(none_mean) or none_mean == 0:
        return np.nan, np.nan
    delta_anti = (
        (none_mean - anti_mean) / none_mean if np.isfinite(anti_mean) else np.nan
    )
    delta_like = (
        (like_mean - none_mean) / none_mean if np.isfinite(like_mean) else np.nan
    )
    return float(delta_anti), float(delta_like)


def _plot_delta_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    labels: list[str],
    title: str,
    *,
    vmin: float,
    vmax: float,
    tick_fontsize: int,
    show_axis_labels: bool,
    is_bottom: bool,
) -> None:
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")
    im = ax.imshow(
        mat, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest", aspect="equal"
    )
    # With aspect='equal' and an attached colorbar, matplotlib will shrink the axes
    # and center it inside the GridSpec slot, which visually looks like a "gap"
    # between the histogram columns and the heatmap column. Anchor left to remove it.
    ax.set_anchor("W")
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=tick_fontsize)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)
    # Place "Source"/"Target" labels without changing layout: anchored to the first
    # tick label (L2/3_Exc) rather than as axis labels.
    if show_axis_labels:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        # "Source" above the first x tick (L2/3_Exc)
        ax.text(
            -1.5,
            1.00,
            "Source",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            clip_on=False,
        )
        # "Target" vertically to the left of the first y tick (L2/3_Exc)
        ax.text(
            -0.05,
            ymin + 1,
            "Target",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="top",
            rotation=90,
            fontsize=7,
            fontweight="bold",
            clip_on=False,
        )
    ax.set_title(title, fontsize=9, pad=2)
    trim_spines(ax)

    # NA markers: thin diagonal line in each NaN cell (match original Fig5-style heatmaps)
    na = ~np.isfinite(mat)
    for i, j in zip(*np.where(na)):
        ax.plot(
            [j - 0.5, j + 0.5],
            [i + 0.5, i - 0.5],
            color="#666666",
            linewidth=0.5,
            alpha=0.8,
        )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser(
        description="Generate simplified correlation panel for Fig.5"
    )
    ap.add_argument("--bases", nargs="*", default=None)
    ap.add_argument("--network-type", default="bio_trained")
    ap.add_argument("--bin-size", type=float, default=0.1)
    ap.add_argument(
        "--out", default="figures/paper/figure5/corr_final_panel_examples.png"
    )
    ap.add_argument(
        "--min-bin-count",
        type=int,
        default=5,
        help="Minimum datapoints required per x-bin to show the bin and to include it in range averages.",
    )
    ap.add_argument("--cache-dir", default="figures/paper/cache_corr_final")
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument(
        "--heatmap-delta-like",
        default=None,
        help="Path to the Δlike inh-aggregated heatmap PNG (optional).",
    )
    ap.add_argument(
        "--heatmap-delta-anti",
        default=None,
        help="Path to the Δanti inh-aggregated heatmap PNG (optional).",
    )
    ap.add_argument(
        "--also-pdf",
        action="store_true",
        help="Also save a PDF next to the output image.",
    )
    # Layout knobs (exposed for manual tuning)
    ap.add_argument("--fig-w", type=float, default=6, help="Figure width (inches).")
    ap.add_argument("--fig-h", type=float, default=3, help="Figure height (inches).")
    ap.add_argument(
        "--wr",
        type=float,
        nargs=4,
        default=[1.0, 1.0, 1.0, 1.2],
        help="GridSpec width ratios for 4 columns (e.g., 1 1 1 2.4).",
    )
    ap.add_argument("--wspace", type=float, default=0.25, help="GridSpec wspace.")
    ap.add_argument("--hspace", type=float, default=0.65, help="GridSpec hspace.")
    ap.add_argument("--left", type=float, default=0.06, help="Figure left margin.")
    ap.add_argument("--right", type=float, default=0.98, help="Figure right margin.")
    ap.add_argument("--top", type=float, default=0.93, help="Figure top margin.")
    ap.add_argument("--bottom", type=float, default=0.12, help="Figure bottom margin.")
    # Heatmap embedding knobs
    ap.add_argument(
        "--heatmap-crop-pad",
        type=int,
        default=2,
        help="Crop padding (pixels) when trimming whitespace around embedded heatmaps.",
    )
    ap.add_argument(
        "--heatmap-aspect",
        choices=["equal", "auto"],
        default="equal",
        help="Aspect mode for embedded heatmap images.",
    )
    ap.add_argument(
        "--heatmap-title-fontsize",
        type=int,
        default=8,
        help="Fontsize for embedded heatmap titles.",
    )
    ap.add_argument(
        "--heatmap-title-pad",
        type=float,
        default=2.0,
        help="Title pad for embedded heatmap titles.",
    )
    ap.add_argument(
        "--heatmap-mode",
        choices=["compute", "embed"],
        default="compute",
        help="Heatmap rendering mode: compute from data (abbrev labels) or embed existing PNGs.",
    )
    ap.add_argument(
        "--heatmap-vmax",
        type=float,
        default=None,
        help="Optional fixed |vmax| for computed heatmaps. If omitted, uses max abs across both mats.",
    )
    ap.add_argument(
        "--heatmap-label-style",
        choices=["full", "abbrev"],
        default="full",
        help="Tick-label style for computed heatmaps.",
    )
    ap.add_argument(
        "--heatmap-tick-fontsize",
        type=int,
        default=6,
        help="Tick-label fontsize for computed heatmaps.",
    )
    ap.add_argument(
        "--heatmap-axis-labels",
        action="store_true",
        default=True,
        help="Show 'Source'/'Target' axis labels on computed heatmaps.",
    )
    args = ap.parse_args()

    apply_pub_style()
    # Try Arial if available
    try:
        import matplotlib as mpl
        from matplotlib import font_manager as fm

        _ = fm.findfont("Arial", fallback_to_default=False)
        mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"]})
    except Exception:
        pass

    bases = (
        args.bases
        if args.bases
        else [f"core_nll_{i}" for i in range(10) if os.path.isdir(f"core_nll_{i}")]
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    xticks = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # Helper to pull pair entry
    def get_entry(cache_dict, pair):
        entry = cache_dict["pairs"].get(pair)
        if not entry:
            raise ValueError(f"Missing pair {pair}")
        return entry

    # 2x3: user-specified example pairs (omit coarse EI panels)
    example_pairs = [
        ("L2/3_Exc", "L6_Exc"),
        ("L2/3_Exc", "L4_Exc"),
        # Use layer-specific inhibitory labels (old notation) for L5-targeted examples.
        ("L5_PV", "L5_IT"),
        ("L5_PV", "L5_ET"),
        # For I->I examples, use layer-agnostic inhibitory families (PV/SST/VIP),
        # consistent with the inh-aggregated heatmaps.
        ("SST", "VIP"),
        ("VIP", "PV"),
    ]

    # Compute per-bin stats with min-bin-count (min5) for ONLY these pairs.
    # This keeps the panel consistent with newer Fig5-style analysis (range averages, not fit),
    # and avoids showing noisy/rare bins.
    sel_pkl = os.path.join(
        args.cache_dir,
        f"corr_selected_pairs_v3_{args.network_type}_inhAgnosticII_min{args.min_bin_count}.pkl",
    )
    if (not args.force_recompute) and os.path.isfile(sel_pkl):
        with open(sel_pkl, "rb") as f:
            sel_cache = pickle.load(f)
    else:
        parts = []
        for bd in bases:
            df = process_network_data((bd, args.network_type))
            df = df[
                ["source_type", "target_type", "Response Correlation", "syn_weight"]
            ]

            # Build a simplified view for layer-agnostic inhibitory families (PV/SST/VIP).
            df_simpl = apply_inh_simplification(df)

            def _is_layer_agnostic_inh(lbl: str) -> bool:
                return lbl in {"PV", "SST", "VIP", "L1_Inh"}

            for s, t in example_pairs:
                view = (
                    df_simpl
                    if (_is_layer_agnostic_inh(s) or _is_layer_agnostic_inh(t))
                    else df
                )
                sub = view[(view["source_type"] == s) & (view["target_type"] == t)]
                if not sub.empty:
                    parts.append(sub)
        if parts:
            import pandas as pd

            df_sel = pd.concat(parts, ignore_index=True)
        else:
            import pandas as pd

            df_sel = pd.DataFrame(
                columns=[
                    "source_type",
                    "target_type",
                    "Response Correlation",
                    "syn_weight",
                ]
            )

        x_min, x_max = -1.0, 1.0
        bins = np.arange(x_min, x_max + args.bin_size, args.bin_size)
        centers = (bins[:-1] + bins[1:]) / 2.0

        pairs_out = {}
        for s, t in example_pairs:
            sub = df_sel[
                (df_sel["source_type"] == s) & (df_sel["target_type"] == t)
            ].dropna()
            if sub.empty:
                pairs_out[(s, t)] = None
                continue
            x = sub["Response Correlation"].to_numpy()
            y = sub["syn_weight"].to_numpy()
            m = (x >= x_min) & (x <= x_max) & np.isfinite(x) & np.isfinite(y)
            x = x[m]
            y = y[m]
            if x.size < 3:
                pairs_out[(s, t)] = None
                continue
            cts, means, sems, counts = _binstats_min_count(
                x, y, bins, min_count=args.min_bin_count
            )
            pairs_out[(s, t)] = {
                "centers": cts,
                "means": means,
                "sems": sems,
                "counts": counts,
                "N": int(x.size),
            }

        sel_cache = {
            "centers": centers,
            "bin_size": float(args.bin_size),
            "x_min": float(x_min),
            "x_max": float(x_max),
            "min_bin_count": int(args.min_bin_count),
            "pairs": pairs_out,
        }
        with open(sel_pkl, "wb") as f:
            pickle.dump(sel_cache, f)

    fig = plt.figure(figsize=(args.fig_w, args.fig_h))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        width_ratios=list(args.wr),
        wspace=args.wspace,
        hspace=args.hspace,
        left=args.left,
        right=args.right,
        top=args.top,
        bottom=args.bottom,
    )

    for k, (s, t) in enumerate(example_pairs):
        r, c = divmod(k, 3)
        ax = fig.add_subplot(gs[r, c])
        entry = get_entry(sel_cache, (s, t))
        # Histogram titles: use old/full cell-type notation.
        title = f"{s}→{t}"
        delta_top_left = (s == "L2/3_Exc") and (t == "L6_Exc")
        _render_corr_hist_rangeavg(
            ax,
            centers=entry["centers"],
            means=entry["means"],
            sems=entry["sems"],
            n_conn=entry.get("N"),
            title=title,
            xlim=(-1.0, 1.0),
            xticks=xticks,
            delta_top_left=delta_top_left,
        )
        if c == 0:
            ax.set_ylabel("Weight (pA)", fontsize=8)
        if r == 1:
            ax.set_xlabel("Response Corr.", fontsize=8)

    # Right column: inh-aggregated delta heatmaps (prefer computed heatmaps for clean styling)
    if args.heatmap_mode == "embed":
        default_like, default_anti = _default_heatmap_paths(args.network_type)
        like_path = args.heatmap_delta_like or default_like
        anti_path = args.heatmap_delta_anti or default_anti

        ax_like = fig.add_subplot(gs[0, 3])
        _draw_heatmap_image(
            ax_like,
            like_path,
            title="",
            crop_pad=args.heatmap_crop_pad,
            aspect=args.heatmap_aspect,
            title_fontsize=args.heatmap_title_fontsize,
            title_pad=args.heatmap_title_pad,
        )
        if args.heatmap_axis_labels:
            _overlay_source_target_labels(ax_like)
        ax_like.text(
            0.5,
            1.02,
            "Δlike (inh-agg)",
            ha="center",
            va="bottom",
            transform=ax_like.transAxes,
            fontsize=9,
        )

        ax_anti = fig.add_subplot(gs[1, 3])
        _draw_heatmap_image(
            ax_anti,
            anti_path,
            title="",
            crop_pad=args.heatmap_crop_pad,
            aspect=args.heatmap_aspect,
            title_fontsize=args.heatmap_title_fontsize,
            title_pad=args.heatmap_title_pad,
        )
        if args.heatmap_axis_labels:
            _overlay_source_target_labels(ax_anti)
        ax_anti.text(
            0.5,
            1.02,
            "Δanti (inh-agg)",
            ha="center",
            va="bottom",
            transform=ax_anti.transAxes,
            fontsize=9,
        )
    else:
        hm_types = [
            "L2/3_Exc",
            "L4_Exc",
            "L5_IT",
            "L5_ET",
            "L6_Exc",
            "L1_Inh",
            "PV",
            "SST",
            "VIP",
        ]
        hm_labels = (
            hm_types
            if args.heatmap_label_style == "full"
            else abbrev_cell_types(hm_types)
        )
        hm_cache = os.path.join(
            args.cache_dir,
            f"corr_delta_heatmaps_{args.network_type}_inhAgg_noNP_min{args.min_bin_count}_bin{args.bin_size}.pkl",
        )
        if (not args.force_recompute) and os.path.isfile(hm_cache):
            with open(hm_cache, "rb") as f:
                heat = pickle.load(f)
            delta_like = np.asarray(heat["delta_like"], dtype=float)
            delta_anti = np.asarray(heat["delta_anti"], dtype=float)
        else:
            parts = []
            for bd in bases:
                df = process_network_data((bd, args.network_type))
                df = df[
                    (df["source_type"] != "L5_NP") & (df["target_type"] != "L5_NP")
                ].copy()
                df = apply_inh_simplification(df)
                df = df[
                    df["source_type"].isin(hm_types) & df["target_type"].isin(hm_types)
                ]
                parts.append(
                    df[
                        [
                            "source_type",
                            "target_type",
                            "Response Correlation",
                            "syn_weight",
                        ]
                    ]
                )
            import pandas as pd

            df_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            x_min, x_max = -1.0, 1.0
            bins = np.arange(x_min, x_max + args.bin_size, args.bin_size)
            n = len(hm_types)
            delta_like = np.full((n, n), np.nan, dtype=float)
            delta_anti = np.full((n, n), np.nan, dtype=float)
            for i, s in enumerate(hm_types):
                for j, t in enumerate(hm_types):
                    sub = df_all[
                        (df_all["source_type"] == s) & (df_all["target_type"] == t)
                    ].dropna()
                    if sub.empty:
                        continue
                    x = sub["Response Correlation"].to_numpy()
                    y = sub["syn_weight"].to_numpy()
                    m = (x >= x_min) & (x <= x_max) & np.isfinite(x) & np.isfinite(y)
                    x = x[m]
                    y = y[m]
                    if x.size < 3:
                        continue
                    centers, means, _sems, _counts = _binstats_min_count(
                        x, y, bins, min_count=args.min_bin_count
                    )
                    d_anti, d_like = _compute_delta_from_bin_means(centers, means)
                    delta_anti[i, j] = d_anti
                    delta_like[i, j] = d_like
            with open(hm_cache, "wb") as f:
                pickle.dump(
                    {
                        "delta_like": delta_like,
                        "delta_anti": delta_anti,
                        "types": hm_types,
                    },
                    f,
                )

        finite = np.isfinite(delta_like) | np.isfinite(delta_anti)
        if args.heatmap_vmax is not None:
            vmax = float(args.heatmap_vmax)
        elif finite.any():
            vmax = float(
                np.nanmax(
                    np.abs(np.concatenate([delta_like[finite], delta_anti[finite]]))
                )
            )
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0
        else:
            vmax = 1.0

        ax_like = fig.add_subplot(gs[0, 3])
        _plot_delta_heatmap(
            ax_like,
            delta_like,
            hm_labels,
            "Δlike",
            vmin=-vmax,
            vmax=vmax,
            tick_fontsize=args.heatmap_tick_fontsize,
            show_axis_labels=args.heatmap_axis_labels,
            is_bottom=False,
        )
        ax_anti = fig.add_subplot(gs[1, 3])
        _plot_delta_heatmap(
            ax_anti,
            delta_anti,
            hm_labels,
            "Δanti",
            vmin=-vmax,
            vmax=vmax,
            tick_fontsize=args.heatmap_tick_fontsize,
            show_axis_labels=args.heatmap_axis_labels,
            is_bottom=True,
        )

    fig.savefig(args.out, dpi=400, bbox_inches="tight")
    if args.also_pdf:
        base, ext = os.path.splitext(args.out)
        pdf_path = base + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
