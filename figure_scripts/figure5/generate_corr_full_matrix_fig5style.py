#!/usr/bin/env python3
"""
Generate a Fig.5-style correlation matrix with multiple fit options:
- Linear (OLS)
- Legendre polynomial (up to 3rd order)
- Piecewise linear with shared intercept at x=0

Output defaults to aggregated_response_correlation/bio_trained_fig5style.png
using all available core_nll_* directories for the bio-trained network.
"""
from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis_shared.style import apply_pub_style, trim_spines
from analysis_shared.grouping import (
    INH_SIMPLE_MAP,
    aggregate_l5,
    apply_inh_simplification,
)
from analysis_shared.stats import bin_mean_sem
from analysis_shared.corr import EXC_ALL, EXC_L5_AGG
from analysis_shared.celltype_labels import abbrev_cell_type, abbrev_cell_types
from aggregate_correlation_plot import process_network_data


def _detect_default_bases() -> list[str]:
    return [f"core_nll_{i}" for i in range(10) if os.path.isdir(f"core_nll_{i}")]


def _png_to_pdf_path(path: str) -> str:
    """Convert a .png path to a .pdf path (best-effort).

    Used for optional dual outputs (PNG + PDF) without changing existing defaults.
    """
    if not isinstance(path, str):
        path = str(path)
    if path.lower().endswith(".png"):
        return path[:-4] + ".pdf"
    return path + ".pdf"


def _save_png_and_optional_pdf(
    fig: plt.Figure, png_path: str, *, dpi: int, also_pdf: bool
) -> None:
    fig.savefig(png_path, dpi=dpi)
    if also_pdf:
        fig.savefig(_png_to_pdf_path(png_path))


def _sanitize_args_for_consistency(args: argparse.Namespace) -> argparse.Namespace:
    """Coerce incompatible flag combos into a consistent configuration.

    Currently:
    - When `--no-simplify-inh` (full 19-type inhibitory labels) is selected, the
      `--inh-respective-layer` (within-layer Exc↔Inh restriction) is not meaningful.
      We automatically disable it to avoid accidental filtering.
    """
    if getattr(args, "no_simplify_inh", False) and getattr(
        args, "inh_respective_layer", False
    ):
        print(
            "NOTE: Ignoring --inh-respective-layer because --no-simplify-inh (full 19 types) is enabled.",
            file=sys.stderr,
        )
        args.inh_respective_layer = False
    return args


def _ordered_types(df, aggregate_l5_types: bool) -> list[str]:
    # Backwards-compatible default ordering (assumes inhibitory simplification is enabled).
    present = set(df["source_type"].astype(str)).union(
        set(df["target_type"].astype(str))
    )
    exc_candidates = EXC_L5_AGG if aggregate_l5_types else EXC_ALL
    exc_types = [t for t in exc_candidates if t in present]
    inh_order = ["L1_Inh", "PV", "SST", "VIP"]
    inh_types = [t for t in inh_order if t in present and t not in exc_types]
    return exc_types + inh_types


EXC_FULL_19 = ["L2/3_Exc", "L4_Exc", "L5_IT", "L5_ET", "L5_NP", "L6_Exc"]
INH_FULL_19 = [
    "L1_Inh",
    "L2/3_PV",
    "L4_PV",
    "L5_PV",
    "L6_PV",
    "L2/3_SST",
    "L4_SST",
    "L5_SST",
    "L6_SST",
    "L2/3_VIP",
    "L4_VIP",
    "L5_VIP",
    "L6_VIP",
]


def _ordered_types_full19(df: pd.DataFrame, *, omit_np: bool) -> list[str]:
    present = set(df["source_type"].astype(str)).union(
        set(df["target_type"].astype(str))
    )
    exc = [t for t in EXC_FULL_19 if t in present and not (omit_np and t == "L5_NP")]
    inh = [t for t in INH_FULL_19 if t in present]
    return exc + inh


_LAYER_ALIASES = {
    "L1": "L1",
    "L2/3": "L2/3",
    "L23": "L2/3",
    "23": "L2/3",
    "L4": "L4",
    "4": "L4",
    "L5": "L5",
    "5": "L5",
    "L6": "L6",
    "6": "L6",
}


def _filter_inh_layer(df: pd.DataFrame, layer: str | None) -> pd.DataFrame:
    """Keep only inhibitory populations from a given layer (e.g., L5_PV) before aggregation."""
    if layer is None:
        return df

    norm_layer = _LAYER_ALIASES.get(layer, layer)
    if not isinstance(norm_layer, str):
        return df
    norm_layer = norm_layer.strip()

    # Collect inhibitory types that belong to the requested layer
    allowed_inh = {
        ct
        for fullset in INH_SIMPLE_MAP.values()
        for ct in fullset
        if str(ct).startswith(f"{norm_layer}_")
    }
    if norm_layer == "L1":
        allowed_inh.add("L1_Inh")
    if not allowed_inh:
        raise ValueError(f"No inhibitory populations match layer '{layer}'.")

    inh_flat = {ct for fullset in INH_SIMPLE_MAP.values() for ct in fullset}

    def _keep(cell_type: str) -> bool:
        if cell_type in allowed_inh:
            return True
        if cell_type == "L1_Inh":
            return norm_layer == "L1"
        # Drop inhibitory types from other layers
        if cell_type in inh_flat:
            return False
        return True

    mask = df["source_type"].map(_keep) & df["target_type"].map(_keep)
    return df[mask]


def _extract_layer(cell_type: str | None) -> str | None:
    if not isinstance(cell_type, str):
        return None
    if cell_type.startswith("L1_"):
        return "L1"
    for prefix in ("L2/3_", "L4_", "L5_", "L6_"):
        if cell_type.startswith(prefix):
            return prefix[:-1] if prefix.endswith("_") else prefix
    if cell_type == "L2/3_Exc":
        return "L2/3"
    if cell_type in ("L5_Exc", "L5_IT", "L5_ET", "L5_NP"):
        return "L5"
    if cell_type == "L6_Exc":
        return "L6"
    return None


def _filter_inh_respective_layer(df: pd.DataFrame) -> pd.DataFrame:
    """For Exc↔Inh pairs, keep inhibitory rows only when layers match the Exc layer."""
    inh_layered = {ct for fullset in INH_SIMPLE_MAP.values() for ct in fullset} | {
        "L1_Inh"
    }

    def _row_ok(row) -> bool:
        s, t = row["source_type"], row["target_type"]
        s_inh = s in inh_layered
        t_inh = t in inh_layered
        s_layer = _extract_layer(s)
        t_layer = _extract_layer(t)

        if s_inh and not t_inh:
            if s_layer == "L1":
                return True  # Allow L1 inhibitory to pair with all Exc
            return s_layer is None or t_layer is None or s_layer == t_layer
        if t_inh and not s_inh:
            if t_layer == "L1":
                return True  # Allow L1 inhibitory to pair with all Exc
            return t_layer is None or s_layer is None or s_layer == t_layer
        return True

    return df[df.apply(_row_ok, axis=1)]


def _domain_mean(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    mask = (x >= lo) & (x <= hi) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(y[mask]))


def _domain_mean_from_bin_means(
    centers: np.ndarray, bin_means: np.ndarray, lo: float, hi: float
) -> float:
    """Two-step average: mean over bins, then mean over bin-means within [lo, hi].

    This avoids aggregating raw points across the whole domain range; instead each x-bin
    contributes equally (ignoring empty/NaN bins).
    """
    centers = np.asarray(centers)
    bin_means = np.asarray(bin_means)
    mask = (
        (centers >= lo)
        & (centers <= hi)
        & np.isfinite(centers)
        & np.isfinite(bin_means)
    )
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(bin_means[mask]))


def _bin_mean_sem_min_count(
    x: np.ndarray, y: np.ndarray, bins: np.ndarray, *, min_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin x and compute (centers, mean, sem, count) with a minimum count per bin.

    For bins with count < min_count, mean/sem are set to NaN (so they are neither plotted
    nor used in downstream range averaging).
    """
    if min_count < 1:
        raise ValueError("min_count must be >= 1")

    df = pd.DataFrame({"x": x, "y": y})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    centers = (bins[:-1] + bins[1:]) / 2
    if df.empty:
        nan = np.full_like(centers, np.nan, dtype=float)
        return centers, nan, nan, np.zeros_like(centers, dtype=int)

    df["bin"] = pd.cut(df["x"], bins, include_lowest=True)
    grouped = df.groupby("bin", observed=False)["y"]
    means = grouped.mean()
    sems = grouped.sem()
    counts = grouped.count()

    # Align to the full set of bins (including empty bins)
    idx = pd.cut(centers, bins, include_lowest=True)
    means = means.reindex(idx).to_numpy(dtype=float)
    sems = sems.reindex(idx).to_numpy(dtype=float)
    counts = counts.reindex(idx).fillna(0).to_numpy(dtype=int)

    bad = counts < int(min_count)
    means[bad] = np.nan
    sems[bad] = np.nan
    return centers, means, sems, counts


def _compute_deltas(
    x: np.ndarray, y: np.ndarray, bins: np.ndarray, *, min_bin_count: int
) -> tuple[float, float]:
    """Return (delta_anti, delta_like) normalized by none-domain mean.

    Uses two-step averaging (domain mean computed from per-bin means) to match the
    Fig5-style panels where y is binned along x.
    """
    centers, means, _sems, _counts = _bin_mean_sem_min_count(
        x, y, bins, min_count=min_bin_count
    )
    anti_mean = _domain_mean_from_bin_means(centers, means, -1.0, -0.5)
    none_mean = _domain_mean_from_bin_means(centers, means, -0.25, 0.25)
    like_mean = _domain_mean_from_bin_means(centers, means, 0.5, 1.0)
    if not np.isfinite(none_mean) or none_mean == 0:
        return np.nan, np.nan
    # Flip sign so positive indicates "more like-to-like" in BOTH delta_like and delta_anti.
    # (anti domain is anti-correlated; lower anti_mean corresponds to stronger like-to-like selectivity)
    delta_anti = (
        (none_mean - anti_mean) / none_mean if np.isfinite(anti_mean) else np.nan
    )
    delta_like = (
        (like_mean - none_mean) / none_mean if np.isfinite(like_mean) else np.nan
    )
    return float(delta_anti), float(delta_like)


def _plot_delta_heatmaps(
    types: list[str],
    delta_anti: np.ndarray,
    delta_like: np.ndarray,
    out_base: str,
    *,
    labels: list[str] | None = None,
    dpi: int = 350,
    also_pdf: bool = False,
) -> None:
    apply_pub_style()
    try:
        import matplotlib as mpl
        from matplotlib import font_manager as fm

        _ = fm.findfont("Arial", fallback_to_default=False)
        mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"]})
    except Exception:
        pass

    n = len(types)
    mats = {"delta_anti": delta_anti, "delta_like": delta_like}
    finite = np.isfinite(delta_anti) | np.isfinite(delta_like)
    if finite.any():
        max_abs = float(
            np.nanmax(np.abs(np.concatenate([delta_anti[finite], delta_like[finite]])))
        )
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1.0
    else:
        max_abs = 1.0

    for name, mat in mats.items():
        fig, ax = plt.subplots(figsize=(min(8.0, 0.22 * n), min(8.0, 0.20 * n)))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad(color="white")
        im = ax.imshow(
            mat, vmin=-max_abs, vmax=max_abs, cmap=cmap, interpolation="nearest"
        )

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        tick_labels = labels if labels is not None else types
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticklabels(tick_labels, fontsize=6)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
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

        # NA markers: thin diagonal line in each NA cell
        na = ~np.isfinite(mat)
        for i, j in zip(*np.where(na)):
            ax.plot(
                [j - 0.5, j + 0.5],
                [i + 0.5, i - 0.5],
                color="#666666",
                linewidth=0.4,
                alpha=0.7,
            )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        # cbar.set_label(name, fontsize=8)
        # do instead at the title
        title = "Δanti" if name == "delta_anti" else ("Δlike" if name == "delta_like" else name)
        ax.set_title(title, fontsize=8)

        trim_spines(ax)

        fig.tight_layout()
        out_png = f"{out_base}_{name}_heatmap.png"
        _save_png_and_optional_pdf(fig, out_png, dpi=dpi, also_pdf=also_pdf)
        plt.close(fig)


def _render_pair(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: np.ndarray,
    bin_size: float,
    x_min: float,
    x_max: float,
    xticks: list[float],
    show_legend: bool = False,
    min_bin_count: int = 5,
):
    if x.size < 3 or y.size < 3:
        ax.text(
            0.5,
            0.5,
            "No data / N<3",
            ha="center",
            va="center",
            fontsize=6,
            transform=ax.transAxes,
        )
        ax.set_xlim(x_min, x_max)
        trim_spines(ax)
        return None

    centers, means, sems, _counts = _bin_mean_sem_min_count(
        x, y, bins, min_count=min_bin_count
    )
    # Match original response-correlation hue (pink from earlier panels)
    bar_color = "#f4b6c2"
    finite_bar = np.isfinite(means)
    ax.bar(
        centers[finite_bar],
        means[finite_bar],
        width=bin_size,
        color=bar_color,
        edgecolor="none",
        alpha=0.8,
    )
    ax.errorbar(
        centers[finite_bar],
        means[finite_bar],
        yerr=sems[finite_bar],
        fmt="none",
        ecolor="#4a4a4a",
        elinewidth=0.8,
        capsize=2,
    )

    # Preserve y-limits based on bar heights only (fits should not rescale)
    finite_mean = np.isfinite(means)
    finite_sem = np.isfinite(sems)
    if finite_mean.any():
        top = (means + np.where(finite_sem, sems, 0))[finite_mean].max()
        bottom = (means - np.where(finite_sem, sems, 0))[finite_mean].min()
        if not np.isfinite(top):
            top = np.nanmax(means)
        if not np.isfinite(bottom):
            bottom = np.nanmin(means)
    else:
        top, bottom = 0.05, -0.05
    span = top - bottom
    if span <= 0 or not np.isfinite(span):
        span = max(abs(top), abs(bottom), 1e-3)
    pad = 0.08 * span
    # Anchor the x-axis at y=0: if data are all positive, start at 0; if all negative, cap at 0.
    if top <= 0:
        y_hi = 0.0
        y_lo = bottom - pad
    elif bottom >= 0:
        y_lo = 0.0
        y_hi = top + pad
    else:
        y_lo = bottom - pad
        y_hi = top + pad

    xs = np.linspace(x_min, x_max, 200)

    # Domain means and normalized differences (anti/none/like)
    # NOTE: two-step averaging: compute per-bin mean first, then average bin-means in each range.
    anti_mean = _domain_mean_from_bin_means(centers, means, -1.0, -0.5)
    none_mean = _domain_mean_from_bin_means(centers, means, -0.25, 0.25)
    like_mean = _domain_mean_from_bin_means(centers, means, 0.5, 1.0)
    delta_anti = (
        (none_mean - anti_mean) / none_mean
        if np.isfinite(anti_mean) and np.isfinite(none_mean) and none_mean != 0
        else np.nan
    )
    delta_like = (
        (like_mean - none_mean) / none_mean
        if np.isfinite(like_mean) and np.isfinite(none_mean) and none_mean != 0
        else np.nan
    )

    text_lines = []
    if np.isfinite(delta_anti):
        text_lines.append(f"Δanti={delta_anti:.2f}")
    if np.isfinite(delta_like):
        text_lines.append(f"Δlike={delta_like:.2f}")
    if text_lines:
        ax.text(
            0.03,
            0.95,
            "\n".join(text_lines),
            transform=ax.transAxes,
            fontsize=6,
            va="top",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(xticks)
    ax.set_ylim(y_lo, y_hi)
    # Visual anchor at x=0 to reinforce alignment
    if x_min <= 0 <= x_max:
        ax.axvline(
            0.0, color="#999999", linewidth=0.7, linestyle="-", alpha=0.6, zorder=0
        )
    # Visual anchor at y=0 so the x-axis aligns with zero
    if y_lo <= 0 <= y_hi:
        ax.axhline(
            0.0, color="#aaaaaa", linewidth=0.7, linestyle="-", alpha=0.7, zorder=0
        )

    # Correlation-domain markers (anti / none / like): horizontal segments at the domain mean weight
    seg_color = "#4C78A8"  # bluish
    domain_segments = [
        ("anti", -1.0, -0.5, -0.75, anti_mean),
        ("none", -0.25, 0.25, 0.0, none_mean),
        ("like", 0.5, 1.0, 0.75, like_mean),
    ]
    for _, xmin, xmax, _, yseg in domain_segments:
        if not np.isfinite(yseg):
            continue
        if xmax < x_min or xmin > x_max:
            continue
        if not (y_lo <= yseg <= y_hi):
            continue
        ax.hlines(
            y=yseg,
            xmin=max(xmin, x_min),
            xmax=min(xmax, x_max),
            colors=seg_color,
            linewidth=2.4,
            alpha=0.38,
            zorder=1,
        )

    # Connect midpoints of the domain bars (anti->none->like)
    pts = [
        (xmid, ymean) for _, _, _, xmid, ymean in domain_segments if np.isfinite(ymean)
    ]
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        if not (x_min <= x1 <= x_max and x_min <= x2 <= x_max):
            continue
        if not (y_lo <= y1 <= y_hi and y_lo <= y2 <= y_hi):
            continue
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=seg_color,
            linewidth=1.0,
            alpha=0.35,
            zorder=1,
        )
    trim_spines(ax)
    handles, labels = ax.get_legend_handles_labels()
    if show_legend and handles:
        ax.legend(handles, labels, fontsize=6, frameon=False, loc="lower left")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Fig.5-style correlation matrix with extra fits."
    )
    ap.add_argument(
        "--bases",
        nargs="*",
        default=None,
        help="Base directories (defaults to core_nll_* present).",
    )
    ap.add_argument("--network-type", default="bio_trained")
    ap.add_argument("--bin-size", type=float, default=0.1)
    ap.add_argument("--x-min", type=float, default=-1.0)
    ap.add_argument("--x-max", type=float, default=1.0)
    ap.add_argument(
        "--out", default="aggregated_response_correlation/bio_trained_fig5style.png"
    )
    ap.add_argument(
        "--min-bin-count",
        type=int,
        default=5,
        help="Minimum datapoints required per x-bin to plot/compute SEM and to include in range averages.",
    )
    ap.add_argument(
        "--aggregate-l5", action="store_true", help="Aggregate L5 IT/ET/NP into L5_Exc."
    )
    ap.add_argument(
        "--no-simplify-inh",
        action="store_true",
        help="Disable PV/SST/VIP aggregation across layers.",
    )
    ap.add_argument(
        "--inh-layer",
        default=None,
        help="Restrict inhibitory populations to this layer before aggregation (e.g., L5 keeps only L5_PV/L5_SST/L5_VIP).",
    )
    ap.add_argument(
        "--inh-respective-layer",
        action="store_true",
        help="For Exc↔Inh pairs, keep only inhibitory populations from the corresponding Exc layer (e.g., E23↔PV uses L2/3_PV).",
    )
    ap.add_argument(
        "--omit-np",
        action="store_true",
        help="Omit L5_NP populations from source/target types.",
    )
    ap.add_argument(
        "--heatmaps",
        action="store_true",
        help="Write delta_anti and delta_like heatmaps instead of the matrix plot. Uses --out as a prefix (without .png).",
    )
    ap.add_argument(
        "--abbrev-labels",
        action="store_true",
        help="Use abbreviated cell-type labels (E23, PV23, etc.) for plot/heatmap tick labels and titles.",
    )
    ap.add_argument(
        "--also-pdf",
        action="store_true",
        help="Also write a PDF version alongside each PNG output (same basename).",
    )
    ap.add_argument(
        "--png-dpi",
        type=int,
        default=350,
        help="DPI used for PNG outputs (PDF is vector and does not use DPI).",
    )
    args = _sanitize_args_for_consistency(ap.parse_args())

    apply_pub_style()
    try:
        import matplotlib as mpl
        from matplotlib import font_manager as fm

        _ = fm.findfont("Arial", fallback_to_default=False)
        mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"]})
    except Exception:
        pass

    bases = args.bases if args.bases else _detect_default_bases()
    if not bases:
        raise RuntimeError("No base directories found (expected core_nll_*).")

    # Force symmetric x-limits around zero to keep 0 centered across all panels
    span = max(abs(args.x_min), abs(args.x_max))
    x_min = -span
    x_max = span

    edge_dfs = [process_network_data((bd, args.network_type)) for bd in bases]
    df = pd.concat(edge_dfs, ignore_index=True)

    if args.omit_np:
        df = df[(df["source_type"] != "L5_NP") & (df["target_type"] != "L5_NP")]

    df = _filter_inh_layer(df, args.inh_layer)
    if args.inh_respective_layer:
        df = _filter_inh_respective_layer(df)

    simplify_inh = not args.no_simplify_inh
    if simplify_inh:
        df = apply_inh_simplification(df)
    if args.aggregate_l5:
        df = aggregate_l5(df)

    # Type ordering: if not simplifying inhibitory, use full 19-type ordering.
    if simplify_inh:
        types = _ordered_types(df, args.aggregate_l5)
    else:
        types = _ordered_types_full19(df, omit_np=args.omit_np)
    df = df[df["source_type"].isin(types) & df["target_type"].isin(types)].copy()

    if args.heatmaps:
        bins = np.arange(x_min, x_max + args.bin_size, args.bin_size)
        n = len(types)
        anti_mat = np.full((n, n), np.nan, dtype=float)
        like_mat = np.full((n, n), np.nan, dtype=float)
        for i, s in enumerate(types):
            for j, t in enumerate(types):
                sub = df[(df["source_type"] == s) & (df["target_type"] == t)][
                    ["Response Correlation", "syn_weight"]
                ].dropna()
                x = sub["Response Correlation"].to_numpy()
                y = sub["syn_weight"].to_numpy()
                mask = (x >= x_min) & (x <= x_max) & np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]
                if x.size < 3 or y.size < 3:
                    continue
                d_anti, d_like = _compute_deltas(
                    x, y, bins, min_bin_count=args.min_bin_count
                )
                anti_mat[i, j] = d_anti
                like_mat[i, j] = d_like

        out_base = args.out[:-4] if args.out.endswith(".png") else args.out
        os.makedirs(
            os.path.dirname(out_base) if os.path.dirname(out_base) else ".",
            exist_ok=True,
        )
        labels = abbrev_cell_types(types) if args.abbrev_labels else None
        _plot_delta_heatmaps(
            types,
            anti_mat,
            like_mat,
            out_base,
            labels=labels,
            dpi=args.png_dpi,
            also_pdf=args.also_pdf,
        )
        print(
            f"Saved heatmaps: {out_base}_delta_anti_heatmap.png and {out_base}_delta_like_heatmap.png"
        )
        return

    bins = np.arange(x_min, x_max + args.bin_size, args.bin_size)
    xticks = [x_min, -0.5 * span, 0.0, 0.5 * span, x_max]

    n = len(types)
    fig, axes = plt.subplots(
        n, n, figsize=(n * 1.6, n * 1.6), sharex=False, sharey=False
    )
    axes = np.atleast_2d(axes)

    for i, s in enumerate(types):
        for j, t in enumerate(types):
            ax = axes[i, j]
            sub = df[(df["source_type"] == s) & (df["target_type"] == t)][
                ["Response Correlation", "syn_weight"]
            ].dropna()
            x = sub["Response Correlation"].to_numpy()
            y = sub["syn_weight"].to_numpy()
            mask = (x >= x_min) & (x <= x_max) & np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            _render_pair(
                ax,
                x,
                y,
                bins=bins,
                bin_size=args.bin_size,
                x_min=x_min,
                x_max=x_max,
                xticks=xticks,
                show_legend=(i == 0 and j == 0),
                min_bin_count=args.min_bin_count,
            )

            if i == n - 1:
                ax.set_xlabel("Response Corr.", fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(
                    abbrev_cell_type(s) if args.abbrev_labels else s, fontsize=7
                )
            ax.tick_params(axis="y", labelsize=6)
            if i == 0:
                ax.set_title(
                    abbrev_cell_type(t) if args.abbrev_labels else t, fontsize=7, pad=2
                )

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.out.lower().endswith(".png"):
        _save_png_and_optional_pdf(
            fig, args.out, dpi=args.png_dpi, also_pdf=args.also_pdf
        )
    else:
        # If the user explicitly requests a PDF (or other extension), keep that behavior.
        fig.savefig(args.out, dpi=args.png_dpi)
        if args.also_pdf and not args.out.lower().endswith(".pdf"):
            fig.savefig(_png_to_pdf_path(args.out))
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
