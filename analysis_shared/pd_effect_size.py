from __future__ import annotations
import os
import pickle
from typing import Dict, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_shared.io import load_edges_with_pref_dir, load_edges_with_computed_pref_dir
from analysis_shared.grouping import (
    apply_inh_simplification,
    aggregate_l5,
    filter_inh_respective_layer,
)
from analysis_shared.stats import fit_cosine_series_deg
from analysis_shared.style import apply_pub_style, trim_spines
from analysis_shared.celltype_labels import abbrev_cell_type


def compute_effect_size_matrix(
    base_dirs: Sequence[str],
    network_type: str,
    *,
    simplify_inh: bool = True,
    inh_respective_layer: bool = False,
    aggregate_l5_types: bool = False,
    omit_np: bool = False,
    cache_path: str | None = None,
    loader=None,
) -> Dict[str, np.ndarray]:
    """Compute effect sizes a/c and b/c for all source/target cell-type pairs (signed normalization).
    Returns dict with keys: 'types', 'a_over_c', 'b_over_c'.
    Cache results to cache_path (pickle) if provided.
    """
    if loader is None:
        loader = load_edges_with_pref_dir
    # Load and concatenate PD edges with types
    dfs = []
    for bd in base_dirs:
        e = loader(bd, network_type)
        # Attach types via response-correlation pipeline
        from aggregate_correlation_plot import process_network_data

        typed = process_network_data((bd, network_type))
        typed = typed[["source_id", "target_id", "source_type", "target_type"]]
        e = e.merge(typed, on=["source_id", "target_id"], how="left").dropna(
            subset=["source_type", "target_type"]
        )
        dfs.append(e)
    df = pd.concat(dfs, ignore_index=True)

    if omit_np:
        df = df[(df["source_type"] != "L5_NP") & (df["target_type"] != "L5_NP")]
    if simplify_inh and inh_respective_layer:
        df = filter_inh_respective_layer(df)
    if simplify_inh:
        df = apply_inh_simplification(df)
    if aggregate_l5_types:
        df = aggregate_l5(df)

    # Cell-type-first ordering: Exc first (by layer or split L5),
    # then inhibitory families PV → SST → VIP → L1_Inh.
    present_types = sorted(
        list(set(df["source_type"].astype(str)) | set(df["target_type"].astype(str)))
    )

    # Excitatory order
    exc_order_agg = ["L2/3_Exc", "L4_Exc", "L5_Exc", "L6_Exc"]
    exc_order_split = ["L2/3_Exc", "L4_Exc", "L5_IT", "L5_ET", "L5_NP", "L6_Exc"]
    exc_order = exc_order_agg if aggregate_l5_types else exc_order_split
    exc_types = [t for t in exc_order if t in present_types]

    # Inhibitory order
    if simplify_inh:
        inh_order = ["L1_Inh", "PV", "SST", "VIP"]
        inh_types = [t for t in inh_order if t in present_types]
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
    a_over_c = np.full((n, n), np.nan, dtype=float)
    b_over_c = np.full((n, n), np.nan, dtype=float)
    p_a = np.full((n, n), np.nan, dtype=float)
    p_b = np.full((n, n), np.nan, dtype=float)

    for s in types:
        for t in types:
            sub = df[(df["source_type"] == s) & (df["target_type"] == t)][
                ["pref_dir_diff_deg", "syn_weight"]
            ].dropna()
            if sub.empty or len(sub) < 10:
                continue
            x = sub["pref_dir_diff_deg"].to_numpy()
            y = sub["syn_weight"].to_numpy()
            fit = fit_cosine_series_deg(x, y)
            denom = fit.c if np.isfinite(fit.c) and fit.c != 0 else np.nan
            i = type_to_idx[s]
            j = type_to_idx[t]
            if np.isfinite(denom):
                a_over_c[i, j] = fit.a / denom
                b_over_c[i, j] = fit.b / denom
            if np.isfinite(fit.p_a):
                p_a[i, j] = fit.p_a
            if np.isfinite(fit.p_b):
                p_b[i, j] = fit.p_b

    result = {
        "types": types,
        "a_over_c": a_over_c,
        "b_over_c": b_over_c,
        "p_a": p_a,
        "p_b": p_b,
    }
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
    return result


def plot_effect_size_heatmaps(
    effect_data: Dict[str, np.ndarray],
    out_dir: str,
    *,
    vmax_percentile: float = 95.0,
    label_style: str = "abbrev",
    also_pdf: bool = False,
    png_dpi: int = 300,
    file_prefix: str = "",
) -> None:
    apply_pub_style()
    types = effect_data["types"]
    A = effect_data["a_over_c"]
    B = effect_data["b_over_c"]
    P_A = effect_data.get("p_a")
    P_B = effect_data.get("p_b")

    os.makedirs(out_dir, exist_ok=True)

    def _plot_diverging(mat: np.ndarray, title: str, fname: str):
        vals = mat[np.isfinite(mat)]
        if vals.size:
            vmax = np.percentile(np.abs(vals), vmax_percentile)
        else:
            vmax = 1.0
        vmin, vmax = -vmax, vmax
        n = len(types)
        # Compact figure
        fig_w = min(12, max(1, n * 0.23))
        fig_h = fig_w
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(
            mat,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))

        if label_style == "full":
            tick = [str(x) for x in types]
        else:
            tick = [abbrev_cell_type(x) for x in types]
        ax.set_xticklabels(tick, rotation=90, fontsize=6)
        ax.set_yticklabels(tick, fontsize=6)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        _ = (xmin, xmax, ymax)
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

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9)
        trim_spines(ax)
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"{file_prefix}{fname}")
        fig.savefig(out_png, dpi=int(png_dpi))
        if also_pdf:
            base, _ext = os.path.splitext(out_png)
            fig.savefig(base + ".pdf")
        plt.close(fig)

    def _plot_pvalues(mat: np.ndarray, title: str, fname: str):
        # Render -log10(p). Higher = more significant.
        arr = np.asarray(mat, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = -np.log10(arr)
        z[~np.isfinite(z)] = np.nan
        z[z < 0] = np.nan

        vals = z[np.isfinite(z)]
        if vals.size:
            vmax = float(np.percentile(vals, vmax_percentile))
        else:
            vmax = 1.0
        vmax = max(vmax, 1e-6)
        vmin = 0.0

        n = len(types)
        fig_w = min(12, max(1, n * 0.23))
        fig_h = fig_w
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="white")
        im = ax.imshow(
            z,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest",
        )

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        if label_style == "full":
            tick = [str(x) for x in types]
        else:
            tick = [abbrev_cell_type(x) for x in types]
        ax.set_xticklabels(tick, rotation=90, fontsize=6)
        ax.set_yticklabels(tick, fontsize=6)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        _ = (xmin, xmax, ymax)
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

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9)
        trim_spines(ax)
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"{file_prefix}{fname}")
        fig.savefig(out_png, dpi=int(png_dpi))
        if also_pdf:
            base, _ext = os.path.splitext(out_png)
            fig.savefig(base + ".pdf")
        plt.close(fig)

    _plot_diverging(A, "a/c", "effect_size_a_over_c.png")
    _plot_diverging(B, "b/c", "effect_size_b_over_c.png")
    if P_A is not None:
        _plot_pvalues(P_A, "-log10 p(a)", "effect_size_p_a.png")
    if P_B is not None:
        _plot_pvalues(P_B, "-log10 p(b)", "effect_size_p_b.png")
