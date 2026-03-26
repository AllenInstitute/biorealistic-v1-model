#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure repository root is on PYTHONPATH when run from any working directory
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _short_label(name: str) -> str:
    # Keep full, standard cell-type strings for publication panels
    # (e.g., "L2/3_Exc", "L4_Exc", "L5_ET", "L1_Inh", "PV", "SST", "VIP").
    return str(name)


def _reorder_types_matrix(eff: dict) -> dict:
    types = eff["types"]

    def _order_key(t: str) -> tuple[int, str]:
        exc_order = [
            "L2/3_Exc",
            "L4_Exc",
            "L5_Exc",
            "L5_IT",
            "L5_ET",
            "L5_NP",
            "L6_Exc",
        ]
        inh_order = ["L1_Inh", "PV", "SST", "VIP"]
        if t in exc_order:
            return (0, exc_order.index(t))
        if t in inh_order:
            return (1, inh_order.index(t))
        return (2, t)

    order_idx = sorted(range(len(types)), key=lambda i: _order_key(types[i]))
    import numpy as _np

    A = eff["a_over_c"][_np.ix_(order_idx, order_idx)]
    B = eff["b_over_c"][_np.ix_(order_idx, order_idx)]
    types_new = [types[i] for i in order_idx]
    return {"types": types_new, "a_over_c": A, "b_over_c": B}


def _drop_types_from_matrix(eff: dict, drop: set[str]) -> dict:
    """Remove specified cell types from the effect-size matrices (rows/cols)."""
    if not drop:
        return eff
    types = list(eff["types"])
    keep_idx = [i for i, t in enumerate(types) if t not in drop]
    import numpy as _np

    A = eff["a_over_c"][_np.ix_(keep_idx, keep_idx)]
    B = eff["b_over_c"][_np.ix_(keep_idx, keep_idx)]
    types_new = [types[i] for i in keep_idx]
    return {"types": types_new, "a_over_c": A, "b_over_c": B}


from analysis_shared.style import apply_pub_style, trim_spines
from analysis_shared.pd import (
    compute_pd_full_matrix_cache,
    compute_pd_ei2x2_cache,
)
from analysis_shared.pd_effect_size import compute_effect_size_matrix
from analysis_shared.em_compare import load_em_pd_pickle, compute_em_pd_pvalues
from analysis_shared.pd_mc import compute_pd_mc_pvalues
from analysis_shared.stats import fit_cosine_series_deg, bin_mean_sem


def _format_n_millions(n_conn: int | None) -> str:
    if n_conn is None:
        return ""
    m = float(n_conn) / 1e6
    if m >= 10.0:
        s = f"{m:.1f}M"
    else:
        s = f"{m:.2f}M"
    return s


def _render_hist_fit(
    ax: plt.Axes,
    *,
    centers: np.ndarray,
    means: np.ndarray,
    sems: np.ndarray | None,
    fit: dict,
    show_pvals: bool,
    n_conn: int | None,
    bar_color: str,
    xticks: list[int] | None = None,
    title: str | None = None,
    title_pad: float = 1.0,
    xlabel: str | None = None,
    xlabel_pad: float = 1.0,
    text_xy: tuple[float, float] | None = None,
    text_ha: str | None = None,
    format_n_millions: bool = True,
) -> None:
    x_min, x_max = 0.0, 180.0
    xs = np.linspace(x_min, x_max, 361)
    # Draw histogram with SEM bars
    bin_width = (
        (centers[1] - centers[0])
        if (centers is not None and len(centers) > 1)
        else 20.0
    )
    ax.bar(centers, means, width=bin_width, color=bar_color, edgecolor="none")
    if sems is not None:
        ax.errorbar(
            centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2
        )
    # Fit overlay
    ys = (
        fit["a"] * np.cos(np.radians(xs))
        + fit["b"] * np.cos(2 * np.radians(xs))
        + fit["c"]
    )
    ax.plot(xs, ys, color="crimson", linewidth=1.0)
    # Effect-size labels
    denom = fit.get("c", np.nan)
    a_over_c = (fit["a"] / denom) if np.isfinite(denom) and denom != 0 else np.nan
    b_over_c = (fit["b"] / denom) if np.isfinite(denom) and denom != 0 else np.nan
    text_y = 0.08 if np.nanmean(means) >= 0 else 0.92
    label = f"a/c={a_over_c:.2f}\nb/c={b_over_c:.2f}"
    if show_pvals:
        label = (
            f"a/c={a_over_c:.2f}  p(a)={fit.get('p_a', np.nan):.1e}\n"
            f"b/c={b_over_c:.2f}  p(b)={fit.get('p_b', np.nan):.1e}"
        )
    tx = 0.03 if text_xy is None else text_xy[0]
    ty = text_y if text_xy is None else text_xy[1]
    va = (
        ("bottom" if text_y < 0.5 else "top")
        if text_xy is None
        else ("top" if ty > 0.5 else "bottom")
    )
    ha = "left" if text_xy is None else (text_ha or "center")
    ax.text(tx, ty, label, transform=ax.transAxes, fontsize=6, va=va, ha=ha)
    if n_conn is not None:
        n_y = 0.05 if np.nanmean(means) >= 0 else 0.95
        n_va = "bottom" if n_y < 0.5 else "top"
        n_str = _format_n_millions(n_conn) if format_n_millions else f"{n_conn}"
        ax.text(
            0.97,
            n_y,
            f"N={n_str}",
            transform=ax.transAxes,
            fontsize=6,
            ha="right",
            va=n_va,
        )
    # Axes cosmetics
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(xticks if xticks is not None else [0, 90, 180])
    ax.tick_params(axis="x", labelsize=6, pad=1)
    ax.tick_params(axis="y", labelsize=6, pad=1)
    if title is not None:
        ax.set_title(title, fontsize=7, pad=title_pad)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=7, labelpad=xlabel_pad)
    trim_spines(ax)


def _draw_pd_panel_from_cache(cache: dict, s: str, t: str, ax: plt.Axes) -> None:
    centers = cache["centers"]
    entry = cache["pairs"].get((s, t))
    if not entry:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            fontsize=7,
            transform=ax.transAxes,
        )
        trim_spines(ax)
        return
    _render_hist_fit(
        ax,
        centers=centers,
        means=entry["means"],
        sems=entry["sems"],
        fit=entry["fit"],
        show_pvals=False,
        n_conn=entry.get("N"),
        bar_color="#aec7e8",
    )


def _em_group_hist_pd(
    ax: plt.Axes,
    em: dict,
    src: list[str],
    tgt: list[str],
    *,
    bin_step: float = 20.0,
    text_xy: tuple[float, float] | None = None,
    text_ha: str | None = None,
) -> None:
    parts = []
    for s in src:
        for t in tgt:
            df = em.get((s, t))
            if df is not None and not df.empty:
                parts.append(
                    df[["x", "y"]].rename(
                        columns={"x": "pref_dir_diff_deg", "y": "syn_weight"}
                    )
                )
    if parts:
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["pref_dir_diff_deg", "syn_weight"])  # empty

    # Histogram + cosine fit
    x_min, x_max = 0.0, 180.0
    bins = np.arange(x_min, x_max + bin_step, bin_step)
    if df.empty:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            fontsize=7,
            transform=ax.transAxes,
        )
        ax.set_xlim(x_min, x_max)
        trim_spines(ax)
        return
    xx = df["pref_dir_diff_deg"].to_numpy()
    yy = df["syn_weight"].to_numpy()
    centers, means, sems = bin_mean_sem(xx, yy, bins)
    fit = fit_cosine_series_deg(xx, yy)
    _render_hist_fit(
        ax,
        centers=centers,
        means=means,
        sems=sems,
        fit={"a": fit.a, "b": fit.b, "c": fit.c, "p_a": fit.p_a, "p_b": fit.p_b},
        show_pvals=True,
        n_conn=len(xx),
        bar_color="#cfcfcf",
        text_xy=text_xy if text_xy is not None else (0.97, 0.95),
        text_ha=text_ha if text_ha is not None else "right",
        format_n_millions=False,
    )


def _compute_violin_long_df(
    bases: list[str],
    network_type: str,
    *,
    aggregate_l5: bool,
    pair_limits_csv: str | None,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    em_pd = load_em_pd_pickle(
        os.path.join(
            "analysis_shared", "delta_pref_direction_vs_weight_v1dd_250827.pkl"
        )
    )
    _ = compute_em_pd_pvalues(em_pd)
    mc = compute_pd_mc_pvalues(
        bases,
        network_type,
        aggregate_l5_types=aggregate_l5,
        resamples=resamples,
        connections_per_draw=None,
        seed=seed,
        pair_limits_csv=pair_limits_csv,
    )
    types = ["L2/3", "L4", "L5"]
    pairs_full = [(s, t) for s in types for t in types]
    rows = []
    for s, t in pairs_full:
        d = mc.get((f"{s}_Exc", f"{t}_Exc"), {})
        for param in ("p_a", "p_b"):
            arr = d.get(param, np.array([]))
            if arr.size:
                z = np.log10(np.clip(arr.astype(float), 1e-12, 1.0))
                for v in z:
                    rows.append(
                        {"pair": f"{s}→{t}", "param": param, "log10_p": float(v)}
                    )
    df_base = pd.DataFrame(rows)

    # Add simulation All E→E MC p-values by sampling exactly 426 connections to match experimental data
    # This ensures fair comparison with experimental E→E (426 total connections)
    try:
        from analysis_shared.io import load_edges_with_pref_dir
        from analysis_shared.grouping import aggregate_l5 as _agg_l5
        import numpy as _np

        rng = _np.random.RandomState(seed)

        # Load and pool all E→E simulation connections
        pooled = []
        for bd in bases:
            e = load_edges_with_pref_dir(bd, network_type)
            try:
                from aggregate_correlation_plot import process_network_data

                typed = process_network_data((bd, network_type))
                typed = typed[["source_id", "target_id", "source_type", "target_type"]]
                e = e.merge(typed, on=["source_id", "target_id"], how="left")
            except Exception:
                pass
            pooled.append(e)
        edge_all = pd.concat(pooled, ignore_index=True)
        edge_all = edge_all.dropna(
            subset=["source_type", "target_type", "pref_dir_diff_deg", "syn_weight"]
        ).copy()
        if aggregate_l5:
            edge_all = _agg_l5(edge_all)

        # Filter to only E→E connections
        mask_exc = edge_all["source_type"].astype(str).str.contains("Exc") & edge_all[
            "target_type"
        ].astype(str).str.contains("Exc")
        edge_all = edge_all[mask_exc]

        x_all = edge_all["pref_dir_diff_deg"].to_numpy()
        y_all = edge_all["syn_weight"].to_numpy()
        n_total = len(x_all)

        if n_total >= 426:
            # Sample exactly 426 connections per MC iteration to match experimental data
            pa = []
            pb = []
            for _ in range(resamples):
                idx = rng.choice(n_total, size=426, replace=False)
                x_sample = x_all[idx]
                y_sample = y_all[idx]
                fit = fit_cosine_series_deg(x_sample, y_sample)
                if _np.isfinite(fit.p_a) and _np.isfinite(fit.p_b):
                    pa.append(float(fit.p_a))
                    pb.append(float(fit.p_b))

            # Add E→E entries from 426-connection sampling
            for v in _np.log10(_np.clip(_np.array(pa), 1e-12, 1.0)):
                if _np.isfinite(v):
                    df_base = pd.concat(
                        [
                            df_base,
                            pd.DataFrame(
                                {
                                    "pair": ["E→E"],
                                    "param": ["p_a"],
                                    "log10_p": [float(v)],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
            for v in _np.log10(_np.clip(_np.array(pb), 1e-12, 1.0)):
                if _np.isfinite(v):
                    df_base = pd.concat(
                        [
                            df_base,
                            pd.DataFrame(
                                {
                                    "pair": ["E→E"],
                                    "param": ["p_b"],
                                    "log10_p": [float(v)],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
    except Exception:
        pass

    return df_base


def _plot_pd_violin_on_ax(
    ax: plt.Axes, df_long: pd.DataFrame, em_overlay: dict | None = None
) -> None:
    if df_long is None or df_long.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        trim_spines(ax)
        return
    # Horizontal violin on the right
    base_pairs = [
        f"{s}→{t}" for s in ["L2/3", "L4", "L5"] for t in ["L2/3", "L4", "L5"]
    ]
    present = set(df_long["pair"].unique())
    order = ["E→E"] + [p for p in base_pairs if p in present]
    # Ensure category for E→E exists so seaborn reserves a row
    import numpy as _np

    df_long = pd.concat(
        [
            df_long,
            pd.DataFrame(
                {
                    "pair": ["E→E", "E→E"],
                    "param": ["p_a", "p_b"],
                    "log10_p": [_np.nan, _np.nan],
                }
            ),
        ],
        ignore_index=True,
    )
    v = sns.violinplot(
        data=df_long,
        y="pair",
        x="log10_p",
        hue="param",
        order=order,
        orient="h",
        split=True,
        cut=0,
        inner="quartile",
        palette={"p_a": "C0", "p_b": "C3"},
        linewidth=0.8,
        saturation=0.9,
        scale="width",
        ax=ax,
    )
    tick_vals = list(range(0, -11, -2))  # log10 scale from 0 to -10 (fewer ticks)
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([r"$10^{%d}$" % k for k in tick_vals])
    ax.set_xlim(0, -10)  # 10^0 on the left edge
    p_thresh_log = np.log10(0.05)
    p_label_y = 1.02  # adjust here if label placement needs tweaking
    ax.axvline(x=p_thresh_log, color="gray", linestyle=":", linewidth=1)
    ax.text(
        p_thresh_log,
        p_label_y,
        "p=0.05",
        ha="center",
        va="bottom",
        fontsize=6,
        rotation=0,
        transform=ax.get_xaxis_transform(),
    )
    # Relabel ticks for display (E23/E4/E5/E6, I1 before PV)
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    new_tick_labels = []
    for lbl in tick_labels:
        if "→" in lbl:
            a, b = lbl.split("→")
            new_tick_labels.append(f"{_short_label(a)}→{_short_label(b)}")
        else:
            new_tick_labels.append(_short_label(lbl))
    ytick_pos = ax.get_yticks()
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(new_tick_labels)
    # Standardize font sizes for tick labels
    ax.tick_params(axis="x", labelsize=6, pad=1)
    ax.tick_params(axis="y", labelsize=6, pad=1)
    # Add gray experimental legend dots (p_a and p_b)
    from matplotlib.lines import Line2D

    custom = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#8c8c8c",
            markeredgecolor="k",
            markersize=5,
            label="exp",
        )
    ]
    leg = ax.get_legend()
    if leg is not None:
        # Merge existing handles with experimental handle and remove title
        handles, labels = ax.get_legend_handles_labels()
        # Avoid duplicate 'exp'
        if "exp" not in labels:
            handles += custom
            labels += ["exp"]
        leg.remove()
        ax.legend(
            handles=handles,
            labels=labels,
            frameon=False,
            fontsize=7,
            loc="lower right",
            bbox_to_anchor=(1.02, 0.02),
        )
    ax.set_xlabel("p-value", fontsize=7)
    ax.set_ylabel("")
    # Overlay experimental points (p_a and p_b) if provided
    if em_overlay:
        # Build mapping from relabeled tick labels to y-coordinates
        pair_to_y = {lbl: i for i, lbl in enumerate(new_tick_labels)}
        for (s, t), vals in em_overlay.items():
            lbl = f"{_short_label(s)}→{_short_label(t)}"
            if lbl not in pair_to_y:
                continue
            y = pair_to_y[lbl]
            # Ensure 'a' is plotted slightly above center, 'b' slightly below
            for param, color, yoff in (("p_a", "C0", -0.12), ("p_b", "C3", +0.12)):
                p = vals.get(param)
                if p is not None and np.isfinite(p) and p > 0:
                    # Gray experimental points to match EM histograms
                    ax.scatter(
                        np.log10(p),
                        y + yoff,
                        s=18,
                        color="#8c8c8c",
                        edgecolor="k",
                        linewidths=0.3,
                        zorder=5,
                    )
    trim_spines(ax)


def main():
    ap = argparse.ArgumentParser(
        description="Compose final PD figure (4x4) with caching"
    )
    ap.add_argument("--bases", nargs="*", default=None)
    ap.add_argument("--network-type", default="bio_trained")
    ap.add_argument("--out", default="figures/paper/figure5/pd_final_panel.png")
    ap.add_argument(
        "--also-pdf",
        action="store_true",
        help="Also save a PDF next to --out (same basename).",
    )
    ap.add_argument("--bin-step", type=float, default=20.0)
    ap.add_argument("--resamples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair-limits-csv", default=None)
    ap.add_argument("--cache-dir", default="figures/paper/cache_pd_final")
    ap.add_argument("--replot-only", action="store_true")
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument(
        "--effect-split-e5",
        action="store_true",
        help="Use split L5 excitatory types (L5_IT/L5_ET/L5_NP) for the effect-size heatmaps (a/c, b/c).",
    )
    ap.add_argument(
        "--effect-omit-np",
        action="store_true",
        help="Omit L5_NP from the effect-size heatmaps (only relevant when --effect-split-e5).",
    )
    args = ap.parse_args()

    apply_pub_style()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    bases = (
        args.bases
        if args.bases
        else [f"core_nll_{i}" for i in range(10) if os.path.isdir(f"core_nll_{i}")]
    )
    # Default pair limits for PD if not provided
    if args.pair_limits_csv is None and os.path.isfile("pair_limits_pd.csv"):
        args.pair_limits_csv = "pair_limits_pd.csv"

    os.makedirs(args.cache_dir, exist_ok=True)
    import pickle

    # EI cache
    # Include within-layer inh aggregation in cache key to avoid stale loads.
    ei_pkl = os.path.join(
        args.cache_dir, f"pd_ei2x2_{args.network_type}_inhWithinLayer.pkl"
    )
    ei_cache = None
    if (not args.force_recompute) and os.path.isfile(ei_pkl):
        try:
            with open(ei_pkl, "rb") as f:
                ei_cache = pickle.load(f)
        except Exception:
            ei_cache = None

    def _cache_has_N(cache_obj: dict) -> bool:
        try:
            pairs = cache_obj.get("pairs", {})
            for v in pairs.values():
                if v and ("N" not in v):
                    return False
            return True
        except Exception:
            return False

    if ei_cache is None and not args.replot_only:
        ei_cache = compute_pd_ei2x2_cache(
            bases,
            args.network_type,
            simplify_inh=True,
            inh_respective_layer=True,
            aggregate_l5_types=True,
            bin_step=args.bin_step,
        )
        with open(ei_pkl, "wb") as f:
            pickle.dump(ei_cache, f)
    # Ensure N exists; if missing, recompute
    if ei_cache is not None and not _cache_has_N(ei_cache) and not args.replot_only:
        ei_cache = compute_pd_ei2x2_cache(
            bases,
            args.network_type,
            simplify_inh=True,
            inh_respective_layer=True,
            aggregate_l5_types=True,
            bin_step=args.bin_step,
        )
        with open(ei_pkl, "wb") as f:
            pickle.dump(ei_cache, f)

    # Full cache
    full_pkl = os.path.join(
        args.cache_dir, f"pd_full_{args.network_type}_inhWithinLayer.pkl"
    )
    full_cache = None
    if (not args.force_recompute) and os.path.isfile(full_pkl):
        try:
            with open(full_pkl, "rb") as f:
                full_cache = pickle.load(f)
        except Exception:
            full_cache = None
    if full_cache is None and not args.replot_only:
        full_cache = compute_pd_full_matrix_cache(
            bases,
            args.network_type,
            simplify_inh=True,
            inh_respective_layer=True,
            aggregate_l5_types=True,
            bin_step=args.bin_step,
        )
        with open(full_pkl, "wb") as f:
            pickle.dump(full_cache, f)
    # Ensure N exists; if missing, recompute
    if full_cache is not None and not _cache_has_N(full_cache) and not args.replot_only:
        full_cache = compute_pd_full_matrix_cache(
            bases,
            args.network_type,
            simplify_inh=True,
            inh_respective_layer=True,
            aggregate_l5_types=True,
            bin_step=args.bin_step,
        )
        with open(full_pkl, "wb") as f:
            pickle.dump(full_cache, f)

    # Effect size cache
    eff_l5_tag = "splitE5" if args.effect_split_e5 else "aggE5"
    eff_np_tag = "_omitNP" if args.effect_omit_np else ""
    eff_pkl = os.path.join(
        args.cache_dir,
        f"pd_effect_{args.network_type}_inhaggWithinLayer_{eff_l5_tag}{eff_np_tag}.pkl",
    )
    if (not args.force_recompute) and os.path.isfile(eff_pkl):
        try:
            with open(eff_pkl, "rb") as f:
                eff = pickle.load(f)
        except Exception:
            eff = compute_effect_size_matrix(
                bases,
                args.network_type,
                simplify_inh=True,
                inh_respective_layer=True,
                aggregate_l5_types=(not args.effect_split_e5),
                cache_path=eff_pkl,
            )
    else:
        if not args.replot_only:
            eff = compute_effect_size_matrix(
                bases,
                args.network_type,
                simplify_inh=True,
                inh_respective_layer=True,
                aggregate_l5_types=(not args.effect_split_e5),
                cache_path=eff_pkl,
            )
        else:
            with open(eff_pkl, "rb") as f:
                eff = pickle.load(f)
    if args.effect_split_e5 and args.effect_omit_np:
        eff = _drop_types_from_matrix(eff, {"L5_NP"})

    # Violin DF cache (encode limits + resamples in name so content is valid)
    lim_tag = (
        os.path.basename(args.pair_limits_csv) if args.pair_limits_csv else "nolimits"
    )
    vio_pkl = os.path.join(
        args.cache_dir,
        f"pd_violin_long_{args.network_type}_{lim_tag}_r{args.resamples}_s{args.seed}.pkl",
    )
    if (not args.force_recompute) and os.path.isfile(vio_pkl):
        try:
            with open(vio_pkl, "rb") as f:
                df_long = pickle.load(f)
        except Exception:
            df_long = None
    else:
        df_long = None
    if df_long is None and not args.replot_only:
        df_long = _compute_violin_long_df(
            bases,
            args.network_type,
            aggregate_l5=True,
            pair_limits_csv=args.pair_limits_csv,
            resamples=args.resamples,
            seed=args.seed,
        )
        with open(vio_pkl, "wb") as f:
            pickle.dump(df_long, f)
    if df_long is None:
        df_long = pd.DataFrame()

    # EM for small panels and violin overlay
    em_pd = load_em_pd_pickle(
        os.path.join(
            "analysis_shared", "delta_pref_direction_vs_weight_v1dd_250827.pkl"
        )
    )
    em_pvals_overlay = compute_em_pd_pvalues(em_pd)
    # Add aggregated E→E experimental p-values by combining all excitatory layer pairs
    try:
        import numpy as _np

        xs = []
        ys = []
        for s in ["L2/3", "L4", "L5"]:
            for t in ["L2/3", "L4", "L5"]:
                df = em_pd.get((s, t))
                if df is None or df.empty:
                    continue
                xs.append(df["x"].to_numpy())
                ys.append(df["y"].to_numpy())
        if xs and ys:
            xcat = _np.concatenate(xs)
            ycat = _np.concatenate(ys)
            fit = fit_cosine_series_deg(xcat, ycat)
            em_pvals_overlay[("E", "E")] = {
                "p_a": float(fit.p_a),
                "p_b": float(fit.p_b),
            }
    except Exception:
        pass

    # Layout using GridSpec 4x4 for PD-only half page
    # Give a bit more width to the heatmap/violin columns, and a bit more height
    # so the heatmaps have more vertical extent (~20% taller overall).
    fig = plt.figure(figsize=(7.2, 5.4))
    # Give a bit more vertical separation between the top-row heatmaps and the panels below,
    # so the heatmap x-label ("target") doesn't collide with the next row titles.
    gs = fig.add_gridspec(
        nrows=4,
        ncols=4,
        left=0.06,
        right=0.93,
        # Leave extra headroom so the slightly raised heatmaps don't clip outside the canvas.
        top=0.955,
        bottom=0.09,
        width_ratios=[1.0, 1.0, 1.18, 1.18],
        height_ratios=[1.2, 1.0, 1.0, 1.0],
        wspace=0.48,
        hspace=0.62,
    )

    # Panels 1,2,5,6: reserved 2x2 block (rows 0-1, cols 0-1)
    # We now show only:
    # - bottom-right: E→E (moved to the old I→I slot)
    # - bottom-left: L2/3_Exc→L1_Inh (replacing the old I→E slot)
    # and leave the top row empty for later additions.
    sub = gs[:2, :2].subgridspec(2, 2, wspace=0.4, hspace=0.5)

    # Leave space for top row (reserved)
    ax_empty_00 = fig.add_subplot(sub[0, 0])
    ax_empty_00.axis("off")
    ax_empty_01 = fig.add_subplot(sub[0, 1])
    ax_empty_01.axis("off")

    # Bottom-left: L2/3_Exc → L1_Inh (from full cache)
    ax_l23_l1 = fig.add_subplot(sub[1, 0])
    s_new, t_new = "L2/3_Exc", "L1_Inh"
    if (s_new, t_new) in full_cache.get("pairs", {}):
        _draw_pd_panel_from_cache(full_cache, s_new, t_new, ax_l23_l1)
    else:
        ax_l23_l1.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            fontsize=7,
            transform=ax_l23_l1.transAxes,
        )
        trim_spines(ax_l23_l1)
    ax_l23_l1.set_title(f"{s_new}→{t_new}", fontsize=7, pad=1.0)
    ax_l23_l1.set_ylabel("Weight (pA)", fontsize=7)
    ax_l23_l1.set_xlabel(r"$\delta$ Pref. Dir. (deg)", fontsize=7, labelpad=1.0)

    # Bottom-right: E → E (from EI cache), moved to old I→I slot
    ax_ee = fig.add_subplot(sub[1, 1])
    entry = ei_cache["pairs"].get(("E", "E")) if ei_cache is not None else None
    if entry is None:
        ax_ee.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            fontsize=7,
            transform=ax_ee.transAxes,
        )
        trim_spines(ax_ee)
    else:
        centers = ei_cache["centers"]
        fit = entry.get("fit", {})
        _render_hist_fit(
            ax_ee,
            centers=centers,
            means=entry["means"],
            sems=entry["sems"],
            fit=fit,
            show_pvals=False,
            n_conn=entry.get("N"),
            bar_color="#aec7e8",
            xticks=[0, 90, 180],
            title="E→E",
            title_pad=1.0,
            xlabel=r"$\delta$ Pref. Dir. (deg)",
            xlabel_pad=1.0,
        )

    # Bottom-left: specific sim pairs (rows 2-3, cols 0-1) 2x2 grid
    sub2 = gs[2:, :2].subgridspec(2, 2, wspace=0.4, hspace=0.5)
    pairs_sim = [
        ("L6_Exc", "L6_Exc"),
        ("L2/3_Exc", "L5_Exc"),
        ("PV", "L4_Exc"),
        ("L5_Exc", "L5_Exc"),
    ]
    _display_label = {"PV": "L4_PV"}
    for idx, (s, t) in enumerate(pairs_sim):
        ax = fig.add_subplot(sub2[idx // 2, idx % 2])
        # If inhibitory types not present in full_cache, fall back to EI split by EM; here full_cache includes inh aggregated
        if (s, t) in full_cache["pairs"]:
            ent = full_cache["pairs"].get((s, t))
            if ent and s == "L6_Exc" and t == "L6_Exc":
                _render_hist_fit(
                    ax,
                    centers=full_cache["centers"],
                    means=ent["means"],
                    sems=ent["sems"],
                    fit=ent["fit"],
                    show_pvals=False,
                    n_conn=ent.get("N"),
                    bar_color="#aec7e8",
                    text_xy=(0.5, 0.95),
                    text_ha="center",
                )
            else:
                _draw_pd_panel_from_cache(full_cache, s, t, ax)
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            trim_spines(ax)
        # Title without N
        s_disp = _display_label.get(s, _short_label(s))
        t_disp = _display_label.get(t, _short_label(t))
        ax.set_title(f"{s_disp}→{t_disp}", fontsize=7, pad=1.0)
        ax.set_xticks([0, 90, 180])
        ax.tick_params(axis="x", labelsize=6, pad=1)
        ax.tick_params(axis="y", labelsize=6, pad=1)
        if (idx % 2) == 0:
            ax.set_ylabel("Weight (pA)", fontsize=7)
        if idx // 2 == 1:
            ax.set_xlabel(r"$\delta$ Pref. Dir. (deg)", fontsize=7, labelpad=1.0)

    # Panels 3 and 4: Effect size heatmaps at top-right (rows 0, cols 2 and 3) directly to keep heights equal
    ax_a = fig.add_subplot(gs[:1, 2])
    ax_b = fig.add_subplot(gs[:1, 3])
    eff = _reorder_types_matrix(eff)
    types = eff["types"]
    A = eff["a_over_c"]
    B = eff["b_over_c"]
    valsA = A[np.isfinite(A)]
    vmaxA = np.percentile(np.abs(valsA), 95) if valsA.size else 1.0
    im = ax_a.imshow(
        A,
        cmap="RdBu_r",
        vmin=-vmaxA,
        vmax=vmaxA,
        aspect="equal",
        interpolation="nearest",
    )
    ax_a.set_title("a/c", fontsize=7, pad=2)
    ax_a.set_xticks(range(len(types)))
    ax_a.set_yticks(range(len(types)))
    ax_a.set_xticklabels([_short_label(x) for x in types], rotation=90, fontsize=6)
    ax_a.set_yticklabels([_short_label(x) for x in types], fontsize=6)
    ax_a.tick_params(axis="x", pad=1)
    ax_a.tick_params(axis="y", pad=1)
    # Place "Source"/"Target" labels like the corr panel (anchored to the first tick).
    if True:
        xmin, xmax = ax_a.get_xlim()
        ymin, ymax = ax_a.get_ylim()
        _ = (xmin, xmax, ymax)  # kept for parity/readability with corr script
        ax_a.text(
            -1.5,
            1.00,
            "Source",
            transform=ax_a.get_xaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            clip_on=False,
        )
        ax_a.text(
            -0.05,
            ymin + 1,
            "Target",
            transform=ax_a.get_yaxis_transform(),
            ha="right",
            va="top",
            rotation=90,
            fontsize=7,
            fontweight="bold",
            clip_on=False,
        )
    # Tighten gap between heatmap and colorbar.
    cbar_a = fig.colorbar(im, ax=ax_a, fraction=0.2)
    cbar_a.ax.tick_params(labelsize=7)
    # With aspect='equal' + colorbar, matplotlib centers/shrinks the axes; anchor left.
    ax_a.set_anchor("W")
    trim_spines(ax_a)
    valsB = B[np.isfinite(B)]
    vmaxB = np.percentile(np.abs(valsB), 95) if valsB.size else 1.0
    im2 = ax_b.imshow(
        B,
        cmap="RdBu_r",
        vmin=-vmaxB,
        vmax=vmaxB,
        aspect="equal",
        interpolation="nearest",
    )
    ax_b.set_title("b/c", fontsize=7, pad=2)
    ax_b.set_xticks(range(len(types)))
    ax_b.set_yticks(range(len(types)))
    ax_b.set_xticklabels([_short_label(x) for x in types], rotation=90, fontsize=6)
    ax_b.set_yticklabels([_short_label(x) for x in types], fontsize=6)
    ax_b.tick_params(axis="x", pad=1)
    ax_b.tick_params(axis="y", pad=1)
    # Also place "Source"/"Target" labels on ax_b (match corr script params).
    xmin, xmax = ax_b.get_xlim()
    ymin, ymax = ax_b.get_ylim()
    _ = (xmin, xmax, ymax)  # kept for parity/readability with corr script
    ax_b.text(
        -1.5,
        1.00,
        "Source",
        transform=ax_b.get_xaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        clip_on=False,
    )
    ax_b.text(
        -0.05,
        ymin + 1,
        "Target",
        transform=ax_b.get_yaxis_transform(),
        ha="right",
        va="top",
        rotation=90,
        fontsize=7,
        fontweight="bold",
        clip_on=False,
    )
    cbar_b = fig.colorbar(im2, ax=ax_b, fraction=0.2)
    cbar_b.ax.tick_params(labelsize=7)
    ax_b.set_anchor("W")
    trim_spines(ax_b)

    # Panels 7,11,15: Experimental small panels (rows 1-3, col 2): stacked 3 rows
    sub3 = gs[1:, 2].subgridspec(3, 1, hspace=0.5)
    exp_pairs = [
        (["L2/3", "L4", "L5"], ["L2/3", "L4", "L5"], "E→E"),
        (["L2/3"], ["L5"], "L2/3_Exc→L5_Exc"),
        (["L5"], ["L5"], "L5_Exc→L5_Exc"),
    ]
    for k, (src, tgt, title) in enumerate(exp_pairs):
        ax = fig.add_subplot(sub3[k, 0])
        # For L5→L5, move annotation slightly left (x=0.95)
        if src == ["L5"] and tgt == ["L5"]:
            _em_group_hist_pd(
                ax,
                em_pd,
                src,
                tgt,
                bin_step=args.bin_step,
                text_xy=(0.95, 0.95),
                text_ha="right",
            )
        else:
            _em_group_hist_pd(ax, em_pd, src, tgt, bin_step=args.bin_step)
        ax.set_title(title, fontsize=7, pad=1.0)
        ax.set_ylabel("Total PSD (µm²)", fontsize=7)
        if k == 2:
            ax.set_xlabel(r"$\delta$ Pref. Dir. (deg)", fontsize=7, labelpad=1.0)

    # Panels 8,12,16: horizontal violin spanning rows 1-3, col 3
    # Slightly nudge the violin plot right by shrinking its axes and shifting
    ax_vio = fig.add_subplot(gs[1:, 3])
    pos = ax_vio.get_position()
    ax_vio.set_position([pos.x0 + 0.03, pos.y0, pos.width, pos.height])
    _plot_pd_violin_on_ax(ax_vio, df_long, em_overlay=em_pvals_overlay)
    # ax_vio.set_title("Monte Carlo p-values (PD)", fontsize=7)
    leg = ax_vio.get_legend()
    if leg is not None:
        leg.set_title(leg.get_title().get_text(), prop={"size": 7})
        for txt in leg.get_texts():
            txt.set_fontsize(7)

    fig.savefig(args.out, dpi=300)
    if args.also_pdf:
        base, _ext = os.path.splitext(args.out)
        fig.savefig(base + ".pdf")
    plt.close(fig)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
