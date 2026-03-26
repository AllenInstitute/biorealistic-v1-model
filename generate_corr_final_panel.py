#!/usr/bin/env python3
from __future__ import annotations
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_shared.style import apply_pub_style, trim_spines
from analysis_shared.corr import (
    compute_corr_ei2x2_cache,
    compute_corr_full_matrix_cache,
)
from analysis_shared.corr_mc import compute_corr_mc_pvalues
from analysis_shared.em_compare import load_em_corr_pickle
from generate_corr_compare import compute_corr_effect_size


def _render_corr_hist_fit(
    ax: plt.Axes,
    *,
    centers: np.ndarray,
    means: np.ndarray,
    sems: np.ndarray | None,
    ols: dict,
    show_p: bool,
    n_conn: int | None,
    bar_color: str,
    bar_alpha: float = 0.5,
    n_fmt_millions: bool = False,
    xlim: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    title: str | None = None,
    title_pad: float = 1.0,
    xlabel: str | None = None,
    xlabel_pad: float = 1.0,
    ylabel: str | None = None,
    ylabel_pad: float = 1.0,
    text_xy: tuple[float, float] | None = None,
    text_ha: str = "left",
) -> None:
    ax.bar(centers, means, width=(centers[1] - centers[0]) if centers.size > 1 else 0.05, color=bar_color, edgecolor="none", alpha=bar_alpha)
    if sems is not None:
        ax.errorbar(centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2)
    if xlim is not None:
        x_min, x_max = float(xlim[0]), float(xlim[1])
    else:
        x_min = float(np.min(centers)) if centers.size else -0.2
        x_max = float(np.max(centers)) if centers.size else 0.5
    xs = np.array([x_min, x_max])
    ax.plot(xs, ols["intercept"] + ols["slope"] * xs, color="crimson", linewidth=1.0)
    slope = float(ols.get("slope", np.nan)); intercept = float(ols.get("intercept", np.nan))
    ratio = slope / intercept if (np.isfinite(slope) and np.isfinite(intercept) and intercept != 0.0) else np.nan
    label = f"m/c={ratio:.3f}" if np.isfinite(ratio) else f"m={slope:.3f}"
    if show_p:
        label = f"{label}  p={ols.get('p_value', np.nan):.1e}"
    # Dynamic placement: bottom for excitatory-dominated (positive mean), top for inhibitory-dominated (negative)
    overall_mean = float(np.nanmean(means)) if means.size else 0.0
    if text_xy is None:
        text_xy = (0.03, 0.08 if overall_mean >= 0 else 0.92)
        va = "bottom" if overall_mean >= 0 else "top"
    else:
        va = "top"
    ax.text(text_xy[0], text_xy[1], label, transform=ax.transAxes, fontsize=6, va=va, ha=text_ha)
    if n_conn is not None:
        if n_fmt_millions:
            millions = n_conn / 1e6
            if millions >= 0.995:
                n_str = f"N={millions:.2f}M"
            else:
                n_str = f"N={millions:.2f}M"
        else:
            n_str = f"N={n_conn}"
        # Place N symmetric with label: bottom-right for positive, top-right for negative
        ax.text(0.97, 0.08 if overall_mean >= 0 else 0.92, n_str, transform=ax.transAxes, fontsize=6, ha="right", va="bottom" if overall_mean >= 0 else "top")
    ax.set_xlim(x_min, x_max)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelsize=6, pad=1)
    ax.tick_params(axis="y", labelsize=6, pad=1)
    if title is not None:
        ax.set_title(title, fontsize=7, pad=title_pad)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=7, labelpad=xlabel_pad)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=7, labelpad=ylabel_pad)
    trim_spines(ax)


def _compute_violin_long_df_corr(bases: list[str], network_type: str, *, pair_limits_csv: str | None, resamples: int, seed: int) -> pd.DataFrame:
    """Match the MC pipeline used in corr_compare aggregated figure.
    Uses monte_carlo_pvalue_matrix with aggregated L5 and CSV pair limits.
    """
    focus = ("L2/3_Exc", "L4_Exc", "L5_Exc", "L6_Exc")
    # Load edges and run the same MC as corr_compare
    from aggregate_correlation_plot import process_network_data, monte_carlo_pvalue_matrix
    from analysis_shared.sampling import read_pair_limits_csv
    import pandas as _pd
    dfs = []
    for bd in bases:
        df = process_network_data((bd, network_type))
        dfs.append(df[["source_type", "target_type", "Response Correlation", "syn_weight"]].copy())
    edge_df = _pd.concat(dfs, ignore_index=True)
    limits = read_pair_limits_csv(pair_limits_csv) if pair_limits_csv else None
    pmap, _exc_types = monte_carlo_pvalue_matrix(edge_df, aggregate_l5=True, runs=resamples, max_per_pair=None, pair_limits=limits, base_seed=seed)
    rows = []
    # Individual pairs
    for s in focus:
        for t in focus:
            arr = pmap.get((s, t), [])
            if arr:
                z = np.log10(np.clip(np.asarray(arr, dtype=float), 1e-12, 1.0))
                for v in z:
                    rows.append({"pair": f"{s.replace('_Exc','')}→{t.replace('_Exc','')}", "log10_p": float(v)})
    # Aggregated E→E pooled from all excitatory layer pairs
    pooled = []
    for s in focus:
        for t in focus:
            arr = pmap.get((s, t), [])
            if arr:
                pooled.extend(list(arr))
    if pooled:
        z = np.log10(np.clip(np.array(pooled, dtype=float), 1e-12, 1.0))
        for v in z:
            rows.append({"pair": "E→E", "log10_p": float(v)})
    return pd.DataFrame(rows)


def _plot_corr_violin_on_ax(ax: plt.Axes, df_long: pd.DataFrame, em_overlay: dict | None = None, allowed_pairs: set[str] | None = None) -> None:
    import seaborn as sns
    if df_long is None or df_long.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        trim_spines(ax); return
    base_pairs = [f"{s}→{t}" for s in ("L2/3","L4","L5","L6") for t in ("L2/3","L4","L5","L6")]
    present = set(df_long["pair"].unique())
    if allowed_pairs is not None:
        present = present.intersection(allowed_pairs)
    order = []
    if (allowed_pairs is None) or ("E→E" in allowed_pairs):
        order.append("E→E")
        # Ensure E→E exists to reserve row if allowed
        df_long = pd.concat([df_long, pd.DataFrame({"pair": ["E→E"], "log10_p": [np.nan]})], ignore_index=True)
    order.extend([p for p in base_pairs if p in present])
    if allowed_pairs is not None:
        # Filter df to allowed pairs only (and E→E if included)
        keep = set(order)
        df_long = df_long[df_long["pair"].isin(keep)]
    sns.violinplot(data=df_long, y="pair", x="log10_p", order=order, orient="h", cut=0, inner="quartile", color="#f4b6c2", linewidth=0.8, saturation=0.9, scale="width", ax=ax)
    # Ensure a thin black outline around violins (match PD format)
    try:
        from matplotlib.collections import PolyCollection
        for art in ax.collections:
            if isinstance(art, PolyCollection):
                art.set_edgecolor("k"); art.set_linewidth(0.6)
    except Exception:
        pass
    ax.axvline(x=np.log10(0.05), color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("log10 p-value", fontsize=7); ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=6, pad=1); ax.tick_params(axis="y", labelsize=6, pad=1)
    # Overlay EM points
    if em_overlay:
        tick_labels = [t.get_text() for t in ax.get_yticklabels()]
        pair_to_y = {lbl: i for i, lbl in enumerate(tick_labels)}
        for (s, t), p in em_overlay.items():
            if (s == "E") and (t == "E"):
                lbl = "E→E"
            else:
                lbl = f"{s}→{t}"
            if lbl not in pair_to_y or (p is None) or (not np.isfinite(p)) or (p <= 0):
                continue
            y = pair_to_y[lbl]
            ax.scatter(np.log10(p), y, s=18, color="#8c8c8c", edgecolor="k", linewidths=0.3, zorder=5)
        # Add legend entry
        from matplotlib.lines import Line2D
        custom = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#8c8c8c', markeredgecolor='k', markersize=5, label='exp')]
        leg = ax.get_legend()
        if leg is not None:
            handles, labels = ax.get_legend_handles_labels()
            handles += custom; labels += ['exp']
            leg.remove(); ax.legend(handles=handles, labels=labels, frameon=False, fontsize=7)
    ax.invert_xaxis(); trim_spines(ax)


def _em_group_hist_corr(ax: plt.Axes, em_corr: dict, src_groups: list[str], tgt_groups: list[str], *, bin_size: float = 0.05, text_xy: tuple[float,float] = (0.97, 0.95), text_ha: str = "right") -> None:
    xs = []; ys = []; n_total = 0
    for s in src_groups:
        for t in tgt_groups:
            df = em_corr.get((s, t))
            if df is None or df.empty:
                continue
            xv = df["x"].to_numpy(dtype=float)
            yv = df["y"].to_numpy(dtype=float)
            if xv.size and yv.size:
                xs.append(xv); ys.append(yv); n_total += xv.size
    if not xs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes); trim_spines(ax); return
    x = np.concatenate(xs); y = np.concatenate(ys)
    x_min, x_max = -0.2, 0.5
    bins = np.arange(x_min, x_max + bin_size, bin_size)
    from analysis_shared.stats import bin_mean_sem, ols_slope_p
    centers, means, sems = bin_mean_sem(x, y, bins)
    ax.bar(centers, means, width=bin_size, color="#cfcfcf", edgecolor="none")
    ax.errorbar(centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2)
    res = ols_slope_p(x, y)
    xs_line = np.array([x_min, x_max])
    ax.plot(xs_line, res.intercept + res.slope * xs_line, color="crimson", linewidth=1.0)
    ratio = res.slope / res.intercept if (res.intercept != 0 and np.isfinite(res.intercept)) else np.nan
    label = f"m/c={ratio:.3f}  p={res.p_value:.1e}" if np.isfinite(ratio) else f"m={res.slope:.3f}  p={res.p_value:.1e}"
    ax.text(text_xy[0], text_xy[1], label, transform=ax.transAxes, fontsize=6, va="top" if text_xy[1]>0.5 else "bottom", ha=text_ha)
    ax.text(0.97, 0.05, f"N={n_total}", transform=ax.transAxes, fontsize=6, ha="right", va="bottom")
    ax.set_xlim(x_min, x_max); ax.tick_params(axis='x', labelsize=6, pad=1); ax.tick_params(axis='y', labelsize=6, pad=1)
    trim_spines(ax)


def _compute_em_hist_cache_corr(em_corr: dict, *, bin_size: float, cache_path: str) -> dict:
    """Compute and cache EM histogram bins and OLS fit for required experimental panels.
    Returns dict keyed by panel name with fields: centers, means, sems, ols{ slope, intercept, p_value }, N.
    """
    from analysis_shared.stats import bin_mean_sem, ols_slope_p
    import numpy as _np
    needed = {
        "E→E": ( ["L2/3","L4","L5","L6"], ["L2/3","L4","L5","L6"] ),
        "L4→L4": ( ["L4"], ["L4"] ),
        "L4→L5": ( ["L4"], ["L5"] ),
        "L2/3→L2/3": ( ["L2/3"], ["L2/3"] ),
    }
    out = {}
    x_min, x_max = -0.2, 0.5
    bins = _np.arange(x_min, x_max + bin_size, bin_size)
    for name, (srcs, tgts) in needed.items():
        xs = []; ys = []; n_total = 0
        for s in srcs:
            for t in tgts:
                df = em_corr.get((s, t))
                if df is None or df.empty:
                    continue
                xv = df["x"].to_numpy(dtype=float)
                yv = df["y"].to_numpy(dtype=float)
                if xv.size and yv.size:
                    xs.append(xv); ys.append(yv); n_total += xv.size
        if not xs:
            continue
        x = _np.concatenate(xs); y = _np.concatenate(ys)
        centers, means, sems = bin_mean_sem(x, y, bins)
        res = ols_slope_p(x, y)
        out[name] = {
            "centers": centers,
            "means": means,
            "sems": sems,
            "ols": {"slope": float(res.slope), "intercept": float(res.intercept), "p_value": float(res.p_value)},
            "N": int(n_total),
        }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out


def main():
    ap = argparse.ArgumentParser(description="Compose final Corr figure (4x4) with caching")
    ap.add_argument("--bases", nargs="*", default=None)
    ap.add_argument("--network-type", default="bio_trained")
    ap.add_argument("--out", default="figures/paper/corr_final_panel.png")
    ap.add_argument("--bin-size", type=float, default=0.05)
    ap.add_argument("--resamples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair-limits-csv", default=None)
    ap.add_argument("--cache-dir", default="figures/paper/cache_corr_final")
    ap.add_argument("--replot-only", action="store_true")
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    apply_pub_style()
    # Force Arial if available
    try:
        import matplotlib as mpl
        from matplotlib import font_manager as _fm
        _ = _fm.findfont("Arial", fallback_to_default=False)
        mpl.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        })
    except Exception:
        pass
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    bases = args.bases if args.bases else [f"core_nll_{i}" for i in range(10) if os.path.isdir(f"core_nll_{i}")]
    if args.pair_limits_csv is None and os.path.isfile("pair_limits_corr_from_em.csv"):
        args.pair_limits_csv = "pair_limits_corr_from_em.csv"

    os.makedirs(args.cache_dir, exist_ok=True)

    # EI cache (full analysis; distinct filename to avoid old pair-limited cache)
    ei_pkl = os.path.join(args.cache_dir, f"corr_ei2x2_full_{args.network_type}.pkl")
    ei_cache = None
    if (not args.force_recompute) and os.path.isfile(ei_pkl):
        try:
            with open(ei_pkl, "rb") as f:
                ei_cache = pickle.load(f)
        except Exception:
            ei_cache = None
    if ei_cache is None and not args.replot_only:
        # Use full analysis (no pair limits) for left histogram panels
        ei_cache = compute_corr_ei2x2_cache(bases, args.network_type, simplify_inh=True, aggregate_l5_types=True, bin_size=args.bin_size, pair_limits_csv=None)
        with open(ei_pkl, "wb") as f:
            pickle.dump(ei_cache, f)

    # Full cache (full analysis; distinct filename to avoid old pair-limited cache)
    full_pkl = os.path.join(args.cache_dir, f"corr_full_full_{args.network_type}.pkl")
    full_cache = None
    if (not args.force_recompute) and os.path.isfile(full_pkl):
        try:
            with open(full_pkl, "rb") as f:
                full_cache = pickle.load(f)
        except Exception:
            full_cache = None
    if full_cache is None and not args.replot_only:
        # Use full analysis (no pair limits) for left histogram panels
        full_cache = compute_corr_full_matrix_cache(bases, args.network_type, simplify_inh=True, aggregate_l5_types=True, bin_size=args.bin_size, pair_limits_csv=None)
        with open(full_pkl, "wb") as f:
            pickle.dump(full_cache, f)

    # EM for small panels and violin overlay
    em_corr = load_em_corr_pickle(os.path.join("analysis_shared", "corr_vs_weight_minnie_250828.pkl"), split_l5=False)

    # Violin long DF from MC (with caching)
    vio_key = f"corr_violin_{args.network_type}_res{args.resamples}_seed{args.seed}_limits{os.path.basename(args.pair_limits_csv) if args.pair_limits_csv else 'none'}.pkl"
    vio_pkl = os.path.join(args.cache_dir, vio_key)
    if (not args.force_recompute) and os.path.isfile(vio_pkl):
        try:
            with open(vio_pkl, 'rb') as f:
                df_long = pickle.load(f)
        except Exception:
            df_long = None
    else:
        df_long = _compute_violin_long_df_corr(bases, args.network_type, pair_limits_csv=args.pair_limits_csv, resamples=args.resamples, seed=args.seed)
        with open(vio_pkl, 'wb') as f:
            pickle.dump(df_long, f)

    # Layout using GridSpec 4x4 for Corr-only half page
    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(nrows=4, ncols=4, left=0.06, right=0.93, top=0.95, bottom=0.09, wspace=0.5, hspace=0.5)

    # Panels 1,2,5,6: 2x2 EI matrix
    sub = gs[:2, :2].subgridspec(2, 2, wspace=0.4, hspace=0.5)
    order = [("E", "E"), ("E", "I"), ("I", "E"), ("I", "I")]
    for idx, (se, te) in enumerate(order):
        i, j = divmod(idx, 2)
        ax = fig.add_subplot(sub[i, j])
        entry = None if ei_cache is None else ei_cache.get("pairs", {}).get((se, te))
        if not entry:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=7, transform=ax.transAxes)
            trim_spines(ax)
        else:
            centers = ei_cache.get("centers")
            _render_corr_hist_fit(
                ax,
                centers=centers,
                means=entry["means"],
                sems=entry["sems"],
                ols=entry["ols"],
                show_p=False,
                n_conn=entry.get("N"), n_fmt_millions=True,
                bar_color="#f4b6c2", bar_alpha=0.35,
                title=f"{se}→{te}",
                text_xy=None,
                xlim=(-0.2, 0.5),
                xticks=[-0.2, 0.0, 0.2, 0.4],
            )
            if j == 0:
                ax.set_ylabel("Weight (pA)", fontsize=7)
    
    # Bottom-left: specific sim pairs (rows 2-3, cols 0-1)
    sub2 = gs[2:, :2].subgridspec(2, 2, wspace=0.4, hspace=0.5)
    sim_pairs = [("L2/3_Exc", "L6_Exc", "L2/3→L6"), ("L4_Exc", "L4_Exc", "L4→L4"), ("PV", "L6_Exc", "PV→L6"), ("L4_Exc", "L5_Exc", "L4→L5")]
    for idx, (s, t, title) in enumerate(sim_pairs):
        i, j = divmod(idx, 2)
        ax = fig.add_subplot(sub2[i, j])
        entry = None if full_cache is None else full_cache.get("pairs", {}).get((s, t))
        if not entry:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            trim_spines(ax)
        else:
            _render_corr_hist_fit(
                ax,
                centers=full_cache.get("centers"),
                means=entry["means"],
                sems=entry["sems"],
                ols=entry["ols"],
                show_p=False,
                n_conn=entry.get("N"), n_fmt_millions=True,
                bar_color="#f4b6c2", bar_alpha=0.35,
                title=title,
                text_xy=None,
                xlim=(-0.2, 0.5),
                xticks=[-0.2, 0.0, 0.2, 0.4],
            )
            if (idx % 2) == 0:
                ax.set_ylabel("Weight (pA)", fontsize=7)
            if idx // 2 == 1:
                ax.set_xlabel("Response correlation", fontsize=7, labelpad=1.0)

    # Panel 3: enlarge effect size heatmap to span top two rows of rightmost column (aggregate L5 Exc)
    ax_eff = fig.add_subplot(gs[:2, 3])
    eff_pkl = os.path.join(args.cache_dir, f"corr_effect_{args.network_type}_inhagg_l5agg.pkl")
    eff = None
    if (not args.force_recompute) and os.path.isfile(eff_pkl):
        try:
            with open(eff_pkl, "rb") as f:
                eff = pickle.load(f)
        except Exception:
            eff = None
    if eff is None and not args.replot_only:
        eff = compute_corr_effect_size(bases, args.network_type, cache_path=eff_pkl, simplify_inh=True, aggregate_l5_types=True)
    types = eff["types"]; A = eff["slope_over_abs_intercept"]
    vals = A[np.isfinite(A)]; vmax = np.percentile(np.abs(vals), 95) if vals.size else 1.0
    def _simplify(lbl: str) -> str:
        m = {"L2/3_Exc":"E23","L4_Exc":"E4","L5_Exc":"E5","L6_Exc":"E6","L1_Inh":"L1"}
        return m.get(lbl, lbl)
    im = ax_eff.imshow(A, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal", interpolation="nearest")
    ax_eff.set_title("m/c (slope/intercept)", fontsize=7, pad=2)
    ax_eff.set_xticks(range(len(types))); ax_eff.set_yticks(range(len(types)))
    ax_eff.set_xticklabels([_simplify(x) for x in types], rotation=90, fontsize=6); ax_eff.set_yticklabels([_simplify(x) for x in types], fontsize=6)
    ax_eff.set_xlabel("Target", fontsize=7, labelpad=2)
    ax_eff.set_ylabel("Source", fontsize=7, labelpad=2)
    cbar = fig.colorbar(im, ax=ax_eff, fraction=0.035, pad=0.02); cbar.ax.tick_params(labelsize=7)
    trim_spines(ax_eff)

    # Panels 7,11,15,? replaced by 4-row experimental panels in right middle column
    # Precompute and cache EM histograms for instant replot
    em_cache_pkl = os.path.join(args.cache_dir, f"corr_em_panels_bins_bin{args.bin_size:.3f}.pkl")
    em_bins = None
    if (not args.force_recompute) and os.path.isfile(em_cache_pkl):
        try:
            with open(em_cache_pkl, "rb") as f:
                em_bins = pickle.load(f)
        except Exception:
            em_bins = None
    if em_bins is None and not args.replot_only:
        em_bins = _compute_em_hist_cache_corr(em_corr, bin_size=args.bin_size, cache_path=em_cache_pkl)

    sub3 = gs[:, 2].subgridspec(4, 1, hspace=0.5)
    exp_titles = ["E→E", "L2/3→L2/3", "L4→L4", "L4→L5"]
    for k, title in enumerate(exp_titles):
        ax = fig.add_subplot(sub3[k, 0])
        data = None if em_bins is None else em_bins.get(title)
        if not data:
            # Fallback compute if cache missing a panel
            if title == "E→E":
                _em_group_hist_corr(ax, em_corr, ["L2/3","L4","L5","L6"], ["L2/3","L4","L5","L6"], bin_size=args.bin_size)
            elif title == "L4→L4":
                _em_group_hist_corr(ax, em_corr, ["L4"], ["L4"], bin_size=args.bin_size)
            elif title == "L4→L5":
                _em_group_hist_corr(ax, em_corr, ["L4"], ["L5"], bin_size=args.bin_size)
            elif title == "L2/3→L2/3":
                _em_group_hist_corr(ax, em_corr, ["L2/3"], ["L2/3"], bin_size=args.bin_size)
        else:
            # Experimental: m/c and p at top; N at bottom; unify x-range and ticks
            _render_corr_hist_fit(
                ax,
                centers=data["centers"],
                means=data["means"],
                sems=data["sems"],
                ols=data["ols"],
                show_p=True,
                n_conn=data.get("N"),
                bar_color="#cfcfcf",
                xlim=(-0.2, 0.5),
                xticks=[-0.2, 0.0, 0.2, 0.4],
                title=title,
                ylabel="Total PSD (µm²)",
                text_xy=(0.03, 0.92),
            )
        if k == 3:
            ax.set_xlabel("Response correlation", fontsize=7, labelpad=1.0)

    # Panels 8,12,16: horizontal violin (rows 2-3, col 3) leaving more room for effect size
    ax_vio = fig.add_subplot(gs[2:, 3])
    # Build EM overlay p-values for individual pairs and aggregated E→E (ensure E→E present)
    from scipy.stats import linregress
    em_overlay: dict[tuple[str,str], float] = {}
    # Individual pairs
    for s in ("L2/3","L4","L5","L6"):
        for t in ("L2/3","L4","L5","L6"):
            df = em_corr.get((s, t))
            if df is None or df.empty:
                continue
            x = df["x"].to_numpy(); y = df["y"].to_numpy()
            if x.size >= 10:
                slope, intercept, r, p, se = linregress(x, y)
                em_overlay[(s, t)] = float(p)
    # Aggregated E→E
    xs=[]; ys=[]
    for s in ("L2/3","L4","L5","L6"):
        for t in ("L2/3","L4","L5","L6"):
            df = em_corr.get((s, t))
            if df is None or df.empty:
                continue
            xs.append(df["x"].to_numpy()); ys.append(df["y"].to_numpy())
    if xs and ys:
        xcat = np.concatenate(xs); ycat = np.concatenate(ys)
        slope, intercept, r, p, se = linregress(xcat, ycat)
        em_overlay[("E","E")] = float(p)
    
    if df_long is not None and not df_long.empty:
        # Determine allowed pairs by requiring at least 100 experimental pairs where available
        allowed = set()
        for s in ("L2/3","L4","L5","L6"):
            for t in ("L2/3","L4","L5","L6"):
                df = em_corr.get((s, t))
                if df is not None and len(df.get("x", [])) >= 100:
                    allowed.add(f"{s}→{t}")
        # Always allow aggregated E→E row
        allowed.add("E→E")
        _plot_corr_violin_on_ax(ax_vio, df_long, em_overlay=em_overlay, allowed_pairs=allowed)
        # Add legend labels for violin (MC) and exp overlay; move to lower right
        from matplotlib.lines import Line2D
        mc_handle = Line2D([0], [0], color="#f4b6c2", lw=6, label="MC (sim)")
        exp_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='#8c8c8c', markeredgecolor='k', markersize=5, label='exp')
        leg = ax_vio.legend(handles=[mc_handle, exp_handle], frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.20, 0.02), borderaxespad=0.0)
    else:
        ax_vio.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_vio.transAxes); trim_spines(ax_vio)

    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
