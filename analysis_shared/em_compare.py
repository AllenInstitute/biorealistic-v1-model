from __future__ import annotations
import os
from typing import Dict, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_shared.style import apply_pub_style, trim_spines
from analysis_shared.stats import bin_mean_sem, fit_cosine_series_deg, ols_slope_p


# ---- EM data loaders ----

def load_em_pd_pickle(path: str) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Load EM PD dict: keys like 'L2/3->L2/3', values DataFrame with columns ['delta_pref_direction','weight'].
    Returns mapping ((source,target)->DataFrame) with normalized labels 'L2/3','L4','L5'."""
    with open(path, 'rb') as f:
        obj = pd.read_pickle(f)  # pickle.load
    if not isinstance(obj, dict):
        raise ValueError("Expected dict in EM PD pickle")
    out: Dict[Tuple[str,str], pd.DataFrame] = {}
    for k, df in obj.items():
        if not isinstance(df, pd.DataFrame):
            continue
        try:
            src, tgt = k.split('->')
        except ValueError:
            continue
        src = src.strip()
        tgt = tgt.strip()
        # Normalize to labels without '_Exc'
        out[(src, tgt)] = df.rename(columns={"delta_pref_direction": "x", "weight": "y"}).copy()
    return out


def load_em_corr_pickle(path: str, *, split_l5: bool = False) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Load EM corr dict: keys like '23P->4P' or '4P->5P-ET', values with ['correlation','weight'].
    If split_l5 is True, map '5P-ET'→'L5_ET', '5P-IT'→'L5_IT', '5P-NP'→'L5_NP'; otherwise map all 5P-*→'L5'."""
    with open(path, 'rb') as f:
        obj = pd.read_pickle(f)
    if not isinstance(obj, dict):
        raise ValueError("Expected dict in EM corr pickle")
    def norm(label: str) -> str:
        s = label.strip()
        if s.startswith('23P'): return 'L2/3'
        if s.startswith('4P'): return 'L4'
        if s.startswith('5P'):
            if not split_l5:
                return 'L5'
            # subtype mapping
            if 'ET' in s: return 'L5_ET'
            if 'IT' in s: return 'L5_IT'
            if 'NP' in s: return 'L5_NP'
            return 'L5'
        if s.startswith('6P'): return 'L6'
        return s
    out: Dict[Tuple[str,str], pd.DataFrame] = {}
    for k, df in obj.items():
        if not isinstance(df, pd.DataFrame):
            continue
        try:
            a, b = k.split('->')
        except ValueError:
            continue
        out[(norm(a), norm(b))] = df.rename(columns={"correlation": "x", "weight": "y"}).copy()
    return out


# ---- EM matrix plots ----

def em_pd_matrix_plot(em_pd: Dict[Tuple[str,str], pd.DataFrame], out_png: str, *, bin_step: float = 20.0) -> None:
    apply_pub_style()
    types = ["L2/3", "L4", "L5"]
    n = len(types)
    fig, axes = plt.subplots(n, n, figsize=(n * 2.2, n * 2.2), sharex=False, sharey=False)
    x_min, x_max = 0.0, 180.0
    bins = np.arange(x_min, x_max + bin_step, bin_step)
    for i, s in enumerate(types):
        for j, t in enumerate(types):
            ax = axes[i, j]
            df = em_pd.get((s+"->"+t,))  # not used
            df = em_pd.get((s, t))
            if df is None or df.empty:
                ax.set_axis_off(); continue
            xx = df["x"].to_numpy(dtype=float)
            yy = df["y"].to_numpy(dtype=float)
            centers, means, sems = bin_mean_sem(xx, yy, bins)
            ax.bar(centers, means, width=bin_step, color="#aec7e8", edgecolor="none")
            ax.errorbar(centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2)
            fit = fit_cosine_series_deg(xx, yy)
            xs = np.linspace(x_min, x_max, 361)
            ys = fit.a * np.cos(np.radians(xs)) + fit.b * np.cos(2 * np.radians(xs)) + fit.c
            ax.plot(xs, ys, color="crimson", linewidth=1.2)
            ax.text(0.03, 0.92, f"a={fit.a:.2f} (p={fit.p_a:.1e})\nb={fit.b:.2f} (p={fit.p_b:.1e})", transform=ax.transAxes, fontsize=7, va="top")
            ax.set_xlim(x_min, x_max)
            if i == n - 1:
                ax.set_xlabel("Pref. dir. diff (deg)")
            if j == 0:
                ax.set_ylabel(s, fontsize=8)
            if i == 0:
                ax.set_title(t, fontsize=8)
            trim_spines(ax)
    plt.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True); plt.savefig(out_png, dpi=150); plt.close()


def em_corr_matrix_plot(em_corr: Dict[Tuple[str,str], pd.DataFrame], out_png: str, *, split_l5: bool = False, bin_size: float = 0.05) -> None:
    apply_pub_style()
    if split_l5:
        types = ["L2/3", "L4", "L5_IT", "L5_ET", "L5_NP", "L6"]
    else:
        types = ["L2/3", "L4", "L5", "L6"]
    n = len(types)
    fig, axes = plt.subplots(n, n, figsize=(n * 2.0, n * 2.0), sharex=False, sharey=False)
    for i, s in enumerate(types):
        for j, t in enumerate(types):
            ax = axes[i, j]
            df = em_corr.get((s, t))
            if df is None or df.empty:
                ax.set_axis_off(); continue
            xx = df["x"].to_numpy(dtype=float)
            yy = df["y"].to_numpy(dtype=float)
            x_min, x_max = np.nanmin(xx), np.nanmax(xx)
            if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
                x_min, x_max = -0.2, 0.5
            bins = np.arange(x_min, x_max + bin_size, bin_size)
            centers, means, sems = bin_mean_sem(xx, yy, bins)
            ax.bar(centers, means, width=(bins[1]-bins[0]) if len(bins)>1 else 0.05, color="#b3b3b3", edgecolor="none")
            ax.errorbar(centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2)
            res = ols_slope_p(xx, yy)
            ax.plot([x_min, x_max], [res.intercept + res.slope * x_min, res.intercept + res.slope * x_max], color="crimson", linewidth=1.2)
            ax.text(0.03, 0.92, f"m={res.slope:.3f}\np={res.p_value:.2e}", transform=ax.transAxes, fontsize=7, va="top")
            ax.set_xlim(x_min, x_max)
            if i == n - 1:
                ax.set_xlabel("Correlation")
            if j == 0:
                ax.set_ylabel(s, fontsize=8)
            if i == 0:
                ax.set_title(t, fontsize=8)
            trim_spines(ax)
    plt.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True); plt.savefig(out_png, dpi=150); plt.close()


# ---- EM PD p-values and violin overlay ----

def compute_em_pd_pvalues(em_pd: Dict[Tuple[str,str], pd.DataFrame]) -> Dict[Tuple[str,str], Dict[str, float]]:
    out: Dict[Tuple[str,str], Dict[str,float]] = {}
    for (s, t), df in em_pd.items():
        xx = df["x"].to_numpy(dtype=float)
        yy = df["y"].to_numpy(dtype=float)
        fit = fit_cosine_series_deg(xx, yy)
        out[(s, t)] = {"p_a": float(fit.p_a), "p_b": float(fit.p_b)}
    return out


def plot_pd_violin_with_em(mc_results: Dict[Tuple[str,str], Dict[str, np.ndarray]], em_pvals: Dict[Tuple[str,str], Dict[str,float]], out_png: str) -> None:
    apply_pub_style()
    types = ["L2/3", "L4", "L5"]
    pairs_full = [(s, t) for s in types for t in types]
    # Build long-form MC dataframe
    rows = []
    for (s, t) in pairs_full:
        d = mc_results.get((f"{s}_Exc" if "_Exc" not in s else s, f"{t}_Exc" if "_Exc" not in t else t)) or mc_results.get((s, t), {})
        for param in ("p_a", "p_b"):
            arr = d.get(param, np.array([]))
            if arr.size:
                z = np.log10(np.clip(arr.astype(float), 1e-12, 1.0))
                for v in z:
                    rows.append({"pair": f"{s}→{t}", "param": param, "log10_p": float(v)})
    if not rows:
        return
    df_long = pd.DataFrame(rows)
    # Only keep pairs that have data (drop empty elements such as missing L4 connections)
    order = [p for p in [f"{s}→{t}" for s, t in pairs_full] if p in set(df_long["pair"].unique())]

    # Smaller overall figure
    fig_width = max(4, 0.7 * len(order))
    fig_height = 2.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    import seaborn as sns
    sns.set(style="ticks")
    v = sns.violinplot(
        data=df_long,
        x="pair",
        y="log10_p",
        hue="param",
        order=order,
        split=True,
        cut=0,
        inner="quartile",
        palette={"p_a": "C0", "p_b": "C3"},
        linewidth=0.8,
        saturation=0.9,
        scale="width",
        ax=ax,
    )
    # p=0.05 horizontal line in log10 space
    thresh = np.log10(0.05)
    ax.axhline(y=thresh, color="gray", linestyle=":", linewidth=1)

    # Overlay EM points and collect one legend handle for "experimental"
    exp_handle = None
    for i, pair_lbl in enumerate(order):
        s, t = pair_lbl.split("→")
        em = em_pvals.get((s, t))
        if not em:
            continue
        if np.isfinite(em.get("p_a", np.nan)):
            ya = np.log10(max(min(em["p_a"], 1.0), 1e-12))
            h = ax.scatter(i - 0.12, ya, facecolors="white", edgecolors="black", s=50, zorder=5, linewidth=0.8)
            exp_handle = exp_handle or h
        if np.isfinite(em.get("p_b", np.nan)):
            yb = np.log10(max(min(em["p_b"], 1.0), 1e-12))
            h = ax.scatter(i + 0.12, yb, facecolors="white", edgecolors="black", s=50, zorder=5, linewidth=0.8)
            exp_handle = exp_handle or h

    ax.set_ylabel("log10(p-value)")
    ax.set_xlabel("Cell-type pair")
    ax.set_xticklabels(order, rotation=30, ha="right")

    # Compose legend: keep p_a/p_b entries, add experimental dot
    handles, labels = ax.get_legend_handles_labels()
    if exp_handle is not None:
        handles.append(exp_handle)
        labels.append("experimental")
    if handles:
        ax.legend(handles, labels, title="", fontsize=8, loc="lower right")
    trim_spines(ax)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
