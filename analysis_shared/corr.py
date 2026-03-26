from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence

from analysis_shared.grouping import aggregate_l5, apply_inh_simplification
from analysis_shared.stats import bin_mean_sem, ols_slope_p
from aggregate_correlation_plot import process_network_data
from analysis_shared.sampling import apply_per_pair_sampling, read_pair_limits_csv
from analysis_shared.style import apply_pub_style, trim_spines


EXC_L5 = ["L5_IT", "L5_ET", "L5_NP"]
EXC_ALL = ["L2/3_Exc", "L4_Exc", "L5_IT", "L5_ET", "L5_NP", "L6_Exc"]
EXC_L5_AGG = ["L2/3_Exc", "L4_Exc", "L5_Exc", "L6_Exc"]


def corr_exc_matrix_plot(base_dirs: Sequence[str], network_type: str, out_png: str, *, aggregate_l5_types: bool = True, bin_size: float = 0.05, max_per_pair: int | None = None, pair_limits_csv: str | None = None, sample_seed: int = 0) -> None:
    apply_pub_style()
    # Load and concatenate
    edge_dfs = [process_network_data((bd, network_type)) for bd in base_dirs]
    df = pd.concat(edge_dfs, ignore_index=True)
    if aggregate_l5_types:
        df = aggregate_l5(df)
        exc_types = EXC_L5_AGG
    else:
        exc_types = EXC_ALL

    # Filter to excitatory pairs
    df = df[df["source_type"].isin(exc_types) & df["target_type"].isin(exc_types)]

    # Reduced sampling
    limits = read_pair_limits_csv(pair_limits_csv) if pair_limits_csv else None
    if (max_per_pair is not None) or (limits is not None):
        df = apply_per_pair_sampling(df, max_per_pair=max_per_pair, pair_limits=limits, rng=np.random.RandomState(sample_seed))

    n = len(exc_types)
    fig, axes = plt.subplots(n, n, figsize=(n * 2.2, n * 2.2), sharex=False, sharey=False)
    if n == 1:
        axes = np.array([[axes]])

    x_min, x_max = -0.2, 0.5
    bins = np.arange(x_min, x_max + bin_size, bin_size)

    for i, s in enumerate(exc_types):
        for j, t in enumerate(exc_types):
            ax = axes[i, j]
            sub = df[(df["source_type"] == s) & (df["target_type"] == t)][["Response Correlation", "syn_weight"]].dropna()
            if sub.empty or len(sub) < 2:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=7, transform=ax.transAxes)
                ax.set_xlim(x_min, x_max)
            else:
                xx = sub["Response Correlation"].to_numpy()
                yy = sub["syn_weight"].to_numpy()
                mask = (xx >= x_min) & (xx <= x_max)
                xx = xx[mask]; yy = yy[mask]
                centers, means, sems = bin_mean_sem(xx, yy, bins)
                ax.bar(centers, means, width=bin_size, color="#b3b3b3", edgecolor="none")
                ax.errorbar(centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2)
                res = ols_slope_p(xx, yy)
                line_x = np.array([x_min, x_max])
                ax.plot(line_x, res.intercept + res.slope * line_x, color="crimson", linewidth=1.2)
                sig = "" if not np.isfinite(res.p_value) else ("***" if res.p_value < 1e-3 else "**" if res.p_value < 1e-2 else "*" if res.p_value < 5e-2 else "")
                ax.text(0.03, 0.92, f"m={res.slope:.3f} {sig}\np={res.p_value:.2e}", transform=ax.transAxes, fontsize=7, va="top")
                ax.set_xlim(x_min, x_max)
            if i == n - 1:
                ax.set_xlabel("Res. corr.")
            if j == 0:
                ax.set_ylabel(s, fontsize=8)
            if i == 0:
                ax.set_title(t, fontsize=8)
            trim_spines(ax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def compute_corr_full_matrix_cache(
    base_dirs: Sequence[str],
    network_type: str,
    *,
    simplify_inh: bool = True,
    aggregate_l5_types: bool = True,
    bin_size: float = 0.05,
    x_min: float = -0.2,
    x_max: float = 0.5,
    max_per_pair: int | None = None,
    pair_limits_csv: str | None = None,
    sample_seed: int = 0,
) -> dict:
    """Compute cached plotting data for correlation full-matrix.

    Returns dict with keys:
      - 'types': cell types (Exc +/- aggregated inhibitory present)
      - 'centers': bin centers
      - 'bin_size', 'x_min', 'x_max'
      - 'pairs': dict[(s,t)] -> { 'means','sems','ols': {'slope','intercept','p_value'} }
    """
    edge_dfs = [process_network_data((bd, network_type)) for bd in base_dirs]
    df = pd.concat(edge_dfs, ignore_index=True)

    if simplify_inh:
        df = apply_inh_simplification(df)
    if aggregate_l5_types:
        df = aggregate_l5(df)

    exc_types = EXC_L5_AGG if aggregate_l5_types else EXC_ALL
    present_types = sorted(set(df["source_type"].astype(str)).union(set(df["target_type"].astype(str))))
    inh_base = ["PV", "SST", "VIP", "L1_Inh"]
    inh_types = [t for t in inh_base if t in present_types]
    types = exc_types + [t for t in inh_types if t not in exc_types]

    df = df[df["source_type"].isin(types) & df["target_type"].isin(types)]

    limits = read_pair_limits_csv(pair_limits_csv) if pair_limits_csv else None
    if (max_per_pair is not None) or (limits is not None):
        df = apply_per_pair_sampling(df, max_per_pair=max_per_pair, pair_limits=limits, rng=np.random.RandomState(sample_seed))

    bins = np.arange(x_min, x_max + bin_size, bin_size)
    centers = bin_mean_sem(np.array([x_min]), np.array([0.0]), bins)[0]

    pairs = {}
    for s in types:
        for t in types:
            sub = df[(df["source_type"] == s) & (df["target_type"] == t)][["Response Correlation", "syn_weight"]].dropna()
            if sub.empty or len(sub) < 2:
                pairs[(s, t)] = None
                continue
            xx = sub["Response Correlation"].to_numpy()
            yy = sub["syn_weight"].to_numpy()
            mask = (xx >= x_min) & (xx <= x_max)
            xx = xx[mask]; yy = yy[mask]
            cts, means, sems = bin_mean_sem(xx, yy, bins)
            res = ols_slope_p(xx, yy)
            pairs[(s, t)] = {
                "means": means,
                "sems": sems,
                "ols": {"slope": res.slope, "intercept": res.intercept, "p_value": res.p_value},
                "N": int(xx.size),
            }

    return {"types": types, "centers": centers, "bin_size": float(bin_size), "x_min": float(x_min), "x_max": float(x_max), "pairs": pairs}


def plot_corr_full_matrix_from_cache(cache_data: dict, out_png: str) -> None:
    apply_pub_style()
    types = cache_data.get("types", [])
    centers = cache_data.get("centers")
    pairs = cache_data.get("pairs", {})
    bin_size = float(cache_data.get("bin_size", 0.05))
    x_min = float(cache_data.get("x_min", -0.2))
    x_max = float(cache_data.get("x_max", 0.5))

    n = len(types)
    if n == 0:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        trim_spines(ax)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return

    fig, axes = plt.subplots(n, n, figsize=(n * 1.6, n * 1.6), sharex=False, sharey=False)
    if n == 1:
        axes = np.array([[axes]])

    line_x = np.array([x_min, x_max])

    for i, s in enumerate(types):
        for j, t in enumerate(types):
            ax = axes[i, j]
            entry = pairs.get((s, t))
            if not entry:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=7, transform=ax.transAxes)
                ax.set_xlim(x_min, x_max)
            else:
                means = entry["means"]
                sems = entry["sems"]
                ols = entry["ols"]
                ax.bar(centers, means, width=bin_size, color="#b3b3b3", edgecolor="none")
                ax.errorbar(centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2)
                ax.plot(line_x, ols["intercept"] + ols["slope"] * line_x, color="crimson", linewidth=1.2)
                p = ols["p_value"]
                sig = "" if not np.isfinite(p) else ("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "")
                ax.text(0.03, 0.92, f"m={ols['slope']:.3f} {sig}\np={p:.2e}", transform=ax.transAxes, fontsize=7, va="top")
                ax.set_xlim(x_min, x_max)
            if i == n - 1:
                ax.set_xlabel("Res. corr.")
            if j == 0:
                ax.set_ylabel(s, fontsize=8)
            if i == 0:
                ax.set_title(t, fontsize=8)
            trim_spines(ax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def compute_corr_ei2x2_cache(
    base_dirs: Sequence[str],
    network_type: str,
    *,
    simplify_inh: bool = True,
    aggregate_l5_types: bool = True,
    bin_size: float = 0.05,
    x_min: float = -0.2,
    x_max: float = 0.5,
    max_per_pair: int | None = None,
    pair_limits_csv: str | None = None,
    sample_seed: int = 0,
) -> dict:
    edge_dfs = [process_network_data((bd, network_type)) for bd in base_dirs]
    df = pd.concat(edge_dfs, ignore_index=True)

    if simplify_inh:
        df = apply_inh_simplification(df)
    if aggregate_l5_types:
        df = aggregate_l5(df)

    def to_ei(label: str) -> str:
        s = str(label)
        if ("Exc" in s) or (s in ("L5_IT", "L5_ET", "L5_NP")):
            return "E"
        return "I"

    df["src_ei"] = df["source_type"].astype(str).map(to_ei)
    df["tgt_ei"] = df["target_type"].astype(str).map(to_ei)
    df = df[df["src_ei"].isin(["E", "I"]) & df["tgt_ei"].isin(["E", "I"])].copy()

    limits = read_pair_limits_csv(pair_limits_csv) if pair_limits_csv else None
    if (max_per_pair is not None) or (limits is not None):
        df = apply_per_pair_sampling(df, max_per_pair=max_per_pair, pair_limits=limits, rng=np.random.RandomState(sample_seed))

    bins = np.arange(x_min, x_max + bin_size, bin_size)
    centers = bin_mean_sem(np.array([x_min]), np.array([0.0]), bins)[0]

    order = [("E", "E"), ("E", "I"), ("I", "E"), ("I", "I")]
    pairs = {}
    for (se, te) in order:
        sub = df[(df["src_ei"] == se) & (df["tgt_ei"] == te)][["Response Correlation", "syn_weight"]].dropna()
        if sub.empty or len(sub) < 2:
            pairs[(se, te)] = None
            continue
        xx = sub["Response Correlation"].to_numpy()
        yy = sub["syn_weight"].to_numpy()
        mask = (xx >= x_min) & (xx <= x_max)
        xx = xx[mask]; yy = yy[mask]
        cts, means, sems = bin_mean_sem(xx, yy, bins)
        res = ols_slope_p(xx, yy)
        pairs[(se, te)] = {
            "means": means,
            "sems": sems,
            "ols": {"slope": res.slope, "intercept": res.intercept, "p_value": res.p_value},
            "N": int(xx.size),
        }

    return {"centers": centers, "bin_size": float(bin_size), "x_min": float(x_min), "x_max": float(x_max), "pairs": pairs, "order": order}


def plot_corr_ei2x2_from_cache(cache_data: dict, out_png: str) -> None:
    apply_pub_style()
    centers = cache_data.get("centers")
    pairs = cache_data.get("pairs", {})
    order = cache_data.get("order", [("E", "E"), ("E", "I"), ("I", "E"), ("I", "I")])
    bin_size = float(cache_data.get("bin_size", 0.05))
    x_min = float(cache_data.get("x_min", -0.2))
    x_max = float(cache_data.get("x_max", 0.5))

    fig, axes = plt.subplots(2, 2, figsize=(3.8, 3.8), sharex=False, sharey=False, constrained_layout=True)
    line_x = np.array([x_min, x_max])

    for idx, (se, te) in enumerate(order):
        i, j = divmod(idx, 2)
        ax = axes[i, j]
        entry = pairs.get((se, te))
        if not entry:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=7, transform=ax.transAxes)
        else:
            means = entry["means"]
            sems = entry["sems"]
            ols = entry["ols"]
            ax.bar(centers, means, width=bin_size, color="#b3b3b3", edgecolor="none")
            ax.errorbar(centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2)
            ax.plot(line_x, ols["intercept"] + ols["slope"] * line_x, color="crimson", linewidth=1.0)
            p = ols["p_value"]
            sig = "" if not np.isfinite(p) else ("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "")
            ax.text(0.03, 0.92, f"m={ols['slope']:.3f} {sig}\np={p:.2e}", transform=ax.transAxes, fontsize=7, va="top")
        ax.set_title(f"{se}→{te}", fontsize=9)
        if i == 1:
            ax.set_xlabel("Res. corr.")
        if j == 0:
            ax.set_ylabel("Weight")
        trim_spines(ax)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
