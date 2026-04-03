from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Sequence

from analysis_shared.grouping import aggregate_l5, apply_inh_simplification
from analysis_shared.grouping import filter_inh_respective_layer
from analysis_shared.stats import bin_mean_sem, fit_cosine_series_deg
from analysis_shared.io import load_edges_with_pref_dir, load_edges_with_computed_pref_dir
from analysis_shared.sampling import apply_per_pair_sampling, read_pair_limits_csv
from analysis_shared.style import apply_pub_style, trim_spines

EXC_ALL = ["L2/3_Exc", "L4_Exc", "L5_IT", "L5_ET", "L5_NP", "L6_Exc"]
EXC_L5_AGG = ["L2/3_Exc", "L4_Exc", "L5_Exc", "L6_Exc"]


def pd_exc_matrix_plot(
    base_dirs: Sequence[str],
    network_type: str,
    out_png: str,
    *,
    aggregate_l5_types: bool = True,
    bin_step: float = 20.0,
    max_per_pair: int | None = None,
    pair_limits_csv: str | None = None,
    sample_seed: int = 0,
    loader=None,
) -> None:
    if loader is None:
        loader = load_edges_with_pref_dir
    apply_pub_style()
    # Concatenate PD edges across bases
    dfs = []
    for bd in base_dirs:
        e = loader(bd, network_type)
        # Attach types if available via response_correlation pipeline
        try:
            from aggregate_correlation_plot import process_network_data

            typed = process_network_data((bd, network_type))
            typed = typed[["source_id", "target_id", "source_type", "target_type"]]
            e = e.merge(typed, on=["source_id", "target_id"], how="left")
        except Exception:
            pass
        dfs.append(e)
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["source_type", "target_type"])  # ensure types present
    if aggregate_l5_types:
        df = aggregate_l5(df)
        exc_types = EXC_L5_AGG
    else:
        exc_types = EXC_ALL

    df = df[df["source_type"].isin(exc_types) & df["target_type"].isin(exc_types)]

    # Reduced sampling
    limits = read_pair_limits_csv(pair_limits_csv) if pair_limits_csv else None
    if (max_per_pair is not None) or (limits is not None):
        df = apply_per_pair_sampling(
            df,
            max_per_pair=max_per_pair,
            pair_limits=limits,
            rng=np.random.RandomState(sample_seed),
        )

    n = len(exc_types)
    fig, axes = plt.subplots(
        n, n, figsize=(n * 2.2, n * 2.2), sharex=False, sharey=False
    )
    if n == 1:
        axes = np.array([[axes]])

    x_min, x_max = 0.0, 180.0
    bins = np.arange(x_min, x_max + bin_step, bin_step)

    for i, s in enumerate(exc_types):
        for j, t in enumerate(exc_types):
            ax = axes[i, j]
            sub = df[(df["source_type"] == s) & (df["target_type"] == t)][
                ["pref_dir_diff_deg", "syn_weight"]
            ].dropna()
            if sub.empty or len(sub) < 2:
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
            else:
                xx = sub["pref_dir_diff_deg"].to_numpy()
                yy = sub["syn_weight"].to_numpy()
                centers, means, sems = bin_mean_sem(xx, yy, bins)
                ax.bar(
                    centers, means, width=bin_step, color="#aec7e8", edgecolor="none"
                )
                ax.errorbar(
                    centers,
                    means,
                    yerr=sems,
                    fmt="none",
                    ecolor="k",
                    elinewidth=1,
                    capsize=2,
                )
                # Cosine-series fit
                fit = fit_cosine_series_deg(xx, yy)
                xs = np.linspace(x_min, x_max, 361)
                ys = (
                    fit.a * np.cos(np.radians(xs))
                    + fit.b * np.cos(2 * np.radians(xs))
                    + fit.c
                )
                ax.plot(xs, ys, color="crimson", linewidth=1.2)
                ax.text(
                    0.03,
                    0.92,
                    f"a={fit.a:.2f} (p={fit.p_a:.1e})\nb={fit.b:.2f} (p={fit.p_b:.1e})",
                    transform=ax.transAxes,
                    fontsize=7,
                    va="top",
                )
                ax.set_xlim(x_min, x_max)
            if i == n - 1:
                ax.set_xlabel("Pref. dir. diff (deg)")
            if j == 0:
                ax.set_ylabel(s, fontsize=8)
            if i == 0:
                ax.set_title(t, fontsize=8)
            trim_spines(ax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def pd_full_matrix_plot(
    base_dirs: Sequence[str],
    network_type: str,
    out_png: str,
    *,
    simplify_inh: bool = True,
    inh_respective_layer: bool = False,
    aggregate_l5_types: bool = True,
    bin_step: float = 20.0,
    max_per_pair: int | None = None,
    pair_limits_csv: str | None = None,
    sample_seed: int = 0,
    loader=None,
) -> None:
    """Plot a full matrix (histogram + cosine fit) for simulation across
    all Exc types plus aggregated inhibitory types (PV/SST/VIP, L1_Inh if present).

    Matches the cell-type set used in the effect-size figure when simplify_inh=True
    and aggregate_l5_types=True.
    """
    if loader is None:
        loader = load_edges_with_pref_dir
    apply_pub_style()

    # Concatenate PD edges across bases
    dfs = []
    for bd in base_dirs:
        e = loader(bd, network_type)
        # Attach types if available via response_correlation pipeline
        try:
            from aggregate_correlation_plot import process_network_data

            typed = process_network_data((bd, network_type))
            typed = typed[["source_id", "target_id", "source_type", "target_type"]]
            e = e.merge(typed, on=["source_id", "target_id"], how="left")
        except Exception:
            pass
        dfs.append(e)
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["source_type", "target_type"])  # ensure types present

    # Apply grouping options
    if simplify_inh and inh_respective_layer:
        df = filter_inh_respective_layer(df)
    if simplify_inh:
        df = apply_inh_simplification(df)
    if aggregate_l5_types:
        df = aggregate_l5(df)

    # Determine allowed types: Exc set + aggregated inhibitory labels found in data
    exc_types = EXC_L5_AGG if aggregate_l5_types else EXC_ALL
    inh_base = ["PV", "SST", "VIP", "L1_Inh"]
    present_types = sorted(
        set(df["source_type"].astype(str)).union(set(df["target_type"].astype(str)))
    )
    inh_types = [t for t in inh_base if t in present_types]
    types = exc_types + [t for t in inh_types if t not in exc_types]

    # Filter to allowed cross pairs
    df = df[df["source_type"].isin(types) & df["target_type"].isin(types)]

    # Optional per-pair sampling
    limits = read_pair_limits_csv(pair_limits_csv) if pair_limits_csv else None
    if (max_per_pair is not None) or (limits is not None):
        df = apply_per_pair_sampling(
            df,
            max_per_pair=max_per_pair,
            pair_limits=limits,
            rng=np.random.RandomState(sample_seed),
        )

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

    # Slightly smaller per-panel size to keep figure manageable
    fig, axes = plt.subplots(
        n, n, figsize=(n * 1.6, n * 1.6), sharex=False, sharey=False
    )
    if n == 1:
        axes = np.array([[axes]])

    x_min, x_max = 0.0, 180.0
    bins = np.arange(x_min, x_max + bin_step, bin_step)

    for i, s in enumerate(types):
        for j, t in enumerate(types):
            ax = axes[i, j]
            sub = df[(df["source_type"] == s) & (df["target_type"] == t)][
                ["pref_dir_diff_deg", "syn_weight"]
            ].dropna()
            if sub.empty or len(sub) < 2:
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
            else:
                xx = sub["pref_dir_diff_deg"].to_numpy()
                yy = sub["syn_weight"].to_numpy()
                centers, means, sems = bin_mean_sem(xx, yy, bins)
                ax.bar(
                    centers, means, width=bin_step, color="#aec7e8", edgecolor="none"
                )
                ax.errorbar(
                    centers,
                    means,
                    yerr=sems,
                    fmt="none",
                    ecolor="k",
                    elinewidth=1,
                    capsize=2,
                )
                fit = fit_cosine_series_deg(xx, yy)
                xs = np.linspace(x_min, x_max, 361)
                ys = (
                    fit.a * np.cos(np.radians(xs))
                    + fit.b * np.cos(2 * np.radians(xs))
                    + fit.c
                )
                ax.plot(xs, ys, color="crimson", linewidth=1.2)
                ax.text(
                    0.03,
                    0.92,
                    f"a={fit.a:.2f} (p={fit.p_a:.1e})\nb={fit.b:.2f} (p={fit.p_b:.1e})",
                    transform=ax.transAxes,
                    fontsize=7,
                    va="top",
                )
                ax.set_xlim(x_min, x_max)
            if i == n - 1:
                ax.set_xlabel("Pref. dir. diff (deg)")
            if j == 0:
                ax.set_ylabel(s, fontsize=8)
            if i == 0:
                ax.set_title(t, fontsize=8)
            trim_spines(ax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def compute_pd_full_matrix_cache(
    base_dirs: Sequence[str],
    network_type: str,
    *,
    simplify_inh: bool = True,
    inh_respective_layer: bool = False,
    aggregate_l5_types: bool = True,
    bin_step: float = 20.0,
    max_per_pair: int | None = None,
    pair_limits_csv: str | None = None,
    sample_seed: int = 0,
    loader=None,
) -> dict:
    """Compute and return compact plotting data for the full PD matrix.

    Returns a dict with keys:
      - 'types': ordered list of cell types
      - 'bin_step': float
      - 'centers': np.ndarray of bin centers
      - 'pairs': dict[(s,t)] -> { 'means', 'sems', 'fit': {'a','b','c','p_a','p_b'} }
    """
    if loader is None:
        loader = load_edges_with_pref_dir
    # Load and type-annotate edges
    dfs = []
    for bd in base_dirs:
        e = loader(bd, network_type)
        try:
            from aggregate_correlation_plot import process_network_data

            typed = process_network_data((bd, network_type))
            typed = typed[["source_id", "target_id", "source_type", "target_type"]]
            e = e.merge(typed, on=["source_id", "target_id"], how="left")
        except Exception:
            pass
        dfs.append(e)
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["source_type", "target_type"]).copy()

    if simplify_inh and inh_respective_layer:
        df = filter_inh_respective_layer(df)
    if simplify_inh:
        df = apply_inh_simplification(df)
    if aggregate_l5_types:
        df = aggregate_l5(df)

    exc_types = EXC_L5_AGG if aggregate_l5_types else EXC_ALL
    present_types = sorted(
        set(df["source_type"].astype(str)).union(set(df["target_type"].astype(str)))
    )
    inh_base = ["PV", "SST", "VIP", "L1_Inh"]
    inh_types = [t for t in inh_base if t in present_types]
    types = exc_types + [t for t in inh_types if t not in exc_types]

    df = df[df["source_type"].isin(types) & df["target_type"].isin(types)]

    limits = read_pair_limits_csv(pair_limits_csv) if pair_limits_csv else None
    if (max_per_pair is not None) or (limits is not None):
        df = apply_per_pair_sampling(
            df,
            max_per_pair=max_per_pair,
            pair_limits=limits,
            rng=np.random.RandomState(sample_seed),
        )

    x_min, x_max = 0.0, 180.0
    bins = np.arange(x_min, x_max + bin_step, bin_step)
    centers = bin_mean_sem(np.array([0.0]), np.array([0.0]), bins)[0]  # centers only

    pairs = {}
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
            cts, means, sems = bin_mean_sem(xx, yy, bins)
            fit = fit_cosine_series_deg(xx, yy)
            pairs[(s, t)] = {
                "N": int(len(xx)),
                "means": means,
                "sems": sems,
                "fit": {
                    "a": fit.a,
                    "b": fit.b,
                    "c": fit.c,
                    "p_a": fit.p_a,
                    "p_b": fit.p_b,
                },
            }

    return {
        "types": types,
        "bin_step": float(bin_step),
        "centers": centers,
        "pairs": pairs,
    }


def plot_pd_full_matrix_from_cache(cache_data: dict, out_png: str) -> None:
    """Render the full PD matrix figure from cached data."""
    apply_pub_style()

    types = cache_data.get("types", [])
    centers = cache_data.get("centers")
    pairs = cache_data.get("pairs", {})
    bin_step = float(cache_data.get("bin_step", 20.0))

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

    fig, axes = plt.subplots(
        n, n, figsize=(n * 1.6, n * 1.6), sharex=False, sharey=False
    )
    if n == 1:
        axes = np.array([[axes]])

    x_min, x_max = 0.0, 180.0
    xs = np.linspace(x_min, x_max, 361)

    for i, s in enumerate(types):
        for j, t in enumerate(types):
            ax = axes[i, j]
            entry = pairs.get((s, t))
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
                ax.set_xlim(x_min, x_max)
            else:
                means = entry["means"]
                sems = entry["sems"]
                fit = entry["fit"]
                ax.bar(
                    centers, means, width=bin_step, color="#aec7e8", edgecolor="none"
                )
                ax.errorbar(
                    centers,
                    means,
                    yerr=sems,
                    fmt="none",
                    ecolor="k",
                    elinewidth=1,
                    capsize=2,
                )
                ys = (
                    fit["a"] * np.cos(np.radians(xs))
                    + fit["b"] * np.cos(2 * np.radians(xs))
                    + fit["c"]
                )
                ax.plot(xs, ys, color="crimson", linewidth=1.2)
                ax.text(
                    0.03,
                    0.92,
                    f"a={fit['a']:.2f} (p={fit['p_a']:.1e})\nb={fit['b']:.2f} (p={fit['p_b']:.1e})",
                    transform=ax.transAxes,
                    fontsize=7,
                    va="top",
                )
                ax.set_xlim(x_min, x_max)
            if i == n - 1:
                ax.set_xlabel("Pref. dir. diff (deg)")
            if j == 0:
                ax.set_ylabel(s, fontsize=8)
            if i == 0:
                ax.set_title(t, fontsize=8)
            trim_spines(ax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def compute_pd_ei2x2_cache(
    base_dirs: Sequence[str],
    network_type: str,
    *,
    simplify_inh: bool = True,
    inh_respective_layer: bool = False,
    aggregate_l5_types: bool = True,
    bin_step: float = 20.0,
    max_per_pair: int | None = None,
    pair_limits_csv: str | None = None,
    sample_seed: int = 0,
    loader=None,
) -> dict:
    """Compute cached data for a 2x2 E/I matrix (E→E, E→I, I→E, I→I)."""
    if loader is None:
        loader = load_edges_with_pref_dir
    # Load and type-annotate edges
    dfs = []
    for bd in base_dirs:
        e = loader(bd, network_type)
        try:
            from aggregate_correlation_plot import process_network_data

            typed = process_network_data((bd, network_type))
            typed = typed[["source_id", "target_id", "source_type", "target_type"]]
            e = e.merge(typed, on=["source_id", "target_id"], how="left")
        except Exception:
            pass
        dfs.append(e)
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["source_type", "target_type"]).copy()

    if simplify_inh and inh_respective_layer:
        df = filter_inh_respective_layer(df)
    if simplify_inh:
        df = apply_inh_simplification(df)
    if aggregate_l5_types:
        df = aggregate_l5(df)

    # Map to E/I
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
        # sampling by original types is fine; it reduces overall connections
        df = apply_per_pair_sampling(
            df,
            max_per_pair=max_per_pair,
            pair_limits=limits,
            rng=np.random.RandomState(sample_seed),
        )

    x_min, x_max = 0.0, 180.0
    bins = np.arange(x_min, x_max + bin_step, bin_step)
    centers = bin_mean_sem(np.array([0.0]), np.array([0.0]), bins)[0]

    order = [("E", "E"), ("E", "I"), ("I", "E"), ("I", "I")]
    pairs = {}
    for se, te in order:
        sub = df[(df["src_ei"] == se) & (df["tgt_ei"] == te)][
            ["pref_dir_diff_deg", "syn_weight"]
        ].dropna()
        if sub.empty or len(sub) < 2:
            pairs[(se, te)] = None
            continue
        xx = sub["pref_dir_diff_deg"].to_numpy()
        yy = sub["syn_weight"].to_numpy()
        cts, means, sems = bin_mean_sem(xx, yy, bins)
        fit = fit_cosine_series_deg(xx, yy)
        pairs[(se, te)] = {
            "N": int(len(xx)),
            "means": means,
            "sems": sems,
            "fit": {"a": fit.a, "b": fit.b, "c": fit.c, "p_a": fit.p_a, "p_b": fit.p_b},
        }

    return {
        "bin_step": float(bin_step),
        "centers": centers,
        "pairs": pairs,
        "order": order,
    }


def plot_pd_ei2x2_from_cache(cache_data: dict, out_png: str) -> None:
    apply_pub_style()
    centers = cache_data.get("centers")
    pairs = cache_data.get("pairs", {})
    order = cache_data.get("order", [("E", "E"), ("E", "I"), ("I", "E"), ("I", "I")])
    bin_step = float(cache_data.get("bin_step", 20.0))

    fig, axes = plt.subplots(
        2, 2, figsize=(3.8, 3.8), sharex=False, sharey=False, constrained_layout=True
    )
    x_min, x_max = 0.0, 180.0
    xs = np.linspace(x_min, x_max, 361)

    for idx, (se, te) in enumerate(order):
        i, j = divmod(idx, 2)
        ax = axes[i, j]
        entry = pairs.get((se, te))
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
            ax.set_xlim(x_min, x_max)
        else:
            means = entry["means"]
            sems = entry["sems"]
            fit = entry["fit"]
            ax.bar(centers, means, width=bin_step, color="#aec7e8", edgecolor="none")
            ax.errorbar(
                centers,
                means,
                yerr=sems,
                fmt="none",
                ecolor="k",
                elinewidth=1,
                capsize=2,
            )
            ys = (
                fit["a"] * np.cos(np.radians(xs))
                + fit["b"] * np.cos(2 * np.radians(xs))
                + fit["c"]
            )
            ax.plot(xs, ys, color="crimson", linewidth=1.0)
            ax.text(
                0.03,
                0.92,
                f"a={fit['a']:.2f}\nb={fit['b']:.2f}",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
            )
            ax.set_xlim(x_min, x_max)
        ax.set_title(f"{se}→{te}", fontsize=9)
        if i == 1:
            ax.set_xlabel("Pref. dir. diff (deg)")
        if j == 0:
            ax.set_ylabel("Weight")
        trim_spines(ax)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
