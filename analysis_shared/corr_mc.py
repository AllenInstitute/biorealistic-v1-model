from __future__ import annotations
import os
from typing import Dict, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_shared.style import apply_pub_style, trim_spines
from analysis_shared.grouping import aggregate_l5

# Default focus when not provided
DEFAULT_FOCUS = ("L2/3_Exc", "L4_Exc", "L5_Exc", "L6_Exc")


def compute_corr_mc_pvalues(bases: Sequence[str], network_type: str, *, resamples: int, connections_per_draw: int | None, seed: int = 0, pair_limits_csv: str | None = None, focus_types: Sequence[str] | None = None) -> Dict[Tuple[str, str], np.ndarray]:
    apply_pub_style()
    rng = np.random.RandomState(seed)

    # Lazy import to avoid circular deps
    from analysis_shared.sampling import read_pair_limits_csv, apply_per_pair_sampling
    from aggregate_correlation_plot import process_network_data
    from analysis_shared.stats import ols_slope_p

    limits = read_pair_limits_csv(pair_limits_csv) if pair_limits_csv else None

    focus = list(focus_types) if focus_types else list(DEFAULT_FOCUS)
    pairs = [(s, t) for s in focus for t in focus]

    # Load and concatenate
    dfs = []
    for bd in bases:
        df = process_network_data((bd, network_type))
        dfs.append(df[["source_type", "target_type", "Response Correlation", "syn_weight"]].copy())
    df_all = pd.concat(dfs, ignore_index=True)

    # Aggregate L5 subtypes if aggregated focus is requested
    if any(ct == "L5_Exc" for ct in focus):
        df_all = aggregate_l5(df_all)

    if (connections_per_draw is not None) or (limits is not None):
        df_all = apply_per_pair_sampling(df_all, max_per_pair=connections_per_draw, pair_limits=limits, rng=rng)

    results: Dict[Tuple[str, str], np.ndarray] = {}
    min_n = 30  # enforce at least 30 connections per pair
    # Match earlier behavior: restrict regression to a stable correlation range
    x_min, x_max = -0.2, 0.5
    for (s, t) in pairs:
        sub = df_all[(df_all["source_type"] == s) & (df_all["target_type"] == t)][["Response Correlation", "syn_weight"]].dropna()
        if not sub.empty:
            sub = sub[(sub["Response Correlation"] >= x_min) & (sub["Response Correlation"] <= x_max)]
        n = len(sub)
        if sub.empty or n < min_n:
            print(f"[mc-corr] skip {s}->{t}: N={n} (<{min_n})")
            continue
        x = sub["Response Correlation"].to_numpy()
        y = sub["syn_weight"].to_numpy()
        pvals = []
        for _ in range(resamples):
            idx = rng.choice(n, size=n, replace=True)
            xb = x[idx]
            yb = y[idx]
            res = ols_slope_p(xb, yb)
            pvals.append(res.p_value)
        if pvals:
            results[(s, t)] = np.array(pvals)
            print(f"[mc-corr] kept {s}->{t}: N={n}, resamples={resamples}")
    return results


def plot_corr_violin_with_em(mc_results: Dict[Tuple[str, str], np.ndarray], em_pvals: Dict[Tuple[str, str], float], out_png: str, *, focus_types: Sequence[str] | None = None) -> None:
    apply_pub_style()
    focus = list(focus_types) if focus_types else list(DEFAULT_FOCUS)
    pairs_full = [(s, t) for s in focus for t in focus]

    # Detect if subtypes are present in focus
    has_l5_subtypes = any(x in ('L5_IT','L5_ET','L5_NP') for x in focus)

    def simple(lbl: str) -> str:
        if has_l5_subtypes:
            return lbl.replace('_Exc','')  # keep L5_IT/L5_ET/L5_NP intact
        return lbl.replace('_Exc','').replace('L5_IT','L5').replace('L5_ET','L5').replace('L5_NP','L5')

    rows = []
    for (s, t) in pairs_full:
        arr = mc_results.get((s, t), np.array([]))
        # require EM overlay exists for this aggregated or split pair
        if arr.size:
            # Use log10(p) to invert orientation (more negative = more significant)
            z = np.log10(np.clip(arr, 1e-12, 1.0))
            for v in z:
                rows.append({"pair": f"{simple(s)}→{simple(t)}", "log10_p": float(v)})
    if not rows:
        return
    df_long = pd.DataFrame(rows)
    order = [f"{simple(s)}→{simple(t)}" for s in focus for t in focus if f"{simple(s)}→{simple(t)}" in set(df_long["pair"].unique())]

    # Shrink figure to relatively enlarge font
    fig_w = max(3.2, 0.5 * len(order)); fig_h = 1.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    import seaborn as sns
    sns.set(style="ticks")
    sns.violinplot(data=df_long, x="pair", y="log10_p", order=order, cut=0, inner="quartile", color="#88ccee", linewidth=0.8, saturation=0.9, scale="width", ax=ax)
    ax.axhline(y=np.log10(0.05), color="gray", linestyle=":", linewidth=1)
    for i, pair_lbl in enumerate(order):
        s, t = pair_lbl.split("→")
        pexp = em_pvals.get((s, t))
        if pexp is not None and np.isfinite(pexp):
            y = np.log10(max(min(pexp, 1.0), 1e-12))
            ax.scatter(i, y, facecolors="white", edgecolors="black", s=50, zorder=5, linewidth=0.8)
    ax.set_ylabel("log10(p-value)")
    ax.set_xlabel("Cell-type pair")
    ax.set_xticklabels(order, rotation=30, ha="right")
    trim_spines(ax)
    plt.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True); plt.savefig(out_png, dpi=150); plt.close(fig)
