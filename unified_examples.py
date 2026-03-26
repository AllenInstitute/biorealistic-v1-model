#!/usr/bin/env python3
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_shared.grouping import aggregate_l5
from analysis_shared.stats import ols_slope_p, bin_mean_sem

# Reuse existing pipeline loader for typed edge DataFrame
from aggregate_correlation_plot import process_network_data
from analysis_shared.io import load_edges_with_pref_dir


def demo_response_corr(network_dir: str, network_type: str, out_png: str) -> None:
    df = process_network_data((network_dir, network_type))
    df = aggregate_l5(df)
    sub = df[["Response Correlation", "syn_weight"]].dropna()
    if sub.empty:
        print("No data for response correlation demo.")
        return
    x = sub["Response Correlation"].to_numpy()
    y = sub["syn_weight"].to_numpy()

    bins = np.arange(-0.2, 0.5 + 0.05, 0.05)
    centers, means, sems = bin_mean_sem(x, y, bins)
    res = ols_slope_p(x, y)

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.bar(centers, means, width=0.05, color="#b3b3b3", edgecolor="none")
    ax.errorbar(centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2)
    xx = np.array([bins[0], bins[-1]])
    ax.plot(xx, res.intercept + res.slope * xx, color="crimson", linewidth=1.2)
    sig = "" if not np.isfinite(res.p_value) else ("***" if res.p_value < 1e-3 else "**" if res.p_value < 1e-2 else "*" if res.p_value < 5e-2 else "")
    ax.text(0.03, 0.95, f"m={res.slope:.3f} {sig}\np={res.p_value:.2e}", transform=ax.transAxes, fontsize=8, va="top")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("Res. corr.")
    ax.set_ylabel("Syn weight")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def demo_preferred_direction(network_dir: str, network_type: str, out_png: str) -> None:
    feat = load_edges_with_pref_dir(network_dir, network_type)
    if feat.empty:
        print("No data for preferred-direction demo.")
        return
    x = feat["pref_dir_diff_deg"].to_numpy()
    y = feat["syn_weight"].to_numpy()

    bins = np.arange(0, 180 + 20, 20)
    centers, means, sems = bin_mean_sem(x, y, bins)
    res = ols_slope_p(x, y)

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.bar(centers, means, width=20, color="#aec7e8", edgecolor="none")
    ax.errorbar(centers, means, yerr=sems, fmt="none", ecolor="k", elinewidth=1, capsize=2)
    xx = np.array([bins[0], bins[-1]])
    ax.plot(xx, res.intercept + res.slope * xx, color="crimson", linewidth=1.2)
    sig = "" if not np.isfinite(res.p_value) else ("***" if res.p_value < 1e-3 else "**" if res.p_value < 1e-2 else "*" if res.p_value < 5e-2 else "")
    ax.text(0.03, 0.95, f"m={res.slope:.3f} {sig}\np={res.p_value:.2e}", transform=ax.transAxes, fontsize=8, va="top")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("Pref. dir. diff (deg)")
    ax.set_ylabel("Syn weight")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    base = \
        "/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/glif_builder_test/biorealistic-v1-model"
    net_dir = os.path.join(base, "core_nll_0")
    out_dir = os.path.join(net_dir, "figures", "unified_demo")

    demo_response_corr(net_dir, "bio_trained", os.path.join(out_dir, "response_corr_demo.png"))
    demo_preferred_direction(net_dir, "bio_trained", os.path.join(out_dir, "preferred_direction_demo.png"))


if __name__ == "__main__":
    main()
