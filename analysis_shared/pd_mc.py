from __future__ import annotations
import os
from typing import Dict, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_shared.io import load_edges_with_pref_dir, load_edges_with_computed_pref_dir
from analysis_shared.grouping import aggregate_l5
from analysis_shared.stats import fit_cosine_series_deg
from analysis_shared.style import apply_pub_style, trim_spines


EXC_ALL = ["L2/3_Exc", "L4_Exc", "L5_IT", "L5_ET", "L5_NP", "L6_Exc"]
EXC_L5_AGG = ["L2/3_Exc", "L4_Exc", "L5_Exc", "L6_Exc"]
PD_EXC_FOCUS = ["L2/3_Exc", "L4_Exc", "L5_Exc"]


def _normalize_label(lbl: str) -> str:
    x = lbl.strip().replace(" ", "_")
    x = x.replace("Exc", "Exc")  # keep token
    x = x.replace("L5_Exc", "L5_Exc")
    x = x.replace("L4_Exc", "L4_Exc")
    x = x.replace("L2/3_Exc", "L2/3_Exc").replace("L2/3_", "L2/3_")
    return x


def _read_pair_limits_csv(path: str) -> Dict[Tuple[str, str], int]:
    df = pd.read_csv(path)
    required = {"source", "target", "connections"}
    if not required.issubset(df.columns):
        raise ValueError(f"Pair limits CSV must contain columns {required}")
    mapping: Dict[Tuple[str, str], int] = {}
    for _, row in df.iterrows():
        s = _normalize_label(str(row["source"]))
        t = _normalize_label(str(row["target"]))
        try:
            n = None if pd.isna(row["connections"]) else int(row["connections"])
        except Exception:
            n = None
        if n is not None and n >= 0:
            mapping[(s, t)] = n
    return mapping


def _prepare_pd_dataframe(base_dirs: Sequence[str], network_type: str, aggregate_l5_types: bool, loader=None) -> pd.DataFrame:
    if loader is None:
        loader = load_edges_with_pref_dir
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
    df = df.dropna(subset=["source_type", "target_type"])  # ensure types present
    if aggregate_l5_types:
        df = aggregate_l5(df)
        exc_types = EXC_L5_AGG
    else:
        exc_types = EXC_ALL
    df["source_type"] = df["source_type"].astype(str)
    df["target_type"] = df["target_type"].astype(str)
    df = df[df["source_type"].isin(exc_types) & df["target_type"].isin(exc_types)]
    return df


def compute_pd_mc_pvalues(base_dirs: Sequence[str], network_type: str, *, aggregate_l5_types: bool, resamples: int, connections_per_draw: int | None, seed: int = 0, pair_limits_csv: str | None = None, loader=None) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    df = _prepare_pd_dataframe(base_dirs, network_type, aggregate_l5_types, loader=loader)
    # Focus pairs for PD reduced stats
    exc_types = PD_EXC_FOCUS

    # Optional per-pair CSV limits
    limits = _read_pair_limits_csv(pair_limits_csv) if pair_limits_csv else None

    results: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(seed)

    for s in exc_types:
        for t in exc_types:
            sub = df[(df["source_type"] == s) & (df["target_type"] == t)][["pref_dir_diff_deg", "syn_weight"]].dropna()
            if sub.empty or len(sub) < 10:
                results[(s, t)] = {"p_a": np.array([]), "p_b": np.array([])}
                continue
            x_all = sub["pref_dir_diff_deg"].to_numpy()
            y_all = sub["syn_weight"].to_numpy()
            n = x_all.size
            # Determine draw size: CSV per-pair override > uniform > full
            draw_csv = limits.get((s, t)) if limits else None
            draw_uniform = int(connections_per_draw) if connections_per_draw is not None else None
            draw = n
            if draw_csv is not None:
                draw = min(n, int(draw_csv))
            elif draw_uniform is not None:
                draw = min(n, int(draw_uniform))
            p_as: list[float] = []
            p_bs: list[float] = []
            for _ in range(int(resamples)):
                if draw < 10:
                    break
                take = rng.choice(n, size=draw, replace=False)
                x = x_all[take]
                y = y_all[take]
                fit = fit_cosine_series_deg(x, y)
                if np.isfinite(fit.p_a) and np.isfinite(fit.p_b):
                    p_as.append(float(fit.p_a))
                    p_bs.append(float(fit.p_b))
            results[(s, t)] = {"p_a": np.array(p_as), "p_b": np.array(p_bs)}
    return results


def plot_pd_mc_histograms(mc_results: Dict[Tuple[str, str], Dict[str, np.ndarray]], *, aggregate_l5_types: bool, out_png: str) -> None:
    apply_pub_style()
    exc_types = PD_EXC_FOCUS
    n = len(exc_types)
    # Flip to log10(p) and shrink figure height
    fig, axes = plt.subplots(n, n, figsize=(n * 1.8, n * 1.6), sharex=True, sharey=True)
    if n == 1:
        axes = np.array([[axes]])

    bins = np.linspace(-12, 0, 21)  # log10(p) from 1e-12 to 1
    for i, s in enumerate(exc_types):
        for j, t in enumerate(exc_types):
            ax = axes[i, j]
            d = mc_results.get((s, t), {"p_a": np.array([]), "p_b": np.array([])})
            pa = d.get("p_a", np.array([]))
            pb = d.get("p_b", np.array([]))
            if pa.size == 0 and pb.size == 0:
                ax.set_axis_off()
                continue
            if pa.size:
                ax.hist(np.log10(np.clip(pa, 1e-12, 1.0)), bins=bins, alpha=0.6, color="C0", label="log10 p(a)")
                med_a = float(np.nanmedian(np.log10(np.clip(pa, 1e-12, 1.0))))
                ax.axvline(med_a, color="C0", linestyle="--", linewidth=1)
            if pb.size:
                ax.hist(np.log10(np.clip(pb, 1e-12, 1.0)), bins=bins, alpha=0.6, color="C3", label="log10 p(b)")
                med_b = float(np.nanmedian(np.log10(np.clip(pb, 1e-12, 1.0))))
                ax.axvline(med_b, color="C3", linestyle="--", linewidth=1)
            ax.set_xlim(bins[0], bins[-1])
            if i == 0:
                ax.set_title(t, fontsize=9)
            if j == 0:
                ax.set_ylabel(s, fontsize=9)
            if i == n - 1:
                ax.set_xlabel("log10(p)", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper right")
            trim_spines(ax)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
