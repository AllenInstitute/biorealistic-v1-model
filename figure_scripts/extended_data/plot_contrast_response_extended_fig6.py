"""Extended Data Figure 6: Contrast response line plots for all cell types (unconstrained networks).

Aggregates across all core_nll_* networks for the naive (unconstrained) condition.
Produces one figure per metric (firing rate, OSI, DSI, lifetime sparsity) with
4 panels (one per layer group), each line = one cell type, shaded region = 95% CI
across networks.

Output: figures/extended_data/contrast_response_naive/
"""

import os
import glob
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.rcParams["font.family"] = "Arial"

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import network_utils as nu
import stimulus_trials as st
from analysis_shared.celltype_labels import abbrev_cell_type

LAYER_GROUPS = {
    "L1+L2/3": ["L1_Inh", "L2/3_Exc", "L2/3_PV", "L2/3_SST", "L2/3_VIP"],
    "L4":      ["L4_Exc", "L4_PV", "L4_SST", "L4_VIP"],
    "L5":      ["L5_ET", "L5_IT", "L5_NP", "L5_PV", "L5_SST", "L5_VIP"],
    "L6":      ["L6_Exc", "L6_PV", "L6_SST", "L6_VIP"],
}


def get_color_map() -> dict:
    scheme_path = Path(ROOT) / "base_props" / "cell_type_naming_scheme.csv"
    df = pd.read_csv(scheme_path, sep=r"\s+", engine="python")
    return dict(zip(df["cell_type"], df["hex"]))


def compute_osi(rates: np.ndarray, angles: np.ndarray) -> float:
    if rates.sum() == 0:
        return np.nan
    rad = np.deg2rad(angles)
    vec = (rates * np.exp(2j * rad)).sum() / rates.sum()
    return float(np.abs(vec))


def compute_dsi(rates: np.ndarray, angles: np.ndarray) -> float:
    if rates.sum() == 0:
        return np.nan
    rad = np.deg2rad(angles)
    vec = (rates * np.exp(1j * rad)).sum() / rates.sum()
    return float(np.abs(vec))


def compute_lifetime_sparsity(rates: np.ndarray) -> float:
    n = len(rates)
    if n == 0 or np.sum(rates) == 0:
        return np.nan
    mean_r = np.mean(rates)
    mean_r2 = np.mean(rates ** 2)
    if mean_r == 0:
        return np.nan
    return float((1 - (mean_r ** 2 / mean_r2)) / (1 - 1.0 / n))


def process_one_network(basedir: str, network_option: str) -> pd.DataFrame | None:
    spike_file = os.path.join(basedir, f"contrasts_{network_option}", "spike_counts.npz")
    if not os.path.isfile(spike_file):
        print(f"[WARN] Not found: {spike_file}")
        return None

    data = np.load(spike_file)
    evoked_spikes = data["evoked_spikes"]          # (angles, contrasts, trials, cells)
    int_len_ms = float(data["interval_length"])
    evoked_rates = evoked_spikes * (1000.0 / int_len_ms)

    contrast_stim = st.ContrastStimulus()
    contrast_vals = np.array(contrast_stim.contrasts)
    angles = np.linspace(0, 315, evoked_rates.shape[0])

    nodes = nu.load_nodes(basedir, core_radius=200, expand=True)
    nodes_core = nodes[nodes["core"]]

    rows = []
    for cell_type, df_type in nodes_core.groupby("cell_type"):
        cell_ids = df_type.index.to_numpy()
        if cell_ids.size == 0:
            continue
        ct_rates = evoked_rates[:, :, :, cell_ids]  # (angles, contrasts, trials, cells)

        for c_idx, contrast in enumerate(contrast_vals):
            contrast_rates = ct_rates[:, c_idx, :, :]      # (angles, trials, cells)
            mean_by_angle = contrast_rates.mean(axis=1)    # (angles, cells)

            rate_values = mean_by_angle.mean(axis=0)
            n_cells = mean_by_angle.shape[1]
            osi_values = np.array([
                compute_osi(mean_by_angle[:, ci], angles)
                for ci in range(n_cells)
            ])
            dsi_values = np.array([
                compute_dsi(mean_by_angle[:, ci], angles)
                for ci in range(n_cells)
            ])
            sparsity_values = np.array([
                compute_lifetime_sparsity(mean_by_angle[:, ci])
                for ci in range(n_cells)
            ])

            rows.append({
                "network": basedir,
                "condition": network_option,
                "cell_type": cell_type,
                "contrast": contrast,
                "mean_rate": float(np.nanmean(rate_values)),
                "mean_osi": float(np.nanmean(osi_values)),
                "mean_dsi": float(np.nanmean(dsi_values)),
                "mean_sparsity": float(np.nanmean(sparsity_values)),
                "n_cells": int(cell_ids.size),
            })

    return pd.DataFrame(rows)


def _contrast_stats(cell_data: pd.DataFrame, metric: str) -> pd.DataFrame:
    grp = cell_data.groupby("contrast")[metric].agg(["mean", "std", "count"]).reset_index()
    grp["sem"] = grp["std"] / np.sqrt(grp["count"])
    grp["ci"] = grp.apply(
        lambda r: stats.t.ppf(0.975, r["count"] - 1) * r["sem"] if r["count"] > 1 else r["sem"],
        axis=1,
    )
    return grp


def make_figure(
    all_data: pd.DataFrame,
    color_map: dict,
    condition: str,
    metric: str,
    ylabel: str,
    ylim,
    save_dir: Path,
):
    condition_data = all_data[all_data["condition"] == condition]

    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.4), sharey=(ylim is not None))
    axes = np.array(axes).flatten()

    for ax_idx, (layer_name, cell_types) in enumerate(LAYER_GROUPS.items()):
        ax = axes[ax_idx]

        for cell_type in cell_types:
            cd = condition_data[condition_data["cell_type"] == cell_type]
            if cd.empty:
                continue
            cs = _contrast_stats(cd, metric)
            color = color_map.get(cell_type, None)
            label = abbrev_cell_type(cell_type)

            ax.plot(cs["contrast"], cs["mean"], color=color, linewidth=1.5, label=label)
            ax.fill_between(
                cs["contrast"],
                cs["mean"] - cs["ci"],
                cs["mean"] + cs["ci"],
                color=color, alpha=0.2,
            )

        ax.set_xlabel("Contrast", fontsize=8)
        ax.set_ylabel(ylabel if ax_idx == 0 else "", fontsize=8)
        ax.set_title(layer_name, fontsize=9, pad=3)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
        ax.tick_params(axis="both", labelsize=7, pad=1)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(fontsize=6, frameon=False, loc="upper left", ncol=1)

    fig.tight_layout(w_pad=1.0)

    stem = f"contrast_{metric.replace('mean_', '')}_{condition}"
    out_pdf = save_dir / f"{stem}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extended Data Fig. 6: contrast response line plots for all cell types (unconstrained)."
    )
    parser.add_argument(
        "--basedirs", nargs="*", default=None,
        help="Network directories. Defaults to all core_nll_* in the project root.",
    )
    parser.add_argument(
        "--conditions", nargs="*", default=["naive"],
        help="Training conditions to plot (default: naive).",
    )
    args = parser.parse_args()

    os.chdir(ROOT)

    basedirs = args.basedirs or sorted(
        d for d in glob.glob("core_nll_*") if os.path.isdir(d)
    )
    if not basedirs:
        raise RuntimeError("No core_nll_* directories found.")

    color_map = get_color_map()

    all_results = []
    for d in basedirs:
        for cond in args.conditions:
            df = process_one_network(d, cond)
            if df is not None:
                all_results.append(df)

    if not all_results:
        raise RuntimeError("No spike data found for any network/condition.")

    all_data = pd.concat(all_results, ignore_index=True)

    save_dir = Path(ROOT) / "figures" / "paper" / "extended_data" / "contrast_response_naive"
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("mean_rate",     "Firing rate (Hz)",       None),
        ("mean_osi",      "OSI",                    (0, 1)),
        ("mean_dsi",      "DSI",                    (0, 1)),
        ("mean_sparsity", "Stimulus selectivity",   (0, 1)),
    ]

    for condition in args.conditions:
        for metric, ylabel, ylim in metrics:
            make_figure(all_data, color_map, condition, metric, ylabel, ylim, save_dir)

    print("Done.")
