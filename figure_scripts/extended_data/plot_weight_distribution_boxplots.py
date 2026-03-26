#!/usr/bin/env python3
"""Box plots of synaptic weight distributions (per source → target cell type).

Three data series per panel: Untrained, Bio-trained, Naive.
One panel per source cell type (19 types, 4×5 grid).
X-axis: 19 target cell types.
Pooled over 10 networks (core_nll_0..9).

Output: figures/paper/figure5/weight_distribution_boxplots.pdf + .png
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import colorsys

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_shared.style import apply_pub_style, trim_spines
from analysis_shared.celltype_labels import abbrev_cell_type, abbrev_cell_types


def _hue_hex(hue_deg: float, s: float = 0.75, v: float = 0.85) -> str:
    r, g, b = colorsys.hsv_to_rgb((hue_deg % 360) / 360.0, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

OUTPUT_DIR = PROJECT_ROOT / "figures" / "paper" / "extended_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cell type order consistent with image_decoding/plot_utils.cell_type_order()
# Exc by layer → L1_Inh → PV by layer → SST by layer → VIP by layer
CT_ORDER = [
    "L2/3_Exc", "L4_Exc", "L5_ET", "L5_IT", "L5_NP", "L6_Exc",
    "L1_Inh",
    "L2/3_PV", "L4_PV", "L5_PV", "L6_PV",
    "L2/3_SST", "L4_SST", "L5_SST", "L6_SST",
    "L2/3_VIP", "L4_VIP", "L5_VIP", "L6_VIP",
]

CONDITIONS = [
    ("untrained",   "v1_v1_edges.h5"),
    ("bio_trained", "v1_v1_edges_bio_trained.h5"),
    ("naive",       "v1_v1_edges_naive.h5"),
]

# Colors matching dataset_palette() in image_decoding/plot_utils.py
COND_COLORS = {
    "untrained":   _hue_hex(45),    # warm yellow
    "bio_trained": _hue_hex(135),   # green
    "naive":       _hue_hex(315),   # pink/magenta
}
COND_LABELS = {
    "untrained":   "Untrained",
    "bio_trained": "Trained (Constrained)",
    "naive":       "Trained (Unconstrained)",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _build_pop_to_celltype() -> dict[str, str]:
    scheme = pd.read_csv(
        PROJECT_ROOT / "base_props" / "cell_type_naming_scheme.csv",
        sep=" ", index_col=0,
    )
    return scheme["cell_type"].to_dict()


def build_edgetype_mapping(basedir: str, pop_to_ct: dict[str, str]) -> dict[int, tuple[str, str]]:
    """Map edge_type_id → (source_cell_type, target_cell_type)."""
    node_types = pd.read_csv(
        f"{basedir}/network/v1_node_types.csv", sep=" ", index_col="node_type_id"
    )
    ntid_to_ct: dict[int, str] = {
        int(ntid): pop_to_ct.get(row["pop_name"], row["pop_name"])
        for ntid, row in node_types.iterrows()
    }

    edge_types = pd.read_csv(
        f"{basedir}/network/v1_v1_edge_types.csv", sep=" ", index_col="edge_type_id"
    )

    mapping: dict[int, tuple[str, str]] = {}
    for etid, row in edge_types.iterrows():
        m_src = re.search(r"pop_name=='([^']+)'", str(row.get("source_query", "")))
        src_ct = pop_to_ct.get(m_src.group(1), m_src.group(1)) if m_src else "unknown"

        m_tgt = re.search(r"node_type_id=='(\d+)'", str(row.get("target_query", "")))
        tgt_ct = ntid_to_ct.get(int(m_tgt.group(1)), "unknown") if m_tgt else "unknown"

        mapping[int(etid)] = (src_ct, tgt_ct)

    return mapping


def load_weights_by_pair(
    basedir: str,
    h5_filename: str,
    mapping: dict[int, tuple[str, str]],
) -> dict[tuple[str, str], np.ndarray]:
    """Return absolute syn_weight arrays keyed by (src_cell_type, tgt_cell_type)."""
    h5_path = f"{basedir}/network/{h5_filename}"
    with h5py.File(h5_path, "r") as f:
        grp = f["edges"]["v1_to_v1"]
        eids = grp["edge_type_id"][:].astype(np.int32)
        weights = np.abs(grp["0"]["syn_weight"][:].astype(np.float32))

    # Vectorised grouping via np.unique (only ~4422 unique IDs → fast loop)
    unique_eids, inverse = np.unique(eids, return_inverse=True)

    pair_chunks: dict[tuple[str, str], list[np.ndarray]] = {}
    for local_i, eid in enumerate(unique_eids):
        pair = mapping.get(int(eid))
        if pair is None:
            continue
        chunk = weights[inverse == local_i]
        pair_chunks.setdefault(pair, []).append(chunk)

    return {pair: np.concatenate(chunks) for pair, chunks in pair_chunks.items()}


def collect_all_weights(n_networks: int = 10) -> dict[str, dict[tuple[str, str], list[np.ndarray]]]:
    """
    Returns all_weights[condition][(src_ct, tgt_ct)] = list of per-network arrays.
    """
    pop_to_ct = _build_pop_to_celltype()

    # structure: cond -> pair -> [array_net0, array_net1, ...]
    all_weights: dict[str, dict[tuple[str, str], list[np.ndarray]]] = {
        cond: {} for cond, _ in CONDITIONS
    }

    for net_idx in range(n_networks):
        basedir = str(PROJECT_ROOT / f"core_nll_{net_idx}")
        if not os.path.isdir(basedir):
            print(f"  Skipping {basedir} (not found)")
            continue

        t0 = time.time()
        print(f"Network {net_idx}: building edge-type mapping...", end=" ", flush=True)
        mapping = build_edgetype_mapping(basedir, pop_to_ct)
        print(f"{time.time() - t0:.1f}s")

        for cond, h5_file in CONDITIONS:
            h5_path = f"{basedir}/network/{h5_file}"
            if not os.path.exists(h5_path):
                print(f"  Missing: {h5_path}")
                continue

            t0 = time.time()
            print(f"  [{cond}] loading...", end=" ", flush=True)
            pair_w = load_weights_by_pair(basedir, h5_file, mapping)
            print(f"{time.time() - t0:.1f}s  ({len(pair_w)} pairs)")

            for pair, w in pair_w.items():
                all_weights[cond].setdefault(pair, []).append(w)

    return all_weights


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _spearman_medians(
    weights_a: dict[tuple[str, str], list[np.ndarray]],
    weights_b: dict[tuple[str, str], list[np.ndarray]],
    src_ct: str,
    ct_order: list[str],
) -> float | None:
    """Spearman r between per-target median weights for two conditions."""
    meds_a, meds_b = [], []
    for tgt_ct in ct_order:
        pair = (src_ct, tgt_ct)
        wa = weights_a.get(pair)
        wb = weights_b.get(pair)
        if wa and wb:
            meds_a.append(float(np.median(np.concatenate(wa))))
            meds_b.append(float(np.median(np.concatenate(wb))))
    if len(meds_a) < 3:
        return None
    r, _ = spearmanr(meds_a, meds_b)
    return float(r)


def plot_weight_boxplots(
    all_weights: dict[str, dict[tuple[str, str], list[np.ndarray]]],
    output_path: Path,
) -> None:
    apply_pub_style()

    n_ct = len(CT_ORDER)

    # 5 rows × 4 cols (last slot = legend)
    n_cols, n_rows = 4, 5
    panel_w, panel_h = 2.3, 2.05
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_w * n_cols, panel_h * n_rows),
        squeeze=False,
    )

    box_w = 0.27
    cond_offsets = np.array([-(box_w + 0.02), 0.0, (box_w + 0.02)])
    tgt_labels = abbrev_cell_types(CT_ORDER)

    for src_idx, src_ct in enumerate(CT_ORDER):
        row, col = divmod(src_idx, n_cols)
        ax = axes[row][col]

        x_pos = np.arange(n_ct, dtype=float)

        # Alternating grey stripe background
        for xi in range(n_ct):
            if xi % 2 == 1:
                ax.axvspan(xi - 0.5, xi + 0.5, color="#efefef", zorder=0, lw=0)

        for ci, (cond, _) in enumerate(CONDITIONS):
            color = COND_COLORS[cond]
            bp_data: list[np.ndarray] = []
            bp_pos: list[float] = []

            for tgt_idx, tgt_ct in enumerate(CT_ORDER):
                pair = (src_ct, tgt_ct)
                arrays = all_weights[cond].get(pair)
                if arrays:
                    combined = np.concatenate(arrays)
                    combined = combined[combined > 0]
                else:
                    combined = np.full(1, np.nan)
                bp_data.append(combined)
                bp_pos.append(x_pos[tgt_idx] + cond_offsets[ci])

            ax.boxplot(
                bp_data,
                positions=bp_pos,
                widths=box_w * 0.90,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(
                    marker=".",
                    markersize=0.7,
                    alpha=0.15,
                    markerfacecolor=color,
                    markeredgewidth=0,
                ),
                medianprops=dict(color="black", linewidth=0.8),
                boxprops=dict(facecolor=color, alpha=0.80, linewidth=0.5),
                whiskerprops=dict(linewidth=0.5, color="#444444"),
                capprops=dict(linewidth=0.5, color="#444444"),
            )

        # Two Spearman annotations: Sc = untrained vs constrained, Su = untrained vs unconstrained
        sc = _spearman_medians(all_weights["untrained"], all_weights["bio_trained"], src_ct, CT_ORDER)
        su = _spearman_medians(all_weights["untrained"], all_weights["naive"], src_ct, CT_ORDER)
        ann_lines = []
        if sc is not None:
            ann_lines.append(f"Sc:{sc:.2f}")
        if su is not None:
            ann_lines.append(f"Su:{su:.2f}")
        if ann_lines:
            ax.text(
                0.98, 0.98,
                "\n".join(ann_lines),
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=5.0,
                linespacing=1.4,
            )

        ax.set_yscale("log")
        ax.set_xlim(-0.6, n_ct - 0.4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tgt_labels, rotation=90, fontsize=5.0)
        ax.set_title(f"Source: {src_ct.replace('_', ' ')}", fontsize=6.5, pad=2)
        # Y-axis label only on leftmost column
        if col == 0:
            ax.set_ylabel("|Weight| (pA)", fontsize=6.0)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=5, which="both")
        ax.tick_params(axis="x", length=2, pad=1)
        trim_spines(ax)

    # Hide unused axes
    for idx in range(len(CT_ORDER), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    # Legend in the empty last panel slot
    last_row, last_col = divmod(len(CT_ORDER), n_cols)
    legend_ax = axes[last_row][last_col]
    legend_ax.set_visible(True)
    legend_ax.axis("off")
    handles = [
        mpatches.Patch(facecolor=COND_COLORS[cond], alpha=0.80, label=COND_LABELS[cond])
        for cond, _ in CONDITIONS
    ]
    legend_ax.legend(
        handles=handles,
        loc="center left",
        fontsize=7,
        frameon=False,
        title="Condition",
        title_fontsize=7.5,
    )

    fig.tight_layout(pad=0.4, h_pad=0.7, w_pad=0.2)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    print(f"Saved:\n  {pdf_path}\n  {png_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing data cache")
    args = parser.parse_args()

    CACHE_PATH = OUTPUT_DIR / "weight_distribution_cache.pkl"

    t_start = time.time()
    if not args.no_cache and CACHE_PATH.exists():
        print(f"Loading cached data from {CACHE_PATH}...")
        with open(CACHE_PATH, "rb") as f:
            all_weights = pickle.load(f)
        print(f"Cache loaded in {time.time() - t_start:.1f}s")
    else:
        print("=== Collecting weights from all networks ===")
        all_weights = collect_all_weights(n_networks=10)
        print(f"\nData loaded in {time.time() - t_start:.1f}s")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(all_weights, f)
        print(f"Cache saved to {CACHE_PATH}")

    print("\n=== Generating figure ===")
    out_path = OUTPUT_DIR / "weight_distribution_boxplots"
    plot_weight_boxplots(all_weights, out_path)
    print(f"\nTotal time: {time.time() - t_start:.1f}s")
