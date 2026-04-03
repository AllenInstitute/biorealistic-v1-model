"""
Extended Data Figure: Raster plots for core_nll_0 at different contrast levels.

Generates one raster-only PNG per contrast (0.05, 0.2, 0.4, 0.6, 0.8) using
trial 0 at angle 0, bio_trained weights. Format matches plot_raster.py
(--raster-only --full-network --grouping four).

Must be run from the project root:
    conda run -n new_v1 python figure_scripts/extended_data/contrast_rasters/plot_contrast_rasters.py
"""

import sys
import os
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# Ensure project root is on path (script is 3 levels deep: figure_scripts/extended_data/contrast_rasters/)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
# Must run from project root so that pu.form_network(infer=True) resolves correctly
os.chdir(PROJECT_ROOT)

import seaborn as sns
import plotting_utils as pu

# ---------------------------------------------------------------------------
# Font: Arial via font_manager
# ---------------------------------------------------------------------------
_arial_candidates = [f.fname for f in fm.fontManager.ttflist if "Arial" in f.name]
if _arial_candidates:
    matplotlib.rcParams["font.family"] = "Arial"
else:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ---------------------------------------------------------------------------
# Parameters  (mirror plot_raster.py --raster-only --full-network --grouping four)
# ---------------------------------------------------------------------------
NET = "core_nll_0"
CONDITION = "contrasts_bio_trained"
ANGLE = 45
TRIAL = 0
CONTRASTS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]

# plot_raster.py: settings["core"] = {radius:200, s:1}; --full-network sets radius=None, s*=0.25
MARKER_S = 0.25
SORTBY = "tuning_angle"
GROUPING = "four"
BASE_FONTSIZE = 16.0   # plot_raster.py --raster-only default
LEGEND_MARKERSIZE = 8  # fixed pt size for legend handles (visible at 16pt font)

OUT_DIR = PROJECT_ROOT / "figures" / "paper" / "extended_data" / "contrast_rasters"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_contrast_raster(contrast: float):
    contrast_str = str(contrast)
    # Pass the trial directory as the "config file" path.
    # pu.get_spikes(infer=True) reads {dir}/spikes.csv
    # pu.form_network(infer=True) takes path.split("/")[0] = "core_nll_0"
    trial_dir = f"{NET}/{CONDITION}/angle{ANGLE}_contrast{contrast_str}_trial{TRIAL}"
    spikes_csv = pathlib.Path(trial_dir) / "spikes.csv"
    if not spikes_csv.exists():
        print(f"[WARN] spikes.csv not found: {spikes_csv}")
        return
    # pu.get_spikes(infer=True) reads Path(config_file).parent / "spikes.csv"
    # pu.form_network(infer=True) takes config_file.split("/")[0] as net name
    fake_config = f"{trial_dir}/config.json"

    plt.rcParams.update({
        "font.size": BASE_FONTSIZE,
        "axes.titlesize": BASE_FONTSIZE,
        "axes.labelsize": BASE_FONTSIZE,
        "legend.fontsize": BASE_FONTSIZE,
        "legend.title_fontsize": BASE_FONTSIZE,
        "xtick.labelsize": BASE_FONTSIZE,
        "ytick.labelsize": BASE_FONTSIZE,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    ax = pu.plot_raster(
        fake_config,
        sortby=SORTBY,
        infer=True,
        ax=ax,
        grouping=GROUPING,
        layer_label_fontsize=BASE_FONTSIZE,
        s=MARKER_S,
        radius=None,  # all cells
        legend_markerscale=None,  # we rebuild the legend below
    )

    ax.set_xlim([0, 2500])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.01)
    ax.tick_params(axis="both", labelsize=BASE_FONTSIZE)

    # Rebuild legend with fixed-size markers so they are clearly visible
    existing_leg = ax.get_legend()
    if existing_leg is not None:
        labels = [t.get_text() for t in existing_leg.get_texts()]
        colors = [h.get_markerfacecolor() if hasattr(h, "get_markerfacecolor") else h.get_color()
                  for h in existing_leg.legend_handles]
        existing_leg.remove()
        handles = [
            mlines.Line2D(
                [], [], marker="o", color="w", markerfacecolor=c,
                markersize=LEGEND_MARKERSIZE, label=lbl, linewidth=0,
            )
            for lbl, c in zip(labels, colors)
        ]
        leg = ax.legend(
            handles=handles, loc="upper right",
            fontsize=BASE_FONTSIZE, frameon=True,
        )

    plt.tight_layout()

    out_path = OUT_DIR / f"contrast_raster_{NET}_c{contrast_str}.png"
    plt.savefig(str(out_path), dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for c in CONTRASTS:
        print(f"Plotting contrast {c} ...")
        plot_contrast_raster(c)

    print("Done.")
