from __future__ import annotations
import matplotlib as mpl


def apply_pub_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def trim_spines(ax):
    for sp in ("top", "right"):
        try:
            ax.spines[sp].set_visible(False)
        except Exception:
            pass

