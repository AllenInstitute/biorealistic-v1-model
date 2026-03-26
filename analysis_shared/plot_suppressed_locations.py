#!/usr/bin/env python3
"""Plot x-z distributions of suppressed (clamped) neurons for outgoing-weight perturbations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "core_nll_0"
NODE_SET_DIR = BASE_DIR / "node_sets"
OUTPUT_DIR = BASE_DIR / "figures" / "selectivity_outgoing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORE_RADIUS = 200.0

SCENARIOS = {
    "exc_high": "high_outgoing_exc_nodes.json",
    "exc_low": "low_outgoing_exc_nodes.json",
    "inh_high": "high_outgoing_inh_nodes.json",
    "inh_low": "low_outgoing_inh_nodes.json",
}

CORE_SCENARIOS = {
    "exc_high_core": "high_outgoing_exc_core_nodes.json",
    "exc_low_core": "low_outgoing_exc_core_nodes.json",
    "inh_high_core": "high_outgoing_inh_core_nodes.json",
    "inh_low_core": "low_outgoing_inh_core_nodes.json",
}

PERIPHERY_SCENARIOS = {
    "exc_high_periphery": "high_outgoing_exc_periphery_nodes.json",
    "exc_low_periphery": "low_outgoing_exc_periphery_nodes.json",
    "inh_high_periphery": "high_outgoing_inh_periphery_nodes.json",
    "inh_low_periphery": "low_outgoing_inh_periphery_nodes.json",
}


def load_node_coordinates() -> pd.DataFrame:
    """Return DataFrame indexed by node_id with x, z, radius."""
    with h5py.File(BASE_DIR / "network" / "v1_nodes.h5", "r") as f:
        grp = f["nodes"]["v1"]
        node_ids = grp["node_id"][:]
        node_type_ids = grp["node_type_id"][:]
        coords = grp["0"]
        x = coords["x"][:]
        z = coords["z"][:]
    radius = np.sqrt(x**2 + z**2)
    df = pd.DataFrame(
        {
            "node_type_id": node_type_ids,
            "x": x,
            "z": z,
            "radius": radius,
        },
        index=node_ids,
    )
    df.index.name = "node_id"
    return df


NODE_COORDS = load_node_coordinates()


def load_node_ids(name: str) -> np.ndarray:
    data = json.loads((NODE_SET_DIR / name).read_text())
    return np.array(data.get("node_id", []), dtype=np.int64)


def plot_points(
    node_ids: Iterable[int],
    title: str,
    filename: str,
    *,
    highlight_core: bool = False,
    core_only: bool = False,
) -> None:
    ids = np.asarray(list(node_ids), dtype=np.int64)
    if ids.size == 0:
        print(f"[warn] No nodes for {title}")
        return

    df = NODE_COORDS.reindex(ids).dropna()
    if df.empty:
        print(f"[warn] No coordinates for {title}")
        return

    if core_only:
        df = df[df["radius"] <= CORE_RADIUS]
        if df.empty:
            print(f"[warn] No core coordinates after filtering for {title}")
            return

    fig, ax = plt.subplots(figsize=(6, 6))

    if highlight_core:
        core_mask = df["radius"] <= CORE_RADIUS
        ax.scatter(
            df.loc[core_mask, "x"],
            df.loc[core_mask, "z"],
            s=12,
            c="#1f77b4",
            alpha=0.7,
            label=f"core (n={core_mask.sum()})",
        )
        ax.scatter(
            df.loc[~core_mask, "x"],
            df.loc[~core_mask, "z"],
            s=12,
            c="#ff7f0e",
            alpha=0.7,
            label=f"periphery (n={(~core_mask).sum()})",
        )
        ax.legend(loc="upper right", fontsize=9)
    else:
        ax.scatter(df["x"], df["z"], s=12, c="#2ca02c", alpha=0.7)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.4)
    circle = plt.Circle(
        (0, 0), CORE_RADIUS, color="grey", linestyle="--", fill=False, alpha=0.4
    )
    ax.add_patch(circle)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def main() -> None:
    for key, node_file in SCENARIOS.items():
        ids = load_node_ids(node_file)
        plot_points(
            ids,
            f"{key.replace('_', ' ').title()} (combined)",
            f"suppressed_{key}_combined_xz.png",
        )
        plot_points(
            ids,
            f"{key.replace('_', ' ').title()} (core vs periphery)",
            f"suppressed_{key}_core_periphery_xz.png",
            highlight_core=True,
        )

    for key, node_file in CORE_SCENARIOS.items():
        ids = load_node_ids(node_file)
        plot_points(
            ids,
            f"{key.replace('_', ' ').title()} (core only)",
            f"suppressed_{key}_xz.png",
            core_only=True,
        )

    for key, node_file in PERIPHERY_SCENARIOS.items():
        ids = load_node_ids(node_file)
        plot_points(
            ids,
            f"{key.replace('_', ' ').title()} (periphery only)",
            f"suppressed_{key}_xz.png",
        )


if __name__ == "__main__":
    main()
