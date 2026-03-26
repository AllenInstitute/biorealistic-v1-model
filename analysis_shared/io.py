from __future__ import annotations
import os
import pandas as pd
from typing import Tuple

import numpy as np
import network_utils as nu


def _format_appendix(network_type: str) -> str:
    if network_type and not network_type.startswith("_"):
        return f"_{network_type}"
    return network_type or ""


def load_edges_for_network(network_dir: str, network_type: str) -> pd.DataFrame:
    """Load edges with core fields using network_utils; returns DataFrame with source_type, target_type, syn_weight, Response Correlation if present.
    network_type: one of {bio_trained, naive} used to select correct appendix if needed by nu.load_edges_pl.
    """
    appendix = _format_appendix(network_type)
    edges = nu.load_edges_pl(network_dir, appendix=appendix)
    # Ensure expected columns exist
    expected = ["source_type", "target_type", "syn_weight"]
    for col in expected:
        if col not in edges.columns:
            # Note: edges here may be a polars LazyFrame; defer to higher-level loaders elsewhere.
            raise ValueError(f"Missing required column in edges: {col}")
    return edges


def load_v1_features(network_dir: str, label: str) -> pd.DataFrame:
    """Load v1_features_df_{label}.csv placed by migrator into network_dir."""
    path = os.path.join(network_dir, f"v1_features_df_{label}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_edges_with_pref_dir(network_dir: str, network_type: str) -> pd.DataFrame:
    """Compute per-edge preferred direction difference (degrees) for core-to-core edges.
    Returns DataFrame with columns: source_id, target_id, syn_weight, pref_dir_diff_deg.
    """
    appendix = _format_appendix(network_type)
    edges = nu.load_edges(network_dir, src="v1", tgt="v1", appendix=appendix)
    nodes = nu.load_nodes(network_dir, loc="v1", core_radius=200)

    src_ids = edges["source_id"]
    tgt_ids = edges["target_id"]
    core_src = nodes["core"][src_ids]
    core_tgt = nodes["core"][tgt_ids]
    both_core = core_src & core_tgt

    src_ids = src_ids[both_core]
    tgt_ids = tgt_ids[both_core]
    syn_w = edges["syn_weight"][both_core]

    pre_theta = nodes["tuning_angle"][src_ids]
    post_theta = nodes["tuning_angle"][tgt_ids]
    pref_dir = nu.angle_difference(pre_theta, post_theta, mode="direction")

    return pd.DataFrame({
        "source_id": src_ids,
        "target_id": tgt_ids,
        "syn_weight": syn_w,
        "pref_dir_diff_deg": pref_dir,
    })
