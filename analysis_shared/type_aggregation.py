#!/usr/bin/env python3
"""Utilities for remapping and aggregating cell-type indexed matrices."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

L5_TYPES = ("L5_IT", "L5_ET", "L5_NP")
L5_AGGREGATE_LABEL = "L5_Exc"

INH_SUFFIXES = ("PV", "SST", "VIP")
INH_AGGREGATE_LABELS = {suffix: suffix for suffix in INH_SUFFIXES}

LAYER_ORDER = {"L1": 0, "L2/3": 1, "L4": 2, "L5": 3, "L6": 4, "All": 5}
CLASS_ORDER = {"Exc": 0, "PV": 1, "SST": 2, "VIP": 3, "Inh": 4}
EXC_ORDER = {
    "L2/3_Exc": 0,
    "L4_Exc": 1,
    "L5_IT": 2,
    "L5_ET": 3,
    "L5_NP": 4,
    "L5_Exc": 5,
    "L6_Exc": 6,
}


@dataclass(frozen=True)
class TypeMapping:
    labels: List[str]
    indices: np.ndarray


def map_l5(label: str, mode: str) -> str:
    if mode == "aggregate" and label in L5_TYPES:
        return L5_AGGREGATE_LABEL
    return label


def map_inhibitory(label: str, mode: str) -> str:
    if mode != "family":
        return label
    for suffix in INH_SUFFIXES:
        if label.endswith(suffix):
            return INH_AGGREGATE_LABELS[suffix]
    return label


def layer_part(label: str) -> str:
    if "_" in label:
        return label.split("_", 1)[0]
    if label in INH_SUFFIXES:
        return "All"
    return label


def class_part(label: str) -> str:
    if "_" in label:
        return label.split("_", 1)[1]
    return label


def sort_types(labels: Sequence[str], mode: str) -> List[str]:
    def layer_key(lbl: str) -> Tuple[int, int, str]:
        layer = layer_part(lbl)
        cls = class_part(lbl)
        layer_idx = LAYER_ORDER.get(layer, len(LAYER_ORDER))
        class_idx = CLASS_ORDER.get(cls, len(CLASS_ORDER))
        return (layer_idx, class_idx, lbl)

    def class_key(lbl: str) -> Tuple[int, int, str]:
        cls = "Exc" if lbl in EXC_ORDER else class_part(lbl)
        if cls == "Exc":
            order_idx = EXC_ORDER.get(lbl, 100)
            return (CLASS_ORDER.get(cls, len(CLASS_ORDER)), order_idx, lbl)
        layer_idx = LAYER_ORDER.get(layer_part(lbl), len(LAYER_ORDER))
        return (CLASS_ORDER.get(cls, len(CLASS_ORDER)), layer_idx, lbl)

    key_fn = layer_key if mode == "layer" else class_key
    return sorted(labels, key=key_fn)


def build_type_mapping(
    types: Sequence[str],
    l5_mode: str = "split",
    inh_mode: str = "layer",
    sort_mode: str = "layer",
) -> TypeMapping:
    remapped = []
    for t in types:
        label = map_l5(t, l5_mode)
        label = map_inhibitory(label, inh_mode)
        remapped.append(label)
    ordered = sort_types(sorted(set(remapped)), sort_mode)
    index_lookup = {label: i for i, label in enumerate(ordered)}
    indices = np.array([index_lookup[label] for label in remapped], dtype=np.int64)
    return TypeMapping(labels=ordered, indices=indices)


def aggregate_square_sum(
    values: np.ndarray,
    indices: np.ndarray,
    out_size: int,
) -> np.ndarray:
    agg = np.zeros((out_size, out_size), dtype=np.float64)
    n = len(indices)
    for i in range(n):
        src_idx = int(indices[i])
        for j in range(n):
            tgt_idx = int(indices[j])
            val = values[i, j]
            if np.isfinite(val):
                agg[src_idx, tgt_idx] += float(val)
    return agg


def aggregate_square_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    out_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    weight_sum = np.zeros((out_size, out_size), dtype=np.float64)
    total = np.zeros((out_size, out_size), dtype=np.float64)
    n = len(indices)
    for i in range(n):
        src_idx = int(indices[i])
        for j in range(n):
            tgt_idx = int(indices[j])
            w = weights[i, j]
            if not np.isfinite(w) or w <= 0:
                continue
            v = values[i, j]
            if not np.isfinite(v):
                continue
            weight_sum[src_idx, tgt_idx] += float(w)
            total[src_idx, tgt_idx] += float(v) * float(w)
    out = np.full((out_size, out_size), np.nan, dtype=np.float64)
    mask = weight_sum > 0
    out[mask] = total[mask] / weight_sum[mask]
    return out, weight_sum
