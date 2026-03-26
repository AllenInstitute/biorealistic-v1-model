from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def read_pair_limits_csv(path: str) -> Dict[Tuple[str, str], int]:
    df = pd.read_csv(path)
    # Accept columns: source,target,connections
    required = {"source", "target", "connections"}
    if not required.issubset(df.columns):
        raise ValueError(f"Pair limits CSV must contain columns {required}")
    mapping: Dict[Tuple[str, str], int] = {}
    for _, row in df.iterrows():
        s = str(row["source"]).strip()
        t = str(row["target"]).strip()
        try:
            n = None if pd.isna(row["connections"]) else int(row["connections"])
        except Exception:
            n = None
        if n is not None and n >= 0:
            mapping[(s, t)] = n
    return mapping


def apply_per_pair_sampling(df: pd.DataFrame, *, max_per_pair: int | None, pair_limits: Dict[Tuple[str, str], int] | None, rng: np.random.RandomState | None = None) -> pd.DataFrame:
    """Downsample per (source_type,target_type) to the provided limits; prefer pair_limits over max_per_pair."""
    if max_per_pair is None and not pair_limits:
        return df
    if rng is None:
        rng = np.random.RandomState(0)
    chunks = []
    for (s, t), g in df.groupby(["source_type", "target_type"], sort=False, observed=False):
        limit = None
        if pair_limits and (s, t) in pair_limits:
            limit = int(pair_limits[(s, t)])
        elif max_per_pair is not None:
            limit = int(max_per_pair)
        if limit is None or len(g) <= limit:
            chunks.append(g)
        else:
            chunks.append(g.sample(n=limit, random_state=rng))
    if not chunks:
        return df.iloc[0:0]
    return pd.concat(chunks, ignore_index=True)

