#!/usr/bin/env python3
"""Build 4x4 outgoing weight fraction table (source high/low exc/inh to target high/low exc/inh)."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "core_nll_0"
NODE_SET_DIR = BASE_DIR / "node_sets"
OUTPUT_DIR = BASE_DIR / "figures" / "selectivity_outgoing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_FILE = BASE_DIR / "network" / "v1_v1_edges_bio_trained.h5"

GROUPS = {
    'exc_high': 'high_outgoing_exc_nodes.json',
    'exc_low': 'low_outgoing_exc_nodes.json',
    'inh_high': 'high_outgoing_inh_nodes.json',
    'inh_low': 'low_outgoing_inh_nodes.json',
}


def load_node_set(name: str) -> set[int]:
    data = json.loads((NODE_SET_DIR / name).read_text())
    return {int(x) for x in data['node_id']}


def main() -> None:
    source_sets = {label: load_node_set(fname) for label, fname in GROUPS.items()}
    target_sets = source_sets  # same mapping

    with h5py.File(EDGE_FILE, 'r') as f:
        src = f['edges']['v1_to_v1']['source_node_id'][:].astype(np.int64)
        tgt = f['edges']['v1_to_v1']['target_node_id'][:].astype(np.int64)
        weights = np.abs(f['edges']['v1_to_v1']['0']['syn_weight'][:].astype(np.float64))

    results = []
    for src_label, src_ids in source_sets.items():
        src_mask = np.isin(src, list(src_ids))
        if not src_mask.any():
            continue
        total_weight = weights[src_mask].sum()
        for tgt_label, tgt_ids in target_sets.items():
            mask = src_mask & np.isin(tgt, list(tgt_ids))
            weight = weights[mask].sum()
            fraction = weight / total_weight if total_weight > 0 else np.nan
            results.append(
                {
                    'source': src_label,
                    'target': tgt_label,
                    'abs_weight': weight,
                    'fraction': fraction,
                }
            )

    df = pd.DataFrame(results)
    table = df.pivot(index='source', columns='target', values='fraction').reindex(index=GROUPS.keys(), columns=GROUPS.keys())
    abs_table = df.pivot(index='source', columns='target', values='abs_weight').reindex(index=GROUPS.keys(), columns=GROUPS.keys())

    out_fraction = OUTPUT_DIR / 'outgoing_weight_fraction_4x4.csv'
    out_abs = OUTPUT_DIR / 'outgoing_weight_abs_4x4.csv'
    table.to_csv(out_fraction)
    abs_table.to_csv(out_abs)

    print('Fraction table:
', table)
    print('
Absolute weight table:
', abs_table)
    print(f'Wrote {out_fraction} and {out_abs}')


if __name__ == '__main__':
    main()
