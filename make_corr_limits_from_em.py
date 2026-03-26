#!/usr/bin/env python3
from __future__ import annotations
import os
import pickle
import pandas as pd

EM_PKL = os.path.join('analysis_shared', 'corr_vs_weight_minnie_250828.pkl')
OUT_CSV = 'pair_limits_corr_from_em.csv'


def map_label(s: str) -> str:
    s = s.strip()
    if s.startswith('23P'): return 'L2/3_Exc'
    if s.startswith('4P'): return 'L4_Exc'
    if s.startswith('5P'): return 'L5_Exc'  # aggregate all 5P-* into L5_Exc
    if s.startswith('6P'): return 'L6_Exc'
    return s


def main():
    with open(EM_PKL, 'rb') as f:
        obj = pickle.load(f)
    rows = []
    for k, df in obj.items():
        if not isinstance(df, pd.DataFrame):
            continue
        try:
            a, b = k.split('->')
        except Exception:
            continue
        rows.append({'source': map_label(a), 'target': map_label(b), 'connections': int(len(df))})
    df = pd.DataFrame(rows, columns=['source','target','connections'])
    # Aggregate duplicates by (source,target)
    agg = df.groupby(['source','target'], as_index=False)['connections'].sum()
    agg.to_csv(OUT_CSV, index=False)
    print(f'Wrote {OUT_CSV} with {len(agg)} rows (aggregated from {len(df)} entries)')


if __name__ == '__main__':
    main()
