#!/usr/bin/env python3
from __future__ import annotations
import os
from analysis_shared.em_compare import (
    load_em_pd_pickle,
    load_em_corr_pickle,
    em_pd_matrix_plot,
    em_corr_matrix_plot,
    compute_em_pd_pvalues,
    plot_pd_violin_with_em,
)
from analysis_shared.pd_mc import compute_pd_mc_pvalues


def main():
    out_dir = os.path.join('figures', 'em_compare')
    os.makedirs(out_dir, exist_ok=True)

    pd_pkl = os.path.join('analysis_shared', 'delta_pref_direction_vs_weight_v1dd_250827.pkl')
    corr_pkl = os.path.join('analysis_shared', 'corr_vs_weight_minnie_250828.pkl')

    # Load EM data
    em_pd = load_em_pd_pickle(pd_pkl)
    em_corr = load_em_corr_pickle(corr_pkl)

    # 1) EM matrix plots
    em_pd_matrix_plot(em_pd, os.path.join(out_dir, 'em_pd_matrix.png'))
    em_corr_matrix_plot(em_corr, os.path.join(out_dir, 'em_corr_matrix.png'))

    # 2) EM vs simulation violin: compute EM pvals and MC
    em_pvals = compute_em_pd_pvalues(em_pd)

    # Use per-pair PD CSV limits if available; else none
    pd_limits_csv = 'pair_limits_pd.csv' if os.path.isfile('pair_limits_pd.csv') else None

    # bio_trained and naive across all core_nll_*
    bases = [f'core_nll_{i}' for i in range(10) if os.path.isdir(f'core_nll_{i}')]

    mc_bio = compute_pd_mc_pvalues(bases, 'bio_trained', aggregate_l5_types=True, resamples=100, connections_per_draw=None, seed=0, pair_limits_csv=pd_limits_csv)
    plot_pd_violin_with_em(mc_bio, em_pvals, os.path.join(out_dir, 'pd_violin_bio_trained.png'))

    mc_naive = compute_pd_mc_pvalues(bases, 'naive', aggregate_l5_types=True, resamples=100, connections_per_draw=None, seed=0, pair_limits_csv=pd_limits_csv)
    plot_pd_violin_with_em(mc_naive, em_pvals, os.path.join(out_dir, 'pd_violin_naive.png'))

    for f in ['em_pd_matrix.png','em_corr_matrix.png','pd_violin_bio_trained.png','pd_violin_naive.png']:
        p = os.path.join(out_dir, f)
        print(('OK ' if os.path.isfile(p) else 'MISSING ') + p)


if __name__ == '__main__':
    main()

