import argparse
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from image_decoding.run_decoding_cached import load_cached, decode_one, get_cell_type_order


DEFAULT_NETWORKS = list(range(10))
REP_OPTIONS = [10, 30, 50]
CELL_SAMPLE_SIZES = [10, 30, 50]
NETWORK_TYPES = ["bio_trained", "naive"]


def run_full_analysis(
    networks=DEFAULT_NETWORKS,
    rep_options=REP_OPTIONS,
    sample_sizes=CELL_SAMPLE_SIZES,
    network_types=NETWORK_TYPES,
    cv_folds: int = 10,
    min_cells: int = 30,
    seed: int = 0,
    n_jobs: int = 4,
    fit_n_jobs: int = -1,
    outdir: Path = Path("image_decoding/summary_full"),
):
    outdir.mkdir(parents=True, exist_ok=True)
    results = []
    rng = random.Random(seed)

    # Build list of decoding tasks -----------------------------------------
    tasks = []
    for net in networks:
        for net_type in network_types:
            # cached tensors ------------------------------------------------
            rates, labels, meta = load_cached(net, network_type=net_type)
            # rates shape: (50, cells, 118); labels shape: (50, 118)
            for n_reps in rep_options:
                rates_rep = rates[:n_reps]
                labels_rep = labels[:n_reps]

                for ctype, grp in meta.groupby("cell_type"):
                    if len(grp) < min_cells:
                        continue
                    idx_pool = grp.index.to_numpy()
                    # exclude completely silent cells --------------------
                    activity = rates_rep[:, idx_pool, :].sum(axis=(0, 2))
                    active_mask = activity > 0
                    if active_mask.sum() == 0:
                        continue
                    cell_pool = idx_pool[active_mask]

                    for n_cells in sample_sizes:
                        if len(cell_pool) < n_cells:
                            continue
                        tasks.append(
                            dict(
                                net=net,
                                net_type=net_type,
                                reps=n_reps,
                                ctype=ctype,
                                n_cells=n_cells,
                                cell_pool=cell_pool,
                            )
                        )

    print(f"Prepared {len(tasks)} decoding tasks …")

    # Helper for parallel execution ----------------------------------------
    def _run_task(task):
        net = task["net"]
        net_type = task["net_type"]
        reps = task["reps"]
        ctype = task["ctype"]
        n_cells = task["n_cells"]
        cell_pool = task["cell_pool"]

        local_rng = random.Random(seed + net * 991 + reps * 13 + n_cells)
        cell_idx = np.array(local_rng.sample(list(cell_pool), n_cells))

        # Reload tensors (memory-mapped) per worker to avoid pickle overhead
        rates, labels, _ = load_cached(net, network_type=net_type)
        rates_rep = rates[:reps]
        labels_rep = labels[:reps]

        acc = decode_one(
            cell_idx,
            rates_rep,
            labels_rep,
            cv_folds=cv_folds,
            seed=seed,
            fit_n_jobs=fit_n_jobs,
        )
        return {
            "network": net,
            "network_type": net_type,
            "n_reps": reps,
            "cell_type": ctype,
            "sample_size": n_cells,
            "accuracy": acc,
        }

    # Run tasks -------------------------------------------------------------
    with joblib.Parallel(n_jobs=n_jobs, verbose=10) as parallel:
        results.extend(parallel(joblib.delayed(_run_task)(t) for t in tasks))

    df = pd.DataFrame(results)
    csv_path = outdir / "decoding_summary_full.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to {csv_path}")

    # Plotting --------------------------------------------------------------
    cell_order = get_cell_type_order()
    available = set(df["cell_type"].unique())
    cell_order = [c for c in cell_order if c in available]

    for net_type in network_types:
        for reps in rep_options:
            sub = df[(df["network_type"] == net_type) & (df["n_reps"] == reps)]
            if sub.empty:
                continue
            plt.figure(figsize=(12, 4))
            sns.barplot(
                data=sub,
                x="cell_type",
                y="accuracy",
                hue="sample_size",
                errorbar="se",
                order=cell_order,
            )
            plt.title(f"{net_type.replace('_', ' ')} • {reps} repetitions")
            plt.xticks(rotation=90)
            sns.despine()
            plt.tight_layout()
            fig_path = outdir / f"accuracy_{net_type}_reps{reps}.png"
            plt.savefig(fig_path, dpi=300)
            plt.close()
            print(f"Saved figure {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Full decoding analysis (bio-trained vs naive)")
    parser.add_argument("--networks", type=int, nargs="*", default=DEFAULT_NETWORKS)
    parser.add_argument("--n_jobs", type=int, default=4, help="Parallel workers for task level")
    parser.add_argument("--fit_n_jobs", type=int, default=-1, help="n_jobs for scikit-learn LogisticRegression")
    parser.add_argument("--outdir", type=str, default="image_decoding/summary_full")
    args = parser.parse_args()

    run_full_analysis(
        networks=args.networks,
        n_jobs=args.n_jobs,
        fit_n_jobs=args.fit_n_jobs,
        outdir=Path(args.outdir),
    )
