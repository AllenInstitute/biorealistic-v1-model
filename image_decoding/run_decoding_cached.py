import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from image_decoding.plot_utils import (
    cell_type_order as ordered_cell_types,
    load_cell_type_colors,
    get_subtype_colors_from_scheme,
    add_background_shading,
)
import joblib  # import Parallel, delayed

from image_decoding.decode import logistic_decoder, accuracy


def get_cell_type_order():
    # Use shared ordering (L1_Inh at the end)
    return ordered_cell_types()


def load_cached(network: int, network_type: str = "bio_trained"):
    base = Path(f"core_nll_{network}") / f"cached_rates_{network_type}"
    rates = np.load(base / "rates_core.npy", mmap_mode="r")  # (50, cells, 118)
    labels = np.load(base / "labels_core.npy")  # (50, 118)
    meta = pd.read_parquet(base / "meta_core.parquet")
    return rates, labels, meta


def decode_one(
    sample_size: int,
    tensor: np.ndarray,
    labels_tensor: np.ndarray,
    idx_pool: np.ndarray,
    *,
    cv_folds: int = 10,
    seed: int = 0,
    fit_n_jobs: int = -1,
    sample_strategy: str = "fixed",
):
    """Decode with 10-fold CV using a fixed neuron subset or resampling per fold.

    tensor shape: (reps, cells, images). We transpose to (reps, images, cells) after selecting cells.
    sample_strategy: "fixed" uses one subset across folds; "per_fold" resamples each fold.
    """
    rng = np.random.default_rng(seed)
    reps = tensor.shape[0]
    fold_size = reps // cv_folds
    rep_idx = np.arange(reps)
    rng.shuffle(rep_idx)

    labels_rep_img = labels_tensor  # (rep, img)
    accs = []

    fixed_subset = None
    if sample_strategy == "fixed":
        fixed_subset = np.array(rng.choice(idx_pool, size=sample_size, replace=False))

    for f in range(cv_folds):
        if sample_strategy == "per_fold":
            fold_rng = np.random.default_rng(seed * 101 + f)
            cell_idx = np.array(fold_rng.choice(idx_pool, size=sample_size, replace=False))
        else:
            cell_idx = fixed_subset

        # Select cells then transpose to (rep, img, cell)
        X_rep_cell_img = tensor[:, cell_idx, :]
        X_rep_img_cell = X_rep_cell_img.transpose(0, 2, 1)

        test_reps = rep_idx[f * fold_size : (f + 1) * fold_size]
        train_reps = np.setdiff1d(rep_idx, test_reps)
        train_X = X_rep_img_cell[train_reps].reshape(-1, X_rep_img_cell.shape[-1])
        test_X = X_rep_img_cell[test_reps].reshape(-1, X_rep_img_cell.shape[-1])
        train_y = labels_rep_img[train_reps].reshape(-1)
        test_y = labels_rep_img[test_reps].reshape(-1)

        # drop gray frames (id == 118)
        mask_train = train_y < 118
        mask_test = test_y < 118
        train_X = train_X[mask_train]
        train_y = train_y[mask_train]
        test_X = test_X[mask_test]
        test_y = test_y[mask_test]

        pred, _ = logistic_decoder(train_X, train_y, test_X, max_iter=1000, n_jobs=fit_n_jobs)
        accs.append(accuracy(pred, test_y))
    return float(np.mean(accs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--networks", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--sample_sizes", type=int, nargs="*", default=[10, 30, 50])
    parser.add_argument("--cv_folds", type=int, default=10)
    parser.add_argument("--min_cells", type=int, default=30)
    parser.add_argument("--outdir", type=str, default="image_decoding/summary50")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs for outer task-level joblib (-1 uses all cores)")
    parser.add_argument("--fit_n_jobs", type=int, default=-1, help="n_jobs passed to scikit-learn LogisticRegression")
    parser.add_argument("--verbose", type=int, default=5, help="Joblib verbosity (prints ETA)")
    parser.add_argument("--network_type", type=str, default="bio_trained", choices=["bio_trained", "naive", "checkpoint", "plain", "adjusted"], help="Suffix used in cached_rates_<network_type> directory (default: bio_trained)")
    parser.add_argument("--combine_cell_types", action="store_true", help="Pool all cells regardless of cell type")
    parser.add_argument("--sample_strategy", type=str, default="fixed", choices=["fixed", "per_fold"], help="Neuron sampling strategy across CV folds")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    rng = random.Random(args.seed)

    # Prepare list of tasks -----------------------------------------------
    tasks = []
    for net in args.networks:
        rates, labels, meta = load_cached(net, network_type=args.network_type)
        if args.combine_cell_types:
            idx_pool = meta.index.to_numpy()
            activity = rates[:, idx_pool, :].sum(axis=(0, 2))
            cell_pool = idx_pool[activity > 0]
            for n in args.sample_sizes:
                if len(cell_pool) < n:
                    print(f"net{net} All: only {len(cell_pool)} cells (<{n}); skipping")
                    continue
                tasks.append((net, "All", n, cell_pool, rates, labels))
        else:
            for ctype, grp in meta.groupby("cell_type"):
                if len(grp) < args.min_cells:
                    continue
                grp_idx = grp.index.to_numpy()
                activity = rates[:, grp_idx, :].sum(axis=(0, 2))
                active_mask = activity > 0
                if active_mask.sum() == 0:
                    print(f"net{net} {ctype}: no active cells, skipping")
                    continue
                cell_pool = grp_idx[active_mask]
                for n in args.sample_sizes:
                    if len(cell_pool) < n:
                        print(f"net{net} {ctype}: only {len(cell_pool)} cells (<{n}); skipping")
                        continue
                    tasks.append((net, ctype, n, cell_pool, rates, labels))

    def _decode_task(task):
        net, ctype, n, cell_pool, rates_t, labels_t = task
        acc = decode_one(
            n,
            rates_t,
            labels_t,
            np.array(cell_pool),
            cv_folds=args.cv_folds,
            seed=args.seed,
            fit_n_jobs=args.fit_n_jobs,
            sample_strategy=args.sample_strategy,
        )
        return {
            "network": net,
            "cell_type": ctype,
            "sample_size": n,
            "accuracy": acc,
        }

    # Run in parallel ------------------------------------------------------
    with joblib.Parallel(n_jobs=args.n_jobs, verbose=args.verbose) as parallel:
        task_results = parallel(joblib.delayed(_decode_task)(t) for t in tasks)

    results.extend(task_results)

    df = pd.DataFrame(results)
    # Avoid overwriting: suffix indicates pooled vs per-cell-type
    mode_suffix = "_pooled" if args.combine_cell_types else "_by_type"
    df.to_csv(out_dir / f"decoding_summary_{args.network_type}{mode_suffix}.csv", index=False)

    # Get custom ordering
    cell_type_order = get_cell_type_order()
    available_types = list(df["cell_type"].unique())
    filtered_order = (["All"] if "All" in available_types else []) + [ct for ct in cell_type_order if ct in available_types]

    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    # Background shading by subtype blocks
    colors_df = load_cell_type_colors()
    subtype_bg = get_subtype_colors_from_scheme(colors_df)
    add_background_shading(ax, filtered_order, subtype_bg)

    sns.barplot(
        data=df,
        x="cell_type",
        y="accuracy",
        hue="sample_size",
        errorbar="se",
        order=filtered_order,
        ax=ax,
    )
    plt.xticks(rotation=90)
    sns.despine(ax=ax)
    # Horizontal legend
    leg = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.17), ncol=len(sorted(df["sample_size"].unique())))
    plt.tight_layout()
    plt.savefig(out_dir / f"accuracy_barplot_{args.network_type}{mode_suffix}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main() 