import argparse
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from image_decoding.decode import logistic_decoder, accuracy


def get_cell_type_order():
    """Return custom cell type ordering for Neuropixels data."""
    order = [
        # Excitatory cells by layer
        "L1_Htr3a",  # L1 inhibitory in Neuropixels
        "L2/3_Exc",
        "L4_Exc", 
        "L5_Exc",    # Simplified L5 naming in Neuropixels
        "L6_Exc",
        # PV cells by layer
        "L2/3_PV",
        "L4_PV",
        "L5_PV",
        "L6_PV",
        # SST cells by layer
        "L2/3_SST",
        "L4_SST", 
        "L5_SST",
        "L6_SST",
        # VIP cells by layer
        "L2/3_VIP",
        "L4_VIP",
        "L5_VIP",
        "L6_VIP",
    ]
    return order

CACHE_ROOT = Path(__file__).resolve().parent / "cached_rates"


def load_cached_np(session_id: str):
    base = CACHE_ROOT / session_id
    rates = np.load(base / "rates_core.npy", mmap_mode="r")  # (reps, images, cells)
    labels = np.load(base / "labels_core.npy")  # (reps, images)
    meta = pd.read_csv(base / "meta_core.csv")
    return rates, labels, meta


def decode_one(
    sample_size: int,
    tensor: np.ndarray,
    labels_tensor: np.ndarray,
    idx_pool: np.ndarray,
    cv_folds: int = 10,
    seed: int = 0,
    fit_n_jobs: int = -1,
    sample_strategy: str = "fixed",
):
    """Decode with 10-fold CV using a fixed neuron subset or resampling per fold.

    sample_strategy: "fixed" uses one subset across folds; "per_fold" resamples each fold.
    """
    rng = np.random.default_rng(seed)
    reps = tensor.shape[0]
    fold_size = reps // cv_folds
    rep_idx = np.arange(reps)
    rng.shuffle(rep_idx)

    labels_rep_img = labels_tensor
    accs: list[float] = []

    fixed_subset = None
    if sample_strategy == "fixed":
        fixed_subset = np.array(rng.choice(idx_pool, size=sample_size, replace=False))

    for f in range(cv_folds):
        if sample_strategy == "per_fold":
            fold_rng = np.random.default_rng(seed * 101 + f)
            cell_idx = np.array(fold_rng.choice(idx_pool, size=sample_size, replace=False))
        else:
            cell_idx = fixed_subset

        X_rep_img_cell = tensor[:, :, cell_idx]
        test_reps = rep_idx[f * fold_size: (f + 1) * fold_size]
        train_reps = np.setdiff1d(rep_idx, test_reps)
        train_X = X_rep_img_cell[train_reps].reshape(-1, X_rep_img_cell.shape[-1])
        test_X = X_rep_img_cell[test_reps].reshape(-1, X_rep_img_cell.shape[-1])
        train_y = labels_rep_img[train_reps].reshape(-1)
        test_y = labels_rep_img[test_reps].reshape(-1)
        pred, _ = logistic_decoder(train_X, train_y, test_X, max_iter=1000, n_jobs=fit_n_jobs)
        accs.append(accuracy(pred, test_y))

    return float(np.mean(accs))


def main():
    parser = argparse.ArgumentParser(description="Decoding of Neuropixels natural-scene data from cached tensors")
    parser.add_argument("--sessions", nargs="*", default=None, help="Session IDs to include; default = all cached")
    parser.add_argument("--sample_sizes", type=int, nargs="*", default=[10, 30, 50])
    parser.add_argument("--cv_folds", type=int, default=10)
    parser.add_argument("--min_cells", type=int, default=30)
    parser.add_argument("--outdir", type=str, default="image_decoding/neuropixels/summary")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--fit_n_jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=5)
    parser.add_argument("--combine_cell_types", action="store_true", help="Pool all cells regardless of cell type")
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="fixed",
        choices=["fixed", "per_fold"],
        help="Neuron sampling strategy across CV folds",
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sessions --------------------------------------------------------------
    sessions = sorted([p.name for p in CACHE_ROOT.iterdir() if p.is_dir()])
    if args.sessions:
        sessions = [s for s in sessions if s in args.sessions]

    results = []
    rng = random.Random(args.seed)

    tasks = []
    for sess in sessions:
        try:
            rates, labels, meta = load_cached_np(sess)
        except FileNotFoundError:
            print(f"Session cache {sess} missing, skipping")
            continue

        if args.combine_cell_types:
            idx_pool = meta.index.to_numpy()
            activity = rates[:, :, idx_pool].sum(axis=(0, 1))
            cell_pool = idx_pool[activity > 0]
            if len(cell_pool) >= max(args.sample_sizes):
                for n in args.sample_sizes:
                    if len(cell_pool) >= n:
                        tasks.append((sess, "All", n, cell_pool))
        else:
            for ctype, grp in meta.groupby("cell_type"):
                if len(grp) < args.min_cells:
                    continue
                idx_pool = grp.index.to_numpy()
                activity = rates[:, :, idx_pool].sum(axis=(0, 1))
                cell_pool = idx_pool[activity > 0]
                if len(cell_pool) == 0:
                    continue
                for n in args.sample_sizes:
                    if len(cell_pool) < n:
                        continue
                    tasks.append((sess, ctype, n, cell_pool))

    def _task(t):
        sess, ctype, n, pool = t
        rates, labels, _ = load_cached_np(sess)
        acc = decode_one(
            n,
            rates,
            labels,
            pool,
            cv_folds=args.cv_folds,
            seed=args.seed,
            fit_n_jobs=args.fit_n_jobs,
            sample_strategy=args.sample_strategy,
        )
        return {"session": sess, "cell_type": ctype, "sample_size": n, "accuracy": acc}

    with joblib.Parallel(n_jobs=args.n_jobs, verbose=args.verbose) as parallel:
        results.extend(parallel(joblib.delayed(_task)(t) for t in tasks))

    df = pd.DataFrame(results)
    # Avoid overwriting: suffix indicates pooled vs per-cell-type
    mode_suffix = "_pooled" if args.combine_cell_types else "_by_type"
    csv_path = out_dir / f"decoding_summary_neuropixels{mode_suffix}.csv"
    df.to_csv(csv_path, index=False)

    if df.empty:
        print("No cell types met the minimum cell-count criterion; no plot generated.")
        return

    order = get_cell_type_order()
    uniq = df["cell_type"].unique()
    order = (["All"] if "All" in uniq else []) + [o for o in order if o in uniq]

    plt.figure(figsize=(12, 4))
    sns.barplot(data=df, x="cell_type", y="accuracy", hue="sample_size", errorbar="se", order=order)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_dir / f"accuracy_neuropixels{mode_suffix}.png", dpi=300)
    plt.close()
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
