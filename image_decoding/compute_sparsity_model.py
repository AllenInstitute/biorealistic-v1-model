import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_cached(network: int, network_type: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    base = Path(f"core_nll_{network}") / f"cached_rates_{network_type}"
    rates = np.load(base / "rates_core.npy", mmap_mode="r")  # (reps, cells, images)
    labels = np.load(base / "labels_core.npy")  # (reps, images)
    meta = pd.read_parquet(base / "meta_core.parquet")
    return rates, labels, meta


def calculate_lifetime_sparsity(rates_img: np.ndarray) -> float:
    # rates_img: (n_images,) image-wise responses
    if rates_img.size == 0:
        return np.nan
    mean_rate = float(np.mean(rates_img))
    if mean_rate == 0.0:
        return np.nan
    mean_sq = float(np.mean(rates_img ** 2))
    n = rates_img.size
    return (1.0 - (mean_rate ** 2) / mean_sq) / (1.0 - 1.0 / n)


def compute_network_sparsity(network: int, network_type: str) -> pd.DataFrame:
    rates, _labels, meta = load_cached(network, network_type)
    # Expected shape (reps, cells, images)
    if rates.ndim != 3:
        raise ValueError(f"Unexpected rates shape {rates.shape}")
    if rates.shape[2] not in (118, 119):
        raise ValueError(f"Unexpected image dimension {rates.shape}")
    # Average repetitions
    mean_rates = rates.mean(axis=0)  # (cells, images)
    # Restrict to 118 natural images if 119 present
    if mean_rates.shape[1] == 119:
        mean_rates = mean_rates[:, :118]

    values = []
    n_cells_tensor = mean_rates.shape[0]
    if len(meta) != n_cells_tensor:
        meta = meta.iloc[:n_cells_tensor].reset_index(drop=True)
    for ci in range(n_cells_tensor):
        ls = calculate_lifetime_sparsity(mean_rates[ci, :])
        row = meta.iloc[ci]
        values.append(
            {
                "network": network,
                "network_type": network_type,
                "node_id": row.get("node_id", np.nan),
                "pop_name": row.get("pop_name", ""),
                "cell_type": row.get("cell_type", ""),
                # Rename to image_selectivity; keep legacy column too
                "image_selectivity": ls,
                "lifetime_sparsity": ls,
            }
        )
    return pd.DataFrame(values)


def main():
    parser = argparse.ArgumentParser(description="Compute lifetime sparsity across images for simulated networks.")
    parser.add_argument("--networks", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--network_types", nargs="*", default=["bio_trained", "naive"], help="e.g. bio_trained naive")
    parser.add_argument("--outdir", type=str, default="image_decoding/summary")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True, parents=True)

    frames = []
    for nt in args.network_types:
        for net in args.networks:
            cache_dir = Path(f"core_nll_{net}") / f"cached_rates_{nt}"
            if not cache_dir.exists():
                # Skip if cache missing; user can run image_decoding/cache_precompute.py
                continue
            try:
                df = compute_network_sparsity(net, nt)
            except Exception as e:
                print(f"Skipping net{net} {nt}: {e}")
                continue
            frames.append(df)

    if not frames:
        raise SystemExit("No networks processed. Ensure caches exist (image_decoding/cache_precompute.py).")

    by_unit = pd.concat(frames, ignore_index=True)
    by_unit.to_csv(out_dir / "selectivity_model_by_unit.csv", index=False)
    by_unit.to_csv(out_dir / "sparsity_model_by_unit.csv", index=False)

    # Aggregate by cell type per network_type
    agg = (
        by_unit.groupby(["network_type", "cell_type"])  # type: ignore[arg-type]
        ["image_selectivity"]
        .agg(["mean", "std", "count"])  # type: ignore[list-item]
        .reset_index()
        .rename(columns={"mean": "mean_image_selectivity", "std": "std_image_selectivity", "count": "n_units"})
    )
    agg_legacy = agg.rename(columns={
        "mean_image_selectivity": "mean_lifetime_sparsity",
        "std_image_selectivity": "std_lifetime_sparsity",
    })
    agg.to_csv(out_dir / "selectivity_model_by_type.csv", index=False)
    agg_legacy.to_csv(out_dir / "sparsity_model_by_type.csv", index=False)

    # Per-network aggregation
    agg_net = (
        by_unit.groupby(["network", "network_type", "cell_type"])  # type: ignore[arg-type]
        ["image_selectivity"]
        .agg(["mean", "std", "count"])  # type: ignore[list-item]
        .reset_index()
        .rename(columns={"mean": "mean_image_selectivity", "std": "std_image_selectivity", "count": "n_units"})
    )
    agg_net_legacy = agg_net.rename(columns={
        "mean_image_selectivity": "mean_lifetime_sparsity",
        "std_image_selectivity": "std_lifetime_sparsity",
    })
    agg_net.to_csv(out_dir / "selectivity_model_by_network_and_type.csv", index=False)
    agg_net_legacy.to_csv(out_dir / "sparsity_model_by_network_and_type.csv", index=False)


if __name__ == "__main__":
    main()



