import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


NP_DIR = Path(__file__).resolve().parent
CACHE_DIR = NP_DIR / "cached_rates"
OUT_DIR = NP_DIR / "summary"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def load_cached_np(session_id: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    base = CACHE_DIR / session_id
    rates = np.load(base / "rates_core.npy", mmap_mode="r")  # (reps, images, cells)
    labels = np.load(base / "labels_core.npy")  # (reps, images)
    meta = pd.read_csv(base / "meta_core.csv")
    return rates, labels, meta


def calculate_lifetime_sparsity(rates: np.ndarray) -> float:
    """Rolls & Tovee lifetime sparsity across stimuli for one neuron.

    rates: (n_images,) firing rates in Hz
    returns NaN if mean rate is zero
    """
    if rates.size == 0:
        return np.nan
    mean_rate = float(np.mean(rates))
    if mean_rate == 0.0:
        return np.nan
    mean_sq = float(np.mean(rates ** 2))
    n = rates.size
    return (1.0 - (mean_rate ** 2) / mean_sq) / (1.0 - 1.0 / n)


def compute_session_sparsity(session_id: str) -> pd.DataFrame:
    rates, _labels, meta = load_cached_np(session_id)
    # Align meta length and tensor cells if there is any mismatch
    n_cells_tensor = rates.shape[2]
    n_meta = len(meta)
    if n_meta != n_cells_tensor:
        n = min(n_meta, n_cells_tensor)
        meta = meta.iloc[:n].reset_index(drop=True)
        rates = rates[:, :, :n]
    # Average across repetitions → image-wise mean response per unit
    mean_rates = rates.mean(axis=0)  # (images, cells)

    values = []
    for unit_idx in range(mean_rates.shape[1]):
        ls = calculate_lifetime_sparsity(mean_rates[:, unit_idx])
        row = meta.iloc[unit_idx]
        values.append(
            {
                "session": session_id,
                "unit_index": unit_idx,
                "unit_id": row["unit_id"],
                "cell_type": row["cell_type"],
                # Rename to image_selectivity for clarity; keep legacy column too
                "image_selectivity": ls,
                "lifetime_sparsity": ls,
            }
        )
    return pd.DataFrame(values)


def main():
    parser = argparse.ArgumentParser(description="Compute lifetime sparsity for Neuropixels cached sessions.")
    parser.add_argument("--sessions", nargs="*", default=None, help="Optional list of NP session IDs to include")
    args = parser.parse_args()

    session_dirs = sorted([p.name for p in CACHE_DIR.iterdir() if p.is_dir()])
    if args.sessions:
        session_dirs = [s for s in session_dirs if s in set(args.sessions)]
    if len(session_dirs) == 0:
        raise SystemExit(f"No cached sessions found in {CACHE_DIR}")

    by_unit_frames = []
    for sess in session_dirs:
        try:
            df_sess = compute_session_sparsity(sess)
        except FileNotFoundError:
            # Skip incomplete sessions
            continue
        by_unit_frames.append(df_sess)

    if len(by_unit_frames) == 0:
        raise SystemExit("No session data available for sparsity computation.")

    by_unit = pd.concat(by_unit_frames, ignore_index=True)
    # Save with new naming (image_selectivity) and keep legacy filenames
    by_unit.to_csv(OUT_DIR / "selectivity_neuropixels_by_unit.csv", index=False)
    by_unit.to_csv(OUT_DIR / "sparsity_neuropixels_by_unit.csv", index=False)

    # Aggregate by cell type (across all sessions)
    grp = by_unit.groupby("cell_type")["image_selectivity"]
    by_type = (
        pd.DataFrame(
            {
                "cell_type": grp.mean().index,
                "mean_image_selectivity": grp.mean().values,
                "std_image_selectivity": grp.std(ddof=1).values,
                "n_units": grp.count().values,
            }
        )
        .sort_values("mean_lifetime_sparsity")
        .reset_index(drop=True)
    )
    # Keep legacy column names for compatibility
    by_type_legacy = by_type.rename(columns={
        "mean_image_selectivity": "mean_lifetime_sparsity",
        "std_image_selectivity": "std_lifetime_sparsity",
    })
    by_type.to_csv(OUT_DIR / "selectivity_neuropixels_by_type.csv", index=False)
    by_type_legacy.to_csv(OUT_DIR / "sparsity_neuropixels_by_type.csv", index=False)

    # Also provide per-session per-type summary
    sess_type = (
        by_unit.groupby(["session", "cell_type"])  # type: ignore[arg-type]
        ["image_selectivity"]
        .agg(["mean", "std", "count"])  # type: ignore[list-item]
        .reset_index()
        .rename(columns={"mean": "mean_image_selectivity", "std": "std_image_selectivity", "count": "n_units"})
    )
    # Save both new and legacy
    sess_type.to_csv(OUT_DIR / "selectivity_neuropixels_by_session_and_type.csv", index=False)
    sess_type_legacy = sess_type.rename(columns={
        "mean_image_selectivity": "mean_lifetime_sparsity",
        "std_image_selectivity": "std_lifetime_sparsity",
    })
    sess_type_legacy.to_csv(OUT_DIR / "sparsity_neuropixels_by_session_and_type.csv", index=False)


if __name__ == "__main__":
    main()


