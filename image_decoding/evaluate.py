from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

from .preprocess import summarise_spikes_to_rates
from .decode import correlation_decoder, logistic_decoder, accuracy


@dataclass
class DecodingResult:
    cell_type: str
    network: int
    decoder: str
    sample_size: int
    n_reps: int
    accuracy: float
    confusion: np.ndarray  # shape (n_images, n_images)

    def as_dict(self):
        d = self.__dict__.copy()
        d["confusion"] = self.confusion  # may be large; caller may drop
        return d


def _build_rate_tensor(cell_ids: List[int], base_dir: Path, n_reps: int):
    """Return tensor of shape (n_reps, 118, n_cells)."""
    rates_list = []
    for rep in range(n_reps):
        spike_path = base_dir / "output_bo_bio_trained" / f"chunk_{rep:02d}" / "spikes.h5"
        rates, _ = summarise_spikes_to_rates(spike_path, cell_ids)
        rates_list.append(rates.T)  # (118, n_cells)
    return np.stack(rates_list, axis=0)


def decode_crossval(
    cell_type: str,
    cell_ids: List[int],
    base_dir: Path,
    *,
    network_idx: int,
    n_reps: int,
    sample_size: int,
    decoder: str = "logistic",
    seed: int = 0,
) -> DecodingResult:
    """Decode with cross-validation."""
    rng = random.Random(seed)
    if len(cell_ids) > sample_size:
        cell_ids = rng.sample(cell_ids, sample_size)

    tensor = _build_rate_tensor(cell_ids, base_dir, n_reps)  # (R, 118, C)
    # Load stimulus identity table to get true label per repetition & frame
    stim_ids_path = Path("/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/digital_twin/bo_movies/stim_ids.npy")
    stim_ids_full = np.load(stim_ids_path)[:, :118]  # (50,118)
    stim_ids = stim_ids_full[:n_reps]
    if stim_ids.shape[0] < n_reps:
        raise ValueError(f"stim_ids contains {stim_ids.shape[0]} reps but n_reps={n_reps}")

    n_images = tensor.shape[1]
    confusion = np.zeros((n_images, n_images), dtype=int)
    correct = total = 0

    # 10-fold leave-repetition-out CV: each fold holds out 5 repetitions
    cv_folds = 10
    reps_per_fold = n_reps // cv_folds
    if reps_per_fold * cv_folds != n_reps:
        raise ValueError(
            f"n_reps={n_reps} is not divisible by {cv_folds}; expected multiples of 5 reps per fold"
        )

    for fold in range(cv_folds):
        test_slice = slice(fold * reps_per_fold, (fold + 1) * reps_per_fold)
        train_mask = np.ones(n_reps, dtype=bool)
        train_mask[test_slice] = False
        X_train = tensor[train_mask].reshape(-1, tensor.shape[2])
        y_train = stim_ids[train_mask].reshape(-1)
        X_test = tensor[test_slice].reshape(-1, tensor.shape[2])
        y_test = stim_ids[test_slice].reshape(-1)

        if decoder == "logistic":
            pred, _ = logistic_decoder(X_train, y_train, X_test)
        elif decoder == "correlation":
            pred, _ = correlation_decoder(X_train, y_train, X_test)
        else:
            raise ValueError(decoder)

        confusion += np.histogram2d(y_test, pred, bins=n_images, range=[[0, n_images], [0, n_images]])[0].astype(int)
        correct += (pred == y_test).sum()
        total += y_test.size

    acc = correct / total
    return DecodingResult(
        cell_type=cell_type,
        network=network_idx,
        decoder=decoder,
        sample_size=sample_size,
        n_reps=n_reps,
        accuracy=acc,
        confusion=confusion,
    ) 