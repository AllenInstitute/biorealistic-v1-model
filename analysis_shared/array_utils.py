from __future__ import annotations

import numpy as np


def safe_bincount_sum(
    keys: np.ndarray, values: np.ndarray, *, minlength: int | None = None
) -> np.ndarray:
    """Fast sum over `keys` using bincount, with basic validation.

    - keys must be integer-like and non-negative
    - values must be same length as keys
    - minlength (if provided) forces output length
    """
    keys = np.asarray(keys, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if keys.shape[0] != values.shape[0]:
        raise ValueError("keys and values must have the same length")
    if keys.size == 0:
        n = int(minlength) if minlength is not None else 0
        return np.zeros(n, dtype=np.float64)
    if keys.min() < 0:
        raise ValueError("keys must be non-negative")
    if minlength is None:
        minlength = int(keys.max()) + 1
    return np.bincount(keys, weights=values, minlength=int(minlength)).astype(np.float64)


