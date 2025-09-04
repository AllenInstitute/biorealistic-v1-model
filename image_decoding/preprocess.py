"""Image-presentation spike binning utilities.

This module converts the raw *spikes.h5* output of a BMTK simulation that
uses the Allen Brain Observatory natural-image paradigm into compact
firing-rate matrices that can be fed into decoders.

Important implementation details
--------------------------------
1.  Timestamps in the SONATA spike files are stored in **milliseconds**.
    We convert them to seconds immediately so that subsequent arithmetic
    on ``start_offset`` and ``img_dur`` (specified in seconds) is
    consistent.
2.  The stimulus protocol is ::
        0.0   – 0.25  s   grey screen (discard)
        0.25  – 30.25 s   118 images presented back-to-back, 250 ms each
        30.25 – 30.50 s   trailing grey (ignored)

    Therefore we normally keep *exactly* 118 bins of width 0.25 s
    starting at ``start_offset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Sequence, Optional

import h5py
import numpy as np

# -----------------------------------------------------------------------------
# Low-level spike loader
# -----------------------------------------------------------------------------

def _load_spikes(spikes_path: str | Path, pop_name: str = "v1") -> Dict[str, np.ndarray]:
    """Return ``node_ids`` and ``timestamps`` from a BMTK *spikes.h5* file.

    The BMTK biorealistic model stores timestamps in **milliseconds**; we
    keep them as read here and let the caller decide on units.
    """
    with h5py.File(spikes_path, "r") as h5:
        grp = h5["spikes"][pop_name]
        node_ids = grp["node_ids"][()]
        timestamps = grp["timestamps"][()]
    return {"node_ids": node_ids, "timestamps": timestamps}

# -----------------------------------------------------------------------------
# Main helper: spikes → (cells × images) rate matrix
# -----------------------------------------------------------------------------

def summarise_spikes_to_rates(
    spikes_path: str | Path,
    cell_ids: Optional[Sequence[int]] = None,
    *,
    start_offset: float = 0.25,  # seconds
    img_dur: float = 0.25,       # seconds
    n_images: int = 118,
    pop_name: str = "v1",
    dtype: str | np.dtype = np.float32,
) -> Tuple[np.ndarray, Dict]:
    """Bin spikes into image presentations and return firing rates (Hz).

    Parameters
    ----------
    spikes_path
        Path to *spikes.h5* file.
    cell_ids
        Optional iterable of node IDs to include; if *None* uses all IDs
        present in the spike file.
    start_offset, img_dur, n_images
        Timing parameters **in seconds**.
    pop_name
        Population name in the SONATA hierarchy (default "v1").
    dtype
        Output dtype (``float32`` gives enough dynamic range).
    """
    data = _load_spikes(spikes_path, pop_name=pop_name)
    node_ids = data["node_ids"]
    # convert ms → seconds once here
    timestamps = data["timestamps"] * 1e-3

    # Restrict to requested cells ------------------------------------------------
    if cell_ids is None:
        cell_ids = np.unique(node_ids)
    else:
        cell_ids = np.asarray(cell_ids)
    id_to_row = {nid: i for i, nid in enumerate(cell_ids)}

    # Initialise counts
    counts = np.zeros((len(cell_ids), n_images), dtype=dtype)

    # Discard everything before grey-screen offset
    valid = timestamps >= start_offset
    node_ids = node_ids[valid]
    timestamps = timestamps[valid]

    # Convert to image index (0 … n_images-1)
    rel_t = timestamps - start_offset
    bin_idx = np.floor(rel_t / img_dur).astype(int)
    in_window = (bin_idx >= 0) & (bin_idx < n_images)
    node_ids = node_ids[in_window]
    bin_idx = bin_idx[in_window]

    # Vectorised accumulation --------------------------------------------------
    # Map node_ids → row indices (returns -1 for cells outside *cell_ids*)
    row_idx = np.array([id_to_row.get(gid, -1) for gid in node_ids], dtype=np.int32)
    valid_rows = row_idx >= 0
    if valid_rows.any():
        flat_index = row_idx[valid_rows] * n_images + bin_idx[valid_rows]
        flat_counts = np.bincount(flat_index, minlength=len(cell_ids) * n_images)
        counts = flat_counts.reshape(len(cell_ids), n_images).astype(dtype, copy=False)

    rates = counts / img_dur  # Hz
    meta = dict(
        cell_ids=cell_ids,
        start_offset=start_offset,
        img_dur=img_dur,
        n_images=n_images,
        pop_name=pop_name,
        spikes_path=str(spikes_path),
    )
    return rates.astype(dtype, copy=False), meta

# -----------------------------------------------------------------------------
# Convenience IO wrappers
# -----------------------------------------------------------------------------

def save_rates(rates: np.ndarray, meta: Dict, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, rates=rates, **meta)


def load_rates(rate_path: str | Path) -> Tuple[np.ndarray, Dict]:
    with np.load(rate_path, allow_pickle=True) as npz:
        rates = npz["rates"]
        meta_keys = [k for k in npz.keys() if k != "rates"]
        meta = {k: npz[k].item() if npz[k].shape == () else npz[k] for k in meta_keys}
    return rates, meta
 