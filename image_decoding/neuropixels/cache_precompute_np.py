import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import re

# Directory containing the raw ns_spike_counts_*.npz files
NP_DIR = Path(__file__).resolve().parent
RAW_DIR = NP_DIR
CACHE_DIR = NP_DIR / "cached_rates"
CACHE_DIR.mkdir(exist_ok=True)

METRICS_CSV = "neuropixels/metrics/OSI_DSI_neuropixels_v4.csv"

# Map "L2/3 PV" → "L2/3_PV" etc. so it matches the 19-type naming scheme.

def normalise_cell_type(name: str) -> str:
    return name.replace(" ", "_")


def build_cache_for_session(npz_path: Path, metrics_df: pd.DataFrame):
    session_id_match = re.search(r"ns_spike_counts_(\d+)\.npz", npz_path.name)
    if not session_id_match:
        raise ValueError(f"Unexpected file name {npz_path.name}")
    session_id = session_id_match.group(1)

    data = np.load(npz_path)
    counts = data["spike_counts"].astype(np.float32)  # (presentations, units)
    unit_ids = data["unit_ids"]
    scene_ids = data["scene_ids"]

    # Filter out blank-screen presentations (scene_id == -1)
    valid = scene_ids >= 0
    counts = counts[valid]
    scene_ids = scene_ids[valid]

    # Determine repetition count (presentations should be reps*118)
    present_per_rep = 118
    reps = counts.shape[0] // present_per_rep
    if reps * present_per_rep != counts.shape[0]:
        print(f"{npz_path.name}: rows ({counts.shape[0]}) not divisible by 118, skipping session")
        return

    counts = counts.reshape(reps, present_per_rep, counts.shape[1])  # (R,118,N)
    rates = counts / 0.25  # Hz (same convention as model)

    # Build labels tensor (scene ids 0..117) per rep; assume same order each rep
    labels = scene_ids.reshape(reps, present_per_rep)

    # Map unit IDs → cell_type using metrics CSV
    m = metrics_df.set_index("ecephys_unit_id").reindex(unit_ids)
    cell_types_raw = m["cell_type"].fillna("").values
    cell_types = np.array([normalise_cell_type(ct) for ct in cell_types_raw])

    # Keep only units with a recognised cell_type string (non-empty)
    keep = cell_types != ""
    if keep.sum() < 10:
        print(f"{session_id}: <10 valid units after filtering, skipping")
        return

    # Apply unit filter to the rates tensor to keep meta and data aligned
    rates = rates[:, :, keep]
    unit_ids_keep = unit_ids[keep]
    cell_types_keep = cell_types[keep]

    # Save ------------------------------------------------------------------
    save_dir = CACHE_DIR / session_id
    save_dir.mkdir(exist_ok=True)

    np.save(save_dir / "rates_core.npy", rates.astype(np.float16))
    np.save(save_dir / "labels_core.npy", labels.astype(np.int16))

    meta = pd.DataFrame({"unit_id": unit_ids_keep, "cell_type": cell_types_keep})
    meta.to_csv(save_dir / "meta_core.csv", index=False)
    print(f"Cached session {session_id}: reps={reps} units={len(unit_ids_keep)} -> {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Precompute cached firing-rate tensors for Neuropixels natural-scene data.")
    parser.add_argument("--sessions", nargs="*", default=None, help="Optional list of session IDs to process; default = all available")
    args = parser.parse_args()

    # Use regex separator to replace deprecated delim_whitespace
    metrics_df = pd.read_csv(METRICS_CSV, sep='\s+')

    npz_files = sorted(RAW_DIR.glob("ns_spike_counts_*.npz"))
    if args.sessions:
        npz_files = [p for p in npz_files if any(s in p.name for s in args.sessions)]

    for p in npz_files:
        try:
            build_cache_for_session(p, metrics_df)
        except Exception as e:
            print(f"[warning] Session {p.name} failed: {e}")


if __name__ == "__main__":
    main()
