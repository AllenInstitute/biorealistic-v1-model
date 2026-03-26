import argparse
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from image_decoding.preprocess import summarise_spikes_to_rates
from image_decoding.decode import correlation_decoder, logistic_decoder, accuracy
from image_decoding.utils import pop_to_cell_type


DEF_MIN_CELLS = 20  # skip cell-types with fewer cells than this


def gather_node_metadata(nodes_csv: Path, nodes_h5: Path, core_radius: float = 200.0):
    """Return DataFrame with node_id, pop_name, cell_type."""
    pop2ctype = pop_to_cell_type()
    types_df = pd.read_csv(nodes_csv, delim_whitespace=True, usecols=["node_type_id", "pop_name"])
    # Load node_type_id per node_id from h5
    with h5py.File(nodes_h5, "r") as h5:
        node_grp = h5["nodes"]["v1"]
        node_ids = node_grp["node_id"][()]
        ntype_ids = node_grp["node_type_id"][()]
        subgrp = node_grp["0"]
        x = subgrp["x"][()]
        z = subgrp["z"][()]

    core_mask = (x ** 2 + z ** 2) < core_radius**2
    meta = pd.DataFrame({"node_id": node_ids, "node_type_id": ntype_ids, "core": core_mask})
    # add pop_name by merging types
    meta = meta.merge(types_df, on="node_type_id", how="left")
    # map to broader cell_type naming
    meta["cell_type"] = meta["pop_name"].map(pop2ctype)
    # keep only core neurons
    meta_core = meta[meta["core"]].copy()
    return meta_core


def decode_one_celltype(cell_ids, base_dir: Path, n_reps: int, n_sample: int = 30, seed: int = 0):
    """Decode a single cell-type using the generic 10-fold CV routine."""
    from image_decoding.evaluate import decode_crossval

    res = decode_crossval(
        cell_type="UNKNOWN",  # placeholder; caller can ignore
        cell_ids=list(cell_ids),
        base_dir=base_dir,
        network_idx=-1,
        n_reps=n_reps,
        sample_size=n_sample,
        decoder="logistic",
        seed=seed,
    )
    return res.accuracy



def main():
    parser = argparse.ArgumentParser(description="Decode images per cell type.")
    parser.add_argument("--network", type=int, default=0)
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--min_cells", type=int, default=DEF_MIN_CELLS)
    parser.add_argument("--sample_per_type", type=int, default=30, help="Subsample this many cells from each cell type")
    args = parser.parse_args()

    base = Path(f"core_nll_{args.network}")
    nodes_csv = base / "network" / "v1_node_types.csv"
    nodes_h5 = base / "network" / "v1_nodes.h5"
    meta = gather_node_metadata(nodes_csv, nodes_h5)

    results = {}
    for ctype, group in meta.groupby("cell_type"):
        if len(group) < args.min_cells:
            continue
        print(f"Processing {ctype} (n={len(group)}) …", flush=True)
        acc = decode_one_celltype(group["node_id"].tolist(), base, args.n_reps, n_sample=args.sample_per_type)
        results[ctype] = acc
        print(f"  → accuracy={acc:.3f}")

    # Save to csv
    out_dir = Path("image_decoding/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"decoding_accuracy_net{args.network}.csv"
    pd.Series(results, name="accuracy").to_csv(out_path)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main() 