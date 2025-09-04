import argparse
from pathlib import Path

import numpy as np
import h5py
import pandas as pd

from image_decoding.preprocess import summarise_spikes_to_rates
from image_decoding.utils import pop_to_cell_type


def load_core_ids(base_dir: Path, radius: float = 200.0):
    nodes_h5 = base_dir / "network" / "v1_nodes.h5"
    node_types_csv = base_dir / "network" / "v1_node_types.csv"

    # positions
    with h5py.File(nodes_h5, "r") as h5:
        v1_grp = h5["nodes"]["v1"]
        node_ids = v1_grp["node_id"][()]
        ntype_ids = v1_grp["node_type_id"][()]
        pos = v1_grp["0"]
        x = pos["x"][()]
        z = pos["z"][()]
    core_mask = (x ** 2 + z ** 2) < radius ** 2

    # pop_name mapping to cell_type
    pop_map = pop_to_cell_type()
    types_df = pd.read_csv(node_types_csv, delim_whitespace=True, usecols=["node_type_id", "pop_name"])
    pop_names = types_df.set_index("node_type_id").loc[ntype_ids, "pop_name"].values
    cell_types = np.vectorize(pop_map.get)(pop_names)

    ids_core = node_ids[core_mask]
    pop_core = pop_names[core_mask]
    ctype_core = cell_types[core_mask]
    return ids_core, pop_core, ctype_core


def build_cache(network: int, network_type: str = "bio_trained"):
    """Build the cached firing-rate tensor for *network*.

    Parameters
    ----------
    network
        Integer index of the core_nll_* directory.
    network_type
        Suffix that identifies which simulation output to use, e.g.
        ``"bio_trained"`` (default) or ``"naive"``. The spike files are
        expected to live under
        ``core_nll_{network}/output_bo_{network_type}/chunk_##/spikes.h5``.
        The generated cache is written to a sibling directory named
        ``cached_rates_{network_type}`` to avoid clobbering different
        versions.
    """

    base = Path(f"core_nll_{network}")
    # keep separate caches per network_type so we can compare later
    save_dir = base / f"cached_rates_{network_type}"
    save_dir.mkdir(exist_ok=True)

    ids, pop_names, cell_types = load_core_ids(base)
    n_cells = len(ids)
    tensor = np.empty((50, n_cells, 118), dtype=np.float16)
    # load global stimulus id table
    stim_ids_path = Path("/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/digital_twin/bo_movies/stim_ids.npy")
    stim_ids = np.load(stim_ids_path)  # shape (50, 119) (118 natural + gray)
    labels_tensor = stim_ids[:, :118].astype(np.int16)  # drop last gray frame

    for rep in range(50):
        h5_path = base / f"output_bo_{network_type}" / f"chunk_{rep:02d}" / "spikes.h5"
        rates, _ = summarise_spikes_to_rates(h5_path, ids, dtype=np.float16)  # (cells, 118)
        tensor[rep] = rates  # store as (cells, 118)
        print(f"net{network} ({network_type}): rep {rep} cached", flush=True)

    # save results ---------------------------------------------------------
    np.save(save_dir / "rates_core.npy", tensor)
    np.save(save_dir / "labels_core.npy", labels_tensor)
    meta = pd.DataFrame({"node_id": ids, "pop_name": pop_names, "cell_type": cell_types})
    meta.to_parquet(save_dir / "meta_core.parquet")
    print(f"net{network} ({network_type}): cache written to {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--networks", type=int, nargs="*", default=list(range(10)))
    parser.add_argument(
        "--network_type",
        type=str,
        default="bio_trained",
        choices=["bio_trained", "naive", "checkpoint", "plain", "adjusted"],
        help="Suffix identifying simulation output folder (default: bio_trained)",
    )
    args = parser.parse_args()
    for net in args.networks:
        build_cache(net, network_type=args.network_type)


if __name__ == "__main__":
    main() 