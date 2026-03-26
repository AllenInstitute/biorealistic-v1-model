import argparse
import random
from pathlib import Path

import h5py
import numpy as np

from image_decoding.preprocess import summarise_spikes_to_rates, save_rates
from image_decoding.decode import correlation_decoder, accuracy


def main():
    parser = argparse.ArgumentParser(description="Prototype image decoding using correlation-based classifier.")
    parser.add_argument("--network", type=int, default=0, help="core_nll network index (0-9)")
    parser.add_argument("--n_chks", type=int, default=5, help="Number of chunks (repetitions) to use")
    parser.add_argument("--cells", type=int, default=100, help="Number of random cells to sample")
    args = parser.parse_args()

    base = Path(f"core_nll_{args.network}")
    nodes_path = base / "network" / "v1_nodes.h5"
    with h5py.File(nodes_path, "r") as h5:
        all_ids = h5["nodes"]["v1"]["node_id"][()]
    random.seed(0)
    cell_sample = random.sample(list(all_ids), args.cells)

    # prepare data containers
    all_rates = []
    all_labels = []
    rep_ids = []

    for rep in range(args.n_chks):
        spike_path = base / "output_bo_bio_trained" / f"chunk_{rep:02d}" / "spikes.h5"
        rates, meta = summarise_spikes_to_rates(spike_path, cell_sample)
        # persist to disk for reuse
        out_file = Path("image_decoding/data") / f"rates_net{args.network}_rep{rep:02d}.npz"
        save_rates(rates, meta, out_file)
        # transpose to (n_images, n_cells)
        all_rates.append(rates.T)
        all_labels.extend(range(rates.shape[1]))  # 0..117
        rep_ids.extend([rep] * rates.shape[1])

    all_rates = np.concatenate(all_rates, axis=0)
    all_labels = np.asarray(all_labels)
    rep_ids = np.asarray(rep_ids)

    # train on first n_chks-1 rep, test on last rep
    train_mask = rep_ids < (args.n_chks - 1)
    test_mask = ~train_mask

    pred, _ = correlation_decoder(all_rates[train_mask], all_labels[train_mask], all_rates[test_mask])
    acc = accuracy(pred, all_labels[test_mask])
    print(f"Correlation decoder accuracy (rep {args.n_chks-1} held out): {acc:.3f}")


if __name__ == "__main__":
    main() 