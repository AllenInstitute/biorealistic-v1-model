import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from image_decoding.run_celltype_decoding import gather_node_metadata
from image_decoding.evaluate import decode_crossval


DEFAULT_SAMPLE_SIZES = [10, 30, 50]
DEFAULT_DECODERS = ["logistic"]  # can add "correlation"


def get_cell_type_order():
    """Return custom cell type ordering: Exc, PV, SST, VIP, Inh subtypes."""
    # Define the desired order based on subtypes
    order = [
        # Excitatory cells by layer
        "L1_Inh",  # L1 only has inhibitory cells
        "L2/3_Exc",
        "L4_Exc", 
        "L5_ET",   # L5 excitatory subtypes between L4 and L6
        "L5_IT",
        "L5_NP", 
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


def run_analysis(
    networks: List[int],
    n_reps: int,
    sample_sizes: List[int],
    decoders: List[str],
    min_cells: int,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for net in networks:
        base = Path(f"core_nll_{net}")
        meta = gather_node_metadata(
            base / "network" / "v1_node_types.csv", base / "network" / "v1_nodes.h5"
        )
        for ctype, grp in meta.groupby("cell_type"):
            if len(grp) < min_cells:
                continue
            cell_ids = grp["node_id"].tolist()
            for dec in decoders:
                for sample in sample_sizes:
                    res = decode_crossval(
                        ctype,
                        cell_ids,
                        base,
                        network_idx=net,
                        n_reps=n_reps,
                        sample_size=sample,
                        decoder=dec,
                    )
                    results.append(res.as_dict())
                    print(f"net{net} {ctype} {dec} n{sample}: acc={res.accuracy:.3f}")

    # save raw results
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "decoding_summary.csv", index=False)

    # Get custom ordering
    cell_type_order = get_cell_type_order()
    available_types = set(df["cell_type"].unique())
    filtered_order = [ct for ct in cell_type_order if ct in available_types]

    # plot mean accuracy across networks vs cell_type
    plt.figure(figsize=(12, 4))
    sns.barplot(data=df, x="cell_type", y="accuracy", hue="sample_size", errorbar="se", order=filtered_order)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_barplot.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full decoding analysis over networks & parameters.")
    parser.add_argument("--networks", type=int, nargs="*", default=list(range(10)))
    parser.add_argument("--n_reps", type=int, default=50)
    parser.add_argument("--sample_sizes", type=int, nargs="*", default=DEFAULT_SAMPLE_SIZES)
    parser.add_argument("--decoders", nargs="*", default=DEFAULT_DECODERS)
    parser.add_argument("--min_cells", type=int, default=30)
    parser.add_argument("--outdir", type=str, default="image_decoding/summary")
    args = parser.parse_args()

    run_analysis(
        networks=args.networks,
        n_reps=args.n_reps,
        sample_sizes=args.sample_sizes,
        decoders=args.decoders,
        min_cells=args.min_cells,
        out_dir=Path(args.outdir),
    ) 