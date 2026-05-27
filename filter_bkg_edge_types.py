import argparse

import h5py
import pandas as pd


def filter_bkg_edge_types(bkg_weights_path, edges_path, output_path):
    bkg_weights = pd.read_csv(bkg_weights_path, sep=" ")
    with h5py.File(edges_path, "r") as h5:
        used_edge_type_ids = sorted(set(h5["/edges/bkg_to_v1/edge_type_id"][()]))

    if len(used_edge_type_ids) != len(bkg_weights):
        raise ValueError(
            f"Found {len(used_edge_type_ids)} edge_type_ids in {edges_path}, "
            f"but {bkg_weights_path} has {len(bkg_weights)} rows."
        )

    edge_types = pd.DataFrame(
        {
            "edge_type_id": used_edge_type_ids,
            "target_query": [
                f"node_type_id=='{model_id}'" for model_id in bkg_weights["model_id"]
            ],
            "source_query": "*",
            "delay": 1.0,
            "weight_function": "weight_function_bkg",
            "dynamics_params": bkg_weights["dynamics_params"],
            "model_template": "static_synapse",
            "syn_weight": bkg_weights["syn_weight"],
        }
    )

    edge_types.to_csv(output_path, sep=" ", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate bkg_v1 edge types compatible with generated edge IDs."
    )
    parser.add_argument("bkg_weights_path")
    parser.add_argument("edges_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    filter_bkg_edge_types(args.bkg_weights_path, args.edges_path, args.output_path)
