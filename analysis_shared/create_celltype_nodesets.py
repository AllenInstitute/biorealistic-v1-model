#!/usr/bin/env python3
"""Create node sets for PV, SST, and VIP cells for suppression experiments."""
import json
from pathlib import Path
import h5py
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "core_nll_0"
NODE_SET_DIR = BASE_DIR / "node_sets"
NODE_FILE = BASE_DIR / "network" / "v1_nodes.h5"
NODE_TYPES_FILE = BASE_DIR / "network" / "v1_node_types.csv"

def main():
    # Load node types
    node_types = pd.read_csv(NODE_TYPES_FILE, sep=r"\s+")
    node_types["cell_marker"] = node_types["pop_name"].str.extract(r"[ei]\d+(\w+)")
    type_lookup = node_types.set_index("node_type_id")["cell_marker"]

    # Load nodes
    with h5py.File(NODE_FILE, "r") as f:
        node_ids = f["nodes"]["v1"]["node_id"][:]
        node_type_ids = f["nodes"]["v1"]["node_type_id"][:]

    # Create dataframe
    df = pd.DataFrame({"node_type_id": node_type_ids}, index=node_ids)
    df["cell_marker"] = df["node_type_id"].map(type_lookup)

    # Create node sets for each cell type
    cell_types = {
        "pv": "Pvalb",
        "sst": "Sst",
        "vip": "Vip"
    }

    for abbrev, marker in cell_types.items():
        nodes = df[df["cell_marker"] == marker].index.tolist()

        node_set = {
            "population": "v1",
            "node_id": [int(n) for n in nodes]
        }

        output_file = NODE_SET_DIR / f"{abbrev}_nodes.json"
        with open(output_file, 'w') as f:
            json.dump(node_set, f, indent=2)

        print(f"Created {abbrev.upper()} node set: {len(nodes)} cells → {output_file}")

if __name__ == "__main__":
    main()
