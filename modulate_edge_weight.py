# %%

import pandas as pd
import re
import sys


def filter(str):
    return int(re.search("node_type_id=='([0-9]+)'", str).group(1))


if __name__ == "__main__":
    basedir = sys.argv[1]
    popmult = pd.read_csv(
        basedir + "/configs/lgn_v1_population_multiplier.csv", sep=" "
    )

    lgn_edges = pd.read_csv(basedir + "/network_nomod/lgn_v1_edge_types.csv", sep=" ")
    lgn_keys = lgn_edges.keys()

    v1_nodes = pd.read_csv(basedir + "/network_nomod/v1_node_types.csv", sep=" ")

    lgn_edges["node_type_id"] = lgn_edges["target_query"].apply(filter)
    pop_edges = lgn_edges.merge(
        v1_nodes[["node_type_id", "pop_name"]], on="node_type_id"
    )
    mult_edges = pop_edges.merge(popmult, on="pop_name")
    mult_edges["syn_weight"] = mult_edges["syn_weight"] * mult_edges["multiplier"]

    lgn_edges_mod = mult_edges[lgn_keys]
    lgn_edges_mod.to_csv(basedir + "/network/lgn_v1_edge_types.csv", sep=" ")
    print("Edge weights are updated.")
