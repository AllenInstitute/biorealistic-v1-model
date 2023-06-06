import pandas as pd
import h5py
import itertools
import os
import json
import numpy as np
import pprint
import argparse

from bmtk.simulator.core.simulation_config import SimulationConfig
from bmtk.utils import sonata

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


def convert_glif_lif_asc_psc(dynamics_params, syn_params):
    config = dynamics_params
    coeffs = config["coeffs"]
    basedict = {
        "V_m": config["El"] * 1.0e03 + config["El_reference"] * 1.0e03,
        "V_th": coeffs["th_inf"] * config["th_inf"] * 1.0e03
        + config["El_reference"] * 1.0e03,
        "g": coeffs["G"] / config["R_input"] * 1.0e09,
        "E_L": config["El"] * 1.0e03 + config["El_reference"] * 1.0e03,
        "C_m": coeffs["C"] * config["C"] * 1.0e12,
        "t_ref": config["spike_cut_length"] * config["dt"] * 1.0e03,
        "V_reset": config["El_reference"] * 1.0e03,
        "asc_init": list(np.array(config["init_AScurrents"]) * 1.0e12),
        "asc_decay": list(1.0 / np.array(config["asc_tau_array"]) * 1.0e-03),
        "asc_amps": list(
            np.array(config["asc_amp_array"])
            * np.array(coeffs["asc_amp_array"])
            * 1.0e12
        ),
        # "tau_syn": syn_params,
        "spike_dependent_threshold": False,
        "after_spike_currents": True,
        "adapting_threshold": False,
    }
    # if syn_params is a dict
    if isinstance(syn_params, dict):
        basedict.update(syn_params)
    else:
        basedict["tau_syn"] = syn_params

    return basedict


def load_v1_nodes(base_dir):
    v1_sonata = sonata.File(
        f"{base_dir}/network/v1_nodes.h5", f"{base_dir}/network/v1_node_types.csv"
    )
    nodes_df = v1_sonata.nodes["v1"].to_dataframe(index_by_id=False)
    nodes_df = nodes_df[["node_id", "dynamics_params"]]
    nodes_df = nodes_df.rename(columns={"dynamics_params": "node_params"})
    return nodes_df


def load_edges(base_dir, src_pop):
    edge_types_df = pd.read_csv(
        f"{base_dir}/network/{src_pop}_v1_edge_types.csv", sep=" "
    )
    edge_types_df = edge_types_df[["edge_type_id", "dynamics_params"]]
    edge_types_df = edge_types_df.rename(columns={"dynamics_params": "edge_params"})
    edges_h5 = h5py.File(f"{base_dir}/network/{src_pop}_v1_edges.h5", "r")
    edges_grp = edges_h5[f"/edges/{src_pop}_to_v1/"]
    edges_df = pd.DataFrame(
        {
            "node_id": edges_grp["target_node_id"],
            "edge_type_id": edges_grp["edge_type_id"],
        }
    )

    edges_df = edges_df.merge(edge_types_df, how="left", on="edge_type_id")
    edges_df = edges_df[["node_id", "edge_params"]].drop_duplicates()
    return edges_df


def convert_params(components_orig_dir, components_new_base_dir, verbose):
    components_new_dir = components_new_base_dir + "/components"
    # Build a table so that we can map each [edge]_dynamics_params json to each [node]_dynamics_params json. Essentialy
    #   let's us know which node-types are being used by any given edge-type and vice-versa.
    print(" > Loading network files.")
    nodes_df = load_v1_nodes(components_new_base_dir)
    v1_edges = load_edges(components_new_base_dir, "v1")
    lgn_edges = load_edges(components_new_base_dir, "lgn")
    bkg_edges = load_edges(components_new_base_dir, "bkg")
    edges_df = pd.concat((v1_edges, lgn_edges, bkg_edges))
    parms_map = edges_df.merge(nodes_df, how="left", on="node_id")
    parms_map = parms_map[["edge_params", "node_params"]].drop_duplicates()

    # For each edge_dynamics_params file, find a list of all correspodning nodes (dynamics_params) used in edge type.
    # both edge_params and node_params are a list of size N with corresponding indices. Used two lists instead of dict
    #  to make it easier to modify the edge_dynamics_params set
    print("Generating shared tau_syns")
    edge_params = []
    node_params = []
    for edge_params_file, edge_params_df in parms_map.groupby("edge_params"):
        edge_params.append({edge_params_file})
        node_params.append(set(edge_params_df["node_params"].values))

    # If two edge-types (dynamics_params) use any intersecting same node-type (dynamic_params) then merge them together.
    #  eg {e2e.json} and {i2e.json} --> {e2e.json,i2e.json}. This way we can know which tau_syn values need to be merged
    #  together.
    has_intersection = True
    while has_intersection:
        # iterate over each combination of node_params items, if two indices have any intersection node-types then
        #  combine the node_params and edge_params sets. Keep repeating until there are no more intersections left.
        has_intersection = False
        for i, j in itertools.combinations(range(len(node_params)), 2):
            set_i = node_params[i]
            set_j = node_params[j]
            if set_i & set_j:
                edge_params[i] |= edge_params.pop(j)
                node_params[i] |= node_params.pop(j)
                has_intersection = True
                break
    assert len(edge_params) == len(node_params)

    # Create the new point_components/ directories
    print(" > Generating new component/ files")
    os.makedirs("{}/synaptic_models".format(components_new_dir), exist_ok=True)
    os.makedirs("{}/cell_models".format(components_new_dir), exist_ok=True)
    edges_table = {
        "synaptic_model": [],
        "params": [],
        "index": [],
        "group": [],
    }  # For printing later
    nodes_table = {}  # {'tau_syn': [], 'cell_models': []}
    group = 0
    for eset, nset in zip(edge_params, node_params):
        syn_params = {}
        for i, efile in enumerate(eset):
            # Get the "tau_syn" values from synaptic_models dynamics_params and create a list of tau_syns. eg
            #  {e2e.json, i2e.json} => tau_syns=[5.5, 8.5]
            input_json_path = os.path.join(
                components_orig_dir, "synaptic_models", efile
            )
            dyn_params = json.load(open(input_json_path, "r"))
            print(input_json_path)
            try:
                # the new format should have 3 parameters: tau_syn, tau_syn_slow, and amp_slow
                # try to retrieve all, and determine the version.
                params = ["tau_syn", "tau_syn_slow", "amp_slow"]
                syn_params_one = {k: dyn_params[k] for k in params}
            except KeyError:
                # show warning only once
                if verbose:
                    print(
                        f'Warning: No "tau_syn_slow" or "amp_slow" found in "{input_json_path}". Falling back to single tau_syn.'
                    )
                try:
                    params = ["tau_syn"]
                    syn_params_one = {k: dyn_params[k] for k in params}
                    # tau = dyn_params["tau_syn"]
                    # syn_params.append(tau)
                    # del dyn_params["tau_syn"]
                except:
                    print(
                        f'Warning: No "tau_syn" found in "{input_json_path}". Setting it to 1.0.'
                    )
                    # tau = 1.0
                    params = ["tau_syn"]
                    syn_params_one.append({"tau_syn": 1.0})

            # once the keys are determined, append the values to the list for each parameter.
            for key in params:
                # if entry does not exist, create a list
                if key not in syn_params:
                    syn_params[key] = []
                # append the value
                syn_params[key].append(syn_params_one[key])
            # reset dyn_params for recycling.
            dyn_params = {}

            output_json = os.path.join(components_new_dir, "synaptic_models", efile)
            dyn_params["receptor_type"] = (
                i + 1
            )  # in NEST it looks like access to receptor type is 1 based
            json.dump(dyn_params, open(output_json, "w"))

            # For displaying table at the end
            edges_table["synaptic_model"].append(efile)
            edges_table["params"].append(syn_params_one)
            edges_table["index"].append(i + 1)
            edges_table["group"].append(group)
        group += 1

        for nfile in nset:
            # Insert the tau_syns array into the updated cell_models/ dynamics_params file
            input_json_path = os.path.join(components_orig_dir, "cell_models", nfile)
            dyn_params = json.load(open(input_json_path, "r"))

            updated_params = convert_glif_lif_asc_psc(dyn_params, syn_params)
            output_json_path = os.path.join(components_new_dir, "cell_models", nfile)
            json.dump(updated_params, open(output_json_path, "w"), indent=2)

        nodes_table[tuple(syn_params)] = nset

    print(" > Done.")
    print(" > synaptic_models indices:")
    print(pd.DataFrame(edges_table))

    if verbose:
        print(" > cell_models tau_syns:")
        pprint.pprint(nodes_table)
        # print(pd.DataFrame(nodes_table))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert generic GLIF components to build-specific components"
    )
    parser.add_argument("basedir", type=str, help="Base directory of the build")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")

    args = parser.parse_args()
    convert_params(
        # components_orig_dir="point_components_orig",
        components_orig_dir="glif_models",
        components_new_base_dir=f"{args.basedir}",  # /components automatically added.
        verbose=args.verbose,
    )
