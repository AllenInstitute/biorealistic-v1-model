import os
import json
import numpy as np
import pandas as pd
import argparse
import random

from mpi4py import MPI

from node_funcs import (
    generate_random_positions,
    generate_positions_grids,
    get_filter_spatial_size,
    get_filter_temporal_params,
)
from edge_funcs import (
    compute_pair_type_parameters,
    connect_cells,
    select_lgn_sources,
    select_lgn_sources_powerlaw,
)

# from node_funcs import generate_random_positions, generate_positions_grids, get_filter_spatial_size, \
#     get_filter_temporal_params
# from connection_rules import compute_pair_type_parameters, connect_cells, select_lgn_sources
# from bmtk.builder.networks import MPIBuilder as NetworkBuilder # NetworkBuilder
from bmtk.builder import NetworkBuilder  # NetworkBuilder

# print(NetworkBuilder)
# exit()

pd.set_option("display.max_columns", None)


def add_nodes_v1(fraction=0.50, miniature=False):
    if miniature:
        node_props = "glif_props/v1_node_models_miniature.json"
    else:
        node_props = "glif_props/v1_node_models.json"
    v1_models = json.load(open(node_props, "r"))

    min_radius = 1.0  # to avoid diverging density near 0
    radius = v1_models["radius"]
    radial_range = [min_radius, radius]

    net = NetworkBuilder("v1")

    for location, loc_dict in v1_models["locations"].items():
        for pop_name, pop_dict in loc_dict.items():
            pop_size = pop_dict["ncells"]
            depth_range = -np.array(pop_dict["depth_range"], dtype=np.float)
            ei = pop_dict["ei"]

            for model in pop_dict["models"]:
                if "N" not in model:
                    # Assumes a 'proportion' key with a value from 0.0 to 1.0, N will be a proportion of pop_size
                    model["N"] = model["proportion"] * pop_size
                    del model["proportion"]

                if fraction != 1.0:
                    # Each model will use only a fraction of the of the number of cells for each model
                    # NOTE: We are using a ceiling function so there is atleast 1 cell of each type - however for models
                    #  with only a few initial cells they can be over-represented.
                    model["N"] = int(np.ceil(fraction * model["N"]))

                # create a list of randomized cell positions for each cell type
                N = model["N"]
                positions = generate_random_positions(N, depth_range, radial_range)

                # properties used to build the cells for each cell-type
                node_props = {
                    "N": N,
                    "node_type_id": model["node_type_id"],
                    "model_type": model["model_type"],
                    "model_template": model["model_template"],
                    "dynamics_params": model["dynamics_params"],
                    "ei": ei,
                    "location": location,
                    "pop_name": pop_name,
                    # "pop_name": (
                    #     "LIF" if model["model_type"] == "point_process" else ""
                    # )
                    # + pop_name,
                    "population": "v1",
                    "x": positions[:, 0],
                    "y": positions[:, 1],
                    "z": positions[:, 2],
                    "tuning_angle": np.linspace(0.0, 360.0, N, endpoint=False),
                }
                if model["model_type"] == "biophysical":
                    # for biophysically detailed cell-types add info about rotations and morphollogy
                    node_props.update(
                        {
                            # for RTNeuron store explicity store the x-rotations (even though it should be 0 by default).
                            "rotation_angle_xaxis": np.zeros(N),
                            "rotation_angle_yaxis": 2 * np.pi * np.random.random(N),
                            # for RTNeuron we need to store z-rotation in the h5 file.
                            "rotation_angle_zaxis": np.full(
                                N, model["rotation_angle_zaxis"]
                            ),
                            "model_processing": model["model_processing"],
                            "morphology": model["morphology"],
                        }
                    )

                net.add_nodes(**node_props)

    return net


def find_direction_rule(src_label, trg_label):
    src_ei = "e" if src_label.startswith("e") or src_label.startswith("LIFe") else "i"
    trg_ei = "e" if trg_label.startswith("e") or trg_label.startswith("LIFe") else "i"

    if src_ei == "e" and trg_ei == "e":
        return "DirectionRule_EE", 30.0

    elif src_ei == "e" and trg_ei == "i":
        return "DirectionRule_others", 90.0

    elif src_ei == "i" and trg_ei == "e":
        return "DirectionRule_others", 90.0

    else:
        return "DirectionRule_others", 50.0


def add_edges_v1(net):
    cc_prob_dict = json.load(open("biophys_props/v1_conn_props.json", "r"))
    conn_weight_df = pd.read_csv("biophys_props/v1_edge_models.csv", sep=" ")

    conn_weight_df = conn_weight_df[~(conn_weight_df["source_label"] == "LGN")]
    for _, row in conn_weight_df.iterrows():
        node_type_id = row["target_model_id"]
        src_type = row["source_label"]
        trg_type = row["target_label"]
        src_trg_params = compute_pair_type_parameters(src_type, trg_type, cc_prob_dict)
        # print(src_trg_params)

        weight_fnc, weight_sigma = find_direction_rule(src_type, trg_type)
        if src_trg_params["A_new"] > 0.0:
            if src_type.startswith("LIF"):
                net.add_edges(
                    source={"pop_name": src_type},
                    target={"node_type_id": node_type_id},
                    iterator="all_to_one",
                    connection_rule=connect_cells,
                    connection_params={"params": src_trg_params},
                    dynamics_params=row["params_file"],
                    syn_weight=row["weight_max"],
                    delay=row["delay"],
                    weight_function=weight_fnc,
                    weight_sigma=weight_sigma,
                )
            else:
                net.add_edges(
                    source={"pop_name": src_type},
                    target={"node_type_id": node_type_id},
                    iterator="all_to_one",
                    connection_rule=connect_cells,
                    connection_params={"params": src_trg_params},
                    dynamics_params=row["params_file"],
                    syn_weight=row["weight_max"],
                    delay=row["delay"],
                    weight_function=weight_fnc,
                    weight_sigma=weight_sigma,
                    distance_range=row["distance_range"],
                    target_sections=row["target_sections"],
                )
    return net


def add_nodes_lgn(X_grids=15, Y_grids=10, x_block=16.0, y_block=12.0):
    lgn_models = json.load(open("base_props/lgn_models.json", "r"))

    lgn = NetworkBuilder("lgn")
    # X_grids = 15  # 15#15      #15
    # Y_grids = 10  # 10#10#10      #10
    X_len = x_block * X_grids  # 240.0  # In linear degrees
    Y_len = y_block * Y_grids  # 120.0  # In linear degrees

    xcoords = []
    ycoords = []
    for model, params in lgn_models.items():
        # Get position of lgn cells and keep track of the averaged location
        # For now, use randomly generated values
        total_N = params["N"] * X_grids * Y_grids

        # Get positional coordinates of cells
        positions = generate_positions_grids(
            params["N"], X_grids, Y_grids, X_len, Y_len
        )
        xcoords += [p[0] for p in positions]
        ycoords += [p[1] for p in positions]

        # Get spatial filter size of cells
        filter_sizes = get_filter_spatial_size(
            params["N"], X_grids, Y_grids, params["size_range"]
        )

        # Get filter temporal parameters
        filter_params = get_filter_temporal_params(params["N"], X_grids, Y_grids, model)

        # Get tuning angle for LGN cells
        # tuning_angles = get_tuning_angles(params['N'], X_grids, Y_grids, model)

        lgn.add_nodes(
            N=total_N,
            pop_name=params["model_id"],
            model_type="virtual",
            ei="e",
            location="LGN",
            x=positions[:, 0],
            y=positions[:, 1],
            spatial_size=filter_sizes,
            kpeaks_dom_0=filter_params[:, 0],
            kpeaks_dom_1=filter_params[:, 1],
            weight_dom_0=filter_params[:, 2],
            weight_dom_1=filter_params[:, 3],
            delay_dom_0=filter_params[:, 4],
            delay_dom_1=filter_params[:, 5],
            kpeaks_non_dom_0=filter_params[:, 6],
            kpeaks_non_dom_1=filter_params[:, 7],
            weight_non_dom_0=filter_params[:, 8],
            weight_non_dom_1=filter_params[:, 9],
            delay_non_dom_0=filter_params[:, 10],
            delay_non_dom_1=filter_params[:, 11],
            tuning_angle=filter_params[:, 12],
            sf_sep=filter_params[:, 13],
        )

    return lgn


def add_lgn_v1_edges_experimental(v1_net, lgn_net, x_len=240.0, y_len=120.0):
    # conn_weight_df = pd.read_csv("base_props/lgn_weights_population.csv", sep=" ")
    conn_weight_df = pd.read_csv("glif_props/lgn_weights_model.csv", sep=" ")
    lgn_mean = (x_len / 2.0, y_len / 2.0)
    lgn_models = pd.read_json("base_props/lgn_models.json", orient="index")

    prop_query = ["node_id", "x", "y", "pop_name", "tuning_angle"]
    lgn_nodes = pd.DataFrame([{q: s[q] for q in prop_query} for s in lgn_net.nodes()])

    # this regular expression is picking up a number after TF
    lgn_nodes["temporal_freq"] = lgn_nodes["pop_name"].str.extract("TF(\d+)")
    # make a complex version beforehand for easy shift/rotation
    lgn_nodes["xy_complex"] = lgn_nodes["x"] + 1j * lgn_nodes["y"]

    for _, row in conn_weight_df.iterrows():
        target_pop_name = row["population"]
        target_model_id = row["model_id"]
        e_or_i = target_pop_name[0]
        if e_or_i == "e":
            sigma = [0.0, 150.0]
        elif e_or_i == "i":
            sigma = [0.0, 1e20]
        else:
            # Additional care for LIF will be necessary if applied for Biophysical
            raise (f"Unknown e_or_i value: {e_or_i} from {target_pop_name}")

        edge_params = {
            "source": lgn_net.nodes(),
            # "target": v1_net.nodes(pop_name=target_pop_name),
            "target": v1_net.nodes(node_type_id=target_model_id),
            "iterator": "all_to_one",
            "connection_rule": select_lgn_sources_powerlaw,
            "connection_params": {"lgn_mean": lgn_mean, "lgn_nodes": lgn_nodes},
            # "dynamics_params": row["params_file"],
            "dynamics_params": f"e2{e_or_i}.json",
            # "syn_weight": row["weight_max"],
            "syn_weight": row["syn_weight_psp"],
            # "delay": row["delay"],
            "delay": 1.7,
            # "weight_function": row["weight_func"],
            "weight_function": "",
            "weight_sigma": sigma,
            "model_template": "static_synapse",
        }

        lgn_net.add_edges(**edge_params)

    return lgn_net


def add_lgn_v1_edges(v1_net, lgn_net, x_len=240.0, y_len=120.0):
    # conn_weight_df = pd.read_csv("conn_props/edge_type_models.csv", sep=" ")
    conn_weight_df = pd.read_csv("base_props/lgn_weights_population.csv", sep=" ")
    # conn_weight_df = conn_weight_df[(conn_weight_df["source_label"] == "LGN")]

    lgn_mean = (x_len / 2.0, y_len / 2.0)
    lgn_models = json.load(open("base_props/lgn_models.json", "r"))

    # it is faster to precompute necessary properties of LGN nework.
    # cell_type_dict
    lgn_ids = [s.node_id for s in lgn_net.nodes()]
    cell_type_dict = {}
    lgn_x = {}
    lgn_y = {}
    for lgn_model in lgn_models:
        # initialize the individual list
        cell_type_dict[lgn_model] = []
        lgn_x[lgn_model] = []
        lgn_y[lgn_model] = []
    for src_id, src_dict in zip(lgn_ids, lgn_net.nodes()):
        cell_type_dict[src_dict["pop_name"]].append((src_id, src_dict))
        lgn_x[src_dict["pop_name"]].append(src_dict["x"])
        lgn_y[src_dict["pop_name"]].append(src_dict["y"])
    for lgn_model in lgn_models:
        # convert to numpy array for later computation
        lgn_x[lgn_model] = np.array(lgn_x[lgn_model])
        lgn_y[lgn_model] = np.array(lgn_y[lgn_model])

    for _, row in conn_weight_df.iterrows():
        # src_type = row["source_label"]
        # trg_type = row["target_label"]
        target_pop_name = row["population"]
        e_or_i = target_pop_name[0]
        if e_or_i == "e":
            sigma = [0.0, 150.0]
        elif e_or_i == "i":
            sigma = [0.0, 1e20]
        else:
            raise (f"Unknown e_or_i value: {e_or_i} from {target_pop_name}")

        edge_params = {
            "source": lgn_net.nodes(),
            "target": v1_net.nodes(pop_name=target_pop_name),
            "iterator": "all_to_one",
            "connection_rule": select_lgn_sources,
            "connection_params": {
                "lgn_mean": lgn_mean,
                "lgn_ids": np.array(lgn_ids),
                "lgn_x": lgn_x,
                "lgn_y": lgn_y,
                "cell_type_dict": cell_type_dict,
            },
            # "dynamics_params": row["params_file"],
            "dynamics_params": f"e2{e_or_i}.json",
            # "syn_weight": row["weight_max"],
            "syn_weight": row["syn_weight"],
            # "delay": row["delay"],
            "delay": 1.7,
            # "weight_function": row["weight_func"],
            "weight_function": "",
            "weight_sigma": sigma,
            "model_template": "static_synapse",
        }
        # if row["target_sections"] is not None:
        #     edge_params.update(
        #         {
        #             "target_sections": row["target_sections"],
        #             "distance_range": row["distance_range"],
        #         }
        #     )

        lgn_net.add_edges(**edge_params)

    return lgn_net


def add_nodes_bkg():
    bkg = NetworkBuilder("bkg")
    bkg.add_nodes(
        N=1,
        pop_name="SG_001",
        ei="e",
        location="BKG",
        model_type="virtual",
        # dynamics_params="spike_generator_bkg.json",
        x=[-91.23767151810344],
        y=[233.43548226294524],
    )
    return bkg


def add_bkg_v1_edges(v1_net, bkg_net):
    conn_weight_df = pd.read_csv("base_props/bkg_weights_population.csv", sep=" ")

    for _, row in conn_weight_df.iterrows():
        # src_type = row['source_label']
        # trg_type = row["target_label"]
        # target_node_type = row["target_model_id"]
        target_pop_name = row["population"]
        nsyns = row.get("nsyns")

        edge_params = {
            "source": bkg_net.nodes(),
            # "target": v1_net.nodes(node_type_id=target_node_type),
            "target": v1_net.nodes(pop_name=target_pop_name),
            "connection_rule": lambda s, t, n: n,
            "connection_params": {"n": nsyns},
            # "connection_params": {"nsyns": 1},
            # "dynamics_params": row["dynamics_params"],
            "dynamics_params": f"e2{target_pop_name[0]}.json",
            "syn_weight": row["syn_weight"],
            # "delay": row["delay"],
            "delay": 1.0,
            "model_template": "static_synapse",
        }
        # if trg_type == "biophysical":
        #     edge_params.update(
        #         {
        #             "target_sections": row["target_sections"],
        #             "distance_range": row["distance_range"],
        #         }
        #     )
        bkg_net.add_edges(**edge_params)

    return bkg_net


def check_files_exists(output_dir, src_net, trg_net, force_overwrite):
    if force_overwrite:
        return

    files = [
        os.path.join(output_dir, "{}_nodes.h5".format(src_net)),
        os.path.join(output_dir, "{}_node_types.csv".format(src_net)),
        os.path.join(output_dir, "{}_{}_edges.h5".format(src_net, trg_net)),
        os.path.join(output_dir, "{}_{}_edge_types.csv".format(src_net, trg_net)),
    ]
    for f in files:
        if os.path.exists(f):
            raise Exception(
                "file {} already exists. Use --force-overwrite to overwrite exists file or --output-dir to change path to file".format(
                    f
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the (GLIF) V1 (and lgn/background inputs) SONATA network files"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="network",
        help="directory where network files will be saved.",
    )
    parser.add_argument(
        "--v1-nodes-dir",
        help="directory containing existing v1 nodes. Used when building lgn/bkg network only",
    )
    parser.add_argument(
        "-f",
        "--force-overwrite",
        action="store_true",
        default=False,
        help="force existings network files to be overwritten",
    )
    parser.add_argument(
        "--no-recurrent",
        action="store_true",
        default=False,
        help="Make no recurrent connections in V1. Just nodes and feed-forward connections.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Specify a value between (0, 1.0) to build a network with only a given fraction of the V1 nodes",
    )
    parser.add_argument(
        "--miniature",
        action="store_true",
        default=False,
        help="Make a miniture network with with a small LGN. Only for debugging",
    )
    parser.add_argument(
        "--feed-forward-v2",
        action="store_true",
        default=False,
        help="use a version 2 of the feed-forward thalamocortical connection",
    )
    parser.add_argument("networks", type=str, nargs="*", default=["v1", "bkg", "lgn"])
    args = parser.parse_args()

    # set random number seed for reproducibility
    # The strategy is to use the common seed for all MPI ranks for nodes, and use
    # separate ones for edges. Though this may not be optimal, I reset the seed
    # multiple times for that reason.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    seed_v1_nodes = 153
    seed_v1_edges = 154 + rank
    seed_lgn_nodes = 253
    seed_lgn_edges = 254 + rank

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    # print(rank)
    # exit()
    # seed = 153 + rank  # modified for MPI running (Please use 8 cores)
    # random.seed(seed)
    # np.random.seed(seed)

    nets = set(args.networks)
    if nets - {"v1", "lgn", "bkg"}:
        # check specified networks
        raise Exception(
            "Uknown network(s) {}. valid networks: v1, lgn, bkg".format(
                set(nets) - {"v1", "lgn", "bkg"}
            )
        )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    v1 = None
    if "v1" in nets:
        print("Building v1 network")
        # check_files_exists(args.output_dir, 'v1', 'v1', args.force_overwrite)
        set_seed(seed_v1_nodes)
        v1 = add_nodes_v1(fraction=args.fraction, miniature=args.miniature)
        if not args.no_recurrent:
            set_seed(seed_v1_edges)
            v1 = add_edges_v1(v1)
        v1.build()
        print("Saving v1 network")
        v1.save(args.output_dir)
        print("  done.")
        nets.remove("v1")

    if len(nets) == 0:
        exit(0)

    if v1 is None:
        print("loading in v1 nodes from {}".format(args.v1_nodes_dir))
        v1 = NetworkBuilder("v1")
        v1.import_nodes(
            os.path.join(args.v1_nodes_dir, "v1_nodes.h5"),
            os.path.join(args.v1_nodes_dir, "v1_node_types.csv"),
        )
        print("  done.")

    if "lgn" in nets:
        print("Building lgn network")
        check_files_exists(args.output_dir, "lgn", "v1", args.force_overwrite)

        if args.feed_forward_v2:
            lgn_v1_edge_func = add_lgn_v1_edges_experimental
            x_block_unit = 8.0  # spherical coordinate
            y_block_unit = 8.0
        else:
            lgn_v1_edge_func = add_lgn_v1_edges
            x_block_unit = 16.0
            y_block_unit = 12.0

        if args.miniature:
            set_seed(seed_lgn_nodes)
            lgn = add_nodes_lgn(
                X_grids=15, Y_grids=10, x_block=x_block_unit, y_block=y_block_unit
            )
            set_seed(seed_lgn_edges)
            lgn = lgn_v1_edge_func(
                v1, lgn, x_len=15 * x_block_unit, y_len=10 * y_block_unit
            )

            # if args.feed_forward_v2:
            #     lgn = add_nodes_lgn(X_grids=15, Y_grids=10, x_block=8.0, y_block=8.0)
            #     lgn = add_lgn_v1_edges_experimental(v1, lgn, x_len=15 * 8.0, y_len=10 * 8.0)
            # else:
            #     lgn = add_nodes_lgn(X_grids=15, Y_grids=10, x_block=16.0, y_block=12.0)
            #     lgn = add_lgn_v1_edges(v1, lgn, x_len=15 * 16.0, y_len=10 * 12.0)
            # lgn = add_nodes_lgn(X_grids=15, Y_grids=10, x_block=8.0, y_block=8.0)
            # lgn = lgn_v1_edge_func(v1, lgn, x_len=15 * 8.0, y_len=10 * 8.0)
        else:
            set_seed(seed_lgn_nodes)
            lgn = add_nodes_lgn()
            set_seed(seed_lgn_edges)
            lgn = lgn_v1_edge_func(v1, lgn)
        lgn.build()
        lgn.save(args.output_dir)
        print("  done.")

    if "bkg" in nets:
        print("Building bkg network")
        check_files_exists(args.output_dir, "bkg", "v1", args.force_overwrite)
        bkg = add_nodes_bkg()
        bkg = add_bkg_v1_edges(v1, bkg)
        bkg.build()
        bkg.save(args.output_dir)
        print("  done.")

