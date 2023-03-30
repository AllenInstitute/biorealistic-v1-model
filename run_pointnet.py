import os, sys
import math
import numpy as np
from bmtk.simulator import pointnet
from bmtk.simulator.pointnet.pyfunction_cache import synaptic_weight
from bmtk.simulator.pointnet.io_tools import io
import argparse
import pandas as pd

import nest

try:
    nest.Install("glifmodule")
except Exception as e:
    pass


# If you want to turn off modulation entirely, set this to True
# turn_off_modulation = False  # this option is no longer needed.
modulation_df = None  # Defining it as a global variable


@synaptic_weight
def weight_function_recurrent(edges, src_nodes, trg_nodes):
    if modulation_df is None:
        return edges["syn_weight"].values
    # if turn_off_modulation:
    # return edges["syn_weight"].values
    # write your modulation I'll put down some example.
    # src_nodes is a pandas dataframe with keys:
    # node_type_id, target_sizes, tuning_angle, x, y, z,
    # model_template, dynamic_params, nsyn_size_shape,
    # nsyn_size_scale, nsyn_size_mean, population, pop_name, ei,  model_type
    # location

    # All the connections in each row of the edge_types file is given.
    # so in all of the case, pop_name is the same for all the neurons.
    # print(src_nodes.pop_name.value_counts())
    # print(trg_nodes.pop_name.value_counts())

    # now we have modulation_df.
    # modulation_df should have the following columns:
    # src_property, src_substring, trg_property, trg_substring, operation, value
    for row in modulation_df.itertuples():
        src_prop = src_nodes[row.src_property].iloc[0]
        tgt_prop = trg_nodes[row.trg_property].iloc[0]
        if (row.src_substring in src_prop) and (row.trg_substring in tgt_prop):
            if row.operation == "*":
                return edges["syn_weight"].values * row.value
            elif row.operation == "+":
                return edges["syn_weight"].values + row.value
            elif row.operation == "-":
                return edges["syn_weight"].values - row.value
            elif row.operation == "/":
                return edges["syn_weight"].values / row.value
            else:
                raise ValueError("Operation not recognized")

    # if ("i4V" in src_pop) and ("i4V" in tgt_pop):
    # print("i4V to i4V")
    # return edges["syn_weight"].values * 10.0

    # if nothing is wanted, you can just return the original weight
    return edges["syn_weight"].values


@synaptic_weight
def weight_function_lgn(edges, src_nodes, trg_nodes):
    return weight_function_recurrent(edges, src_nodes, trg_nodes)
    # if turn_off_modulation:
    # return edges["syn_weight"].values

    # return edges["syn_weight"].values


@synaptic_weight
def weight_function_bkg(edges, src_nodes, trg_nodes):
    return weight_function_recurrent(edges, src_nodes, trg_nodes)
    # if turn_off_modulation:
    #     return edges["syn_weight"].values
    # return edges["syn_weight"].values


"""
@synaptic_weight
def DendriticConstancy_LGN(edges, src_nodes, trg_nodes):
    # normalize by the total amount of inputs of the target cells
    # the total number of inputs are pre computed
    # total_input = trg_nodes["total_input_nsyns_lgn.0"].values
    # total_input = trg_nodes["total_input_nsyns_lgn"].values
    nsyns_correction = trg_nodes["nsyns_correction_lgn"].values
    # mean_total_input = np.mean(total_input)
    nsyns = edges["nsyns"].values
    syn_weight = edges["syn_weight"].values
    return syn_weight * nsyns / nsyns_correction
"""


@synaptic_weight
def ConstantMultiplier_LGN(edges, src_nodes, trg_nodes):
    """
    Multiply a constant for all the LGN connections.
    """
    return edges["syn_weight"].values * 1.0


@synaptic_weight
def ConstantMultiplier_BKG(edges, src_nodes, trg_nodes):
    """
    Multiply a constant for all the BKG connections.
    """
    return edges["syn_weight"].values * 1.0


@synaptic_weight
def DirectionRule_others(edges, src_nodes, trg_nodes):
    src_tuning = src_nodes["tuning_angle"].values
    tar_tuning = trg_nodes["tuning_angle"].values
    sigma = edges["weight_sigma"].values
    nsyn = edges["nsyns"].values
    syn_weight = edges["syn_weight"].values

    delta_tuning_180 = np.abs(
        np.abs(np.mod(np.abs(tar_tuning - src_tuning), 360.0) - 180.0) - 180.0
    )
    w_multiplier_180 = np.exp(-((delta_tuning_180 / sigma) ** 2))

    return syn_weight * w_multiplier_180 * nsyn


@synaptic_weight
def DirectionRule_EE(edges, src_nodes, trg_nodes):
    src_tuning = src_nodes["tuning_angle"].values
    tar_tuning = trg_nodes["tuning_angle"].values
    x_tar = trg_nodes["x"].values
    x_src = src_nodes["x"].values
    z_tar = trg_nodes["z"].values
    z_src = src_nodes["z"].values
    sigma = edges["weight_sigma"].values
    nsyn = edges["nsyns"].values
    syn_weight = edges["syn_weight"].values

    delta_tuning_180 = np.abs(
        np.abs(np.mod(np.abs(tar_tuning - src_tuning), 360.0) - 180.0) - 180.0
    )
    w_multiplier_180 = np.exp(-((delta_tuning_180 / sigma) ** 2))

    delta_x = (x_tar - x_src) * 0.07
    delta_z = (z_tar - z_src) * 0.04

    theta_pref = tar_tuning * (np.pi / 180.0)
    xz = delta_x * np.cos(theta_pref) + delta_z * np.sin(theta_pref)
    sigma_phase = 1.0
    phase_scale_ratio = np.exp(-(xz**2 / (2 * sigma_phase**2)))

    # To account for the 0.07 vs 0.04 dimensions. This ensures the horizontal neurons are scaled by 5.5/4 (from the
    # midpoint of 4 & 7). Also, ensures the vertical is scaled by 5.5/7. This was a basic linear estimate to get the
    # numbers (y = ax + b).
    theta_tar_scale = abs(
        abs(abs(180.0 - np.mod(np.abs(tar_tuning), 360.0)) - 90.0) - 90.0
    )
    phase_scale_ratio = phase_scale_ratio * (
        5.5 / 4.0 - 11.0 / 1680.0 * theta_tar_scale
    )

    return syn_weight * w_multiplier_180 * phase_scale_ratio * nsyn


def override_output(config, output_dir):
    outfiles = {
        "log_file": "log.txt",
        "spikes_file": "spikes.h5",
        "spikes_file_csv": "spikes.csv",
    }
    if output_dir is not None:
        output_fullpath = os.path.abspath(output_dir)
        config["manifest"]["OUTPUT_DIR"] = output_fullpath
        config["output"]["output_dir"] = output_fullpath
        config.output_dir = output_fullpath
        config.log_file = os.path.join(output_fullpath, "log.txt")
        for key, value in outfiles.items():
            config["output"][key] = os.path.join(output_fullpath, value)
    return config

def insert_modfile_to_config(config, modfilename):
    # insert the mod file to the config
    if modfilename is not None:
        modfile_fullpath = os.path.abspath(modfilename)
        config["inputs"]["MODFILE"] = modfile_fullpath
    return config

def main(config_file, output_dir, modfilename):
    configure = pointnet.Config.from_json(config_file)
    # change the output directory if specified
    # it will do nothing if output_dir is None
    override_output(configure, output_dir)
    insert_modfile_to_config(configure, modfilename)

    configure.build_env()

    graph = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, graph)

    # if you want to initialize the network with random membrane potentials,
    # uncomment the following line
    # set_random_potentials(sim)
    sim.run()


def get_v1_node_nums(sim):
    node_ids = sim.net._node_sets["v1"]._populations[0]._node_pop.node_ids
    return len(node_ids)


def set_random_potentials(sim):
    node_nums = get_v1_node_nums(sim)
    random_potentials = np.random.uniform(low=-75.0, high=-55.0, size=node_nums)
    nest.SetStatus(range(1, node_nums + 1), "V_m", random_potentials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the pointnet simulation with the given config file."
    )
    parser.add_argument(
        "-m", "--modfile", type=str, default=None, help="The modulation file to use."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="This option will override the output directory specified in the config file.",
    )
    parser.add_argument(
        "config_file",
        type=str,
        nargs="?",
        default="config.json",
        help="The config file to use for the simulation.",
    )
    args = parser.parse_args()

    if args.modfile is not None:
        # assign the modulation file to the global variable.
        # index column is not defined in the file, so make it up.
        modulation_df = pd.read_csv(args.modfile, sep=" ", index_col=False)

    main(args.config_file, output_dir=args.output_dir, args.modfile)
