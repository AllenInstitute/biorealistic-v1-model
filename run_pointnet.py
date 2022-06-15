import os, sys
import math
import numpy as np
from bmtk.simulator import pointnet
from bmtk.simulator.pointnet.pyfunction_cache import synaptic_weight
from bmtk.simulator.pointnet.io_tools import io

import nest

try:
    nest.Install("glifmodule")
except Exception as e:
    pass


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
def ConstantMultiplier_LGN(edges, src_nodes, trg_nodes):
    """
    Multiply a constant for all the LGN connections.
    """
    return edges['syn_weight'].values * 5.0
    

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
    phase_scale_ratio = np.exp(-(xz ** 2 / (2 * sigma_phase ** 2)))

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


def main(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    graph = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, graph)
    sim.run()


if __name__ == "__main__":
    if __file__ != sys.argv[-1]:
        main(sys.argv[-1])
    else:
        main("config.json")
