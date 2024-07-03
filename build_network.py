import os
import json
import numpy as np
import pandas as pd
import argparse
import random
from scipy import interpolate, integrate
from scipy.stats import lognorm, rv_continuous
from pathlib import Path
from math import sqrt, exp, log
from scipy.special import erfinv

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
    select_lgn_sources_powerlaw,
    select_bkg_sources,
)

from bmtk.builder import NetworkBuilder
from bmtk.builder.node_pool import NodePool
from numba import njit

import logging

# logging.basicConfig(level=logging.DEBUG)

# print(NetworkBuilder)
# exit()

pd.set_option("display.max_columns", None)


def add_nodes_v1(fraction=1.00, miniature=False, flat=False):
    if miniature:
        node_props = "glif_props/v1_node_models_miniature.json"
    else:
        node_props = "glif_props/v1_node_models.json"
    # v1_models = json.load(open(node_props, "r"))
    with open(node_props, "r") as f:
        v1_models = json.load(f)

    min_radius = 1.0  # to avoid diverging density near 0
    radius = v1_models["radius"] * np.sqrt(fraction)
    radial_range = [min_radius, radius]

    net = NetworkBuilder("v1")

    for location, loc_dict in v1_models["locations"].items():
        for pop_name, pop_dict in loc_dict.items():
            pop_size = pop_dict["ncells"]
            depth_range = -np.array(pop_dict["depth_range"], dtype=float)
            ei = pop_dict["ei"]
            nsyn_lognorm_shape = pop_dict["nsyn_lognorm_shape"]
            nsyn_lognorm_scale = pop_dict["nsyn_lognorm_scale"]

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

                if flat:
                    N = 100
                else:
                    N = model["N"]
                # create a list of randomized cell positions for each cell type
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
                    "target_sizes": generate_target_sizes(
                        N, nsyn_lognorm_shape, nsyn_lognorm_scale
                    ),
                    # "EPSP_unitary": model["EPSP_unitary"],
                    # "IPSP_unitary": model["IPSP_unitary"],
                    "nsyn_size_shape": nsyn_lognorm_shape,
                    "nsyn_size_scale": nsyn_lognorm_scale,
                    "nsyn_size_mean": int(
                        lognorm(
                            s=nsyn_lognorm_shape, loc=0, scale=nsyn_lognorm_scale
                        ).stats(moments="m")
                    ),
                    # "size_connectivity_correction":
                }
                if model["model_type"] == "biophysical":
                    # for biophysically detailed cell-types add info about rotations and morphology
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


def orientation_dependence_fns(intercept_in, grad_in):
    intercept1 = intercept_in
    grad = grad_in
    y_90 = intercept1 + grad * 90.0
    intercept2 = 2 * y_90 - intercept1
    x = np.linspace(0, 180, 1000000)
    my_pdf = np.piecewise(
        x,
        [x < 90, x >= 90],
        [
            lambda x: (intercept1 + x * grad)
            / (2 * (intercept1 * 90 + grad * 90**2 / 2)),
            lambda x: (intercept2 - x * grad)
            / (
                2
                * (
                    intercept2 * 180
                    - grad * 180**2 / 2
                    - intercept2 * 90
                    + grad * 90**2 / 2
                )
            ),
        ],
    )
    discrete_cdf1 = np.append(integrate.cumtrapz(y=my_pdf, x=x), 1)
    # discrete_cdf1 = np.append(discrete_cdf1_,1.)
    pdf_out = interpolate.interp1d(x, my_pdf)
    cdf_out = interpolate.interp1d(x, discrete_cdf1)
    ppf_out = interpolate.interp1d(discrete_cdf1, x)
    return pdf_out, cdf_out, ppf_out


def syn_weight_by_experimental_distribution(
    source,
    target,
    src_pop_name,
    trg_pop_name,
    PSP_correction,
    PSP_lognorm_shape,
    PSP_lognorm_scale,
    connection_params,
    # delta_theta_dist,
):
    src_ei = src_pop_name[0]
    trg_ei = trg_pop_name[0]

    src_tuning = source["tuning_angle"]
    tar_tuning = target["tuning_angle"]

    #
    if PSP_lognorm_shape < target["nsyn_size_shape"]:
        weight_shape = 0.001
    else:
        weight_shape = sqrt(PSP_lognorm_shape**2 - target["nsyn_size_shape"] ** 2)
    weight_scale = exp(
        log(PSP_lognorm_scale)
        + log(target["nsyn_size_scale"])
        - log(target["nsyn_size_mean"])
    )
    # weight_rv = lognorm(weight_shape, loc=0, scale=weight_scale)

    # To set syn_weight, use the PPF with the orientation difference:
    # if not np.isnan(src_trg_params["gradient"]):

    # randomizing_factor = 0.1 #  8/15/2022 after discussion, decided to weaken the randomness
    randomizing_factor = (
        1.0  #  8/18/2022 reverting as it didn't affect the results much
    )
    # Original if condition:
    # if src_ei == "e" and trg_ei == "e" and (not type(delta_theta_dist) == float):
    # TODO: Please make sure if I'm doing it right.
    if (
        src_ei == "e"
        and trg_ei == "e"
        and (not np.isnan(connection_params["gradient"]))
    ):
        # For e-to-e, there is a non-uniform distribution of delta_orientations.
        # These need to be ordered and mapped uniformly over [0,1] using the cdf:

        # adds some randomization to like-to-like and avoids 0-degree delta
        # tuning_rnd = float(np.random.randn(1) * 5) * randomizing_factor
        tuning_rnd = 0.0

        delta_tuning_180 = np.abs(
            np.abs(np.mod(np.abs(tar_tuning - src_tuning + tuning_rnd), 360.0) - 180.0)
            - 180.0
        )

        # orient_temp = 1 - delta_theta_dist.cdf(delta_tuning_180)
        orient_temp = 1 - delta_theta_cdf(
            connection_params["intercept"], delta_tuning_180
        )
        # orient_temp = np.min([0.999, orient_temp])
        # orient_temp = np.max([0.001, orient_temp])
        orient_temp = min(0.999, orient_temp)
        orient_temp = max(0.001, orient_temp)
        syn_weight = lognorm_ppf(orient_temp, weight_shape, loc=0, scale=weight_scale)
        # weight_rv = lognorm(weight_shape, loc=0, scale=weight_scale)
        # syn_weight = weight_rv.ppf(orient_temp)
        n_syns_ = 1

    elif (src_ei == "e" and trg_ei == "i") or (src_ei == "i" and trg_ei == "e"):
        # If there was no like-to-like connection rule for the population, we can use
        # delta_orientation directly with the PPF

        # adds some randomization to like-to-like and avoids 0-degree delta
        tuning_rnd = float(np.random.randn(1) * 5) * randomizing_factor

        delta_tuning_180 = np.abs(
            np.abs(np.mod(np.abs(tar_tuning - src_tuning + tuning_rnd), 360.0) - 180.0)
            - 180.0
        )

        orient_temp = 1 - (delta_tuning_180 / 180)
        # orient_temp = np.min([0.999, orient_temp])
        # orient_temp = np.max([0.001, orient_temp])
        orient_temp = min(0.999, orient_temp)
        orient_temp = max(0.001, orient_temp)
        syn_weight = lognorm_ppf(orient_temp, weight_shape, loc=0, scale=weight_scale)
        # syn_weight = weight_rv.ppf(orient_temp)
        n_syns_ = 1

    elif src_ei == "i" and trg_ei == "i":
        # If there was no like-to-like connection rule for the population, we can use
        # delta_orientation directly with the PPF

        # adds some randomization to like-to-like and avoids 0-degree delta
        tuning_rnd = float(np.random.randn(1) * 10) * randomizing_factor

        delta_tuning_180 = np.abs(
            np.abs(np.mod(np.abs(tar_tuning - src_tuning + tuning_rnd), 360.0) - 180.0)
            - 180.0
        )

        orient_temp = 1 - (delta_tuning_180 / 180)
        # orient_temp = np.min([0.999, orient_temp])
        # orient_temp = np.max([0.001, orient_temp])
        orient_temp = min(0.999, orient_temp)
        orient_temp = max(0.001, orient_temp)

        syn_weight = lognorm_ppf(orient_temp, weight_shape, loc=0, scale=weight_scale)
        # syn_weight = weight_rv.ppf(orient_temp)
        n_syns_ = 1

    else:
        # If there was no like-to-like connection rule for the population, we can use
        # delta_orientation directly with the PPF

        # adds some randomization to like-to-like and avoids 0-degree delta
        tuning_rnd = float(np.random.randn(1) * 5) * randomizing_factor

        delta_tuning_180 = np.abs(
            np.abs(np.mod(np.abs(tar_tuning - src_tuning + tuning_rnd), 360.0) - 180.0)
            - 180.0
        )

        orient_temp = 1 - (delta_tuning_180 / 180)
        # orient_temp = np.min([0.999, orient_temp])
        # orient_temp = np.max([0.001, orient_temp])
        orient_temp = min(0.999, orient_temp)
        orient_temp = max(0.001, orient_temp)
        syn_weight = lognorm_ppf(orient_temp, weight_shape, loc=0, scale=weight_scale)
        # syn_weight = weight_rv.ppf(orient_temp)
        n_syns_ = 1

    # Below was copied from Billeh to use as an initial correction factor, but it is not clear how applicable
    # it is to the current Rossi Rule
    # delta_x = (x_tar - x_src) * 0.07
    # delta_z = (z_tar - z_src) * 0.04

    # theta_pref = tar_tuning * (np.pi / 180.0)
    # xz = delta_x * np.cos(theta_pref) + delta_z * np.sin(theta_pref)
    # sigma_phase = 1.0
    # phase_scale_ratio = np.exp(-(xz**2 / (2 * sigma_phase**2)))

    # # To account for the 0.07 vs 0.04 dimensions. This ensures the horizontal neurons are scaled by 5.5/4 (from the
    # # midpoint of 4 & 7). Also, ensures the vertical is scaled by 5.5/7. This was a basic linear estimate to get the
    # # numbers (y = ax + b).
    # theta_tar_scale = abs(
    #     abs(abs(180.0 - np.mod(np.abs(tar_tuning), 360.0)) - 90.0) - 90.0
    # )
    # phase_scale_ratio = phase_scale_ratio * (
    #     5.5 / 4.0 - 11.0 / 1680.0 * theta_tar_scale
    # )

    syn_weight = (
        syn_weight
        * target["nsyn_size_mean"]
        / (PSP_correction * target["target_sizes"])
    )
    return syn_weight, n_syns_


def lognorm_ppf(x, shape, loc=0, scale=1):
    # definition from wikipedia (quantile)
    return scale * exp(sqrt(2 * shape**2) * erfinv(2 * x - 1)) + loc


def delta_theta_cdf(intercept, d_theta):
    B1 = intercept
    # B1 = 2.0 / (1.0 + Q)
    Q = 2.0 / B1 - 1.0
    B2 = B1 * Q
    G = (B2 - B1) / 90.0
    norm = 90 * (B1 + B2)  # total area for normalization
    x = d_theta - 90
    if d_theta < 0:
        raise "d_theta must be >= 0, but was {}".format(d_theta)
    elif d_theta < 90:
        # analytical integration of the pdf to get this cdf
        return (0.5 * G * x**2 + B2 * x) / norm + 0.5
    elif d_theta <= 180:
        return (-0.5 * G * x**2 + B2 * x) / norm + 0.5
    else:
        raise "d_theta must be <= 180, but was {}".format(d_theta)


def add_edges_v1(net, core_radius):
    # pop to pop parameters:
    # cc_prob_dict = json.load(open("base_props/v1_conn_props_new.json", "r"))
    # cc_prob_dict = json.load(open("base_props/v1_conn_props_March28_2023.json", "r"))
    # cc_prob_dict = json.load(
    # open("base_props/v1_conn_props_March12_2024_Q0.5.json", "r")
    # open("base_props/v1_conn_props_April2.json", "r")
    # open("base_props/v1_conn_props_March13_2024_Q0.2.json", "r")
    # open("base_props/v1_conn_props_March14_2024_Q0.4.json", "r")
    # )
    # pop to specific model parameters:
    # conn_weight_df = pd.read_csv("base_props/v1_edge_models_lognorm_Jan_3_2022.csv")
    # conn_weight_df = pd.read_csv("base_props/v1_edge_models_lognorm_June_20_2022.csv")
    # conn_weight_df = pd.read_csv("base_props/v1_edge_models_double_alpha.csv")
    # conn_weight_df = pd.read_csv("base_props/v1_edge_models_April2.csv")
    # cc_prob_dict = json.load(open("biophys_props/v1_conn_props.json", "r"))
    # conn_weight_df = pd.read_csv("biophys_props/v1_edge_models.csv", sep=" ")

    conn_df = pd.read_csv("glif_props/v1_edge_models.csv", sep=" ")

    # conn_weight_df = conn_weight_df[~(conn_weight_df["source_label"] == "LGN")]
    for _, row in conn_df.iterrows():
        node_type_id = row["target_model_id"]
        src_pop_name = row["source_pop_name"]
        trg_pop_name = row["target_pop_name"]
        src_trg_params = compute_pair_type_parameters(row)

        prop_query = ["x", "z", "tuning_angle"]
        src_criteria = {"pop_name": src_pop_name}
        net.nodes()  # this line is necessary to activate nodes... (I don't know why.)
        source_nodes = NodePool(net, **src_criteria)
        source_nodes_df = pd.DataFrame(
            [{q: s[q] for q in prop_query} for s in source_nodes]
        )

        # TODO: check if these values should be used
        weight_fnc, weight_sigma = find_direction_rule(src_pop_name, trg_pop_name)
        if src_trg_params["pmax"] > 0.0:
            # if src_type.startswith("LIF"):
            #     net.add_edges(
            #         source={"pop_name": src_type},
            #         target={"node_type_id": node_type_id},
            #         iterator="all_to_one",
            #         connection_rule=connect_cells,
            #         connection_params={"params": src_trg_params},
            #         dynamics_params=row["params_file"],
            #         syn_weight=row["weight_max"],
            #         delay=row["delay"],
            #         weight_function=weight_fnc,
            #         weight_sigma=weight_sigma,
            #     )
            # else:
            # tentative fix for non-negative inhibitory connections
            if src_pop_name[0] == "i":
                pspsign = -1
            else:
                pspsign = 1
            cm = net.add_edges(
                source=src_criteria,
                target={"node_type_id": node_type_id},
                iterator="all_to_one",
                connection_rule=connect_cells,
                connection_params={
                    "params": src_trg_params,
                    "source_nodes": source_nodes_df,
                    "core_radius": core_radius,
                },
                dynamics_params=row["params_file"],
                # syn_weight_max=row["weight_max"],
                delay=row["delay"],
                weight_function="weight_function_recurrent",
                # weight_sigma=weight_sigma,
                # distance_range=row["distance_range"],
                # target_sections=row["target_sections"],
                # PSP_correction=row["PSP_scale_factor"],  # original there is one more line to fix ~30 lines below.
                PSP_correction=np.abs(row["PSP_scale_factor"]) * pspsign,
                PSP_lognorm_shape=row["lognorm_shape"],
                PSP_lognorm_scale=row["lognorm_scale"],
                model_template="static_synapse",
            )
            # replaced with custom analytic cdf function
            # if not np.isnan(src_trg_params["gradient"]):
            #     pdf1, cdf1, ppf1 = orientation_dependence_fns(
            #         src_trg_params["intercept"], src_trg_params["gradient"]
            #     )

            #     class orientation_dependence_dist(rv_continuous):
            #         def _pdf(self, x):
            #             return pdf1(x)

            #         def _cdf(self, x):
            #             return cdf1(x)

            #         def _ppf(self, x):
            #             return ppf1(x)

            #     delta_theta_dist = orientation_dependence_dist()
            # else:
            #     delta_theta_dist = np.NaN

            cm.add_properties(
                ["syn_weight", "n_syns_"],
                rule=syn_weight_by_experimental_distribution,
                rule_params={
                    "src_pop_name": src_pop_name,
                    "trg_pop_name": trg_pop_name,
                    # "PSP_correction": row["PSP_scale_factor"],
                    "PSP_correction": np.abs(row["PSP_scale_factor"]) * pspsign,
                    "PSP_lognorm_shape": row["lognorm_shape"],
                    "PSP_lognorm_scale": row["lognorm_scale"],
                    "connection_params": src_trg_params,
                    # "delta_theta_dist": delta_theta_dist,
                    # "lognorm_shape": row["lognorm_shape"],
                    # "lognorm_scale": row["lognorm_scale"],
                },
                dtypes=[float, np.int64],
            )
    return net


def add_nodes_lgn(X_grids=15, Y_grids=10, x_block=8.0, y_block=8.0):
    lgn_models = json.load(open("base_props/lgn_models.json", "r"))

    lgn = NetworkBuilder("lgn")
    X_len = x_block * X_grids  # default is 120 degrees
    Y_len = y_block * Y_grids  # default is 80 degrees

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


def add_lgn_v1_edges(v1_net, lgn_net, x_len=240.0, y_len=120.0, miniature=False):
    if miniature:
        node_props = "glif_props/v1_node_models_miniature.json"
    else:
        node_props = "glif_props/v1_node_models.json"
    v1_models = json.load(open(node_props, "r"))

    # skipping the 'locations' (e.g. VisL1) key and make a population-based
    # (e.g. i1Htr3a) dictionary
    v1_models_pop = {}
    for l in v1_models["locations"]:
        v1_models_pop.update(v1_models["locations"][l])

    # in this file, the values are specified for each target model
    conn_weight_df = pd.read_csv("glif_props/lgn_weights_model.csv", sep=" ")
    lgn_mean = (x_len / 2.0, y_len / 2.0)

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

        # LGN is configured based on e4 response. Here we use the mean target sizes of
        # the e4 neurons and normalize all the cells using these values. By doing this,
        # we can avoid injecting too much current to the populations with large target
        # sizes.
        lognorm_shape = v1_models_pop["e4other"]["nsyn_lognorm_shape"]
        lognorm_scale = v1_models_pop["e4other"]["nsyn_lognorm_scale"]
        e4_mean_size = np.exp(np.log(lognorm_scale) + (lognorm_shape**2) / 2)

        edge_params = {
            "source": lgn_net.nodes(),
            "target": v1_net.nodes(node_type_id=target_model_id),
            "iterator": "all_to_one",
            "connection_rule": select_lgn_sources_powerlaw,
            "connection_params": {"lgn_mean": lgn_mean, "lgn_nodes": lgn_nodes},
            "dynamics_params": row["dynamics_params"],
            "delay": 1.7,
            # "weight_function": "ConstantMultiplier_LGN",
            "weight_function": "weight_function_lgn",
            "weight_sigma": sigma,
            "model_template": "static_synapse",
        }

        cm = lgn_net.add_edges(**edge_params)
        cm.add_properties(
            "syn_weight",
            rule=lgn_synaptic_weight_rule,
            rule_params={
                "base_weight": row["syn_weight_psp"],
                "mean_size": e4_mean_size,
            },
            dtypes=float,
        )

    return lgn_net


def lgn_synaptic_weight_rule(source, target, base_weight, mean_size):
    return base_weight * mean_size / target["target_sizes"]


def add_nodes_bkg(n_unit):
    bkg = NetworkBuilder("bkg")
    bkg.add_nodes(
        # N=1,
        N=n_unit,
        pop_name="SG_001",
        ei="e",
        location="BKG",
        model_type="virtual",
        x=np.zeros(n_unit),
        y=np.zeros(n_unit),  # are these necessary?
        # x=[-91.23767151810344],
        # y=[233.43548226294524],
    )
    return bkg


def add_bkg_v1_edges(v1_net, bkg_net, n_conn):
    conn_weight_df = pd.read_csv("glif_props/bkg_weights_model.csv", sep=" ")
    # this file should contain the following parameters:
    # model_id (of targets), syn_weight_psp, dynamics_params, nsyns

    for _, row in conn_weight_df.iterrows():
        edge_params = {
            "source": bkg_net.nodes(),
            "target": v1_net.nodes(node_type_id=row["model_id"]),
            # "connection_rule": lambda s, t, n: n,
            "connection_rule": select_bkg_sources,
            "iterator": "all_to_one",
            "connection_params": {"n_syns": row["nsyns"], "n_conn": n_conn},
            "dynamics_params": row["dynamics_params"],
            # "syn_weight": row["syn_weight_psp"],
            "syn_weight": row["syn_weight"],
            "delay": 1.0,
            "model_template": "static_synapse",
            # "weight_function": "ConstantMultiplier_BKG",
            "weight_function": "weight_function_bkg",
        }
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


def generate_target_sizes(N, ln_shape, ln_scale):
    ln_rv = lognorm(s=ln_shape, loc=0, scale=ln_scale)
    ln_rvs = ln_rv.rvs(N).round()
    return ln_rvs


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
        help="Specify a value between (0, 1.0) to build a network with only a given fraction of the V1 nodes (radius is reduced; density is kept)",
    )
    parser.add_argument(
        "--miniature",
        action="store_true",
        default=False,
        help="Make a miniture network with a small LGN. Only for debugging",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        default=False,
        help="Make the number of neurons for each population 100. For BKG tuning.",
    )
    parser.add_argument(
        "--bkg-unit-num",
        type=int,
        default=100,
        help="Number of units in the background population",
    )
    parser.add_argument(
        "--bkg-conn-num",
        type=int,
        default=4,
        help="Number of connections from the background population",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        help="Compression algorithm to use for the HDF5 files (none, 1-9, gzip, lzf, etc.) \
              If you provide a number, it will be used as the compression level for gzip., \
        ",
    )
    parser.add_argument(
        "--core-radius",
        type=float,
        default=400.0,
        help="The radius of the core region. This will be used to determine how the \
              Rossi rule will be applied. It will be applied within 1.5 * core_radius.",
    )
    # This option is now obsolete.
    # parser.add_argument(
    #     "--feed-forward-v2",
    #     action="store_true",
    #     default=True,
    #     help="use a version 2 of the feed-forward thalamocortical connection",
    # )
    parser.add_argument("networks", type=str, nargs="*", default=["v1", "bkg", "lgn"])
    args = parser.parse_args()

    # if args.compression is a single letter string with a digit, convert it to int.
    if len(args.compression) == 1 and args.compression[0].isdigit():
        args.compression = int(args.compression)

    # set random number seed for reproducibility
    # The strategy is to use the common seed for all MPI ranks for nodes, and use
    # separate ones for edges. Though this may not be optimal, I reset the seed
    # multiple times for that reason.
    # To reproduce the same resutls, please use the same number of MPI processes.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    seed_v1_nodes = 153
    seed_v1_edges = 154 + rank
    seed_lgn_nodes = 253
    seed_lgn_edges = 254 + rank
    seed_bkg_nodes = 353
    seed_bkg_edges = 354 + rank

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    nets = set(args.networks)
    if nets - {"v1", "lgn", "bkg"}:
        # check specified networks
        raise Exception(
            "Uknown network(s) {}. valid networks: v1, lgn, bkg".format(
                set(nets) - {"v1", "lgn", "bkg"}
            )
        )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    v1 = None
    if "v1" in nets:
        print("Building v1 network")
        # check_files_exists(args.output_dir, 'v1', 'v1', args.force_overwrite)
        set_seed(seed_v1_nodes)
        v1 = add_nodes_v1(
            fraction=args.fraction, miniature=args.miniature, flat=args.flat
        )
        if not args.no_recurrent:
            set_seed(seed_v1_edges)
            v1 = add_edges_v1(v1, args.core_radius)
        v1.build()
        print("Saving v1 network")
        v1.save(args.output_dir, compression=args.compression)
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

        lgn_v1_edge_func = add_lgn_v1_edges
        x_block_unit = 8.0  # spherical coordinate
        y_block_unit = 8.0

        # now regardless of settings, LGN models are the same
        set_seed(seed_lgn_nodes)
        if args.miniature:
            lgn = add_nodes_lgn(
                X_grids=1, Y_grids=2, x_block=x_block_unit, y_block=y_block_unit
            )
        else:
            lgn = add_nodes_lgn(x_block=x_block_unit, y_block=y_block_unit)
        set_seed(seed_lgn_edges)
        lgn = lgn_v1_edge_func(
            v1, lgn, x_len=15 * x_block_unit, y_len=10 * y_block_unit
        )

        lgn.build()
        lgn.save(args.output_dir, compression=args.compression)
        print("  done.")

    if "bkg" in nets:
        print("Building bkg network")
        check_files_exists(args.output_dir, "bkg", "v1", args.force_overwrite)
        set_seed(seed_bkg_nodes)
        bkg = add_nodes_bkg(args.bkg_unit_num)
        set_seed(seed_bkg_edges)
        bkg = add_bkg_v1_edges(v1, bkg, args.bkg_conn_num)
        bkg.build()
        bkg.save(args.output_dir, compression=args.compression)
        print("  done.")
