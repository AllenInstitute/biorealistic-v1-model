import json
import numpy as np
import math
from random import random
from numba import njit, jit
from itertools import compress
from scipy.stats import yulesimon


# cc_prob_dict = json.load(open('biophys_props/v1_conn_props.json', 'r'))
lgn_params = json.load(open("base_props/lgn_params.json", "r"))
power_law_exponent = 3.58  # calculated from the EM data
power_law_mean = 1.1737  # above exponent numerically define this


def compute_pair_type_parameters(source_type, target_type, cc_prob_dict):
    """ Takes in two strings for the source and target type. It determined the connectivity parameters needed based on
    distance dependence and orientation tuning dependence and returns a dictionary of these parameters. A description
    of the calculation and resulting formulas used herein can be found in the accompanying documentation. Note that the
    source and target can be the same as this function works on cell TYPES and not individual nodes. The first step of
    this function is getting the parameters that determine the connectivity probabilities reported in the literature.
    From there the calculation proceed based on adapting these values to our model implementation.

    :param source_type: string of the cell type that will be the source (pre-synaptic)
    :param target_type: string of the cell type that will be the targer (post-synaptic)
    :return: dictionary with the values to be used for distance dependent connectivity
             and orientation tuning dependent connectivity (when applicable, else nans populate the dictionary).
    """
    src_new = source_type[3:] if source_type[0:3] == "LIF" else source_type
    trg_new = target_type[3:] if target_type[0:3] == "LIF" else target_type

    src_tmp = src_new[0:2]
    if src_new[0] == "i":
        src_tmp = src_new[0:3]
    if src_new[0:2] == "i2":
        src_tmp = src_new[0:2] + src_new[3]

    trg_tmp = trg_new[0:2]
    if trg_new[0] == "i":
        trg_tmp = trg_new[0:3]
    if trg_new[0:2] == "i2":
        trg_tmp = trg_new[0:2] + trg_new[3]

    cc_props = cc_prob_dict[src_tmp + "-" + trg_tmp]

    ##### For distance dependence which is modeled as a Gaussian ####
    # P = A * exp(-r^2 / sigma^2)
    # Since papers reported probabilities of connection having measured from 50um to 100um intersomatic distance
    # we had to normalize ensure A. In short, A had to be set such that the integral from 0 to 75um of the above
    # function was equal to the reported A in the lietature. Please see accompanying documentation for the derivation
    # of equations which should explain how A_new is determined.
    # Note we intergrate upto 75um as an approximate mid-point from the reported literature.

    # A_literature is different for every source-target pair and was estimated from the literature.
    A_literature = cc_props["A_literature"]

    # R0 read from the dictionary, but setting it now at 75um for all cases but this allows us to change it
    R0 = cc_props["R0"]

    # Sigma is measure from the literature or internally at the Allen Institute
    sigma = cc_props["sigma"]

    # Gaussian equation was intergrated to and solved to calculate new A_new. See accompanying documentation.
    A_new = A_literature / ((sigma / R0) ** 2 * (1 - np.exp(-((R0 / sigma) ** 2))))

    # Due to the measured values in the literature being from multiple sources and approximations that were
    # made by us and the literature (for instance R0 = 75um and sigma from the literature), we found that it is
    # possible for A_new to go slightly above 1.0 in which case we rescale it to 1.0. We confirmed that if this
    # does happen, it is for a few cases and is not much higher than 1.0.
    if A_new > 1.0:
        # print('WARNING: Adjusted calculated probability based on distance dependence is coming out to be ' \
        #       'greater than 1 for ' + source_type + ' and ' + target_type + '. Setting to 1.0')
        A_new = 1.0

    ##### To include orientation tuning ####
    # Many cells will show orientation tuning and the relative different in orientation tuning angle will influence
    # probability of connections as has been extensively report in the literature. This is modeled here with a linear
    # where B in the largest value from 0 to 90 (if the profile decays, then B is the intercept, if the profile
    # increases, then B is the value at 90). The value of G is the gradient of the curve.
    # The calculations and explanation can be found in the accompanying documentation with this code.

    # Extract the values from the dictionary for B, the maximum value and G the gradient
    B_ratio = cc_props["B_ratio"]
    B_ratio = np.nan if B_ratio is None else B_ratio

    # Check if there is orientation dependence in this source-target pair type. If yes, then a parallel calculation
    # to the one done above for distance dependence is made though with the assumption of a linear profile.
    if not np.isnan(B_ratio):
        # The scaling for distance and orientation must remain less than 1 which is calculated here and reset
        # if it is greater than one. We also ensure that the area under the p(delta_phi) curve is always equal
        # to one (see documentation). Hence the desired ratio by the user may not be possible, in which case
        # an warning message appears indicating the new ratio used. In the worst case scenario the line will become
        # horizontal (no orientation tuning) but will never "reverse" slopes.

        # B1 is the intercept which occurs at (0, B1)
        # B2 is the value when delta_phi equals 90 degree and hence the point (90, B2)
        B1 = 2.0 / (1.0 + B_ratio)
        B2 = B_ratio * B1

        AB = A_new * max(B1, B2)
        if AB > 1.0:
            if B1 >= B2:
                B1_new = 1.0 / A_new
                delta = B1 - B1_new
                B1 = B1_new
                B2 = B2 + delta
            elif B2 > B1:
                B2_new = 1.0 / A_new
                delta = B2 - B2_new
                B2 = B2_new
                B1 = B1 + delta

            B_ratio = B2 / B1
            print(
                "WARNING: Could not satisfy the desired B_ratio (probability of connectivity would become "
                "greater than one in some cases). Rescaled and now for "
                + source_type
                + " --> "
                + target_type
                + " the ratio is set to: ",
                B_ratio,
            )

        G = (B2 - B1) / 90.0

    # If there is no orientation dependent, record this by setting the intercept to Not a Number (NaN).
    else:
        B1 = np.NaN
        G = np.NaN

    # Return the dictionary. Note, the new values are A_new and intercept. The rest are from CC_prob_dict.
    return {
        "A_new": A_new,
        "sigma": sigma,
        "gradient": G,
        "intercept": B1,
        "nsyn_range": cc_props["nsyn_range"],
    }


def connect_cells(sources, target, params):
    """This function determined which nodes are connected based on the parameters in the dictionary params. The
    function iterates through every cell pair when called and hence no for loop is seen iterating pairwise
    although this is still happening.

    By iterating though every cell pair, a decision is made whether or not two cells are connected and this
    information is returned by the function. This function calculates these probabilities based on the distance between
    two nodes and (if applicable) the orientation tuning angle difference.

    :param sid: the id of the source node
    :param source: the attributes of the source node
    :param tid: the id of the target node
    :param target: the attributes of the target node
    :param params: parameters dictionary for probability of connection (see function: compute_pair_type_parameters)
    :return: if two cells are deemed to be connected, the return function automatically returns the source id
             and the target id for that connection. The code further returns the number of synapses between
             those two neurons
    """

    sources_x = np.array([s["x"] for s in sources])
    sources_z = np.array([s["z"] for s in sources])
    sources_tuning_angle = [s["tuning_angle"] for s in sources]

    # Get target id
    tid = target.node_id
    # if tid % 1000 == 0:
    #     print('target {}'.format(tid))

    # Read parameter values needed for distance and orientation dependence
    A_new = params["A_new"]
    sigma = params["sigma"]
    gradient = params["gradient"]
    intercept = params["intercept"]
    nsyn_range = params["nsyn_range"]

    # Calculate the intersomatic distance between the current two cells (in 2D - not including depth)
    intersomatic_distance = np.sqrt(
        (sources_x - target["x"]) ** 2 + (sources_z - target["z"]) ** 2
    )

    # if target.node_id % 10000 == 0:
    #     print("Working on tid: ", target.node_id)

    ### Check if there is orientation dependence
    if not np.isnan(gradient):
        # Calculate the difference in orientation tuning between the cells
        delta_orientation = np.array(sources_tuning_angle, dtype=float) - float(
            target["tuning_angle"]
        )

        # For OSI, convert to quadrant from 0 - 90 degrees
        delta_orientation = abs(abs(abs(180.0 - abs(delta_orientation)) - 90.0) - 90.0)

        # Calculate the probability two cells are connected based on distance and orientation
        p_connect = (
            A_new
            * np.exp(-((intersomatic_distance / sigma) ** 2))
            * (intercept + gradient * delta_orientation)
        )

    ### If no orienatation dependence
    else:
        # Calculate the probability two cells are connection based on distance only
        p_connect = A_new * np.exp(-((intersomatic_distance / sigma) ** 2))

    # # Sanity check warning
    # if p_connect > 1:
    #     print 'WARNING WARNING WARNING: p_connect is greater that 1.0 it is: ', p_connect

    # If not the same cell (no self-connections)
    if 0.0 in intersomatic_distance:
        p_connect[np.where(intersomatic_distance == 0.0)[0][0]] = 0

    # Decide which cells get a connection based on the p_connect value calculated
    p_connected = np.random.binomial(1, p_connect)
    p_connected[p_connected == 1] = np.random.randint(
        nsyn_range[0], nsyn_range[1], len(p_connected[p_connected == 1])
    )

    nsyns_ret = [Nsyn if Nsyn != 0 else None for Nsyn in p_connected]
    return nsyns_ret


def get_selection_probability(src_type, lgn_models_subtypes_dictionary):

    current_model_subtypes = lgn_models_subtypes_dictionary[src_type[0:4]]["sub_types"]
    current_model_probabilities = lgn_models_subtypes_dictionary[src_type[0:4]][
        "probabilities"
    ]
    lgn_model_idx = [
        i for i, model in enumerate(current_model_subtypes) if src_type == model
    ][0]
    return current_model_probabilities[lgn_model_idx]


def convert_x_to_lindegs(xcoords):
    return np.tan(0.07 * np.array(xcoords) * np.pi / 180.0) * 180.0 / np.pi


def convert_z_to_lindegs(zcoords):
    return np.tan(0.04 * np.array(zcoords) * np.pi / 180.0) * 180.0 / np.pi


@njit
def within_ellipse(x, y, tuning_angle, e_x, e_y, e_cos, e_sin, e_a, e_b):
    """ check if x, y are within the ellipse """
    x0 = x - e_x
    y0 = y - e_y
    if tuning_angle is None:
        x_rot = x0
        y_rot = y0
    else:
        x_rot = x0 * e_cos - y0 * e_sin
        y_rot = x0 * e_sin + y0 * e_cos
    return ((x_rot / e_a) ** 2 + (y_rot / e_b) ** 2) <= 1.0


# @profile
def select_lgn_sources(
    sources, target, lgn_mean, lgn_ids, lgn_x, lgn_y, cell_type_dict
):
    target_id = target.node_id

    parametersDictionary = lgn_params
    pop_name = [key for key in parametersDictionary if key in target["pop_name"]][0]

    # Check if target supposed to get a connection and if not, then no need to keep calculating.
    if np.random.random() > parametersDictionary[pop_name]["probability"]:
        return [None] * len(lgn_ids)

    if target_id % 250 == 0:
        print("connection LGN cells to V1 cell #", target_id)

    subfields_centers_distance_min = parametersDictionary[pop_name][
        "centers_d_min"
    ]  # 10.0
    subfields_centers_distance_max = parametersDictionary[pop_name][
        "centers_d_max"
    ]  # 11.0  # 11.0
    subfields_centers_distance_L = (
        subfields_centers_distance_max - subfields_centers_distance_min
    )

    subfields_ON_OFF_width_min = parametersDictionary[pop_name][
        "ON_OFF_w_min"
    ]  # 6.0 8.0 #10.0 #8.0 #8.0 #14.0 #15.0
    subfields_ON_OFF_width_max = parametersDictionary[pop_name][
        "ON_OFF_w_max"
    ]  # 8.0 10.0 #12.0 #10.0 #15.0 #20.0
    subfields_ON_OFF_width_L = subfields_ON_OFF_width_max - subfields_ON_OFF_width_min

    subfields_width_aspect_ratio_min = parametersDictionary[pop_name][
        "aspectRatio_min"
    ]  # 2.8  # 1.9 #1.4 #0.9 #1.0
    subfields_width_aspect_ratio_max = parametersDictionary[pop_name][
        "aspectRatio_max"
    ]  # 3.0  # 2.0 #1.5 #1.1 #1.0
    subfields_width_aspect_ratio_L = (
        subfields_width_aspect_ratio_max - subfields_width_aspect_ratio_min
    )

    # Convert to lin_degrees as what is used by the function select_source_cells below
    # There is a corresponding write-up with screen shots that explains the below conversion. Briefly:
    # From Niell et. al, chapter 29 of "The New Visual Neurosciences", Fig 29.1D, a square with ~0.5 mm on a side
    # corresponds to ~35 degrees in x (azimuth) and ~20 degrees in z (elevation).
    # Also, from the figure, we can assume that in both the azimuth and elevation directions, the scale
    # is approximately constant and not warped. Hence, for the x (azimuth) and z (elevation),
    # the visual degree traversed per mm of cortex can be determined:
    # In azimuth, 35/0.5 = 70 degs/mm
    # In elevation, 20/0.5 = 40 degs/mm
    # From this we can convert a translation in x & z in cortex to a translation in visual space.
    # For example, consider moving 0.85mm in the azimuth, the movement in visual space is then estimated
    # to be 0.85 * 70 = 59.5 degrees.
    # The x and z poistions are then converted to linear degrees: tan(x) * (180/pi)
    # Note that before the tangent is taken, the angle is converted to radians
    # The same conversion was done for the mean and dimensions.
    x_position_lin_degrees = convert_x_to_lindegs(target["x"])
    y_position_lin_degrees = convert_z_to_lindegs(target["z"])

    vis_x = lgn_mean[0] + (
        (x_position_lin_degrees)
    )  # - l4_mean[0]) / l4_dim[0]) * lgn_dim[0]
    vis_y = lgn_mean[1] + (
        (y_position_lin_degrees)
    )  # - l4_mean[2]) / l4_dim[2]) * lgn_dim[1]

    ellipse_center_x0 = vis_x  # tar_cells[tar_gid]['vis_x']
    ellipse_center_y0 = vis_y  # tar_cells[tar_gid]['vis_y']

    tuning_angle = float(target["tuning_angle"])
    tuning_angle = None if math.isnan(tuning_angle) else tuning_angle
    # tuning_angle = None if math.isnan(target['tuning_angle']) else target['tuning_angle']
    if tuning_angle is None:
        ellipse_b0 = (
            subfields_ON_OFF_width_min + random() * subfields_ON_OFF_width_L
        ) / 2.0  # Divide by 2 to convert from width to radius.
        ellipse_b0 = 2.5 * ellipse_b0  # 1.5 * ellipse_b0
        ellipse_a0 = ellipse_b0  # ellipse_b0
        top_N_src_cells_subfield = 15  # 20
        ellipses_centers_halfdistance = 0.0
    else:
        tuning_angle_value = float(tuning_angle)
        ellipses_centers_halfdistance = (
            subfields_centers_distance_min + random() * subfields_centers_distance_L
        ) / 2.0
        ellipse_b0 = (
            subfields_ON_OFF_width_min + random() * subfields_ON_OFF_width_L
        ) / 2.0  # Divide by 2 to convert from width to radius.
        ellipse_a0 = ellipse_b0 * (
            subfields_width_aspect_ratio_min + random() * subfields_width_aspect_ratio_L
        )
        ellipse_phi = (
            tuning_angle_value + 180.0 + 90.0
        )  # Angle, in degrees, describing the rotation of the canonical ellipse away from the x-axis.
        ellipse_cos_mphi = math.cos(-math.radians(ellipse_phi))
        ellipse_sin_mphi = math.sin(-math.radians(ellipse_phi))
        top_N_src_cells_subfield = 8  # 10 #9

        ###############################################################################################################
        probability_sON = parametersDictionary[pop_name]["sON_ratio"]
        if np.random.random() < probability_sON:
            cell_sustained_unit = "sON_"
        else:
            cell_sustained_unit = "sOFF_"

    cell_TF = np.random.poisson(parametersDictionary[pop_name]["poissonParameter"])
    while cell_TF <= 0:
        cell_TF = np.random.poisson(parametersDictionary[pop_name]["poissonParameter"])

    sON_subunits = np.array([1.0, 2.0, 4.0, 8.0])
    sON_sum = np.sum(abs(cell_TF - sON_subunits))
    p_sON = (1 - abs(cell_TF - sON_subunits) / sON_sum) / (len(sON_subunits) - 1)

    sOFF_subunits = np.array([1.0, 2.0, 4.0, 8.0, 15.0])
    sOFF_sum = np.sum(abs(cell_TF - sOFF_subunits))
    p_sOFF = (1 - abs(cell_TF - sOFF_subunits) / sOFF_sum) / (len(sOFF_subunits) - 1)

    tOFF_subunits = np.array([4.0, 8.0, 15.0])
    tOFF_sum = np.sum(abs(cell_TF - tOFF_subunits))
    p_tOFF = (1 - abs(cell_TF - tOFF_subunits) / tOFF_sum) / (len(tOFF_subunits) - 1)

    # to match previous algorithm reorganize source cells by type
    # cell_type_dict = {}
    # for lgn_model in lgn_models:
    #     # print(lgn_model)
    #     cell_type_dict[lgn_model] = [
    #         (src_id, src_dict)
    #         for src_id, src_dict in zip(source_ids, sources)
    #         if src_dict["pop_name"] == lgn_model
    #     ]

    # the above part has been a bottleneck. it is now precomputed.

    lgn_models_subtypes_dictionary = {
        "sON_": {
            "sub_types": ["sON_TF1", "sON_TF2", "sON_TF4", "sON_TF8"],
            "probabilities": p_sON,
        },
        "sOFF": {
            "sub_types": ["sOFF_TF1", "sOFF_TF2", "sOFF_TF4", "sOFF_TF8", "sOFF_TF15"],
            "probabilities": p_sOFF,
        },
        "tOFF": {
            "sub_types": ["tOFF_TF4", "tOFF_TF8", "tOFF_TF15"],
            "probabilities": p_tOFF,
        },
    }
    ##################################################################################################################

    # For this target cell, if it has tuning, select the input cell types
    # Note these parameters will not matter if the cell does not have tuning but are calculated anyway
    # Putting it here instead of previous if-else statement for clarity
    # cumulativeP = np.cumsum(connectivityRatios[pop_name]['probabilities'])
    # lgn_model_idx = np.where((np.random.random() < np.array(cumulativeP)) == True)[0][0]
    # sustained_Subunit = connectivityRatios[pop_name]['lgn_models'][lgn_model_idx][0]
    # transient_Subunit = connectivityRatios[pop_name]['lgn_models'][lgn_model_idx][1]
    src_cells_selected = {}
    for src_type in cell_type_dict.keys():
        src_cells_selected[src_type] = []

        if tuning_angle is None:
            ellipse_center_x = ellipse_center_x0
            ellipse_center_y = ellipse_center_y0
            ellipse_a = ellipse_a0
            ellipse_b = ellipse_b0
        else:
            if "tOFF_" in src_type[0:5]:
                ellipse_center_x = (
                    ellipse_center_x0 + ellipses_centers_halfdistance * ellipse_sin_mphi
                )
                ellipse_center_y = (
                    ellipse_center_y0 + ellipses_centers_halfdistance * ellipse_cos_mphi
                )
                ellipse_a = ellipse_a0
                ellipse_b = ellipse_b0
            elif "sON_" in src_type[0:5] or "sOFF_" in src_type[0:5]:
                ellipse_center_x = (
                    ellipse_center_x0 - ellipses_centers_halfdistance * ellipse_sin_mphi
                )
                ellipse_center_y = (
                    ellipse_center_y0 - ellipses_centers_halfdistance * ellipse_cos_mphi
                )
                ellipse_a = ellipse_a0
                ellipse_b = ellipse_b0
            else:
                # Make this a simple circle.
                ellipse_center_x = ellipse_center_x0
                ellipse_center_y = ellipse_center_y0
                # Make the region from which source cells are selected a bit smaller for the transient_ON_OFF cells,
                # since each source cell in this case produces both ON and OFF responses.
                ellipse_b = (
                    ellipses_centers_halfdistance / 2.0
                )  # 0.01 #ellipses_centers_halfdistance + 1.0*ellipse_b0 #0.01 #0.5 * ellipse_b0 # 0.8 * ellipse_b0
                ellipse_a = ellipse_b0  # 0.01 #ellipse_b0

        # Find those source cells of the appropriate type that have their visual space coordinates within the ellipse.
        ellipse_params = (
            ellipse_center_x,
            ellipse_center_y,
            ellipse_cos_mphi,
            ellipse_sin_mphi,
            ellipse_a,
            ellipse_b,
        )

        # speeding up by extracting this element outside
        in_ellipse = within_ellipse(
            lgn_x[src_type], lgn_y[src_type], tuning_angle, *ellipse_params
        )

        for src_id, src_dict in compress(cell_type_dict[src_type], in_ellipse):
            if tuning_angle is not None:
                if src_type == "sONsOFF_001" or src_type == "sONtOFF_001":
                    src_tuning_angle = float(src_dict["tuning_angle"])
                    delta_tuning = abs(
                        abs(
                            abs(
                                180.0
                                - abs(tuning_angle_value - src_tuning_angle) % 360.0
                            )
                            - 90.0
                        )
                        - 90.0
                    )
                    if delta_tuning < 15.0:
                        src_cells_selected[src_type].append(src_id)

                # elif src_type in ['sONtOFF_001']:
                #     src_cells_selected[src_type].append(src_id)

                elif cell_sustained_unit in src_type[:5]:
                    selection_probability = get_selection_probability(
                        src_type, lgn_models_subtypes_dictionary
                    )
                    if np.random.random() < selection_probability:
                        src_cells_selected[src_type].append(src_id)

                elif "tOFF_" in src_type[:5]:
                    selection_probability = get_selection_probability(
                        src_type, lgn_models_subtypes_dictionary
                    )
                    if np.random.random() < selection_probability:
                        src_cells_selected[src_type].append(src_id)

            else:
                if src_type == "sONsOFF_001" or src_type == "sONtOFF_001":
                    src_cells_selected[src_type].append(src_id)
                else:
                    selection_probability = get_selection_probability(
                        src_type, lgn_models_subtypes_dictionary
                    )
                    if np.random.random() < selection_probability:
                        src_cells_selected[src_type].append(src_id)

    select_cell_ids = [
        id for _, selected in src_cells_selected.items() for id in selected
    ]

    # this section has been replaced with the code below
    # if len(select_cell_ids) > 30:
    #     select_cell_ids = np.random.choice(select_cell_ids, 30, replace=False)
    # nsyns_ret = [
    #     parametersDictionary[pop_name]["N_syn"] if id in select_cell_ids else None
    #     for id in lgn_ids
    # ]

    selected_index = np.where(np.isin(lgn_ids, select_cell_ids))[0]
    nsyns_ret = [None] * len(lgn_ids)
    for i in selected_index:
        nsyns_ret[i] = parametersDictionary[pop_name]["N_syn"]
    return nsyns_ret


def select_lgn_sources_powerlaw(sources, target, lgn_mean, lgn_nodes):
    target_id = target.node_id
    # it's a little scary to ahve lgn_params here...
    pop_name = [key for key in lgn_params if key in target["pop_name"]][0]

    if target_id % 250 == 0:
        print("connection LGN cells to V1 cell #", target_id)

    # x_position_lin_degrees = convert_x_to_lindegs(target["x"])
    # y_position_lin_degrees = convert_z_to_lindegs(target["z"])
    # this is simpler for the new coordinate
    x_position_lin_degrees = target["x"] * 0.07
    y_position_lin_degrees = target["z"] * 0.04

    # center of the visual RF
    vis_x = lgn_mean[0] + x_position_lin_degrees
    vis_y = lgn_mean[1] + y_position_lin_degrees
    rf_center = vis_x + 1j * vis_y

    tuning_angle = float(target["tuning_angle"])
    tuning_angle = None if math.isnan(tuning_angle) else tuning_angle
    testing = False

    if tuning_angle is not None:
        tuning_angle_rad = tuning_angle / 180.0 * math.pi
        if testing:
            rf_shift_vector = -np.exp(1j * tuning_angle_rad)  # tentatively flipping
            probability_sON = 1.0  # for testing purpose fix later
        else:
            rf_shift_vector = np.exp(1j * tuning_angle_rad)  # using complex expression
            probability_sON = lgn_params[pop_name]["sON_ratio"]
        if np.random.random() < probability_sON:
            cell_ignore_unit = "sOFF"  # sON cell. ignore sOFF
        else:
            cell_ignore_unit = "sON"
            rf_shift_vector = -rf_shift_vector  # This will be flipped.

    # not very comfortable with this.
    cell_TF = np.random.poisson(lgn_params[pop_name]["poissonParameter"])
    while cell_TF <= 0:
        cell_TF = np.random.poisson(lgn_params[pop_name]["poissonParameter"])

    subunit_freqs = {"sON": [1, 2, 4, 8], "sOFF": [1, 2, 4, 8, 15], "tOFF": [4, 8, 15]}
    # calculate probability for each subunit types separately

    # start calculating the relative probability for each candidate neuron
    # make a candidate pool (circular)

    # circle with radius 40 centered at vis_x, vis_y
    big_circle = (vis_x, vis_y, 1.0, 0.0, 40, 40)
    in_circle = within_ellipse(
        np.array(lgn_nodes["x"]), np.array(lgn_nodes["y"]), tuning_angle, *big_circle
    )

    lgn_circle = lgn_nodes[in_circle]

    # RF center of LGN cell as a complex number
    lgn_complex = np.array(lgn_circle["x"] + 1j * lgn_circle["y"])

    # shift sustained and transient units
    if testing:
        shift = 2.5  # amount to shift the RF
    else:
        shift = 2.5  # amount to shift the RF
    lgn_complex[lgn_circle["pop_name"].str.contains("sON_")] -= rf_shift_vector * shift
    lgn_complex[lgn_circle["pop_name"].str.contains("sOFF_")] -= rf_shift_vector * shift
    lgn_complex[lgn_circle["pop_name"].str.contains("tOFF_")] += rf_shift_vector * shift

    # next, elongate the LGN complex orthogonal to the shift vector
    # rotate by shift vector to adjust the angle, strech, and rotate back.
    # sq_asr = np.sqrt(1.15)  # sqrt of aspect ratio (take from data later)
    sq_asr = np.sqrt(1.5)  # sqrt of aspect ratio (take from data later)
    lgn_relative = lgn_complex - rf_center
    lgn_rotate = lgn_relative / rf_shift_vector
    lgn_strech = lgn_rotate.real * sq_asr + 1j * lgn_rotate.imag / sq_asr
    lgn_back = lgn_strech * rf_shift_vector

    # relative_rf_dist = np.abs(lgn_complex - rf_center)
    relative_rf_dist = np.abs(lgn_back)

    gauss_radius = 5.0  # extention of LGN axons in degrees
    gaussian_prob = gaussian_probability(relative_rf_dist, gauss_radius)

    # assign relative probability calculated above
    subunit_prob = np.zeros_like(gaussian_prob)
    for subname, freqs in subunit_freqs.items():
        if cell_ignore_unit in subname:
            # ignore either sON or sOFF
            continue

        probs = calculate_subunit_probs(cell_TF, [float(f) for f in freqs])
        for i in range(len(probs)):
            # construct the name
            typename = f"{subname}_TF{freqs[i]}"
            subunit_prob[lgn_circle["pop_name"].str.contains(typename)] = probs[i]

    # treatments for sONsOFF and sONtOFF cells
    # tuning_angles are defined only for sONsOFF and sONtOFF cells, so this should be fine.
    lgn_ori_eligible = (
        delta_ori(lgn_circle["tuning_angle"] - target["tuning_angle"]) < 15
    )
    subunit_prob[lgn_ori_eligible] = 1.0

    total_prob = gaussian_prob * subunit_prob
    total_prob = total_prob / sum(total_prob)  # normalize
    # num_cons = np.random.randint(100, 700)
    original_exp = lgn_params[pop_name]["probability"] * lgn_params[pop_name]["N_syn"]
    num_syns_mean = 500 / 80 * original_exp  # so that e4 becomes 500
    logn_sigma = 0.66  # from data
    yule_param = lgn_params[pop_name]["yuleParameter"]
    num_cons_mean = num_syns_mean * (yule_param - 1) / yule_param
    logn_mu = np.log(num_cons_mean) - logn_sigma ** 2 / 2

    num_cons = int(np.random.lognormal(logn_mu, logn_sigma))

    # num_cons = int(np.random.lognormal(5.8, 0.661))  # from LGN statistics.
    # num_cons = int(np.random.lognormal(5.8, 0.2))  # more realistic value...
    # num_cons = 400

    # this line avoid crashing when you don't have sufficinet number of source LGN neurons
    num_cons = int(np.random.lognormal(logn_mu, logn_sigma))
    num_cons = min(num_cons, sum(total_prob > 0))
    selected_locs = pick_from_probs(num_cons, total_prob)

    # now neurons are selected. Next set the number of synapses
    # draw from the Yule distribution (scipy)
    num_syns_indv = yulesimon.rvs(yule_param, size=num_cons)
    # num_syns_indv = yulesimon.rvs(4.2, size=num_cons)
    num_syns, num_neurons = np.unique(num_syns_indv, return_counts=True)
    selected_lgn_inds = lgn_circle.index[selected_locs]
    selected_lgn_dist = relative_rf_dist[selected_locs]

    nsyns_ret = np.zeros(len(lgn_nodes), dtype=int)
    for i in range(len(num_syns) - 1, 0, -1):
        gaussian_prob = gaussian_probability(
            selected_lgn_dist, gauss_radius / np.sqrt(num_syns[i])
        )
        gaussian_prob = gaussian_prob / sum(gaussian_prob)
        syn_selected = pick_from_probs(num_neurons[i], gaussian_prob)
        nsyns_ret[selected_lgn_inds[syn_selected]] = num_syns[i]

        selected_lgn_inds = np.delete(selected_lgn_inds, syn_selected)
        selected_lgn_dist = np.delete(selected_lgn_dist, syn_selected)

    # there should be 1 synapse connections remaining
    nsyns_ret[selected_lgn_inds] = 1
    nsyns_ret = list(nsyns_ret)

    # getting back to list
    nsyns_ret = [None if n == 0 else n for n in nsyns_ret]
    # return
    return nsyns_ret


def pick_from_probs(n, prob_dist):
    # pick n item based on prob_dist and return index of the choice
    return np.random.choice(
        list(range(len(prob_dist))), size=n, replace=False, p=prob_dist
    )


def gaussian_probability(x, sigma):
    return np.exp(-(x ** 2 / (2 * sigma ** 2)))


def calculate_subunit_probs(cell_TF, tf_list):
    tf_array = np.array(tf_list)
    tf_sum = np.sum(abs(cell_TF - tf_array))
    p = (1 - abs(cell_TF - tf_array) / tf_sum) / (len(tf_array) - 1)
    return p


def general_candidate_pool_ellipse(vis_x, vis_y):
    visx_width = 5 * 4  # to cover most candidate cells
    visy_width = 5 * 4
    ellipse_center_x = vis_x
    ellipse_center_y = vis_y
    ellipse_cos_mphi = (1.0,)
    ellipse_sin_mphi = (0.0,)
    ellipse_a = visx_width
    ellipse_b = visy_width
    ellipse_params = (
        ellipse_center_x,
        ellipse_center_y,
        ellipse_cos_mphi,
        ellipse_sin_mphi,
        ellipse_a,
        ellipse_b,
    )
    return ellipse_params


# orientation comparator
def delta_ori(angle):
    return np.abs(np.abs(angle - 90) % 180 - 90)
