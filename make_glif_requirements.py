""" This is a support script for build_network.py.
This is supposed to read necessary human editable files and generate files required by build_network.py

input files:
base_props/V1model_seed_file.xlsx: human editable file that contains information of each population
glif_requisite/glif_models_prop.csv: file that contains LIF cell types from cell type database

output files: glif_props/v1_node_models.json
# dependencies are written in makefile, so refer to that as a part of documentation.


Necessary information:
locations e.g. 'VisL23'
pop_name e.g. 'e23Cux2'
ncells 56057
ei 'e'
depth_range [100.0, 310.0]
and models nested

models
N: 43368
node_type_id: 100000102,
model_type: 'point_process'
model_template: 'nrn:IntFire1'
dynamic_params: e23Cux2_avg_lif.json

for biophysical models there are more requirements
morphology: Cux2-CreERT2_Ai14... take from the file...
model_processing: 'aibs_perisomatic'
rotation_angle_zaxis: -2.728849956000004
"""


# %% reading excel file to get the population info
import pylightxl as xl
import numpy as np
import pandas as pd
import json
import os
import argparse


def extract_info(row):
    d = {}
    d["ncells"] = row["pop_combined_count"]
    d["ei"] = row["ei"]
    d["depth_range"] = [row["upper_bound"], row["lower_bound"]]
    # Adding lognormal parameters for distribution of "size"/synapse numbers
    d["nsyn_lognorm_shape"] = row["nsyn_lognorm_shape"]
    d["nsyn_lognorm_scale"] = row["nsyn_lognorm_scale"]
    return d


def extract_criteria(row):
    d = {}
    query = ["ei", "location", "reporter_status"]
    for q in query:
        d[q] = row[q]
    # for non-simple ones, do manually
    d["cre_line"] = row["cre_line"].split(",")
    return d


def pick_lif_model(models_df, row):
    selected_df = models_df[models_df["pop_name"] == row["pop_name"]]
    model_dict = {}
    model_dict["N"] = int(row["pop_peripheral_count"])
    model_dict["node_type_id"] = int(selected_df["node_type_id"])
    model_dict["model_type"] = "point_process"
    model_dict["model_template"] = "nrn:IntFire1"
    model_dict["dynamics_params"] = selected_df.iloc[0]["parameters_file"]
    return model_dict


def distribute_nums(n, m):
    # distribute n to m entities. used to distribute cells into models.
    base_num = n // m
    residual = n % m
    counts = np.array([base_num] * m)
    counts[:residual] += 1  # add number to distribute the residual
    return counts


def pick_glif_models(models_df, row, v1_synapse_amps):
    # Need short names for indexing:
    pop_name_long2short = {
        "i1Htr3a": "vip",
        "e23Cux2": "e23",
        "i23Vip": "vip",
        "i23Pvalb": "pv",
        "i23Sst": "sst",
        "e4Nr5a1": "e4",
        "e4Rorb": "e4",
        "e4Scnn1a": "e4",
        "e4other": "e4",
        "i4Vip": "vip",
        "i4Pvalb": "pv",
        "i4Sst": "sst",
        "e5IT": "e5it",
        "e5ET": "e5et",
        "e5NP": "e5np",
        "i5Vip": "vip",
        "i5Pvalb": "pv",
        "i5Sst": "sst",
        "e6Ntsr1": "e6",
        "i6Vip": "vip",
        "i6Pvalb": "pv",
        "i6Sst": "sst",
    }
    cell_pops_pre = [
        "e23",
        "e4",
        "e5et",
        "e5it",
        "e5np",
        "e6",
        "pv",
        "sst",
        "vip",
        "lgn",
    ]
    cell_pops_post = ["e23", "e4", "e5et", "e5it", "e5np", "e6", "pv", "sst", "vip"]

    # models are pre-selected, so you can directly search with pop_name
    selected_df = models_df[models_df["pop_name"] == row["pop_name"]]

    ncell_all = int(row["pop_combined_count"])
    n_models = selected_df.shape[0]
    assert n_models > 0
    model_cell_count = distribute_nums(ncell_all, n_models)

    post_pop = row["pop_name"]
    models = []
    for i in range(n_models):
        poprow = selected_df.iloc[i]
        model_dict = {}
        model_dict["N"] = int(model_cell_count[i])
        model_dict["node_type_id"] = int(poprow["specimen__id"])
        model_dict["model_type"] = "point_process"
        model_dict["model_template"] = "nest:glif_psc"
        model_dict["dynamics_params"] = poprow["parameters_file"]
        models.append(model_dict)

    return models


def pick_bio_models(models_df, row):
    criteria = extract_criteria(row)
    selected = (
        (models_df["ei"] == criteria["ei"])
        & (models_df["reporter_status"] == "positive")
        & (models_df["location"] == criteria["location"])
    )
    selected &= (
        models_df["cre_line"].isin(criteria["cre_line"])
        if criteria["reporter_status"] == "positive"
        else ~models_df["cre_line"].isin(criteria["cre_line"])
    )
    ncell_all = row["pop_core_count"]
    models_pop_df = models_df[selected]
    # set number of cells here
    n_models = np.sum(selected)
    # print(row["pop_name"])
    assert n_models > 0
    model_cell_count = distribute_nums(ncell_all, n_models)
    # this will create something like [5 5 4 4 4], ncell_all == 22 & n_models == 5

    models = []
    for i in range(n_models):
        poprow = models_pop_df.iloc[i]
        model_dict = {}
        model_dict["N"] = int(model_cell_count[i])
        model_dict["node_type_id"] = int(poprow["node_type_id"])
        model_dict["model_type"] = "biophysical"
        model_dict["model_template"] = "ctdb:Biophys1.hoc"
        model_dict["dynamics_params"] = poprow["parameters_file"]
        model_dict["morphology"] = poprow["morphology_file"]
        model_dict["model_processing"] = "aibs_perisomatic"
        model_dict["rotation_angle_zaxis"] = poprow["rotation_angle_zaxis"]
        models.append(model_dict)

    return models


def make_v1_node_models(miniature=False):
    if miniature:
        filepath = "base_props/V1model_seed_file_miniature.xlsx"
        outfilepath = "glif_props/v1_node_models_miniature.json"
    else:
        filepath = "base_props/V1model_seed_file.xlsx"
        outfilepath = "glif_props/v1_node_models.json"

    db = xl.readxl(filepath)
    table = db.ws("cell_models").ssd(keycols="pop_id", keyrows="pop_id")
    t0 = table[0]
    seed_df = pd.DataFrame(data=t0["data"], index=t0["keyrows"], columns=t0["keycols"])
    glif_models_df = pd.read_csv("glif_requisite/glif_models_prop.csv", sep=" ")
    node_models = {"locations": {}}
    # Load unitary v1 synapse amps:

    for location, subdf in seed_df.groupby("location"):
        location_dict = {}
        for pop_id, row in subdf.iterrows():
            pop_dict = extract_info(row)
            models = pick_glif_models(glif_models_df, row)
            pop_dict["models"] = models
            location_dict[pop_name_change(row["pop_name"])] = pop_dict

        node_models["locations"][location] = location_dict

    # node_models["inner_radial_range"] = [1.0, 400.0]
    # node_models["outer_radial_range"] = [400.0, 845.0]

    # process general properties
    general_table = db.ws("general_parameters").ssd(
        keycols="properties", keyrows="properties"
    )
    tg = general_table[0]
    general_df = pd.DataFrame(
        data=tg["data"], index=tg["keyrows"], columns=tg["keycols"]
    )
    node_models["core_radius"] = float(general_df.loc["core_radius"])
    node_models["radius"] = float(general_df.loc["radius"])

    if not os.path.exists("glif_props"):
        os.mkdir("glif_props")

    with open(outfilepath, "w") as f:
        json.dump(node_models, f, indent=2)


def pop_name_change(pop_name):
    # handles two exceptions in pop_name
    if pop_name == "VisL2/3":
        return "VisL23"
    if pop_name == "VisL6a":
        return "VisL6"
    return pop_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make a required node definition file for the GLIF model"
    )
    parser.add_argument(
        "-m",
        "--miniature",
        action="store_true",
        default=False,
        help="make a miniature version of the simualtion for debugging",
    )
    args = parser.parse_args()

    make_v1_node_models(args.miniature)
