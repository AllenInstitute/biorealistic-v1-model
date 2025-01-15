""" This is a support script for build_network.py.
This is supposed to read necessary human editable files and generate files required by build_network.py

input files:
base_props/V1model_seed_file.xlsx: human editable file that contains information of each population
glif_requisite/glif_models_prop.csv: file that contains LIF cell types from cell type database

output files: glif_props/v1_node_models.csv
# dependencies are written in makefile, so refer to that as a part of documentation.


Necessary information:
locations e.g. 'VisL23'
pop_name e.g. 'e23Cux2'
ncells 56057
ei 'e'
upper_bound e.g. 100.0
lower_bound e.g. 310.0
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
import argparse
from pathlib import Path


def extract_info(row):
    d = {}
    # exception
    d["ncells"] = row["pop_combined_count"]

    prop_names = [
        "ei",
        "pop_name",
        "upper_bound",
        "lower_bound",
        "nsyn_lognorm_shape",
        "nsyn_lognorm_scale",
    ]

    for prop in prop_names:
        d[prop] = row[prop]

    return d


def extract_criteria(row):
    d = {}
    query = ["ei", "location", "reporter_status"]
    for q in query:
        d[q] = row[q]
    # for non-simple ones, do manually
    d["cre_line"] = row["cre_line"].split(",")
    return d


def distribute_nums(n, m):
    # distribute n to m entities. used to distribute cells into models.
    base_num = n // m
    residual = n % m
    counts = np.array([base_num] * m)
    counts[:residual] += 1  # add number to distribute the residual
    return counts


def pick_glif_models(models_df, row, douple_alpha=False):
    # models are pre-selected, so you can directly search with pop_name
    selected_df = models_df[models_df["pop_name"] == row["pop_name"]]

    ncell_all = int(row["pop_combined_count"])
    n_models = selected_df.shape[0]
    assert n_models > 0
    model_cell_count = distribute_nums(ncell_all, n_models)

    models = []
    for i in range(n_models):
        poprow = selected_df.iloc[i]
        model_dict = {}
        model_dict["N"] = int(model_cell_count[i])
        model_dict["node_type_id"] = int(poprow["specimen__id"])
        model_dict["model_type"] = "point_process"
        if douple_alpha:
            model_dict["model_template"] = "nest:glif_psc_double_alpha"
        else:
            model_dict["model_template"] = "nest:glif_psc"
        model_dict["dynamics_params"] = poprow["parameters_file"]
        models.append(model_dict)

    return models


def make_v1_node_models(args):
    filepath = "base_props/V1model_seed_file.xlsx"
    outfilepath = "glif_props/v1_node_models.csv"

    db = xl.readxl(filepath)
    table = db.ws("cell_models").ssd(keycols="pop_id", keyrows="pop_id")
    t0 = table[0]
    seed_df = pd.DataFrame(data=t0["data"], index=t0["keyrows"], columns=t0["keycols"])
    glif_models_df = pd.read_csv("glif_requisite/glif_models_prop.csv", sep=" ")
    # node_models = {"locations": {}}
    # Load unitary v1 synapse amps:
    node_models = {}  # make a dataframe to store in csv

    for location, subdf in seed_df.groupby("location"):
        location_dict = {}
        for pop_name, row in subdf.iterrows():
            pop_dict = extract_info(row)
            models = pick_glif_models(glif_models_df, row, args.double_alpha)
            # pop_dict["models"] = models
            # instead of containing it, let's update the dict.
            for m in models:
                m.update(pop_dict)
                m["locations"] = location
                node_type_id = m.pop("node_type_id")
                node_models[node_type_id] = m

            # location_dict[pop_name_change(row["pop_name"])] = pop_dict

        # node_models["locations"][location] = location_dict

    # change it to pathlib versino
    Path("glif_props").mkdir(parents=True, exist_ok=True)

    # with open(outfilepath, "w") as f:
    # json.dump(node_models, f, indent=2)

    node_models_df = pd.DataFrame(node_models).T
    node_models_df.index.name = "node_type_id"
    node_models_df.to_csv(outfilepath, sep=" ")


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
        "-d",
        "--double-alpha",
        action="store_true",
        default=False,
        help="use nest glif model with double alpha synapses",
    )
    args = parser.parse_args()

    make_v1_node_models(args)
