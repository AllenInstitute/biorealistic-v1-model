# %%simple adjustment of the bkg weight with recurrent connections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bkg_weight_adjustment as bwa
import bkg_weight_adjustment_minuit as bwam
import sys
import utils
from importlib import reload


basedir = "small"
v1df = bwa.get_v1_dfs(basedir)
v1df["cell_type"] = v1df["pop_name"].map(utils.pop_name_to_cell_type)
target = "median"
# tfr = bwa.get_target_fr(basedir, target="target_median_fr")
tfr = pd.read_csv("base_props/bkg_weights_population_init.csv", sep=" ")
tfr["cell_type"] = tfr["population"].map(utils.pop_name_to_cell_type)
tfr = tfr[["cell_type", "target_median_fr"]].drop_duplicates()
tfr.index = tfr["cell_type"]
tfr = tfr["target_median_fr"]
# tfr.index = tfr.population.map(utils.pop_name_to_cell_type)
# tfr.index.name = "cell_type"
# tfr = tfr.drop_duplicates()["target_median_fr"]

# make a table of node_type_id to cell_type
node_id_to_cell_type = v1df[["node_type_id", "cell_type"]].copy().drop_duplicates()
node_id_to_cell_type.index = node_id_to_cell_type["node_type_id"]
node_id_to_cell_type = node_id_to_cell_type["cell_type"]
# node_id_to_cell_type


# %% load initial weights
reload(bwa)
duration = 10
# initial_file = "precomputed_props/bkg_v1_edge_types_0605_250Hz.csv"
initial_file = "precomputed_props/bkg_v1_edge_types_0607_noon.csv"
bkg_types = pd.read_csv(initial_file, index_col=0, sep=" ")
bkg_types["target_type_id"] = bkg_types["target_query"].str.extract("(\d+)").astype(int)
bkg_types.index = bkg_types["target_type_id"]


for i in range(10000):
    # write the weights and run simulation
    print(f"iteration {i}")
    bkg_types["syn_weight"]
    bkg_types["cell_type"] = node_id_to_cell_type
    df = bwa.update_bkg_weights(basedir, bkg_types["syn_weight"])

    # run
    bwa.run_simulation(basedir, recurrent=True, ncore=6)
    # %%
    # get the model firing rates
    model_fr = bwa.get_model_fr(
        basedir, recurrent=True, duration=duration, target="type_median"
    )
    model_fr
    # %%

    # calculate the score
    # rel_diff = (model_fr - tfr) / tfr
    rel_diff = np.log2(model_fr / tfr)
    # pick the one with the largest diff
    worst = rel_diff.abs().idxmax()
    # check the sign
    sign = np.sign(rel_diff.loc[worst])
    # update the weight
    increment = 0.05  # 5%
    ratio = 1.0 - sign * increment
    # identiry ids to update
    ids = bkg_types[bkg_types["cell_type"] == worst].index
    bkg_types.loc[ids, "syn_weight"] *= ratio

    # display the weights
    show_df = tfr.to_frame()
    show_df["model_fr"] = model_fr
    show_df["rel_diff"] = rel_diff
    show_df["picked"] = show_df.index == worst
    # change picked to "*" or ""
    show_df["picked"] = show_df["picked"].map(lambda x: "*" if x else "")
    print(show_df)


# %%
