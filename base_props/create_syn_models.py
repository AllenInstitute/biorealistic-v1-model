import pandas as pd
import json
import numpy as np
import os


syn_types = pd.read_csv("syn_types_table.csv", index_col=0)
syn_types_syn_tau = pd.read_csv("syn_types_syn_tau.csv", index_col=0)
syn_models_dir = "syn_models_new_May31/"

os.mkdir(syn_models_dir)


cell_pops_pre = ["e23", "e4", "e5et", "e5it", "e5np", "e6", "pv", "sst", "vip", "lgn"]
cell_pops_post = ["e23", "e4", "e5et", "e5it", "e5np", "e6", "pv", "sst", "vip"]

for pre_pop in cell_pops_pre:
    for post_pop in cell_pops_post:
        syn_name = syn_types[post_pop].loc[pre_pop]
        syn = {"tau_syn": float(syn_types_syn_tau[post_pop].loc[pre_pop])}
        with open(syn_models_dir + str(syn_name) + ".json", "w") as f:
            json.dump(syn, f)
