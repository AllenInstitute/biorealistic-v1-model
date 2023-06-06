import pandas as pd
import json
import numpy as np

# import os
import pathlib


# syn_types = pd.read_csv("base_props/syn_types_table.csv", index_col=0)
# syn_types_syn_tau = pd.read_csv("base_props/syn_types_syn_tau.csv", index_col=0)
tau_syn = pd.read_csv("base_props/tau_syn.csv", index_col=0)
tau_syn_slow = pd.read_csv("base_props/tau_syn_slow.csv", index_col=0)
amp_slow = pd.read_csv("base_props/amp_slow.csv", index_col=0)
# syn_types_syn_tau = pd.read_csv("base_props/syn_types_syn_tau.csv", index_col=0)
syn_models_dir = "glif_models/synaptic_models/"

# os.mkdir(syn_models_dir)
# make syn_model_dir if it doesn't exist
pathlib.Path(syn_models_dir).mkdir(parents=True, exist_ok=True)


cell_pops_pre = ["e23", "e4", "e5et", "e5it", "e5np", "e6", "pv", "sst", "vip", "exc"]
cell_pops_post = ["e23", "e4", "e5et", "e5it", "e5np", "e6", "pv", "sst", "vip"]

for pre_pop in cell_pops_pre:
    for post_pop in cell_pops_post:
        # syn_name = syn_types[post_pop].loc[pre_pop]
        syn_name = f"{pre_pop}_to_{post_pop}"
        syn = {
            "tau_syn": float(tau_syn[post_pop].loc[pre_pop]) * 1000,
            "tau_syn_slow": float(tau_syn_slow[post_pop].loc[pre_pop]) * 1000,
            "amp_slow": float(amp_slow[post_pop].loc[pre_pop]),
        }
        with open(syn_models_dir + str(syn_name) + ".json", "w") as f:
            json.dump(syn, f)
