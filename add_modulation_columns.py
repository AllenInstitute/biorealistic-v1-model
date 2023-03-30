# %% This function adds the modulation columns to the edge types.
# If the function already exists, it will be overwritten with the new name.
# The name of the column is "weight_function"

# For v1 to v1 connections, fill the elements with "weight_function_recurrent"
# for LGN to v1 connection, fill the elements with "weight_function_lgn"
# for BKG to v1 connection, fill the elements with "weight_function_bkg"
import pandas as pd
import sys

# basedir = "small"  # eventually becomes sys.argv[1]
basedir = sys.argv[1]
types = ["recurrent", "lgn", "bkg"]

types_file_names = {
    "recurrent": basedir + "/network/v1_v1_edge_types.csv",
    "lgn": basedir + "/network/lgn_v1_edge_types.csv",
    "bkg": basedir + "/network/bkg_v1_edge_types.csv",
}

func_names = {
    "recurrent": "weight_function_recurrent",
    "lgn": "weight_function_lgn",
    "bkg": "weight_function_bkg",
}

for t in types:
    df = pd.read_csv(types_file_names[t], sep=" ", index_col=1)
    df["weight_function"] = func_names[t]
    df.to_csv(types_file_names[t], sep=" ")
