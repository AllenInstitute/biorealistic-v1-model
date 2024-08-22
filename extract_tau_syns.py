# extract a unique set of tau_syn, tau_syn_slow, amp_slow from the base_props

# %%
import pandas as pd
import numpy as np
from pathlib import Path


# base_props_dir = "../../glif_builder_test/biorealistic-v1-model/base_props/"
base_props_dir = "base_props/"
params = ["tau_syn_fast", "tau_syn_slow", "amp_slow"]

# each parameter is stored in a .csv file, as a matrix. load each matrix, and flatten
# and combile all of the values into a single dataframe.
dfs = []
for param in params:
    df = pd.read_csv(base_props_dir + param + ".csv", sep=" ", index_col=0)
    row_names = df.index
    col_names = df.columns
    # flatten
    df = df.values.flatten()
    dfs.append(df)

# combine into a single dataframe
df = pd.DataFrame(dfs).T
df.columns = params
df["tau_syn_fast"] = df["tau_syn_fast"] * 1000
df["tau_syn_slow"] = df["tau_syn_slow"] * 1000

# take out a unique rows out of the dataframe.
df_unique = df.drop_duplicates().copy()

# change the unit of tau_syn and tau_syn_slow from s to ms.


# make the directory if not yet
Path("tf_props").mkdir(parents=True, exist_ok=True)


# store them in a file.
df_unique.to_csv("tf_props/double_alpha_params.csv", sep=" ", index=False)

# %% full version is also stored.
df_full = df.copy()

# construct index for these.
index = ["{}_to_{}".format(row, col) for row in row_names for col in col_names]
df_full.index = index


df_full.to_csv("tf_props/double_alpha_params_full.csv", sep=" ")
