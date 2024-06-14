# %%
# using the actuation matrix and the target current, figure out how the recurrent
# connections should be adjusted.
import pandas as pd
import numpy as np


target_current = pd.read_csv("glif_models/target_currents.csv", sep=" ", index_col=0)
actuation_matrix = pd.read_csv(
    "core/metrics/actuation_matrix.csv", sep=" ", index_col=0
)


# name conversion again...
neme_conv = {
    "i1H": "L1 Inh",
    "e2": "L2/3 Exc",
    "i23P": "L2/3 PV",
    "i23S": "L2/3 SST",
    "i23V": "L2/3 VIP",
    "e4": "L4 Exc",
    "i4P": "L4 PV",
    "i4S": "L4 SST",
    "i4V": "L4 VIP",
    "e5ET": "L5 ET",
    "e5IT": "L5 IT",
    "e5NP": "L5 NP",
    "i5P": "L5 PV",
    "i5S": "L5 SST",
    "i5V": "L5 VIP",
    "e6": "L6 Exc",
    "i6P": "L6 PV",
    "i6S": "L6 SST",
    "i6V": "L6 VIP",
}
rev_name_conv = {v: k for k, v in neme_conv.items()}

# first, change the row and column names of the actuation matrix,
# if they are contained in name_conv.
actuation_matrix = actuation_matrix.rename(index=neme_conv, columns=neme_conv)
# add the row of the target_current
actuation_matrix = pd.concat([actuation_matrix, target_current.T])

# let's calculate the adjusted LGN rates.
# the original is the average of the spont and stim rates.
# let's change it to the evoked rate.
lgn_spont = 3.84
lgn_evoked = 6.13
lgn_averge = 3.84 / 3 + 6.13 / 3 * 2
lgn_adjust = lgn_evoked / lgn_averge


actuation_matrix.loc["needed_rec"] = (
    actuation_matrix.loc["target_current"]
    - actuation_matrix.loc["bkg"]
    - actuation_matrix.loc["lgn"] * lgn_adjust
)

actuation_matrix.loc["total_exc"] = actuation_matrix.loc[
    ["L2/3 Exc", "L4 Exc", "L5 ET", "L5 IT", "L5 NP", "L6 Exc"]
].sum()
actuation_matrix.loc["total_inh"] = actuation_matrix.loc[
    [
        "L2/3 PV",
        "L2/3 SST",
        "L2/3 VIP",
        "L4 PV",
        "L4 SST",
        "L4 VIP",
        "L5 PV",
        "L5 SST",
        "L5 VIP",
        "L6 PV",
        "L6 SST",
        "L6 VIP",
    ]
].sum()

# the balancing point would be where
# needed_rec = total_exc * alpha + total_inh / alpha
# let's solve for alpha
te = actuation_matrix.loc["total_exc"].to_numpy()
ti = actuation_matrix.loc["total_inh"].to_numpy()
nr = actuation_matrix.loc["needed_rec"].to_numpy()

# zip them and do as a list.
alpha = []
for tev, tiv, nrv in zip(te, ti, nr):
    roots = np.roots([tev, -nrv, tiv])
    # pick the one that's positive.
    root = [r for r in roots if r > 0][0]
    alpha.append(root)

alpha
actuation_matrix.loc["adjustment_factor"] = alpha


actuation_matrix

# %% let's make a modulation file for this network

# the modulation file is a csv file with " " as a separator.
# the columns are, "src_property", "src_substring", "trg_property", "trg_substring", "operation", "value"

# initialize
df_data = {}
df_data["src_property"] = []
df_data["src_substring"] = []
df_data["trg_property"] = []
df_data["trg_substring"] = []
df_data["operation"] = []
df_data["value"] = []


# process for each column of the actuation matrix.
for col in actuation_matrix.columns:
    # two entries for each column. for E and I.
    # basics things first
    df_data["src_property"].append("pop_name")
    df_data["src_property"].append("pop_name")
    df_data["trg_property"].append("pop_name")
    df_data["trg_property"].append("pop_name")
    df_data["src_substring"].append("e")
    df_data["src_substring"].append("i")
    df_data["operation"].append("*")  # first one is for E connection
    df_data["operation"].append("/")  # second one is for I connection

    factor = actuation_matrix.loc["adjustment_factor", col]
    df_data["value"].append(factor)
    df_data["value"].append(factor)

    # determine the substrings.
    substring = rev_name_conv[col]
    df_data["trg_substring"].append(substring)
    df_data["trg_substring"].append(substring)

# actuation_matrix.columns[0]
# actuation_matrix.loc["adjustment_factor", "L1 Inh"]

mod_df = pd.DataFrame(df_data)

# write it down to csv, excluding the index.
mod_df.to_csv("core/metrics/modulation.csv", sep=" ", index=False)


# %%
actuation_matrix.loc["adjustment_factor"]