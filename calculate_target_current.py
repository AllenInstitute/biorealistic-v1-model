# %%

import pandas as pd
import numpy as np
from utils import pop_name_to_cell_type as pnames
from scipy.interpolate import interp1d

# rev_pname = {v: k for k, v in pnames.items()}


npdf = pd.read_csv("neuropixels/metrics/OSI_DSI_DF.csv", sep=" ")
sp_rates = npdf.groupby("cell_type").mean()["Avg_Rate(Hz)"]
# sp_rates = npdf.groupby("cell_type").median()["Avg_Rate(Hz)"]
sp_rates
target_fr = sp_rates

# if index contain white space, replace it with _
target_fr.index = target_fr.index.str.replace(" ", "_")

# if target_fr has a key called "L1 Htr3a", change it to "L1 Inh"
if "L1_Htr3a" in target_fr:
    target_fr["L1_Inh"] = target_fr.pop("L1_Htr3a")

# Also, L5 Exc should be expanded to L5 IT, L5 ET, and L5 NP
if "L5_Exc" in target_fr:
    target_fr["L5_IT"] = target_fr["L5_Exc"]
    target_fr["L5_ET"] = target_fr["L5_Exc"]
    target_fr["L5_NP"] = target_fr["L5_Exc"]
    target_fr.pop("L5_Exc")


df = pd.read_csv("glif_requisite/glif_models_prop.csv", sep=" ", index_col=0)
df["cell_type"] = df.pop_name.apply(lambda x: pnames[x])


# if curves are necessary, and I'll prepare a way to get that.
# if_curves = np.load("glif_models/if_curves_all.npy", allow_pickle=True)
if_curves = pd.read_csv("glif_models/if_curves_all.csv", sep=" ", index_col=0)

# get the stim_amps from the index.
stim_amps = if_curves.columns.values.astype(float)


def calculate_target_current(type, subdf):
    # type_in_fr = rev_pnames[type]
    # target_fr_type = target_fr[target_fr.cell_type == type_in_fr]["firing_rate_dg_mean"]
    target_fr_type = target_fr[type]

    ind = subdf.index
    if_curve = if_curves.loc[ind].mean(axis=0)
    # this if_curve is the average firing rate.
    # solve for the target current.
    target_current = interp1d(if_curve, stim_amps)(target_fr_type)
    return target_current


types = []
target_currents = []
for type, subdf in df.groupby("cell_type"):
    target_currents.append(calculate_target_current(type, subdf))
    types.append(type)

target_currents = np.array(target_currents)

target_current_df = pd.DataFrame(
    target_currents, index=types, columns=["target_current"]
)
target_current_df.index.name = "cell_type"
target_current_df.to_csv("glif_models/target_currents.csv", sep=" ")
