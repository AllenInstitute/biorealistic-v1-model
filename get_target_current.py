# %%

import pandas as pd
import numpy as np
from utils import pop_name_to_cell_type2 as pnames
from scipy.interpolate import interp1d

# rev_pname = {v: k for k, v in pnames.items()}


npdf = pd.read_csv("neuropixels/metrics/OSI_DSI_DF.csv", sep=" ")
sp_rates = npdf.groupby("cell_type").mean()["Avg_Rate(Hz)"]
target_fr = sp_rates

# if target_fr has a key called "L1 Htr3a", change it to "L1 Inh"
if "L1 Htr3a" in target_fr:
    target_fr["L1 Inh"] = target_fr.pop("L1 Htr3a")

# Also, L5 Exc should be expanded to L5 IT, L5 ET, and L5 NP
if "L5 Exc" in target_fr:
    target_fr["L5 IT"] = target_fr["L5 Exc"]
    target_fr["L5 ET"] = target_fr["L5 Exc"]
    target_fr["L5 NP"] = target_fr["L5 Exc"]
    target_fr.pop("L5 Exc")


# TODO: fix this later... is shouldn't depend on a network instance
df = pd.read_csv("small/network/v1_node_types.csv", sep=" ")
df["cell_type"] = df.pop_name.apply(lambda x: pnames[x])

# if curves are necessary, and I'll prepare a way to get that.
if_curves = np.load("glif_models/if_curves_all.npy", allow_pickle=True)

# somehow each item is a list of float, so let's convert them to numpy array...
if_curves_np = np.array([np.array(x) for x in if_curves])
if_curves_np.shape
# infer the stimulus amplitude from the shape.
stim_amps = np.linspace(0, 500, if_curves_np.shape[1])


def get_target_current(type, subdf):
    # type_in_fr = rev_pnames[type]
    # target_fr_type = target_fr[target_fr.cell_type == type_in_fr]["firing_rate_dg_mean"]
    target_fr_type = target_fr[type]

    ind = subdf.index
    if_curve = if_curves_np[ind].mean(axis=0)
    # this if_curve is the average firing rate.
    # solve for the target current.
    target_current = interp1d(if_curve, stim_amps)(target_fr_type)
    return target_current


types = []
target_currents = []
for type, subdf in df.groupby("cell_type"):
    target_currents.append(get_target_current(type, subdf))
    types.append(type)

target_currents = np.array(target_currents)

target_current_df = pd.DataFrame(
    target_currents, index=types, columns=["target_current"]
)
target_current_df.index.name = "cell_type"
target_current_df.to_csv("glif_models/target_currents.csv", sep=" ")


# %%
target_current_df
