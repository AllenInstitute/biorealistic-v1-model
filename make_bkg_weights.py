# read bkg_weights_population.csv and write bkg_weights_model.csv
# %%
import pandas as pd
import json
import numpy as np
import pop_names


v1unitary = json.load(open("precomputed_props/v1_synapse_amps.json", "r"))
v1unitary_ser_e23 = pd.Series(v1unitary["lgn_to_e23"], name="Unitary PSP")
v1unitary_ser_e4 = pd.Series(v1unitary["lgn_to_e4"], name="Unitary PSP")
v1unitary_ser_e5et = pd.Series(v1unitary["lgn_to_e5et"], name="Unitary PSP")
v1unitary_ser_e5it = pd.Series(v1unitary["lgn_to_e5it"], name="Unitary PSP")
v1unitary_ser_e5np = pd.Series(v1unitary["lgn_to_e5np"], name="Unitary PSP")
v1unitary_ser_e6 = pd.Series(v1unitary["lgn_to_e6"], name="Unitary PSP")
v1unitary_ser_pv = pd.Series(v1unitary["lgn_to_pv"], name="Unitary PSP")
v1unitary_ser_sst = pd.Series(v1unitary["lgn_to_sst"], name="Unitary PSP")
v1unitary_ser_vip = pd.Series(v1unitary["lgn_to_vip"], name="Unitary PSP")

v1unitary_ser_e23.index = pd.to_numeric(v1unitary_ser_e23.index)
v1unitary_ser_e4.index = pd.to_numeric(v1unitary_ser_e4.index)
v1unitary_ser_e5et.index = pd.to_numeric(v1unitary_ser_e5et.index)
v1unitary_ser_e5it.index = pd.to_numeric(v1unitary_ser_e5it.index)
v1unitary_ser_e5np.index = pd.to_numeric(v1unitary_ser_e5np.index)
v1unitary_ser_e6.index = pd.to_numeric(v1unitary_ser_e6.index)
v1unitary_ser_pv.index = pd.to_numeric(v1unitary_ser_pv.index)
v1unitary_ser_sst.index = pd.to_numeric(v1unitary_ser_sst.index)
v1unitary_ser_vip.index = pd.to_numeric(v1unitary_ser_vip.index)
v1unitary_ser_e23

v1unitary_ser = pd.concat(
    [
        v1unitary_ser_e23,
        v1unitary_ser_e4,
        v1unitary_ser_e5et,
        v1unitary_ser_e5it,
        v1unitary_ser_e5np,
        v1unitary_ser_e6,
        v1unitary_ser_pv,
        v1unitary_ser_sst,
        v1unitary_ser_vip,
    ]
)

# also load up population weight


pop_weights = pd.read_csv("base_props/bkg_weights_population_init.csv", sep=" ")
v1_node_model = json.load(open("glif_props/v1_node_models.json"))

pop_model_dict = {}
for loc, locdict in v1_node_model["locations"].items():
    for pop, popdict in locdict.items():
        for m in popdict["models"]:

            pop_model_dict[m["node_type_id"]] = pop

# %%

pop_df = pd.DataFrame.from_dict(pop_model_dict, orient="index", columns=["population"])
pop_df["model_id"] = pop_df.index
pop_psp = pop_df.merge(v1unitary_ser, left_index=True, right_index=True)


pop_bkg = pop_psp.merge(pop_weights, on="population")


group_ave = pop_bkg.groupby("population").mean()["Unitary PSP"]
pop_bkg = pop_bkg.merge(
    group_ave, left_on="population", right_index=True, suffixes=["", "_pop"]
)

pop_bkg["PSP coef"] = pop_bkg["Unitary PSP_pop"] / pop_bkg["Unitary PSP"]
pop_bkg["syn_weight_psp"] = pop_bkg["syn_weight"] * pop_bkg["PSP coef"]

pop_bkg.index = pop_bkg["model_id"]
del pop_bkg["model_id"]

pop_bkg.to_csv("glif_props/bkg_weights_model.csv", sep=" ")


pop_bkg


# pop_df.loc[np.setdiff1d(pop_df.index, pop_bkg.index)]
