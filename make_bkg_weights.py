# read bkg_weights_population.csv and write bkg_weights_model.csv
# %%
import pandas as pd
import json
import numpy as np


v1unitary = json.load(open("base_props/OLD_v1_synapse_amps.json", "r"))
v1unitary_ser = pd.Series(v1unitary["e2e"], name="Unitary PSP")
v1unitary_ser_i = pd.Series(v1unitary["e2i"], name="Unitary PSP")
v1unitary_ser.index = pd.to_numeric(v1unitary_ser.index)
v1unitary_ser_i.index = pd.to_numeric(v1unitary_ser_i.index)
v1unitary_ser_both = v1unitary_ser.append(v1unitary_ser_i)

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
pop_psp = pop_df.merge(v1unitary_ser_both, left_index=True, right_index=True)


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
