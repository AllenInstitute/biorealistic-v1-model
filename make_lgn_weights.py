# %%
import pandas as pd
import json

v1unitary = json.load(open("base_props/v1_synapse_amps.json", "r"))
v1unitary_ser = pd.Series(v1unitary["e2e"], name="Unitary PSP")
v1unitary_ser_i = pd.Series(v1unitary["e2i"], name="Unitary PSP")
v1unitary_ser.index = pd.to_numeric(v1unitary_ser.index)
v1unitary_ser_i.index = pd.to_numeric(v1unitary_ser_i.index)
v1unitary_ser_both = v1unitary_ser.append(v1unitary_ser_i)

# also load up population weight
pop_weights = pd.read_csv("base_props/lgn_weights_population.csv", sep=" ")


# want to know which models are in which population
v1_node_model = json.load(open("glif_props/v1_node_models.json"))

# flatten down to population
pop_model_dict = {}
for loc, locdict in v1_node_model["locations"].items():
    for pop, popdict in locdict.items():
        for m in popdict["models"]:
            pop_model_dict[m["node_type_id"]] = pop


# %% make a dataframe that contains all these info
pop_df = pd.DataFrame.from_dict(pop_model_dict, orient="index", columns=["population"])
pop_df.index
pop_psp = pop_df.merge(v1unitary_ser_both, left_index=True, right_index=True)
pop_psp
pop_weights
pop_psp["model_id"] = pop_psp.index


pop_all = pop_psp.merge(pop_weights, on="population")

pop_all
group_ave = pop_all.groupby("population").mean()["Unitary PSP"]
pop_all = pop_all.merge(
    group_ave, left_on="population", right_index=True, suffixes=["", "_pop"]
)

pop_all["PSP coef"] = pop_all["Unitary PSP_pop"] / pop_all["Unitary PSP"]
pop_all["syn_weight_psp"] = pop_all["syn_weight"] * pop_all["PSP coef"]

pop_all.index = pop_all["model_id"]
del pop_all["model_id"]
pop_all.to_csv("glif_props/lgn_weights_model.csv", sep=" ")

