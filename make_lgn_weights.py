# %%
import pandas as pd
import json

def make_pop_model_dict():
    types = ['e23', 'e4', 'e5et', 'e5it', 'e5np', 'e6', 'pv', 'sst', 'vip']

    v1unitary = json.load(open("precomputed_props/v1_synapse_amps.json", "r"))
    v1unitary_series = {}

    for t in types:
        v1unitary_series[t] = pd.Series(v1unitary['lgn_to_' + t], name="Unitary PSP")
        v1unitary_series[t].index = pd.to_numeric(v1unitary_series[t].index)

    v1unitary_ser = pd.concat(v1unitary_series.values())
    v1_node_model = json.load(open("glif_props/v1_node_models.json"))

    # flatten down to population
    pop_model_dict = {}
    for loc, locdict in v1_node_model["locations"].items():
        for pop, popdict in locdict.items():
            for m in popdict["models"]:
                pop_model_dict[m["node_type_id"]] = pop
    return v1unitary_ser, pop_model_dict

v1unitary_ser, pop_model_dict = make_pop_model_dict()


# %% make a dataframe that contains all these info
pop_df = pd.DataFrame.from_dict(pop_model_dict, orient="index", columns=["population"])
pop_psp = pop_df.merge(v1unitary_ser, left_index=True, right_index=True)
pop_psp["model_id"] = pop_psp.index


# load population weight and merge it
pop_weights = pd.read_csv("base_props/lgn_weights_population.csv", sep=" ")
pop_all = pop_psp.merge(pop_weights, on="population")

# pop_all
# group_ave = pop_all.groupby("population").mean()["Unitary PSP"]
# pop_all = pop_all.merge(
#     group_ave, left_on="population", right_index=True, suffixes=["", "_pop"]
# )

# normalize the coefficient per population, so that we can use the same weight from
# the old models
# pop_all["PSP coef"] = pop_all["Unitary PSP_pop"] / pop_all["Unitary PSP"]
# pop_all["syn_weight_psp"] = pop_all["syn_weight"] * pop_all["PSP coef"]

# revert it. Just divide by the unitary PSP to get the final value
pop_all["syn_weight_psp"] = pop_all["syn_weight"] / pop_all["Unitary PSP"]

pop_all.index = pop_all["model_id"]
del pop_all["model_id"]
pop_all.to_csv("glif_props/lgn_weights_model.csv", sep=" ")
