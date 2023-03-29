# %%
import pandas as pd
import json


def make_pop_model_dict():
    types = ["e23", "e4", "e5et", "e5it", "e5np", "e6", "pv", "sst", "vip"]

    v1unitary = json.load(open("precomputed_props/v1_synapse_amps.json", "r"))
    v1unitary_series = {}

    for t in types:
        v1unitary_series[t] = pd.Series(v1unitary["lgn_to_" + t], name="Unitary PSP")
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

basename = "base_props/lgn_weights_population.csv"
targetname = "glif_props/lgn_weights_model.csv"


def mix_in_population_weights(base_name, pop_model_dict, v1unitary_ser):
    pop_df = pd.DataFrame.from_dict(
        pop_model_dict, orient="index", columns=["population"]
    )
    pop_psp = pop_df.merge(v1unitary_ser, left_index=True, right_index=True)
    pop_psp["model_id"] = pop_psp.index

    pop_weights = pd.read_csv(base_name, sep=" ")
    pop_all = pop_psp.merge(pop_weights, on="population")

    # just devide by the unitary PSP
    pop_all["syn_weight_psp"] = pop_all["syn_weight"] / pop_all["Unitary PSP"]

    pop_all.index = pop_all["model_id"]
    del pop_all["model_id"]
    return pop_all


pop_all = mix_in_population_weights(basename, pop_model_dict, v1unitary_ser)

# do a bit of treatment of the highly active models.
# these models are too active just with the LGN inputs, so reduce the weights by 10%
active_models = [
    479179020,
    484679812,
    480169202,
    488689403,
    535728342,
    478793814,
    569997187,
    579414994,
]
# check if these are in the dataframe
for m in active_models:
    if m not in pop_all.index:
        print(f"model {m} not in dataframe")
    else:  # take out syn_weight_psp by 20%
        pop_all.loc[m, "syn_weight_psp"] = pop_all.loc[m, "syn_weight_psp"] * 0.8


pop_all.to_csv(targetname, sep=" ")
# %%

pop_all
