# %%
import pandas as pd
import json
import argparse
import utils


# parse args. available one is --double-alpha, which saves true.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Make a required node definition file for the GLIF model"
    )
    parser.add_argument(
        "-d",
        "--double-alpha",
        action="store_true",
        default=True,
        help="use nest glif model with double alpha synapses",
    )
    args = parser.parse_args()
    return args


def make_pop_model_dict(double_alpha=False):
    # make the pop_model_dict first.
    #
    v1_node_model = json.load(open("glif_props/v1_node_models.json"))
    # flatten down to population
    pop_model_dict = {}
    for loc, locdict in v1_node_model["locations"].items():
        for pop, popdict in locdict.items():
            for m in popdict["models"]:
                pop_model_dict[m["node_type_id"]] = pop

    # pop_models_name replaces the pop_name with syn_name.
    id_to_syn_type = {
        k: utils.pop_name_to_syn_type[v] for k, v in pop_model_dict.items()
    }
    syn_type_to_ids = {}
    types = ["e23", "e4", "e5et", "e5it", "e5np", "e6", "pv", "sst", "vip"]
    for t in types:
        syn_type_to_ids[t] = [k for k, v in id_to_syn_type.items() if v == t]

    v1unitary = json.load(open("precomputed_props/v1_synapse_amps.json", "r"))
    v1unitary_series = {}

    for t in types:
        if double_alpha:
            pre_type = "exc"
        else:
            pre_type = "lgn"

        type_unitary = pd.Series(v1unitary[f"{pre_type}_to_{t}"], name="Unitary PSP")
        type_unitary.index = pd.to_numeric(type_unitary.index)
        # limit to the ids that match postsynaptic type
        v1unitary_series[t] = type_unitary.loc[syn_type_to_ids[t]]

    v1unitary_ser = pd.concat(v1unitary_series.values())
    return v1unitary_ser, pop_model_dict


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


# %%
if __name__ == "__main__":
    args = parse_args()
    v1unitary_ser, pop_model_dict = make_pop_model_dict(double_alpha=args.double_alpha)
    # v1unitary_ser, pop_model_dict = make_pop_model_dict(True)

    basename = "base_props/lgn_weights_population.csv"
    targetname = "glif_props/lgn_weights_model.csv"

    pop_all = mix_in_population_weights(basename, pop_model_dict, v1unitary_ser)

    # do a bit of treatment of the highly active models.
    # these models are too active just with the LGN inputs, so reduce the weights by 10%
    # [model_id, reduction_factor]
    active_models = [
        [479179020, 0.9],
        [484679812, 0.9],
        [517647182, 0.9],
        [480169202, 0.8],
        [488689403, 0.8],
        [535728342, 0.5],
        [478793814, 0.7],
        [478958894, 0.9],
        [569997187, 0.7],
        [572375809, 0.8],
        [579414994, 0.6],
    ]
    # check if these are in the dataframe
    for m in active_models:
        if m[0] not in pop_all.index:
            print(f"model {m[0]} not in dataframe")
        else:  # take out syn_weight_psp by 20%
            pop_all.loc[m[0], "syn_weight_psp"] = (
                pop_all.loc[m[0], "syn_weight_psp"] * m[1]
            )

    pop_all.to_csv(targetname, sep=" ")

# %%
