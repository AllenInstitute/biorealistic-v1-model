# %% script to make a v1 edge model.
import pandas as pd
import json

cell_type_df = pd.read_csv(
    "base_props/cell_type_naming_scheme.csv", sep=" ", index_col=0
)

# the granularity of the edge models will be:
# pre: cell_type
# post: cell_models

# %% load necessary tables.

# cell models
glif_models = pd.read_csv("glif_requisite/glif_models_prop.csv", sep=" ", index_col=0)

# psp to psc correction (scaling) factor
with open("precomputed_props/v1_synapse_amps.json", "r") as f:
    psc_psp = json.load(f)

# connection probabilities (sigma, pmax, b_ratio)
sigma = pd.read_csv("base_props/sigma.csv", sep=" ", index_col=0)
pmax = pd.read_csv("base_props/pmax_matrix_v1dd.csv", sep=" ", index_col=0)
b_ratio = pd.read_csv("base_props/b_ratio.csv", sep=" ", index_col=0)

# connection weights
lookup_table = pd.read_csv("base_props/psp_lookup_table.csv", sep=" ", index_col=0)
psp_df = pd.read_csv("base_props/psp_characterization.csv", index_col=0, sep=" ")
lookup_table

# connection latency
latency = pd.read_csv("base_props/syn_types_latency.csv", sep=" ", index_col=0)


# %% make the edge models
glif_models
cell_type_df.index


# initialize the connections dictionary
conn = {}
variable_names = [
    "target_model_id",
    "target_pop_name",
    "source_pop_name",
    "params_file",
    "PSP_scale_factor",
    "lognorm_shape",
    "lognorm_scale",
    "sigma",
    "b_ratio",
    "pmax",
    "delay",
]
for variable in variable_names:
    conn[variable] = []


for pre_pop in cell_type_df.index:  # source
    for model_id, model in glif_models.iterrows():  # target
        post_pop = model["pop_name"]

        # define cell types
        pre_ei = cell_type_df.loc[pre_pop, "ei"]
        post_ei = cell_type_df.loc[post_pop, "ei"]
        pre_type = cell_type_df.loc[pre_pop, "cell_type"]
        post_type = cell_type_df.loc[post_pop, "cell_type"]
        pre_type_synaptic = cell_type_df.loc[pre_pop, "cell_type_synaptic"]
        post_type_synaptic = cell_type_df.loc[post_pop, "cell_type_synaptic"]

        psp_name = lookup_table.loc[pre_pop, post_pop]
        syn_pair_name = f"{pre_type_synaptic}_to_{post_type_synaptic}"

        # set each value from the table.
        conn["target_model_id"].append(model_id)
        conn["target_pop_name"].append(post_pop)
        conn["source_pop_name"].append(pre_pop)
        conn["params_file"].append(f"{syn_pair_name}.json")

        conn["PSP_scale_factor"].append(psc_psp[syn_pair_name][str(model_id)])
        conn["lognorm_shape"].append(psp_df.loc[psp_name, "logn_shape_90th"])
        conn["lognorm_scale"].append(psp_df.loc[psp_name, "logn_scale_90th"])

        conn["sigma"].append(sigma.loc[pre_ei, post_ei])
        conn["b_ratio"].append(b_ratio.loc[pre_ei, post_ei])
        conn["pmax"].append(pmax.loc[pre_type, post_type])

        conn["delay"].append(latency.loc[pre_type_synaptic, post_type_synaptic])


conn_df = pd.DataFrame(conn)
conn_df.to_csv("glif_props/v1_edge_models.csv", sep=" ", index=False)
