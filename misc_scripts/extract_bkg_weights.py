# %%
# This script extract weights (per population) from the previous model
import pandas as pd

bkg_edge_df = pd.read_csv(
    "../../network_builder/glif_props/bkg_v1_edge_types.csv", sep=" "
)
len(bkg_edge_df.syn_weight.unique())
bkg_edge_df["target_model"] = (
    bkg_edge_df["target_query"].str.extract(r"'([0-9]+?)'").astype(int)
)


v1_node_df = pd.read_json("../../network_builder/glif_props/v1_node_models.json")
v1_node_df["locations"]["VisL4"].keys()


# %% LGN part
lgn_edge_df = pd.read_csv(
    "../../network_builder/glif_props/lgn_v1_edge_types.csv", sep=" "
)
len(lgn_edge_df.syn_weight.unique())
lgn_edge_df["target_model"] = (
    lgn_edge_df["target_query"].str.extract(r"'([0-9]+?)'").astype(int)
)


# %% V1 part

# flatten the node df
v1l = v1_node_df["locations"]

model_dict = {}

for l in v1l:
    populations = list(l.keys())
    for p in populations:
        models = l[p]["models"]
        for m in models:
            model_dict[m["node_type_id"]] = p


# print(model_dict)
model_pop = pd.DataFrame.from_dict(model_dict, orient="index", columns=["population"])

# bkg_edge_df['target_model'] = bkg_edge_df['target_query'].


# %% merging part
merged_bkg = bkg_edge_df.merge(model_pop, left_on="target_model", right_index=True)
simple_bkg = merged_bkg[["population", "syn_weight"]].drop_duplicates()
simple_bkg.to_csv("bkg_weights_population.csv", index=False)

# %% merging for lgn
merged_lgn = lgn_edge_df.merge(model_pop, left_on="target_model", right_index=True)
simple_lgn = merged_lgn.groupby("population").mean()[["syn_weight"]]
# merged_lgn.groupby('population').std()
# simple_lgn = merged_lgn[["population", "syn_weight"]].drop_duplicates()
simple_lgn.to_csv("lgn_weights_population.csv")


# note that this does not care about changes of e5 populations.
# The generated files should be manually changed to include e5IT, e5ET, and e5NP
# (I used the average values of old e5 to fill them in.)
