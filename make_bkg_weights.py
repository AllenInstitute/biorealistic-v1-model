# read bkg_weights_population.csv and write bkg_weights_model.csv
# %%
import pandas as pd
from make_lgn_weights import make_pop_model_dict

v1unitary_ser, pop_model_dict = make_pop_model_dict()

# %%

pop_df = pd.DataFrame.from_dict(pop_model_dict, orient="index", columns=["population"])
pop_df["model_id"] = pop_df.index
pop_psp = pop_df.merge(v1unitary_ser, left_index=True, right_index=True)


pop_weights = pd.read_csv("base_props/bkg_weights_population_init.csv", sep=" ")
pop_bkg = pop_psp.merge(pop_weights, on="population")


# group_ave = pop_bkg.groupby("population").mean()["Unitary PSP"]
# pop_bkg = pop_bkg.merge(
    # group_ave, left_on="population", right_index=True, suffixes=["", "_pop"]
# )

# pop_bkg["PSP coef"] = pop_bkg["Unitary PSP_pop"] / pop_bkg["Unitary PSP"]
# pop_bkg["syn_weight_psp"] = pop_bkg["syn_weight"] * pop_bkg["PSP coef"]

pop_bkg['syn_weight_psp'] = pop_bkg['syn_weight'] / pop_bkg['Unitary PSP']

pop_bkg.index = pop_bkg["model_id"]
del pop_bkg["model_id"]

pop_bkg.to_csv("glif_props/bkg_weights_model.csv", sep=" ")


pop_bkg


# pop_df.loc[np.setdiff1d(pop_df.index, pop_bkg.index)]
