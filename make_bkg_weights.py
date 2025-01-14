# read bkg_weights_population.csv and write bkg_weights_model.csv
# %%
import pandas as pd
from make_lgn_weights import make_pop_model_dict, mix_in_population_weights
import sys


v1unitary_ser, pop_model_dict = make_pop_model_dict(double_alpha=True)

# this is the default basename
basename = "base_props/bkg_weights_population_init.csv"
if sys.argv[0] == "make_bkg_weights.py":  # this script is run from command line
    # check the additional argument and use it for the basename
    # print('ran from command line')
    if len(sys.argv) > 1:
        # print('basename is now: ' + sys.argv[1])
        basename = sys.argv[1]

targetname = "glif_props/bkg_weights_model.csv"

pop_all = mix_in_population_weights(basename, pop_model_dict, v1unitary_ser)
pop_all.to_csv(targetname, sep=" ")

# pop_df = pd.DataFrame.from_dict(pop_model_dict, orient="index", columns=["population"])
# pop_df["model_id"] = pop_df.index
# pop_psp = pop_df.merge(v1unitary_ser, left_index=True, right_index=True)


# If you want to start from scratch, please load this version.
# pop_weights = pd.read_csv("base_props/bkg_weights_population_init.csv", sep=" ")

# We have optimized the background connections. Load that version instead.
# pop_weights = pd.read_csv("precomputed_props/bkg_weights_population_init.csv", sep=" ")
# pop_bkg = pop_psp.merge(pop_weights, on="population")


# group_ave = pop_bkg.groupby("population").mean()["Unitary PSP"]
# pop_bkg = pop_bkg.merge(
# group_ave, left_on="population", right_index=True, suffixes=["", "_pop"]
# )

# pop_bkg["PSP coef"] = pop_bkg["Unitary PSP_pop"] / pop_bkg["Unitary PSP"]
# pop_bkg["syn_weight_psp"] = pop_bkg["syn_weight"] * pop_bkg["PSP coef"]

# pop_bkg['syn_weight_psp'] = pop_bkg['syn_weight'] / pop_bkg['Unitary PSP']

# pop_bkg.index = pop_bkg["model_id"]
# del pop_bkg["model_id"]

# pop_bkg.to_csv("glif_props/bkg_weights_model.csv", sep=" ")


# pop_df.loc[np.setdiff1d(pop_df.index, pop_bkg.index)]
