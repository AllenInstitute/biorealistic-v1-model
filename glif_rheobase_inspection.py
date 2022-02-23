# %% doing additional analysis on rheobase


# get the rheobase from the cell-type database


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("cell_types/cells_with_glif_pop_name.csv", index_col=0)

# %% Now I'm ready to pick up the rheobase

sid = 5533  # put in a right number
df[df["specimen__id"] == sid]["ef__threshold_i_long_square"]

# %% get the cell's prpoerties from the model files


from pathlib import Path
import json


model_dir = Path("glif_models/cell_models")

paths = list(model_dir.glob("*.json"))
json.load(open(paths[0], "r"))

df_elems = []
int(paths[0].name[:9])


# %%

elements_to_pick = ["El_reference", "C", "th_inf", "R_input"]
# spec_ids = []

dics = []

for p in model_dir.glob("*.json"):
    model = json.load(open(p, "r"))
    # spec_ids.append(int(p.name[:9]))  # first 9 letters are the specimen ID
    dic = {e: model[e] for e in elements_to_pick}
    dic.update({"th_inf_coef": model["coeffs"]["th_inf"]})
    dic.update({"specimen__id": int(p.name[:9])})
    dics.append(dic)

# %%
model_df = pd.DataFrame(dics)

# %%
used_models = df.merge(model_df, on="specimen__id")
# used_models = used_models[used_models["pop_name"].str.contains("e4")]
# used_models = used_models[used_models["pop_name"].str.contains("e4Scnn")]
used_models

# %% plotting
used_models["model_rheo"] = (
    # (used_models["th_inf"] - used_models["El_reference"])
    used_models["th_inf"]
    * used_models["th_inf_coef"]
    / used_models["R_input"]
    * 1e12
)
used_models.plot(
    "ef__threshold_i_long_square", "El_reference", linestyle="", marker="."
)
used_models.plot("ef__threshold_i_long_square", "th_inf", linestyle="", marker=".")
used_models.plot("ef__threshold_i_long_square", "R_input", linestyle="", marker=".")

used_models.plot("ef__threshold_i_long_square", "model_rheo", linestyle="", marker=".")

used_models.plot("ef__ri", "R_input", linestyle="", marker=".")
used_models.plot("ef__vrest", "El_reference", linestyle="", marker=".")
used_models.plot("ef__threshold_v_long_square", "th_inf", linestyle="", marker=".")

used_models.keys()

used_models["th_inf"]
used_models["El_reference"]

# %%
a = used_models.plot(
    "ef__threshold_i_long_square", "model_rheo", linestyle="", marker="."
)
a.set_xlabel("Measured Rheobase")
a.set_ylabel("Model Rheobase")
a.axis("image")
a.set_xlim([0, 370])
a.set_ylim([0, 330])
a.legend().remove()
a.plot([0, 350], [0, 350])

# %% save the table
used_models.to_csv("misc_files/used_models.csv")


# %% testing lognormal
ln = np.random.lognormal(np.log(46), 0.2, 100000)

print(ln.mean())
print(ln.std())


# %%
