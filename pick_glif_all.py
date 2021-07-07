# %%
import json
import pandas as pd
import numpy as np
from allensdk.core.cell_types_cache import CellTypesCache

ctc = CellTypesCache()


with open("cell_types/cells.json", "r") as f:
    celljson = json.load(f)
df = pd.DataFrame(celljson)
df.keys()

v1cells = df.structure__name.str.contains("Primary visual area")
candidates = (df.donor__species == "Mus musculus") & (df.m__glif >= 2) & v1cells
# candidates = (df.donor__species == "Mus musculus") & (df.m__biophys_perisomatic >= 1) & v1cells
print(f"{candidates.sum()} mouse V1 cells with 2 or more GLIF models.")
df = df[candidates]

# mix in explained variance ratio, and type database
glif_df = pd.read_csv("glif_explained_variance_ratio.csv", index_col=0)
df = df.join(glif_df, on="specimen__id")

type_df = pd.read_excel(
    "41593_2019_417_MOESM5_ESM.xlsx", engine="openpyxl", index_col=0
)
df = df.join(type_df, on="specimen__id")
print("GLIF explained variance ratio and ME types mixed in")

# remove cells in the exclude_list
excl_df = pd.read_csv("exclude_list.csv", index_col=0)
df = df[~df["specimen__id"].isin(excl_df.index)]

print(f"{len(df)} cells are available after excluding specific cells from a list.")


# %% now I set criteria for each population (read from seed file)
import pylightxl as xl


db = xl.readxl("../network_builder/V1model_seed_file.xlsx")
table = db.ws(db.ws_names[0]).ssd(keycols="pop_id", keyrows="pop_id")
t0 = table[0]
seed_df = pd.DataFrame(data=t0["data"], index=t0["keyrows"], columns=t0["keycols"])


# %% Let's see if I can get any cells using these criteria
evr_thresh = 0.7
tot_candidates = 0
candidate_dict = {}

df["pop_name"] = None

# pop = seed_df.iloc[0]
for _, pop in seed_df.iterrows():

    # Layer name can be uniquely extracted from the last letter of location
    pop_layer = pop.location[-1]
    cre_line = pop.cre_line

    df.keys()
    # Layer name can be uniquely extracted from the last letter of location
    layer_good = df.structure__layer.str.contains(pop.location[-1])
    lines = pop.cre_line.split(",")
    line_good = np.any([df.line_name.str.contains(s) for s in lines], axis=0)
    # line_good = df.line_name.str.contains(pop.cre_line)
    cre_good = df.cell_reporter_status.str.contains(pop.reporter_status)
    evr_good = df["explained variance ratio"] > evr_thresh

    me_types = pop.me_type.split(",")
    if len(me_types[0]) > 0:  # non empty string
        if me_types[0][0] == "M":  # works for IT, ET cells
            me_good = np.any([df["me-type"] == s for s in me_types], axis=0)
        elif me_types[0][0] == "E":  # so that it works for NP cells
            me_good = np.any([df["e-type"] == s for s in me_types], axis=0)
            # ignore ones with ME type defined.
            me_good &= pd.isna(df["me-type"])
    else:  # if not specified, pass all
        me_good = True

    matched_cells = layer_good & line_good & cre_good & evr_good & me_good
    n_candidates = matched_cells.sum()
    tot_candidates += n_candidates

    df["pop_name"][matched_cells] = pop.pop_name

    candidate_dict[pop.pop_name] = df[matched_cells]

    print(f"{pop.pop_name}: {n_candidates}")

print(f"Total: {tot_candidates}")

# %% generate URL list

urlbase = "http://celltypes.brain-map.org/experiment/electrophysiology/"

for pop_name, pop_df in candidate_dict.items():
    print(f"Population: {pop_name}")
    for _, cell in pop_df.iterrows():
        print(f"{urlbase}{cell.specimen__id}")
    print("")


# %% put it in context
h = df.hist(column="explained variance ratio", bins=np.linspace(0, 1, 20))
h[0, 0].plot([evr_thresh, evr_thresh], [0, 100], "r")


# %% save the IDs into a file

# let's just save the entire labeld df as a file.
df.to_csv("cell_types/cells_with_glif_pop_name.csv")
