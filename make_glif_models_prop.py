# %%
# given the datafrom that contains population information, create property files
import pandas as pd
import os

if not os.path.exists("glif_requisite"):
    os.mkdir("glif_requisite")


df = pd.read_csv("cell_types/cells_with_glif_pop_name.csv", index_col=0)

df_used = df[pd.notna(df.pop_name)]

df_store = pd.DataFrame(index=df_used.specimen__id)
df_store["level_of_details"] = "GLIF3"
df_store["pop_name"] = list(df_used.pop_name)

ids = list(df_used.specimen__id)

df_store["parameters_file"] = [str(i) + "_glif_lif_asc_config.json" for i in ids]

df_sorted = df_store.sort_values(["pop_name", "specimen__id"])

df_sorted.to_csv("glif_requisite/glif_models_prop.csv", sep=" ")

