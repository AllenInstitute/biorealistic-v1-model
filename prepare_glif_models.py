# %%
# given the DataFrame that contains population information, copy glif3 models
#

import pandas as pd
import numpy as np
from allensdk.api.queries.glif_api import GlifApi
from shutil import copyfile
import os

glif_api = GlifApi()

df = pd.read_csv("cell_types/cells_with_glif_pop_name.csv", index_col=0)


def get_glif3_model(id):
    models = glif_api.get_neuronal_models(id)[0]["neuronal_models"]
    glif3id = int(np.where(["3 LIF" in m["name"] for m in models])[0])
    modelpath = models[glif3id]["well_known_files"][0]["path"]
    destination = f"glif_models/cell_models/{id}_glif_lif_asc_config.json"
    copyfile(modelpath, destination)


if not os.path.exists("glif_models/cell_models"):
    if not os.path.exists("glif_models"):
        os.mkdir("glif_models")
    os.mkdir("glif_models/cell_models")


for _, cell in df.iterrows():
    if not pd.isna(cell.pop_name):
        id = cell.specimen__id
        get_glif3_model(id)

