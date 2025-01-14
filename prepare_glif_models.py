# %%
# given the DataFrame that contains population information, copy glif3 models
#

import pandas as pd
import numpy as np
from allensdk.api.queries.glif_api import GlifApi
from shutil import copyfile
import os
import pathlib
import urllib.request
import zipfile
from tqdm import tqdm
from query_glif_expvar import safe_get_neuronal_models
import time


def get_glif3_model(id):
    # models = glif_api.get_neuronal_models(id)[0]["neuronal_models"]
    models = safe_get_neuronal_models(id)[0]["neuronal_models"]
    glif3id = int(np.where(["3 LIF" in m["name"] for m in models])[0])
    modelpath = models[glif3id]["well_known_files"][0]["path"]
    destination = f"glif_models/cell_models/{id}_glif_lif_asc_config.json"
    # if the model path is availabe, simply copy it to the destination
    if os.path.exists(modelpath):
        # if False:
        copyfile(modelpath, destination)
    else:  # otherwise, download the file from the http link.
        modelid = models[glif3id]["well_known_files"][0]["attachable_id"]
        url = f"http://api.brain-map.org/neuronal_model/download/{modelid}"

        # a layer of safety here as well.
        for i in range(10):
            try:
                urllib.request.urlretrieve(url, "./tmpfile.zip")
                with zipfile.ZipFile("./tmpfile.zip", "r") as zip_ref:
                    zip_ref.extract("neuron_config.json", "./")
                copyfile("./neuron_config.json", destination)
                # and remove the temporary file
                os.remove("./tmpfile.zip")
                os.remove("./neuron_config.json")
                break
            except:
                print(f"Failed to download {id} model, retrying, trial {i+1} of 10")
                time.sleep(0.5)


glif_api = GlifApi()
df = pd.read_csv("cell_types/cells_with_glif_pop_name.csv", index_col=0)

pathlib.Path("glif_models/cell_models").mkdir(parents=True, exist_ok=True)

print("Downloading GLIF3 models to glif_models/cell_models")
count = 0
for _, cell in tqdm(df.iterrows(), total=len(df)):
    if not pd.isna(cell.pop_name):
        id = cell.specimen__id
        get_glif3_model(id)
        count += 1
print(f"Downloaded {count} models")
