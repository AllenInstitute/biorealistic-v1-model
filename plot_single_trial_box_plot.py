# %% plot single trial box plot... as the name suggests.

import plotting_utils as pu
import numpy as np
from utils import pop_name_to_cell_type as pnames

config_file = "core/output_adjusted/config_plain.json"

radius = 200
sortby = "tuning_angle"
infer = True

spike_df, hue_order, color_dict, layer_divisions = pu.make_figure_elements(
    config_file, radius, sortby, infer
)
net = pu.form_network(config_file)
v1df = net.nodes["v1"].to_dataframe()
v1df = pu.pick_core(v1df, radius=radius)

v1df["Cell Type"] = v1df["pop_name"].apply(lambda x: pnames[x])
v1df.value_counts("Cell Type")

spike_df
spike_df


np.digitize(spike_df["Sorted ID"].to_numpy(), list(layer_divisions.values()))

spike_df["Sorted ID"].to_numpy()


# count the number of spikes for each cell type!

# let's also get the node df
pu.get


# %% get the firing rates

spike_df.groupby(["Sorted ID"]).size()


# %%
layer_divisions.values()
