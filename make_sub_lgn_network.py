# %% subsampled version of the LGN network.

import numpy as np
import h5py

# the structure of the lgn_nodes.h5 is as follows:
# /nodes/lgn/* variable group 1,
# /nodes/lgn/0/* variable group 2,

vg1 = ["node_group_id", "node_group_index", "node_id", "node_type_id"]

vg2 = [
    "delay_dom_0",
    "delay_dom_1",
    "delay_non_dom_0",
    "delay_non_dom_1",
    "kpeaks_dom_0",
    "kpeaks_dom_1",
    "kpeaks_non_dom_0",
    "kpeaks_non_dom_1",
    "sf_sep",
    "spatial_size",
    "tuning_angle",
    "weight_dom_0",
    "weight_dom_1",
    "weight_non_dom_0",
    "weight_non_dom_1",
    "x",
    "y",
]

# subnumber = 1
subrange = slice(1052, 1053)
# for all variables, pick up to subnumber for each variable.

orig_file_name = "tiny/network/lgn_nodes.h5"
sub_file_name = "tiny/network/lgn_nodes_sub.h5"

with h5py.File(orig_file_name, "r") as orig_file:
    with h5py.File(sub_file_name, "w") as sub_file:
        for var in vg1:
            vname = "/nodes/lgn/" + var
            data = np.array(orig_file[vname])
            # subdata = data[:subnumber]
            subdata = data[subrange]
            if var == "node_id" or var == "node_group_index":
                subdata = list(range(len(subdata)))
            sub_file.create_dataset(vname, data=subdata)
        for var in vg2:
            vname = "/nodes/lgn/0/" + var
            data = np.array(orig_file[vname])
            # subdata = data[:subnumber]
            subdata = data[subrange]
            sub_file.create_dataset(vname, data=subdata)

# this should be it.
