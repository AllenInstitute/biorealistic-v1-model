# %% make one raster plot for the network

import plotting_utils as pu
import matplotlib.pyplot as plt
import argparse
import os


# net = "core"
# sortby = "tuning_angle"

parser = argparse.ArgumentParser(description="Make a raster plot of the network.")
parser.add_argument("outputdir", type=str, help="The network to plot.")
parser.add_argument(
    "--sortby",
    "-s",
    type=str,
    default="tuning_angle",
    help="The variable to sort by. Available options are 'tuning_angle', 'node_type_ids', 'x', 'y', etc...",
)
args = parser.parse_args()
# net = args.network
sortby = args.sortby


# config_file = f"{net}/output_plain/config.json"
config_file = f"{args.outputdir}/config.json"


net = os.path.split(args.outputdir)[0]

plt.figure(figsize=(10, 6))


ax = pu.plot_raster(config_file, sortby=sortby, infer=True, **pu.settings[net])
ax.set_xlim([0, 2500])
plt.tight_layout()
config_folder = os.path.dirname(config_file)
plt.savefig(f"{config_folder}/raster_by_{sortby}.png", dpi=300)
