# %% make one raster plot for the network

import argparse

parser = argparse.ArgumentParser(description="Make a raster plot of the network.")
parser.add_argument(
    "outputdir",
    type=str,
    help="The output folder to plot that contains spikes file. e.g. core/output_adjusted.",
)
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


# delayed import for faster help response
import plotting_utils as pu
import matplotlib.pyplot as plt
import os
import pathlib

# net = "core"
# sortby = "tuning_angle"

# config_file = f"{net}/output_plain/config.json"
config_file = f"{args.outputdir}/config.json"


# net = os.path.split(args.outputdir)[0]
net = pathlib.Path(args.outputdir).parts[0]

plt.figure(figsize=(10, 6))


if "core" in net:
    net = "core"  # fall back not to cause error in setting keys.


ax = pu.plot_raster(config_file, sortby=sortby, infer=True, **pu.settings[net])
ax.set_xlim([0, 2500])
plt.tight_layout()
config_folder = os.path.dirname(config_file)
plt.savefig(f"{config_folder}/raster_by_{sortby}.png", dpi=300)
