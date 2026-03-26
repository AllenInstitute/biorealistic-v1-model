# %%
from matplotlib import pyplot as plt
import plotting_utils as pu
from glob import glob


#
net = "small"
base_result_dir = f"{net}/output_spont_lgn_5s"

# check config*.json in all the subfolders of base_result_dir
config_files = glob(f"{base_result_dir}/*/config*.json")

# for each config file, plot the raster and save it.
for config_file in config_files:
    print(config_file)
    plt.figure(figsize=(15, 10))
    ax = pu.plot_raster(config_file, s=3, radius=100.0)
    ax.set_xlim([0, 2500])
    plt.tight_layout()
    plt.savefig(f"{config_file.replace('.json', '.png')}", dpi=300)
    plt.close()
