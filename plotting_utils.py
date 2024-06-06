# %% provides utilities for plotting a nice figures of simulation

import os
import numpy as np
import matplotlib.pyplot as plt
from sonata.circuit import File
import json
import pandas as pd
import re
import seaborn as sns
import pathlib


# in principle, if you provide the config file, you should be able to reproduce all the
# metadata


def pick_core(df, radius=400.0):
    """return if the neuron is at the core."""
    lateral = np.sqrt(df["x"] ** 2 + df["z"] ** 2)
    return df[lateral <= radius]


def read_config(config_file):
    # js = json.load(open(config_file, "r"))
    # let's close the file once opened...
    with open(config_file, "r") as f:
        js = json.load(f)
    return js


def form_network(config_file, infer=False):
    if infer:
        # in this case, config_js is the file pat, so trim the file name, and guess
        # the main directory name
        # pick up the top directory
        net_name = config_file.split("/")[0]
        print(f"inferring network from {net_name}")
        net = File(
            f"{net_name}/network/v1_nodes.h5", f"{net_name}/network/v1_node_types.csv"
        )
    else:
        # get the network structure out of the simulation
        config_js = read_config(config_file)
        node_files = [e["nodes_file"] for e in config_js["networks"]["nodes"]]
        node_type_files = [e["node_types_file"] for e in config_js["networks"]["nodes"]]
        net = File(node_files[0], node_type_files[0])
    return net


def get_spikes(config_file, infer=False):
    if infer:
        dir_name = str(pathlib.Path(config_file).parent)
        spike_file_name = f"{dir_name}/spikes.csv"
    else:
        config_js = read_config(config_file)
        spike_file_name = config_js["output"]["spikes_file_csv"]
    spike_df = pd.read_csv(spike_file_name, sep=" ", index_col=2)
    return spike_df


def identify_cell_type(pop_name: str):
    if pop_name.startswith("e"):
        return "Exc"
    else:
        # return the string after the first number
        return re.search(r"\d+(.*)", pop_name).groups()[0]


# this is destructive method (adds columns to v1df)
def determine_sort_position(v1df, sortby):
    if v1df["location"].iloc[0] == "Cortex":  # Old model
        layer = v1df["pop_name"].apply(lambda x: x[1])
        v1df["location"] = layer
    if sortby is not None:
        sorter = ["location", "Cell Type", sortby]
    else:
        sorter = ["location", "Cell Type"]
    reset_v1 = v1df.sort_values(sorter).reset_index()
    # reset_v1 = v1df.sort_values(["location", "Cell Type", "x"]).reset_index()
    # reset_v1 = v1df.sort_values(["location", "Cell Type"]).reset_index()
    sort_position = reset_v1.sort_values("index").index
    return sort_position


def determine_layer_divisions(v1df):
    """Given the dataframe, determine the layer divisions"""
    layers = ["", "L1", "L2/3", "L4", "L5", "L6"]
    divisions = list(np.cumsum(v1df.value_counts("location").sort_index()))
    divisions = [0] + divisions
    return dict(zip(layers, divisions))


def plot_raster(config_file, s=1, radius=400.0, sortby=None, infer=True, **kwarg):
    # try:
    #     config_js = read_config(config_file)
    #     net = form_network(config_js)
    #     spike_df = get_spikes(config_js)
    # except FileNotFoundError:
    #     # fall back to infer the network from the directory name.
    #     # <net>/network/ should contain the necessary node files.
    #     print("config file not found, inferring network from directory name.")
    #     net = form_network(config_file, infer=True)
    #     spike_df = get_spikes(config_file, infer=True)

    # defaulting to infer. because if I change the directory name, it tries to read from the new one.
    spike_df, hue_order, color_dict, layer_divisions = make_figure_elements(
        config_file, radius, sortby, infer
    )

    ax = sns.scatterplot(
        data=spike_df,
        x="timestamps",
        y="Sorted ID",
        hue="Cell Type",
        s=s,
        hue_order=hue_order,
        palette=color_dict,
        **kwarg,
    )
    # change the x label to Time (ms)
    ax.set_xlabel("Time (ms)")
    ax.invert_yaxis()
    for name, div in layer_divisions.items():
        ax.axhline(y=div, color="black", linestyle="-", linewidth=0.3)
        ax.text(
            0,
            div,
            name,
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize=9,
        )

    ax.legend(loc="upper right")

    return ax


def plot_fr_histogram(
    config_file,
    radius=400.0,
    sortby=None,
    infer=True,
    start_time=600,
    end_time=3000,
    ax=None,
    s=None,
):
    spike_df, hue_order, color_dict, layer_divisions = make_figure_elements(
        config_file, radius, sortby, infer
    )
    # make a histogram of the firing rates for all the neurons within the radius.
    # trim down spike_df with the start and end time
    spike_df = spike_df[
        (spike_df["timestamps"] >= start_time) & (spike_df["timestamps"] <= end_time)
    ]
    last_neuron_id = spike_df["Sorted ID"].max()
    if ax is None:
        # make a new figure.
        fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.histplot(
        ax=ax,
        data=spike_df,
        x="Sorted ID",
        hue="Cell Type",
        hue_order=hue_order,
        palette=color_dict,
        stat="count",
        bins=np.arange(0, last_neuron_id + 1, 1),
        weights=1000 / (end_time - start_time),  # convert to Hz
        kde=True,
    )
    ax.set_xlabel("Sorted ID")
    for name, div in layer_divisions.items():
        ax.axvline(x=div, color="black", linestyle="-", linewidth=0.3)
        ax.text(
            div,
            0,
            name,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=9,
        )
    ax.legend(loc="upper right")
    return ax


def make_figure_elements(config_file, radius, sortby, infer):
    net = form_network(config_file, infer=infer)
    spike_df = get_spikes(config_file, infer=infer)

    v1df = net.nodes["v1"].to_dataframe()
    v1df = pick_core(v1df, radius=radius)
    v1df["Cell Type"] = v1df["pop_name"].apply(identify_cell_type)
    v1df["Sort Position"] = determine_sort_position(v1df, sortby)

    spike_df = spike_df.loc[spike_df.index.isin(v1df.index)]

    spike_df["Sorted ID"] = v1df["Sort Position"].loc[spike_df.index]
    spike_df["Cell Type"] = v1df["Cell Type"].loc[spike_df.index]

    hue_order = ["Exc", "Pvalb", "Sst", "Vip", "Htr3a"]
    color_order = ["tab:red", "tab:blue", "tab:olive", "tab:purple", "tab:purple"]
    # color_order = ["tab:red", "tab:blue", "yellowgreen", "violet", "violet"]
    color_dict = dict(zip(hue_order, color_order))

    layer_divisions = determine_layer_divisions(v1df)
    return spike_df, hue_order, color_dict, layer_divisions


settings = {
    "full": {"radius": 400.0, "s": 0.5},
    "small": {"radius": 200.0, "s": 1},
    "small_0330": {"radius": 100.0, "s": 2},
    "small_0522": {"radius": 100.0, "s": 2},
    "small_0427": {"radius": 100.0, "s": 2},
    "small_0202": {"radius": 100.0, "s": 2},
    "core": {"radius": 200.0, "s": 1},
    "core_0312_24": {"radius": 200.0, "s": 1},
    "core_like05": {"radius": 200.0, "s": 1},
    "core_like02": {"radius": 200.0, "s": 1},
    "flat": {"radius": 850.0, "s": 1},
    "tensorflow": {"radius": 177.3, "s": 1},
    "tensorflow_new": {"radius": 400, "s": 1},
    # "tensorflow": {"radius": 400, "s": 1},
}
# %%time
if __name__ == "__main__":
    simple = True
    # net = "full"
    net = "core"
    # net = "small"
    # net = "tensorflow"
    # net = "tensorflow_new"

    sortby = "tuning_angle"
    # sortby = None  # model ID
    # sortby = "node_type_d"
    # sortby = "x"
    # sortby = "z"
    # sortby = "y"

    if simple:
        # config_file = "full/8dir_10trials/angle90_trial0/config_20.json"
        # config_file = f"{net}/8dir_10trials/angle0_trial0/config_0.json"
        # config_file = f"{net}/8dir_10trials/angle45_trial0/config_10.json"
        # config_file = f"{net}/8dir_10trials/angle90_trial0/config_20.json"
        # config_file = f"{net}/8dir_10trials/angle135_trial0/config_20.json"
        # config_file = (
        #     f"{net}/8dir_10trials_10k_rand_ckpt10/angle90_trial0/config_20.json"
        # )
        # config_file = f"../../glif_builder_test_0407/biorealistic-v1-model/{net}/8dir_10trials/angle90_trial0/config_20.json"
        # config_file = f"flat_0203/output_bkgtune/config_bkgtune.json"
        # config_file = f"{net}/output_bkgtune/config_bkgtune.json"
        # config_file = "small/8dir_10trials/angle0_trial0/config_0.json"
        config_file = f"{net}/output/config.json"
        # config_file = f"{net}/output/config_plain.json"
        # config_file = f"{net}/output_adjusted/config_plain.json"
        # config_file = f"{net}/output_2x/config_plain.json"
        # config_file = f"{net}/output_lgnbkg/config_lgnbkg.json"
        # config_file = f"{net}/output_lgn/config_lgn.json"
        # config_file = f"{net}/output_multimeter/config_multimeter.json"
        # config_file = (
        #     f"{net}/output_lgnbkg_borrow_filter/config_lgnbkg_borrow_filter.json"
        # )
        # config_file = f"{net}/output_bkg/config_bkg.json"
        # config_file = "full/output/config_plain.json"
        # config_file = "../../GLIF_network/output_lgnbkg/config_lgnbkg.json"
        # config_file = (
        #     "../../GLIF_network/output_lgn_os_prefix/config_lgn_os_prefix.json"
        # )
        # config_file = "../../GLIF_network/output_lgn_os_fixed/config_lgn_os_fixed.json"
        # config_file = f"{net}/output/angle0_trial0/configs.json"
        # config_file = f"{net}/output/ckpt-13/configs.json"
        # plt.figure(figsize=(15, 10))
        # plt.figure(figsize=(5, 3))
        plt.figure(figsize=(6, 3.5))
        # plt.figure(figsize=(10, 6))
        if net in settings:
            ax = plot_raster(config_file, sortby=sortby, infer=True, **settings[net])
            # ax = plot_fr_histogram(
            # config_file, sortby=sortby, infer=False, **settings[net]
            # )
        else:
            ax = plot_raster(config_file, sortby=sortby, s=1)
        # ax = plot_raster(config_file, s=1)  # for ful
        # ax.set_xlim([0, 1000])
        ax.set_xlim([0, 2500])
        # ax.set_ylim([0, 20])
        # ax.set_xlim([0, 1000])
        plt.tight_layout()
        # ax.legend_.texts[4].set_text("Inh. L1")
        # ax.legend_.text

        config_folder = os.path.dirname(config_file)
        plt.savefig(f"{config_folder}/raster_by_{sortby}.png", dpi=300)
        # plt.savefig(f"{config_folder}/hist_by_{sortby}.png", dpi=300)
    else:
        config_file = f"{net}/output_lgn/config_lgn.json"
        fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
        ax = plot_raster(config_file, ax=axs[0], **settings[net])
        ax.set_xlim([0, 1000])
        ax.legend(loc="upper right")
        ax.set_title("LGN only")

        config_file = f"{net}/output_lgnbkg/config_lgnbkg.json"
        ax = plot_raster(config_file, ax=axs[1], **settings[net])
        ax.legend(loc="upper right")
        ax.set_title("LGN + BKG connection")

        config_file = f"{net}/output/config.json"
        ax = plot_raster(config_file, ax=axs[2], **settings[net])
        ax.legend(loc="upper right")
        ax.set_title("LGN + BKG + recurrent connection")

        plt.tight_layout()
        plt.savefig("nice_ratser_reoptim.png", dpi=300)


# %% development block
# v1df.value_counts("location").sort_index()
