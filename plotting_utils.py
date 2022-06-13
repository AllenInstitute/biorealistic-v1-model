# %% provides utilities for plotting a nice figures of simulation

import numpy as np
import matplotlib.pyplot as plt
from sonata.circuit import File
import json
import pandas as pd
import re
import seaborn as sns


# in principle, if you provide the config file, you should be able to reproduce all the
# metadata


def read_config(config_file):
    js = json.load(open(config_file, "r"))
    return js


def form_network(config_js):
    # get the network structure out of the simulation
    node_files = [e["nodes_file"] for e in config_js["networks"]["nodes"]]
    # edge_files = [e["edges_file"] for e in config_js["networks"]["edges"]]
    node_type_files = [e["node_types_file"] for e in config_js["networks"]["nodes"]]
    # edge_type_files = [e["edge_types_file"] for e in config_js["networks"]["edges"]]
    net = File(node_files[0], node_type_files[0])
    return net


def get_spikes(config_js):
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
def determine_sort_position(v1df):
    reset_v1 = v1df.sort_values(["location", "pop_name", "tuning_angle"]).reset_index()
    sort_position = reset_v1.sort_values("index").index
    return sort_position


def determine_layer_divisions(v1df):
    """ Given the dataframe, determine the layer divisions """
    layers = ["", "L1", "L2/3", "L4", "L5", "L6"]
    divisions = list(np.cumsum(v1df.value_counts("location").sort_index()))
    divisions = [0] + divisions
    return dict(zip(layers, divisions))


def plot_raster(config_file, s=1, **kwarg):
    config_js = read_config(config_file)
    net = form_network(config_js)
    spike_df = get_spikes(config_js)

    v1df = net.nodes["v1"].to_dataframe()
    v1df["Sort Position"] = determine_sort_position(v1df)
    v1df["Cell Type"] = v1df["pop_name"].apply(identify_cell_type)

    layer_divisions = determine_layer_divisions(v1df)

    spike_df["Sorted ID"] = v1df["Sort Position"].loc[spike_df.index]
    spike_df["Cell Type"] = v1df["Cell Type"].loc[spike_df.index]

    hue_order = ["Exc", "Pvalb", "Sst", "Vip", "Htr3a"]
    color_order = ["tab:red", "tab:blue", "tab:olive", "tab:purple", "tab:purple"]
    # color_order = ["tab:red", "tab:blue", "yellowgreen", "violet", "violet"]
    color_dict = dict(zip(hue_order, color_order))

    ax = sns.scatterplot(
        data=spike_df,
        x="timestamps",
        y="Sorted ID",
        hue="Cell Type",
        s=s,
        hue_order=hue_order,
        palette=color_dict,
        **kwarg
    )
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


# %%
# %%time
if __name__ == "__main__":
    simple = False
    if simple:
        # config_file = "fullmodel56/output/config_plain.json"
        config_file = "miniature/output/config_plain.json"
        plt.figure(figsize=(15, 10))
        ax = plot_raster(config_file, s=1)
    else:
        config_file = "miniature/output/config_plain.json"
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        ax = plot_raster(config_file, ax=axs[0])
        ax.set_xlim([0, 1000])
        ax.legend(loc="upper right")
        ax.set_title("With Recurrent")

        config_file = "miniature/output_lgnbkg/config_lgnbkg.json"
        ax = plot_raster(config_file, ax=axs[1])
        ax.set_xlim([0, 1000])
        ax.legend(loc="upper right")
        ax.set_title("Without Recurrent")

        plt.tight_layout()
        plt.savefig("nice_ratser_reoptim.png")


# %% development block
v1df.value_counts("location").sort_index()

