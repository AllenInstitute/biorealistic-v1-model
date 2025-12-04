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


def plot_raster(
    config_file,
    s=1,
    radius=400.0,
    sortby=None,
    infer=True,
    grouping="full",
    legend_markerscale=None,
    layer_label_fontsize=None,
    **kwarg,
):
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
        config_file, radius, sortby, infer, grouping
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
    # remove y label since we now show layer ticks with counts
    ax.set_ylabel("")
    ax.invert_yaxis()
    # Determine fontsize for layer labels
    _lbl_fs = layer_label_fontsize if layer_label_fontsize is not None else plt.rcParams.get("font.size", 10)
    # Draw layer division lines
    for name, div in layer_divisions.items():
        ax.axhline(y=div, color="black", linestyle="-", linewidth=0.3)

    # Build y-ticks and labels with counts per layer, aligned at the division lines
    layers_order = [k for k in layer_divisions.keys() if k != ""]
    ticks, labels = [], []
    prev = layer_divisions.get("", 0)
    for name in layers_order:
        this = layer_divisions[name]
        count = max(int(this - prev), 0)
        ticks.append(this)
        # labels.append(f"{name}\n(n={count})")
        labels.append(f"{name}")
        prev = this

    if ticks:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        for t in ax.get_yticklabels():
            t.set_verticalalignment("bottom")
            t.set_fontsize(_lbl_fs)

    # Add total neuron count at top-left, slightly above plotting area
    if layer_divisions:
        try:
            total_n = max(layer_divisions.values())
        except Exception:
            total_n = None
        if total_n is not None and total_n > 0:
            annot_fs = max(int(_lbl_fs * 0.8), 8)
            ax.text(
                0.0,
                1.02,
                f"N={total_n}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=annot_fs,
            )

    if legend_markerscale is not None:
        leg = ax.legend(loc="upper right", scatterpoints=1, markerscale=legend_markerscale)
    else:
        leg = ax.legend(loc="upper right")
    # Ensure legend text sizes are consistent with base font
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(_lbl_fs)
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(_lbl_fs)

    # Ensure tick and axis label sizes match the base font
    ax.tick_params(axis="both", labelsize=_lbl_fs)
    ax.xaxis.label.set_size(_lbl_fs)
    ax.yaxis.label.set_size(_lbl_fs)

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


def make_figure_elements(config_file, radius, sortby, infer, grouping="full"):
    net = form_network(config_file, infer=infer)
    spike_df = get_spikes(config_file, infer=infer)

    v1df = net.nodes["v1"].to_dataframe()
    if radius is not None:
        v1df = pick_core(v1df, radius=radius)
    # Map each pop_name to its canonical cell_type using the naming scheme CSV
    import network_utils as nu
    ctdf = nu.get_cell_type_table()  # index = pop_name
    pop_to_celltype = ctdf["cell_type"]
    pop_to_hex = ctdf[["cell_type", "hex"]].drop_duplicates().set_index("cell_type")["hex"]

    # Use the mapping (fall back to generic categories if missing)
    v1df["Cell Type"] = v1df["pop_name"].map(pop_to_celltype)
    missing_mask = v1df["Cell Type"].isna()
    if missing_mask.any():
        # fallback to previous identify_cell_type for unseen pop_names
        v1df.loc[missing_mask, "Cell Type"] = v1df.loc[missing_mask, "pop_name"].apply(identify_cell_type)

    # If simplified grouping is requested, map to four categories before sorting
    if grouping == "four":
        # Determine layer number from pop_name to find L1 inhibitory
        layer_num = v1df["pop_name"].str.extract(r"[ei](\d)").astype(int)[0]
        base_labels = v1df["Cell Type"].astype(str)

        def map_to_group(cell_type: str) -> str:
            # Merge L5 IT/ET/NP into Exc
            if cell_type in ("L5_IT", "L5_ET", "L5_NP"):
                return "Exc"
            if cell_type.endswith("_Exc") or cell_type == "Exc":
                return "Exc"
            if cell_type.endswith("_PV") or cell_type in ("Pvalb", "PV"):
                return "PV"
            if cell_type.endswith("_SST") or cell_type in ("Sst", "SST"):
                return "SST"
            if cell_type.endswith("_VIP") or cell_type in ("Vip", "Htr3a", "VIP"):
                return "VIP"
            return cell_type

        grouped = base_labels.map(map_to_group)
        grouped.loc[layer_num.eq(1) & ~base_labels.str.contains("Exc")] = "L1"

        # Overwrite Cell Type for sorting and downstream labeling
        v1df["Cell Type"] = grouped

    v1df["Sort Position"] = determine_sort_position(v1df, sortby)

    spike_df = spike_df.loc[spike_df.index.isin(v1df.index)]

    spike_df["Sorted ID"] = v1df["Sort Position"].loc[spike_df.index]
    spike_df["Cell Type"] = v1df["Cell Type"].loc[spike_df.index]

    # ---------------------------------------------
    # Build palettes and labels
    # ---------------------------------------------
    if grouping == "four":
        # Colors: use L2/3 colours for all layers
        l23_keys = {
            "Exc": "L2/3_Exc",
            "PV": "L2/3_PV",
            "SST": "L2/3_SST",
            "VIP": "L2/3_VIP",
        }
        color_dict = {}
        for k, canonical in l23_keys.items():
            if canonical in pop_to_hex.index:
                color_dict[k] = pop_to_hex.loc[canonical]
        # L1 in neutral gray for visibility but equal brightness feel
        color_dict["L1"] = "#9E9E9E"

        # Keep order concise, put L1 first
        order_pref = ["L1", "Exc", "PV", "SST", "VIP"]
        present = list(spike_df["Cell Type"].dropna().unique())
        hue_order = [k for k in order_pref if k in present]
    else:
        # Full palette: use CSV colours directly; add fallbacks for generic labels
        color_dict = pop_to_hex.to_dict()
        fallback_palette = {
            'Exc': '#D61515',
            'PV' : '#157C0C',
            'SST': '#0C7979',
            'VIP': '#8F40DF',
        }
        color_dict.update({
            "Exc": fallback_palette["Exc"],
            "Pvalb": fallback_palette["PV"],
            "Sst": fallback_palette["SST"],
            "Vip": fallback_palette["VIP"],
            "Htr3a": fallback_palette["VIP"],
        })
        hue_order = list(color_dict.keys())

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

for i in range(10):
    settings[f"core_{i}"] = {"radius": 200.0, "s": 1}
# %%time
if __name__ == "__main__":
    simple = True
    # net = "full"
    # net = "core"
    net = "small"
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
        # config_file = f"{net}/output/config.json"
        # config_file = f"{net}/output_plain/config_plain.json"
        config_file = f"{net}/output_adjusted/config_adjusted.json"
        # config_file = f"{net}/output_checkpoint/config_checkpoint.json"
        # config_file = f"{net}/output_checkpoint_1k/config_checkpoint_1k.json"
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
        # plt.figure(figsize=(6, 3.5))
        plt.figure(figsize=(10, 6))
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
