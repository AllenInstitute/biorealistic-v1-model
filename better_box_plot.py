# %% the goal of this script is to make a better box plot that compares with billeh

# What we need is cell types for each dataset, and OS/DS metrics for them.
import pandas as pd
import numpy as np
from sonata.circuit import File
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pathlib
from matplotlib.patches import Rectangle
from plotting_utils import pick_core


def pop_name_to_cell_type(pop_name):
    """convert pop_name in the old format to cell types.
    for example,
    'e4Rorb' -> 'L4 Exc'
    'i4Pvalb' -> 'L4 PV'
    'i23Sst' -> 'L2/3 SST'
    """
    shift = 0  # letter shift for L23
    layer = pop_name[1]
    if layer == "2":
        layer = "2/3"
        shift = 1
    elif layer == "1":
        return "L1 Htr3a"  # special case

    class_name = pop_name[2 + shift :]
    if class_name == "Pvalb":
        subclass = "PV"
    elif class_name == "Sst":
        subclass = "SST"
    elif (class_name == "Vip") or (class_name == "Htr3a"):
        subclass = "VIP"
    else:  # excitatory
        subclass = "Exc"

    return f"L{layer} {subclass}"


# basedir = 'small'
def get_osi_df(basedir, metric_file="OSI_DSI_DF.csv", metric_name="", radius=400.0):
    # special case is neurpixels experimental data
    if basedir == "neuropixels":
        df = pd.read_csv(f"{basedir}/metrics/{metric_file}", sep=" ")
    else:
        net = File(
            f"{basedir}/network/v1_nodes.h5", f"{basedir}/network/v1_node_types.csv"
        )
        v1df = net.nodes["v1"].to_dataframe()
        osi_df = pd.read_csv(f"{basedir}/metrics/{metric_file}", sep=" ")
        df = v1df.merge(osi_df)
        df["cell_type"] = df["pop_name"].apply(pop_name_to_cell_type)
        df = pick_core(df, radius=radius)

    df.rename(
        columns={"max_mean_rate(Hz)": "Rate at preferred direction (Hz)"}, inplace=True
    )

    # exclude OSI and DSI for neurons with <0.5 Hz preferred direction response
    nonresponding = df["Rate at preferred direction (Hz)"] < 0.5
    df.loc[nonresponding, "OSI"] = np.nan
    df.loc[nonresponding, "DSI"] = np.nan

    df = df.sort_values(by="cell_type")

    if len(metric_name) > 0:
        df["data_type"] = metric_name
    else:
        df["data_type"] = basedir
    return df


def get_borders(ticklabel):
    prev_layer = "1"
    borders = [-0.5]
    for i in ticklabel:
        x = i.get_position()[0]
        text = i.get_text()
        if text[1] != prev_layer:
            # detected layer change
            borders.append(x - 0.5)
            prev_layer = text[1]
    borders.append(x + 0.5)
    return borders


def draw_borders(ax, borders, ylim):
    for i in range(0, len(borders), 2):
        w = borders[i + 1] - borders[i]
        h = ylim[1] - ylim[0]
        ax.add_patch(
            Rectangle((borders[i], ylim[0]), w, h, alpha=0.08, color="k", zorder=-10)
        )
    return ax


def plot_one(ax, df, metric_name, ylim, cpal=None, e_only=False):
    if e_only:
        # first, filter out cell type == nan
        df = df[~df["cell_type"].isna()]
        df = df[df["cell_type"].str.contains("Exc")]
    sns.boxplot(
        x="cell_type",
        y=metric_name,
        hue="data_type",
        data=df,
        ax=ax,
        width=0.7,
        palette=cpal,
    )
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(loc="upper right")
    ax.set_ylim(ylim)
    ax.set_xlabel("")
    # apply shadings to each layer
    ticklabel = ax.get_xticklabels()
    borders = get_borders(ticklabel)
    draw_borders(ax, borders, ylim)
    return ax


def plot_scat(ax, df, x, y, cpal=None, s=1):
    sns.scatterplot(x=x, y=y, hue="data_type", data=df, ax=ax, s=s, palette=cpal)
    ax.legend(loc="upper right")
    # make x axis log scale
    ax.set_xscale("log")
    return ax


osi_dfs = []

color_pal = {
    "Billeh 2020, GLIF final": "tab:orange",
    "Neuropixels": "tab:gray",
    # "Neuropixels": "k",
    "core 9am": "tab:pink",
    "core bkg minus4": "tab:cyan",
    "core bkg minus6": "tab:pink",
    "Tensorflow": "tab:blue",
    "TF FR tuning": "tab:cyan",
    "TF default": "tab:blue",
    "Random weights": "tab:blue",
    "After 1000 iter.": "tab:pink",
    "After 7500 iter.": "tab:red",
    "TF rand ckpt5": "tab:pink",
    "TF rand ckpt10": "tab:red",
    "TF rand ckpt50": "tab:green",
    "TF rand ckpt75": "tab:blue",
    "Trained": "tab:red",
    "TF rand ckpt100": "tab:blue",
    "TF OS tuning": "tab:pink",
    "TF OS, core, 5": "tab:brown",
    "Lognorm, norecurrent": "tab:pink",
    "New model": "tab:blue",
    "small_new": "tab:blue",
}


# osi_dfs.append(get_osi_df("billeh", "OSI_DSI_DF.csv", "Billeh 2020, GLIF final"))
osi_dfs.append(get_osi_df("neuropixels", "OSI_DSI_DF.csv", "Neuropixels"))
# osi_dfs.append(
#     get_osi_df(
#         "core", "OSI_DSI_DF_recurrent_minus4.csv", "core bkg minus4", radius=200.0
#     )
# )
# osi_dfs.append(
#     get_osi_df(
#         "core", "OSI_DSI_DF_recurrent_minus6.csv", "core bkg minus6", radius=200.0
#     )
# )
# osi_dfs.append(get_osi_df("core", "OSI_DSI_DF_lgn.csv", "core lgn", radius=200.0))
# osi_dfs.append(get_osi_df("core", "OSI_DSI_DF_lgnbkg.csv", "core lgnbkg", radius=200.0))
# osi_dfs.append(get_osi_df("core", "OSI_DSI_DF.csv", "core new", radius=200.0))
# osi_dfs.append(
# get_osi_df("full", "OSI_DSI_DF_recurrent.csv", "full recurrent", radius=200.0)
# )
# osi_dfs.append(
# get_osi_df("tensorflow", "OSI_DSI_DF_before_tuning.csv", "TF default", radius=200.0)
# )
osi_dfs.append(
    get_osi_df(
        "small",
        "OSI_DSI_DF.csv",
        "small_new",
        radius=100.0,
    )
)
# osi_dfs.append(
#     get_osi_df(
#         "tensorflow",
#         "OSI_DSI_DF_10k_random_pre.csv",
#         "Random weights",
#         radius=100.0,
#     )
# )
# osi_dfs.append(
#     get_osi_df(
#         "tensorflow",
#         "OSI_DSI_DF_10k_random_ckpt10.csv",
#         "After 1000 iter.",
#         radius=100.0,
#     )
# )
# osi_dfs.append(
#     get_osi_df(
#         "tensorflow",
#         "OSI_DSI_DF_10k_random_ckpt50.csv",
#         "TF rand ckpt50",
#         radius=100.0,
#     )
# )
# osi_dfs.append(
#     get_osi_df(
#         "tensorflow",
#         "OSI_DSI_DF_10k_random_ckpt75.csv",
#         "After 7500 iter.",
#         radius=100.0,
#     )
# )
# osi_dfs.append(
#     get_osi_df(
#         "tensorflow",
#         "OSI_DSI_DF_10k_random_ckpt100.csv",
#         "TF rand ckpt100",
#         radius=100.0,
#     )
# )
# osi_dfs.append(
#     get_osi_df(
#         "tensorflow",
#         "OSI_DSI_DF_firing_rate_tuning_ckpt3.csv",
#         "TF FR tuning",
#         radius=200.0,
#     )
# )
# osi_dfs.append(
#     get_osi_df("tensorflow", "OSI_DSI_DF_os_tuning.csv", "TF OS tuning", radius=200.0)
# )
# osi_dfs.append(
#     get_osi_df(
#         "tensorflow", "OSI_DSI_DF_core_os_ckpt5.csv", "TF OS, core, 5", radius=400.0
#     )
# )
# osi_dfs.append(
#     get_osi_df(
#         "core", "OSI_DSI_DF_lgnbkg_minus4.csv", "core lgnbkg minus4", radius=200.0
#     )
# )
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF.csv", "full bad bkg", radius=400.0))
# osi_dfs.append(get_osi_df("core", "OSI_DSI_DF.csv", "core bad bkg", radius=200.0))

df = pd.concat(osi_dfs)

# for_sac = True
pattern = "scats"

# fig, axs = plt.subplots(4, 1, figsize=(12, 20))
if pattern == "sac":
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    plot_one(
        axs[0], df, "Rate at preferred direction (Hz)", [0, 50], color_pal, e_only=True
    )
    plot_one(axs[1], df, "OSI", [0, 1], color_pal, e_only=True)
    # remove the legend from axs[0], move the legend of axs[1] to outside the plot
    axs[0].get_legend().remove()
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig("box_simplified_SAC.png", dpi=150)
elif pattern == "normal":
    fig, axs = plt.subplots(4, 2, figsize=(24, 12))
    # fig, axs = plt.subplots(3, 1, figsize=(9, 12))
    plot_one(axs[0, 0], df, "Spont_Rate(Hz)", [0, 50], color_pal)
    plot_one(axs[0, 1], df, "Avg_Rate(Hz)", [0, 50], color_pal)
    plot_one(axs[1, 0], df, "Rate at preferred direction (Hz)", [0, 50], color_pal)
    plot_one(axs[2, 0], df, "DSI", [0, 1], color_pal)
    plot_one(axs[2, 1], df, "OSI", [0, 1], color_pal)
    plot_scat(axs[3, 0], df, "Rate at preferred direction (Hz)", "DSI", color_pal, s=5)
    plot_scat(axs[3, 1], df, "Rate at preferred direction (Hz)", "OSI", color_pal, s=5)

    plt.tight_layout()
    plt.savefig("box_CorrectRecurrent_Feb6_full.png", dpi=150)
    # plt.savefig("box_Dec12.svg", bbox="tight")
    # plt.savefig("box_Dec12.png", dpi=150)
elif pattern == "scats":
    # scatter plots of the rates.
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    plot_scat(axs[0, 0], df, "Spont_Rate(Hz)", "Avg_Rate(Hz)", color_pal, s=5)
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_aspect("equal")
    # equality line
    axs[0, 0].plot([0, 100], [0, 100], "k--")
    plot_scat(
        axs[0, 1],
        df,
        "Spont_Rate(Hz)",
        "Rate at preferred direction (Hz)",
        color_pal,
        s=5,
    )
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_aspect("equal")
    axs[0, 1].plot([0, 100], [0, 100], "k--")
    plt.tight_layout()


# %%
l4df = df.query("data_type == 'full round9' and location == 'VisL4'")
# print the following 4 items
print(l4df.groupby("cell_type").mean()["Spont_Rate(Hz)"])
print(l4df.groupby("cell_type").mean()["Avg_Rate(Hz)"])
print(l4df.groupby("ei").mean()["Spont_Rate(Hz)"])
print(l4df.groupby("ei").mean()["Avg_Rate(Hz)"])


# %% experimenting shading


# ax.fill_between(range(len(ticklabel)), 0, 1, shades, alpha=0.1, color='k', step='mid')

# ax.get_xlim(

a = df.query("data_type=='Neuropixels'")
# remove nans
# (a.query("cell_type == 'L6 Exc'")["Spont_Rate(Hz)"] == 0).sum()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# plot neuropixels scatter plot with different cell types highlighted.
sns.scatterplot(
    ax=ax, data=a, x="Rate at preferred direction (Hz)", y="OSI", s=5, hue="cell_type"
)
# set xscale log
ax.set_xscale("log")

# %%
