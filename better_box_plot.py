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
import utils


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
        # df["cell_type"] = df["pop_name"].map(utils.cell_type_df["cell_type"])
        df["cell_type"] = df["pop_name"].map(utils.cell_type_df["cell_type_old"])
        df["ei"] = df["pop_name"].map(utils.cell_type_df["ei"])
        df = pick_core(df, radius=radius)

    df.rename(
        columns={"max_mean_rate(Hz)": "Rate at preferred direction (Hz)"}, inplace=True
    )

    # replace white spaces in cell_type with underscore
    df["cell_type"] = df["cell_type"].str.replace(" ", "_")
    # also L1_Htr3a should be L1_Inh
    df["cell_type"] = df["cell_type"].str.replace("L1_Htr3a", "L1_Inh")

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
    for i in range(0, len(borders) - 1, 2):
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
        # df = df[df["cell_type"].str.contains("Exc")]
        df = df[df["ei"] == "e"]
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


def plot_normal(df, cpal):
    fig, axs = plt.subplots(4, 2, figsize=(24, 12))
    plot_one(axs[0, 0], df, "Spont_Rate(Hz)", [0, 50], cpal)
    plot_one(axs[0, 1], df, "Avg_Rate(Hz)", [0, 50], cpal)
    plot_one(axs[1, 0], df, "Rate at preferred direction (Hz)", [0, 50], cpal)
    plot_one(axs[2, 0], df, "DSI", [0, 1], cpal)
    plot_one(axs[2, 1], df, "OSI", [0, 1], cpal)
    plot_scat(axs[3, 0], df, "Rate at preferred direction (Hz)", "DSI", cpal, s=5)
    plot_scat(axs[3, 1], df, "Rate at preferred direction (Hz)", "OSI", cpal, s=5)
    plt.tight_layout()
    return fig, axs


def plot_sac(df, cpal):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    # plot_one(axs[0], df, "Rate at preferred direction (Hz)", [0, 50], cpal, e_only=True)
    plot_one(axs[0], df, "Avg_Rate(Hz)", [0, 25], cpal, e_only=True)
    plot_one(axs[1], df, "OSI", [0, 1], cpal, e_only=True)
    # remove the legend from axs[0], move the legend of axs[1] to outside the plot
    axs[0].get_legend().remove()
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()
    return fig, axs


osi_dfs = []

color_pal = {
    "Billeh 2020, GLIF final": "tab:orange",
    "Neuropixels data": "tab:gray",
    "Neuropixels v3": "tab:gray",
    # "Neuropixels": "k",
    "core 9am": "tab:pink",
    "core bkg minus4": "tab:cyan",
    "core bkg minus6": "tab:pink",
    "full recurrent": "tab:blue",
    "core recurrent": "tab:pink",
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
    "New model": "tab:orange",
    "small_new": "tab:blue",
    "core new": "tab:red",
    "core default": "tab:blue",
    "core adjusted": "tab:blue",
    "core trained": "tab:red",
    "small_before_tuning": "tab:blue",
    "small_after_tuning": "tab:pink",
    "Baseline Model": "tab:orange",
    "Rate Adjusted Model": "tab:orange",
    "Optimized Model": "tab:blue",
}


# osi_dfs.append(get_osi_df("billeh", "OSI_DSI_DF.csv", "Billeh 2020, GLIF final"))
osi_dfs.append(get_osi_df("neuropixels", "OSI_DSI_DF_data.csv", "Neuropixels data"))
# osi_dfs.append(get_osi_df("core", "OSI_DSI_DF_orig.csv", "core adjusted", radius=200.0))
osi_dfs.append(
    get_osi_df("core_1", "OSI_DSI_DF_adjusted.csv", "Rate Adjusted Model", radius=200.0)
)
osi_dfs.append(
    get_osi_df("core", "OSI_DSI_DF_checkpoint.csv", "Optimized Model", radius=200.0)
)
# osi_dfs.append(
#     get_osi_df("core_0427", "OSI_DSI_DF_recurrent.csv", "core recurrent", radius=200.0)
# )
# osi_dfs.append(get_osi_df("core", "OSI_DSI_DF.csv", "core new", radius=200.0))
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF.csv", "full recurrent", radius=400.0))
df = pd.concat(osi_dfs).reset_index()

pattern = "normal"
# pattern = "sac"

# fig, axs = plt.subplots(4, 1, figsize=(12, 20))
if pattern == "sac":
    fig, axs = plot_sac(df, color_pal)
    # fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    # plot_one(
    #     axs[0], df, "Rate at preferred direction (Hz)", [0, 50], color_pal, e_only=True
    # )
    # plot_one(axs[1], df, "OSI", [0, 1], color_pal, e_only=True)
    # # remove the legend from axs[0], move the legend of axs[1] to outside the plot
    # axs[0].get_legend().remove()
    # axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig("box_simplified_SAC.png", dpi=150)
elif pattern == "normal":
    # color_pal = None
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
    plt.savefig("core/figures/box_plain.png", dpi=150)
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
    plot_scat(axs[1, 0], df, "Spont_Rate(Hz)", "OSI", color_pal, s=5)
    plot_scat(axs[1, 1], df, "Spont_Rate(Hz)", "DSI", color_pal, s=5)
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

osi_dfs[0].groupby("cell_type").median()["Spont_Rate(Hz)"]


osi_dfs[0].value_counts("cell_type")

osi_dfs[1].groupby("cell_type").median()

# %%

np_df = get_osi_df("neuropixels", "OSI_DSI_DF_data.csv", "Neuropixels")

# for each cell type, plot the scatter plot of spont vs prefered rate.


fig, axs = plt.subplots(4, 5, figsize=(15, 15))
axcount = 0
for cell_type, df in np_df.groupby("cell_type"):
    ax = axs[axcount // 5, axcount % 5]
    sns.scatterplot(
        ax=ax,
        data=df,
        y="OSI",
        x="Spont_Rate(Hz)",
        s=5,
        hue="cell_type",
    )
    ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.plot([0, 100], [0, 100], "k--")
    # ax.set_aspect("equal")
    ax.set_title(cell_type)
    # turn off legend
    ax.get_legend().remove()
    axcount += 1
plt.tight_layout()


# %%

# df.index.duplicated().sum()

# df.index
