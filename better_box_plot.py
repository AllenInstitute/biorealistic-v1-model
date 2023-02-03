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


def plot_one(ax, df, metric_name, ylim, cpal=None):
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
    "full, bkg=1.0": "tab:blue",
    "full, bkg=0.5": "tab:cyan",
    #  'small, bkg=1.0': 'tab:red',
    #  'small, bkg=0.5': 'tab:pink',
    "small all round1": "tab:red",
    "small dendritic norm fix": "tab:pink",
    "small dendritic norm fix, no rand": "tab:cyan",
    "small dendritic norm fix, full bkg": "tab:cyan",
    "small all round2": "tab:pink",
    "small all round3": "tab:red",
    "small all round4": "tab:pink",
    "full all round4": "tab:blue",
    "full all round3": "tab:blue",
    "full all round2": "tab:cyan",
    "full all round1": "tab:red",
    "full, all": "tab:blue",
    "small, lgnbkg": "tab:pink",
    "small constBKG round6": "tab:cyan",
    "full round6": "tab:pink",
    "full round7": "tab:blue",
    "full round9": "tab:cyan",
    "small round7": "tab:cyan",
    "small round9": "tab:blue",
    "small newest": "tab:blue",
    "Lognorm, norecurrent": "tab:pink",
    "New model": "tab:blue",
    #  'small, lgnonly': 'tab:olive',
    #  'small, lgn+bkg': 'tab:brown',
    #  'small, +recurrent': 'tab:red',
    "Neuropixels": "tab:gray",
    # "Neuropixels": "k",
}


# Billeh, small, neuropixels set
# osi_dfs.append(get_osi_df("billeh", "OSI_DSI_DF.csv", "Billeh 2020, GLIF final"))
# osi_dfs.append(get_osi_df("small", "OSI_DSI_DF_27pBKG_lgnbkg.csv", "small, lgnbkg", radius=100.0))

osi_dfs.append(get_osi_df("billeh", "OSI_DSI_DF.csv", "Billeh 2020, GLIF final"))
osi_dfs.append(get_osi_df("neuropixels", "OSI_DSI_DF.csv", "Neuropixels"))
osi_dfs.append(
    get_osi_df("flat", "OSI_DSI_DF_round9.csv", "Lognorm, norecurrent", radius=850.0)
)
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF_round6.csv", "full round6"))
osi_dfs.append(get_osi_df("full", "OSI_DSI_DF_round9.csv", "full round9"))
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF_round7.csv", "full round7"))

osi_dfs.append(
    get_osi_df("small", "OSI_DSI_DF_round9.csv", "small round9", radius=100.0)
)
# osi_dfs.append(
#     get_osi_df("small", "OSI_DSI_DF_round7.csv", "small round7", radius=100.0)
# )
# osi_dfs.append(
#     get_osi_df("full", "OSI_DSI_DF.csv", "New model")
# )
# osi_dfs.append(get_osi_df("small", "OSI_DSI_DF.csv", "small all round4", radius=100.0))
# osi_dfs.append(
#     get_osi_df("small", "OSI_DSI_DF_bkg1.0.csv", "small all round1", radius=100.0)
# )
# osi_dfs.append(
#     get_osi_df(
#         "small", "OSI_DSI_DF_normfix.csv", "small dendritic norm fix", radius=100.0
#     )
# )
# osi_dfs.append(
#     get_osi_df(
#         "small",
#         "OSI_DSI_DF.csv",
#         "small dendritic norm fix, full bkg",
#         radius=100.0,
#     )
# )
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF_round3_all.csv", "full all round3"))
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF.csv", "full all round4"))
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF_bkg1.0.csv", "full all round1"))
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF_27pBKG_all.csv", "full all round2"))
# osi_dfs.append(get_osi_df("small", "OSI_DSI_DF_round3_all.csv", "small all round3", radius=100.0))
# osi_dfs.append(get_osi_df("small", "OSI_DSI_DF_27pBKG_all.csv", "small all round2", radius=100.0))
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF_27pBKG_all.csv", "full, all", radius=400.0))

# Billeh set
# osi_dfs.append(get_osi_df("billeh", "OSI_DSI_DF.csv", "Billeh 2020, GLIF final"))
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF_bkg1.0.csv", "full, bkg=1.0"))
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF_bkg0.5.csv", "full, bkg=0.5"))

# full-small set
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF_bkg1.0.csv", "full, bkg=1.0"))
# osi_dfs.append(get_osi_df("small", "OSI_DSI_DF_bkg1.0.csv", "small, bkg=1.0", radius=100.0))
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF_bkg0.5.csv", "full, bkg=0.5"))
# osi_dfs.append(get_osi_df("small", "OSI_DSI_DF_bkg0.5.csv", "small, bkg=0.5", radius=100.0))

# recurrent effect set
# osi_dfs.append(get_osi_df("small", "OSI_DSI_DF_lgnonly.csv", "small, lgnonly", radius=100.0))
# osi_dfs.append(get_osi_df("small", "OSI_DSI_DF_lgnbkg.csv", "small, lgn+bkg", radius=100.0))
# osi_dfs.append(get_osi_df("small", "OSI_DSI_DF_bkg1.0.csv", "small, +recurrent", radius=100.0))
# osi_dfs.append(get_osi_df("small", "OSI_DSI_DF_bkg0.5.csv", "small, bkg=0.5"))

# osi_dfs.append(get_osi_df("billeh", "OSI_DSI_DF.csv", "Billeh 2020, GLIF final"))
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF_bkg1.0.csv", "full, bkg=1.0"))
# osi_dfs.append(get_osi_df("small", "OSI_DSI_DF_bkg1.0.csv", "small, bkg=1.0"))
# osi_dfs.append(get_osi_df("full", "OSI_DSI_DF_bkg0.5.csv", "full, bkg=0.5"))
# osi_dfs.append(get_osi_df("small", "OSI_DSI_DF_bkg0.5.csv", "small, bkg=0.5"))
# osi_dfs.append(get_osi_df("small", "OSI_DSI_DF_lgnbkg.csv", "small, lgn+bkg"))

df = pd.concat(osi_dfs)


fig, axs = plt.subplots(4, 1, figsize=(12, 20))
# fig, axs = plt.subplots(3, 1, figsize=(9, 12))
plot_one(axs[0], df, "Spont_Rate(Hz)", [0, 50], color_pal)
plot_one(axs[1], df, "Rate at preferred direction (Hz)", [0, 50], color_pal)
plot_one(axs[2], df, "DSI", [0, 1], color_pal)
plot_one(axs[3], df, "OSI", [0, 1], color_pal)
# plot_scat(axs[4], df, "Rate at preferred direction (Hz)", "DSI", color_pal, s=5)
# plot_scat(axs[4], df, "Rate at preferred direction (Hz)", "OSI", color_pal, s=5)

plt.tight_layout()
# plt.savefig("box_Dec12.svg", bbox="tight")
# plt.savefig("box_Dec12.png", dpi=150)

# %%
l4df = df.query("data_type == 'full round9' and location == 'VisL4'")
# print the following 4 items
print(l4df.groupby('cell_type').mean()['Spont_Rate(Hz)'])
print(l4df.groupby('cell_type').mean()['Avg_Rate(Hz)'])
print(l4df.groupby('ei').mean()['Spont_Rate(Hz)'])
print(l4df.groupby('ei').mean()['Avg_Rate(Hz)'])




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
