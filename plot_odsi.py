# %%

import pandas as pd
import numpy as np
from sonata.circuit import File
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pathlib


billeh_compare = False # mode to compare with billeh et al (for e4)
keys = ["whislo", "q1", "med", "q3", "whishi"]
billeh_rate = [dict(zip(keys, [0, 0.5, 2, 3, 8]))]
billeh_DSI = [dict(zip(keys, [0, 0.08, 0.14, 0.24, 0.4]))]
# billeh_DSI = [0, 0.08, 0.14, 0.24, 0.4]
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
colors = ["black", "blue", "green", "blue", "black"]
lengths = [0.1, 0.2, 0.2, 0.2, 0.1]


def draw_billeh_line(ax, x, y, color, length):
    # draw a horizontal line at the specified x and y
    ax.plot([x - length, x + length], [y, y], color=color)
    return


if __name__ == "__main__":
    basedir = sys.argv[1]
    # basedir = "original_mini"
    # basedir = "miniature"

    net = File(basedir + "/network/v1_nodes.h5", basedir + "/network/v1_node_types.csv")
    v1df = net.nodes["v1"].to_dataframe()

    osi_df = pd.read_csv(basedir + "/metrics/OSI_DSI_DF.csv", sep=" ")

    df = v1df.merge(osi_df)

    cores = np.sqrt(df["x"] ** 2 + df["z"] ** 2) <= 400
    df_core = df[cores]
    resp = df_core["max_mean_rate(Hz)"] > 0.5

    f, ax = plt.subplots(4, 1, figsize=(10, 15))

    if billeh_compare:
        df_core.boxplot("max_mean_rate(Hz)", ax=ax[0])
        x = ax[0].get_xlim()[1] + 0.5
        ax[0].bxp(billeh_rate, showfliers=False, positions=[x])
        # add billeh tickmark
        ax[0].get_xticklabels()[int(x - 1)].set_text("Billeh2020")
    else:
        df_core.boxplot("max_mean_rate(Hz)", "pop_name", ax=ax[0])
    ax[0].set_ylim([0, 35])


    if billeh_compare:
        df_core[resp].boxplot("DSI", ax=ax[1])
        ax[1].bxp(billeh_DSI, showfliers=False, positions=[x])
        ax[1].get_xticklabels()[int(x - 1)].set_text("Billeh2020")
    else:
        df_core[resp].boxplot("DSI", "pop_name", ax=ax[1])
    ax[1].set_ylim([0, 1])
        

    if billeh_compare:
        df_core[resp].boxplot("OSI", ax=ax[2])
        ax[2].set_xlim([0, x + 0.5])
    else:
        df_core[resp].boxplot("OSI", "pop_name", ax=ax[2])
    ax[2].set_ylim([0, 1])

    df_core.plot.scatter(x="max_mean_rate(Hz)", y="DSI", s=0.3, ax=ax[3])
    ax[3].set_xscale("log")
    # sns.scatterplot(data=df_core, x="max_mean_rate(Hz)", y="DSI", s=4)
    # ax[3].set_xscale("log")
    for i in range(3):
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=30)

    plt.tight_layout()
    pathlib.Path(basedir + "/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(basedir + "/figures/OSI_DSI.png")

