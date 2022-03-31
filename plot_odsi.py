# %%

import pandas as pd
import numpy as np
from sonata.circuit import File
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pathlib


if __name__ == "__main__":
    # basedir = sys.argv[1]
    # basedir = "original_mini"
    basedir = "miniature"

    net = File(basedir + "/network/v1_nodes.h5", basedir + "/network/v1_node_types.csv")
    v1df = net.nodes["v1"].to_dataframe()

    osi_df = pd.read_csv(basedir + "/metrics/OSI_DSI_DF.csv", sep=" ")

    df = v1df.merge(osi_df)

    cores = np.sqrt(df["x"] ** 2 + df["z"] ** 2) <= 400
    df_core = df[cores]
    resp = df_core["max_mean_rate(Hz)"] > 0.5

    f, ax = plt.subplots(4, 1, figsize=(10, 15))

    df_core.boxplot("max_mean_rate(Hz)", "pop_name", ax=ax[0])
    ax[0].set_ylim([0, 35])

    df_core[resp].boxplot("DSI", "pop_name", ax=ax[1])
    ax[1].set_ylim([0, 1])
    # sns.boxplot(data=df_core[resp], x="pop_name", y="DSI", hue=None, ax=ax[1])

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

