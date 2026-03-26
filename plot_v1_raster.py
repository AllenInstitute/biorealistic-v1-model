# %% simple script for plotting spike rasters

from bmtk.analyzer.spike_trains import plot_raster, plot_rates, plot_rates_boxplot
from numpy.core.numeric import count_nonzero
import matplotlib.pyplot as plt

networks = ["miniature"]
# networks = ["fullmodel56"]
# networks = ["normalize_by_type"]
for n in networks:
    fig = plot_raster(
        # config_file=f"{n}/8dir_10trials/angle0_trial0/config_0.json",
        config_file=f"{n}/configs/config_plain.json",
        # config_file=f"{n}/output_lgnbkg/config_lgnbkg.json",
        # config_file=f"{n}/output/config_lgnonly.json",
        group_by="pop_name",
        times=(0.0, 3000.0),
        show=False,
    )
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(f"Raster_{n}.png")


# %% rates
for n in networks:
    fig = plot_rates(
        config_file=f"{n}/configs/config_plain.json",
        # config_file=f"{n}/configs/config_lgnbkg.json",
        group_by="pop_name",
        times=(0.0, 1000.0),
        show=False,
    )
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(f"Rates_{n}.png")

# %%
for n in networks:
    fig = plot_rates_boxplot(
        config_file=f"{n}/configs/config_plain.json",
        # config_file=f"{n}/configs/config_lgnbkg.json",
        group_by="pop_name",
        times=(0.0, 1000.0),
        show=False,
    )
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(f"Rates_bx_{n}.png")


#%%
fig

# %%


from bmtk.utils.reports.spike_trains.plotting import plot_raster

plot_raster("output/spikes.h5")


# pop_name = ...
# spikes_p = SpikeTrains.load("output/spikes.h5", populations=[pop_name])


# %%

import pandas as pd
from bmtk.utils.reports.spike_trains import SpikeTrains

sim_type = "lgnbkg"
outputdir = "output_" + sim_type

spikes = SpikeTrains.load(outputdir + "/spikes.h5")
df = spikes.to_dataframe()

spont_df = df[(df.timestamps >= 100.0) & (df.timestamps < 500.0)]
evoked_df = df[(df.timestamps >= 1000.0) & (df.timestamps < 3000.0)]

spont_fr = pd.DataFrame(spont_df["node_ids"].value_counts() / 400.0 * 1000)
evoked_fr = pd.DataFrame(evoked_df["node_ids"].value_counts() / 2000.0 * 1000)

spont_fr = spont_fr.rename(columns={"node_ids": "spont_fr"})
evoked_fr = evoked_fr.rename(columns={"node_ids": "evoked_fr"})


# %% get node list
from sonata.circuit import File
import numpy as np

net = File(
    data_files="network/v1_nodes.h5", data_type_files="network/v1_node_types.csv"
)


v1_df = net.nodes.get_population("v1").to_dataframe()
v1_df = v1_df.merge(spont_fr, left_on="node_id", right_index=True, how="left")
v1_df = v1_df.merge(evoked_fr, left_on="node_id", right_index=True, how="left")

v1_df["spont_fr"] = v1_df["spont_fr"].fillna(0)
v1_df["evoked_fr"] = v1_df["evoked_fr"].fillna(0)
v1_df["log_spont_fr"] = np.log10(v1_df["spont_fr"])
v1_df["log_evoked_fr"] = np.log10(v1_df["evoked_fr"])

v1_df["is_core"] = ((v1_df.x) ** 2 + (v1_df.z) ** 2) < 400.0 ** 2
v1_df["is_core"].sum()
v1_df

# %% now ready to plot for each population
import seaborn as sns
import matplotlib.pyplot as plt
import os


# pop_name = "i4Pvalb"
# pop = v1_df.groupby("pop_name").get_group("i4Pvalb")
simoutdir = f"FRdists_{sim_type}"
if not os.path.exists(simoutdir):
    os.mkdir(simoutdir)

for (pop_name, pop) in v1_df.groupby("pop_name"):
    f, ax = plt.subplots(4, 2, figsize=(15, 15))

    for (iscore, inout) in pop.groupby("is_core"):

        bins = (np.linspace(0, 90, 46), np.linspace(-0.5, 2.0, 31))
        target = ["spont_fr", "log_spont_fr", "evoked_fr", "log_evoked_fr"]
        # inout = [core, noncore]
        # titlestr += f'{pop_name}: Mean: {meanfr:.2} ± {std:.2}'

        for i in range(4):
            sns.histplot(
                data=inout,
                ax=ax[i, int(not iscore)],
                stat="count",
                multiple="stack",
                x=target[i],
                bins=bins[i % 2],
                kde=False,
                palette="colorblind",
                hue="node_type_id",
                element="bars",
                legend=((not iscore) & (i == 0)),
                # legend=False,
            )

            ax[i, int(not iscore)].set_title(f"Core:{iscore}    {target[i]}")
            ax[i, int(not iscore)].set_xlabel("log10(FR (Hz))")
            ax[i, int(not iscore)].set_ylabel("count")

    f.suptitle(f"{pop_name}")
    f.tight_layout(h_pad=0.5)
    plt.savefig(f"FRdists_{sim_type}/FR_dist_{pop_name}.png", transparent=False)


# %% make a table too.

pop_table = pd.DataFrame(v1_df.groupby("pop_name")["spont_fr", "evoked_fr"].mean())
pop_table = pop_table.rename(
    columns={"spont_fr": "Mean spont.", "evoked_fr": "Mean evoked"}
)
new_table = v1_df.groupby("pop_name")["spont_fr", "evoked_fr"].std()
pop_table = pop_table.merge(new_table, left_index=True, right_index=True)
pop_table = pop_table.rename(
    columns={"spont_fr": "STD spont.", "evoked_fr": "STD evoked"}
)
pop_table

new_table = v1_df.groupby("pop_name")["spont_fr", "evoked_fr"].median()
pop_table = pop_table.merge(new_table, left_index=True, right_index=True)
pop_table = pop_table.rename(
    columns={"spont_fr": "Med. spont.", "evoked_fr": "Med. evoked"}
)

pop_table = pop_table.iloc[:, [0, 2, 4, 1, 3, 5]]
pop_table
# pop_table = pd.DataFrame(v1_df.groupby("pop_name")["_fr"].mean())

# for (pop_name, pop) in v1_df.groupby("pop_name"):
#    for (iscore, inout) in pop.groupby("is_core"):

# %%
import matplotlib.pyplot as plt

# df["pop_name"] = list(v1_df.pop_name[df["node_ids"]])

v1_df.pop_name[list(df["node_ids"])]
df["node_ids"]

# df short
dfs = df[df.timestamps < 1000]


# %%
def pop_name_color(pop_name):
    if "e" in pop_name:
        return "tab:blue"
    if "Pvalb" in pop_name:
        return "tab:orange"
    if "Sst" in pop_name:
        return "tab:green"
    if ("Vip" in pop_name) or ("Htr3a" in pop_name):
        return "tab:red"


plt.figure(figsize=(15, 10))
plt.scatter(df.timestamps, df.node_ids, c=df.pop_name.map(pop_name_color), s=0.3)
# df.plot.scatter(x='timestamps', y='node_ids', )

# %%
import vaex

dfv = vaex.from_pandas(df)

dfv.plot_bq(dfv.timestamps, dfv.node_ids)


# %%
df = vaex.read_csv("output_lgnbkg/spikes.csv", sep=" ")

import matplotlib.pyplot as plt

# %%
plt.figure(figsize=(15, 10))
df.plot("timestamps", "node_ids", f="log1p", shape=1000)


pdf = df.to_pandas_df()


# %%
# sns.heatmap(pdf[]

# sns.heatmap(pdf[['timestamps', 'node_ids']])
sns.jointplot(data=pdf, x="timestamps", y="node_ids")


# %%
# v1_df
# v1_df.pop_name.unique()
import pylab as plt

plt.scatter(df.timestamps, df.node_ids)

