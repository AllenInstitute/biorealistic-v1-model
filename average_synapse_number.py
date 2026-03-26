# %% load a network and calculate average number of synapses.

import network_utils as nu
import pandas as pd
import polars as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


basedir = "core"
core_radius = 200

nodes = nu.load_nodes(basedir, core_radius=core_radius)
edges = nu.load_edges(basedir)


full_node_df = nodes["types"].loc[nodes["node_type_id"]]
node_df_pl = pl.DataFrame(full_node_df)


# limit target edges to the core
core_edge = nodes["core"][edges["target_id"]]
core_source = edges["source_id"][core_edge]
core_target = edges["target_id"][core_edge]

source_pop = node_df_pl["pop_name"][core_source]
target_pop = node_df_pl["pop_name"][core_target]


# %%
pop_dfs = []

populations = ["e23", "e4", "e5IT", "e5ET", "e5NP", "e6"]

# estimate total number of synapses on these populations
for population in populations:
    pop_ind = target_pop.str.contains(population)
    counts = np.unique(edges["target_id"][core_edge][pop_ind], return_counts=True)
    # make a dataframe for the counts
    pop_df = pd.DataFrame(
        {
            "pop_name": population,
            "target_id": counts[0],
            "count": counts[1],
        }
    )
    pop_dfs.append(pop_df)

# vstack the dataframes
pop_df = pd.concat(pop_dfs)
# pop_df.plot.boxplot(by="pop_name", column="count")


# pop_df.groupby("pop_name").agg(pl.median("count"))
# pop_df.plot.boxplot(by="pop_name", column="count")

# make a box plot of the count for each pop_name
# pop_df.to_pandas().boxplot(column="count", by="pop_name")
# %%
fig = plt.figure(figsize=(3.5, 7))
sns.boxplot(data=pop_df, x="pop_name", y="count")
plt.ylim(0, 3000)
plt.xlabel("Target population")
plt.ylabel("Number of inputs")
