# %% simple script to inspect miniature network's tuning properties]
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sonata.circuit import File


net = File("miniature/v1_nodes.h5", "miniature/v1_node_types.csv")
v1df = net.nodes["v1"].to_dataframe()

n_neu = len(v1df)

spikes_df = pd.read_csv("miniature_output/spikes.csv", sep=" ")
spikes_df = spikes_df.query("timestamps > 500")

spike_rates = np.zeros(n_neu)
for node_id, count in spikes_df.value_counts("node_ids").items():
    spike_rates[node_id] = count / 2.5  # to get FR

v1df["FR"] = spike_rates


v1df_s = v1df.sort_values("tuning_angle")

fig, ax = plt.subplots(1, 1, figsize=(15, 5))

v1df_s.plot.scatter("tuning_angle", "FR", ax=ax)
v1df_s.rolling(100).mean().plot(
    "tuning_angle", "FR", ax=ax, color="tab:orange", linewidth=3
)

