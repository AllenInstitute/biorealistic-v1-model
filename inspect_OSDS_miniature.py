# %% simple script to inspect miniature network's tuning properties]
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as pltticker
import numpy as np
from sonata.circuit import File
import h5py

# d = "miniature/"
d = "small/"
# d = "single/"
# d = "original_mini/"
dnet = d + "network/"
# dout = d + "output_multimeter_bkg/"
# dout = d + "output_single_spikes/"
dout = d + "output_multimeter/"
dfiles = [dnet + "lgn_nodes.h5", dnet + "v1_nodes.h5", dnet + "lgn_v1_edges.h5"]
dtfiles = [
    dnet + "lgn_node_types.csv",
    dnet + "v1_node_types.csv",
    dnet + "lgn_v1_edge_types.csv",
]

net = File(dfiles, dtfiles)
v1df = net.nodes["v1"].to_dataframe()


n_neu = len(v1df)

spikes_df = pd.read_csv(dout + "spikes.csv", sep=" ")
spikes_df = spikes_df.query("timestamps > 500")

spike_rates = np.zeros(n_neu)
for node_id, count in spikes_df.value_counts("node_ids").items():
    spike_rates[node_id] = count / 2.5  # to get FR

v1df["FR"] = spike_rates


# v1df_one_type = v1df[v1df["node_type_id"] == 479993900]
# v1df_one_type = v1df[v1df["node_type_id"] == 483020137]
# v1df_one_type = v1df[v1df["node_type_id"] == 488689403]
# v1df_s = v1df_one_type.sort_values("tuning_angle")
v1df_s = v1df.sort_values("tuning_angle")
# v1df_s = v1df_s[v1df_s.FR < 5]

v1df
v1df_s = v1df_s.query('location == "VisL4" & ei == "e"')

fig, ax = plt.subplots(1, 1, figsize=(15, 5))

v1df_s.plot.scatter("tuning_angle", "FR", ax=ax)
v1df_s.rolling(100).mean().plot(
    "tuning_angle", "FR", ax=ax, color="tab:orange", linewidth=3
)
ax.set_ylim([0, 30])


#  look at the traces
v1df.to_csv("v1df_mini.csv")


# %% determining which cell type is firing
used_models_df = pd.read_csv("misc_files/used_models.csv", index_col=0)
used_models_df.specimen__id

rheo_df = used_models_df[["specimen__id", "model_rheo"]]
v1df_r = v1df.merge(rheo_df, left_on="node_type_id", right_on="specimen__id")

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
v1df_r.plot.scatter("model_rheo", "FR", ax=ax)
# v1df_r.plot.hexbin('model_rheo', 'FR', ax=ax, bins='log')


# %% comparison with the unitary inputs
import json

v1unitary = json.load(open("misc_files/v1_synapse_amps.json", "r"))
v1unitary_ser = pd.Series(v1unitary["e2e"], name="Unitary PSP")
v1unitary_ser_i = pd.Series(v1unitary["e2i"], name="Unitary PSP")
v1unitary_ser.index = pd.to_numeric(v1unitary_ser.index)
v1unitary_ser_i.index = pd.to_numeric(v1unitary_ser_i.index)

v1unitary_ser_both = v1unitary_ser.append(v1unitary_ser_i)


unit_df = used_models_df.merge(
    v1unitary_ser_both, left_on="specimen__id", right_index=True
)
unit_df.keys()


unit_df.plot.scatter("model_rheo", "Unitary PSP")
unit_df.plot.scatter("R_input", "Unitary PSP")
unit_df.plot.scatter("C", "Unitary PSP")
unit_df["R/C"] = unit_df["R_input"] / unit_df["C"]
unit_df.plot.scatter("R/C", "Unitary PSP")


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py

v1df = pd.read_csv("v1df_mini.csv")


def get_np_data(filename):
    with open(filename, "rb") as file:
        f = h5py.File(file, "r")
        f["report/v1/data"]
        f["report/v1"].keys()
        time = np.array(f["report/v1/mapping/time"])
        time = np.arange(0, *time[1:])
        time.shape
        np.array(f["report/v1/mapping/node_ids"])
        data = f["report/v1/data"]
        npd = np.array(data)
        f.close()

        return (time, npd)


time, npd = get_np_data(f"{dout}cai_traces.h5")
# time, npd = get_np_data(f"{dout}cai_traces_single.h5")

# plt.plot(time, npd[:, [276, 284, 300]])
# plt.plot(time, npd[:, [276]])
# plt.plot(time, npd[:, [284]])
# plt.plot(time, npd[:, [300]])
# # from another group
# plt.plot(time, npd[:, [370]])
# plt.plot(time, npd[:, [394]])
# f1 analysis

# pick e4 cells
# v1df_sub = v1df.query("ei=='e' and location=='VisL4'").sample(2000)
v1df_sub = v1df.query("ei=='e' and location=='VisL4'").copy()
# v1df_sub = v1df


spont_time = time < 500
stim_time = time > 1000
spont_I = npd[spont_time, :].mean(axis=0)
stim_I = npd[stim_time, :].mean(axis=0)
freq = 2.0
f1_I = npd.transpose() * np.exp(-2j * np.pi * freq * time / 1000)
f1_I = np.abs(f1_I[:, stim_time].mean(axis=1))
# plt.plot(time, np.exp(2j*np.pi*freq*time/1000).real)
# plt.plot(spont_I)
# plt.plot(stim_I)
# plt.plot(f1_I)
# v1df_sub = v1df.loc[range(0, 17001, 100)]


vi = v1df_sub.index
npd[spont_time, :][:, vi[0:5]].mean(axis=0)
npd[stim_time, :][:, vi[0:5]].mean(axis=0)

vi


stim_I[vi].mean()
v1df_sub["evoked I"] = stim_I[vi] - spont_I[vi]
v1df_sub["f1 I"] = f1_I[vi]
v1df_sub["f1/evoked"] = f1_I[vi] / (stim_I[vi] - spont_I[vi])

print(v1df_sub["evoked I"].mean())
plt.hist(v1df_sub["evoked I"], bins=100)
plt.xlabel("Evoked current (pA)")
plt.ylabel("counts")

v1df_sub.query("ei=='e' and location=='VisL4'")["evoked I"].mean()


v1df_sub.plot.scatter("tuning_angle", "evoked I")
v1df_sub.plot.scatter("tuning_angle", "f1 I")
v1df_sub.plot.scatter("tuning_angle", "f1/evoked")
plt.ylim([0.0, 1.5])

# print((stim_I - spont_I).mean())
# (spont_I).mean()
# (stim_I).mean()

# %% try to characterize for each subpopulations
locations = ["VisL2/3", "VisL4", "VisL5", "VisL6"]
for loc in locations:
    v1df_sub = v1df.query(f"ei=='i' and location=='{loc}'")
    vi = v1df_sub.index
    print(f"{loc}: {stim_I[vi].mean() - spont_I[vi].mean()}")


# %% It'll look great if you chop off neurons with high FR.

plt.plot(npd[:, vi[0:5]])
# plt.xlim([2000, 2100])
plt.xlim([0, 100])

# %%
plt.figure(figsize=(15, 5))
plt.plot(time, npd[:, vi[0]])
plt.xlim([400.8, 405])
plt.grid(True)

# %% report the distribution of the number of synapses form LGN to v1


# def get_ncon(v1_nid):
#    return len([[] for i in net.edges["lgn_to_v1"].get_target(v1_nid)])


# ncons = [get_ncon(nid) for nid in v1df.node_id]


# %%
np.mean(ncons)

# %%
plt.scatter(ncons, v1df["FR"])

sum(v1df["FR"] == 0)


# %%
def plot_one(df_e4, df_lgn, nid=None, nsyn_th=3):
    plt.plot(df_lgn["x"], df_lgn["y"], ".", markersize=0.1)
    v1cell = df_e4.sample()
    if nid is None:
        nid = int(v1cell["node_id"])
    else:
        v1cell = df_e4.loc[nid]
    # get all the sources
    sources = [
        edge.source_node_id
        for edge in net.edges["lgn_to_v1"].get_target(nid)
        # if edge["nsyns"] >= nsyn_th
    ]
    # nsyns = [edge["nsyns"] for edge in net.edges["lgn_to_v1"].get_target(nid)]
    # plot them
    df_lgnsub = df_lgn[df_lgn["node_id"].isin(sources)]
    toff = df_lgnsub["pop_name"].str.contains("tOFF_")
    sus = df_lgnsub["pop_name"].str.contains("sOFF_")
    # sus = sus | df_lgnsub["pop_name"].str.contains("sON_")
    others = np.logical_not(toff | sus)

    plt.plot(df_lgnsub[toff]["x"], df_lgnsub[toff]["y"], "r.", markersize=2)
    plt.plot(df_lgnsub[sus]["x"], df_lgnsub[sus]["y"], "g.", markersize=2)
    plt.plot(df_lgnsub[others]["x"], df_lgnsub[others]["y"], "b.", markersize=2)
    plt.axis("equal")

    # locx = pltticker.MultipleLocator(base=16.0)
    # locy = pltticker.MultipleLocator(base=12.0)
    locx = pltticker.MultipleLocator(base=40.0)
    locy = pltticker.MultipleLocator(base=40.0)
    plt.gca().xaxis.set_major_locator(locx)
    plt.gca().yaxis.set_major_locator(locy)
    plt.gca().grid(which="major", axis="both")
    plt.gca().set_ylabel(f'id:{nid}, angle:{float(v1cell["tuning_angle"]):0.0f}')

    return df_lgnsub


lgndf = net.nodes["lgn"].to_dataframe()
# np.random.seed(0)
df_sources = pd.DataFrame()
fig = plt.figure(figsize=(10, 6))
for i in range(1, 10):
    ax = fig.add_subplot(3, 3, i)
    plt.axes(ax)
    df_sources = pd.concat((df_sources, plot_one(v1df, lgndf, nsyn_th=1)))

plt.tight_layout()


# %% diagnosis
v1cell = v1df.sample()
nid = int(v1cell["node_id"])
sources = [edge.source_node_id for edge in net.edges["lgn_to_v1"].get_target(nid)]
nsyns = [edge["nsyns"] for edge in net.edges["lgn_to_v1"].get_target(nid)]

sources


# %%

v1df[v1df["node_type_id"] == 479993900]
v1df[v1df["node_type_id"] == 483020137]
v1df[v1df["node_type_id"] == 488689403]
plot_one(v1df, lgndf, nsyn_th=2, nid=276)
plot_one(v1df, lgndf, nsyn_th=2, nid=284)
plot_one(v1df, lgndf, nsyn_th=2, nid=300)

plot_one(v1df, lgndf, nsyn_th=2, nid=370)
plot_one(v1df, lgndf, nsyn_th=2, nid=394)

v1df.groupby("node_type_id").mean()


# %% lognorm demo
from scipy.stats import lognorm

x = np.array(range(1, 7000))

s = 0.661

lognorm.pdf(x, s, 0, 10)

np.exp(7.456)


# plt.plot(x, lognorm.pdf(x, s, 0, np.exp(7.456)))

# %%
samples = np.round(lognorm.rvs(s, 0, np.exp(7.456), size=44))
plt.hist(samples, bins=np.linspace(0, 7000, 30))
