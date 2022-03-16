# %% develop a method to adjust bkg strengths

# develop the following methods

# run the simulation
# collect firing rate stats for each model
# simple solver for the firing rates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sonata.circuit
import subprocess


# %% try getting FR
# get the spike dataframe


def get_spike_df(basedir, query="timestamps < 500"):
    outputdir = basedir + "/output_lgnbkg"
    spike_df = pd.read_csv(outputdir + "/spikes.csv", sep=" ")
    spike_df = spike_df.query(query)
    return spike_df


def get_v1_dfs(basedir):
    data_files = [basedir + "/network/v1_nodes.h5"]
    data_type_files = [basedir + "/network/v1_node_types.csv"]
    v1 = sonata.circuit.File(data_files=data_files, data_type_files=data_type_files)
    v1df = v1.nodes["v1"].to_dataframe()
    return v1df


def get_model_fr(basedir):
    spike_df = get_spike_df(basedir)
    v1df = get_v1_dfs(basedir)

    v1df["node_ids"] = v1df["node_id"]
    spike_df = spike_df.merge(v1df[["node_ids", "node_type_id"]], on="node_ids")
    v1df["spike_rate"] = spike_df.value_counts("node_ids") / 0.5
    v1df["spike_rate"][np.isnan(v1df["spike_rate"])] = 0
    model_fr = v1df.groupby("node_type_id")["spike_rate"].mean()
    return model_fr


# %% calculate the new FR
# bisection search seems reasonable


class BisectionSolver:
    def __init__(self, lx, rx, target_fr, tolerance=1e-2):
        self.lx = lx
        self.rx = rx
        self.target_fr = target_fr
        self.tolerance = tolerance
        self.ly = np.nan
        self.ry = np.nan

    def mid(self):
        return (self.lx + self.rx) / 2

    def solved(self, new_y):
        # solution is within tolerance
        cond1 = new_y > self.target_fr * (1 - self.tolerance)
        cond2 = new_y < self.target_fr * (1 + self.tolerance)
        return cond1 & cond2

    def step(self, new_y):
        if np.isnan(new_y):
            raise ValueError("New FR is nan. Something is wrong.")

        if np.isnan(self.ly):  # initial condition
            if (new_y - self.target_fr) > 0:  # ill initial condition
                # raise ValueError("Starting left edge is already positive")
                print("Starting left edge is already positive")
            self.ly = new_y
            return self.rx
        elif np.isnan(self.ry):  # next condition
            if (new_y - self.target_fr) < 0:  # ill initial condition
                # raise ValueError("Starting right edge is already negative")
                print("Starting right edge is already negative")
            self.ry = new_y  # now we collected all initial conditions, so return the mid point)
            return self.mid()
        else:  # normal interation
            if self.solved(new_y):  # tolerance achieved. return -1
                return -1.0
            if (new_y - self.target_fr) < 0:  # update the left edge
                self.lx = self.mid()
                return self.mid()
            else:  # update the right edge
                self.rx = self.mid()
                return self.mid()


# %%  let's do simple testing
"""
def func(x):
    return x**2

bs = BisectionSolver(0, 1, 4, 0.00001)
x = 0
for i in range(100):
    val = func(x)
    x = bs.step(val)
    print(x)
    if x < 0:
        break
# OK. it works.
"""

# bkg_edge_df['model_fr'] = model_fr


# %% get the target FR


def get_formatted_edge_df(basedir):
    bkg_edge_name = basedir + "/network/bkg_v1_edge_types.csv"
    bkg_edge_df = pd.read_csv(bkg_edge_name, sep=" ")
    bkg_edge_df["target_type_id"] = (
        bkg_edge_df["target_query"].str.extract("(\d+)").astype(int)
    )
    bkg_edge_df.index = bkg_edge_df["target_type_id"]
    return bkg_edge_df


def get_target_fr(basedir):
    target_pop = pd.read_csv("base_props/bkg_weights_population_init.csv", sep=" ")
    v1df = get_v1_dfs(basedir)
    model_to_pop = v1df[["node_type_id", "pop_name"]].drop_duplicates()
    model_to_fr = model_to_pop.merge(target_pop[['population', 'target_mean_fr']], left_on='pop_name', right_on='population')
    model_to_fr.index = model_to_fr['node_type_id']
    target_fr = model_to_fr['target_mean_fr']
    return target_fr


basedir = "miniature"
get_target_fr(basedir)
# %% update the parameters
basedir = "miniature"


def update_bkg_weights(basedir, new_weight):
    bkg_edge_name = basedir + "/network/bkg_v1_edge_types.csv"
    bkg_edge_df = get_formatted_edge_df(basedir)
    # bkg_edge_df['model_fr'] = model_fr
    # new_weight = bkg_edge_df['syn_weight'].copy()
    # new_weight[:] = 20
    bkg_edge_df["syn_weight"] = new_weight
    bkg_edge_df.index = bkg_edge_df["edge_type_id"]
    del bkg_edge_df["edge_type_id"]
    del bkg_edge_df["target_type_id"]
    bkg_edge_df.to_csv(bkg_edge_name, sep=" ")
    return 0


# %% run simulation with the existing configuration


def run_simulation(basedir, ncore=8):
    config_file = basedir + "/configs/config_lgnbkg.json"
    command = f"mpirun -np {ncore} python run_pointnet.py {config_file}"
    print("running the command below...")
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)
    return process.returncode


# run_simulation(basedir) it works.


# %% let's write the main function

# if __name__ == "__main__":
    # start with forming the problem.

basedir = "miniature"
v1df = get_v1_dfs(basedir)
tfr = get_target_fr(basedir)

tfr.keys()[0]
solvers = {nid: BisectionSolver(0, 100, tfr[nid]) for nid in tfr.keys()}

weight = tfr.copy()
weight[:] = 0.0
weight.name = 'syn_weight'

for i in range (1000):
    print(weight)
    update_bkg_weights(basedir, weight)
    run_simulation(basedir)
    model_fr = get_model_fr(basedir)
    
    new_weight = weight.copy()
    for i, v in model_fr.items():
        new_weight[i] = solvers[i].step(v)
        if new_weight[i] >= 0:
            weight[i] = new_weight[i]

    if all(new_weight < 0): # solution achieved
        break

# %%
# (new_weight>0).sum()




# %%

"""
def func(x):
    return x**2

bs = BisectionSolver(0, 1, 4, 0.00001)
x = 0
for i in range(100):
    val = func(x)
    x = bs.step(val)
    print(x)
    if x < 0:
        break
# OK. it works.
"""

