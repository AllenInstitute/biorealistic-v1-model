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


def get_spike_df(basedir, query="timestamps < 100000", recurrent=False, outdir=None):
    if recurrent:
        outputdir = basedir + "/output_bkgtune_recurrent"
    else:
        outputdir = basedir + "/output_bkgtune"

    if outdir is not None:
        outputdir = basedir + "/" + outdir
    spike_df = pd.read_csv(outputdir + "/spikes.csv", sep=" ")
    spike_df = spike_df.query(query)
    return spike_df


def get_v1_dfs(basedir):
    data_files = [basedir + "/network/v1_nodes.h5"]
    data_type_files = [basedir + "/network/v1_node_types.csv"]
    v1 = sonata.circuit.File(data_files=data_files, data_type_files=data_type_files)
    v1df = v1.nodes["v1"].to_dataframe().copy()
    return v1df


def get_model_fr(basedir, recurrent=False, duration=100.0, target="mean", outdir=None):
    spike_df = get_spike_df(basedir, recurrent=recurrent, outdir=outdir)
    v1df = get_v1_dfs(basedir)

    v1df["node_ids"] = v1df["node_id"]
    spike_df = spike_df.merge(v1df[["node_ids", "node_type_id"]], on="node_ids")
    v1df["spike_rate"] = spike_df.value_counts("node_ids") / duration
    # v1df["spike_rate"][np.isnan(v1df["spike_rate"])] = 0
    # replace nan with 0
    v1df["spike_rate"] = v1df["spike_rate"].fillna(0)
    if target == "mean":
        model_fr = v1df.groupby("node_type_id")["spike_rate"].mean()
    elif target == "median":
        model_fr = v1df.groupby("node_type_id")["spike_rate"].median()
    elif target == "array":
        # return the array of firing rates
        model_fr = v1df.groupby("node_type_id")["spike_rate"].apply(np.array)
    else:
        raise ValueError(f"Unknown target: {target}")
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
        self.force_solved = False

    def mid(self):
        return (self.lx + self.rx) / 2

    def solved(self, new_y):
        # solution is within tolerance
        cond1 = new_y > self.target_fr * (1 - self.tolerance)
        cond2 = new_y < self.target_fr * (1 + self.tolerance)
        return (cond1 and cond2) or self.force_solved

    def step(self, new_y):
        if np.isnan(new_y):
            raise ValueError("New FR is nan. Something is wrong.")

        if np.isnan(self.ly):  # initial condition
            if (new_y - self.target_fr) > 0:  # ill initial condition
                # raise ValueError("Starting left edge is already positive")
                diff = new_y - self.target_fr
                print(f"Starting left edge is already positive, exceeding by {diff}")
                # print("The weight is set to 0, and considered solved.")
                # instead, set the target firing rate to 10% more of the observed rate.
                new_rate = new_y * 1.1
                print(f"The target firing rate is increased by 10% to {new_rate}")
                self.target_fr = new_rate
                # self.force_solved = True
            self.ly = new_y
            return self.rx
        elif np.isnan(self.ry):  # next condition
            if (new_y - self.target_fr) < 0:  # ill initial condition
                # raise ValueError("Starting right edge is already negative")
                diff = new_y - self.target_fr
                print(f"Starting right edge is already negative by {diff}")
            self.ry = new_y  # now we collected all initial conditions, so return the mid point)
            return self.mid()
        else:  # normal interation
            if self.solved(new_y):  # tolerance achieved. return -1
                return -1.0
            if (new_y - self.target_fr) < 0:  # update the left edge
                self.lx = self.mid()
                return self.mid()
            else:  # update the right edge
                if abs(self.rx - self.mid()) < self.tolerance:
                    self.force_solved = True
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


def get_target_fr(basedir, target="target_mean_fr"):
    target_pop = pd.read_csv("base_props/bkg_weights_population_init.csv", sep=" ")
    v1df = get_v1_dfs(basedir)
    model_to_pop = v1df[["node_type_id", "pop_name"]].drop_duplicates()
    model_to_fr = model_to_pop.merge(
        target_pop[["population", target]],
        left_on="pop_name",
        right_on="population",
    )
    model_to_fr.index = model_to_fr["node_type_id"]
    target_fr = model_to_fr[target]
    return target_fr


# basedir = "single"
# get_target_fr(basedir)
# %% update the parameters
# basedir = "single"


def run_command(command):
    print("running the command below...")
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(f"Return code: {process.returncode}")
    return process.returncode


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


# this is a new version after synaptic weight dynamics change
# def update_bkg_weights2(basedir, new_weight):


# write the new weights to a temporary file
def write_new_weights(new_weight):
    # first, update the temporay file for the new weights
    orig_filename = "base_props/bkg_weights_population_init.csv"
    tmp_filename = "bkg_weights_population_tmp.csv"

    bkg_pop_df = pd.read_csv(orig_filename, sep=" ")
    bkg_pop_df["syn_weight"] = new_weight

    bkg_pop_df.to_csv(tmp_filename, sep=" ", index=False)
    return


# %% run simulation with the existing configuration


def run_simulation(basedir, ncore=8, recurrent=False):
    if recurrent:
        config_file = basedir + "/configs/config_bkgtune_recurrent.json"
    else:
        config_file = basedir + "/configs/config_bkgtune.json"
    command = f"mpirun -np {ncore} python run_pointnet.py {config_file}"
    return run_command(command)


# run_simulation(basedir) it works.


# %% let's write the main function


if __name__ == "__main__":
    # start with forming the problem.
    mode = "small_lgnbkg"
    # mode = "flat_wasser"
    target = "median"

    if mode == "small_lgnbkg":
        basedir = "small"
        duration = 10.0
    elif mode == "flat_wasser":
        # flat population (100 neurons for each model) with wasserstein distance
        basedir = "flat"
        duration = 100.0
    elif mode == "single":
        basedir = "single"
        duration = 100.0
    else:
        raise ValueError(
            "Unknown mode. Please choose from small_lgnbkg, flat_wasser, or single."
        )
    v1df = get_v1_dfs(basedir)
    if target == "mean":
        tfr = get_target_fr(basedir, target="target_mean_fr")
    elif target == "median":
        tfr = get_target_fr(basedir, target="target_median_fr")

    recurrent = False

    # based on Reinhold et al., 2015, we try to set the background so that the
    # spontaneous firing rates are 27% of the measured rates that include the LGN.
    if mode == "single":
        tfr = tfr * 0.27

    tfr.keys()[0]
    if mode == "flat_wasser":
        solvers = {nid: MinuitPipeSolver(0, 64, tfr[nid]) for nid in tfr.keys()}
    else:
        solvers = {nid: BisectionSolver(0, 256, tfr[nid]) for nid in tfr.keys()}
        # solvers = {nid: BisectionSolver(0, 16, tfr[nid]) for nid in tfr.keys()}

    weight = tfr.copy()
    weight[:] = 0.0
    weight.name = "syn_weight"

    for i in range(1000):
        if mode == "flat_wasser":
            update_bkg_weights_lognormal(basedir, weight)
        else:
            update_bkg_weights(basedir, weight)
        run_simulation(basedir, recurrent=recurrent, ncore=6)
        model_fr = get_model_fr(basedir, recurrent, duration=duration, target=target)

        # if new_weight does not exist, create it.
        if "new_weight" not in locals():
            new_weight = weight.copy()
        dd = pd.DataFrame([weight, new_weight, tfr, model_fr]).T
        # show the entire df
        with pd.option_context("display.max_rows", None):
            print(dd)

        new_weight = weight.copy()
        for i, v in model_fr.items():
            new_weight[i] = solvers[i].step(v)
            if new_weight[i] >= 0:
                weight[i] = new_weight[i]

        if all(new_weight < 0):  # solution achieved
            break

# %%

if False:
    basedir = "small"
    v1df = get_v1_dfs(basedir)
    tfr = get_target_fr(basedir)
    tfr_27 = tfr * 0.27

    tuned_fr = get_model_fr(basedir)
    tuned_fr_recurrent = get_model_fr(basedir, recurrent=True)

    # showing some results here

    tuned_fr
    tuned_fr_recurrent
# %%
if False:
    df = pd.DataFrame([tfr_27, tuned_fr]).T
    df = pd.DataFrame([tfr_27, tuned_fr_recurrent]).T
    plt.plot(df["target_mean_fr"], df["spike_rate"], "o")
    plt.plot([0, 3.5], [0, 3.5], "k--")
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.axis("image")

    tuned_fr.mean() / tuned_fr_recurrent.mean() * 0.27
    # %%
    from plotting_utils import plot_raster

    plot_raster("small/output_bkgtune_recurrent/config_bkgtune_recurrent.json")


# solvers
# model_fr


# %%

# model_fr
# run_simulation(basedir)
# df = get_spike_df(basedir)
# fr = get_model_fr(basedir)

# fr.plot()


# def func(x):
#     return x**2

# bs = BisectionSolver(0, 10, 4, 0.00001)
# x = 0
# for i in range(100):
#     print(x, val)
#     val = func(x)
#     x = bs.step(val)
#     if x < 0:
#         break
# # OK. it works.


# %% looking at the raster
# from plotting_utils import plot_raster

# plot_raster("small/output_bkgtune/config_bkgtune.json")
