# adjust the background weight distribution so that the resulting firing rate distribution
# matches the target firing rate distribution

# %% import modules
from multiprocessing import Pipe, Process
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import sonata.circuit
import scipy.stats as stats
import scipy.optimize
import shutil
import bkg_weight_adjustment as bwa
import os
from importlib import reload
import iminuit

reload(bwa)

# %% start testing simple things.

# first, let's see if the distribution can be generated as expected.

# lognormal
# v1 = stats.lognorm.rvs(0.5, loc=10, scale=15, size=10000)
# v2 = stats.lognorm.rvs(1, loc=10, scale=10, size=10000)
# plot them in log space
# bins = np.linspace(0, 3, 100)
# plt.hist(v1, bins=bins, alpha=0.5)
# plt.hist(v1, bins=20)
# plt.hist(v2, bins=bins, alpha=0.5)

# Great. These are right parameters for the lognormal distribution.

# %% next, let's come up with a way to rewrite the synaptic weights
# the s, loc, scale values will be stored in the edge_types file.

# so, in the new structure, the bkg edge types will contain the s, loc, scale values,
# and the synaptic weight will be generated using these values.

# firt, lets copy the original files to keep them safe.
def backup_files(basedir):
    # basedir = "small"
    bkg_edge_file = f"{basedir}/network/bkg_v1_edges.h5"
    bkg_edge_type_file = f"{basedir}/network/bkg_v1_edge_types.csv"
    bkg_edge_file_bkup = f"{bkg_edge_file}.bak"
    bkg_edge_type_file_bkup = f"{bkg_edge_type_file}.bak"
    # copy without overwriting
    if not os.path.exists(bkg_edge_file_bkup):
        shutil.copy2(bkg_edge_file, bkg_edge_file_bkup)
    else:
        print("bkg_edge_file_bkup already exists. skipping copy.")
    if not os.path.exists(bkg_edge_type_file_bkup):
        shutil.copy2(bkg_edge_type_file, bkg_edge_type_file_bkup)
    else:
        print("bkg_edge_type_file_bkup already exists. skipping copy.")
    return (
        bkg_edge_file,
        bkg_edge_file_bkup,
        bkg_edge_type_file,
        bkg_edge_type_file_bkup,
    )


# and let's open the bkup files.


# so, let's just copy the synaptic weights from the types file to the edges file.
def prepare_new_file_structure(
    bkg_edge_file, bkg_edge_file_bkup, bkg_edge_type_file, bkg_edge_type_file_bkup
):
    bkg = h5py.File(bkg_edge_file, "r+")
    bkg_types = pd.read_csv(bkg_edge_type_file_bkup, index_col="edge_type_id", sep=" ")

    edge_type_ids = bkg["edges/bkg_to_v1/edge_type_id"][:]
    # get the weights from the types file
    weights = bkg_types.loc[edge_type_ids, "syn_weight"].values
    # and write them to the edges file. (overwrite the variable if it already exists)
    if "syn_weight" in bkg["edges/bkg_to_v1/0"]:
        del bkg["edges/bkg_to_v1/0"]["syn_weight"]
    bkg["edges/bkg_to_v1/0/syn_weight"] = weights
    # bkg["edges/bkg_to_v1/0/syn_weight"] = weights
    bkg.close()

    # remove the syn_weight column from the types file.
    bkg_types_drop = bkg_types.drop("syn_weight", axis=1)
    bkg_types_drop.to_csv(bkg_edge_type_file, sep=" ")
    bkg_types.loc[:, "done"] = False
    return bkg_types, edge_type_ids


# OK. it works.


def write_weights(bkg_file, weights):
    bkg = h5py.File(bkg_file, "r+")
    # and write them to the edges file. (overwrite the variable if it already exists)
    if "syn_weight" in bkg["edges/bkg_to_v1/0"]:
        del bkg["edges/bkg_to_v1/0"]["syn_weight"]
    bkg["edges/bkg_to_v1/0/syn_weight"] = weights
    bkg.close()
    return


def set_init_params(bkg_types):
    # for each edge type, come up with the initial parameters.
    # loc can be the original weight of the df.
    bkg_types["loc"] = bkg_types["syn_weight"]
    bkg_types["s"] = 1
    bkg_types["scale"] = 10
    return bkg_types


def load_init_params(bkg_types, bkg_edge_type_file):
    # load the initial parameters from the file.
    bkg_types_loaded = pd.read_csv(
        bkg_edge_type_file, index_col="edge_type_id", sep=" "
    )
    bkg_types["loc"] = bkg_types_loaded["loc"]
    bkg_types["s"] = bkg_types_loaded["s"]
    bkg_types["scale"] = bkg_types_loaded["scale"]
    return bkg_types


def set_ncells(bkg_types, edge_type_ids):
    # count cells with the id in edge_type_ids, and set the number of cells
    # in the bkg_types dataframe.
    bkg_types["ncells"] = 0
    for edge_type_id in np.unique(edge_type_ids):
        bkg_types.loc[edge_type_id, "ncells"] = np.sum(edge_type_ids == edge_type_id)
    return bkg_types


def generate_new_weights(bkg_types, seed=0):
    # generate the weight distribution from the parameters.
    weights = []
    np.random.seed(seed)
    for edge_type_id in bkg_types.index:
        loc = bkg_types.loc[edge_type_id, "loc"]
        s = bkg_types.loc[edge_type_id, "s"]
        scale = bkg_types.loc[edge_type_id, "scale"]
        ncells = bkg_types.loc[edge_type_id, "ncells"]
        try:
            weights.append(stats.lognorm.rvs(s, loc=loc, scale=scale, size=ncells))
        except ValueError:
            print(f"ValueError for edge_type_id {edge_type_id}")
            print(f"loc: {loc}, s: {s}, scale: {scale}, ncells: {ncells}")
            print("falling back to no weight")
            weights.append(np.zeros(ncells))

    weights = np.concatenate(weights)
    return weights


def get_target_frs():
    # get target fr distribution from the neuropixels data
    npdf = pd.read_csv("neuropixels/metrics/OSI_DSI_DF.csv", index_col=0, sep=" ")

    # get spontaneous firing rate as an array for each cell
    arraydic = {}
    for cell_type in sorted(npdf.cell_type.dropna().unique()):
        arraydic[cell_type] = (
            npdf.loc[npdf.cell_type == cell_type, "Spont_Rate(Hz)"].dropna().values
        )

    for cell_type in arraydic.keys():
        print(f"{cell_type}: {len(arraydic[cell_type])}")

    return arraydic


def set_target_types(bkg_types, v1types):
    # identify the cell types of the target neurons
    # first, convert the target_query to the target node type in int
    bkg_types["target_type_id"] = bkg_types["target_query"].apply(
        lambda x: int(x.split("==")[1].strip("'"))  # drop the quotes
    )
    bkg_types["target_pop"] = v1types.loc[
        bkg_types["target_type_id"], "pop_name"
    ].values
    bkg_types["cell_type"] = bkg_types["target_pop"].apply(bwa.pop_name_to_cell_type)
    return bkg_types


def calculate_wasserstein(target_frs, model_frs, bkg_types):
    was_dist = {}
    for edge_type_id in bkg_types.index:
        cell_type = bkg_types.loc[edge_type_id, "cell_type"]
        target_type_id = bkg_types.loc[edge_type_id, "target_type_id"]
        was_dist[edge_type_id] = stats.wasserstein_distance(
            model_frs[target_type_id], target_frs[cell_type]
        )
    return was_dist


# solver-related functions
def get_from_main(params, conn):
    conn.send(params)
    value = conn.recv()
    return value


solution_dict = {}


def run_solver(inits, conn, id):
    solution = scipy.optimize.minimize(
        get_from_main,
        inits,
        args=(conn,),
        tol=1e-2,
        bounds=((0.01, 200), (0.01, 100), (0.01, 100)),  # bounds for loc, s, scale
        method="Nelder-Mead",
        options={"eps": 1e-2},
    )
    solution_dict[id] = solution
    return solution


def run_solver2(inits, conn, id):
    # this one uses iminuit
    solution = iminuit.minimize(
        get_from_main,
        inits,
        args=(conn,),
        tol=1e-3,
        bounds=((0.01, 200), (0.01, 100), (0.01, 100)),  # bounds for loc, s, scale
        method="simplex",
    )
    solution_dict[id] = solution
    return solution


class PipeSolver:
    def __init__(self, inits, id):
        self.parent_conn, self.child_conn = Pipe()
        self.process = Process(target=run_solver2, args=(inits, self.child_conn, id))
        self.process.start()

    def get_parameters(self):
        if self.process.is_alive():
            return self.parent_conn.recv()
        else:
            return None

    def give_loss(self, loss):
        self.parent_conn.send(loss)
        self.process.join(0.1)
        return


def get_params_all(solvers, bkg_types):
    params = {}
    for edge_type_id in bkg_types.index:
        params[edge_type_id] = solvers[edge_type_id].get_parameters()
    # also update the bkg_types dataframe
    for edge_type_id in bkg_types.index:
        if params[edge_type_id] is not None:
            bkg_types.loc[edge_type_id, "loc"] = params[edge_type_id][0]
            bkg_types.loc[edge_type_id, "s"] = params[edge_type_id][1]
            bkg_types.loc[edge_type_id, "scale"] = params[edge_type_id][2]
        else:
            bkg_types.loc[edge_type_id, "done"] = True

    return bkg_types


def set_loss_all(solvers, losses):
    for edge_type_id in losses.keys():
        solvers[edge_type_id].give_loss(losses[edge_type_id])
    return


# %%

# These are preparations
if __name__ == "__main__":
    set_init = True
    # set_init = False
    basedir = "flat"
    bkg_files = backup_files(basedir)
    bkg_types, edge_type_ids = prepare_new_file_structure(*bkg_files)
    bkg_types = set_init_params(bkg_types)
    if set_init:
        # init_file = basedir + "/network/bkg_v1_edge_types_fitted_v4_full_middle.csv"
        init_file = "precomputed_props/bkg_v1_edge_types_fitted_v4_full_done.csv"
        bkg_types = load_init_params(bkg_types, init_file)
    bkg_types = set_ncells(bkg_types, edge_type_ids)
    # identify the cell types of the target neurons
    v1types = pd.read_csv(basedir + "/network/v1_node_types.csv", index_col=0, sep=" ")
    bkg_types = set_target_types(bkg_types, v1types)

    # generate the weight distribution from the parameters.
    weights = generate_new_weights(bkg_types)
    target_frs = get_target_frs()

    # construct the solvers
    solvers = {}
    for edge_type_id in bkg_types.index:
        loc = bkg_types.loc[edge_type_id, "loc"]
        s = bkg_types.loc[edge_type_id, "s"]
        scale = bkg_types.loc[edge_type_id, "scale"]
        solvers[edge_type_id] = PipeSolver([loc, s, scale], edge_type_id)

    # the loop starts from here

    # model_frs = bwa.get_model_fr(basedir, target="array")
    # was_dist = calculate_wasserstein(target_frs, model_frs, bkg_types)

    for i in range(1000):
        bkg_types = get_params_all(solvers, bkg_types)
        weights = generate_new_weights(bkg_types)
        write_weights(bkg_files[0], weights)
        bkg_types.to_csv(basedir + "/network/bkg_v1_edge_types_fitted.csv", sep=" ")
        bwa.run_simulation(basedir, recurrent=False, ncore=8)
        model_frs = bwa.get_model_fr(basedir, target="array", duration=3.0)
        was_dist = calculate_wasserstein(target_frs, model_frs, bkg_types)

        # print the mean value
        print(i)
        print(np.mean(list(was_dist.values())))
        # also plot the results
        # plt.plot(list(was_dist.values()))

        # update the solver with new parameters
        set_loss_all(solvers, was_dist)
        # show how many processes are done
        print(f"Processes done: {bkg_types['done'].sum()}")
        # show the firing rates of the done processes

        # stop the loop if all the solvers are done
        if bkg_types["done"].all():
            break

# %%
