# This file contains functions helpful for understanding the network
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import json
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sonata
from math import exp

debug = False


def get_tau_syn(synaptic_folder="glif_models/synaptic_models/", double_alpha=True):
    tau_syn_dict = {}
    for file in Path(synaptic_folder).glob("*.json"):
        with open(file, "r") as f:
            if double_alpha:
                props = json.load(f)
                tau_syn_dict[file.name] = (
                    props["tau_syn"] + props["tau_syn_slow"] * props["amp_slow"]
                )
            else:
                tau_syn_dict[file.name] = json.load(f)["tau_syn"]
    return tau_syn_dict


def load_edges(basedir, src="v1", tgt="v1"):
    edgeh5 = h5py.File(f"{basedir}/network/{src}_{tgt}_edges.h5", "r")
    edges = {}
    edges["source_id"] = np.array(edgeh5[f"edges/{src}_to_{tgt}/source_node_id"])
    edges["target_id"] = np.array(edgeh5[f"edges/{src}_to_{tgt}/target_node_id"])
    if src == "v1":
        # if there is n_syns_ fine, if not, use nsyns.
        if f"edges/{src}_to_{tgt}/0/n_syns_" in edgeh5:
            edges["n_syns"] = np.array(edgeh5[f"edges/{src}_to_{tgt}/0/n_syns_"])
        elif f"edges/{src}_to_{tgt}/0/nsyns" in edgeh5:
            edges["n_syns"] = np.array(edgeh5[f"edges/{src}_to_{tgt}/0/nsyns"])
        else:  # if there is no n_syns_ or nsyns, make ones.
            print("n_syns_ or nsyns not found in edges file. Using ones.")
            edges["n_syns"] = np.ones(len(edges["source_id"]))
    else:
        # make ones like them.
        edges["n_syns"] = np.ones(len(edges["source_id"]))
    try:
        edges["syn_weight"] = np.array(edgeh5[f"edges/{src}_to_{tgt}/0/syn_weight"])
        use_types = False
    except:
        print("syn_weight not found in edges file. Trying to use types file...")
        use_types = True

    edges["edge_type_id"] = np.array(edgeh5[f"edges/{src}_to_{tgt}/edge_type_id"])
    edges["types"] = pd.read_csv(
        f"{basedir}/network/{src}_{tgt}_edge_types.csv",
        sep=" ",
        index_col="edge_type_id",
    )
    if use_types:
        # find syn_weight that correspond to the edge_type_id, and store it as
        # edges["syn_weight"]
        edges["syn_weight"] = edges["types"]["syn_weight"][edges["edge_type_id"]].values
        print("syn_weight is loaded from types file.")

    tau_syn_dict = get_tau_syn()
    edges["types"]["tau_syn"] = edges["types"]["dynamics_params"].map(tau_syn_dict)

    return edges


def get_all_tau_syn(edges):
    try:
        edge_type_tau_syn = edges["types"]["tau_syn"]
    except KeyError:
        print("tau_syn not found in the types file. Loading tau_syn_fast")
        edge_type_tau_syn = edges["types"]["tau_syn_fast"]
    all_tau_syn = (
        np.array(edge_type_tau_syn[edges["edge_type_id"]]) / 1000.0
    )  # convert to second
    return all_tau_syn


def pop_filter(nodes, pop_name, core_only=False):
    if pop_name == "lgn" or pop_name == "bkg":
        # pass all the cells
        return nodes["node_id"]
    pop_types = nodes["types"][
        nodes["types"]["pop_name"].str.contains(pop_name)
    ].index.values
    if core_only:
        return nodes["node_id"][
            np.isin(nodes["node_type_id"], pop_types) & nodes["core"]
        ]
    else:
        return nodes["node_id"][np.isin(nodes["node_type_id"], pop_types)]


def get_charge(
    edges, src_nodes, tgt_nodes, src_pops, tgt_pops, all_tau_syn, per_target=True
):
    src_ids = np.concatenate([pop_filter(src_nodes, pop) for pop in src_pops])
    tgt_ids = np.concatenate(
        [pop_filter(tgt_nodes, pop, core_only=True) for pop in tgt_pops]
    )
    source_edges = np.isin(edges["source_id"], src_ids)
    target_edges = np.isin(edges["target_id"], tgt_ids)
    contributing_edges = source_edges & target_edges
    n_syns = edges["n_syns"][contributing_edges]
    syn_weight = edges["syn_weight"][contributing_edges]
    tau_syn = all_tau_syn[contributing_edges]

    # the last e is needed because alpha function integral is tau * e
    charge = n_syns * syn_weight * tau_syn * exp(1)

    if debug:
        # show population information
        print("")
        print(f"Source populations: {src_pops}")
        print(f"Target populations: {tgt_pops}")
        # print the number of nodes
        print(f"Number of source nodes: {len(src_ids)}")
        print(f"Number of target nodes: {len(tgt_ids)}")
        # print the number of edges
        print(f"Number of total edges: {len(edges['source_id'])}")
        print(f"Number of source edges: {np.sum(source_edges)}")
        print(f"Number of target edges: {np.sum(target_edges)}")
        print(f"Number of edges: {np.sum(contributing_edges)}")
        # print mean n_syns
        print(f"Mean n_syns: {np.mean(n_syns)}")
        print(f"Mean syn_weight: {np.mean(syn_weight)}")
        print(f"Mean tau_syn: {np.mean(tau_syn) * 1000} (ms)")
        print(f"Mean total syn_weight: {np.mean(edges['syn_weight'])}")
        print(f"Mean total tau_syn: {np.mean(all_tau_syn)}")

    if per_target:
        return np.sum(charge) / len(tgt_ids)
    else:
        return np.sum(charge)


def get_infl_matrix(basedir, src_pops, tgt_pops, core_radius):
    if src_pops[0] == "lgn" or src_pops[0] == "bkg":
        src_loc = src_pops[0]
        src_nodes = load_nodes(basedir, src_loc)  # core radius is irrelevant
        tgt_nodes = load_nodes(basedir, "v1", core_radius)
    else:
        src_loc = "v1"
        src_nodes = load_nodes(basedir, "v1", core_radius)
        tgt_nodes = src_nodes
    edges = load_edges(basedir, src=src_loc, tgt="v1")
    all_tau_syn = get_all_tau_syn(edges)
    infl_matrix = np.zeros((len(src_pops), len(tgt_pops)))
    for i, src_pop in enumerate(src_pops):
        for j, tgt_pop in enumerate(tgt_pops):
            infl_matrix[i, j] = get_charge(
                edges, src_nodes, tgt_nodes, [src_pop], [tgt_pop], all_tau_syn
            )
    return infl_matrix


def filter_by_truthtable(edges, truthtable):
    # replace the edeges with the ones that are in the truthtable.
    n_edges = len(edges["source_id"])
    new_edges = {
        k: (v[truthtable] if len(v) == n_edges else v) for k, v in edges.items()
    }
    return new_edges


def load_nodes(basedir, loc="v1", core_radius=400):
    # nodeh5 = h5py.File(f"{basedir}/network/v1_nodes.h5", "r")
    # nodes = {}
    # nodes["node_id"] = np.array(nodeh5["nodes/v1/node_id"])
    # nodes["node_type_id"] = np.array(nodeh5["nodes/v1/node_type_id"])
    # nodes["x"] = np.array(nodeh5["nodes/v1/0/x"])
    # nodes["z"] = np.array(nodeh5["nodes/v1/0/z"])
    # nodes["tuning_angle"] = np.array(nodeh5["nodes/v1/0/tuning_angle"])
    # nodes["core"] = nodes["x"] ** 2 + nodes["z"] ** 2 < core_radius**2
    # nodes["types"] = pd.read_csv(f"{basedir}/network/v1_node_types.csv", sep=" ")

    # reproduce the code above replacing 'v1' with 'loc' using f-string
    nodeh5 = h5py.File(f"{basedir}/network/{loc}_nodes.h5", "r")
    nodes = {}
    nodes["node_id"] = np.array(nodeh5[f"nodes/{loc}/node_id"])
    nodes["node_type_id"] = np.array(nodeh5[f"nodes/{loc}/node_type_id"])
    if loc == "v1":
        nodes["x"] = np.array(nodeh5[f"nodes/{loc}/0/x"])
        nodes["z"] = np.array(nodeh5[f"nodes/{loc}/0/z"])
        nodes["tuning_angle"] = np.array(nodeh5["nodes/v1/0/tuning_angle"])
        nodes["core"] = nodes["x"] ** 2 + nodes["z"] ** 2 < core_radius**2
    nodes["types"] = pd.read_csv(
        f"{basedir}/network/{loc}_node_types.csv", sep=" ", index_col="node_type_id"
    )
    return nodes


def angle_difference(angle1, angle2, mode="orientation"):
    # Normalize angles to the range [0, 360)
    angle1 = angle1 % 360
    angle2 = angle2 % 360

    # Calculate absolute difference
    diff = np.abs(angle1 - angle2)
    # Adjust differences for direction mode
    if mode == "direction":
        direction_diff = np.where(diff > 180, 360 - diff, diff)
        return direction_diff

    # Calculate orientation difference
    elif mode == "orientation":
        orientation_diff = diff % 180
        orientation_diff = np.where(
            orientation_diff > 90, 180 - orientation_diff, orientation_diff
        )
        return orientation_diff
    else:
        raise ValueError("Mode must be 'orientation' or 'direction'")


def get_delta_theta(type_name, edges, nodes, billeh=False):
    # pick up edge type ids from the dataframe
    if billeh:
        # parse the type_name to get the source and target populations.
        src_pop, tgt_pop = type_name.split("_to_")
        tgt_pop = tgt_pop.split(".")[0]  # remove the .json part
        # if the source_query contains the named string, it is selected.
        src_selected_ids = (
            edges["types"]
            .query(f"source_query.str.contains('{src_pop}')", engine="python")
            .index
        )
        # for targets, consult the nodes first
        tgt_type_ids = nodes["types"].index[
            nodes["types"]["pop_name"].str.contains(tgt_pop)
        ]
        # convert target_query to an array of target node type ids
        target_type_ids = (
            edges["types"]["target_query"].str.extract(r"(\d+)").astype(int)
        )
        selected_target_ids = np.isin(target_type_ids, tgt_type_ids)

        tgt_selected_ids = edges["types"][selected_target_ids].index
        all_selected = np.isin(edges["edge_type_id"], src_selected_ids) & np.isin(
            edges["edge_type_id"], tgt_selected_ids
        )

    else:
        etype_ids = edges["types"].query(f"dynamics_params == '{type_name}'").index
        all_selected = np.isin(edges["edge_type_id"], etype_ids)
    # all_selected.sum()
    new_edges = filter_by_truthtable(edges, all_selected)

    # calculate two versions of delta theta (orientation difference and direction difference)
    pre_theta = nodes["tuning_angle"][new_edges["source_id"]]
    post_theta = nodes["tuning_angle"][new_edges["target_id"]]

    #
    delta_theta_ori = angle_difference(pre_theta, post_theta, mode="orientation")
    delta_theta_dir = angle_difference(pre_theta, post_theta, mode="direction")

    return delta_theta_ori, delta_theta_dir, new_edges


def plot_delta_theta(delta_theta, network_name, ax=None):
    # plotting the direction one is actually sufficient.
    hp = sns.histplot(delta_theta, bins=np.arange(0, 181, 5), stat="density", ax=ax)
    # put x axis label
    hp.set_xlabel("Direction difference (degrees)")
    hp.set_title(network_name)
    return hp


# for each segment of the delta theta, calculate the average weight.
def block_ave_weights(delta_theta, weights, brange):
    # range is expected to be like np.arange(0, 181, 5)
    # first, find out which bin each delta_theta belongs to.
    bin_inds = np.digitize(delta_theta, brange)
    # then, calculate the average weights for each bin.
    ave_weights = np.array([weights[bin_inds == i].mean() for i in range(len(brange))])
    # do the sem as well
    sem_weights = np.array(
        [
            weights[bin_inds == i].std() / np.sqrt((bin_inds == i).sum())
            for i in range(len(brange))
        ]
    )
    return ave_weights, sem_weights


def plot_block_ave_weights(ave_weights, sem_weights, range, ax=None):
    # plot the average weights as a function of delta theta.
    ax.errorbar(range, ave_weights, yerr=sem_weights, fmt="o")
    ax.set_xlabel("Direction difference (degrees)")
    ax.set_ylabel("Average synaptic weight")
    return ax


def plot_rossi_one(nodes, edges, seed=None, core=True):
    if seed is not None:
        np.random.seed(seed)
    e4_type = nodes["types"]["pop_name"].str.contains("e4")
    e4_cells = e4_type[nodes["node_type_id"]]

    if core:
        pick_nodes = nodes["core"]
    else:
        pick_nodes = ~nodes["core"]

    id_e4_core = np.where(e4_cells & pick_nodes)[0]
    rand_e4_core = np.random.choice(id_e4_core)

    # find edges that connects to this neuron
    target_edges = edges["target_id"] == rand_e4_core
    source_nodes = edges["source_id"][target_edges]
    print(f"Number of connections: {source_nodes.size}")
    source_ei = nodes["types"]["ei"][nodes["node_type_id"][source_nodes]]
    source_nodes_e = source_nodes[source_ei == "e"]
    source_nodes_i = source_nodes[source_ei == "i"]

    #  visulalize x and z of them.
    source_nodes_e.size
    source_nodes_i.size
    x_e = nodes["x"][source_nodes_e]
    z_e = nodes["z"][source_nodes_e]
    x_i = nodes["x"][source_nodes_i]
    z_i = nodes["z"][source_nodes_i]
    fig, ax = plt.subplots()
    ax.scatter(x_e, z_e, label="excitatory", alpha=0.2, color="r", s=10)
    ax.scatter(x_i, z_i, label="inhibitory", alpha=0.2, color="b", s=10)

    # also highlight the position of the target neuron with x sign.
    ax.scatter(
        nodes["x"][rand_e4_core], nodes["z"][rand_e4_core], marker="x", s=100, color="k"
    )

    nodes["x"][rand_e4_core], nodes["z"][rand_e4_core]
    nodes["core"][rand_e4_core]
    tuning_angle = nodes["tuning_angle"][rand_e4_core]

    # make the axis equal, and adjust the x and z limits to pm 400.
    ax.axis("square")
    ax.set_title(f"Source nodes to #{rand_e4_core} (tuning angle {tuning_angle})")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")
    ax.legend()

    # also draw a big circle with 400 um radius
    circle = plt.Circle((0, 0), 400, fill=False, color="k", alpha=0.4)
    ax.add_artist(circle)
    circle = plt.Circle((0, 0), 200, fill=False, color="k", alpha=0.4)
    ax.add_artist(circle)
    ax.set_xlim([-410, 410])
    ax.set_ylim([-410, 410])

    return fig


def infer_core_radius(basedir):
    # if basedir contains 'full', 400
    # if basedir contains 'core', 200
    # if basedir contains 'small', 100
    # otherwise, just say 200...

    if "full" in basedir:
        return 400
    elif "core" in basedir:
        return 200
    elif "small" in basedir:
        return 100
    else:
        return 200


# class ReccurentNetwork(object):
#     def __init__(self, basedir, core_radius=200):
#         self.basedir = basedir
#         self.nodes = load_nodes(basedir, "v1", core_radius)
#         self.edges = load_edges(basedir, "v1", "v1")


#     def


class SonataNetwork(object):
    def __init__(self, basedir, exclude=[]):
        self.basedir = basedir
        self.exclude = exclude

    def detect_networks(self):
        # search files in the {base}/network directory and find out the networks.
        # in the network folder, look for files that end with nodes.h5 and pick up the
        # names
        file_list = Path(self.basedir).glob("network/*_nodes.h5")
        network_names = [f.stem.split("_")[0] for f in file_list]
        # if any of the network names are in the exclude list, remove them.
        network_names = [n for n in network_names if n not in self.exclude]
        return network_names

    def node_file_names(self):
        # append "{base}/networks/{network}_nodes.h5" for each network.
        names = [f"{self.basedir}/network/{n}_nodes.h5" for n in self.networks]
        return names

    def node_type_file_names(self):
        # append "{base}/networks/{network}_node_types.csv" for each network.
        names = [f"{self.basedir}/network/{n}_node_types.csv" for n in self.networks]
        return names

    def edge_file_names(self):
        # append "{base}/networks/{network}_{network}_edges.h5" for each network.
        # but only those go into v1 is valid.
        names = [f"{self.basedir}/network/{n}_v1_edges.h5" for n in self.networks]
        return names

    def edge_type_file_names(self):
        # append "{base}/networks/{network}_{network}_edge_types.csv" for each network.
        # but only those go into v1 is valid.
        names = [f"{self.basedir}/network/{n}_v1_edge_types.csv" for n in self.networks]
        return names

    def load_sonata_network(self):
        data_files = self.node_file_names() + self.edge_file_names()
        data_type_files = self.node_type_file_names() + self.edge_type_file_names()
        return sonata.circuit.File(data_files, data_type_files)

    @property
    def net(self):
        if not hasattr(self, "_net"):
            self._net = self.load_sonata_network()
        return self._net

    @property
    def networks(self):
        # a list of strings that are the names of the networks.
        # e.g. ["v1", "lgn", "bkg"]
        if not hasattr(self, "_networks"):
            self._networks = self.detect_networks()
        return self._networks


# self test in the end
# if __name__ == "__main__":
