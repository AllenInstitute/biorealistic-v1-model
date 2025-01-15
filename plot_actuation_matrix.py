import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import argparse
from math import exp
from pathlib import Path

import network_utils as nu

e = exp(1)


def get_rheobase(pops):
    rheo_df = pd.read_csv("Rheobase_slope.csv", index_col=0)
    # for example, if the pop_name is "i4Pvalb", the groupname is "i4P".
    rheo_df["group"] = rheo_df.pop_name.map(lambda x: [pop for pop in pops if pop in x])
    # if the group is empty, change it to nan, otherwise, take the element of it.
    rheo_df["group"] = rheo_df.group.map(lambda x: x[0] if len(x) > 0 else np.nan)
    rheo = rheo_df.groupby("group").mean(numeric_only=True)["rheobase"]
    return rheo


if __name__ == "__main__":
    # parse the argument.
    # argument is the basedir and the core radius.
    # core radius is optional (default, None (auto recognition))
    parser = argparse.ArgumentParser(description="Plot the actuation matrix")
    parser.add_argument("basedir", type=str)
    parser.add_argument("-c", "--core_radius", type=float, default=None)
    parser.add_argument("--swap_etit", action="store_true")
    args = parser.parse_args()

    if args.core_radius is None:
        args.core_radius = nu.infer_core_radius(args.basedir)

    pops = [
        "i1H",
        "e2",
        "i23P",
        "i23S",
        "i23V",
        "e4",
        "i4P",
        "i4S",
        "i4V",
        "e5ET",
        "e5IT",
        "e5NP",
        "i5P",
        "i5S",
        "i5V",
        "e6",
        "i6P",
        "i6S",
        "i6V",
    ]
    print("Getting the influence matrix for LGN")
    infl_lgn = nu.get_infl_matrix(
        args.basedir,
        ["lgn"],
        pops,
        core_radius=args.core_radius,
    )  # unit is pC per spike

    print("Getting the influence matrix for recurrent connections")
    infl_rec = nu.get_infl_matrix(
        args.basedir, pops, pops, core_radius=args.core_radius
    )
    print("Getting the influence matrix for background")
    infl_bkg = nu.get_infl_matrix(
        args.basedir, ["bkg"], pops, core_radius=args.core_radius
    )

    print("Constructing the actuation matrix")
    df_bkg = pd.DataFrame(infl_bkg, index=["bkg"], columns=pops)
    df_lgn = pd.DataFrame(infl_lgn, index=["lgn"], columns=pops)
    df_rec = pd.DataFrame(infl_rec, index=pops, columns=pops)
    df_all = pd.concat([df_bkg, df_lgn, df_rec])

    # converting to actuation matrix

    npdf = pd.read_csv("neuropixels/metrics/OSI_DSI_DF_data.csv", sep=" ")

    sp_rates = npdf.groupby("cell_type").mean()["Avg_Rate(Hz)"]
    # replacing the white space with _ in case the OSI_DSI_DF is old.
    sp_rates.index = sp_rates.index.str.replace(" ", "_")

    # add LGN and bkg firing rates. LGN is mixture of the spont (1/3) and
    # stimulated (2/3) periods.
    lgn_rate = 3.84 / 3 + 6.13 / 3 * 2
    bkg_rate = 250  # new 100 bkg rate
    df_all.shape

    # duplicate L5 FR in sp_rates.
    # delete 'L5 Exc' row and replace it with L5 IT, L5 PT, and L5 NP.
    l5e = sp_rates["L5_Exc"]
    l5e_index = sp_rates.index.get_loc("L5_Exc")
    sp_rates = sp_rates.drop("L5_Exc")

    # insert the new values.
    sp_rates = pd.concat(
        [sp_rates, pd.Series([l5e, l5e, l5e], index=["L5_ET", "L5_IT", "L5_NP"])]
    )

    # sort the index.
    sp_rates = sp_rates.sort_index()

    # add in the lgn and bkg
    sp_rates = pd.concat(
        [pd.Series([bkg_rate, lgn_rate], index=["bkg", "lgn"]), sp_rates]
    )

    # convert to actuation matrix in pA.
    act_mat = df_all * sp_rates.values[:, None]

    # get the rheobase from the file
    rheo = get_rheobase(pops)

    # append the rheobase to the actuation matrix
    act_mat_t = act_mat.T
    act_mat_t.insert(0, "rheobase", rheo)
    act_mat = act_mat_t.T

    if args.swap_etit:
        act_mat.iloc[[12, 13]] = act_mat.iloc[[13, 12]].values
        act_mat.iloc[:, [9, 10]] = act_mat.iloc[:, [10, 9]].values
        # swap labels e5IT to e5ET and e5ET to e5IT
        act_mat = act_mat.rename(
            index={"e5IT": "e5ET", "e5ET": "e5IT"},
            columns={"e5IT": "e5ET", "e5ET": "e5IT"},
        )

    print("Plotting the actuation matrix")
    plt.figure(figsize=(12, 12))
    sns.heatmap(
        act_mat, annot=True, fmt=".1f", cmap="coolwarm", center=0, vmin=-100, vmax=100
    )
    # do not rotate the yticklabels
    plt.yticks(rotation=0)
    plt.xlabel("Target population")
    plt.ylabel("Source population")
    plt.title("Actuation matrix (average rates; pA)")
    # place the white between the layers.
    # for h, at [2, 6, 10, 14]. For v, at [4, 8, 12]
    for h in [1, 3, 4, 8, 12, 18]:
        plt.axhline(h, color="w", linewidth=2)
    for v in [1, 5, 9, 15]:
        plt.axvline(v, color="w", linewidth=2)

    plt.axis("image")
    plt.tight_layout()
    # make figures dir if not exists
    Path(f"{args.basedir}/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{args.basedir}/figures/actuation_matrix.pdf")
    # let's also save the actuation matrix itself.
    Path(f"{args.basedir}/metrics").mkdir(parents=True, exist_ok=True)
    act_mat.to_csv(f"{args.basedir}/metrics/actuation_matrix.csv", sep=" ")

    print("Done")
