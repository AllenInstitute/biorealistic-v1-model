# %%

import json
import numpy as np
import argparse
import os
import make_osi_jobs as moj
import pathlib
import stimulus_trials as st

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Making a cluster-ready contrast job files."
    )
    parser.add_argument("basedir", type=str)
    parser.add_argument("--filternet", action="store_true")
    parser.add_argument(
        "-m", "--modfile", type=str, default=None, help="The modulation file to use."
    )
    parser.add_argument(
        "--memory", type=int, default=20, help="The memory to use in GB."
    )

    args = parser.parse_args()

    filterdir = args.basedir + "/filternet_contrasts"
    if args.filternet:
        base_config = args.basedir + "/configs/config_filternet.json"
        outdir = filterdir
        configdir = args.basedir + "/configs/filternet_contrasts"
    else:
        base_config = args.basedir + "/configs/config.json"
        outdir = args.basedir + "/output_contrasts"
        configdir = args.basedir + "/configs/contrasts"

    jobdir = args.basedir + "/jobs"
    logdir = args.basedir + "/jobs/logs"

    pathlib.Path(configdir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

    # take the base config file
    js = json.load(open(base_config))

    # define necessary elements
    job_name = "contrasts"
    # contrasts = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]  # based on Millman et al. 2020
    # angles = np.linspace(0, 315, 8)
    # trials = range(10)  # for now to save time
    stim_iter = st.ContrastStimulus()
    config_counts = 0
    for angle, contrast, trial in stim_iter:
        filterdir_indv = f"$BASE_DIR/filternet_contrasts/angle{int(angle)}_contrast{contrast}_trial{trial}"
        outdir_indv = (
            f"$BASE_DIR/contrasts/angle{int(angle)}_contrast{contrast}_trial{trial}"
        )

        js["manifest"]["$BASE_DIR"] = "${configdir}/../.."
        if args.filternet:
            js["inputs"]["LGN_spikes"]["theta"] = angle
            js["inputs"]["LGN_spikes"]["contrast"] = contrast
            js["inputs"]["LGN_spikes"]["trial"] = trial
            # important to match TF with Millman et al. 2020
            js["inputs"]["LGN_spikes"]["temroral_f"] = 1.0
            # randomize the phase for each trial
            js["inputs"]["LGN_spikes"]["phase"] = np.random.rand() * 360
            js["manifest"]["$OUTPUT_DIR"] = filterdir_indv
            config_name = configdir + f"/config_filternet_{config_counts}.json"
        else:  # main job
            js["manifest"]["$LGNINPUT_DIR"] = filterdir_indv
            js["manifest"]["$OUTPUT_DIR"] = outdir_indv
            config_name = configdir + f"/config_{config_counts}.json"
            # if the config file contains background input, change the input file
            if "$BKGINPUT_DIR" in js["manifest"].keys():
                bkgdir_indv = f"$BASE_DIR/bkg_contrasts/angle{int(angle)}_contrast{contrast}_trial{trial}"
                js["manifest"]["$BKGINPUT_DIR"] = bkgdir_indv
        config_counts += 1

        json.dump(js, open(config_name, "w"), indent=2)

    if args.filternet:
        moj.write_filternet_job(args.basedir, config_counts, job_name, args.memory)
    else:
        moj.write_job(args.basedir, config_counts, args.modfile, job_name, args.memory)
