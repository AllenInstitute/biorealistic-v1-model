# %%

"""
the goal is to make the cluter running as easy as possible, while permitting necessary
manipulations of the parameters.

Eventually, the command to run is like this
mpirun -np 8 python run_filternet.py <config_file>
or
mpirun -np 8 python run_pointnet.py <config_file>
8 should be changed to an appropriate number of cores (using a big number may not be helpful)

These commands should be stored in the job file


Let's break down into steps

* Get the base config and identify parameters
* Write down config files for each operation (config file number should be a serial number)
* Write into a job file (single job files)

* I'd like to make this compatible with both filternet simulations and main simulations


"""

import json
import numpy as np
import argparse
import pathlib


def sbatch_boilerplate(file, logdir, config_counts, full_memory=False):
    file.write("#!/bin/bash\n")
    file.write("#SBATCH --partition=braintv\n")
    if full_memory:  # full model needs a lot of memory
        file.write("#SBATCH -N1 -c1 -n4\n")
        file.write("#SBATCH --mem-per-cpu=60G\n")
        file.write("#SBATCH -t2:00:00\n")
    else:
        file.write("#SBATCH -N1 -c1 -n8\n")
        file.write("#SBATCH --mem-per-cpu=4G\n")
        file.write("#SBATCH -t1:00:00\n")
    file.write("#SBATCH --qos=braintv\n")
    file.write(f"#SBATCH --output={logdir}/slurm-%A_%a.out\n")
    file.write(f"#SBATCH --error={logdir}/slurm-%A_%a.err\n")
    # array ID range includes both edges
    file.write(f"#SBATCH --array=0-{config_counts-1}\n\n")
    file.write(
        "module use /allen/programs/braintv/workgroups/modelingsdk/modulefiles\n"
    )
    file.write("module load mpich/3.4.1-slurm\n")
    file.write("module load conda/4.5.4\n")
    # f.write("module load nest/2.20.1-py37-slurm\n")
    # f.write("source activate v1_glif_modeling\n")
    file.write("source activate bmtk-latest-py37-slurm\n")


def write_job(basedir, config_counts):
    configdir = basedir + "/configs/8dir_10trials"
    jobdir = basedir + "/jobs"
    logdir = jobdir + "/logs"

    with open(jobdir + "/8dir_10trials.sh", "w") as f:
        sbatch_boilerplate(f, logdir, config_counts, full_memory=("full" in basedir))
        f.write("module load nest/2.20.1-py37-slurm\n")

        config_array = configdir + "/config_$SLURM_ARRAY_TASK_ID.json"
        f.write(f"srun --mpi=pmi2 python run_pointnet.py {config_array}")


def write_filternet_job(basedir, config_counts):
    configdir = basedir + "/configs/filternet_8dir_10trials"
    jobdir = basedir + "/jobs"
    logdir = jobdir + "/logs"

    with open(jobdir + "/filternet_8dir_10trials.sh", "w") as f:
        sbatch_boilerplate(f, logdir, config_counts)
        # f.write("module load nest/2.20.1-py37-slurm\n")

        config_array = configdir + "/config_filternet_$SLURM_ARRAY_TASK_ID.json"
        f.write(f"srun --mpi=pmi2 python run_filternet.py {config_array}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Making a cluster-ready job file.")
    parser.add_argument("basedir", type=str)
    parser.add_argument("--filternet", action="store_true")
    args = parser.parse_args()

    filterdir = args.basedir + "/filternet_8dir_10trials"
    if args.filternet:
        base_config = args.basedir + "/configs/config_filternet.json"
        outdir = filterdir
        configdir = args.basedir + "/configs/filternet_8dir_10trials"
    else:
        base_config = args.basedir + "/configs/config.json"
        outdir = args.basedir + "/output_8dir_10trials"
        configdir = args.basedir + "/configs/8dir_10trials"

    jobdir = args.basedir + "/jobs"
    logdir = args.basedir + "/jobs/logs"
    pathlib.Path(configdir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

    # take the base config file
    js = json.load(open(base_config))

    # define necessary elements
    angles = np.linspace(0, 315, 8)
    trials = range(10)
    config_counts = 0

    for angle in angles:
        for trial in trials:
            filterdir_indv = (
                f"$BASE_DIR/filternet_8dir_10trials/angle{int(angle)}_trial{trial}"
            )
            outdir_indv = f"$BASE_DIR/8dir_10trials/angle{int(angle)}_trial{trial}"

            js["manifest"]["$BASE_DIR"] = "${configdir}/../.."
            if args.filternet:
                js["inputs"]["LGN_spikes"]["theta"] = angle
                js["inputs"]["LGN_spikes"]["trial"] = trial
                js["manifest"]["$OUTPUT_DIR"] = filterdir_indv
                config_name = configdir + f"/config_filternet_{config_counts}.json"
            else:  # main job
                js["manifest"]["$LGNINPUT_DIR"] = filterdir_indv
                js["manifest"]["$OUTPUT_DIR"] = outdir_indv
                config_name = configdir + f"/config_{config_counts}.json"
                # if the config file contains background input, change the input file
                if "$BKGINPUT_DIR" in js["manifest"].keys():
                    bkgdir_indv = (
                        f"$BASE_DIR/bkg_8dir_10trials/angle{int(angle)}_trial{trial}"
                    )
                    js["manifest"]["$BKGINPUT_DIR"] = bkgdir_indv
            config_counts += 1

            json.dump(js, open(config_name, "w"), indent=2)

    if args.filternet:
        write_filternet_job(args.basedir, config_counts)
    else:
        write_job(args.basedir, config_counts)
