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


def sbatch_boilerplate(file, logdir, config_counts, memory=32, jobs=1, threads=8):
    time = "2:00:00" if (jobs * threads) <= 4 else "1:00:00"

    file.write("#!/bin/bash\n")
    file.write("#SBATCH --partition=braintv\n")
    # if full_memory:  # full model needs a lot of memory
    #     file.write("#SBATCH -N1 -c1 -n4\n")
    #     file.write("#SBATCH --mem-per-cpu=60G\n")
    #     file.write("#SBATCH -t2:00:00\n")
    # else:
    #     file.write("#SBATCH -N1 -c1 -n8\n")
    #     file.write("#SBATCH --mem-per-cpu=15G\n")
    #     file.write("#SBATCH -t1:00:00\n")
    file.write(f"#SBATCH -N1 -c1 -n{jobs * threads}\n")
    # file.write(f"#SBATCH --mem-per-cpu={memory}G\n")
    # now better to specify the total memory
    file.write(f"#SBATCH --mem={memory}G\n")
    file.write(f"#SBATCH -t{time}\n")

    file.write("#SBATCH --qos=braintv\n")
    file.write(f"#SBATCH --output={logdir}/slurm-%A_%a.out\n")
    file.write(f"#SBATCH --error={logdir}/slurm-%A_%a.err\n")
    # array ID range includes both edges
    file.write(f"#SBATCH --array=0-{config_counts-1}\n\n")
    file.write(
        # "source /allen/programs/mindscope/workgroups/realistic-model/shinya.ito/activate_custom_nest_sdk_develop.sh\n"
        "source /allen/programs/mindscope/workgroups/realistic-model/shinya.ito/miniconda3/bin/activate new_v1\n"
    )
    # file.write(
    #     "module use /allen/programs/braintv/workgroups/modelingsdk/modulefiles\n"
    # )
    # # activating double alpha version of nest.
    # file.write(
    #     "module load anaconda/22.10 cmake/3.25.2 gcc/10.1.0-centos7 openmpi/4.1.5-gcc10\n"
    # )
    # file.write(
    #     "conda activate /allen/programs/mindscope/workgroups/realistic-model/shinya.ito/miniconda3/envs/custom_nest\n"
    # )
    # file.write(
    #     "source /allen/programs/mindscope/workgroups/realistic-model/shinya.ito/custom_nest/nest-simulator-3.4/build/bin/nest_vars.sh\n"
    # )
    # old code
    # file.write("module load mpich/3.4.1-slurm\n")
    # file.write("module load conda/4.5.4\n")
    # f.write("module load nest/2.20.1-py37-slurm\n")
    # f.write("source activate v1_glif_modeling\n")
    # file.write("source activate bmtk-latest-py37-slurm\n")


def memory_jobs(basedir):
    if "full" in basedir:
        memory = 120
        jobs = 1
        threads = 8
    elif "forty" in basedir:
        memory = 60
        jobs = 1
        threads = 8
    elif ("core" in basedir) or ("twenty" in basedir):
        memory = 30
        jobs = 1
        threads = 8
    else:  # including small, tiny, etc.
        memory = 10
        jobs = 1
        threads = 8
    return memory, jobs, threads


def write_job(basedir, config_counts, modfile, job_name, arg_memory):
    configdir = basedir + f"/configs/{job_name}"
    jobdir = basedir + "/jobs"
    logdir = jobdir + "/logs"

    with open(jobdir + f"/{job_name}.sh", "w") as f:
        memory, jobs, threads = memory_jobs(basedir)
        # override memory and jobs
        memory = arg_memory
        sbatch_boilerplate(
            f, logdir, config_counts, memory=memory, jobs=jobs, threads=threads
        )
        # f.write("module load nest/2.20.1-py37-slurm\n")

        config_array = configdir + "/config_$SLURM_ARRAY_TASK_ID.json"
        if modfile is not None:
            f.write(
                # f"srun --mpi=pmi2 python run_pointnet.py {config_array} -m {modfile}"
                # f"mpirun -np {jobs} python run_pointnet.py {config_array} -m {modfile}"
                f"python run_pointnet.py {config_array} -m {modfile} -n {threads}"
            )
        else:
            # f.write(f"srun --mpi=pmi2 python run_pointnet.py {config_array}")
            f.write(f"python run_pointnet.py {config_array} -n {threads}")


def write_filternet_job(basedir, config_counts, job_name, arg_memory):
    configdir = basedir + f"/configs/filternet_{job_name}"
    jobdir = basedir + "/jobs"
    logdir = jobdir + "/logs"

    with open(jobdir + f"/filternet_{job_name}.sh", "w") as f:
        memory, jobs, cores = memory_jobs(basedir)
        sbatch_boilerplate(f, logdir, config_counts, memory=arg_memory)
        # f.write("module load nest/2.20.1-py37-slurm\n")

        config_array = configdir + "/config_filternet_$SLURM_ARRAY_TASK_ID.json"
        # f.write(f"srun --mpi=pmi2 python run_filternet.py {config_array}")
        f.write(
            f"mpirun --oversubscribe -np {jobs * cores} python run_filternet.py {config_array}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Making a cluster-ready job file.")
    parser.add_argument("basedir", type=str)
    parser.add_argument("--filternet", action="store_true")
    parser.add_argument(
        "-m", "--modfile", type=str, default=None, help="The modulation file to use."
    )
    parser.add_argument(
        "--memory", type=int, default=20, help="The memory to use in GB."
    )
    parser.add_argument(
        "--network_option",
        type=str,
        default="plain",
        help="The network option. (plain, adjusted, checkpoint)",
    )
    args = parser.parse_args()

    filterdir = args.basedir + "/filternet_8dir_10trials"
    if args.filternet:
        base_config = args.basedir + "/configs/config_filternet.json"
        configdir = args.basedir + "/configs/filternet_8dir_10trials"
    else:
        base_config = args.basedir + "/configs/config_plain.json"
        configdir = args.basedir + f"/configs/8dir_10trials_{args.network_option}"

    jobdir = args.basedir + "/jobs"
    logdir = args.basedir + "/jobs/logs"
    pathlib.Path(configdir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

    # take the base config file
    js = json.load(open(base_config))

    # define necessary elements
    if args.filternet:
        job_name = "8dir_10trials"
    else:
        job_name = f"8dir_10trials_{args.network_option}"

    angles = np.linspace(0, 315, 8)
    trials = range(10)
    config_counts = 0

    for angle in angles:
        for trial in trials:
            filterdir_indv = (
                f"$BASE_DIR/filternet_8dir_10trials/angle{int(angle)}_trial{trial}"
            )
            outdir_indv = f"$BASE_DIR/{job_name}/angle{int(angle)}_trial{trial}"

            js["manifest"]["$BASE_DIR"] = "${configdir}/../.."
            if args.filternet:
                js["inputs"]["LGN_spikes"]["theta"] = angle
                js["inputs"]["LGN_spikes"]["trial"] = trial
                js["manifest"]["$OUTPUT_DIR"] = filterdir_indv
                config_name = configdir + f"/config_filternet_{config_counts}.json"
            else:  # main job
                js["manifest"]["$LGNINPUT_DIR"] = filterdir_indv
                js["manifest"]["$OUTPUT_DIR"] = outdir_indv

                # change the edge file if not plain
                if args.network_option != "plain":
                    js["networks"]["edges"][0][
                        "edges_file"
                    ] = f"$NETWORK_DIR/v1_v1_edges_{args.network_option}.h5"
                # also change the bkg for TF checkpoint
                if args.network_option == "checkpoint":
                    js["networks"]["edges"][2][
                        "edges_file"
                    ] = f"$NETWORK_DIR/bkg_v1_edges_checkpoint.h5"

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
        write_filternet_job(args.basedir, config_counts, job_name, args.memory)
    else:
        write_job(args.basedir, config_counts, args.modfile, job_name, args.memory)
