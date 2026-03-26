import argparse
import json
import pathlib

import numpy as np
import make_filternet_jobs as mfj


def write_job(basedir, config_name, config_counts, full_memory):
    configdir = basedir + "/configs/" + config_name
    jobdir = basedir + "/jobs"
    logdir = jobdir + "/logs"

    with open(jobdir + "/" + config_name + ".sh", "w") as f:
        mfj.sbatch_boilerplate(f, logdir, config_counts, full_memory=full_memory)
        f.write("module load nest/2.20.1-py37-slurm\n")

        config_array = configdir + "/config_$SLURM_ARRAY_TASK_ID.json"
        f.write(f"srun --mpi=pmi2 python run_pointnet.py {config_array}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Making a cluster-ready job file.")
    parser.add_argument("basedir", type=str)
    args = parser.parse_args()
    project = "spont_lgn_5s"

    filterdir = args.basedir + f"/../scaled_{project}"
    base_config = args.basedir + "/configs/config.json"
    outdir = args.basedir + f"/output_{project}"
    configdir = args.basedir + f"/configs/{project}"

    jobdir = args.basedir + "/jobs"
    logdir = args.basedir + "/jobs/logs"
    pathlib.Path(configdir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)

    js = json.load(open(base_config))

    # one should be able to change any element here, in this case, LGN FR.
    lgn_frs = np.arange(9.0, 30.0, 1.0)
    config_counts = 0

    for lgn_fr in lgn_frs:
        js["manifest"]["BASE_DIR"] = "${configdir}/../.."

        outdir_indv = f"$BASE_DIR/output_{project}/lgn_fr_{lgn_fr:.1f}Hz"
        js["manifest"]["$OUTPUT_DIR"] = outdir_indv
        js["manifest"]["$LGNINPUT_DIR"] = f"${{BASE_DIR}}/../{filterdir}"
        js["inputs"]["LGN_spikes"]["module"] = "csv"
        js["inputs"]["LGN_spikes"][
            "input_file"
        ] = f"$LGNINPUT_DIR/{lgn_fr:.1f}_Hz_spikes.csv"
        js["inputs"]["BKG_spikes"][
            "input_file"
        ] = f"$BKGINPUT_DIR/bkg_spikes_full_10s.h5"

        json.dump(js, open(configdir + f"/config_{config_counts}.json", "w"))
        config_counts += 1

    # check if the basedir path contains full
    full_memory = "full" in args.basedir
    write_job(args.basedir, project, config_counts - 1, full_memory=full_memory)
