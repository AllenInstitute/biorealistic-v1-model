"""Generate 8dir_10trials configuration sets with inhibitory-selective current clamp."""

from __future__ import annotations

import argparse
import json
import pathlib
import numpy as np


def sbatch_boilerplate(
    file, logdir: pathlib.Path, config_counts: int, memory: int = 30, threads: int = 8
) -> None:
    file.write("#!/bin/bash\n")
    file.write("#SBATCH --partition=d3\n")
    file.write(f"#SBATCH -N1 -n1 -c{threads}\n")
    file.write(f"#SBATCH --mem={memory}G\n")
    file.write("#SBATCH -t4:00:00\n")
    file.write("#SBATCH --qos=d3\n")
    file.write(f"#SBATCH --output={logdir}/slurm-%A_%a.out\n")
    file.write(f"#SBATCH --error={logdir}/slurm-%A_%a.err\n")
    file.write(f"#SBATCH --array=0-{config_counts-1}\n\n")
    file.write(
        "source /allen/programs/mindscope/workgroups/realistic-model/shinya.ito/miniconda3/etc/profile.d/conda.sh\n"
    )
    file.write("conda activate new_v1\n")


def write_job_script(
    base_dir: pathlib.Path,
    config_dir: pathlib.Path,
    job_name: str,
    config_counts: int,
    threads: int = 8,
) -> None:
    jobdir = base_dir / "jobs"
    logdir = jobdir / "logs"
    jobdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(exist_ok=True)

    script_path = jobdir / f"{job_name}.sh"
    with script_path.open("w") as fh:
        sbatch_boilerplate(fh, logdir, config_counts, memory=30, threads=threads)
        config_array = config_dir / "config_$SLURM_ARRAY_TASK_ID.json"
        fh.write(f"python run_pointnet.py {config_array} -n {threads}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 8dir_10trials configs with clamp variants."
    )
    parser.add_argument("basedir", type=pathlib.Path)
    parser.add_argument("base_config", type=pathlib.Path)
    parser.add_argument(
        "suffix",
        type=str,
        help="Experiment suffix appended to config/output directories.",
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    base_dir = args.basedir.resolve()
    base_config = args.base_config.resolve()
    suffix = args.suffix

    config_dir = base_dir / "configs" / f"8dir_10trials_{suffix}"
    output_root = base_dir / f"8dir_10trials_{suffix}"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_template = json.loads(base_config.read_text())

    angles = np.linspace(0, 315, 8)
    trials = range(10)

    config_count = 0
    for angle in angles:
        for trial in trials:
            # Update manifest pointers for this condition
            config_template["manifest"]["$BASE_DIR"] = "${configdir}/../.."
            config_template["manifest"][
                "$LGNINPUT_DIR"
            ] = f"$BASE_DIR/filternet_8dir_10trials/angle{int(angle)}_trial{trial}"
            config_template["manifest"][
                "$BKGINPUT_DIR"
            ] = f"$BASE_DIR/bkg_8dir_10trials/angle{int(angle)}_trial{trial}"
            config_template["manifest"][
                "$OUTPUT_DIR"
            ] = f"$BASE_DIR/8dir_10trials_{suffix}/angle{int(angle)}_trial{trial}"

            config_path = config_dir / f"config_{config_count}.json"
            config_path.write_text(json.dumps(config_template, indent=2))
            config_count += 1

    write_job_script(
        base_dir,
        config_dir,
        f"8dir_10trials_{suffix}",
        config_count,
        threads=args.threads,
    )
    print(f"Wrote {config_count} configs to {config_dir}")
