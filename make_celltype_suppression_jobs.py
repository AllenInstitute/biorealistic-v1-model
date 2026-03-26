#!/usr/bin/env python3
"""Generate 8dir_10trials configuration sets with cell-type specific suppression (PV, SST, VIP)."""

from __future__ import annotations

import json
import pathlib
import numpy as np

BASE_DIR = pathlib.Path("core_nll_0")
NODE_SETS_DIR = BASE_DIR / "node_sets"

# Cell types to suppress
CELL_TYPES = {
    "pv": {"node_set_file": "pv_nodes.json", "label": "PV (Parvalbumin)"},
    "sst": {"node_set_file": "sst_nodes.json", "label": "SST (Somatostatin)"},
    "vip": {"node_set_file": "vip_nodes.json", "label": "VIP"},
}

# Suppression amplitudes to test
AMP_VALUES = [-1000.0]  # Can add more like [-500.0, -1000.0, -2000.0] if needed


def load_node_set(filename: str) -> list[int]:
    """Load node IDs from a JSON node set file."""
    with open(NODE_SETS_DIR / filename) as f:
        data = json.load(f)
    return data["node_id"]


def create_base_config() -> dict:
    """Create base configuration template."""
    return {
        "manifest": {
            "$BASE_DIR": "${configdir}/../..",
            "$NETWORK_DIR": "$BASE_DIR/network",
            "$MODELS_DIR": "$BASE_DIR/components",
            "$OUTPUT_DIR": None,  # Will be set per config
            "$LGNINPUT_DIR": None,  # Will be set per config
            "$BKGINPUT_DIR": None,  # Will be set per config
        },
        "run": {"duration": 3000.0, "dt": 0.25, "block_run": False, "block_size": 1000.0},
        "inputs": {
            "LGN_spikes": {
                "input_type": "spikes",
                "module": "h5",
                "input_file": "$LGNINPUT_DIR/spikes.h5",
                "node_set": "lgn",
            },
            "BKG_spikes": {
                "input_type": "spikes",
                "module": "h5",
                "input_file": "$BKGINPUT_DIR/bkg_spikes_250Hz_3s.h5",
                "node_set": "bkg",
            },
        },
        "output": {
            "log_file": "log.txt",
            "spikes_file": "spikes.h5",
            "spikes_file_csv": "spikes.csv",
            "output_dir": "$OUTPUT_DIR",
            "overwrite_output_dir": True,
            "quiet_simulator": True,
        },
        "target_simulator": "NEST",
        "components": {
            "point_neuron_models_dir": "$MODELS_DIR/cell_models",
            "synaptic_models_dir": "$MODELS_DIR/synaptic_models",
        },
        "networks": {
            "nodes": [
                {
                    "nodes_file": "$NETWORK_DIR/v1_nodes.h5",
                    "node_types_file": "$NETWORK_DIR/v1_node_types.csv",
                },
                {
                    "nodes_file": "$NETWORK_DIR/lgn_nodes.h5",
                    "node_types_file": "$NETWORK_DIR/lgn_node_types.csv",
                },
                {
                    "nodes_file": "$NETWORK_DIR/bkg_nodes.h5",
                    "node_types_file": "$NETWORK_DIR/bkg_node_types.csv",
                },
            ],
            "edges": [
                {
                    "edges_file": "$NETWORK_DIR/v1_v1_edges_bio_trained.h5",
                    "edge_types_file": "$NETWORK_DIR/v1_v1_edge_types.csv",
                },
                {
                    "edges_file": "$NETWORK_DIR/lgn_v1_edges.h5",
                    "edge_types_file": "$NETWORK_DIR/lgn_v1_edge_types.csv",
                },
                {
                    "edges_file": "$NETWORK_DIR/bkg_v1_edges_bio_trained.h5",
                    "edge_types_file": "$NETWORK_DIR/bkg_v1_edge_types.csv",
                },
            ],
        },
        "node_sets": {},  # Will be populated per config
    }


def write_sbatch_script(
    script_path: pathlib.Path,
    config_dir: pathlib.Path,
    n_configs: int,
    memory_gb: int = 20,
    threads: int = 8,
    time_hours: int = 1,
) -> None:
    """Write SLURM batch script."""
    logdir = script_path.parent / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --partition=braintv\n")
        f.write(f"#SBATCH -N1 -c1 -n{threads}\n")
        f.write(f"#SBATCH --mem={memory_gb}G\n")
        f.write(f"#SBATCH -t{time_hours}:00:00\n")
        f.write("#SBATCH --qos=braintv\n")
        f.write(f"#SBATCH --output={logdir}/slurm-%A_%a.out\n")
        f.write(f"#SBATCH --error={logdir}/slurm-%A_%a.err\n")
        f.write(f"#SBATCH --array=0-{n_configs-1}\n\n")
        f.write(
            "source /allen/programs/mindscope/workgroups/realistic-model/shinya.ito/miniconda3/bin/activate new_v1\n"
        )
        rel_config = config_dir.relative_to(BASE_DIR.parent)
        f.write(f"python run_pointnet.py {rel_config}/config_$SLURM_ARRAY_TASK_ID.json -n {threads}\n")

    script_path.chmod(0o755)


def main():
    angles = np.linspace(0, 315, 8)
    trials = range(10)

    for cell_type, info in CELL_TYPES.items():
        print(f"\n{'='*80}")
        print(f"Creating configs for {info['label']} suppression")
        print(f"{'='*80}")

        # Load node IDs for this cell type
        node_ids = load_node_set(info["node_set_file"])
        print(f"  Loaded {len(node_ids)} {cell_type.upper()} cells")

        for amp in AMP_VALUES:
            amp_label = f"neg{abs(int(amp))}" if amp < 0 else f"pos{abs(int(amp))}"
            exp_name = f"{cell_type}_{amp_label}"

            config_dir = BASE_DIR / "configs" / f"8dir_10trials_{exp_name}"
            config_dir.mkdir(parents=True, exist_ok=True)

            config_count = 0
            for angle in angles:
                for trial in trials:
                    config = create_base_config()

                    # Set paths for this specific trial
                    angle_int = int(angle)
                    config["manifest"][
                        "$LGNINPUT_DIR"
                    ] = f"$BASE_DIR/filternet_8dir_10trials/angle{angle_int}_trial{trial}"
                    config["manifest"][
                        "$BKGINPUT_DIR"
                    ] = f"$BASE_DIR/bkg_8dir_10trials/angle{angle_int}_trial{trial}"
                    config["manifest"][
                        "$OUTPUT_DIR"
                    ] = f"$BASE_DIR/8dir_10trials_{exp_name}/angle{angle_int}_trial{trial}"

                    # Add current clamp for this cell type
                    clamp_name = f"{cell_type.upper()}Clamp"
                    node_set_name = f"{cell_type}_suppressed"

                    config["inputs"][clamp_name] = {
                        "input_type": "current_clamp",
                        "module": "IClamp",
                        "node_set": node_set_name,
                        "amp": amp,
                        "delay": 1.0,
                        "duration": 2999.0,
                    }

                    # Add node set definition
                    config["node_sets"][node_set_name] = {
                        "population": "v1",
                        "node_id": node_ids,
                    }

                    # Write config file
                    config_path = config_dir / f"config_{config_count}.json"
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=2)

                    config_count += 1

            print(f"  Created {config_count} configs in {config_dir}")

            # Create SLURM job script
            job_script = BASE_DIR / "jobs" / f"8dir_10trials_{exp_name}.sh"
            write_sbatch_script(
                job_script,
                config_dir,
                config_count,
                memory_gb=20,
                threads=8,
                time_hours=1,
            )
            print(f"  Created job script: {job_script}")
            print(f"  → Submit with: sbatch {job_script}")


if __name__ == "__main__":
    main()
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nTo submit all jobs, run:")
    print("  sbatch core_nll_0/jobs/8dir_10trials_pv_neg1000.sh")
    print("  sbatch core_nll_0/jobs/8dir_10trials_sst_neg1000.sh")
    print("  sbatch core_nll_0/jobs/8dir_10trials_vip_neg1000.sh")
    print("\nTo test one config before submitting:")
    print("  python run_pointnet.py core_nll_0/configs/8dir_10trials_pv_neg1000/config_0.json -n 8")
    print("="*80)
