#!/usr/bin/env python3
"""Generate 8dir_10trials configuration sets with cell-type AND high/low outgoing weight specific suppression."""

from __future__ import annotations

import json
import pathlib
import numpy as np

BASE_DIR = pathlib.Path("core_nll_0")
NODE_SETS_DIR = BASE_DIR / "node_sets"

# Cell types with high/low splits
EXPERIMENTS = [
    {"name": "pv_high", "file": "pv_high_outgoing_nodes.json", "label": "PV High Outgoing"},
    {"name": "pv_low", "file": "pv_low_outgoing_nodes.json", "label": "PV Low Outgoing"},
    {"name": "sst_high", "file": "sst_high_outgoing_nodes.json", "label": "SST High Outgoing"},
    {"name": "sst_low", "file": "sst_low_outgoing_nodes.json", "label": "SST Low Outgoing"},
    {"name": "vip_high", "file": "vip_high_outgoing_nodes.json", "label": "VIP High Outgoing"},
    {"name": "vip_low", "file": "vip_low_outgoing_nodes.json", "label": "VIP Low Outgoing"},
]

# Suppression amplitude
AMP = -1000.0


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
            "$OUTPUT_DIR": None,
            "$LGNINPUT_DIR": None,
            "$BKGINPUT_DIR": None,
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
        "node_sets": {},
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

    print("="*80)
    print("GENERATING CELL-TYPE SPECIFIC HIGH/LOW SUPPRESSION EXPERIMENTS")
    print("="*80)

    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        print(f"\n{exp['label']} ({exp_name}):")
        print("-"*80)

        # Load node IDs
        node_ids = load_node_set(exp["file"])
        print(f"  Cells: {len(node_ids)}")

        # Create config directory
        amp_label = f"neg{abs(int(AMP))}"
        full_name = f"{exp_name}_{amp_label}"
        config_dir = BASE_DIR / "configs" / f"8dir_10trials_{full_name}"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_count = 0
        for angle in angles:
            for trial in trials:
                config = create_base_config()

                # Set paths
                angle_int = int(angle)
                config["manifest"][
                    "$LGNINPUT_DIR"
                ] = f"$BASE_DIR/filternet_8dir_10trials/angle{angle_int}_trial{trial}"
                config["manifest"][
                    "$BKGINPUT_DIR"
                ] = f"$BASE_DIR/bkg_8dir_10trials/angle{angle_int}_trial{trial}"
                config["manifest"][
                    "$OUTPUT_DIR"
                ] = f"$BASE_DIR/8dir_10trials_{full_name}/angle{angle_int}_trial{trial}"

                # Add current clamp
                clamp_name = f"{exp_name.upper()}_Clamp"
                node_set_name = f"{exp_name}_suppressed"

                config["inputs"][clamp_name] = {
                    "input_type": "current_clamp",
                    "module": "IClamp",
                    "node_set": node_set_name,
                    "amp": AMP,
                    "delay": 1.0,
                    "duration": 2999.0,
                }

                # Add node set
                config["node_sets"][node_set_name] = {
                    "population": "v1",
                    "node_id": node_ids,
                }

                # Write config
                config_path = config_dir / f"config_{config_count}.json"
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

                config_count += 1

        print(f"  Configs: {config_count} → {config_dir}")

        # Create job script
        job_script = BASE_DIR / "jobs" / f"8dir_10trials_{full_name}.sh"
        write_sbatch_script(job_script, config_dir, config_count)
        print(f"  Job script: {job_script}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - 6 EXPERIMENTS CREATED")
    print("="*80)
    print("\nTo submit all jobs:\n")
    for exp in EXPERIMENTS:
        full_name = f"{exp['name']}_neg{abs(int(AMP))}"
        print(f"  sbatch core_nll_0/jobs/8dir_10trials_{full_name}.sh  # {exp['label']}")

    print("\n" + "="*80)
    print("To test one config from each:")
    print("="*80)
    print("\nsource /allen/programs/mindscope/workgroups/realistic-model/shinya.ito/miniconda3/bin/activate new_v1\n")
    for exp in EXPERIMENTS:
        full_name = f"{exp['name']}_neg{abs(int(AMP))}"
        print(f"python run_pointnet.py core_nll_0/configs/8dir_10trials_{full_name}/config_0.json -n 8")


if __name__ == "__main__":
    main()
