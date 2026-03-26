#!/usr/bin/env python3
"""Generate 8dir_10trials configuration sets with cell-type specific cohort suppression across multiple networks."""

from __future__ import annotations

import json
import pathlib
import numpy as np

# Suppression amplitudes to test
AMP_VALUES = [-1000.0]

# High and Low cell type cohorts
CELL_COHORTS = {
    "pv_high": {"node_set_file": "pv_high_outgoing_core_nodes.json", "label": "PV High"},
    "pv_low": {"node_set_file": "pv_low_outgoing_core_nodes.json", "label": "PV Low"},
    "sst_high": {"node_set_file": "sst_high_outgoing_core_nodes.json", "label": "SST High"},
    "sst_low": {"node_set_file": "sst_low_outgoing_core_nodes.json", "label": "SST Low"},
    "vip_high": {"node_set_file": "vip_high_outgoing_core_nodes.json", "label": "VIP High"},
    "vip_low": {"node_set_file": "vip_low_outgoing_core_nodes.json", "label": "VIP Low"},
}


def load_node_set(node_sets_dir: pathlib.Path, filename: str) -> list[int]:
    """Load node IDs from a JSON node set file."""
    path = node_sets_dir / filename
    if not path.exists():
        print(f"Warning: {path} not found.")
        return []
    with open(path) as f:
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
    filepath: pathlib.Path,
    config_dir: pathlib.Path,
    num_configs: int,
    memory_gb: int = 20,
    threads: int = 8,
    time_hours: int = 1,
):
    """Write a SLURM job array script."""
    script_content = f"""#!/bin/bash
#SBATCH --job-name={filepath.stem}
#SBATCH --output={config_dir.parent}/logs/{filepath.stem}_%A_%a.out
#SBATCH --error={config_dir.parent}/logs/{filepath.stem}_%A_%a.err
#SBATCH --time={time_hours}:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={threads}
#SBATCH --mem={memory_gb}G
#SBATCH --array=0-{num_configs-1}
#SBATCH --partition=d3

# Create logs directory if it doesn't exist
mkdir -p {config_dir.parent}/logs

source activate new_v1
export OMP_NUM_THREADS={threads}

echo "Starting task $SLURM_ARRAY_TASK_ID"
python run_pointnet.py {config_dir}/config_${{SLURM_ARRAY_TASK_ID}}.json -n {threads}
echo "Finished task $SLURM_ARRAY_TASK_ID"
"""
    with open(filepath, "w") as f:
        f.write(script_content)


def main():
    networks = [f"core_nll_{i}" for i in range(1, 10)] # Skipping core_nll_0 as it's already done
    
    print(f"Setting up simulation environments across networks {networks[0]} to {networks[-1]}")
    
    all_scripts = []
    
    for net in networks:
        base_dir = pathlib.Path(net)
        if not base_dir.exists():
            continue
            
        print(f"Processing {net}...")
        
        node_sets_dir = base_dir / "node_sets"
        jobs_dir = base_dir / "jobs"
        jobs_dir.mkdir(exist_ok=True, parents=True)
        
        for ct_key, ct_info in CELL_COHORTS.items():
            print(f"  Setting up {ct_info['label']} suppression...")
            node_ids = load_node_set(node_sets_dir, ct_info["node_set_file"])
            if not node_ids:
                print(f"    No node IDs found for {ct_key}, skipping.")
                continue

            for amp in AMP_VALUES:
                # Format exp name (e.g., pv_high_neg1000)
                amp_str = f"neg{int(abs(amp))}" if amp < 0 else f"pos{int(amp)}"
                exp_name = f"{ct_key}_{amp_str}"

                # Create output and config directories for this experiment
                exp_dir_name = f"8dir_10trials_{exp_name}"
                config_dir = base_dir / "configs" / exp_dir_name
                config_dir.mkdir(parents=True, exist_ok=True)
                
                # Make sure output dirs will exist
                (base_dir / exp_dir_name).mkdir(parents=True, exist_ok=True)

                config_count = 0
                for angle in range(0, 360, 45):
                    for trial in range(10):
                        # Create config
                        config = create_base_config()
                        
                        # Set paths
                        angle_int = int(angle)
                        lgn_input = f"$BASE_DIR/filternet_8dir_10trials/angle{angle_int}_trial{trial}"
                        bkg_input = f"$BASE_DIR/bkg_8dir_10trials/angle{angle_int}_trial{trial}"
                        out_dir = f"$BASE_DIR/{exp_dir_name}/angle{angle_int}_trial{trial}"

                        config["manifest"]["$OUTPUT_DIR"] = out_dir
                        config["manifest"]["$LGNINPUT_DIR"] = lgn_input
                        config["manifest"]["$BKGINPUT_DIR"] = bkg_input

                        # Add current clamp inputs
                        clamp_name = f"{ct_key.upper()}_Clamp"
                        node_set_name = f"{ct_key}_suppressed"

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

                # Create SLURM job script
                job_script = jobs_dir / f"8dir_10trials_{exp_name}.sh"
                write_sbatch_script(
                    job_script,
                    config_dir,
                    config_count,
                    memory_gb=20,
                    threads=8,
                    time_hours=1,
                )
                all_scripts.append(job_script)
                print(f"    Created job script: {job_script}")
                
    # Write a master submission script
    master_script = pathlib.Path("submit_all_celltype_suppressions.sh")
    with open(master_script, "w") as f:
        f.write("#!/bin/bash\n\n")
        for script in all_scripts:
            f.write(f"sbatch {script}\n")
            
    master_script.chmod(0o755)
    print(f"\nCreated master submission script: {master_script}")
    print(f"Run './{master_script}' to submit all {len(all_scripts)} jobs.")

if __name__ == "__main__":
    main()
