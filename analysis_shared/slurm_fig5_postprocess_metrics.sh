#!/bin/bash
#SBATCH --partition=braintv
#SBATCH -N1 -c1 -n4
#SBATCH --mem=40G
#SBATCH -t4:00:00
#SBATCH --qos=braintv
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --array=4-39

# Postprocess Fig5 outgoing-cohort silencing runs into OSI/DSI CSVs and DG sparsity caches.
# Array index encodes (network_idx, condition_idx):
#   network_idx  : 0..9
#   condition_idx: 0..3
# Total tasks: 10 * 4 = 40

set -eo pipefail

mkdir -p logs

NETWORK_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
COND_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

BASE_DIR="core_nll_${NETWORK_IDX}"

COND_NAMES=(
  "exc_high_core_neg1000"
  "exc_low_core_neg1000"
  "inh_high_core_neg1000"
  "inh_low_core_neg1000"
)

COND="${COND_NAMES[$COND_IDX]}"

source /allen/programs/mindscope/workgroups/realistic-model/shinya.ito/miniconda3/bin/activate new_v1

# 1) OSI/DSI + Rates tables from spikes.csv across the 80 trials
python calculate_odsi.py "${BASE_DIR}" "${COND}"

# 2) DG sparsity cache (reads spikes.h5 across the same 80 trials)
python analysis_shared/compute_dg_sparsity_for_all_networks.py --base-dir "${BASE_DIR}" --network "${COND}"
