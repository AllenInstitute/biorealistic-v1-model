#!/bin/bash
#SBATCH --partition=braintv
#SBATCH -N1 -c1 -n8
#SBATCH --mem=20G
#SBATCH -t1:00:00
#SBATCH --qos=braintv
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --array=322-3199

# Multi-network Fig5 outgoing-cohort silencing runs.
# Array index encodes (network_idx, condition_idx, config_idx):
#   network_idx  : 0..9  -> core_nll_${network_idx}
#   condition_idx: 0..3  -> exc_high_core_neg1000, exc_low_core_neg1000, inh_high_core_neg1000, inh_low_core_neg1000
#   config_idx   : 0..79 -> config_${config_idx}.json within that condition
# Total tasks: 10 * 4 * 80 = 3200

set -eo pipefail

mkdir -p logs

NETWORK_IDX=$(( SLURM_ARRAY_TASK_ID / 320 ))
REM=$(( SLURM_ARRAY_TASK_ID % 320 ))
COND_IDX=$(( REM / 80 ))
CFG_IDX=$(( REM % 80 ))

BASE_DIR="core_nll_${NETWORK_IDX}"

COND_NAMES=(
  "exc_high_core_neg1000"
  "exc_low_core_neg1000"
  "inh_high_core_neg1000"
  "inh_low_core_neg1000"
)

COND="${COND_NAMES[$COND_IDX]}"
CFG_PATH="${BASE_DIR}/configs/8dir_10trials_${COND}/config_${CFG_IDX}.json"

if [ ! -f "${CFG_PATH}" ]; then
  echo "Missing config: ${CFG_PATH}" >&2
  exit 2
fi

set +e
source /allen/programs/mindscope/workgroups/realistic-model/shinya.ito/miniconda3/bin/activate new_v1
set -e

python -c "import nest" >/dev/null 2>&1

python run_pointnet.py "${CFG_PATH}" -n 8
