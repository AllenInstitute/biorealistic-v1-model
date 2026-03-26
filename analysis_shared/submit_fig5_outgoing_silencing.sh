#!/bin/bash

# Submit Fig5 outgoing-cohort silencing simulations + postprocessing as two dependent SLURM jobs.
#
# Usage (on cluster, from repo root):
#   bash analysis_shared/submit_fig5_outgoing_silencing.sh
#
# This script does NOT generate node_sets/configs. Run those beforehand.

set -euo pipefail

ROOT_DIR=$(pwd)

SIM_SCRIPT="analysis_shared/slurm_fig5_outgoing_silencing_array.sh"
POST_SCRIPT="analysis_shared/slurm_fig5_postprocess_metrics.sh"

if [ ! -f "${SIM_SCRIPT}" ]; then
  echo "Missing ${SIM_SCRIPT} (run from repo root)." >&2
  exit 2
fi

if [ ! -f "${POST_SCRIPT}" ]; then
  echo "Missing ${POST_SCRIPT} (run from repo root)." >&2
  exit 2
fi

conds=(
  "exc_high_core_neg1000"
  "exc_low_core_neg1000"
  "inh_high_core_neg1000"
  "inh_low_core_neg1000"
)

# Sanity check: configs exist and are complete (80 each)
for i in {0..9}; do
  base="core_nll_${i}"
  for cond in "${conds[@]}"; do
    d="${base}/configs/8dir_10trials_${cond}"
    if [ ! -d "${d}" ]; then
      echo "Missing config dir: ${d}" >&2
      exit 3
    fi
    n=$(ls -1 "${d}"/config_*.json 2>/dev/null | wc -l | tr -d ' ')
    if [ "${n}" != "80" ]; then
      echo "Unexpected config count: ${d} has ${n} (expected 80)" >&2
      exit 4
    fi
  done
  if [ ! -d "${base}/node_sets" ]; then
    echo "Missing node_sets dir: ${base}/node_sets" >&2
    exit 5
  fi
  if [ ! -f "${base}/node_sets/high_outgoing_exc_core_nodes.json" ]; then
    echo "Missing node set: ${base}/node_sets/high_outgoing_exc_core_nodes.json" >&2
    exit 6
  fi
  if [ ! -f "${base}/node_sets/low_outgoing_exc_core_nodes.json" ]; then
    echo "Missing node set: ${base}/node_sets/low_outgoing_exc_core_nodes.json" >&2
    exit 7
  fi
  if [ ! -f "${base}/node_sets/high_outgoing_inh_core_nodes.json" ]; then
    echo "Missing node set: ${base}/node_sets/high_outgoing_inh_core_nodes.json" >&2
    exit 8
  fi
  if [ ! -f "${base}/node_sets/low_outgoing_inh_core_nodes.json" ]; then
    echo "Missing node set: ${base}/node_sets/low_outgoing_inh_core_nodes.json" >&2
    exit 9
  fi

done

echo "Submitting sims: ${SIM_SCRIPT}"
SIM_SUBMIT_OUT=$(sbatch "${SIM_SCRIPT}")
echo "${SIM_SUBMIT_OUT}"
SIM_JOB_ID=$(echo "${SIM_SUBMIT_OUT}" | awk '{print $4}')

if [ -z "${SIM_JOB_ID}" ]; then
  echo "Failed to parse simulation job id from: ${SIM_SUBMIT_OUT}" >&2
  exit 10
fi

echo "Submitting postprocess (afterok:${SIM_JOB_ID}): ${POST_SCRIPT}"
POST_SUBMIT_OUT=$(sbatch --dependency=afterok:"${SIM_JOB_ID}" "${POST_SCRIPT}")
echo "${POST_SUBMIT_OUT}"

POST_JOB_ID=$(echo "${POST_SUBMIT_OUT}" | awk '{print $4}')

echo "\nSubmitted jobs:"
echo "  sims       : ${SIM_JOB_ID}"
echo "  postprocess: ${POST_JOB_ID}"
echo "\nRepo root: ${ROOT_DIR}"
