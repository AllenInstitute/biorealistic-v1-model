#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Run full image-decoding analysis (bio-trained vs naive) overnight.
# 1. Builds cached firing-rate tensors for each network & condition.
# 2. Executes the Python driver that performs decoding for
#    n_reps = 10,30,50 and sample_size = 10,30,50.
#
# Usage:  bash image_decoding/run_decoding_analysis.sh  [optional NET_LIST]
# If NET_LIST is provided (space-separated numbers) it overrides the default
# 0-9 network range, e.g.:
#     bash image_decoding/run_decoding_analysis.sh "0 1 2"
# -----------------------------------------------------------------------------
set -euo pipefail

# --- configuration -----------------------------------------------------------
NETWORKS_DEFAULT="0 1 2 3 4 5 6 7 8 9"
NETWORKS="${1:-$NETWORKS_DEFAULT}"

# Build caches for both model variants
for NET_TYPE in bio_trained naive; do
    echo "[cache] Building caches for $NET_TYPE …" >&2
    python -m image_decoding.cache_precompute \
        --networks ${NETWORKS} \
        --network_type ${NET_TYPE}
    echo "[cache] Completed for $NET_TYPE." >&2
done

# Neuropixels cache -------------------------------------------------------
echo "[cache] Building Neuropixels caches …" >&2
python -m image_decoding.neuropixels.cache_precompute_np

echo "[analysis] Running full decoding analysis (model) …" >&2
python -m image_decoding.full_analysis \
    --networks ${NETWORKS} \
    --n_jobs "$(nproc)" \
    --fit_n_jobs -1 \
    --outdir image_decoding/summary_full

echo "[analysis] Running Neuropixels decoding …" >&2
python -m image_decoding.neuropixels.run_decoding_np_cached --n_jobs "$(nproc)" --fit_n_jobs -1
echo "[done] All results written to image_decoding/summary_full and neuropixels summaries" >&2
