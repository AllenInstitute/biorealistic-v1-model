# Perturbation Simulation Workflow

## Key data locations
- Neuron annotations incl. activity labels live in `cell_categorization/core_nll_0_neuron_features.parquet`.
- Bio-trained network assets (nodes/edges) are under `core_nll_0/network`.
- Config templates sit in `core_nll_0/configs`; `config_bio_trained_inh_selective_iclamp.json` targets inhibitory-selective cells with a +50 pA current.
- Simulation outputs are typically stored in `core_nll_0/output_*/*/` with `spikes.csv`, `spikes.h5`, and the resolved config.
- Outgoing-weight fraction summaries and stacked-bar figures live in `core_nll_0/figures/selectivity_analysis/` (`complete_outgoing_stacked*.png/svg`, `complete_high_vs_low_comparison*.png/svg`) with the underlying tables in `core_nll_0/figures/selectivity_outgoing/`.
- Cohort-resolved box plots (rate, OSI/DSI, DG sparsity, image selectivity) are exported per network as `core_nll_0/figures/selectivity_analysis/core_*_by_cohort_{network}.png/svg` via `analysis_shared/plot_core_rate_boxplots.py`.

## Current-clamp config
- Current large-scale runs use the generated config sets in `core_nll_0/configs/8dir_10trials_inh_selective_pos200` (amp +200 pA) and `core_nll_0/configs/8dir_10trials_inh_selective_neg200` (amp -200 pA). Submit them with `jobs/8dir_10trials_inh_selective_{suffix}.sh`.
- Submit from the interactive node with `ssh hpc` (Snakefile targets updated); run `sbatch --array` scripts or trigger via Snakemake as needed.
- The current injection uses the BMTK PointNet `IClamp` module with `input_type="current_clamp"`.
- Targeting relies on an inline node-set filter: `{"population": "v1", "node_id": [...]}`. No node file edits are required, but the inhibitory-selective node list comes from the feature parquet (`activity_label == 'selective'` and `is_inhibitory`).
- Clamp parameters are `amp=50` pA, `delay=1.0` ms, `duration=2999.0` ms. The base config keeps LGN/BKG spike inputs unchanged.

## Running simulations
- Always activate the `new_v1` conda environment: `source <miniconda>/etc/profile.d/conda.sh && conda activate new_v1`.
- Launch a run with `python run_pointnet.py -n <threads> -o <output_dir> core_nll_0/configs/config_bio_trained_inh_selective_iclamp.json`.
- A 3 s simulation (~16.7k neurons) takes ≈6 minutes with 8 threads: ~2 min network build, ~3 min simulation, remainder for I/O.
- Reusing the base config (no clamp) under a different `-o` path gives a matched baseline run for comparisons.

## Metrics extraction
- `analysis_shared/perturbation_metrics.py` aggregates firing-rate metrics from one or more run directories.
    - Saves `per_node_rates.parquet`, `cell_type_metrics_by_run.csv`, `cell_type_metrics_summary.csv`, and inhibitory-selective summaries.
    - Accepts `--duration-ms` to override config-derived durations.
- Comparing conditions can be done by merging the summary outputs, e.g. `cell_type_rate_deltas.csv` created in `output_inh_selective_iclamp/metrics`.

## Practical notes
- `spikes.csv` is space-delimited (`timestamps population node_ids`). Only `population == 'v1'` is relevant for cortical metrics.
- For large batch jobs (e.g., 80 runs) consider scripting around `run_pointnet.py` to iterate `-o` directories or dispatch via Slurm.
- Background and LGN inputs are reused from `core_nll_0/bkg` and `core_nll_0/filternet`; if alternative stimulus sets are needed, update manifest paths accordingly.
- Keep Black formatting on new scripts; the project leans on pandas/h5py for data wrangling and BMTK (PointNet + NEST) for simulation.


- New 8dir config sets available for: 
    * Inhibitory selective ±100 pA (`core_nll_0/configs/8dir_10trials_inh_selective_{pos,neg}100`)
    * Excitatory selective ±100 pA (`core_nll_0/configs/8dir_10trials_exc_selective_{pos,neg}100`)
    * Inhibitory non-selective ±100 pA (`core_nll_0/configs/8dir_10trials_inh_nonselective_{pos,neg}100`)
    * Excitatory non-selective ±100 pA (`core_nll_0/configs/8dir_10trials_exc_nonselective_{pos,neg}100`)
  Submit via `sbatch core_nll_0/jobs/8dir_10trials_<suffix>.sh` from `ssh hpc`.


- Matched-population perturbations (randomly sampled to equalize counts):
    * Inh non-selective matched ±100 pA (`core_nll_0/configs/8dir_10trials_inh_nonselective_matched_{pos,neg}100`)
    * Exc selective matched ±100 pA (`core_nll_0/configs/8dir_10trials_exc_selective_matched_{pos,neg}100`)
  Submit via `sbatch core_nll_0/jobs/8dir_10trials_<suffix>.sh`.
