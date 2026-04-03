# biorealistic-v1-model

A project for making biorealistic model of mouse V1.
This is a successor project of making a model of the mouse primary visual cortex
(<https://portal.brain-map.org/explore/models/mv1-all-layers>)

The improvements are:

* Connection probability and weights are derived from coherent datasets, including the Allen Institute synaptic physiology data and MICrONS electron microscopy connectomics dataset (instead from the literature).
* Synaptic connections are now expressed by double alpha functions and the receptor types are more elaborated.
* Segregation of the L5 excitatory cells into IT, ET, and NP types.
* More GLIF cell models are used.

Only PointNet version is available as of now.

## Table of contents

- [Installation instruction](#installation-instruction)
- [How to build a network and run a simulation](#how-to-build-a-network-and-run-a-simulation)
- [Simulation pipelines for analysis](#simulation-pipelines-for-analysis)
  - [Network weight variants](#network-weight-variants)
  - [Drifting-grating simulations (Figures 3, 5, 6)](#drifting-grating-simulations-figures-3-5-6)
  - [Contrast-response simulations (Figures 3, 6)](#contrast-response-simulations-figures-3-6)
  - [Brain Observatory natural-image simulations (Figure 4)](#brain-observatory-natural-image-simulations-figure-4)
  - [Response-correlation computation (Figure 5)](#response-correlation-computation-figure-5)
  - [Suppression / silencing experiments (Figure 5)](#suppression--silencing-experiments-figure-5)
- [Folders and files](#folders-and-files)
- [Analysis and figure generation](#analysis-and-figure-generation)
- [Notes](#notes)
- [Main contributors](#main-contributors)

## Installation instruction

This repository mainly documents the PointNet-based network build and simulation workflow.
If you are interested in TensorFlow-based training workflows, please refer to
`instructions/tensorflow.md` and the related files in the `instructions/` directory.
If you are interested in running simulations using trained networks, trained SONATA
network files are available here:
<https://www.dropbox.com/scl/fo/4i8tsihwokn78jpb6wqls/AE9ukbU8ShG1R5hQEdssxDg?rlkey=332wdyd2ou5yujy5us3eq89pc&st=pg2mabe6&dl=0>

First, clone this repository to your local environment.

```bash
git clone https://github.com/AllenInstitute/biorealistic-v1-model.git
```

I suggest using [miniforge](https://github.com/conda-forge/miniforge) if you are setting up the environment for this. (Other conda variants will also work, if you specify 'conda-forge' as the primary package source.)

The 'conda' command can be replaced with 'mamba' if you have it installed, and it's much faster than 'conda'.

```bash
conda env create -f environment.yml -n <env_name>
```

## How to build a network and run a simulation

The most straightforward way to get a network and its simulation output is doing the following.

```bash
snakemake <network_name>/output_plain/spikes.h5
```

where `<network_name>` is one of the defined names in `Snakefile`. Namely:

* `full`: The full size network (700 µm radius, ~203k neurons)
* `core`: 400 µm radius network. ~67k neurons.
* `core_X`: X is {0-9}. The number of neurons in each cell model has Poisson fluctuation.
* `core_nll`: Same geometry as core, but no weight like-to-like is used.
* `core_nll_X`: Same as core_X, but no weight like-to-like is used (used in the paper)
* `small`: 200 µm radius network.
* `tiny`: 100 µm radius network. Mainly for testing the build script.
* `profile`: For profiling workflows. 200 µm radius.

You can create a custom size if you define parameters in `Snakefile`.

Also, you can edit `V1model_seed_file.xlsx` to change what cell types are be included. For example, if you want to make L4 only network, you can delete all the cell types other than L4 from this file.

Building the `full` model requires hundreds of GBs of memory. It usually needs to be run on a cluster computer, but it is not incorporated in the snakemake workflow yet. If you really need the `full` network, running the build script should be done manually. The subsequent processes should work fine once you build the model.

## Simulation pipelines for analysis

The analysis scripts in this repository expect simulation output that was generated using
specific stimulus paradigms. Each paradigm requires its own LGN (FilterNet) and
background spike inputs, followed by a PointNet V1 simulation. The general workflow for
every paradigm is:

1. **Build the network** — `snakemake <network_name>/network/v1_nodes.h5` (or the full build as described above).
2. **Generate background spikes** — `python bkg_spike_generation.py <network_name>` creates background spike files for all paradigms in one pass (`bkg/`, `bkg_8dir_10trials/`, `bkg_contrasts/`, `bkg_imagenet/`, `bkg_bo/`).
3. **Run FilterNet** (LGN) — produces LGN spike trains from visual stimuli. Config in `config_templates/config_filternet.json`.
4. **Run PointNet** (V1) — uses LGN + background spikes to simulate V1 activity. Config in `config_templates/config_<network_option>.json`.

Steps 3–4 are typically launched via SLURM job scripts generated by `make_*_jobs.py` helpers.
On a local machine with modest resources, you can run them directly with
`python run_filternet.py <config>` and `python run_pointnet.py <config> -n <threads>`.

### Network weight variants

Each network can be simulated with different weight files selected by the `network_option`
parameter. The main options used in the paper are:

* `plain` — Original untrained weights (as built).
* `adjusted` — Weights after the current-adjustment procedure (`make_adjusted_network.py`).
* `bio_trained` — Weights after TensorFlow training with biological constraints.
* `naive` — Weights after TensorFlow training without biological constraints.

These map to config files `config_templates/config_<option>.json`, which point to the
corresponding edge files (e.g., `v1_v1_edges_bio_trained.h5`). Trained edge files must
be placed in `<network_name>/network/` before running simulations.

### Drifting-grating simulations (Figures 3, 5, 6)

8 directions × 10 trials, 3 s each. Used for OSI/DSI metrics and preferred-direction analyses.

```bash
# 1. Generate FilterNet jobs (LGN responses to drifting gratings)
python make_osi_jobs.py <network_name> --filternet --memory 20

# 2. Submit FilterNet array job (cluster) or run locally
sbatch <network_name>/jobs/filternet_8dir_10trials.sh

# 3. Generate PointNet jobs for a given weight variant
python make_osi_jobs.py <network_name> --network_option bio_trained --memory 20

# 4. Submit PointNet array job
sbatch <network_name>/jobs/8dir_10trials_bio_trained.sh
```

Output lands in `<network_name>/8dir_10trials_<option>/angle<A>_trial<T>/spikes.h5`.
After all trials complete, compute OSI/DSI metrics:

```bash
python calculate_odsi.py <network_name> <network_option>
# → <network_name>/metrics/OSI_DSI_DF_<option>.csv
```

Snakemake shortcut (single network, single option):

```bash
snakemake core_nll_0/metrics/OSI_DSI_DF_bio_trained.csv
```

### Contrast-response simulations (Figures 3, 6)

8 directions × 6 contrasts (0.05–0.80) × 10 trials. Stimulus conditions are defined in
`stimulus_trials.py` (`ContrastStimulus` class).

```bash
# Generate and submit FilterNet + PointNet contrast jobs
python make_contrast_jobs.py <network_name> --filternet --memory 20
sbatch <network_name>/jobs/filternet_contrasts.sh

python make_contrast_jobs.py <network_name> --network_option bio_trained --memory 20
sbatch <network_name>/jobs/contrasts_bio_trained.sh
```

Output: `<network_name>/contrasts_<option>/angle<A>_contrast<C>_trial<T>/spikes.h5`.
After completion, aggregate spike counts and plot contrast-response curves:

```bash
python contrast_spike_aggregation.py <network_name> <network_option>
python contrast_analysis.py <network_name> <network_option>
```

### Brain Observatory natural-image simulations (Figure 4)

118 natural images from the Allen Brain Observatory, 250 ms each, repeated for 50
chunks (repetitions). Each chunk is ~30 s. These simulations are **not** included in the
Snakefile and must be set up manually.

Background spikes are generated automatically by `bkg_spike_generation.py` (stored in
`<network_name>/bkg_bo/bkg_bo_chunk_XX.h5`). FilterNet and PointNet configs for each
chunk need to be created by adapting the base config to point to the appropriate
FilterNet output and background file for each chunk.

Output convention: `<network_name>/output_bo_<option>/chunk_XX/spikes.h5`.

Once all 50 chunks are complete, cache firing-rate tensors for the decoding pipeline:

```bash
python -m image_decoding.cache_precompute --networks 0 1 2 3 4 5 6 7 8 9 --network_type bio_trained
```

See `image_decoding/README.md` for the full decoding and plotting workflow.

### Response-correlation computation (Figure 5)

After natural-image simulations (ImageNet or Brain Observatory) are complete, compute
pairwise response correlations between connected neurons:

```bash
python response_correlation_calculations.py --base_dir <network_name> --input_type <option>
# → <network_name>/metrics/response_correlations_<option>.npy
```

These `.npy` files are consumed by the Figure 5 correlation scripts
(`generate_corr_final_panel.py`, etc.).

### Suppression / silencing experiments (Figure 5)

These experiments suppress activity of specific neuron cohorts (e.g., high-outgoing-weight
excitatory cells, or specific inhibitory subtypes) and re-run drifting-grating simulations
to measure the effect on network activity.

```bash
# 1. Generate cohort node sets (identifies which neurons to suppress)
python analysis_shared/create_highlow_outgoing_weight_nodesets.py
python analysis_shared/create_celltype_nodesets.py

# 2. Generate silencing simulation configs
python analysis_shared/generate_fig5_outgoing_silencing_configs.py

# 3. Generate and submit SLURM jobs
python make_celltype_suppression_jobs.py
sbatch <network_name>/jobs/<suppression_job>.sh

# 4. Compute metrics from suppression output
python analysis_shared/calculate_celltype_suppression_metrics.py
```

Results are plotted by `figure_scripts/figure5/plot_celltype_suppression_boxplots.py`
and `plot_perturbation_heatmap_figure5.py`.

## Folders and files

base_props/: A folder that contains seed files that are necessary for building the network. These are mostly designed to be human readable and editable.

base_props/V1model_seed_file.xlsx: An excel file that contains general properties of
each cell population. Edit this when you want to change which cell population is used
and their numbers.

base_props/exclude_list.csv: Cells listed in this file will be excluded from the model. The reasons could vary, but generally due to undesired properties in the model, such as spiny inhibitory cells, aspiny excitatory cells, or that the model does not well reproduce cell's activity.

base_props/bkg_weights_population.csv: This file defines synaptic weights from background cell (currently, there is only one entity for background source) to each cell population. The synaptic weights in this file are copied from the previous model (except for L5 which is average of the two types in the previous model), and are parameters of the optimization stage later.

Preferred data storage option is CSV format, as long as the data are not large. All the csv files are 'space' separarated (to match with SONATA format). Larger data can be stored as h5 data frame. All should be readable in pandas.


## Analysis and figure generation

This repository also contains the analysis code used to generate all figures in the paper.
Analysis results are produced by running standalone Python scripts from the repo root
with the `new_v1` conda environment activated.

The code is organized into four layers:

1. **Root-level utility modules** — Low-level loaders and computation routines (`network_utils.py`, `calculate_odsi.py`, etc.) that most other scripts depend on.
2. **`analysis_shared/`** — A shared library of reusable analysis functions: data I/O, cell-type grouping, statistical fitting, weight-vs-activity analyses, cohort generation, and perturbation metrics.
3. **`figure_scripts/`** — Per-figure panel generators organized by figure number (`figure3/` through `figure6/`, plus `extended_data/`). Each script produces one or more publication panels.
4. **`image_decoding/`** — A self-contained sub-package for natural-image decoding analyses (Figure 4), with its own preprocessing, decoding, and plotting modules. See `image_decoding/README.md`.

Experimental reference data live in `neuropixels/` (Neuropixels OSI/DSI metrics) and
`analysis_shared/*.pkl` (MICrONS EM data). Extended documentation is in `docs/`.

### Root-level utility modules

These modules are imported by many analysis and figure scripts:

- **`network_utils.py`** — Core functions for loading SONATA network nodes/edges, spike data, and cell-type tables. Most analysis code depends on this.
- **`plotting_utils.py`** — General plotting helpers (core-neuron selection, config reading, raster/box-plot routines).
- **`stimulus_trials.py`** — Defines stimulus conditions (angles, contrasts, trials) and path conventions for drifting-grating and contrast experiments.
- **`calculate_odsi.py`** — Computes firing rates, orientation/direction selectivity indices (OSI/DSI), and lifetime/population sparsity from spike files.
- **`response_correlation.py`** / **`response_correlation_calculations.py`** — Compute pairwise response correlations between neurons across stimulus conditions.
- **`aggregate_boxplots_odsi.py`** — Discovers and aggregates OSI/DSI/firing-rate metrics across multiple networks for box-plot generation.
- **`aggregate_similarity_odsi.py`** — Computes distribution-similarity scores (1 − KS) between model and Neuropixels data.
- **`aggregate_correlation_plot.py`** — Loads edge tables with response correlations for weight-vs-correlation analyses.
- **`contrast_aggregated_plots.py`** / **`contrast_analysis.py`** / **`contrast_quantification.py`** — Aggregate and plot contrast-response functions across networks.
- **`plot_raster.py`** / **`plot_v1_raster.py`** — Raster plot generation for drifting-grating and natural-image stimuli.
- **`plot_odsi.py`** — Single-network OSI/DSI box plots.

### `analysis_shared/` — Shared analysis library

A Python package of reusable analysis and plotting utilities:

- **Data I/O & grouping**
  - `io.py` — Load edges with response correlation or preferred direction; wraps `network_utils`.
  - `grouping.py` — Cell-type aggregation (L5 IT/ET/NP → L5 Exc; layered Inh → simplified PV/SST/VIP).
  - `sampling.py` — Per-pair downsampling to match EM connection counts.
  - `celltype_labels.py` — Abbreviation helpers for plot labels (e.g., `L2/3_Exc` → `E23`).
  - `style.py` — Publication style (`apply_pub_style`) and spine trimming.

- **Weight-vs-activity analyses (Figure 5)**
  - `corr.py` / `corr_mc.py` — Weight-vs-response-correlation matrix plots (simulation and Monte Carlo null).
  - `pd.py` / `pd_mc.py` — Weight-vs-preferred-direction-difference plots (simulation and Monte Carlo null).
  - `pd_effect_size.py` — Cosine-fit effect-size heatmaps for PD analyses.
  - `em_compare.py` — EM (MICrONS) data loaders and comparison plots for both correlation and PD.
  - `stats.py` — OLS regression, binned mean/SEM, cosine-series fitting, Legendre polynomial fitting, piecewise-linear fitting.

- **Cohort and perturbation analyses (Figure 5)**
  - `create_highlow_outgoing_weight_nodesets.py` — Generate high/low outgoing-weight cohort node sets.
  - `create_highlow_outgoing_synapsecount_nodesets.py` — Same for synapse-count-based cohorts.
  - `create_highlow_incoming_weight_nodesets.py` — Same for incoming-weight cohorts.
  - `create_celltype_nodesets.py` / `create_celltype_highlow_nodesets.py` — Cell-type-specific node sets.
  - `generate_fig5_outgoing_silencing_configs.py` — Generate simulation configs for silencing experiments.
  - `outgoing_weight_granular_core_to_core.py` / `outgoing_weight_fraction_table.py` — Granular outgoing-weight target fraction tables.
  - `outgoing_synapsecount_complete_targets_core_to_core.py` — Synapse-count target tables.
  - `calculate_celltype_metrics_optimized.py` — Firing rate and selectivity metrics per cell type for suppression experiments.
  - `calculate_celltype_suppression_metrics.py` / `_fast.py` — Compute suppression metric deltas.
  - `perturbation_metrics.py` / `compute_perturbation_slopes.py` — Quantify perturbation effects.
  - `plot_perturbation_heatmap_figure5.py` — Heatmap of percent-change under silencing.

- **Other utilities**
  - `neuron_features.py` — Extract per-neuron features (rates, selectivity) from simulation output.
  - `osi_boxplot_utils.py` — Helpers for OSI/DSI box-plot styling.
  - `weight_property_cache.py` / `weight_property_survey.py` — Cache and survey weight distributions.
  - `array_utils.py` — Array manipulation helpers.

### `figure_scripts/` — Per-figure panel generators

All scripts are standalone and runnable from the repo root with `conda activate new_v1`.

#### `figure_scripts/figure3/` — Drifting-grating activity (Figure 3)

- `generate_odsi_aggregate_boxplots.py` — Aggregate OSI/DSI/firing-rate box plots across networks.
- `generate_similarity_panels_odsi.py` — Per-metric box plot + similarity heatmap panels.
- `generate_similarity_summary_boxplots_odsi.py` — Summary similarity-score box plots.
- `tune_fig3_firing_rate_boxplot.py` — Fine-tuned firing-rate box plot for the paper.
- `aggregate_boxplots_odsi.py` / `aggregate_similarity_odsi.py` — Thin wrappers calling the generators above.

#### `figure_scripts/figure4/` — Natural-image activity (Figure 4)

- `generate_bo_firing_rate_similarity.py` — Firing-rate similarity (model vs Neuropixels) for Brain Observatory natural images.
- `generate_bo_selectivity_similarity.py` — Selectivity/sparsity similarity panels.

#### `figure_scripts/figure5/` — Synaptic weight changes (Figure 5)

- **Correlation panels**: `generate_corr_final_panel.py`, `generate_corr_sim_full_matrix.py`, `generate_corr_sim_ei2x2.py`, `generate_corr_sim_em_panels.py`, `generate_corr_compare.py`, `generate_corr_full_matrix_fig5style.py`, `generate_corr_final_panel_simple.py`
- **Preferred-direction panels**: `generate_pd_final_panel.py`, `generate_pd_sim_full_matrix.py`, `generate_pd_sim_ei2x2.py`, `generate_pd_sim_em_panels.py`, `generate_pd_effect_size.py`
- **Cohort analyses**: `plot_outgoing_weight_distribution_figure5.py`, `plot_outgoing_synapsecount_distribution_figure5.py`, `plot_target_fraction_figure5.py`, `plot_target_fraction_synapsecount_figure5.py`, `plot_complete_outgoing_stacked.py`
- **Suppression/perturbation**: `plot_celltype_suppression_boxplots.py`, `plot_celltype_suppression_heatmap.py`, `plot_perturbation_heatmap_figure5.py`, `plot_core_rate_boxplots_figure5.py`

#### `figure_scripts/figure6/` — Effect of biological weight constraints (Figure 6)

- `generate_bio_vs_naive_side_by_side.py` — Side-by-side weight-vs-activity plots (bio-trained vs naive).
- `generate_pd_fit_heatmaps_figure6.py` — PD cosine-fit effect-size heatmaps.
- `generate_similarity_panels_odsi_figure6.py` / `generate_similarity_summary_boxplots_odsi_figure6.py` — OSI/DSI similarity panels for unconstrained networks.
- `plot_l5_contrast_responses_figure6.py` — L5 contrast response curves.

#### `figure_scripts/extended_data/` — Extended data figures

- `fig5_corr_full19_matrix.py` — Full 19×19 response-correlation matrix + heatmaps.
- `fig5_pd_full19_matrix.py` — Full 19×19 preferred-direction matrix + heatmaps.
- `generate_extended_boxplots_figure6.py` — Extended box plots for Figure 6 metrics.
- `plot_weight_distribution_boxplots.py` — Synaptic weight distribution box plots (untrained/bio-trained/naive).
- `plot_celltype_suppression_boxplots.py` / `plot_celltype_suppression_heatmap.py` — Extended suppression analyses.
- `plot_contrast_response_extended_fig6.py` — Contrast response line plots (unconstrained networks).
- `contrast_rasters/plot_contrast_rasters.py` — Raster plots at different contrast levels.
- `contrast_response/plot_contrast_response_extended.py` — Contrast response line plots (bio-trained).
- `incoming_cohorts/plot_incoming_weight_boxplots_extended.py` — Incoming-weight cohort analyses.

### `image_decoding/` — Natural-image decoding pipeline (Figure 4)

A self-contained sub-package for image-decoding analyses. See `image_decoding/README.md` for full details.

- **Preprocessing**: `preprocess.py`, `cache_precompute.py` — Convert spike trains to firing-rate matrices.
- **Decoding**: `decode.py`, `evaluate.py`, `run_decoding_cached.py`, `run_celltype_decoding.py` — Multinomial logistic regression and correlation-template decoders with cross-validation.
- **Selectivity/sparsity**: `compute_sparsity_model.py`, `compute_firing_rate_similarity.py`, `compute_selectivity_similarity.py` — Compute image selectivity and distribution similarity.
- **Plotting**: `plot_firing_rate_boxplot.py`, `plot_selectivity_boxplot.py`, `plot_selectivity_violin.py`, `plot_decoding_barplots.py`, `plot_decoding_across_modalities.py`, `plot_selectivity_vs_decoding.py`, `plot_sparsity_comparison.py`
- **Neuropixels reference data**: `neuropixels/cache_precompute_np.py`, `neuropixels/compute_sparsity_np.py`, `neuropixels/run_decoding_np_cached.py`
- **Utilities**: `plot_utils.py` (colors, ordering, shading), `utils.py`, `analysis_pipeline.py`

### Root-level figure-generation scripts

Some legacy/canonical scripts live at the repo root (many have wrappers in `figure_scripts/`):

- `generate_corr_final_panel.py` / `generate_corr_compare.py` / `generate_corr_sim_*.py` — Response-correlation panels.
- `generate_pd_final_panel.py` / `generate_pd_effect_size.py` / `generate_pd_sim_*.py` — Preferred-direction panels.
- `generate_bio_vs_naive_side_by_side.py` — Bio-trained vs naive comparison.
- `generate_em_compare.py` — EM comparison panels.
- `generate_syn_weight_distribution.py` — Synaptic weight distribution plots.
- `generate_perturbation_boxplots.py` — Perturbation box plots.

### `neuropixels/` — Experimental reference data

- `metrics/OSI_DSI_neuropixels_v4.csv` — Neuropixels OSI/DSI metrics used for model validation.

## Notes
In the new V1 model, the LGN coordinates are defined as the visual field coordinates
(elevation and azimuth), though these coordinates are not zero-centered.
The LGN’s elevation axis is oriented upward, aligning with the V1 model’s z-axis;
however, this may conflict with conventional image coordinate systems, where row indices
increase downward. Note that this definition differs from our previous model (Billeh et
al., 2020), which defined the LGN coordinate axis as downward.
Use the y_dir and flip_y options in BMTK to control image orientation when presenting
data to this network.



## Main contributors

* Shinya Ito
* Darrell Haufler
* Kael Dai
