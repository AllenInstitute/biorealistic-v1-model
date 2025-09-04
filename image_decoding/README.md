# Decoding analysis details

Simulation data
- 10 independently-seeded networks (core_nll_0 … core_nll_9).
- Each network was stimulated with the Allen Brain Observatory natural-scenes paradigm: 118 natural images (250 ms each) with no intervals. (gray screen is inserted only at the beginning and the end).
- Responses were recorded for 50 repetitions (“reps”) per network.

Neuron selection and cell-type labelling
- Only “core” neurons (Euclidean distance √(x²+z²) < 200 µm from the center of the simulated cylinder) were included.
- 19 canonical cell types were used and the analyses were performed separately for every cell type

Spike preprocessing
- For each repetition the 250-ms window of every image was binned into a single spike count and divided by 0.25 s to yield firing rates (Hz).

Sub-sampling strategy
- To compare scaling, we drew without replacement n = 10, 30, 50 neurons from the available pool of each cell type (sample_size).
- A new random sample was generated for every (network, cell type, sample_size) combination; the random seed was fixed for reproducibility. (Same set of cells for cross-validation)

Decoder models
1. Multinomial logistic regression (primary)
	- Implemented with scikit-learn 1.4.0 LogisticRegression.
	- Solver = ‘lbfgs’; multi_class = ‘multinomial’; L2 penalty; max_iter = 1000; tol = 1 × 10⁻³.
2. Correlation-template decoder (control) (Double check)
	- For each class the mean response vector was computed on the training set; test images were assigned to the class with the highest Pearson correlation.

Cross-validation
- We performed 10-fold leave-repetition-out CV: for each fold 5 reps were held out for testing and 45 reps used for training.
- Classifier accuracy (fraction of correctly identified images) was averaged across folds to give one score per (network, cell type, sample_size).

Aggregation across networks
- The analysis produced a table with 10 (networks) × 19 (cell types) × 3 (sample sizes) accuracy values.
- Final figures report the mean ± s.e.m. across networks for each cell type and sample size.

Software
- Python 3.11, NumPy 1.26, pandas 2.2, scikit-learn 1.4, joblib 1.3.

## How to reproduce plots (commands)

Assumptions
- Activate your conda env first: `conda activate new_v1`
- Run commands from the repository root: `/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/glif_builder_test/biorealistic-v1-model`
- Colors, ordering, background shading, and legends are centralized in `image_decoding/plot_utils.py`.

### 1) Build cached firing-rate tensors (models)
If not already built for a given `network_type`:

```bash
python -m image_decoding.cache_precompute --networks 0 1 2 3 4 5 6 7 8 9 --network_type bio_trained
python -m image_decoding.cache_precompute --networks 0 1 2 3 4 5 6 7 8 9 --network_type naive
python -m image_decoding.cache_precompute --networks 0 1 2 3 4 5 6 7 8 9 --network_type plain
python -m image_decoding.cache_precompute --networks 0 1 2 3 4 5 6 7 8 9 --network_type adjusted
```

### 2) Compute image selectivity (.csv)
- Neuropixels: `image_decoding/neuropixels/compute_sparsity_np.py` creates:
  - `image_decoding/neuropixels/summary/sparsity_neuropixels_by_unit.csv`
- Models: `image_decoding/compute_sparsity_model.py` appends per-network-type to:
  - `image_decoding/summary/sparsity_model_by_unit.csv`

Example (models):
```bash
python -m image_decoding.compute_sparsity_model --networks 0 1 2 3 4 5 6 7 8 9 --network_type bio_trained
python -m image_decoding.compute_sparsity_model --networks 0 1 2 3 4 5 6 7 8 9 --network_type naive
python -m image_decoding.compute_sparsity_model --networks 0 1 2 3 4 5 6 7 8 9 --network_type plain
python -m image_decoding.compute_sparsity_model --networks 0 1 2 3 4 5 6 7 8 9 --network_type adjusted
```

### 3) Image selectivity plots
- Box (log-scale), with and without Naive:
```bash
python -m image_decoding.plot_selectivity_boxplot --out image_decoding/summary/selectivity_boxplot_log.png
python -m image_decoding.plot_selectivity_boxplot --exclude_naive --out image_decoding/summary/selectivity_boxplot_log_no_naive.png
```
- Violin (Neuropixels vs Bio-trained):
```bash
python -m image_decoding.plot_selectivity_violin --out image_decoding/summary/selectivity_violin.png
```
- Bar comparison (legacy):
```bash
python -m image_decoding.plot_sparsity_comparison --out image_decoding/summary/selectivity_barplot.png
```

### 4) Firing-rate boxplots (Neuropixels + models)
- With and without Naive:
```bash
python -m image_decoding.plot_firing_rate_boxplot --out image_decoding/summary/firing_rate_boxplot.png
python -m image_decoding.plot_firing_rate_boxplot --exclude_naive --out image_decoding/summary/firing_rate_boxplot_no_naive.png
```

### 5) Similarity plots (1 − KS similarity)
Computed per cell type vs Neuropixels; output is a combined figure (boxplot + heatmap).

- Firing rate similarity:
```bash
python -m image_decoding.compute_firing_rate_similarity --exclude_naive --out_combined image_decoding/summary/firing_rate_similarity_combined.png
```
- Image selectivity similarity (no Naive and with Naive):
```bash
python -m image_decoding.compute_selectivity_similarity --exclude_naive --out_combined image_decoding/summary/selectivity_similarity_combined.png
python -m image_decoding.compute_selectivity_similarity --out_combined image_decoding/summary/selectivity_similarity_combined_with_naive.png
```

Notes
- Similarity uses log-rates for FR and raw values for selectivity; metric is 1 − KS.
- Colors/order/legend/shading are reused from `plot_utils.py`.

### 6) Run decoding (per network type)
Generates `image_decoding/summary50/decoding_summary_<network>_by_type.csv` and barplots.

```bash
python -m image_decoding.run_decoding_cached --networks 0 1 2 3 4 5 6 7 8 9 --network_type bio_trained --n_jobs -1 --fit_n_jobs -1 --outdir image_decoding/summary50
python -m image_decoding.run_decoding_cached --networks 0 1 2 3 4 5 6 7 8 9 --network_type naive --n_jobs -1 --fit_n_jobs -1 --outdir image_decoding/summary50
python -m image_decoding.run_decoding_cached --networks 0 1 2 3 4 5 6 7 8 9 --network_type plain --n_jobs -1 --fit_n_jobs -1 --outdir image_decoding/summary50
python -m image_decoding.run_decoding_cached --networks 0 1 2 3 4 5 6 7 8 9 --network_type adjusted --n_jobs -1 --fit_n_jobs -1 --outdir image_decoding/summary50
```

### 7) Decoding plots
- Per-network-type barplots (from CSVs):
```bash
python -m image_decoding.plot_decoding_barplots --summary_dir image_decoding/summary50 --network_types bio_trained naive plain adjusted
```

- Across modalities at sample size 30 (with/without Naive):
```bash
python -m image_decoding.plot_decoding_across_modalities --summary_dir image_decoding/summary50 --out image_decoding/summary50/accuracy_barplot_across_modalities_n30.png
python -m image_decoding.plot_decoding_across_modalities --summary_dir image_decoding/summary50 --exclude_naive --out image_decoding/summary50/accuracy_barplot_across_modalities_n30_no_naive.png
```

### 8) Selectivity vs decoding (scatter)
```bash
python -m image_decoding.plot_selectivity_vs_decoding --out image_decoding/summary/selectivity_vs_decoding.png
```

### 9) Data locations (outputs)
- Selectivity CSVs: `image_decoding/neuropixels/summary/` and `image_decoding/summary/`
- FR/Selectivity plots: `image_decoding/summary/`
- Decoding CSVs and plots: `image_decoding/summary50/`
