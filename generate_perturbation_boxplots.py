#!/usr/bin/env python3
"""Generate perturbation OSI/DSI comparison box plots with curated datasets."""
from __future__ import annotations

from pathlib import Path

from analysis_shared.osi_boxplot_utils import DatasetSpec, load_osi_datasets, plot_box_grid

OUTPUT_DIR = Path("core_nll_0/figures/perturbation_boxplots")

NEUROPIXELS = DatasetSpec(
    label="Neuropixels v4",
    basedir=Path("neuropixels"),
    metric_file="OSI_DSI_neuropixels_v4.csv",
)
BIO_TRAINED = DatasetSpec(
    label="Bio-trained Model",
    basedir=Path("core_nll_0"),
    metric_file="OSI_DSI_DF_bio_trained.csv",
)

ALL_DATASETS = {
    "Inh-selective pos 200": DatasetSpec(
        label="Inh-selective pos 200",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_inh_selective_pos200.csv",
    ),
    "Inh-selective neg 200": DatasetSpec(
        label="Inh-selective neg 200",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_inh_selective_neg200.csv",
    ),
    "Inh-selective pos 100": DatasetSpec(
        label="Inh-selective pos 100",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_inh_selective_pos100.csv",
    ),
    "Inh-selective neg 100": DatasetSpec(
        label="Inh-selective neg 100",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_inh_selective_neg100.csv",
    ),
    "Inh-nonselective pos 100": DatasetSpec(
        label="Inh-nonselective pos 100",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_inh_nonselective_pos100.csv",
    ),
    "Inh-nonselective neg 100": DatasetSpec(
        label="Inh-nonselective neg 100",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_inh_nonselective_neg100.csv",
    ),
    "Inh-nonselective matched pos 100": DatasetSpec(
        label="Inh-nonselective matched pos 100",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_inh_nonselective_matched_pos100.csv",
    ),
    "Inh-nonselective matched neg 100": DatasetSpec(
        label="Inh-nonselective matched neg 100",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_inh_nonselective_matched_neg100.csv",
    ),
    "Exc-selective pos 100": DatasetSpec(
        label="Exc-selective pos 100",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_exc_selective_pos100.csv",
    ),
    "Exc-selective neg 100": DatasetSpec(
        label="Exc-selective neg 100",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_exc_selective_neg100.csv",
    ),
    "Exc-selective matched pos 100": DatasetSpec(
        label="Exc-selective matched pos 100",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_exc_selective_matched_pos100.csv",
    ),
    "Exc-selective matched neg 100": DatasetSpec(
        label="Exc-selective matched neg 100",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_exc_selective_matched_neg100.csv",
    ),
    "Exc-nonselective pos 100": DatasetSpec(
        label="Exc-nonselective pos 100",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_exc_nonselective_pos100.csv",
    ),
    "Exc-nonselective neg 100": DatasetSpec(
        label="Exc-nonselective neg 100",
        basedir="core_nll_0",
        metric_file="OSI_DSI_DF_exc_nonselective_neg100.csv",
    ),
}

PALETTE = {
    "Neuropixels v4": "tab:gray",
    "Bio-trained Model": "tab:green",
    "Inh-selective pos 200": "tab:red",
    "Inh-selective neg 200": "tab:blue",
    "Inh-selective pos 100": "tab:purple",
    "Inh-selective neg 100": "tab:olive",
    "Inh-nonselective pos 100": "tab:orange",
    "Inh-nonselective neg 100": "tab:brown",
    "Inh-nonselective matched pos 100": "tab:pink",
    "Inh-nonselective matched neg 100": "tab:cyan",
    "Exc-selective pos 100": "tab:red",
    "Exc-selective neg 100": "tab:blue",
    "Exc-nonselective pos 100": "tab:orange",
    "Exc-nonselective neg 100": "tab:brown",
    "Exc-selective matched pos 100": "tab:pink",
    "Exc-selective matched neg 100": "tab:cyan",
}

PLOTS = {
    "inhibitory_amp_comparison.png": [
        NEUROPIXELS,
        BIO_TRAINED,
        ALL_DATASETS["Inh-selective pos 200"],
        ALL_DATASETS["Inh-selective neg 200"],
        ALL_DATASETS["Inh-selective pos 100"],
        ALL_DATASETS["Inh-selective neg 100"],
    ],
    "inhibitory_nonmatched_100.png": [
        NEUROPIXELS,
        BIO_TRAINED,
        ALL_DATASETS["Inh-selective pos 100"],
        ALL_DATASETS["Inh-selective neg 100"],
        ALL_DATASETS["Inh-nonselective pos 100"],
        ALL_DATASETS["Inh-nonselective neg 100"],
    ],
    "inhibitory_matched_100.png": [
        NEUROPIXELS,
        BIO_TRAINED,
        ALL_DATASETS["Inh-selective pos 100"],
        ALL_DATASETS["Inh-selective neg 100"],
        ALL_DATASETS["Inh-nonselective matched pos 100"],
        ALL_DATASETS["Inh-nonselective matched neg 100"],
    ],
    "excitatory_nonmatched_100.png": [
        NEUROPIXELS,
        BIO_TRAINED,
        ALL_DATASETS["Exc-selective pos 100"],
        ALL_DATASETS["Exc-selective neg 100"],
        ALL_DATASETS["Exc-nonselective pos 100"],
        ALL_DATASETS["Exc-nonselective neg 100"],
    ],
    "excitatory_matched_100.png": [
        NEUROPIXELS,
        BIO_TRAINED,
        ALL_DATASETS["Exc-selective matched pos 100"],
        ALL_DATASETS["Exc-selective matched neg 100"],
        ALL_DATASETS["Exc-nonselective pos 100"],
        ALL_DATASETS["Exc-nonselective neg 100"],
    ],
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, specs in PLOTS.items():
        print(f"Generating {filename} with {len(specs)} datasets")
        df = load_osi_datasets(specs)
        palette = {spec.label: PALETTE.get(spec.label, None) for spec in specs}
        plot_box_grid(df, OUTPUT_DIR / filename, palette)


if __name__ == "__main__":
    main()
