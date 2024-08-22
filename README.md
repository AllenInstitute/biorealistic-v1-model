# biorealistic-v1-model

A project for making biorealistic model of mouse V1.
This is a successor project of making a model of the mouse primary visual cortex
(<https://portal.brain-map.org/explore/models/mv1-all-layers>)

The improvements will be:

* Connection probability and weights are derived from coherent datasets, including the Allen Institute synaptic physiology data and MICrONS electron microscopy connectomics dataset (instead from the literature).
* Synaptic connections are now expressed by double alpha functions and the receptor types are more elaborated.
* Segregation of the L5 excitatory cells into IT, ET, and NP types.
* More GLIF cell models are used.

Only PointNet version is available as of now.

## Installation instruction

First, clone this repository to your local environment.

```bash
git clone https://github.com/AllenInstitute/biorealistic-v1-model.git
```

I suggest using [miniforge](https://github.com/conda-forge/miniforge) if you are setting up the environment for this. (Other conda variants will also work, if you specify 'conda-forge' as the primary package source.)

The 'conda' command can be replaced with 'mamba' if you have it installed, and it's much faster than 'conda'.

```bash
conda env create -f environment.yml -n <env_name>
```

## How to build and run a network (incl. network adjustment)

The most straightforward way to get the final adjusted network and it's output is doing the following.

```bash
snakemake <network_name>/output_adjusted/spikes.h5
```

where `<network_name>` is one of the defined names in `Snakefile`. Namely:
- `full`: The full size network (850 µm radius, ~280k neurons)
- `core`: 400 µm radius network. ~65k neurons
- `small`: For testing. Contains 5% of neurons of the `full` network.
- `tiny`: 0.5% network. Mainly for testing the build script.
- `profile`: For profiling workflows. 5% network.

You can create a custom size if you define parameters in `Snakefile`.

Also, you can edit `V1model_seed_file.xlsx` to change what cell types are be included. For example, if you want to make L4 only network, you can delete all the cell types other than L4 from this file.

Building the `full` model requires a few hundreds of GBs of memory. It usually needs to be run on a cluster computer, but it is not incorporated in the snakemake workflow yet. If you really need the `full` network, running the build script should be done manually. The subsequent processes should work fine once you build the model.

## Folders and files

base_props/: A folder that contains seed files that are necessary for building the network. These are mostly designed to be human readable and editable.

base_props/V1model_seed_file.xlsx: An excel file that contains general properties of
each cell population. Edit this when you want to change which cell population is used
and their numbers.

base_props/exclude_list.csv: Cells listed in this file will be excluded from the model. The reasons could vary, but generally due to undesired properties in the model, such as spiny inhibitory cells, aspiny excitatory cells, or that the model does not well reproduce cell's activity.

base_props/bkg_weights_population.csv: This file defines synaptic weights from background cell (currently, there is only one entity for background source) to each cell population. The synaptic weights in this file are copied from the previous model (except for L5 which is average of the two types in the previous model), and are parameters of the optimization stage later.

Preferred data storage option is CSV format, as long as the data are not large. All the csv files are 'space' separarated (to match with SONATA format). Larger data can be stored as h5 data frame. All should be readable in pandas.

## Main contributors

* Shinya Ito
* Darrell Haufler
* Kael Dai
