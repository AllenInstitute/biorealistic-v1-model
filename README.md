# biorealistic-v1-model

A project for making biorealistic model of mouse V1.
This is a successor project of making a model of the mouse primary visual cortex
(https://portal.brain-map.org/explore/models/mv1-all-layers)

The improvements will be:
* Connectivity matrix based on the Allen Institute synaptic physiology data (instead from the literature)
* Segregation of the L5 excitatory cells into IT, ET, and NP types.
* More cell models are used for the GLIF version of the model

Right now, it can only generate V1 GLIF nodes. No other structures, no edges.

## file descriptions
base_props/: seed files that are necessary for building the network most of them are in
human readable format and editable (except for the LGN part, which didn't change from
the last version).

base_props/V1model_seed_file.xlsx: An excel file that contains general properties of 
each cell population. Edit this when you want to change which cell population is used 
and their numbers.

base_props/exclude_list.csv: Cells listed in this file will be excluded from the model. The reasons could vary, but generally due to undesired properties in the model, such as spiny inhibitory cells, aspiny excitatory cells, or that the model does not well reproduce cell's activity.

base_props/bkg_weights_population.csv: This file defines synaptic weights from background cell (currently, there is only one entity for background source) to each cell population. The synaptic weights in this file are copied from the previous model (except for L5 which is average of the two types in the previous model), and are parameters of the optimization stage later.


## Main contributors:
* Shinya Ito
* Darrell Haufler
* Kael Dai
