# base_props

This directory contains base properties (constraints) of the model that are manually written.

* V1model_seed_file.xlsx: Excel file that contains number of cells, layer information for each population
* lgn_weights_population.csv: Weights for LGN cells for each population
* bkg_weights_nsyns_population.csv: Weights for BKG connections, also contains nsyns (deprecated, also not proper sonata format)
* bkg_weights_population.csv: Weights for BKG connections, nsyns are already multipled to the weights, and therefore fixed to 1.


The bkg files were generated using misc_scripts/extract_bkg_weights.py and manually changed the part for L5 cells (because of the new categorization). 

