# base_props

This directory contains base properties (constraints) of the model that are manually written.


## File summary table

| Filename | Description | Source data | Obsolete |
| --- | --- | --- | --- |
| b_ratio.csv | Ratio between the highest and lowest connection probability in the orientation like-to-like rule | Billeh et al., 2020 | No |
| filter_components/* | Component files for LGN neurons. | Billeh et al., 2020  | No |
| lgn_fitted_models/* | Model files for LGN neurons. | Billeh et al., 2020  | No |
| lgn_models.csv | Model types definitions for LGN cells | Billeh et al., 2020 | No |
| lgn_params.csv | Parameters for the LGN to V1 connections | Billeh et al., 2020 | No |
| psp_characterization.csv | Results of the log-normal fitting to Post-synaptic potential (PSP) data in synaptic physiology dataset | Campagnola, Seeman et al., 2022 | No |
| psp_lookup_table.csv | Look up table for the psp value to fill in missing entries. | Campagnola, Seeman et al., 2022 | No |
| sigma.csv | Sigma for the spatial connection probability distribution, which is a 2D Gaussian that decays with the lateral distance. | Campagnola, Seeman et al., 2022 | No |
| syn_types_latency.csv | Latency to the onset of the synaptic activation function | Campagnola, Seeman et al., 2022  | No |
| tau_syn_fast.csv | Synaptic time constant of the faster of the two alpha functions in double alpha function. | Campagnola, Seeman et al., 2022  | No |
| tau_syn_slow.csv | Synaptic time constant of the slower of the two alpha functions in double alpha function. | Campagnola, Seeman et al., 2022  | No |
| amp_slow.csv | Relative amplitude of the slower part of the double alpha function. | Campagnola, Seeman et al., 2022 | No |
| exclude_list.csv | List of cells in the cell types database that should be excluded for modeling | Cell types database | No |
| 41593_2019_417_MOESM5_ESM.xlsx | Morphological feature comparison for m-types. From Gouwens et al., 2019. | Gouwens et al., 2019 | No |
| V1model_seed_file.xlsx | Excel file that contains number of cells, layer information for each population | MICrONS dataset, Lee et al., 2010 | No |
| pmax_matrix_v1dd.csv | Connection probability matrix derived from V1DD dataset | V1DD dataset | No |
| bkg_weights_population_init.csv | Initial seed values for bkg weights. This will be overridden by fitted value. | - | No |
| cell_type_naming_scheme.csv | Naming scheme of the cell types in different datasets. | - | No |
| lgn_weights_population.csv | Weights for LGN cells for each population (?) | - | No |
| syn_types_syn_tau.csv | Synaptic time constant for a single alpha function (obsolete) | Campagnola, Seeman et al., 2022  | Yes |
| V1model_seed_file_miniature.xlsx | Excel file for making a miniature version of the network (obsolete) | - | Yes |
