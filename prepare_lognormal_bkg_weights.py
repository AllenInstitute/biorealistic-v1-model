# %%
import sys
import bkg_weight_adjustment_minuit as bwam

# basedir = "small"
# take from the input argument instead
basedir = sys.argv[1]
bkg_files = bwam.backup_files(basedir)
bkg_types, edge_type_ids = bwam.prepare_new_file_structure(*bkg_files)

# init_file = "precomputed_props/bkg_v1_edge_types_fitted_v3_simplex_final.csv"
init_file = "precomputed_props/bkg_v1_edge_types_fitted.csv"
bkg_types = bwam.load_init_params(bkg_types, init_file)
bkg_types = bwam.set_ncells(bkg_types, edge_type_ids)
weights = bwam.generate_new_weights(bkg_types)

bwam.write_weights(bkg_files[0], weights)
