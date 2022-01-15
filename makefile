mainscripts := build_network.py edge_funcs.py node_funcs.py
networks = original_mini miniature

# this may not be necessary
lgn_node_targets = $(addsuffix /network/lgn_nodes.h5, $(networks))
config_targets = $(addsuffix /configs/config_filternet.json, $(networks))
filternet_targets = $(addsuffix /filternet/spikes.h5, $(networks))
jobs_8dfilternet_targets = $(addsuffix /jobs/filternet_8dir_10trials.sh, $(networks))
jobs_8d_targets = $(addsuffix /jobs/8dir_10trials.sh, $(networks))
run_8dfilternet_targets = $(addsuffix /filternet_8dir_10trials/angle0_trial0/spikes.csv, $(networks))
run_8d_targets = $(addsuffix /8dir_10trials/angle0_trial0/spikes.csv, $(networks))
odsi_targets = $(addsuffix /metrics/OSI_DSI_DF.csv, $(networks))
odsi_figure_targets = $(addsuffix /figures/OSI_DSI.png, $(networks))
get_figures_targets = $(addsuffix /figures, $(networks))
lgn_weight_targets = $(addsuffix /network/lgn_v1_edge_types.csv, $(networks))


$(config_targets): %/configs/config_filternet.json: config_templates/config_filternet.json
	mkdir -p $*/configs
	cp config_templates/config_multimeter.json $*/configs/
	cp config_templates/config_plain.json $*/configs/
	cp config_templates/config.json $*/configs/
	cp config_templates/config_filternet.json $*/configs/
	cp config_templates/lgn_v1_population_multiplier.csv $*/configs/

$(lgn_weight_targets): %/network/lgn_v1_edge_types.csv: %/network/lgn_nodes.h5 %/network_nomod/lgn_v1_edge_types.csv %/configs/lgn_v1_population_multiplier.csv modulate_edge_weight.py 
	python modulate_edge_weight.py $*

$(filternet_targets): %/filternet/spikes.h5: %/network/lgn_nodes.h5 %/configs/config_filternet.json
	mpirun -np 8 python run_filternet.py $*/configs/config_filternet.json
	
$(jobs_8dfilternet_targets): %/jobs/filternet_8dir_10trials.sh: %/configs/config_filternet.json make_filternet_jobs.py
	python make_filternet_jobs.py $* --filternet
	
$(jobs_8d_targets): %/jobs/8dir_10trials.sh: %/configs/config_filternet.json make_filternet_jobs.py
	python make_filternet_jobs.py $*

$(run_8dfilternet_targets): %/filternet_8dir_10trials/angle0_trial0/spikes.csv: %/jobs/filternet_8dir_10trials.sh %/network/lgn_nodes.h5 
	ssh -t hpc-login 'cd realistic-model/glif_builder; sbatch --wait $*/jobs/filternet_8dir_10trials.sh'

$(run_8d_targets): %/8dir_10trials/angle0_trial0/spikes.csv: %/network/lgn_v1_edge_types.csv %/filternet_8dir_10trials/angle0_trial0/spikes.csv %/jobs/8dir_10trials.sh %/network/lgn_nodes.h5 run_pointnet.py
	ssh -t hpc-login 'cd realistic-model/glif_builder; sbatch --wait $*/jobs/8dir_10trials.sh'

$(odsi_targets): %/metrics/OSI_DSI_DF.csv: %/8dir_10trials/angle0_trial0/spikes.csv calculate_odsi.py
	python calculate_odsi.py $*

$(odsi_figure_targets): %/figures/OSI_DSI.png: %/metrics/OSI_DSI_DF.csv plot_odsi.py
	python plot_odsi.py $*
	
$(get_figures_targets): %/figures: %/figures/OSI_DSI.png
	echo done.

original_mini/network/lgn_nodes.h5: $(mainscripts) glif_props/v1_node_models_miniature.json
	mkdir -p original_mini
	mpirun -np 4 python build_network.py -f -o original_mini/network --no-recurrent --miniature

# override the config settings for the original network
original_mini/configs/config_filternet.json: config_templates/config_filternet.json
	mkdir -p $*/configs
	cp config_templates/config_multimeter.json $*/configs
	cp config_templates/config_plain.json $*/configs
	cp config_templates/config.json $*/configs
	cp config_templates/config_filternet_original.json $*/configs/config_filternet.json
	cp config_templates/lgn_v1_population_multiplier_original.csv $*/configs/lgn_v1_population_multiplier.csv
	
miniature/network/lgn_nodes.h5: $(mainscripts) glif_props/v1_node_models_miniature.json
	mkdir -p miniature
	# it fails saving if more than 4 cores are used... Let's ask Kael.
	mpirun -np 4 python build_network.py -f -o miniature/network --no-recurrent --miniature --feed-forward-v2
	# duplicate the node/edge type files so that we can adjust the weight retroactively
	mkdir -p miniature/network_nomod
	cp miniature/network/*.csv miniature/network_nomod/
	# remove the file to update... (pretty ad-hoc)
	rm miniature/network/lgn_v1_edge_types.csv

# original_mini_run: original_mini/network/lgn_nodes.h5 original_mini/filternet/spikes.h5
# 	python run_pointnet.py original_mini/configs/config.json



# original_mini/network/lgn_nodes.h5: $(mainscripts) glif_props/v1_node_models_miniature.json
# 	mkdir -p original_mini
# 	python build_network.py -f -o original_mini/network --no-recurrent --miniature
	
# original_mini/configs/config_filternet.json: config_templates/config_filternet.json
# 	mkdir -p original_mini/configs
# 	cp config_templates/config_multimeter.json original_mini/configs
# 	cp config_templates/config_plain.json original_mini/configs
# 	cp config_templates/config.json original_mini/configs
# 	cp config_templates/config_filternet_original.json original_mini/configs/config_filternet.json

# original_mini/filternet/spikes.h5: original_mini/network/lgn_nodes.h5 original_mini/configs/config_filternet.json
# 	mpirun -np 8 python run_filternet.py original_mini/configs/config_filternet.json
	
# original_mini/jobs/filternet_8dir_10trials.sh: original_mini/configs/config_filternet.json make_filternet_jobs.py
# 	python make_filternet_jobs.py original_mini --filternet
	
# original_mini/jobs/8dir_10trials.sh: original_mini/configs/config_filternet.json make_filternet_jobs.py
# 	python make_filternet_jobs.py original_mini --filternet

# original_mini/filternet_8dir_10trials/angle0_trial0/spikes.csv: original_mini/jobs/filternet_8dir_10trials.sh original_mini/network/lgn_nodes.h5 
# 	ssh -t hpc-login 'cd realistic-model/glif_builder; sbatch --wait original_mini/jobs/filternet_8dir_10trials.sh'

# original_mini/8dir_10trials/angle0_trial0/spikes.csv: original_mini/jobs/filternet_8dir_10trials.sh original_mini/network/lgn_nodes.h5 
# 	ssh -t hpc-login 'cd realistic-model/glif_builder; sbatch --wait original_mini/jobs/8dir_10trials.sh'

# original_mini/metrics/OSI_DSI_DF.csv: original_mini/8dir_10trials/angle0_trial0/spikes.csv
# 	python calculate_odsi.py original_mini

# original_mini_run: original_mini/network/lgn_nodes.h5 original_mini/filternet/spikes.h5
# 	python run_pointnet.py original_mini/configs/config.json

no_recurrent: glif_props/v1_node_models.json glif_props/bkg_v1_edge_types.csv $(mainscripts)
	python build_network.py -f --fraction 0.001 -o no_recurrent --no-recurrent

no_recurrent_full: glif_props/v1_node_models.json glif_props/bkg_v1_edge_types.csv $(mainscripts)
	python build_network.py -f --fraction 1 -o no_recurrent_full --no-recurrent

v1nodes: glif_props/v1_node_models.json $(mainscripts)
	python build_network.py -f --fraction 0.001 -o v1nodes --no-recurrent v1
	
# miniature/configs/config_filternet.json: config_templates/config_filternet.json
# 	mkdir -p miniature/configs
# 	cp config_templates/config_multimeter.json miniature/configs
# 	cp config_templates/config_plain.json miniature/configs
# 	cp config_templates/config.json miniature/configs
# 	cp config_templates/config_filternet.json miniature/configs
	
glif_props/lgn_weights_model.csv: base_props/lgn_weights_population.csv base_props/v1_synapse_amps.json make_lgn_weights.py
	python make_lgn_weights.py

	
# miniature/filternet/spikes.h5: miniature/network/lgn_nodes.h5 miniature/configs/config_filternet.json
# 	mpirun -np 8 python run_filternet.py miniature/configs/config_filternet.json
	
# miniature/filternet8d/45.0/spikes.h5: miniature/network/lgn_nodes.h5 configs/config_miniature_filternet.json
# 	mpirun -np 8 python run_filternet.py configs/config_miniature_filternet.json

# miniature_run: miniature/network/lgn_nodes.h5 miniature/configs/config.json
# 	mpirun -np 8 python run_pointnet.py miniature/configs/config.json

test: $(mainscripts)
	python build_network.py -f --fraction 0.001 -o test
	
profile: $(mainscripts)
	python -m cProfile -o out.prof build_network.py -f --fraction 0.01 -o profile --no-recurrent --feed-forward-v2
	
glif_models: prepare_glif_models.py cell_types/cells_with_glif_pop_name.csv base_props/synaptic_models
	python prepare_glif_models.py
	cp -r base_props/synaptic_models glif_models/synaptic_models

glif_network: build_network.py
	python build_network.py -o glif_network
	
glif_requisite/glif_models_prop.csv: make_glif_models_prop.py cell_types/cells_with_glif_pop_name.csv
	python make_glif_models_prop.py
	
glif_props/v1_node_models.json: make_glif_requirements.py base_props/V1model_seed_file.xlsx glif_requisite/glif_models_prop.csv
	python make_glif_requirements.py

glif_props/v1_node_models_miniature.json: make_glif_requirements.py base_props/V1model_seed_file_miniature.xlsx glif_requisite/glif_models_prop.csv
	python make_glif_requirements.py --miniature

glif_props/bkg_v1_edge_types.csv: base_props/bkg_weights_population.csv glif_props/v1_node_models.json
	python make_bkg_weights.py
	
cell_types/cells_with_glif_pop_name.csv: pick_glif_all.py base_props/V1model_seed_file.xlsx cell_types/glif_explained_variance_ratio.csv
	python pick_glif_all.py

cell_types/glif_explained_variance_ratio.csv: query_glif_expvar.py
	python query_glif_expvar.py
	
clean:
	rm -r cell_types
	rm -r glif_models
	rm -r glif_props
	rm -r glif_requisite
	rm -r v1nodes
	rm -r no_recurrent
	rm -r no_recurrent_full
	rm -r profile
	rm out.prof
	
