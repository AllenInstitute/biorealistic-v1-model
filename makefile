mainscripts := build_network.py edge_funcs.py node_funcs.py
buildfiles := glif_props/v1_node_models.json glif_props/lgn_weights_model.csv
networks = miniature fullmodel

lgn_node_targets = $(addsuffix /network/lgn_nodes.h5, $(networks))
config_targets = $(addsuffix /configs/config.json, $(networks))
filternet_targets = $(addsuffix /filternet/spikes.h5, $(networks))
run_targets = $(addsuffix /output/spikes.h5, $(networks))
run_lgn_targets = $(addsuffix /output_lgn/spikes.h5, $(networks))
run_lgnbkg_targets = $(addsuffix /output_lgnbkg/spikes.h5, $(networks))
jobs_8dfilternet_targets = $(addsuffix /jobs/filternet_8dir_10trials.sh, $(networks))
jobs_8d_targets = $(addsuffix /jobs/8dir_10trials.sh, $(networks))
run_8dfilternet_targets = $(addsuffix /filternet_8dir_10trials/angle0_trial0/spikes.csv, $(networks))
run_8d_targets = $(addsuffix /8dir_10trials/angle0_trial0/spikes.csv, $(networks))
odsi_targets = $(addsuffix /metrics/OSI_DSI_DF.csv, $(networks))
odsi_figure_targets = $(addsuffix /figures/OSI_DSI.png, $(networks))
get_figures_targets = $(addsuffix /figures, $(networks))
components_targets = $(addsuffix /components/synaptic_models/lgn_2_vip.json, $(networks))

components: miniature/components/synaptic_models/lgn_2_vip.json
build: miniature/network/lgn_nodes.h5

$(config_targets): %/configs/config.json: config_templates/config_plain.json
	mkdir -p $*/configs
	cp config_templates/*.json $*/configs
	#cp config_templates/lgn_v1_population_multiplier.csv $*/configs/
	#cp config_templates/config_multimeter.json $*/configs/
	#cp config_templates/config_plain.json $*/configs/
	#cp config_templates/config_filternet.json $*/configs/
	ln -s config_plain.json $*/configs/config.json

$(components_targets): %/components/synaptic_models/lgn_2_vip.json: %/network/lgn_nodes.h5
	python convert_models.py $*

$(filternet_targets): %/filternet/spikes.h5: %/network/lgn_nodes.h5 %/configs/config.json
	mpirun -np 8 python run_filternet.py $*/configs/config.json
	
$(run_targets): %/output/spikes.h5: %/network/lgn_nodes.h5 %/configs/config.json %/components/synaptic_models/lgn_2_vip.json
	mpirun -np 8 python run_pointnet.py $*/configs/config.json

$(run_lgn_targets): %/output_lgn/spikes.h5: %/network/lgn_nodes.h5 %/configs/config.json %/components/synaptic_models/lgn_2_vip.json
	mpirun -np 8 python run_pointnet.py $*/configs/config_lgn.json

$(run_lgnbkg_targets): %/output_lgnbkg/spikes.h5: %/network/lgn_nodes.h5 %/configs/config.json %/components/synaptic_models/lgn_2_vip.json
	mpirun -np 8 python run_pointnet.py $*/configs/config_lgnbkg.json

$(jobs_8dfilternet_targets): %/jobs/filternet_8dir_10trials.sh: %/configs/config.json make_filternet_jobs.py
	python make_filternet_jobs.py $* --filternet
	
$(jobs_8d_targets): %/jobs/8dir_10trials.sh: %/configs/config.json make_filternet_jobs.py
	python make_filternet_jobs.py $*

$(run_8dfilternet_targets): %/filternet_8dir_10trials/angle0_trial0/spikes.csv: %/jobs/filternet_8dir_10trials.sh %/network/lgn_nodes.h5
	ssh -t hpc-login 'cd $(CURDIR); sbatch --wait $*/jobs/filternet_8dir_10trials.sh'

$(run_8d_targets): %/8dir_10trials/angle0_trial0/spikes.csv: %/filternet_8dir_10trials/angle0_trial0/spikes.csv %/jobs/8dir_10trials.sh %/network/lgn_nodes.h5 run_pointnet.py %/components/synaptic_models/lgn_2_vip.json
	ssh -t hpc-login 'cd $(CURDIR); sbatch --wait $*/jobs/8dir_10trials.sh'

$(odsi_targets): %/metrics/OSI_DSI_DF.csv: %/8dir_10trials/angle0_trial0/spikes.csv calculate_odsi.py
	python calculate_odsi.py $*

$(odsi_figure_targets): %/figures/OSI_DSI.png: %/metrics/OSI_DSI_DF.csv plot_odsi.py
	python plot_odsi.py $*
	
$(get_figures_targets): %/figures: %/figures/OSI_DSI.png
	echo done.

miniature/network/lgn_nodes.h5: $(mainscripts) $(buildfiles) glif_props/v1_node_models_miniature.json
	mkdir -p miniature
	mpirun -np 4 python build_network.py -f -o miniature/network --miniature --feed-forward-v2
	# duplicate the node/edge type files so that we can adjust the weight retroactively
	# This is no longer valid as we swtiched to store weights in h5 files.
	#mkdir -p miniature/network_nomod 
	#cp miniature/network/*.csv miniature/network_nomod/
	# copy optimized background connections
	#cp base_props/bkg_v1_edge_types_optimized.csv miniature/network/bkg_v1_edge_types.csv

glif_props/lgn_weights_model.csv: base_props/lgn_weights_population.csv base_props/v1_synapse_amps.json make_lgn_weights.py
	python make_lgn_weights.py

glif_props/bkg_weights_model.csv: base_props/bkg_weights_population_init.csv base_props/v1_synapse_amps.json make_bkg_weights.py
	python make_bkg_weights.py

test: $(mainscripts)
	python build_network.py -f --fraction 0.001 -o test
	
profile: $(mainscripts)
	python -m cProfile -o out.prof build_network.py -f --fraction 0.1 -o profile --feed-forward-v2
	
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
	rm -rf cell_types
	rm -rf glif_models
	rm -rf glif_props
	rm -rf glif_requisite
	rm -rf v1nodes
	rm -rf no_recurrent
	rm -rf miniature
	rm -rf original_mini
	rm -rf no_recurrent_full
	rm -rf profile
	rm -rf out.prof
	
