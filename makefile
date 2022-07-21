mainscripts := build_network.py edge_funcs.py node_funcs.py

#whatever is needed for above should go here:
buildfiles := glif_props/v1_node_models.json glif_props/v1_node_models_miniature.json glif_props/lgn_weights_model.csv glif_props/bkg_weights_model.csv base_props/lgn_weights_population.csv

networks = miniature full small tiny

lgn_node_targets = $(addsuffix /network/lgn_nodes.h5, $(networks))
config_targets = $(addsuffix /configs/config.json, $(networks))
config_filternet_targets = $(addsuffix /configs/config_filternet.json, $(networks))
filternet_targets = $(addsuffix /filternet/spikes.h5, $(networks))
run_targets = $(addsuffix /output/spikes.h5, $(networks))
run_lgn_targets = $(addsuffix /output_lgn/spikes.h5, $(networks))
run_lgnbkg_targets = $(addsuffix /output_lgnbkg/spikes.h5, $(networks))
run_multimeter_targets = $(addsuffix /output_multimeter/spikes.h5, $(networks))
jobs_8dfilternet_targets = $(addsuffix /jobs/filternet_8dir_10trials.sh, $(networks))
jobs_8d_targets = $(addsuffix /jobs/8dir_10trials.sh, $(networks))
bkg_spikes_targets = $(addsuffix /network/bkg/bkg_spikes_1kHz_3s.h5, $(networks))
run_8dfilternet_targets = $(addsuffix /filternet_8dir_10trials/angle0_trial0/spikes.csv, $(networks))
run_8d_targets = $(addsuffix /8dir_10trials/angle0_trial0/spikes.csv, $(networks))
odsi_targets = $(addsuffix /metrics/OSI_DSI_DF.csv, $(networks))
odsi_figure_targets = $(addsuffix /figures/OSI_DSI.png, $(networks))
get_figures_targets = $(addsuffix /figures, $(networks))
components_targets = $(addsuffix /components/synaptic_models/lgn_to_e4.json, $(networks))

smallfigure: small/figures
tinybuild: tiny/network/lgn_nodes.h5

$(config_targets): %/configs/config.json: config_templates/config_plain.json
	mkdir -p $*/configs
	cp config_templates/*.json $*/configs
	ln -s config_plain.json $*/configs/config.json

$(config_filternet_targets): %/configs/config_filternet.json: config_templates/config_filternet.json
	# this may not do anything special, but convenient not to rerun filternet everytime
	# when config.json is updated.
	mkdir -p $*/configs
	cp config_templates/config_filternet.json $*/configs/config_filternet.json

$(components_targets): %/components/synaptic_models/lgn_to_e4.json: %/network/lgn_nodes.h5 glif_models/synaptic_models/e4_to_e4.json
	python convert_models.py $*

$(filternet_targets): %/filternet/spikes.h5: %/network/lgn_nodes.h5 %/configs/config_filternet.json
	mpirun -np 8 python run_filternet.py $*/configs/config_filternet.json
	
$(run_targets): %/output/spikes.h5: %/network/lgn_nodes.h5 %/configs/config.json %/components/synaptic_models/lgn_to_e4.json %/filternet/spikes.h5
	mpirun -np 8 python run_pointnet.py $*/configs/config.json

$(run_lgn_targets): %/output_lgn/spikes.h5: %/network/lgn_nodes.h5 %/configs/config.json %/components/synaptic_models/lgn_to_e4.json %/filternet/spikes.h5
	mpirun -np 8 python run_pointnet.py $*/configs/config_lgn.json

$(run_lgnbkg_targets): %/output_lgnbkg/spikes.h5: %/network/lgn_nodes.h5 %/configs/config.json %/components/synaptic_models/lgn_to_e4.json %/filternet/spikes.h5 %/bkg/bkg_spikes_1kHz_3s.h5
	mpirun -np 8 python run_pointnet.py $*/configs/config_lgnbkg.json

$(run_multimeter_targets): %/output_multimeter/spikes.h5: %/network/lgn_nodes.h5 %/configs/config.json %/components/synaptic_models/lgn_to_e4.json %/filternet/spikes.h5
	mpirun -np 8 python run_pointnet.py $*/configs/config_multimeter.json

$(jobs_8dfilternet_targets): %/jobs/filternet_8dir_10trials.sh: %/configs/config_filternet.json make_filternet_jobs.py
	python make_filternet_jobs.py $* --filternet
	
$(jobs_8d_targets): %/jobs/8dir_10trials.sh: %/configs/config.json make_filternet_jobs.py
	python make_filternet_jobs.py $*

$(bkg_spikes_targets): %/network/bkg/bkg_spikes_1kHz_3s.h5: %/network/lgn_nodes.h5
	python bkg_spike_generation.py $*

$(run_8dfilternet_targets): %/filternet_8dir_10trials/angle0_trial0/spikes.csv: %/jobs/filternet_8dir_10trials.sh %/network/lgn_nodes.h5
	ssh -t hpc-login 'cd $(CURDIR); sbatch --wait $*/jobs/filternet_8dir_10trials.sh'

$(run_8d_targets): %/8dir_10trials/angle0_trial0/spikes.csv: %/filternet_8dir_10trials/angle0_trial0/spikes.csv %/jobs/8dir_10trials.sh %/network/lgn_nodes.h5 run_pointnet.py %/components/synaptic_models/lgn_to_e4.json %/network/bkg/bkg_spikes_1kHz_3s.h5
	ssh -t hpc-login 'cd $(CURDIR); sbatch --wait $*/jobs/8dir_10trials.sh'

$(odsi_targets): %/metrics/OSI_DSI_DF.csv: %/8dir_10trials/angle0_trial0/spikes.csv calculate_odsi.py
	python calculate_odsi.py $*

$(odsi_figure_targets): %/figures/OSI_DSI.png: %/metrics/OSI_DSI_DF.csv plot_odsi.py
	python plot_odsi.py $*
	
$(get_figures_targets): %/figures: %/figures/OSI_DSI.png
	echo done.

miniature/network/lgn_nodes.h5: $(mainscripts) $(buildfiles) glif_props/v1_node_models_miniature.json
	mkdir -p miniature
	mpirun -np 4 python build_network.py -f -o miniature/network --miniature 
	
tiny/network/lgn_nodes.h5: $(mainscripts) $(buildfiles)
	mkdir -p tiny
	mpirun -np 4 python build_network.py -f -o tiny/network --fraction 0.005

small/network/lgn_nodes.h5: $(mainscripts) $(buildfiles)
	mkdir -p small
	mpirun -np 4 python build_network.py -f -o small/network --fraction 0.05

# full model will not be built without a cluster
full/network/lgn_nodes.h5: $(mainscripts) $(buildfiles)  # most likely this will fail
	mkdir -p full
	# mpirun -np 4 python build_network.py -f -o full/network --fraction 1.00
	ssh -t hpc-login 'cd $(CURDIR); sbatch --wait full_build.sh'
	

glif_props/lgn_weights_model.csv: base_props/lgn_weights_population.csv precomputed_props/v1_synapse_amps.json make_lgn_weights.py
	python make_lgn_weights.py

glif_props/bkg_weights_model.csv: base_props/bkg_weights_population_init.csv precomputed_props/v1_synapse_amps.json make_bkg_weights.py
	python make_bkg_weights.py

test: $(mainscripts) $(buildfiles)
	python build_network.py -f --fraction 0.001 -o test
	
profile: $(mainscripts) $(buildfiles)
	python -m cProfile -o out.prof build_network.py -f --fraction 0.05 -o profile 
	
glif_models: prepare_glif_models.py cell_types/cells_with_glif_pop_name.csv base_props/synaptic_models
	python prepare_glif_models.py
	# cp -r base_props/synaptic_models glif_models/synaptic_models

glif_models/synaptic_models/e4_to_e4.json: create_syn_models.py
	python create_syn_models.py
	# resolve the error for when the directory already exists

glif_network: build_network.py
	python build_network.py -o glif_network
	
glif_requisite/glif_models_prop.csv: make_glif_models_prop.py cell_types/cells_with_glif_pop_name.csv
	python make_glif_models_prop.py
	
glif_props/v1_node_models.json: make_glif_requirements.py base_props/V1model_seed_file.xlsx glif_requisite/glif_models_prop.csv
	python make_glif_requirements.py

glif_props/v1_node_models_miniature.json: make_glif_requirements.py base_props/V1model_seed_file_miniature.xlsx glif_requisite/glif_models_prop.csv
	python make_glif_requirements.py --miniature

cell_types/cells_with_glif_pop_name.csv: pick_glif_all.py base_props/V1model_seed_file.xlsx cell_types/glif_explained_variance_ratio.csv
	python pick_glif_all.py

cell_types/glif_explained_variance_ratio.csv: query_glif_expvar.py
	python query_glif_expvar.py
	

# for background tuning	in small network
small/bkg/bkg_spikes_tuning_1khz_10s.h5: bkg_spike_generation.py
	python bkg_spike_generation.py

	
clean:
	rm -rf cell_types
	rm -rf glif_models
	rm -rf glif_props
	rm -rf glif_requisite
	rm -rf v1nodes
	rm -rf no_recurrent
	rm -rf miniature
	rm -rf no_recurrent_full
	rm -rf profile
	rm -rf out.prof
	
.PHONY: list  # to list all the files. from stackoverflow question # 4219255
list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
