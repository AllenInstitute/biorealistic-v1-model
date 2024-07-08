# set the defalut values for the options to be an empty string
makeopt?=

mainscripts := build_network.py edge_funcs.py node_funcs.py

#whatever is needed for above should go here:
buildfiles := glif_props/v1_node_models.json glif_props/v1_node_models_miniature.json glif_props/lgn_weights_model.csv glif_props/bkg_weights_model.csv base_props/lgn_weights_population.csv glif_models/cell_models

networks = miniature full small tiny forty single flat twenty core diagnose

lgn_node_targets = $(addsuffix /network/bkg_nodes.h5, $(networks))
config_targets = $(addsuffix /configs/config.json, $(networks))
config_filternet_targets = $(addsuffix /configs/config_filternet.json, $(networks))
filternet_targets = $(addsuffix /filternet/spikes.h5, $(networks))
run_targets = $(addsuffix /output/spikes.h5, $(networks))
run_lgn_targets = $(addsuffix /output_lgn/spikes.h5, $(networks))
run_bkg_targets = $(addsuffix /output_bkg/spikes.h5, $(networks))
run_bkgtune_targets = $(addsuffix /output_bkgtune/spikes.h5, $(networks))
run_filternet_bkgtune_targets = $(addsuffix /filternet_bkgtune/spikes.h5, $(networks))
run_lgnbkg_targets = $(addsuffix /output_lgnbkg/spikes.h5, $(networks))
run_nolgn_targets = $(addsuffix /output_nolgn/spikes.h5, $(networks))
run_multimeter_targets = $(addsuffix /output_multimeter/spikes.h5, $(networks))
jobs_8dfilternet_targets = $(addsuffix /jobs/filternet_8dir_10trials.sh, $(networks))
jobs_8d_targets = $(addsuffix /jobs/8dir_10trials.sh, $(networks))
jobs_spont_lgn_targets = $(addsuffix /jobs/spont_lgn_5s.sh, $(networks))
bkg_spikes_targets = $(addsuffix /bkg/bkg_spikes_1kHz_3s.h5, $(networks))
bkg_edge_targets = $(addsuffix /network/bkg_v1_edge_types.csv, $(networks))
run_8dfilternet_targets = $(addsuffix /filternet_8dir_10trials/angle0_trial0/spikes.h5, $(networks))
run_8d_targets = $(addsuffix /8dir_10trials/angle0_trial0/spikes.h5, $(networks))
run_spont_lgn_targets = $(addsuffix /output_spont_lgn_5s/lgn_fr_1.0Hz/spikes.h5, $(networks))
odsi_targets = $(addsuffix /metrics/OSI_DSI_DF.csv, $(networks))
odsi_figure_targets = $(addsuffix /figures/OSI_DSI.png, $(networks))
get_figures_targets = $(addsuffix /figures, $(networks))
components_targets = $(addsuffix /components/synaptic_models/exc_to_e4.json, $(networks))

smallfigure: small/figures
tinybuild: tiny/network/bkg_nodes.h5


$(config_targets): %/configs/config.json: config_templates/config_plain.json
	mkdir -p $*/configs
	cp config_templates/*.json $*/configs
	ln -s config_plain.json $*/configs/config.json

$(config_filternet_targets): %/configs/config_filternet.json: config_templates/config_filternet.json
	# this may not do anything special, but convenient not to rerun filternet everytime
	# when config.json is updated.
	mkdir -p $*/configs
	cp config_templates/config_filternet.json $*/configs/config_filternet.json

$(components_targets): %/components/synaptic_models/exc_to_e4.json: %/network/bkg_nodes.h5 glif_models/synaptic_models/e4_to_e4.json
	python convert_models.py $*

$(filternet_targets): %/filternet/spikes.h5: %/network/bkg_nodes.h5 %/configs/config_filternet.json
	mpirun -np 4 python run_filternet.py $*/configs/config_filternet.json
	
$(run_targets): %/output/spikes.h5: %/network/bkg_nodes.h5 %/configs/config.json %/components/synaptic_models/exc_to_e4.json %/filternet/spikes.h5 %/network/bkg_v1_edge_types.csv %/bkg/bkg_spikes_1kHz_3s.h5
	python run_pointnet.py $*/configs/config.json $(makeopt) -n 8

$(run_lgn_targets): %/output_lgn/spikes.h5: %/network/bkg_nodes.h5 %/configs/config.json %/components/synaptic_models/exc_to_e4.json %/filternet/spikes.h5
	python run_pointnet.py $*/configs/config_lgn.json $(makeopt) -n 8

$(run_bkg_targets): %/output_bkg/spikes.h5: %/network/bkg_nodes.h5 %/configs/config.json %/components/synaptic_models/exc_to_e4.json %/network/bkg_v1_edge_types.csv %/bkg/bkg_spikes_1kHz_3s.h5
	python run_pointnet.py $*/configs/config_bkg.json $(makeopt) -n 8

$(run_bkgtune_targets): %/output_bkgtune/spikes.h5: %/filternet_bkgtune/spikes.h5 %/network/bkg_nodes.h5 %/configs/config.json %/components/synaptic_models/exc_to_e4.json %/network/bkg_v1_edge_types.csv %/bkg/bkg_spikes_1kHz_3s.h5
	python run_pointnet.py $*/configs/config_bkgtune.json -n 8

$(run_filternet_bkgtune_targets): %/filternet_bkgtune/spikes.h5: %/network/bkg_nodes.h5 %/configs/config.json
	mpirun -np 4 python run_filternet.py $*/configs/config_filternet_bkgtune.json

$(run_lgnbkg_targets): %/output_lgnbkg/spikes.h5: %/network/bkg_nodes.h5 %/configs/config.json %/components/synaptic_models/exc_to_e4.json %/filternet/spikes.h5 %/bkg/bkg_spikes_1kHz_3s.h5 %/network/bkg_v1_edge_types.csv
	python run_pointnet.py $*/configs/config_lgnbkg.json $(makeopt) -n 8

$(run_nolgn_targets): %/output_nolgn/spikes.h5: %/network/bkg_nodes.h5 %/configs/config.json %/components/synaptic_models/exc_to_e4.json %/bkg/bkg_spikes_1kHz_3s.h5 %/network/bkg_v1_edge_types.csv
	python run_pointnet.py $*/configs/config_nolgn.json $(makeopt) -n 8

$(run_multimeter_targets): %/output_multimeter/spikes.h5: %/network/bkg_nodes.h5 %/configs/config.json %/components/synaptic_models/exc_to_e4.json %/filternet/spikes.h5 %/network/bkg_v1_edge_types.csv 
	python run_pointnet.py $*/configs/config_multimeter.json $(makeopt) -n 8

$(jobs_8dfilternet_targets): %/jobs/filternet_8dir_10trials.sh: %/configs/config_filternet.json make_filternet_jobs.py
	python make_filternet_jobs.py $* --filternet
	
$(jobs_8d_targets): %/jobs/8dir_10trials.sh: %/configs/config.json make_filternet_jobs.py
	python make_filternet_jobs.py $* $(makeopt)

$(jobs_spont_lgn_targets): %/jobs/spont_lgn_5s.sh: %/configs/config.json make_filternet_jobs.py make_lgn_test_jobs.py
	python make_lgn_test_jobs.py $*

$(bkg_spikes_targets): %/bkg/bkg_spikes_1kHz_3s.h5: %/network/bkg_nodes.h5 bkg_spike_generation.py
	python bkg_spike_generation.py $*

$(bkg_edge_targets): %/network/bkg_v1_edge_types.csv: prepare_lognormal_bkg_weights.py precomputed_props/bkg_v1_edge_types.csv %/network/bkg_nodes.h5 %/components/synaptic_models/exc_to_e4.json 
	cp precomputed_props/bkg_v1_edge_types.csv $*/network/bkg_v1_edge_types.csv

# $(bkg_edge_targets): %/network/bkg_v1_edge_types.csv: prepare_lognormal_bkg_weights.py precomputed_props/bkg_v1_edge_types_fitted.csv %/network/bkg_nodes.h5 %/components/synaptic_models/exc_to_e4.json 
# 	# new scheme (lognormal)
# 	# python prepare_lognormal_bkg_weights.py $*

$(run_8dfilternet_targets): %/filternet_8dir_10trials/angle0_trial0/spikes.h5: %/jobs/filternet_8dir_10trials.sh %/network/bkg_nodes.h5
	# WARNING: Terminaing this command won't stop the jobs running on the cluster.
	#          Make sure you cancel the jobs manually before re-running this.
	ssh -t hpc-login 'cd $(CURDIR); sbatch --wait $*/jobs/filternet_8dir_10trials.sh'

$(run_8d_targets): %/8dir_10trials/angle0_trial0/spikes.h5: %/filternet_8dir_10trials/angle0_trial0/spikes.h5 %/jobs/8dir_10trials.sh %/network/bkg_nodes.h5 run_pointnet.py %/components/synaptic_models/exc_to_e4.json %/bkg/bkg_spikes_1kHz_3s.h5 %/network/bkg_v1_edge_types.csv
	# WARNING: Terminaing this command won't stop the jobs running on the cluster.
	#          Make sure you cancel the jobs manually before re-running this.
	ssh -t hpc-login 'cd $(CURDIR); sbatch --wait $*/jobs/8dir_10trials.sh'
	
$(run_spont_lgn_targets): %/output_spont_lgn_5s/lgn_fr_1.0Hz/spikes.h5: %/jobs/spont_lgn_5s.sh %/network/bkg_nodes.h5 %/network/bkg_v1_edge_types.csv 
	# WARNING: Terminaing this command won't stop the jobs running on the cluster.
	#          Make sure you cancel the jobs manually before re-running this.
	ssh -t hpc-login 'cd $(CURDIR); sbatch --wait $*/jobs/spont_lgn_5s.sh'

$(odsi_targets): %/metrics/OSI_DSI_DF.csv: %/8dir_10trials/angle0_trial0/spikes.h5 calculate_odsi.py
	python calculate_odsi.py $*

$(odsi_figure_targets): %/figures/OSI_DSI.png: %/metrics/OSI_DSI_DF.csv plot_odsi.py
	python plot_odsi.py $*
	
$(get_figures_targets): %/figures: %/figures/OSI_DSI.png
	echo done.
	
flat/network/bkg_nodes.h5: $(mainscripts) $(buildfiles)
	mkdir -p flat
	mpirun -np 4 python build_network.py -f -o flat/network --flat

miniature/network/bkg_nodes.h5: $(mainscripts) $(buildfiles) glif_props/v1_node_models_miniature.json
	mkdir -p miniature
	mpirun -np 4 python build_network.py -f -o miniature/network --miniature 
	
tiny/network/bkg_nodes.h5: $(mainscripts) $(buildfiles)
	mkdir -p tiny
	mpirun -np 4 python build_network.py -f -o tiny/network --fraction 0.005

diagnose/network/bkg_nodes.h5: $(mainscripts) $(buildfiles)
	mkdir -p diagnose
	python build_network.py -f -o diagnose/network --fraction 0.001 --bkg-unit-num 1 --bkg-conn-num 1

small/network/bkg_nodes.h5: $(mainscripts) $(buildfiles)
	mkdir -p small
	mpirun -np 4 python build_network.py -f -o small/network --fraction 0.05 --compression gzip

forty/network/bkg_nodes.h5: $(mainscripts) $(buildfiles)
	# forty is the largest network that can run on a single core.
	# Due to nest limitation, larger network won't run on a single core.
	mkdir -p forty
	ssh -t hpc-login 'cd $(CURDIR); sbatch --wait forty_build.sh'
	
twenty/network/bkg_nodes.h5: $(mainscripts) $(buildfiles)
	mkdir -p twenty
	mpirun -np 8 python build_network.py -f -o twenty/network --fraction 0.2
	
core/network/bkg_nodes.h5: $(mainscripts) $(buildfiles)
	# core is 400 micron diameter network.
	# Due to nest limitation, larger network won't run on a single core.
	mkdir -p core
	mpirun -np 4 python build_network.py -f -o core/network --fraction 0.22145328719723

single/network/bkg_nodes.h5: $(mainscripts) $(buildfiles)
	# single network contains exactly 1 neuron for each model.
	# 201 neurons in total in the network
	mkdir -p single
	mpirun -np 4 python build_network.py -f -o single/network --fraction 0.00001

# full model requires ~400GB RAM. Our personal workstations cannot handle it.
full/network/bkg_nodes.h5: $(mainscripts) $(buildfiles)  # most likely this will fail
	mkdir -p full
	# if you want to run it locally, turn on this option
	# mpirun -np 4 python build_network.py -f -o full/network --fraction 1.00
	#
	# WARNING: Terminaing this command won't stop the jobs running on the cluster.
	#          Make sure you cancel the jobs manually before re-running this.
	ssh -t hpc-login 'cd $(CURDIR); sbatch --wait full_build.sh'
	
full/output/spikes.h5: $(mainscripts)
	ssh -t hpc-login 'cd $(CURDIR); sbatch --wait fullmodel_run.sh'

glif_props/lgn_weights_model.csv: base_props/lgn_weights_population.csv precomputed_props/v1_synapse_amps.json make_lgn_weights.py
	python make_lgn_weights.py --double-alpha

glif_props/bkg_weights_model.csv: base_props/bkg_weights_population_init.csv precomputed_props/v1_synapse_amps.json make_bkg_weights.py
	python make_bkg_weights.py

test: $(mainscripts) $(buildfiles)
	python build_network.py -f --fraction 0.001 -o test
	
profile_build: $(mainscripts) $(buildfiles)
	python -m cProfile -o out.prof build_network.py -f --fraction 0.05 -o profile
	
profile_run: $(mainscripts) $(buildfiles) small/filternet/spikes.h5
	python -m cProfile -o run.prof run_pointnet.py small/configs/config_plain.json

glif_models/cell_models: prepare_glif_models.py cell_types/cells_with_glif_pop_name.csv
	python prepare_glif_models.py
	touch glif_models/cell_models
	# cp -r base_props/synaptic_models glif_models/synaptic_models

glif_models/synaptic_models/e4_to_e4.json: create_syn_models.py
	python create_syn_models.py
	# resolve the error for when the directory already exists

glif_network: build_network.py
	python build_network.py -o glif_network
	
glif_requisite/glif_models_prop.csv: make_glif_models_prop.py cell_types/cells_with_glif_pop_name.csv
	python make_glif_models_prop.py
	
glif_props/v1_node_models.json: make_glif_requirements.py base_props/V1model_seed_file.xlsx glif_requisite/glif_models_prop.csv
	python make_glif_requirements.py --double-alpha

glif_props/v1_node_models_miniature.json: make_glif_requirements.py base_props/V1model_seed_file_miniature.xlsx glif_requisite/glif_models_prop.csv
	python make_glif_requirements.py --miniature --double-alpha

cell_types/cells_with_glif_pop_name.csv: pick_glif_all.py base_props/V1model_seed_file.xlsx cell_types/glif_explained_variance_ratio.csv
	python pick_glif_all.py

cell_types/glif_explained_variance_ratio.csv: query_glif_expvar.py
	python query_glif_expvar.py
	
	
# new bkg adjustment with flat network (optional, precomputed)
# bkg_adjustment: flat/output_bkgtune/spikes.h5 flat/bkg/bkg_spikes_1kHz_3s.h5
#   output will be generates as flat/network/bkg_v1_edge_types.csv
#	python bkg_weight_adjustment_minuit.py
	
# bkg adjustment (renewed), median one
bkg_adjustment: small/output_bkgtune/spikes.h5
#   output will be generates as single/network/bkg_v1_edge_types.csv
	python bkg_weight_adjustment.py

	
clean:
	rm -rf cell_types
	rm -rf glif_models
	rm -rf glif_props
	rm -rf glif_requisite
	rm -rf v1nodes
	rm -rf no_recurrent
	rm -rf miniature
	rm -rf full
	rm -rf small
	rm -rf forty
	rm -rf tiny
	rm -rf no_recurrent_full
	rm -rf profile
	rm -rf out.prof
	
.PHONY: list  # to list all the files. from stackoverflow question # 4219255
list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
