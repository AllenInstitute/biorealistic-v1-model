mainscripts := build_network.py edge_funcs.py node_funcs.py

no_recurrent: glif_props/v1_node_models.json glif_props/bkg_v1_edge_types.csv $(mainscripts)
	python build_network.py -f --fraction 0.001 -o no_recurrent --no-recurrent

v1nodes: glif_props/v1_node_models.json $(mainscripts)
	python build_network.py -f --fraction 0.001 -o v1nodes --no-recurrent v1
	
test: $(mainscripts)
	python build_network.py -f --fraction 0.001 -o test
	
profile: $(mainscripts)
	python -m cProfile -o out.prof build_network.py -f --fraction 0.001 -o profile --no-recurrent
	
glif_models: prepare_glif_models.py cell_types/cells_with_glif_pop_name.csv
	python prepare_glif_models.py

glif_network: build_network.py
	python build_network.py -o glif_network
	
glif_requisite/glif_models_prop.csv: make_glif_models_prop.py cell_types/cells_with_glif_pop_name.csv
	python make_glif_models_prop.py
	
glif_props/v1_node_models.json: make_glif_requirements.py base_props/V1model_seed_file.xlsx glif_requisite/glif_models_prop.csv
	python make_glif_requirements.py

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
	rm -r profile
	rm out.prof
	
