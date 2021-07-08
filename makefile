v1nodes: glif_props/v1_node_models.json
	python build_network.py -f --fraction 0.001 -o v1nodes --no-recurrent v1

test: build_network.py
	python build_network.py -f --fraction 0.001 -o test
	
glif_models: prepare_glif_models.py cell_types/cells_with_glif_pop_name.csv
	python prepare_glif_models.py

glif_network: build_network.py
	python build_network.py -o glif_network
	
glif_requisite/glif_models_prop.csv: make_glif_models_prop.py cell_types/cells_with_glif_pop_name.csv
	python make_glif_models_prop.py
	
glif_props/v1_node_models.json: make_glif_requirements.py V1model_seed_file.xlsx glif_requisite/glif_models_prop.csv
	python make_glif_requirements.py

cell_types/cells_with_glif_pop_name.csv: pick_glif_all.py V1model_seed_file.xlsx cell_types/glif_explained_variance_ratio.csv
	python pick_glif_all.py

cell_types/glif_explained_variance_ratio.csv: query_glif_expvar.py
	python query_glif_expvar.py
	
clean:
	rm -r cell_types
	rm -r glif_models
	rm -r glif_props
	rm -r glif_requisite
	