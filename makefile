test: build_network.py
	python build_network.py -f --fraction 0.001 -o test

glif_network: build_network.py
	python build_network.py -o glif_network
	
glif_props/v1_node_models.json: make_glif_requirements.py
	python make_glif_requirements.py

cell_types/cells_with_glif_pop_name.csv: pick_glif_all.py
	python pick_glif_all.py

cell_types/glif_explained_variance_ratio.csv: query_glif_expvar.py
	python query_glif_expvar.py
	
clean:
	rm -r glif_models
	