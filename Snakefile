rule lgn_weight_model:
    input:
        script="make_lgn_weights.py",
        data=[
            "base_props/lgn_weights_population.csv",
            "precomputed_props/v1_synapse_amps.json"
        ]
    output: "lgn_weights_model.csv"
    shell: "python {input.script} --double-alpha"
    
rule bkg_weight_model:
    input:
        script="make_bkg_weights.py",
        data=[
            "base_props/bkg_weights_population_init.csv",
            "precomputed_props/v1_synapse_amps.json"
        ]
    output: "bkg_weights_model.csv"
    shell: "python {input.script}"
    
rule graph:
    shell: "snakemake --dag | dot -Tpdf > workflow_graph.pdf"