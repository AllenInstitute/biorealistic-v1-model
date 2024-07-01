main_scripts = ["build_network.py", "edge_funcs.py", "node_funcs.py"]
build_files = [
    "base_props/lgn_weights_population.csv",
    "glif_props/v1_node_models.json",
    "glif_props/lgn_weights_model.csv",
    "glif_props/bkg_weights_model.csv",
    "glif_models/cell_models"
]

config_files = [
    "config_bkg.json",
    "config_lgn.json",
    "config_lgnbkg.json",
    "config_plain.json",
    "config_bkgtune.json",
    "config_filternet.json",
    "config_filternet_bkgtune.json",
    "config_multimeter.json",
]

network_files = [
    "bkg_node_types.csv",
    "bkg_nodes.h5",
    "lgn_node_types.csv",
    "lgn_nodes.h5",
    "v1_node_types.csv",
    "v1_nodes.h5",
    "bkg_v1_edge_types.csv",
    "bkg_v1_edges.h5",
    "lgn_v1_edge_types.csv",
    "lgn_v1_edges.h5",
    "v1_v1_edge_types.csv",
    "v1_v1_edges.h5",
]
network_files = [f"/network/{file}" for file in network_files]

# drop bkg_v1_edge_types.csv because it will be overridden.
print(network_files)
network_files = network_files[:6] + network_files[7:]
print(network_files)



filter_files = [network_files[2], network_files[3]] # just lgn nodes


networks = {
    "tiny": {
        "fraction": 0.005
    },
    "small": {
        "fraction": 0.05
    },
    "core": {
        "fraction": 0.22145328719723 # creates a 400 µm radius network
    },
    "full": {
        "fraction": 1  # requires a cluster to build
    }
}

n_threads = 8

rule all:
    input: "tiny/output/spikes.h5"


rule synaptic_models:
    input:
        script="create_syn_models.py",
        data=[
            "base_props/tau_syn.csv",
            "base_props/tau_syn_slow.csv",
            "base_props/amp_slow.csv",
        ]
    output: "glif_models/synaptic_models/e4_to_e4.json"  # representative model
    shell: "python {input.script}"

rule glif_explained_variance_ratio:
    input: "query_glif_expvar.py"
    output: "cell_types/glif_explained_variance_ratio.csv"
    shell: "python {input}"


rule cells_with_glif_pop_name:
    input:
        script="pick_glif_all.py",
        data=[
            "base_props/V1model_seed_file.xlsx",
            "cell_types/glif_explained_variance_ratio.csv"
        ]
    output: "cell_types/cells_with_glif_pop_name.csv"
    shell: "python {input}"


rule cell_models:
    input:
        script="prepare_glif_models.py",
        data="cell_types/cells_with_glif_pop_name.csv"
    output: "glif_models/cell_models/313861608_glif_lif_asc_config.json" # representative model
    shell: "python {input.script}"


rule glif_models_prop:
    input:
        script="make_glif_models_prop.py",
        data="cell_types/cells_with_glif_pop_name.csv"
    output:
        "glif_requisite/glif_models_prop.csv"
    shell: "python {input.script}"


rule glif_requirements:
    input:
        script="make_glif_requirements.py",
        data=[
            "base_props/V1model_seed_file.xlsx",
            "glif_requisite/glif_models_prop.csv"
        ]
    output: "glif_props/v1_node_models.json"
    shell: "python {input.script} --double-alpha"


rule lgn_weight_model:
    input:
        script="make_lgn_weights.py",
        data=[
            "base_props/lgn_weights_population.csv",
            "glif_props/v1_node_models.json",
            "precomputed_props/v1_synapse_amps.json"
        ]
    output: "glif_props/lgn_weights_model.csv"
    shell: "python {input.script} --double-alpha"
    

rule bkg_weight_model:
    input:
        script="make_bkg_weights.py",
        data=[
            "base_props/bkg_weights_population_init.csv",
            "precomputed_props/v1_synapse_amps.json"
        ]
    output: "glif_props/bkg_weights_model.csv"
    shell: "python {input.script}"
    
    
rule build_network:
    input:
        script=main_scripts,
        data=build_files
    output: ["{network_name}" + name for name in network_files]
    params:
        fraction = lambda wildcards: networks[wildcards.network_name]["fraction"]
    shell:
        "mpirun -np {n_threads} python {input.script[0]} -f -o {wildcards.network_name}/network --fraction {params.fraction}"
    

rule bkg_edges_override:
    input:
        "precomputed_props/bkg_v1_edge_types.csv",
        "{network_name}/network/bkg_nodes.h5",
        "{network_name}/components/synaptic_models/e4_to_e4.json"
    input:
    output: "{network_name}/network/bkg_v1_edge_types.csv"
    shell: "cp {input[0]} {output}"


rule bkg_spikes:
    input:
        script="bkg_spike_generation.py",
        network=[
            "{network_name}/network/bkg_nodes.h5",
            "{network_name}/network/bkg_v1_edge_types.csv",
        ]
    output: "{network_name}/bkg/bkg_spikes_250Hz_3s.h5" # representative file
    shell: "python {input.script} {wildcards.network_name}"


rule network_synaptic_components:
    input:
        script="convert_models.py",
        data=[
            "{network_name}/network/bkg_nodes.h5",
            "glif_models/synaptic_models/e4_to_e4.json",
        ]
    output: "{network_name}/components/synaptic_models/e4_to_e4.json"
    shell: "python {input.script} {wildcards.network_name}"


rule config_files:
    input: [f"config_templates/{fname}" for fname in config_files]
    output: ["{network_name}" + f"/configs/{fname}" for fname in config_files] + ["{network_name}/configs/config.json"]
    shell: """
        mkdir -p {wildcards.network_name}/configs
        cp config_templates/*.json {wildcards.network_name}/configs
        ln -s config_plain.json {wildcards.network_name}/configs/config.json
    """
    

rule filternet_spikes:
    input:
        script="run_filternet.py",
        network=["{network_name}" + name for name in filter_files],
        config="{network_name}/configs/config_filternet.json"
    output: "{network_name}/filternet/spikes.h5"
    threads: n_threads
    shell: "mpirun -np {n_threads} python {input.script} {input.config}"
    

rule output_spikes:
    input:
        script="run_pointnet.py",
        network=["{network_name}" + name for name in network_files],
        config="{network_name}/configs/config.json",
        components="{network_name}/components/synaptic_models/e4_to_e4.json",
        data=[
            "{network_name}/filternet/spikes.h5",
            "{network_name}/bkg/bkg_spikes_250Hz_3s.h5",
        ]
    output: "{network_name}/output/spikes.h5"
    threads: n_threads
    shell: "python {input.script} {input.config} -n {n_threads}"


rule graph:
    shell: "snakemake --dag | dot -Tpdf > workflow_graph.pdf"