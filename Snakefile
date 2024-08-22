import os

main_scripts = ["build_network.py", "edge_funcs.py", "node_funcs.py"]
build_files = [
    "base_props/lgn_weights_population.csv",
    "base_props/lgn_params.csv",
    "base_props/lgn_models.csv",
    "glif_props/v1_node_models.csv",
    "glif_props/v1_edge_models.csv",
    "glif_props/lgn_weights_model.csv",
    "glif_props/bkg_weights_model.csv",
    "glif_models/cell_models/313861608_glif_lif_asc_config.json" # representative model
]

config_files = [
    "config_adjusted.json",
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
# print(network_files)
network_files = network_files[:6] + network_files[7:]
# print(network_files)



filter_files = [network_files[2], network_files[3]] # just lgn nodes


networks = {
    "profile": {
        "fraction": 0.05
    },
    "tiny": {
        "fraction": 0.005
    },
    "small": {
        "fraction": 0.05,  # creates ~200 µm radius network
        "core_radius": 100,
        "memory": 10  # GB, to run on HPC
    },
    "core": {
        "fraction": 0.22145328719723,  # creates a 400 µm radius network
        "core_radius": 200,
        "memory": 20  # GB, to run on HPC
    },
    "full": {
        "fraction": 1,  # 850 µm radius.
        "core_radius": 400,
        "memory": 80  # GB, to run on HPC
    }
}

n_threads = 6

rule all:
    input: "small/figures/OSI_DSI.png"

rule core_spikes:
    input: "core/output/spikes.h5"

rule clean:
    shell:
        """
        rm -rf cell_types
        rm -rf glif_models
        rm -rf glif_props
        rm -rf glif_requisite
        rm -rf full
        rm -rf small
        rm -rf core
        rm -rf tiny
        rm -rf profile
        rm -rf out.prof
        """

rule tf_alpha_params:
    input:
        script="extract_tau_syns.py",
        data=[
            "base_props/tau_syn_fast.csv",
            "base_props/tau_syn_slow.csv",
            "base_props/amp_slow.csv",
        ]
    output:
        "tf_props/double_alpha_params.csv",
        "tf_props/double_alpha_params_full.csv",
    shell: "python {input.script}"

rule tf_basis_functions:
    input:
        script="bf_conv.py",
        data=[
            "tf_props/double_alpha_params.csv",
            "tf_props/double_alpha_params_full.csv"
        ]
    output:
        "tf_props/tau_basis.npy",
        "tf_props/basis_function_weights.csv"
    shell: "python {input.script}"
    

rule synaptic_models:
    input:
        script="create_syn_models.py",
        data=[
            "base_props/tau_syn_fast.csv",
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
    output: "glif_props/v1_node_models.csv"
    shell: "python {input.script} --double-alpha"


rule v1_edge_model:
    input:
        script="make_v1_edge_models.py",
        data=[
            "base_props/cell_type_naming_scheme.csv",
            "glif_requisite/glif_models_prop.csv",
            "precomputed_props/v1_synapse_amps.json",
            "base_props/sigma.csv",
            "base_props/pmax_matrix_v1dd.csv",
            "base_props/b_ratio.csv",
            "base_props/psp_lookup_table.csv",
            "base_props/psp_characterization.csv",
            "base_props/syn_types_latency.csv"
        ]
    output: "glif_props/v1_edge_models.csv"
    shell: "python {input.script}"


rule lgn_weight_model:
    input:
        script="make_lgn_weights.py",
        data=[
            "base_props/lgn_weights_population.csv",
            "glif_props/v1_node_models.csv",
            "precomputed_props/v1_synapse_amps.json"
        ]
    output: "glif_props/lgn_weights_model.csv"
    shell: "python {input.script} --double-alpha"
    

rule bkg_weight_model:
    input:
        script="make_bkg_weights.py",
        data=[
            "base_props/bkg_weights_population_init.csv",
            "glif_props/v1_node_models.csv",
            "precomputed_props/v1_synapse_amps.json"
        ]
    output: "glif_props/bkg_weights_model.csv"
    shell: "python {input.script}"
    
    
rule build_network:
    input:
        script=main_scripts,
        data=build_files
    output: ["{network_name}" + name for name in network_files]
    threads: n_threads
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
    output:
        "{network_name}/bkg/bkg_spikes_250Hz_3s.h5", # representative file
        "{network_name}/bkg/bkg_spikes_250Hz_10s.h5"
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
        ln -s config_adjusted.json {wildcards.network_name}/configs/config.json
    """
    

# {opt} can be "" or "_bkgtune"
rule filternet_spikes:
    input:
        script="run_filternet.py",
        network=["{network_name}" + name for name in filter_files],
        config="{network_name}/configs/config_filternet.json"
    output: "{network_name}/filternet/spikes.h5"
    threads: n_threads
    shell: "mpirun -np {n_threads} python {input.script} {input.config}"
    

rule filternet_bkgtune_spikes:
    input:
        script="run_filternet.py",
        network=["{network_name}" + name for name in filter_files],
        config="{network_name}/configs/config_filternet_bkgtune.json"
    output: "{network_name}/filternet_bkgtune/spikes.h5"
    threads: n_threads
    shell: "mpirun -np {n_threads} python {input.script} {input.config}"
 

rule output_spikes:
    input:
        script="run_pointnet.py",
        network=["{network_name}" + name for name in network_files],
        adjusted=lambda wildcards: f"{wildcards.network_name}/network/v1_v1_edges_adjusted.h5" if wildcards.run_opt == "_adjusted" else [],
        config="{network_name}/configs/config{run_opt}.json",
        components="{network_name}/components/synaptic_models/e4_to_e4.json",
        data=[
            "{network_name}/filternet/spikes.h5",
            "{network_name}/bkg/bkg_spikes_250Hz_3s.h5",  # representative file
        ]
    output: "{network_name}/output{run_opt}/spikes.h5"
    threads: n_threads
    shell: "python {input.script} {input.config} -n {n_threads}"


rule output_spikes_bkgtune:
    input:
        script="run_pointnet.py",
        network=["{network_name}" + name for name in network_files],
        config="{network_name}/configs/config_bkgtune.json",
        components="{network_name}/components/synaptic_models/e4_to_e4.json",
        data=[
            "{network_name}/filternet_bkgtune/spikes.h5",
            "{network_name}/bkg/bkg_spikes_250Hz_10s.h5",
        ]
    output: "{network_name}/output_bkgtune/spikes.h5"
    threads: n_threads
    shell: "python {input.script} {input.config} -n {n_threads}"


rule actuation_matrix:
    input:
        script="plot_actuation_matrix.py",
        network=["{network_name}" + name for name in network_files],
        data="neuropixels/metrics/OSI_DSI_DF_data.csv"
    output:
        "{network_name}/metrics/actuation_matrix.csv",
        "{network_name}/figures/actuation_matrix.pdf"
    params:
        core_radius=lambda wildcards: networks[wildcards.network_name]["core_radius"]
    shell: "python {input.script} {wildcards.network_name} -c {params.core_radius}"


rule model_target_current:
    input:
        script="calculate_target_current.py",
        data=[
            "neuropixels/metrics/OSI_DSI_DF_data.csv",
            "glif_requisite/glif_models_prop.csv",
            "glif_models/if_curves_all.csv"
        ]
    output: "glif_models/target_currents.csv"
    shell: "python {input.script}"


rule if_curves:
    input:
        script="calculate_if_curves.py",
        data=[
            "glif_requisite/glif_models_prop.csv",
            "glif_models/cell_models/313861608_glif_lif_asc_config.json"
        ]
    output:
        "glif_models/if_curves_all.csv",
    shell: "python {input.script}"


rule current_adjustment_factor:
    input:
        script="calculate_adjustment_factor.py",
        network="{network_name}/metrics/actuation_matrix.csv",
        data="glif_models/target_currents.csv"
    output: "{network_name}/metrics/modulation.csv"
    shell: "python {input.script} {wildcards.network_name}"


rule recurrent_edge_adjustment:
    input:
        script="make_adjusted_network.py",
        network=["{network_name}" + name for name in network_files],
        data="{network_name}/metrics/modulation.csv"
    output: "{network_name}/network/v1_v1_edges_adjusted.h5"
    shell: "python {input.script} {wildcards.network_name}"


rule filternet_osi_job:
    input:
        script="make_osi_jobs.py",
        network=["{network_name}" + name for name in filter_files],
        data="{network_name}/configs/config_filternet.json"
    output: "{network_name}/jobs/filternet_8dir_10trials.sh"
    shell: "python {input.script} {wildcards.network_name} --filternet"
        

rule osi_job:
    input:
        script="make_osi_jobs.py",
        network=["{network_name}" + name for name in network_files],
        adjusted="{network_name}/network/v1_v1_edges_adjusted.h5",
        data="{network_name}/configs/config.json"
    output: "{network_name}/jobs/8dir_10trials.sh"
    params: memory=lambda wildcards: networks[wildcards.network_name]["memory"]
    shell: "python {input.script} {wildcards.network_name} --memory {params.memory}"


rule filternet_contrast_job:
    input:
        script="make_contrast_jobs.py",
        network=["{network_name}" + name for name in filter_files],
        data="{network_name}/configs/config_filternet.json"
    output: "{network_name}/jobs/filternet_contrasts.sh"
    shell: "python {input.script} {wildcards.network_name} --filternet"
    

rule contrast_job:
    input:
        script="make_contrast_jobs.py",
        network=["{network_name}" + name for name in network_files],
        adjusted="{network_name}/network/v1_v1_edges_adjusted.h5",
        data="{network_name}/configs/config.json"
    output: "{network_name}/jobs/contrasts.sh"
    shell: "python {input.script} {wildcards.network_name}"


curdir = os.getcwd()
rule run_filternet_osi_job:
    input:
        script="{network_name}/jobs/filternet_8dir_10trials.sh"
    output: "{network_name}/filternet_8dir_10trials/angle0_trial0/spikes.h5"
    params: curdir=curdir
    shell: "ssh -t hpc-login 'cd {params.curdir}; sbatch --wait {input.script}'"
    

rule run_osi_job:
    input:
        script="{network_name}/jobs/8dir_10trials.sh",
        data="{network_name}/filternet_8dir_10trials/angle0_trial0/spikes.h5"
    output: "{network_name}/8dir_10trials/angle0_trial0/spikes.h5"
    params: curdir=curdir
    shell: "ssh -t hpc-login 'cd {params.curdir}; sbatch --wait {input.script}'"


rule run_filternet_contrast_job:
    input:
        script="{network_name}/jobs/filternet_contrasts.sh"
    output: "{network_name}/filternet_contrasts/angle0_contrast0.05_trial0/spikes.h5"
    params: curdir=curdir
    shell: "ssh -t hpc-login 'cd {params.curdir}; sbatch --wait {input.script}'"

rule run_contrast_job:
    input:
        script="{network_name}/jobs/contrasts.sh",
        data="{network_name}/filternet_contrasts/angle0_contrast0.05_trial0/spikes.h5"
    output: "{network_name}/contrasts/angle0_contrast0.05_trial0/spikes.h5"
    params: curdir=curdir
    shell: "ssh -t hpc-login 'cd {params.curdir}; sbatch --wait {input.script}'"



rule odsi_metrics:
    input:
        script="calculate_odsi.py",
        data="{network_name}/8dir_10trials/angle0_trial0/spikes.h5"
    output: "{network_name}/metrics/OSI_DSI_DF.csv"
    shell: "python {input.script} {wildcards.network_name}"


rule plot_odsi:
    input:
        script="plot_odsi.py",
        data="{network_name}/metrics/OSI_DSI_DF.csv"
    output: "{network_name}/figures/OSI_DSI.png"
    shell: "python {input.script} {wildcards.network_name}"


rule bkg_adjustment:
    input:
        config="small/output_bkgtune/spikes.h5"
    output: "small/network/bkg_v1_edge_types_adjusted.csv"
    threads: n_threads
    shell: """
        python bkg_weight_adjustment.py
        cp small/network/bkg_v1_edge_types.csv small/network/bkg_v1_edge_types_adjusted.csv
        """


rule graph:
    shell: "snakemake --dag | dot -Tpdf > workflow_graph.pdf"
    

rule profile_build:
    input: ["profile" + name for name in network_files]
    output: "out.prof"
    shell: "python -m cProfile -o out.prof build_network.py -f -o profile/network --fraction 0.05"
    

rule clean_logs:
    shell: "rm */jobs/logs/*"