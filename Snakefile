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
    "config_checkpoint.json",
    "config_noweightloss.json",
    "config_bio_trained.json",
    "config_bio_free.json",
    "config_uni_reg.json",
    "config_naive.json",
    "config_bkgtune.json",
    "config_filternet.json",
    "config_filternet_bkgtune.json",
    "config_multimeter.json",
]

# available network options
network_options = [
    "plain",
    "adjusted",
    "checkpoint",
    "noweightloss",
    "bio_trained",
    "bio_free",
    "uni_reg",
    "naive",
    "checkpoint_random",
]

network_options_1 = [
    "plain",
    "adjusted",
    "bio_trained",
    "naive",
]

network_options_1d = [
    "adjusted",
    "bio_trained",
    "naive",
]

network_options_2 = [
    "plain",
    "adjusted",
    "bio_trained",
    "naive",
    "bio_free",
    "uni_reg",
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
        "radius": 100.0,
        "core_radius": 50.0,
        "other_options": "--small-lgn"
    },
    "tiny": {
        "radius": 50.0,
        "core_radius": 20.0,
        "other_options": "--small-lgn"
    },
    "small": {
        "radius": 200.0,
        "core_radius": 100.0,
        "other_options": "--small-lgn",
        "memory": 10  # GB, to run on HPC
    },
    "core": {
        "radius": 400.0,
        "core_radius": 200.0,
        "memory": 20  # GB, to run on HPC
    },
    "core_nll": {
        "radius": 400.0,
        "core_radius": 200.0,
        "memory": 20,  # GB, to run on HPC
        "other_options": "-nll"
    },
    "full": {
        "radius": 700.0,
        "core_radius": 400.0,
        "memory": 80  # GB, to run on HPC
    },
    "full_nll": {
        "radius": 700.0,
        "core_radius": 400.0,
        "memory": 80,  # GB, to run on HPC
        "other_options": "-nll"
    }
}

# let's make 10 core networks in addition
for i in range(10):
    networks[f"core_{i}"] = {
        "radius": 400.0,
        "core_radius": 200.0,
        "memory": 20,
        "seed": i * 1000,
        "other_options": "--fluctuate-nneu"
    }
    networks[f"core_nll_{i}"] = {
        "radius": 400.0,
        "core_radius": 200.0,
        "memory": 20,
        "seed": i * 1000,
        "other_options": "--fluctuate-nneu -nll"
    }


# only include available ones
wildcard_constraints:
    # network_name = "^(?!neuropixels$).+",
    # network_option = "^(?!data$).+"
    network_name = "|".join(networks.keys()),
    network_option = "|".join(network_options)


n_threads = 4

# rule to make all the core networks
rule all_cores_nll_response_corr_plot:
    input: expand("core_nll_{i}/figures/response_correlation_{input_type}.png", i=range(10), input_type=network_options_1)

rule all_cores_nll_contrast_plot:
    input: expand("core_nll_{i}/figures/contrast_responsive_cells_{input_type}.pdf", i=range(10), input_type=network_options_1)

rule all_cores_nll_contrast_plot_notf:
    input: expand("core_nll_{i}/figures/contrast_responsive_cells_{input_type}.pdf", i=range(10), input_type=["plain", "adjusted"])

rule all_cores_nll_rasters:
    input: expand("core_nll_{i}/output_{input_type}/raster_and_spectra_by_tuning_angle.png", i=range(10), input_type=network_options_2)

rule all_cores_nll_rasters_notf:
    input: expand("core_nll_{i}/output_{input_type}/raster_and_spectra_by_tuning_angle.png", i=range(10), input_type=["plain", "adjusted"])
    
rule all_core_nll_odsi_plots:
    input: expand("core_nll_{i}/figures/OSI_DSI_{input_type}.png", i=range(10), input_type=network_options_1)

rule all_core_nll_odsi_plots_notf:
    input: expand("core_nll_{i}/figures/OSI_DSI_{input_type}.png", i=range(10), input_type=["plain", "adjusted"])
    
rule all_cores_notf:
    input:
        rules.all_cores_nll_contrast_plot_notf.input,
        rules.all_cores_nll_rasters_notf.input,
        rules.all_core_nll_odsi_plots_notf.input

rule all_cores_all:
    input:
        rules.all_cores_nll_contrast_plot.input,
        rules.all_cores_nll_rasters.input,
        rules.all_core_nll_odsi_plots.input,

rule all_core_nlls_plot:
    input: expand("core_nll_{i}/output_adjusted/raster_and_spectra_by_tuning_angle.png", i=range(10))

rule all_cores:
    input: expand("core_{i}/output_adjusted/spikes.h5", i=range(10))

rule all_cores_plot:
    input: expand("core_{i}/output_adjusted/raster_and_spectra_by_tuning_angle.png", i=range(10))

rule all:
    input: "small/figures/OSI_DSI.png"

rule core_spikes:
    input: "core/output_adjusted/spikes.h5"

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
        # ignore=lambda wildcards: ["glif_props/v1_edge_models.csv", "glif_models/cell_models/313861608_glif_lif_asc_config.json", "gilf_props/bkg_weights_model.csv", "glif_props/v1_node_models.csv", "glif_props/lgn_weights_model.csv"]
    output: ["{network_name}" + name for name in network_files]
    threads: n_threads
    params:
        radius = lambda wildcards: networks[wildcards.network_name]["radius"],
        core_radius = lambda wildcards: networks[wildcards.network_name]["core_radius"],
        seed = lambda wildcards: networks[wildcards.network_name].get("seed", 153),
        other_options = lambda wildcards: networks[wildcards.network_name].get("other_options", "")
    shell:
        "mpirun -np {n_threads} python {input.script[0]} -f -o {wildcards.network_name}/network --radius {params.radius} --core-radius {params.core_radius} --seed {params.seed} {params.other_options}"
    

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
        script=["bkg_spike_generation.py", "stimulus_trials.py"],
        network=[
            "{network_name}/network/bkg_nodes.h5",
            "{network_name}/network/bkg_v1_edge_types.csv",
        ]
    output:
        "{network_name}/bkg/bkg_spikes_250Hz_3s.h5", # representative file
        "{network_name}/bkg/bkg_spikes_250Hz_10s.h5"
    shell: "python {input.script[0]} {wildcards.network_name}"


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


rule output_rasters:
    input:
        script="plot_raster.py",
        data="{network_name}/output{run_opt}/spikes.h5",
    output: "{network_name}/output{run_opt}/raster_and_spectra_by_tuning_angle.png"
    shell: "python {input.script} {wildcards.network_name}/output{wildcards.run_opt}"


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
    output: "{network_name}/jobs/8dir_10trials_{network_option}.sh"
    params: memory=lambda wildcards: networks[wildcards.network_name]["memory"]
    shell: "python {input.script} {wildcards.network_name} --memory {params.memory} --network_option {wildcards.network_option}"


rule filternet_contrast_job:
    input:
        script="make_contrast_jobs.py",
        network=["{network_name}" + name for name in filter_files],
        data="{network_name}/configs/config_filternet.json"
    output: "{network_name}/jobs/filternet_contrasts.sh"
    shell: "python {input.script} {wildcards.network_name} --filternet"
    

rule contrast_job:
    input:
        script=["make_contrast_jobs.py", "stimulus_trials.py"],
        network=["{network_name}" + name for name in network_files],
        adjusted="{network_name}/network/v1_v1_edges_adjusted.h5",
        data="{network_name}/configs/config.json"
    output: "{network_name}/jobs/contrasts_{network_option}.sh"
    params: memory=lambda wildcards: networks[wildcards.network_name]["memory"]
    shell: "python {input.script[0]} {wildcards.network_name} --memory {params.memory} --network_option {wildcards.network_option}"


curdir = os.getcwd()
rule run_filternet_osi_job:
    input:
        script="{network_name}/jobs/filternet_8dir_10trials.sh"
    output: "{network_name}/filternet_8dir_10trials/angle0_trial0/spikes.h5"
    params: curdir=curdir
    shell: "ssh hpc-login 'cd {params.curdir}; sbatch --wait {input.script}'"
    

rule run_osi_job:
    input:
        script="{network_name}/jobs/8dir_10trials_{network_option}.sh",
        data=["{network_name}/filternet_8dir_10trials/angle0_trial0/spikes.h5",
              "{network_name}/bkg/bkg_spikes_250Hz_3s.h5"],
        components="{network_name}/components/synaptic_models/e4_to_e4.json"
    output: "{network_name}/8dir_10trials_{network_option}/angle0_trial0/spikes.h5"
    params: curdir=curdir
    shell: "ssh hpc-login 'cd {params.curdir}; sbatch --wait {input.script}'"


rule run_filternet_contrast_job:
    input:
        script="{network_name}/jobs/filternet_contrasts.sh"
    output: "{network_name}/filternet_contrasts/angle0_contrast0.05_trial0/spikes.h5"
    params: curdir=curdir
    shell: "ssh hpc-login 'cd {params.curdir}; sbatch --wait {input.script}'"

rule run_contrast_job:
    input:
        script="{network_name}/jobs/contrasts_{network_option}.sh",
        network="{network_name}/network/v1_v1_edges_checkpoint.h5",
        data=["{network_name}/filternet_contrasts/angle0_contrast0.05_trial0/spikes.h5",
              "{network_name}/bkg/bkg_spikes_250Hz_3s.h5"],
        components="{network_name}/components/synaptic_models/e4_to_e4.json"
    output: "{network_name}/contrasts_{network_option}/angle0_contrast0.05_trial0/spikes.h5"
    params: curdir=curdir
    shell: "ssh hpc-login 'cd {params.curdir}; sbatch --wait {input.script}'"


rule contrast_spike_aggregation:
    input:
        script="contrast_spike_aggregation.py",
        data="{network_name}/contrasts_{network_option}/angle0_contrast0.05_trial0/spikes.h5"
    output: "{network_name}/contrasts_{network_option}/spike_counts.npz"
    shell: "python {input.script} {wildcards.network_name} {wildcards.network_option}"
    
rule plot_contrast_response:
    input:
        script="contrast_analysis.py",
        data="{network_name}/contrasts_{network_option}/spike_counts.npz"
    output: "{network_name}/figures/contrast_responsive_cells_{network_option}.pdf"
    shell: "python {input.script} {wildcards.network_name} {wildcards.network_option}"

rule odsi_metrics:
    input:
        script="calculate_odsi.py",
        data="{network_name}/8dir_10trials_{network_option}/angle0_trial0/spikes.h5"
    output: "{network_name}/metrics/OSI_DSI_DF_{network_option}.csv"
    shell: "python {input.script} {wildcards.network_name} {wildcards.network_option}"


rule plot_odsi:
    input:
        script="plot_odsi.py",
        data="{network_name}/metrics/OSI_DSI_DF_{network_option}.csv"
    output: "{network_name}/figures/OSI_DSI_{network_option}.png"
    shell: "python {input.script} {wildcards.network_name} {wildcards.network_option}"


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
    shell: "python -m cProfile -o out.prof build_network.py -f -o profile/network --radius 100.0 --core-radius 50.0"
    

rule clear_logs:
    shell: "rm */jobs/logs/*"

rule response_correlation_calculation:
    input:
        network_files=[
            "{network_name}/network/v1_nodes.h5",
            "{network_name}/network/v1_v1_edges.h5"
        ],
        activity_files=[
            "{network_name}/metrics/stim_spikes_output_imagenet_checkpoint.npz",
            "{network_name}/metrics/stim_spikes_output_imagenet.npz"
        ]
    output:
        correlations="{network_name}/metrics/response_correlations_{input_type}.npy",
    threads: 4,
    params:
        base_dir="{network_name}",
        input_type="{input_type}"
    shell:
        """
        python response_correlation_calculations.py --base_dir {params.base_dir} --input_type {params.input_type}
        """

rule calculate_distances:
    input:
        network_files=[
            "{network_name}/network/v1_nodes.h5",
            "{network_name}/network/v1_v1_edges.h5"
        ]
    output:
        distances="{network_name}/metrics/distances.npy"
    params:
        base_dir="{network_name}"
    shell:
        """
        python calculate_distances.py --base_dir {params.base_dir}
        """

rule response_correlation_plot:
    input:
        script="response_correlation_plot.py",
        correlations="{network_name}/metrics/response_correlations_{input_type}.npy",
        distances="{network_name}/metrics/distances.npy"
    output:
        plot="{network_name}/figures/response_correlation_{input_type}.png"
    params:
        base_dir="{network_name}",
        input_type="{input_type}"
    shell:
        """
        python response_correlation_plot.py {params.base_dir} {params.input_type}
        """
        
rule aggregate_correlation_plot:
    input:
        script="aggregate_correlation_plot.py",
        data="core_nll_9/metrics/response_correlations_{input_type}.npy", 
    output:
        "aggregated_response_correlation/{input_type}.png"
    shell:
        """
        mkdir -p aggregated_response_correlation
        python {input.script} core_nll_? {wildcards.input_type} aggregated_response_correlation/{wildcards.input_type}.png
        """

# rule to run all respons_correlation calculations for all networks
rule all_response_correlation_plots:
    input:
        expand("core_nll_{i}/figures/response_correlation_{input_type}.png", i=range(10), input_type=network_options_1),
        expand("aggregated_response_correlation/{input_type}.png", input_type=network_options_1)




# Rule to calculate sparsity measures
rule calculate_sparsity:
    input:
        script="sparsity_calculations.py",
        activity_files=[
            "{base_dir}/metrics/stim_spikes_output_imagenet.npz",
            "{base_dir}/metrics/stim_spikes_output_imagenet_checkpoint.npz"
        ]
    output:
        lifetime_sparsity="{base_dir}/metrics/lifetime_sparsity_{input_type}.npy",
        population_sparsity="{base_dir}/metrics/population_sparsity_{input_type}.npy"
    params:
        base_dir="{base_dir}",
        input_type="{input_type}"
    threads: 8
    shell:
        """
        python {input.script} --base_dir {params.base_dir} --input_type {params.input_type}
        """

# Rule to run calculate_sparsity for all core_nll_? networks
rule all_nll_sparsity:
    input:
        expand("core_nll_{i}/metrics/lifetime_sparsity_{input_type}.npy", i=range(10), input_type=network_options_1),
        expand("core_nll_{i}/metrics/population_sparsity_{input_type}.npy", i=range(10), input_type=network_options_1)

# Rule to plot population sparsity for all core_nll_? networks
rule plot_population_sparsity:
    input:
        expand("core_nll_{i}/metrics/population_sparsity_{input_type}.npy", i=range(10), input_type=network_options_1)
    output:
        "population_sparsity_{input_type}.png"
    params:
        script="plot_sparsity.py",
        base_dirs=["core_nll_{i}" for i in range(10)],
        input_type="{input_type}",
        output_dir="."
    shell:
        """
        python {params.script} --base_dirs {params.base_dirs} --input_type {params.input_type} --output_dir {params.output_dir}
        """

# Rule to plot lifetime sparsity for each core_nll_? network
rule plot_lifetime_sparsity:
    input:
        expand("core_nll_{i}/metrics/lifetime_sparsity_{input_type}.npy", i=range(10), input_type=network_options_1)
    output:
        expand("lifetime_sparsity_core_nll_{i}_{input_type}.png", i=range(10), input_type=network_options_1)
    params:
        script="plot_sparsity.py",
        base_dirs=["core_nll_{i}" for i in range(10)],
        input_type="{input_type}",
        output_dir="."
    shell:
        """
        python {params.script} --base_dirs {params.base_dirs} --input_type {params.input_type} --output_dir {params.output_dir}
        """



# Rule to run spectral analysis for a single network and input type
rule spectral_analysis:
    input:
        script="spectral_analysis2.py",
        data="{network_name}/contrasts_{network_option}/angle0_contrast0.05_trial0/spikes.h5"
    output:
        "{network_name}/contrasts_{network_option}/combined_spectra_700to2700.json"
    threads: 4
    shell:
        """
        python {input.script} \
        --basedir {wildcards.network_name} \
        --subdir contrasts_{wildcards.network_option} \
        --n_processes {threads} \
        """

# Rule to generate spectral plots for a single network and input type
rule spectral_plots:
    input:
        script="spectral_analysis2.py",
        data="{network_name}/contrasts_{network_option}/combined_spectra_700to2700.json"
    output:
        directory("{network_name}/contrasts_{network_option}/figures")
    shell:
        """
        python {input.script} \
        --basedir {wildcards.network_name} \
        --subdir contrasts_{wildcards.network_option} \
        --plot_only
        """

# Rule to run spectral analysis for all core_nll_? networks with all input types
rule all_spectral_analysis:
    input:
        expand("core_nll_{i}/contrasts_{input_type}/combined_spectra_700to2700.json", 
              i=range(10), input_type=network_options_1)

# Rule to generate spectral plots for all core_nll_? networks with all input types
rule all_spectral_plots:
    input:
        expand("core_nll_{i}/contrasts_{input_type}/figures", 
              i=range(10), input_type=network_options_1)

# Rule to aggregate spectral analysis from all networks
rule aggregate_spectra:
    input:
        script="aggregate_spectra.py",
        data=expand("core_nll_{i}/contrasts_{input_type}/combined_spectra_700to2700.json", 
                    i=range(10), input_type=network_options_1)
    output:
        directory("aggregate_spectra")
    shell:
        """
        python {input.script} \
        --base_dirs core_nll_0 core_nll_1 core_nll_2 core_nll_3 core_nll_4 core_nll_5 core_nll_6 core_nll_7 core_nll_8 core_nll_9 \
        --input_types plain bio_trained naive adjusted \
        --output_dir {output}
        """

# Rule to generate normalized aggregate plots
rule aggregate_spectra_normalized:
    input:
        script="aggregate_spectra.py",
        data=expand("core_nll_{i}/contrasts_{input_type}/combined_spectra_700to2700.json", 
                    i=range(10), input_type=network_options_1)
    output:
        directory("aggregate_spectra_normalized")
    shell:
        """
        python {input.script} \
        --base_dirs core_nll_0 core_nll_1 core_nll_2 core_nll_3 core_nll_4 core_nll_5 core_nll_6 core_nll_7 core_nll_8 core_nll_9 \
        --input_types plain bio_trained naive adjusted \
        --output_dir {output} \
        --normalized
        """

# Rule for all spectral analysis tasks
rule all_spectral:
    input:
        rules.all_spectral_analysis.input,
        rules.all_spectral_plots.input,
        rules.aggregate_spectra.output,
        # rules.aggregate_spectra_normalized.output

# Rule to run all analysis for core_nll_0
rule all_core_nll_0_all:
    input:
        expand("core_nll_0/figures/contrast_responsive_cells_{input_type}.pdf", input_type=network_options_1),
        expand("core_nll_0/output_{input_type}/raster_and_spectra_by_tuning_angle.png", input_type=network_options_2),
        expand("core_nll_0/figures/OSI_DSI_{input_type}.png", input_type=network_options_1),
        # Add other relevant outputs for network 0 if needed, for example:
        # expand("core_nll_0/figures/response_correlation_{input_type}.png", input_type=network_options_1),
        # expand("core_nll_0/metrics/lifetime_sparsity_{input_type}.npy", input_type=network_options_1),
        # expand("core_nll_0/metrics/population_sparsity_{input_type}.npy", input_type=network_options_1),
        expand("core_nll_0/contrasts_{input_type}/combined_spectra_700to2700.json", input_type=network_options_1),
        expand("core_nll_0/contrasts_{input_type}/figures", input_type=network_options_1),