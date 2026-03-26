# contrast_sparsity_comparison.py
import os
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import network_utils as nu
import stimulus_trials as st


def calculate_lifetime_sparsity(rates):
    """Calculate lifetime sparsity across orientations for each cell."""
    if len(rates) == 0 or np.sum(rates) == 0:
        return np.nan
    
    mean_rate = np.mean(rates)
    mean_squared_rate = np.mean(rates**2)
    
    if mean_rate == 0:
        return np.nan
    
    # Standard lifetime sparsity formula (Rolls & Tovee, 1995)
    sparsity = (1 - (mean_rate**2 / mean_squared_rate)) / (1 - (1 / len(rates)))
    return sparsity


def get_color_map():
    """Return a mapping from cell_type to hex color string."""
    scheme_path = Path("base_props/cell_type_naming_scheme.csv")
    if not scheme_path.exists():
        return {}
    scheme_df = pd.read_csv(scheme_path, sep=r'\s+', engine="python")
    return dict(zip(scheme_df["cell_type"], scheme_df["hex"]))


def process_one_network(basedir: str, network_option: str):
    """Process one network and return sparsity data."""
    spike_file = f"{basedir}/contrasts_{network_option}/spike_counts.npz"
    if not os.path.isfile(spike_file):
        return None

    data = np.load(spike_file)
    evoked_spikes = data["evoked_spikes"]
    int_len_ms = float(data["interval_length"])
    evoked_rates = evoked_spikes * (1000.0 / int_len_ms)

    contrast_stim = st.ContrastStimulus()
    contrast_vals = np.array(contrast_stim.contrasts)

    nodes = nu.load_nodes(basedir, core_radius=200, expand=True)
    nodes_core = nodes[nodes["core"]]

    results = []
    
    for cell_type, df_type in nodes_core.groupby("cell_type"):
        cell_ids = df_type.index.to_numpy()
        if cell_ids.size == 0:
            continue

        ct_rates = evoked_rates[:, :, :, cell_ids]
        
        for c_idx, contrast in enumerate(contrast_vals):
            contrast_rates = ct_rates[:, c_idx, :, :]
            mean_rates = contrast_rates.mean(axis=1)
            
            sparsity_values = []
            for cell_idx in range(mean_rates.shape[1]):
                cell_rates = mean_rates[:, cell_idx]
                sparsity = calculate_lifetime_sparsity(cell_rates)
                sparsity_values.append(sparsity)
            
            results.append({
                'network': basedir,
                'condition': network_option,
                'cell_type': cell_type,
                'contrast': contrast,
                'mean_sparsity': np.nanmean(sparsity_values),
                'std_sparsity': np.nanstd(sparsity_values),
                'n_cells': len(sparsity_values)
            })
    
    return pd.DataFrame(results)


def create_comparison_plot(all_data: pd.DataFrame, color_map: dict):
    """Create side-by-side comparison plot."""
    save_dir = Path("figures")
    save_dir.mkdir(exist_ok=True)
    
    # Group cell types by layer
    layer_groups = {
        'L1+L2/3': ['L1_Inh', 'L2/3_Exc', 'L2/3_PV', 'L2/3_SST', 'L2/3_VIP'],
        'L4': ['L4_Exc', 'L4_PV', 'L4_SST', 'L4_VIP'],
        'L5': ['L5_ET', 'L5_IT', 'L5_NP', 'L5_PV', 'L5_SST', 'L5_VIP'],
        'L6': ['L6_Exc', 'L6_PV', 'L6_SST', 'L6_VIP']
    }
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.5))

    for idx, (layer_name, cell_types) in enumerate(layer_groups.items()):
        ax = axes[idx]

        for cell_type in cell_types:
            color = color_map.get(cell_type, None)

            for condition, linestyle, alpha in [('bio_trained', '-', 0.3), ('naive', '--', 0.15)]:
                condition_data = all_data[(all_data['condition'] == condition) & 
                                           (all_data['cell_type'] == cell_type)]
                if condition_data.empty:
                    continue
                contrast_stats = condition_data.groupby('contrast').agg({
                    'mean_sparsity': ['mean', 'std', 'count']
                }).reset_index()
                contrast_stats.columns = ['contrast', 'mean', 'std', 'count']
                contrast_stats['sem'] = contrast_stats['std'] / np.sqrt(contrast_stats['count'])

                ax.plot(contrast_stats['contrast'], contrast_stats['mean'], 
                        color=color, linestyle=linestyle, linewidth=1.5, markersize=3,
                        marker='o', alpha=1)
                ax.fill_between(contrast_stats['contrast'], 
                                 contrast_stats['mean'] - contrast_stats['sem'],
                                 contrast_stats['mean'] + contrast_stats['sem'],
                                 color=color, alpha=alpha)

        # Formatting
        ax.set_xlabel('Contrast', fontsize=10)
        ax.set_ylabel('Response Sparseness' if idx == 0 else '', fontsize=10)
        ax.set_title(layer_name, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    fig.suptitle('Response Sparseness vs. Contrast: Bio-trained (solid) vs. Naive (dashed)', fontsize=12, y=0.95)
    fig.tight_layout()
    fig.savefig(save_dir / "contrast_sparsity_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[INFO] Saved contrast_sparsity_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare lifetime sparsity between bio-trained and naive conditions.")
    parser.add_argument("--basedirs", nargs="*", default=None, help="List of network directories.")
    args = parser.parse_args()

    # Determine directories to process
    if args.basedirs:
        dirs_to_process = args.basedirs
    else:
        dirs_to_process = [d for d in glob.glob("core_nll_*") if os.path.isdir(d)]

    if not dirs_to_process:
        raise RuntimeError("No network directories found to process.")

    color_map = get_color_map()
    
    # Process all networks
    all_results = []
    for d in dirs_to_process:
        for option in ['bio_trained', 'naive']:
            result = process_one_network(d, option)
            if result is not None:
                all_results.append(result)
    
    if not all_results:
        print("No data found to process.")
        exit(1)
    
    # Combine all results
    combined_data = pd.concat(all_results, ignore_index=True)
    
    # Create comparison plot
    create_comparison_plot(combined_data, color_map)
    
    print("Comparison analysis complete!") 