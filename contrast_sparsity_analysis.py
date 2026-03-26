# contrast_sparsity_analysis.py
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
    """Calculate lifetime sparsity across orientations for each cell.
    
    Parameters
    ----------
    rates : np.ndarray
        Firing rates with shape (n_orientations,) for a single cell.
        
    Returns
    -------
    float
        Lifetime sparsity value between 0 and 1.
    """
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
    """Process one network and return sparsity data.
    
    Parameters
    ----------
    basedir : str
        Directory that contains contrast results.
    network_option : str
        Either 'bio_trained', 'naive', etc.
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with sparsity data for each cell type and contrast.
    """
    spike_file = f"{basedir}/contrasts_{network_option}/spike_counts.npz"
    if not os.path.isfile(spike_file):
        print(f"[WARN] Spike file not found: {spike_file}. Skipping …")
        return None

    data = np.load(spike_file)
    evoked_spikes = data["evoked_spikes"]  # shape (angles, contrasts, trials, cells)
    int_len_ms = float(data["interval_length"])
    evoked_rates = evoked_spikes * (1000.0 / int_len_ms)  # Hz

    # contrast values
    contrast_stim = st.ContrastStimulus()
    contrast_vals = np.array(contrast_stim.contrasts)

    # Load node metadata
    nodes = nu.load_nodes(basedir, core_radius=200, expand=True)
    nodes_core = nodes[nodes["core"]]

    results = []
    
    for cell_type, df_type in nodes_core.groupby("cell_type"):
        cell_ids = df_type.index.to_numpy()
        if cell_ids.size == 0:
            continue

        ct_rates = evoked_rates[:, :, :, cell_ids]  # (angles, contrasts, trials, cells)
        
        # For each contrast level
        for c_idx, contrast in enumerate(contrast_vals):
            # Get rates for this contrast: (angles, trials, cells)
            contrast_rates = ct_rates[:, c_idx, :, :]
            # Average across trials: (angles, cells)
            mean_rates = contrast_rates.mean(axis=1)
            
            # Calculate sparsity for each cell
            sparsity_values = []
            for cell_idx in range(mean_rates.shape[1]):
                cell_rates = mean_rates[:, cell_idx]
                sparsity = calculate_lifetime_sparsity(cell_rates)
                sparsity_values.append(sparsity)
            
            # Store results
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


def create_summary_plots(all_data: pd.DataFrame, color_map: dict):
    """Create summary plots across all networks with error bars.
    
    Parameters
    ----------
    all_data : pd.DataFrame
        Combined data from all networks.
    color_map : dict
        Mapping from cell_type to hex color.
    """
    save_dir = Path("figures")
    save_dir.mkdir(exist_ok=True)
    
    # Group cell types by layer
    layer_groups = {
        'L1+L2/3': ['L1_Inh', 'L2/3_Exc', 'L2/3_PV', 'L2/3_SST', 'L2/3_VIP'],
        'L4': ['L4_Exc', 'L4_PV', 'L4_SST', 'L4_VIP'],
        'L5': ['L5_ET', 'L5_IT', 'L5_NP', 'L5_PV', 'L5_SST', 'L5_VIP'],
        'L6': ['L6_Exc', 'L6_PV', 'L6_SST', 'L6_VIP']
    }
    
    for condition in ['bio_trained', 'naive']:
        condition_data = all_data[all_data['condition'] == condition]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (layer_name, cell_types) in enumerate(layer_groups.items()):
            ax = axes[idx]
            
            for cell_type in cell_types:
                cell_data = condition_data[condition_data['cell_type'] == cell_type]
                if cell_data.empty:
                    continue
                
                # Group by contrast and calculate statistics across networks
                contrast_stats = cell_data.groupby('contrast').agg({
                    'mean_sparsity': ['mean', 'std', 'count']
                }).reset_index()
                
                contrast_stats.columns = ['contrast', 'mean', 'std', 'count']
                
                # Calculate standard error
                contrast_stats['sem'] = contrast_stats['std'] / np.sqrt(contrast_stats['count'])
                
                color = color_map.get(cell_type, None)
                
                # Plot with error bars (using SEM)
                ax.errorbar(contrast_stats['contrast'], contrast_stats['mean'], 
                          yerr=contrast_stats['sem'], 
                          marker='o', label=cell_type, color=color, 
                          linewidth=2, capsize=3)
            
            ax.set_xlabel('Contrast')
            ax.set_ylabel('Response Sparseness')
            ax.set_title(f'{layer_name} - Sparsity vs. Contrast')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Response Sparseness vs. Contrast ({condition})', fontsize=16, y=0.98)
        fig.tight_layout()
        fig.savefig(save_dir / f"contrast_sparsity_summary_{condition}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[INFO] Saved summary plot for {condition}")


def create_individual_network_plots(all_data: pd.DataFrame, color_map: dict):
    """Create individual plots for each network showing sparsity by layer.
    
    Parameters
    ----------
    all_data : pd.DataFrame
        Combined data from all networks.
    color_map : dict
        Mapping from cell_type to hex color.
    """
    # Group cell types by layer
    layer_groups = {
        'L1+L2/3': ['L1_Inh', 'L2/3_Exc', 'L2/3_PV', 'L2/3_SST', 'L2/3_VIP'],
        'L4': ['L4_Exc', 'L4_PV', 'L4_SST', 'L4_VIP'],
        'L5': ['L5_ET', 'L5_IT', 'L5_NP', 'L5_PV', 'L5_SST', 'L5_VIP'],
        'L6': ['L6_Exc', 'L6_PV', 'L6_SST', 'L6_VIP']
    }
    
    for network in all_data['network'].unique():
        network_data = all_data[all_data['network'] == network]
        save_dir = Path(network) / "figures"
        save_dir.mkdir(exist_ok=True)
        
        for condition in ['bio_trained', 'naive']:
            condition_data = network_data[network_data['condition'] == condition]
            if condition_data.empty:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for idx, (layer_name, cell_types) in enumerate(layer_groups.items()):
                ax = axes[idx]
                
                for cell_type in cell_types:
                    cell_data = condition_data[condition_data['cell_type'] == cell_type]
                    if cell_data.empty:
                        continue
                    
                    color = color_map.get(cell_type, None)
                    ax.plot(cell_data['contrast'], cell_data['mean_sparsity'], 
                           marker='o', label=cell_type, color=color, linewidth=2)
                
                ax.set_xlabel('Contrast')
                ax.set_ylabel('Response Sparseness')
                ax.set_title(f'{layer_name} - Sparsity vs. Contrast')
                ax.set_ylim(0, 1)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            fig.suptitle(f'Response Sparseness vs. Contrast ({condition})', fontsize=14, y=0.98)
            fig.tight_layout()
            fig.savefig(save_dir / f"contrast_sparsity_by_layer_{condition}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print(f"[INFO] Saved individual plots for {network}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze lifetime sparsity across orientations for contrast responses.")
    parser.add_argument("--basedirs", nargs="*", default=None, help="List of network directories.")
    parser.add_argument("--network_options", nargs="*", default=["bio_trained", "naive"], help="Training conditions.")
    parser.add_argument("--summary_only", action="store_true", help="Only create summary plots, not individual network plots.")
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
        for option in args.network_options:
            result = process_one_network(d, option)
            if result is not None:
                all_results.append(result)
    
    if not all_results:
        print("No data found to process.")
        exit(1)
    
    # Combine all results
    combined_data = pd.concat(all_results, ignore_index=True)
    
    # Create summary plots
    create_summary_plots(combined_data, color_map)
    
    # Create individual network plots unless summary_only is specified
    if not args.summary_only:
        create_individual_network_plots(combined_data, color_map)
    
    print("Analysis complete!") 