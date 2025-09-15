# contrast_aggregated_plots.py
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
from scipy import stats


def compute_osi(rates, angles):
    """Compute orientation selectivity index (OSI)."""
    if rates.sum() == 0:
        return np.nan
    rad = np.deg2rad(angles)
    vec = (rates * np.exp(2j * rad)).sum() / rates.sum()
    return np.abs(vec)


def calculate_lifetime_sparsity(rates):
    """Calculate lifetime sparsity across orientations for each cell."""
    if len(rates) == 0 or np.sum(rates) == 0:
        return np.nan
    
    mean_rate = np.mean(rates)
    mean_squared_rate = np.mean(rates**2)
    
    if mean_rate == 0:
        return np.nan
    
    sparsity = (1 - (mean_rate**2 / mean_squared_rate)) / (1 - (1 / len(rates)))
    return sparsity


def get_color_map():
    """Return a mapping from cell_type to hex color string."""
    scheme_path = Path("base_props/cell_type_naming_scheme.csv")
    if not scheme_path.exists():
        return {}
    scheme_df = pd.read_csv(scheme_path, sep=r'\s+', engine="python")
    return dict(zip(scheme_df["cell_type"], scheme_df["hex"]))


def process_one_network_complete(basedir: str, network_option: str):
    """Process one network and return complete data (rate, OSI, sparsity)."""
    spike_file = f"{basedir}/contrasts_{network_option}/spike_counts.npz"
    if not os.path.isfile(spike_file):
        return None

    data = np.load(spike_file)
    evoked_spikes = data["evoked_spikes"]  # shape (angles, contrasts, trials, cells)
    int_len_ms = float(data["interval_length"])
    evoked_rates = evoked_spikes * (1000.0 / int_len_ms)

    contrast_stim = st.ContrastStimulus()
    contrast_vals = np.array(contrast_stim.contrasts)
    angles = np.linspace(0, 315, evoked_rates.shape[0])

    nodes = nu.load_nodes(basedir, core_radius=200, expand=True)
    nodes_core = nodes[nodes["core"]]

    results = []
    
    for cell_type, df_type in nodes_core.groupby("cell_type"):
        cell_ids = df_type.index.to_numpy()
        if cell_ids.size == 0:
            continue

        ct_rates = evoked_rates[:, :, :, cell_ids]  # (angles, contrasts, trials, cells)
        
        for c_idx, contrast in enumerate(contrast_vals):
            # Get rates for this contrast: (angles, trials, cells)
            contrast_rates = ct_rates[:, c_idx, :, :]
            # Average across trials: (angles, cells)
            mean_rates = contrast_rates.mean(axis=1)
            
            # Calculate metrics for each cell
            rate_values = []
            osi_values = []
            sparsity_values = []
            
            for cell_idx in range(mean_rates.shape[1]):
                cell_rates = mean_rates[:, cell_idx]
                
                # Mean firing rate across orientations
                mean_rate = np.mean(cell_rates)
                rate_values.append(mean_rate)
                
                # OSI
                osi = compute_osi(cell_rates, angles)
                osi_values.append(osi)
                
                # Sparsity
                sparsity = calculate_lifetime_sparsity(cell_rates)
                sparsity_values.append(sparsity)
            
            # Store results
            results.append({
                'network': basedir,
                'condition': network_option,
                'cell_type': cell_type,
                'contrast': contrast,
                'mean_rate': np.nanmean(rate_values),
                'mean_osi': np.nanmean(osi_values),
                'mean_sparsity': np.nanmean(sparsity_values),
                'n_cells': len(rate_values)
            })
    
    return pd.DataFrame(results)


def create_shaded_plots(all_data: pd.DataFrame, color_map: dict):
    """Create plots with error shading showing variability across networks."""
    save_dir = Path("figures")
    save_dir.mkdir(exist_ok=True)
    
    # Group cell types by layer
    layer_groups = {
        'L1+L2/3': ['L1_Inh', 'L2/3_Exc', 'L2/3_PV', 'L2/3_SST', 'L2/3_VIP'],
        'L4': ['L4_Exc', 'L4_PV', 'L4_SST', 'L4_VIP'],
        'L5': ['L5_ET', 'L5_IT', 'L5_NP', 'L5_PV', 'L5_SST', 'L5_VIP'],
        'L6': ['L6_Exc', 'L6_PV', 'L6_SST', 'L6_VIP']
    }
    
    metrics = ['mean_rate', 'mean_osi', 'mean_sparsity']
    metric_labels = ['Mean Firing Rate (Hz)', 'Mean OSI', 'Response Sparseness']
    metric_ylims = [None, (0, 1), (0, 1)]
    
    for condition in ['bio_trained', 'naive']:
        condition_data = all_data[all_data['condition'] == condition]
        
        for metric_idx, (metric, ylabel, ylim) in enumerate(zip(metrics, metric_labels, metric_ylims)):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for layer_idx, (layer_name, cell_types) in enumerate(layer_groups.items()):
                ax = axes[layer_idx]
                
                for cell_type in cell_types:
                    cell_data = condition_data[condition_data['cell_type'] == cell_type]
                    if cell_data.empty:
                        continue
                    
                    # Group by contrast and calculate statistics across networks
                    contrast_stats = cell_data.groupby('contrast').agg({
                        metric: ['mean', 'std', 'count']
                    }).reset_index()
                    
                    contrast_stats.columns = ['contrast', 'mean', 'std', 'count']
                    
                    # Calculate confidence intervals (95% CI using t-distribution)
                    contrast_stats['sem'] = contrast_stats['std'] / np.sqrt(contrast_stats['count'])
                    contrast_stats['ci'] = contrast_stats.apply(
                        lambda row: stats.t.ppf(0.975, row['count']-1) * row['sem'] 
                        if row['count'] > 1 else row['sem'], axis=1
                    )
                    
                    color = color_map.get(cell_type, None)
                    
                    # Plot line with shaded error region
                    ax.plot(contrast_stats['contrast'], contrast_stats['mean'], 
                           color=color, linewidth=1.5, label=cell_type)
                    
                    ax.fill_between(contrast_stats['contrast'], 
                                   contrast_stats['mean'] - contrast_stats['ci'],
                                   contrast_stats['mean'] + contrast_stats['ci'],
                                   color=color, alpha=0.2)
                
                ax.set_xlabel('Contrast', fontsize=10)
                ax.set_ylabel(ylabel if layer_idx == 0 else '', fontsize=10)
                ax.set_title(f'{layer_name}', fontsize=11)
                if ylim is not None:
                    ax.set_ylim(ylim)
                ax.grid(True, alpha=0.3)
                ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
                for spine in ['top', 'right']:
                    ax.spines[spine].set_visible(False)
            
            # Set overall title based on metric
            if metric == 'mean_rate':
                title = f'Firing Rate vs. Contrast ({condition})'
            elif metric == 'mean_osi':
                title = f'Orientation Selectivity vs. Contrast ({condition})'
            else:
                title = f'Response Sparseness vs. Contrast ({condition})'
            
            fig.suptitle(title, fontsize=16, y=0.98)
            fig.tight_layout()
            
            # Save with appropriate filename
            if metric == 'mean_rate':
                filename = f"contrast_rate_aggregated_{condition}.png"
            elif metric == 'mean_osi':
                filename = f"contrast_osi_aggregated_{condition}.png"
            else:
                filename = f"contrast_sparsity_aggregated_{condition}.png"
            
            fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"[INFO] Saved {filename}")

    # After all metrics processed for this condition, create legend figure once
    create_celltype_legend(color_map)


def create_comparison_shaded_plots(all_data: pd.DataFrame, color_map: dict):
    """Create comparison plots with shaded error regions."""
    save_dir = Path("figures")
    save_dir.mkdir(exist_ok=True)
    
    # Group cell types by layer
    layer_groups = {
        'L1+L2/3': ['L1_Inh', 'L2/3_Exc', 'L2/3_PV', 'L2/3_SST', 'L2/3_VIP'],
        'L4': ['L4_Exc', 'L4_PV', 'L4_SST', 'L4_VIP'],
        'L5': ['L5_ET', 'L5_IT', 'L5_NP', 'L5_PV', 'L5_SST', 'L5_VIP'],
        'L6': ['L6_Exc', 'L6_PV', 'L6_SST', 'L6_VIP']
    }
    
    metrics = ['mean_rate', 'mean_osi', 'mean_sparsity']
    metric_labels = ['Mean Firing Rate (Hz)', 'Mean OSI', 'Response Sparseness']
    metric_ylims = [None, (0, 1), (0, 1)]
    
    for metric_idx, (metric, ylabel, ylim) in enumerate(zip(metrics, metric_labels, metric_ylims)):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for layer_idx, (layer_name, cell_types) in enumerate(layer_groups.items()):
            ax = axes[layer_idx]
            
            for cell_type in cell_types:
                color = color_map.get(cell_type, None)
                
                for condition_idx, condition in enumerate(['bio_trained', 'naive']):
                    condition_data = all_data[all_data['condition'] == condition]
                    cell_data = condition_data[condition_data['cell_type'] == cell_type]
                    if cell_data.empty:
                        continue
                    
                    # Group by contrast and calculate statistics
                    contrast_stats = cell_data.groupby('contrast').agg({
                        metric: ['mean', 'std', 'count']
                    }).reset_index()
                    
                    contrast_stats.columns = ['contrast', 'mean', 'std', 'count']
                    contrast_stats['sem'] = contrast_stats['std'] / np.sqrt(contrast_stats['count'])
                    contrast_stats['ci'] = contrast_stats.apply(
                        lambda row: stats.t.ppf(0.975, row['count']-1) * row['sem'] 
                        if row['count'] > 1 else row['sem'], axis=1
                    )
                    
                    # Different line styles for conditions
                    linestyle = '-' if condition == 'bio_trained' else '--'
                    alpha = 0.3 if condition == 'bio_trained' else 0.15
                    
                    label = f"{cell_type} ({condition})" if layer_idx == 0 else None
                    
                    # Plot line with shaded error region
                    ax.plot(contrast_stats['contrast'], contrast_stats['mean'], 
                           color=color, linewidth=2, linestyle=linestyle, label=label)
                    
                    ax.fill_between(contrast_stats['contrast'], 
                                   contrast_stats['mean'] - contrast_stats['ci'],
                                   contrast_stats['mean'] + contrast_stats['ci'],
                                   color=color, alpha=alpha)
            
            ax.set_xlabel('Contrast')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{layer_name}')
            if ylim is not None:
                ax.set_ylim(ylim)
            # No legend in main figure; handled separately
            ax.grid(True, alpha=0.3)
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
        
        # Set overall title based on metric
        if metric == 'mean_rate':
            title = 'Firing Rate vs. Contrast: Bio-trained (solid) vs. Naive (dashed)'
        elif metric == 'mean_osi':
            title = 'Orientation Selectivity vs. Contrast: Bio-trained (solid) vs. Naive (dashed)'
        else:
            title = 'Response Sparseness vs. Contrast: Bio-trained (solid) vs. Naive (dashed)'
        
        fig.suptitle(title, fontsize=14, y=0.98)
        fig.tight_layout()
        
        # Save with appropriate filename
        if metric == 'mean_rate':
            filename = "contrast_rate_comparison_shaded.png"
        elif metric == 'mean_osi':
            filename = "contrast_osi_comparison_shaded.png"
        else:
            filename = "contrast_sparsity_comparison_shaded.png"
        
        fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[INFO] Saved {filename}")


def create_celltype_legend(color_map: dict):
    """Create a standalone legend image with all cell types."""
    save_dir = Path("figures")
    save_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 0.5))
    handles = []
    labels = []
    for cell_type, color in color_map.items():
        handle, = ax.plot([], [], marker='o', color=color, linestyle='None', markersize=6)
        handles.append(handle)
        labels.append(cell_type)
    ax.axis('off')
    leg = ax.legend(handles, labels, ncol=8, fontsize=6, frameon=False)
    fig.savefig(save_dir / "cell_type_legend.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[INFO] Saved cell_type_legend.png")


def create_filtered_rate_plot(all_data: pd.DataFrame, color_map: dict, condition: str = 'bio_trained'):
    """Create a single-panel aggregated rate plot with shaded error for L2/3_VIP and L4_SST."""
    save_dir = Path("figures")
    save_dir.mkdir(exist_ok=True)

    wanted = ['L2/3_VIP', 'L4_SST']
    condition_data = all_data[(all_data['condition'] == condition) & (all_data['cell_type'].isin(wanted))]
    if condition_data.empty:
        print(f"[WARN] No data for condition {condition} and requested cell types {wanted}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.0))

    for cell_type in wanted:
        cell_data = condition_data[condition_data['cell_type'] == cell_type]
        if cell_data.empty:
            continue

        contrast_stats = cell_data.groupby('contrast').agg({
            'mean_rate': ['mean', 'std', 'count']
        }).reset_index()
        contrast_stats.columns = ['contrast', 'mean', 'std', 'count']
        contrast_stats['sem'] = contrast_stats['std'] / np.sqrt(contrast_stats['count'])
        contrast_stats['ci'] = contrast_stats.apply(
            lambda row: stats.t.ppf(0.975, row['count']-1) * row['sem'] if row['count'] > 1 else row['sem'], axis=1
        )

        color = color_map.get(cell_type, None)
        label = 'L2/3 VIP' if cell_type == 'L2/3_VIP' else 'L4 SST'

        ax.plot(contrast_stats['contrast'], contrast_stats['mean'], color=color, linewidth=2, marker='', label=label)
        ax.fill_between(contrast_stats['contrast'],
                        contrast_stats['mean'] - contrast_stats['ci'],
                        contrast_stats['mean'] + contrast_stats['ci'],
                        color=color, alpha=0.25)

    ax.set_xlabel('Contrast')
    ax.set_ylabel('Mean Firing Rate (Hz)')
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.grid(True, alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()

    outname = save_dir / f"contrast_rate_aggregated_L23VIP_L4SST_{condition}.png"
    fig.savefig(outname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved {outname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create aggregated plots with error shading for contrast responses.")
    parser.add_argument("--basedirs", nargs="*", default=None, help="List of network directories.")
    parser.add_argument("--network_options", nargs="*", default=["bio_trained", "naive"], help="Training conditions.")
    parser.add_argument("--comparison_only", action="store_true", help="Only create comparison plots.")
    parser.add_argument("--filtered_only", action="store_true", help="Only create filtered L2/3 VIP and L4 SST aggregated rate plot.")
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
            result = process_one_network_complete(d, option)
            if result is not None:
                all_results.append(result)
    
    if not all_results:
        print("No data found to process.")
        exit(1)
    
    # Combine all results
    combined_data = pd.concat(all_results, ignore_index=True)
    
    if args.filtered_only:
        create_filtered_rate_plot(combined_data, color_map, condition='bio_trained')
    elif args.comparison_only:
        # Create only comparison plots
        create_comparison_shaded_plots(combined_data, color_map)
    else:
        # Create both individual condition plots and comparison plots
        create_shaded_plots(combined_data, color_map)
        create_comparison_shaded_plots(combined_data, color_map)
    
    print("Aggregated analysis complete!") 