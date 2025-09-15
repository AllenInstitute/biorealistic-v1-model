# contrast_quantification.py
import os
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import network_utils as nu
import stimulus_trials as st


def compute_osi(rates, angles):
    """Compute orientation selectivity index (OSI).

    Parameters
    ----------
    rates : np.ndarray
        Firing rates for each orientation (shape: n_angles,).
    angles : np.ndarray
        Angles in degrees corresponding to the entries in *rates*.

    Returns
    -------
    float
        Orientation selectivity index in the range [0, 1].
    """
    if rates.sum() == 0:
        return np.nan
    rad = np.deg2rad(angles)
    vec = (rates * np.exp(2j * rad)).sum() / rates.sum()
    return np.abs(vec)


def get_color_map():
    """Return a mapping from cell_type (e.g. 'L2/3_Exc') to hex color string."""
    scheme_path = Path("base_props/cell_type_naming_scheme.csv")
    if not scheme_path.exists():
        return {}
    scheme_df = pd.read_csv(scheme_path, delim_whitespace=True, engine="python")
    return dict(zip(scheme_df["cell_type"], scheme_df["hex"]))


def process_one_network(basedir: str, network_option: str, color_map: dict):
    """Generate combined layer-based line plots for a single network and training condition.

    Parameters
    ----------
    basedir : str
        Directory that contains contrast results (e.g. 'core_nll_0').
    network_option : str
        Either 'bio_trained', 'naive', etc.
    color_map : dict
        Mapping from cell_type to hex color for plotting.
    """
    spike_file = f"{basedir}/contrasts_{network_option}/spike_counts.npz"
    if not os.path.isfile(spike_file):
        print(f"[WARN] Spike file not found: {spike_file}. Skipping …")
        return

    data = np.load(spike_file)
    evoked_spikes = data["evoked_spikes"]  # shape (angles, contrasts, trials, cells)
    int_len_ms = float(data["interval_length"])  # milliseconds
    evoked_rates = evoked_spikes * (1000.0 / int_len_ms)  # Hz

    # contrast values are defined in the stimulus helper class
    contrast_stim = st.ContrastStimulus()
    contrast_vals = np.array(contrast_stim.contrasts)

    # Load node metadata
    nodes = nu.load_nodes(basedir, core_radius=200, expand=True)
    nodes_core = nodes[nodes["core"]]

    # orientation angles in degrees (0–315 step 45)
    angles = np.linspace(0, 315, evoked_rates.shape[0])

    save_dir = Path(basedir) / "figures"
    save_dir.mkdir(exist_ok=True)

    # Group cell types by layer
    layer_groups = {
        'L1+L2/3': ['L1_Inh', 'L2/3_Exc', 'L2/3_PV', 'L2/3_SST', 'L2/3_VIP'],
        'L4': ['L4_Exc', 'L4_PV', 'L4_SST', 'L4_VIP'],
        'L5': ['L5_ET', 'L5_IT', 'L5_NP', 'L5_PV', 'L5_SST', 'L5_VIP'],
        'L6': ['L6_Exc', 'L6_PV', 'L6_SST', 'L6_VIP']
    }

    # Collect data for all cell types
    cell_data = {}
    for cell_type, df_type in nodes_core.groupby("cell_type"):
        cell_ids = df_type.index.to_numpy()
        if cell_ids.size == 0:
            continue

        ct_rates = evoked_rates[:, :, :, cell_ids]  # (angles, contrasts, trials, cells)

        # Mean firing rate across orientations, trials, and cells
        mean_rates = ct_rates.mean(axis=(0, 2, 3))  # (contrasts,)

        # Compute OSI for each cell for each contrast
        osi_vals = []
        for c_idx in range(len(contrast_vals)):
            # rates for current contrast -> (angles, trials, cells)
            angle_trial_cell = ct_rates[:, c_idx, :, :]
            # average across trials -> (angles, cells)
            angle_cell = angle_trial_cell.mean(axis=1)
            osi_each = [compute_osi(angle_cell[:, cid], angles) for cid in range(angle_cell.shape[1])]
            osi_vals.append(np.nanmean(osi_each))
        osi_vals = np.array(osi_vals)

        cell_data[cell_type] = {
            'mean_rates': mean_rates,
            'osi_vals': osi_vals,
            'color': color_map.get(cell_type, None)
        }

    # Create combined figures
    # 1. Firing Rate figure
    fig_rate, axes_rate = plt.subplots(2, 2, figsize=(12, 10))
    axes_rate = axes_rate.flatten()
    
    # 2. OSI figure
    fig_osi, axes_osi = plt.subplots(2, 2, figsize=(12, 10))
    axes_osi = axes_osi.flatten()

    for idx, (layer_name, cell_types) in enumerate(layer_groups.items()):
        ax_rate = axes_rate[idx]
        ax_osi = axes_osi[idx]
        
        # Plot each cell type in this layer
        for cell_type in cell_types:
            if cell_type in cell_data:
                data = cell_data[cell_type]
                color = data['color']
                
                # Rate plot
                ax_rate.plot(contrast_vals, data['mean_rates'], 
                           marker='o', label=cell_type, color=color, linewidth=2)
                
                # OSI plot
                ax_osi.plot(contrast_vals, data['osi_vals'], 
                          marker='o', label=cell_type, color=color, linewidth=2)
        
        # Format rate subplot
        ax_rate.set_xlabel('Contrast')
        ax_rate.set_ylabel('Mean firing rate (Hz)')
        ax_rate.set_title(f'{layer_name} - Rate vs. Contrast')
        ax_rate.legend(fontsize=8)
        ax_rate.grid(True, alpha=0.3)
        
        # Format OSI subplot
        ax_osi.set_xlabel('Contrast')
        ax_osi.set_ylabel('Mean OSI')
        ax_osi.set_title(f'{layer_name} - Selectivity vs. Contrast')
        ax_osi.set_ylim(0, 1)
        ax_osi.legend(fontsize=8)
        ax_osi.grid(True, alpha=0.3)

    # Finalize and save figures
    fig_rate.suptitle(f'Firing Rate vs. Contrast ({network_option})', fontsize=14, y=0.98)
    fig_rate.tight_layout()
    fig_rate.savefig(save_dir / f"contrast_rate_by_layer_{network_option}.png", dpi=300, bbox_inches='tight')
    plt.close(fig_rate)
    
    fig_osi.suptitle(f'Orientation Selectivity vs. Contrast ({network_option})', fontsize=14, y=0.98)
    fig_osi.tight_layout()
    fig_osi.savefig(save_dir / f"contrast_selectivity_by_layer_{network_option}.png", dpi=300, bbox_inches='tight')
    plt.close(fig_osi)
    
    print(f"[INFO] Saved combined layer plots for {network_option} in {basedir}/figures")

    # Create filtered figure with both L2/3_VIP and L4_SST on a single panel
    filtered_types = [
        ("L2/3_VIP", "L2/3 VIP"),
        ("L4_SST", "L4 SST"),
    ]
    fig_filt, ax = plt.subplots(1, 1, figsize=(3.5, 3))
    for cell_type, pretty_label in filtered_types:
        if cell_type not in cell_data:
            continue
        d = cell_data[cell_type]
        color = d["color"]
        ax.plot(contrast_vals, d["mean_rates"], marker="o", color=color, linewidth=2, label=pretty_label)
    ax.set_xlabel("Contrast")
    ax.set_ylabel("Mean firing rate (Hz)")
    ax.grid(True, alpha=0.3)
    # Remove top and right spines for publication-quality aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig_filt.tight_layout()
    fig_filt.savefig(save_dir / f"contrast_rate_L23VIP_L4SST_{network_option}.png", dpi=300, bbox_inches="tight")
    plt.close(fig_filt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantify contrast responses (rate & selectivity) for V1 model networks.")
    parser.add_argument("--basedirs", nargs="*", default=None, help="List of network directories. If omitted, all 'core_nll_*' folders in cwd are used.")
    parser.add_argument("--network_options", nargs="*", default=["bio_trained", "naive"], help="Training conditions to evaluate.")
    args = parser.parse_args()

    # Determine directories to process
    if args.basedirs:
        dirs_to_process = args.basedirs
    else:
        dirs_to_process = [d for d in glob.glob("core_nll_*") if os.path.isdir(d)]

    if not dirs_to_process:
        raise RuntimeError("No network directories found to process.")

    color_map = get_color_map()

    for d in dirs_to_process:
        for option in args.network_options:
            process_one_network(d, option, color_map) 