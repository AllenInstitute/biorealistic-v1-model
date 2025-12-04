# %% make one raster plot for the network with power spectra

import argparse

parser = argparse.ArgumentParser(description="Make a raster plot of the network with power spectra.")
parser.add_argument(
    "outputdir",
    type=str,
    help="The output folder to plot that contains spikes file. e.g. core/output_adjusted.",
)
parser.add_argument(
    "--sortby",
    "-s",
    type=str,
    default="tuning_angle",
    help="The variable to sort by. Available options are 'tuning_angle', 'node_type_ids', 'x', 'y', etc...",
)
parser.add_argument(
    "--grouping",
    choices=["full", "four"],
    default="four",
    help="Legend/cell-type grouping: 'full' = all types, 'four' = Exc, PV, SST, VIP (L1 gray)",
)
parser.add_argument(
    "--legend-markerscale",
    type=float,
    default=10,
    help="Increase legend marker size (scatter only). If not set and --grouping four, defaults to 4.",
)
parser.add_argument(
    "--raster-only",
    action="store_true",
    help="Only render the raster panel (no spectra)",
)
parser.add_argument(
    "--fontsize",
    type=float,
    default=10.0,
    help="Base font size for labels and legend.",
)
parser.add_argument(
    "--full-network",
    action="store_true",
    help="Use full network (no core radius selection).",
)
args = parser.parse_args()
sortby = args.sortby
grouping = args.grouping
legend_markerscale = args.legend_markerscale
only_raster = args.raster_only
base_fontsize = args.fontsize
use_full_network = args.full_network

# delayed import for faster help response
import plotting_utils as pu
import matplotlib.pyplot as plt
import os
import pathlib
import numpy as np
import pandas as pd
from scipy import signal
import network_utils as nu

# Evoked period parameters
evoked_start = 700  # ms
evoked_end = 2700   # ms
window_size = 1000  # ms for Welch's method

# Function to calculate power spectra using Welch's method
def calculate_power_spectrum(spike_times, n_neurons, fs=1000, nperseg=1000):
    # Create a binary spike train (0s and 1s)
    bins = np.arange(evoked_start, evoked_end + 1, 1)  # 1ms bins
    spike_counts = np.histogram(spike_times, bins=bins)[0]
    
    # Normalize by number of neurons
    if n_neurons > 0:
        spike_counts = spike_counts / n_neurons
    
    # Calculate power spectrum using Welch's method
    f, Pxx = signal.welch(spike_counts, fs=fs, nperseg=nperseg)
    
    return f, Pxx

# Function to group spikes by layer and cell type using cell_type table
def group_spikes_by_layer_and_type(spike_df, v1df):
    # Get cell type table
    ctdf = nu.get_cell_type_table()
    
    # Extract layer and cell type from pop_name
    v1df = v1df.copy()
    v1df['cell_type'] = pd.Series(ctdf.loc[v1df['pop_name']]['cell_type'].values, index=v1df.index)
    
    # Extract layer from pop_name
    layer_map = {1: 'L1', 2: 'L2/3', 4: 'L4', 5: 'L5', 6: 'L6'}
    v1df['layer_num'] = v1df['pop_name'].str.extract(r'[ei](\d)').astype(int)
    v1df['layer'] = v1df['layer_num'].map(layer_map)
    
    # For L5 E cells, identify subtypes (IT, NP, ET)
    is_l5_exc = (v1df['layer'] == 'L5') & (v1df['cell_type'].str.startswith('Exc'))
    
    # Function to determine L5 E subtypes
    def get_l5_subtype(pop_name):
        if 'IT' in pop_name:
            return 'E-IT'
        elif 'NP' in pop_name:
            return 'E-NP'
        elif 'ET' in pop_name:
            return 'E-ET'
        else:
            return 'E-IT'  # Default to IT if no specific marker
    
    # Apply the function to L5 E cells
    v1df.loc[is_l5_exc, 'cell_type'] = v1df.loc[is_l5_exc, 'pop_name'].apply(get_l5_subtype)
    
    # Map other cell types to simplified representation
    type_map = {
        'Exc': 'E',
        'Pvalb': 'PV',
        'Sst': 'SST', 
        'Vip': 'VIP'
    }
    
    # Apply mapping but preserve the L5 E subtypes
    non_l5_exc = ~is_l5_exc
    for old, new in type_map.items():
        v1df.loc[non_l5_exc & v1df['cell_type'].str.startswith(old), 'cell_type'] = new
    
    # Group by layer and cell type
    grouped = v1df.groupby(['layer', 'cell_type'])
    
    # Calculate power spectra for each group
    power_spectra = {}
    for (layer, cell_type), group in grouped:
        if layer not in power_spectra:
            power_spectra[layer] = {}
            
        # Filter spikes for this group during evoked period
        neuron_ids = group.index
        n_neurons = len(neuron_ids)
        
        if n_neurons > 0:
            group_spikes = spike_df[
                (spike_df.index.isin(neuron_ids)) & 
                (spike_df['timestamps'] >= evoked_start) & 
                (spike_df['timestamps'] < evoked_end)
            ]
            
            if not group_spikes.empty:
                # Calculate power spectrum normalized by neuron count
                f, Pxx = calculate_power_spectrum(group_spikes['timestamps'], n_neurons)
                power_spectra[layer][cell_type] = (f, Pxx, n_neurons)
    
    return power_spectra

# Set up figure
# Robustly pick a config file from the output directory
out_dir = pathlib.Path(args.outputdir)
candidate_configs = [
    out_dir / "config.json",
    out_dir / "config_bio_trained.json",
]
config_file = None
for c in candidate_configs:
    if c.exists():
        config_file = str(c)
        break
if config_file is None:
    # Fallback: first config*.json in the directory
    cfgs = sorted(out_dir.glob("config*.json"))
    if cfgs:
        config_file = str(cfgs[0])
    else:
        raise FileNotFoundError(f"No config json found in {out_dir}")
orig_net_name = pathlib.Path(args.outputdir).parts[0]
net = orig_net_name
if "core" in orig_net_name:
    net = "core"  # used only to index settings

# Create figure with subplots: top for raster, bottom grid for power spectra
plt.rcParams.update({
    "font.size": base_fontsize,
    "axes.titlesize": base_fontsize,
    "axes.labelsize": base_fontsize,
    "legend.fontsize": base_fontsize,
    "legend.title_fontsize": base_fontsize,
    "xtick.labelsize": base_fontsize,
    "ytick.labelsize": base_fontsize,
})
fig = plt.figure(figsize=(10, 12 if not only_raster else 6))

# Get spike data and network info
net_obj = pu.form_network(config_file, infer=True)
spike_df = pu.get_spikes(config_file, infer=True)
v1df = net_obj.nodes["v1"].to_dataframe()
# Handle radius selection: special-case core networks
is_core_net = ("core" in orig_net_name)
if is_core_net:
    # For 'core' networks: core=200 µm, full=400 µm
    desired_radius = 400.0 if use_full_network else 200.0
    v1df = pu.pick_core(v1df, radius=desired_radius)
else:
    # Other networks: use settings radius only when not full-network
    if not use_full_network:
        v1df = pu.pick_core(v1df, radius=pu.settings.get(net, {}).get('radius', 400.0))

# Extract L1 inhibitory neurons for special handling
v1df_copy = v1df.copy()
ctdf = nu.get_cell_type_table()
v1df_copy['cell_type'] = pd.Series(ctdf.loc[v1df_copy['pop_name']]['cell_type'].values, index=v1df_copy.index)
v1df_copy['layer_num'] = v1df_copy['pop_name'].str.extract(r'[ei](\d)').astype(int)

# Identify L1 inhibitory cells
l1_inh_ids = v1df_copy[(v1df_copy['layer_num'] == 1) & ~v1df_copy['cell_type'].str.startswith('Exc')].index
l1_inh_spikes = spike_df[
    (spike_df.index.isin(l1_inh_ids)) & 
    (spike_df['timestamps'] >= evoked_start) & 
    (spike_df['timestamps'] < evoked_end)
]
l1_neurons_count = len(l1_inh_ids)

# Calculate L1 inhibitory power spectrum if there are any L1 inhibitory neurons
l1_power_spectrum = None
if l1_neurons_count > 0 and not l1_inh_spikes.empty:
    l1_freqs, l1_power = calculate_power_spectrum(l1_inh_spikes['timestamps'], l1_neurons_count)
    l1_power_spectrum = (l1_freqs, l1_power, l1_neurons_count)

if only_raster:
    ax_raster = fig.add_subplot(111)
else:
    # Define grid layout with equal height panels for the spectra
    gs = plt.GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1])
    # Top subplot for raster (50% of height)
    ax_raster = fig.add_subplot(gs[0:3, 0:2])

# Plot raster with grouping and legend marker scaling
# Defaults for raster-only mode
if only_raster:
    if legend_markerscale is None:
        legend_markerscale = 8.0
    if args.fontsize == parser.get_default("fontsize"):
        base_fontsize = 16.0
        plt.rcParams.update({
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize,
            "axes.labelsize": base_fontsize,
            "legend.fontsize": base_fontsize,
            "legend.title_fontsize": base_fontsize,
            "xtick.labelsize": base_fontsize,
            "ytick.labelsize": base_fontsize,
        })

auto_markerscale = 4.0 if (legend_markerscale is None and grouping == "four") else legend_markerscale
# Prepare plotting settings, overriding radius for full-network
_plot_settings = dict(pu.settings.get(net, {}))
if use_full_network:
    _plot_settings["radius"] = None
    # Reduce marker area by 1/4 in full-network mode to preserve perceived density
    base_s = _plot_settings.get("s", 1)
    _plot_settings["s"] = base_s * 0.25
    auto_markerscale *= 2.0

raster_plot = pu.plot_raster(
    config_file,
    sortby=sortby,
    infer=True,
    ax=ax_raster,
    grouping=grouping,
    legend_markerscale=auto_markerscale,
    layer_label_fontsize=base_fontsize,
    **_plot_settings,
)
ax_raster.set_xlim([0, 2500])
if not only_raster:
    ax_raster.set_title('Raster Plot')
# Presentation cosmetics for raster
ax_raster.spines["top"].set_visible(False)
ax_raster.spines["right"].set_visible(False)
ax_raster.margins(y=0.01)

# Get the color scheme from the raster plot
legend_handles = raster_plot.get_legend().get_lines()  # Line2D objects for lines
legend_texts = raster_plot.get_legend().get_texts()    # Text objects for labels
raster_colors = {text.get_text(): handle.get_color() for text, handle in zip(legend_texts, legend_handles)}

if not only_raster:
    # Calculate power spectra for each layer and cell type
    power_spectra = group_spikes_by_layer_and_type(spike_df, v1df)

# -----------------------------------------------------------------------------
# Load color scheme from the CSV file so that plots follow project-wide standard
# -----------------------------------------------------------------------------
# The CSV uses the column name "hex" for the color codes and the index contains
# the canonical cell_type names (e.g. "L2/3_Exc", "L4_PV", "L5_IT", etc.).
# We will create a small helper that maps our simplified (layer, cell_type)
# representation used in this script to the canonical key and returns the
# corresponding hex code.
ct_color_table = nu.get_cell_type_table(src="cell_type")[["hex"]]



def get_color(layer: str, cell_type: str) -> str:
    """Return hex color for given (layer, cell_type) combination."""
    # 1) Fast path: if the provided label already matches a row in the table.
    if cell_type in ct_color_table.index:
        return ct_color_table.loc[cell_type, "hex"]

    # 2) Translate shorthand labels (E, PV, SST, VIP, etc.) to canonical keys
    key = None
    if cell_type in ("E", "PV", "SST", "VIP"):
        suffix_map = {"E": "Exc", "PV": "PV", "SST": "SST", "VIP": "VIP"}
        key = f"{layer}_{suffix_map[cell_type]}"
    elif cell_type == "E-IT":
        key = "L5_IT"
    elif cell_type == "E-ET":
        key = "L5_ET"
    elif cell_type == "E-NP":
        key = "L5_NP"
    elif cell_type == "L1-Inh":
        key = "L1_Inh"

    if key is not None and key in ct_color_table.index:
        return ct_color_table.loc[key, "hex"]

    # 3) Fallback: warn and return black
    print(f"[plot_raster] Warning: No colour found for (layer={layer}, cell_type={cell_type}). Using black.")
    return "black"

# -----------------------------------------------------------------------------

# Define colors for different cell types directly using the suffix
# cell_type_colors = {
#     'Exc': 'tab:red',       # Red for excitatory cells
#     'IT': 'tab:red',        # Same red for IT cells
#     'NP': 'tab:orange',     # Orange for NP cells
#     'ET': 'tab:pink',       # Pink for ET cells
#     'PV': 'tab:blue',       # Blue for PV cells
#     'Pvalb': 'tab:blue',    # Same blue for Pvalb cells (alternate naming)
#     'SST': 'tab:olive',     # Olive for SST cells
#     'Sst': 'tab:olive',     # Same olive (alternate naming)
#     'VIP': 'tab:purple',    # Purple for VIP cells
#     'Vip': 'tab:purple',    # Same purple (alternate naming)
#     'L1-Inh': 'darkviolet'  # Dark purple for L1 inhibitory cells
# }

if not only_raster:
    # Create log-log subplots for power spectra (one for each layer)
    layers_to_plot = ['L2/3', 'L4', 'L5', 'L6']

    # Positions for the spectra plots in the bottom half of the figure
    spectra_positions = [
        gs[3, 0],  # L2/3
        gs[3, 1],  # L4
        gs[4, 0],  # L5
        gs[4, 1]   # L6
    ]

    for layer, position in zip(layers_to_plot, spectra_positions):
        ax = fig.add_subplot(position)
        
        # Plot power spectra for each cell type in this layer
        if layer in power_spectra:
            for cell_type, (freqs, power, n_neurons) in power_spectra[layer].items():
                # Use the standardized color from the CSV file
                color = get_color(layer, cell_type)
                
                # Plot with the color
                ax.loglog(freqs, power, label=f'{cell_type} (n={n_neurons})', color=color)
        
        # Add L1 inhibitory cells to the L2/3 plot
        if layer == 'L2/3' and l1_power_spectrum is not None:
            l1_freqs, l1_power, l1_count = l1_power_spectrum
            ax.loglog(l1_freqs, l1_power, label=f'L1-Inh (n={l1_count})', 
                     color=get_color('L1', 'L1-Inh'), linestyle='--')
        
        # Set plot properties
        ax.set_xlim(1, 100)  # Display frequencies from 1 to 100 Hz (log scale)
        ax.set_ylim(1e-10, 1e-5)
        ax.set_title(f'{layer} Power Spectra (Evoked: {evoked_start}-{evoked_end} ms)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (normalized)')
        ax.legend(loc='lower left')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
config_folder = os.path.dirname(config_file)
# Compose informative filename suffixes
scope_suffix = "full" if use_full_network else "core"
group_suffix = grouping
base_name = "raster_by" if only_raster else "raster_and_spectra_by"
outfile = f"{config_folder}/{base_name}_{sortby}_{group_suffix}_{scope_suffix}.png"
plt.savefig(outfile, dpi=300)
