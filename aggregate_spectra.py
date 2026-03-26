import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import argparse
from tqdm import tqdm
import matplotlib.colors as mcolors
import re

# Define a color cycle for simulations, reserving black for experimental
sim_colors = plt.cm.viridis(np.linspace(0, 1, 5))

def extract_layer_from_cell_type(cell_type):
    """Extract cortical layer from cell type string."""
    # Match patterns like L1, L2/3, L4, L5, L6 at the beginning
    match = re.match(r"^(L[1-6](\/[1-6])?)_", cell_type)
    if match:
        return match.group(1)  # Return the layer part (e.g., "L1", "L2/3")

    # Handle potential specific mappings if needed (e.g., if 'e4' means L4)
    # if cell_type == 'e4': return 'L4'

    return None  # Return None if no standard layer format found


def load_spectrum_data(base_dirs, input_types, cell_types_to_plot=None, neuropixels_path=None):
    """
    Load spectral data from multiple networks and input types,
    optionally including experimental NeuroPixels data.

    Parameters:
    - base_dirs: List of base directories (e.g., ["core_nll_0", "core_nll_1", ...])
    - input_types: List of input types (e.g., ["checkpoint", "plain", ...])
    - cell_types_to_plot: List of specific cell types to collect, or None for all
    - neuropixels_path: Path to the combined_spectra.npy file from NeuroPixels analysis

    Returns:
    - DataFrame with the spectral data
    """
    all_data = []

    for base_dir in tqdm(base_dirs, desc="Loading networks"):
        for input_type in tqdm(input_types, desc=f"Processing {base_dir}", leave=False):
            # Load combined spectra file
            combined_file = (
                f"{base_dir}/contrasts_{input_type}/combined_spectra_700to2700.json"
            )
            if not os.path.exists(combined_file):
                print(f"Warning: File {combined_file} not found, skipping")
                continue

            try:
                with open(combined_file, "r") as f:
                    combined_results = json.load(f)

                # Filter cell types if requested
                if cell_types_to_plot:
                    cell_types = [
                        ct
                        for ct in combined_results["cell_types"]
                        if ct in cell_types_to_plot
                    ]
                else:
                    cell_types = combined_results["cell_types"]

                # Process each cell type
                for cell_type in cell_types:
                    # Process each contrast (for contrast 0.8 only)
                    contrast_key = "0.8"
                    if contrast_key not in combined_results["spectra"][cell_type]:
                        continue

                    # Get data for angle 0
                    angle_key = "0"
                    if (
                        angle_key
                        not in combined_results["spectra"][cell_type][contrast_key]
                    ):
                        continue

                    # Extract spectral data
                    spectrum = combined_results["spectra"][cell_type][contrast_key][
                        angle_key
                    ]
                    frequencies = np.array(spectrum["frequencies"])
                    psd = np.array(spectrum["psd"])

                    # Store only data for frequencies up to 100 Hz
                    mask = frequencies <= 100
                    freq = frequencies[mask]
                    power = psd[mask]

                    # Create data entry
                    for f, p in zip(freq, power):
                        all_data.append(
                            {
                                "network": base_dir,
                                "input_type": input_type,
                                "cell_type": cell_type,
                                "contrast": 0.8,
                                "frequency": f,
                                "power": p,
                            }
                        )
            except Exception as e:
                print(f"Error processing {combined_file}: {e}")

    # Load NeuroPixels data if path is provided
    if neuropixels_path and os.path.exists(neuropixels_path):
        print(f"Loading NeuroPixels data from {neuropixels_path}")
        try:
            # Load the dictionary saved via np.save
            neuropixels_results = np.load(neuropixels_path, allow_pickle=True).item()

            # Define mapping from NeuroPixels names to Simulation names
            neuropixels_cell_map = {
                "EXC_L23": "L2/3_Exc",
                "EXC_L4": "L4_Exc",
                "EXC_L5": "L5_IT",  # Assuming L5 EXC maps to L5 IT in simulation
                "EXC_L6": "L6_Exc",
                "PV_L23": "L2/3_PV",
                "PV_L4": "L4_PV",
                "PV_L5": "L5_PV",
                "PV_L6": "L6_PV",
                "SST_L23": "L2/3_SST",
                "SST_L4": "L4_SST",
                "SST_L5": "L5_SST",
                "SST_L6": "L6_SST",
                "VIP_L23": "L2/3_VIP",
                "VIP_L4": "L4_VIP",
                "VIP_L5": "L5_VIP",
                "VIP_L6": "L6_VIP",
                # Add other mappings if necessary based on .npy file contents
            }

            # Determine which cell types to process from the .npy file list
            np_cell_types_in_file = neuropixels_results.get("cell_types", [])
            if cell_types_to_plot:
                # If user specified cell types, filter based on the *mapped* simulation names
                print(f"Filtering NeuroPixels data for specified cell types: {cell_types_to_plot}")
                np_cell_types_to_process = [
                    np_ct
                    for np_ct in np_cell_types_in_file
                    if neuropixels_cell_map.get(np_ct) in cell_types_to_plot
                ]
            else:
                # Otherwise, process all mappable cell types found in the file
                np_cell_types_to_process = [
                    np_ct
                    for np_ct in np_cell_types_in_file
                    if np_ct in neuropixels_cell_map  # Only process if we have a mapping
                ]

            print(f"Processing NeuroPixels cell types: {np_cell_types_to_process}")

            # Extract data for each relevant cell type
            for np_cell_type in np_cell_types_to_process:
                if np_cell_type in neuropixels_results.get("spectra", {}):
                    spectrum_data = neuropixels_results["spectra"][np_cell_type]
                    frequencies = np.array(spectrum_data.get("frequencies", []))

                    # Try getting 'mean_psd', then 'psd', calculating mean if 'psd' is a list
                    psd = np.array(spectrum_data.get("mean_psd", []))
                    psd_key_used = 'mean_psd'
                    if psd.size == 0:
                        psd = np.array(spectrum_data.get("psd", []))
                        psd_key_used = 'psd'
                        # If 'psd' key holds a list/stack of arrays (from sessions), calculate mean
                        if psd.ndim > 1 and psd.shape[0] > 1:  # Check if it's stack of arrays
                            print(f"  Calculating mean PSD from list/stack ({psd.shape}) for NeuroPixels cell type {np_cell_type}")
                            psd = np.mean(psd, axis=0)
                            psd_key_used = 'psd (mean calculated)'
                        elif psd.ndim > 1 and psd.shape[0] == 1:  # If it's saved as [[...]]
                            psd = psd[0]
                            psd_key_used = 'psd (extracted from list/stack)'

                    if frequencies.size == 0 or psd.size == 0 or frequencies.size != psd.size:
                        print(f"Warning: Missing or mismatched frequencies/PSD for NeuroPixels cell type {np_cell_type} (Shape Freq: {frequencies.shape}, Shape PSD: {psd.shape}). Tried keys 'mean_psd', 'psd'. Skipping.")
                        continue

                    # Also load SEM if available
                    sem_psd = np.array(spectrum_data.get("sem_psd", []))
                    power_sem = np.zeros_like(psd)  # Default to zero SEM if not found or mismatch
                    if sem_psd.size == frequencies.size:  # Check if SEM data exists and matches freq size
                        power_sem = sem_psd
                        print(f"    Loaded SEM for NeuroPixels {np_cell_type}")
                    elif sem_psd.size > 0:
                        print(f"    Warning: SEM size mismatch for NeuroPixels {np_cell_type} (SEM: {sem_psd.shape}, Freq: {frequencies.shape}). SEM not loaded.")

                    # Map the cell type name to simulation convention
                    sim_cell_type = neuropixels_cell_map.get(np_cell_type)
                    # We already filtered for mappable types, but double-check
                    if sim_cell_type is None:
                        print(f"Internal Warning: No mapping found for NeuroPixels cell type {np_cell_type} during data appending. Skipping.")
                        continue

                    # Filter frequencies <= 100 Hz
                    mask = frequencies <= 100
                    freq = frequencies[mask]
                    power = psd[mask]
                    power_sem = power_sem[mask]

                    # Append data using the mapped simulation cell type name
                    print(f"  Loaded NeuroPixels data for {np_cell_type} (mapped to {sim_cell_type}), using PSD key: {psd_key_used}")
                    for f, p, s in zip(freq, power, power_sem):
                        all_data.append(
                            {
                                "network": "NeuroPixels",
                                "input_type": "Experimental",
                                "cell_type": sim_cell_type,  # Use mapped simulation name
                                "contrast": 0.8,
                                "frequency": f,
                                "power": p,
                                "power_sem": s,  # Add SEM here
                            }
                        )
                else:
                    print(f"Warning: NeuroPixels cell type {np_cell_type} not found in loaded NeuroPixels spectra dictionary.")

        except Exception as e:
            print(f"Error processing NeuroPixels data {neuropixels_path}: {e}")
    elif neuropixels_path:
        print(f"Warning: NeuroPixels data file not found at {neuropixels_path}")

    # Convert to DataFrame
    if not all_data:
        print("Warning: No data loaded.")
        return pd.DataFrame()  # Return empty DF if no data

    return pd.DataFrame(all_data)


def plot_aggregate_spectra(df, output_dir=".", normalized=False):
    """
    Generate aggregate plots of power spectra across networks

    Parameters:
    - df: DataFrame with spectral data
    - output_dir: Directory to save plots
    - normalized: Whether to show normalized power
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get unique input types and cell types
    input_types = sorted(df["input_type"].unique())
    cell_types = sorted(df["cell_type"].unique())

    # Assign colors, reserving black for Experimental
    sim_input_types = [it for it in input_types if it != "Experimental"]
    # color_map = {it: sim_colors[i % len(sim_colors)] for i, it in enumerate(sim_input_types)}
    # Use a standard color cycle for simulation types
    prop_cycle = plt.rcParams['axes.prop_cycle']
    sim_colors_cycle = prop_cycle.by_key()['color']
    color_map = {it: sim_colors_cycle[i % len(sim_colors_cycle)] for i, it in enumerate(sim_input_types)}

    if "Experimental" in input_types:
        color_map["Experimental"] = "black"

    # Generate plots comparing input types for each cell type
    for cell_type in tqdm(cell_types, desc="Plotting cell types"):
        plt.figure(figsize=(12, 8))
        has_experimental = False

        # Separate loop for simulation to handle colors correctly
        for i, input_type in enumerate(sim_input_types):
            # Get data for this input type and cell type
            subset = df[
                (df["cell_type"] == cell_type) & (df["input_type"] == input_type)
            ]

            if subset.empty:
                continue

            # Group by frequency and calculate mean and SEM for simulations
            grouped = subset.groupby("frequency")["power"]
            mean_power = grouped.mean()
            sem_power = grouped.sem()
            # Ensure frequencies are sorted
            frequencies = np.array(sorted(subset["frequency"].unique()))
            # Reindex mean and sem to match sorted frequencies, filling missing with NaN
            mean_power = mean_power.reindex(frequencies)
            sem_power = sem_power.reindex(frequencies)

            # Plot simulation data with error bands
            line_color = color_map[input_type]
            plt.plot(frequencies, mean_power, label=input_type, linewidth=2, color=line_color)
            plt.fill_between(
                frequencies,
                mean_power - sem_power,
                mean_power + sem_power,
                alpha=0.2,
                color=line_color
            )

        # Plot Experimental data if present (separately, without SEM band)
        if "Experimental" in input_types:
            subset_exp = df[
                (df["cell_type"] == cell_type) & (df["input_type"] == "Experimental")
            ]
            if not subset_exp.empty:
                # Experimental data should already be averaged, no need to group further
                # Sort by frequency just in case
                subset_exp = subset_exp.sort_values("frequency")
                frequencies_exp = subset_exp["frequency"].values
                mean_power_exp = subset_exp["power"].values
                sem_power_exp = subset_exp["power_sem"].values  # Get SEM values

                plt.plot(
                    frequencies_exp,
                    mean_power_exp,
                    label="Experimental",
                    linewidth=2,
                    color=color_map["Experimental"],
                    linestyle="--",  # Dashed line for experimental
                )
                # Add gray shading for experimental error band if SEM is available
                if np.any(sem_power_exp > 0):  # Check if there are non-zero SEM values
                    plt.fill_between(
                        frequencies_exp,
                        mean_power_exp - sem_power_exp,
                        mean_power_exp + sem_power_exp,
                        alpha=0.2,
                        color="gray"  # Use gray for experimental SEM
                    )

                has_experimental = True

        # Format plot
        title_norm = "Normalized " if normalized else ""
        plt.title(
            f"{title_norm}Power Spectra for {cell_type} (High Contrast)\\nAveraged across networks{' and Experimental Data' if has_experimental else ''}",
            fontsize=14,
        )
        plt.xlabel("Frequency (Hz)", fontsize=12)
        plt.ylabel("Power Spectral Density", fontsize=12)
        plt.legend(title="Input Type")
        plt.grid(True, which="both", linestyle="--", alpha=0.7)

        # Scale
        plt.xscale("log")
        plt.yscale("log")

        # Highlight frequency bands of interest
        plt.axvspan(4, 8, alpha=0.1, color="blue", label="Theta (4-8 Hz)")
        plt.axvspan(8, 12, alpha=0.1, color="green", label="Alpha (8-12 Hz)")
        plt.axvspan(30, 80, alpha=0.1, color="red", label="Gamma (30-80 Hz)")

        plt.tight_layout()

        # Save
        norm_suffix = "_normalized" if normalized else ""
        safe_cell_type = cell_type.replace("/", "_").replace("\\", "_")
        output_file = os.path.join(
            output_dir, f"aggregate_spectra_{safe_cell_type}{norm_suffix}.png"
        )
        plt.savefig(output_file)
        plt.close()

    # Create comparison plot for a few key cell types (ONLY FOR SIMULATION INPUT TYPES)
    key_cell_types = ["L4 Exc", "L2/3 Exc", "L5 Exc", "L6 Exc", "VIP", "SST", "PV"]  # Added inhibitory types
    key_cell_types = [ct for ct in key_cell_types if ct in cell_types]

    # Only plot comparisons if there are simulation input types
    if sim_input_types:
        for input_type in tqdm(sim_input_types, desc="Plotting key cell comparisons"):
            plt.figure(figsize=(12, 8))
            # Use a distinct color cycle for cell types within this plot
            cell_color_cycle = plt.cm.tab10(np.linspace(0, 1, len(key_cell_types)))

            for i, cell_type in enumerate(key_cell_types):
                # Get data for this simulation input type and cell type
                subset = df[
                    (df["cell_type"] == cell_type) & (df["input_type"] == input_type)
                ]

                if subset.empty:
                    continue

                # Group by frequency and calculate mean power for simulations
                # SEM for simulation is calculated from variation across networks
                grouped = subset.groupby("frequency")["power"]
                mean_power = grouped.mean()
                # Ensure frequencies are sorted
                frequencies = np.array(sorted(subset["frequency"].unique()))
                # Reindex mean to match sorted frequencies
                mean_power = mean_power.reindex(frequencies)

                # Plot
                plt.plot(frequencies, mean_power, label=cell_type, linewidth=2, color=cell_color_cycle[i])

            # Format plot
            title_norm = "Normalized " if normalized else ""
            plt.title(
                f"{title_norm}Power Spectra for {input_type} input (High Contrast)\\nAveraged across networks",
                fontsize=14,
            )
            plt.xlabel("Frequency (Hz)", fontsize=12)
            plt.ylabel("Power Spectral Density", fontsize=12)
            plt.legend(title="Cell Type")
            plt.grid(True, which="both", linestyle="--", alpha=0.7)

            # Scale
            plt.xscale("log")
            plt.yscale("log")

            # Highlight frequency bands of interest
            plt.axvspan(4, 8, alpha=0.1, color="blue")
            plt.axvspan(8, 12, alpha=0.1, color="green")
            plt.axvspan(30, 80, alpha=0.1, color="red")

            plt.tight_layout()

            # Save
            norm_suffix = "_normalized" if normalized else ""
            output_file = os.path.join(
                output_dir, f"aggregate_spectra_key_cells_{input_type}{norm_suffix}.png"
            )
            plt.savefig(output_file)
            plt.close()
    else:
        print("Skipping key cell type comparison plots as no simulation input types were found.")


def plot_layer_aggregate_spectra(df, output_dir=".", normalized=False):
    """
    Generate aggregate plots of power spectra grouped by layer for a single input type.

    Parameters:
    - df: DataFrame with spectral data (should contain only one input_type).
    - output_dir: Directory to save plots.
    - normalized: Whether to show normalized power (currently not implemented for layers).
    """
    os.makedirs(output_dir, exist_ok=True)

    input_type = df["input_type"].unique()
    if len(input_type) != 1:
        print("Error: plot_layer_aggregate_spectra called with multiple input types.")
        return
    input_type = input_type[0]

    # Add layer information
    df["layer"] = df["cell_type"].apply(extract_layer_from_cell_type)
    df_filtered = df.dropna(subset=["layer"])

    if df_filtered.empty:
        print("No data with recognizable layer information found. Skipping layer aggregation plot.")
        return

    layers = sorted(df_filtered["layer"].unique(), key=lambda x: int(re.search(r'\d+', x).group()))  # Sort layers numerically

    plt.figure(figsize=(12, 8))
    layer_color_cycle = plt.cm.viridis(np.linspace(0, 1, len(layers)))

    is_experimental = (input_type == "Experimental")

    for i, layer in enumerate(layers):
        subset = df_filtered[df_filtered["layer"] == layer]

        # Group by frequency and aggregate power across cell types in the layer
        grouped = subset.groupby("frequency")
        mean_power = grouped["power"].mean()  # Simple mean across cell types
        frequencies = mean_power.index.values

        # Aggregate SEM if it's experimental data
        mean_sem = None
        if is_experimental and 'power_sem' in subset.columns:
            # Average SEM - note this is an approximation
            mean_sem = grouped["power_sem"].mean()
            mean_sem = mean_sem.reindex(frequencies).fillna(0)  # Align with frequencies

        # Plot layer aggregate
        plt.plot(frequencies, mean_power, label=layer, linewidth=2, color=layer_color_cycle[i])

        # Plot SEM band if available (primarily for experimental)
        if mean_sem is not None and np.any(mean_sem > 0):
            plt.fill_between(
                frequencies,
                mean_power - mean_sem,
                mean_power + mean_sem,
                alpha=0.2,
                color=layer_color_cycle[i]
            )

    # Format plot
    title_norm = "Normalized " if normalized else ""
    plt.title(
        f"{title_norm}Layer-Aggregated Power Spectra for {input_type} Input (High Contrast)",
        fontsize=14,
    )
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Average Power Spectral Density (across cell types)", fontsize=12)
    plt.legend(title="Layer")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)

    # Scale
    plt.xscale("log")
    plt.yscale("log")

    # Highlight frequency bands
    plt.axvspan(4, 8, alpha=0.1, color="blue", label="Theta (4-8 Hz)")
    plt.axvspan(8, 12, alpha=0.1, color="green", label="Alpha (8-12 Hz)")
    plt.axvspan(30, 80, alpha=0.1, color="red", label="Gamma (30-80 Hz)")

    plt.tight_layout()

    # Save
    norm_suffix = "_normalized" if normalized else ""
    safe_input_type = input_type.replace("/", "_").replace("\\", "_")
    output_file = os.path.join(
        output_dir, f"layer_aggregate_spectra_{safe_input_type}{norm_suffix}.png"
    )
    plt.savefig(output_file)
    print(f"Saved layer aggregate plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate spectral analysis across multiple networks"
    )
    parser.add_argument(
        "--base_dirs",
        type=str,
        nargs="+",
        default=["core_nll_" + str(i) for i in range(10)],
        help="Base directories to process",
    )
    parser.add_argument(
        "--input_types",
        type=str,
        nargs="+",
        default=["checkpoint", "noweightloss", "adjusted", "plain"],
        help="Input types to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="aggregate_spectra",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--cell_types",
        type=str,
        nargs="+",
        default=None,
        help="Cell types to include (default: all found)",
    )
    parser.add_argument(
        "--neuropixels-data",
        type=str,
        default=None,
        help="Path to the combined_spectra.npy file from NeuroPixels analysis.",
    )
    parser.add_argument(
        "--aggregate-layers",
        action="store_true",
        help="Aggregate spectra by layer (only applies if a single input_type is processed)."
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Plot normalized power spectra"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(
        f"Loading data from {len(args.base_dirs)} networks and {len(args.input_types)} input types"
    )
    df = load_spectrum_data(
        args.base_dirs,
        args.input_types,
        args.cell_types,
        neuropixels_path=args.neuropixels_data,  # Pass the path here
    )

    # Check if DataFrame is empty
    if df.empty:
        print("Resulting DataFrame is empty. Exiting.")
    else:
        print("\nDataFrame Head:")
        print(df.head())
        print(f"\nLoaded data for {len(df['network'].unique())} networks/sources.")
        print(f"Input types: {df['input_type'].unique()}")
        print(f"Cell types: {df['cell_type'].unique()}")

        # Decide which plotting function to use
        input_types_found = df['input_type'].unique()
        if args.aggregate_layers and len(input_types_found) == 1:
            print(f"\nAggregating by layer for single input type: {input_types_found[0]}")
            plot_layer_aggregate_spectra(df.copy(), args.output_dir, args.normalize)
        else:
            if args.aggregate_layers and len(input_types_found) > 1:
                print("\nWarning: --aggregate-layers flag ignored because multiple input types were found. Generating standard plots.")
            # Generate standard plots (comparing input types per cell type, etc.)
            plot_aggregate_spectra(df, args.output_dir, args.normalize)

        print(f"\nPlots saved to {args.output_dir}")
