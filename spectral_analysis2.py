# %% doing spectral analysis for the contrast stimuli

import numpy as np
import pandas as pd
import stimulus_trials as st
from scipy import signal
import matplotlib.pyplot as plt
import h5py
import os
import network_utils as nu
import time
from collections import defaultdict
import multiprocessing as mp
import argparse
from functools import partial
import json
from tqdm import tqdm


# %% let's do for one first.

# load the data.
basedir = "core_nll_1/"
subdir = "contrasts_checkpoint"
trial = "angle0_contrast0.0_trial0"
spike_file_name = f"{basedir}/{subdir}/{trial}/spikes.h5"


# Load the spike data - optimized version
def load_spike_data(spike_file_name, t_start=700, t_end=2700, verbose=False):
    """
    Load spike data and filter by time window (700ms to 2700ms)
    Note: SONATA format uses time in milliseconds
    """
    start_time = time.time()
    with h5py.File(spike_file_name, "r") as f:
        # Load only the data we need
        timestamps = np.array(f["spikes"]["v1"]["timestamps"][:])
        node_ids = np.array(f["spikes"]["v1"]["node_ids"][:])

        # Filter by time window (700ms to 2700ms)
        time_mask = (timestamps >= t_start) & (timestamps <= t_end)
        timestamps = timestamps[time_mask]
        node_ids = node_ids[time_mask]

    if verbose:
        print(f"Loaded {len(timestamps)} spikes in time window {t_start}-{t_end}ms")
        print(f"Time to load spikes: {time.time() - start_time:.2f}s")

    return timestamps, node_ids


# Calculate power spectrum for a spike train
def calculate_power_spectrum(
    spike_times,
    fs=1.0,
    nperseg=1024,
    noverlap=512,
    t_start=700,
    t_end=2700,
    normalize=True,
):
    """
    Calculate power spectrum from spike times

    Parameters:
    - spike_times: array of spike times in milliseconds
    - fs: sampling frequency in kHz (1.0 means 1 sample per ms)
    - t_start, t_end: time window in milliseconds
    - normalize: if True, apply 1/f normalization
    """
    # Duration of the analysis window in milliseconds
    duration = t_end - t_start

    # Create binary spike train directly in the right window
    bins = np.linspace(t_start, t_end, int(duration * fs) + 1)
    spike_train, _ = np.histogram(spike_times, bins=bins)

    # Improved spectral analysis
    # 1. Apply window function to reduce spectral leakage
    # 2. Use larger nperseg for better frequency resolution
    # 3. Normalize the PSD by the sampling rate to get proper units
    frequencies, psd = signal.welch(
        spike_train,
        fs=fs,
        nperseg=min(nperseg, len(spike_train) // 2),  # Better frequency resolution
        noverlap=min(noverlap, len(spike_train) // 4),  # 50% overlap
        window="hann",  # Reduce spectral leakage
        detrend="constant",  # Remove DC component
        scaling="density",  # Use density scaling
    )

    # Convert frequencies from kHz to Hz
    frequencies = frequencies * 1000

    # Correct PSD scaling only if normalization is requested
    if normalize:
        # This normalization helps visualize oscillatory peaks
        psd = psd * frequencies  # Apply 1/f normalization

    return frequencies, psd


# Calculate spectra by cell type - optimized version
def calculate_spectra_by_cell_type(
    timestamps,
    node_ids,
    v1_nodes,
    ctdf,
    fs=1.0,
    t_start=700,
    t_end=2700,
    normalize=True,
    verbose=False,
    population_average=True,
    normalize_by_neurons_squared=True,
):
    """
    Calculate spectra by cell type

    Parameters:
    - timestamps: spike timestamps
    - node_ids: neuron IDs for each spike
    - v1_nodes: dataframe of V1 neurons
    - ctdf: cell type dataframe
    - fs: sampling frequency in kHz
    - t_start, t_end: time window in ms
    - normalize: whether to apply 1/f normalization
    - verbose: print detailed messages
    - population_average: if True, calculate spectra from average spike train
    - normalize_by_neurons_squared: if True, normalize PSD by the square of the number of neurons
    """
    # Filter for core neurons only
    start_time = time.time()
    v1_nodes = v1_nodes[v1_nodes["core"]]

    # Create a mapping from node_id to cell_type
    v1_nodes["cell_type"] = ctdf.loc[v1_nodes["pop_name"]]["cell_type"].values
    node_to_cell_type = dict(zip(v1_nodes["node_id"], v1_nodes["cell_type"]))

    # Group spikes by neuron ID - much faster than pandas groupby
    neuron_spikes = defaultdict(list)
    for t, nid in zip(timestamps, node_ids):
        neuron_spikes[nid].append(t)

    # Group neurons by cell type
    cell_type_neurons = defaultdict(list)
    for nid, spike_times in neuron_spikes.items():
        if nid in node_to_cell_type:  # Only process neurons with known cell type
            cell_type = node_to_cell_type[nid]
            cell_type_neurons[cell_type].append((nid, spike_times))

    if verbose:
        print(f"Grouped {len(neuron_spikes)} neurons by cell type")
        print(f"Found {len(cell_type_neurons)} cell types")

    # Calculate spectra for each cell type
    spectra_by_cell_type = {}
    duration = t_end - t_start

    for cell_type, neurons in cell_type_neurons.items():
        if verbose:
            print(f"Processing {len(neurons)} {cell_type} neurons...")

        if population_average:
            # Calculate average spike train for the population
            # Create a histogram of all spikes from this cell type
            all_spikes = []
            for _, spike_times in neurons:
                all_spikes.extend(spike_times)

            # Skip if no spikes
            if not all_spikes:
                continue

            # Calculate average spikes per neuron
            avg_spikes_per_neuron = len(all_spikes) / len(neurons)

            if verbose:
                print(
                    f"  {cell_type}: {len(all_spikes)} spikes, {avg_spikes_per_neuron:.1f} spikes/neuron"
                )

            # Calculate spectrum from the population spike train
            frequencies, psd = calculate_power_spectrum(
                all_spikes,
                fs=fs,
                t_start=t_start,
                t_end=t_end,
                normalize=normalize,
            )

            # Normalize by the square of the number of neurons if requested
            if normalize_by_neurons_squared:
                psd = psd / (len(neurons) ** 2)
                if verbose:
                    print(f"  Normalized PSD by {len(neurons)}² neurons")

            spectra_by_cell_type[cell_type] = {
                "frequencies": frequencies,
                "psd": psd,
                "num_neurons": len(neurons),
                "avg_spikes_per_neuron": avg_spikes_per_neuron,
            }
        else:
            # Original method: calculate spectrum per neuron, then average
            neuron_spectra = []
            total_spikes = 0

            for _, spike_times in neurons:
                if len(spike_times) > 0:  # Only calculate if there are spikes
                    total_spikes += len(spike_times)
                    frequencies, psd = calculate_power_spectrum(
                        spike_times,
                        fs=fs,
                        t_start=t_start,
                        t_end=t_end,
                        normalize=normalize,
                    )
                    neuron_spectra.append(psd)

            if neuron_spectra:
                # Average across neurons of the same type
                avg_spectrum = np.mean(neuron_spectra, axis=0)
                avg_spikes_per_neuron = total_spikes / len(neurons)

                # Normalize by the square of the number of neurons if requested
                if normalize_by_neurons_squared:
                    avg_spectrum = avg_spectrum / (len(neurons) ** 2)
                    if verbose:
                        print(f"  Normalized PSD by {len(neurons)}² neurons")

                if verbose:
                    print(
                        f"  {cell_type}: {total_spikes} spikes, {avg_spikes_per_neuron:.1f} spikes/neuron"
                    )

                spectra_by_cell_type[cell_type] = {
                    "frequencies": frequencies,
                    "psd": avg_spectrum,
                    "num_neurons": len(neurons),
                    "avg_spikes_per_neuron": avg_spikes_per_neuron,
                }

    if verbose:
        print(f"Time to calculate spectra: {time.time() - start_time:.2f}s")
    return spectra_by_cell_type


# Plot the spectra
def plot_spectra_by_cell_type(spectra_by_cell_type, t_start=700, t_end=2700):
    plt.figure(figsize=(10, 6))

    for cell_type, spectrum in spectra_by_cell_type.items():
        # Limit frequencies to 100 Hz
        freq_mask = spectrum["frequencies"] <= 100
        plt.semilogy(
            spectrum["frequencies"][freq_mask],
            spectrum["psd"][freq_mask],
            label=f"Cell type {cell_type}",
        )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(f"Power Spectra by Cell Type ({t_start/1000:.1f}s-{t_end/1000:.1f}s)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Process a single stimulus trial
def process_trial(
    angle,
    contrast,
    trial_num,
    basedir,
    subdir,
    t_start=700,
    t_end=2700,
    fs=1.0,
    normalize=True,
    verbose=False,
    population_average=True,
    normalize_by_neurons_squared=True,
):
    """
    Process a single stimulus trial

    Parameters:
    - angle: stimulus angle
    - contrast: stimulus contrast
    - trial_num: trial number
    - basedir: base directory
    - subdir: subdirectory for results
    - t_start, t_end: time window in ms
    - fs: sampling frequency in kHz
    - normalize: if True, apply 1/f normalization
    - normalize_by_neurons_squared: if True, normalize PSD by the square of the number of neurons

    Returns:
    - Dictionary with trial info and spectra results
    """
    trial_start = time.time()

    # Create trial ID
    trial_id = f"angle{int(angle)}_contrast{contrast}_trial{trial_num}"
    if verbose:
        print(f"Processing {trial_id}")

    # Define spike file path
    spike_file_name = f"{basedir}/{subdir}/{trial_id}/spikes.h5"

    # Skip if file doesn't exist
    if not os.path.exists(spike_file_name):
        if verbose:
            print(f"Skipping {trial_id}: File not found")
        return None

    try:
        # Load spike data
        timestamps, node_ids = load_spike_data(spike_file_name, t_start, t_end, verbose)

        # Skip if no spikes found
        if len(timestamps) == 0:
            if verbose:
                print(f"Skipping {trial_id}: No spikes found")
            return None

        # Determine core radius based on basedir
        core_radius = nu.infer_core_radius(basedir)

        # Load nodes and cell type information using network_utils
        v1_nodes = (
            nu.load_nodes_pl(basedir, core_radius=core_radius).collect().to_pandas()
        )
        v1_nodes = v1_nodes[v1_nodes["core"]]  # Filter for core neurons only
        ctdf = nu.get_cell_type_table()

        # Calculate spectra by cell type
        spectra_by_cell_type = calculate_spectra_by_cell_type(
            timestamps,
            node_ids,
            v1_nodes,
            ctdf,
            fs=fs,
            t_start=t_start,
            t_end=t_end,
            normalize=normalize,
            verbose=verbose,
            population_average=population_average,
            normalize_by_neurons_squared=normalize_by_neurons_squared,
        )

        # Create output directory if it doesn't exist
        output_dir = f"{basedir}/{subdir}/{trial_id}"
        os.makedirs(output_dir, exist_ok=True)

        # Save the results
        output_file = f"{output_dir}/spectra_by_cell_type_{t_start}to{t_end}.npy"
        np.save(output_file, spectra_by_cell_type)

        if verbose:
            print(f"Completed {trial_id} in {time.time() - trial_start:.2f}s")

        # Return result metadata
        return {
            "angle": angle,
            "contrast": contrast,
            "trial": trial_num,
            "output_file": output_file,
            "cell_types": list(spectra_by_cell_type.keys()),
            "duration": time.time() - trial_start,
        }

    except Exception as e:
        if verbose:
            print(f"Error processing {trial_id}: {e}")
        return None


# Combine all spectra results for analysis and visualization
def combine_spectra_results(
    results, basedir, subdir, t_start=700, t_end=2700, verbose=False
):
    """
    Combine all spectra results from multiple trials

    Parameters:
    - results: list of result metadata from process_trial
    - basedir: base directory
    - subdir: subdirectory for results
    - t_start, t_end: time window used for analysis

    Returns:
    - Dictionary with combined spectra results
    """
    # Filter out None results
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("No valid results found")
        return None

    # Get unique contrasts and angles
    contrasts = sorted(set(r["contrast"] for r in valid_results))
    angles = sorted(set(r["angle"] for r in valid_results))

    # Get all cell types
    all_cell_types = set()
    for r in valid_results:
        all_cell_types.update(r["cell_types"])
    all_cell_types = sorted(all_cell_types)

    # Create combined results structure
    combined = {
        "cell_types": all_cell_types,
        "contrasts": contrasts,
        "angles": angles,
        "t_start": t_start,
        "t_end": t_end,
        "spectra": {cell_type: {} for cell_type in all_cell_types},
    }

    # Load and combine spectra for each cell type, contrast, and angle
    if verbose:
        print(f"Combining results for {len(all_cell_types)} cell types...")
    for cell_type in all_cell_types:
        # Initialize spectra dict for this cell type
        combined["spectra"][cell_type] = {}

        for contrast in contrasts:
            # Use string representation of contrast as key
            contrast_key = str(contrast)
            combined["spectra"][cell_type][contrast_key] = {}

            for angle in angles:
                # Use string representation of angle as key
                angle_key = str(int(angle))

                # Find all trials for this contrast/angle combination
                matching_results = [
                    r
                    for r in valid_results
                    if r["contrast"] == contrast and r["angle"] == angle
                ]

                trial_spectra = []
                frequencies = None

                # Load spectra from each trial
                for r in matching_results:
                    try:
                        spectra_data = np.load(
                            r["output_file"], allow_pickle=True
                        ).item()
                        if cell_type in spectra_data:
                            if frequencies is None:
                                frequencies = spectra_data[cell_type]["frequencies"]
                            trial_spectra.append(spectra_data[cell_type]["psd"])
                    except Exception as e:
                        print(f"Error loading {r['output_file']}: {e}")

                # Average across trials if we have data
                if trial_spectra:
                    avg_spectrum = np.mean(trial_spectra, axis=0)
                    combined["spectra"][cell_type][contrast_key][angle_key] = {
                        "frequencies": frequencies,
                        "psd": avg_spectrum,
                        "n_trials": len(trial_spectra),
                    }

    # Save combined results
    combined_file = f"{basedir}/{subdir}/combined_spectra_{t_start}to{t_end}.json"

    # Convert numpy arrays to lists for JSON serialization
    for cell_type in all_cell_types:
        for contrast_key in combined["spectra"][cell_type]:
            for angle_key in combined["spectra"][cell_type][contrast_key]:
                data = combined["spectra"][cell_type][contrast_key][angle_key]
                if "frequencies" in data:
                    data["frequencies"] = data["frequencies"].tolist()
                if "psd" in data:
                    data["psd"] = data["psd"].tolist()

    with open(combined_file, "w") as f:
        json.dump(combined, f)

    return combined


# Plot spectra for different contrasts for a given cell type
def plot_contrast_spectra(
    combined_results,
    cell_type,
    angle=0,
    max_freq=100,
    output_dir=".",
    normalize=True,
    normalize_by_neurons_squared=True,
    verbose=False,
):
    """Plot spectra for different contrasts for a given cell type"""
    # Create a unique filename based on plot parameters - this allows different visualization options
    # without overwriting existing plots
    plot_variant = os.environ.get("PLOT_VARIANT", "")
    norm_suffix = "" if normalize else "_unnormalized"
    neuron_norm_suffix = "" if normalize_by_neurons_squared else "_noN2norm"

    # Set up plot
    plt.figure(figsize=(12, 8))

    contrasts = combined_results["contrasts"]
    angle_key = str(int(angle))

    # Create a color map for different contrasts
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(contrasts)))

    has_data = False
    for idx, contrast in enumerate(contrasts):
        contrast_key = str(contrast)

        if (
            contrast_key in combined_results["spectra"][cell_type]
            and angle_key in combined_results["spectra"][cell_type][contrast_key]
        ):
            has_data = True
            spectrum = combined_results["spectra"][cell_type][contrast_key][angle_key]
            frequencies = np.array(spectrum["frequencies"])
            psd = np.array(spectrum["psd"])

            # Limit to max_freq
            mask = frequencies <= max_freq

            # Plot using different linestyles for clarity
            plt.loglog(
                frequencies[mask],
                psd[mask],
                label=f"Contrast {contrast}",
                color=colors[idx],
                linewidth=2,
            )

            # Add a smoothed version to help identify peaks
            if contrast in [
                0.0,
                0.4,
                0.8,
            ]:  # Only for selected contrasts to avoid clutter
                from scipy.signal import savgol_filter

                smoothed = savgol_filter(
                    psd[mask], 21, 3
                )  # Window size 21, polynomial order 3
                plt.loglog(
                    frequencies[mask],
                    smoothed,
                    color=colors[idx],
                    linestyle=":",
                    alpha=0.7,
                )

    if has_data:
        # Highlight frequency bands of interest
        plt.axvspan(4, 8, alpha=0.1, color="blue", label="Theta (4-8 Hz)")
        plt.axvspan(8, 12, alpha=0.1, color="green", label="Alpha (8-12 Hz)")
        plt.axvspan(30, 80, alpha=0.1, color="red", label="Gamma (30-80 Hz)")

        # Better axis formatting
        plt.xlabel("Frequency (Hz)", fontsize=12)

        # Update ylabel based on normalization
        if normalize_by_neurons_squared:
            y_label = "Power Spectral Density (normalized by N²)"
        else:
            y_label = (
                "Normalized Power Spectral Density"
                if normalize
                else "Power Spectral Density (1/f trend visible)"
            )
        plt.ylabel(y_label, fontsize=12)

        # Update title based on normalization
        title_norm = "Normalized " if normalize else "Unnormalized "
        pop_avg_text = (
            "Population Average " if "num_neurons" in spectrum else "Neuron Average "
        )
        neuron_norm_text = "N² normalized " if normalize_by_neurons_squared else ""
        plt.title(
            f"{title_norm}{neuron_norm_text}{pop_avg_text}Power Spectra for {cell_type} at {angle}° "
            f"({combined_results['t_start']/1000:.1f}s-{combined_results['t_end']/1000:.1f}s)",
            fontsize=14,
        )

        # Add text annotation with neuron count and average spike count if available
        if "num_neurons" in spectrum:
            num_neurons = spectrum["num_neurons"]
            annotation_text = f"Neurons: {num_neurons}"
            if "avg_spikes_per_neuron" in spectrum:
                avg_spikes = spectrum["avg_spikes_per_neuron"]
                annotation_text += f", Avg: {avg_spikes:.1f} spikes/neuron"
            plt.annotate(
                annotation_text,
                xy=(0.02, 0.95),
                xycoords="axes fraction",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

        # Better legend placement
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, which="both", ls="--", alpha=0.5)

        # Add second y-axis with linear scale for better peak visualization
        ax2 = plt.gca().twinx()
        for idx, contrast in enumerate(contrasts):
            contrast_key = str(contrast)
            if (
                contrast_key in combined_results["spectra"][cell_type]
                and angle_key in combined_results["spectra"][cell_type][contrast_key]
            ):
                spectrum = combined_results["spectra"][cell_type][contrast_key][
                    angle_key
                ]
                frequencies = np.array(spectrum["frequencies"])
                psd = np.array(spectrum["psd"])
                mask = frequencies <= max_freq
                if contrast == max(contrasts):  # Only show for highest contrast
                    ax2.plot(frequencies[mask], psd[mask], color="gray", alpha=0.3)
        ax2.set_ylabel("Linear PSD (for peak detection)", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        plt.tight_layout()

        # Save with unique name including normalization info
        safe_cell_type = cell_type.replace("/", "_").replace("\\", "_")
        plot_filename = f"spectra_celltype{safe_cell_type}_angle{angle}{norm_suffix}{neuron_norm_suffix}"
        if plot_variant:
            plot_filename += f"_{plot_variant}"
        plot_filename += ".png"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, plot_filename)

        try:
            plt.savefig(output_file)
            if verbose:
                print(f"Saved figure to {output_file}")
        except Exception as e:
            print(f"Error saving figure: {e}")
        finally:
            plt.close()
    else:
        plt.close()
        if verbose:
            print(f"No data available for cell type {cell_type} at angle {angle}")


# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Calculate spectra for contrast stimuli"
    )
    parser.add_argument(
        "--basedir", type=str, default="core_nll_1/", help="Base directory"
    )
    parser.add_argument(
        "--subdir", type=str, default="contrasts_checkpoint", help="Subdirectory"
    )
    parser.add_argument("--t_start", type=float, default=700, help="Start time (ms)")
    parser.add_argument("--t_end", type=float, default=2700, help="End time (ms)")
    parser.add_argument(
        "--n_processes", type=int, default=8, help="Number of parallel processes"
    )
    parser.add_argument(
        "--max_freq", type=float, default=100, help="Maximum frequency to plot (Hz)"
    )
    parser.add_argument(
        "--plot_only", action="store_true", help="Only plot existing results"
    )
    # Add output directory argument
    parser.add_argument(
        "--output_dir", type=str, default="figures", help="Directory to save figures"
    )
    # Add arguments for different plot options
    parser.add_argument(
        "--plot_variant",
        type=str,
        default="",
        help="Variant name for plots (to avoid overwriting)",
    )
    parser.add_argument(
        "--show_bands",
        action="store_true",
        help="Show frequency bands (theta, gamma, etc)",
    )
    parser.add_argument("--smooth", action="store_true", help="Show smoothed spectra")
    parser.add_argument(
        "--angles", type=int, nargs="+", default=[0], help="Angles to plot (default: 0)"
    )
    # Add normalization argument - now defaults to unnormalized (switched default)
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Plot normalized spectra to see oscillation peaks (default is unnormalized for 1/f trend)",
    )
    # Add neuron count normalization argument - changed to default True with --no-normalize-by-neurons-squared to disable
    parser.add_argument(
        "--no-normalize-by-neurons-squared",
        dest="normalize_by_neurons_squared",
        action="store_false",
        help="Disable normalization of PSD by the square of the number of neurons (normalization is on by default)",
    )
    parser.set_defaults(normalize_by_neurons_squared=True)
    # Add verbose flag
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress messages"
    )
    # Add population average flag
    parser.add_argument(
        "--neuron_average",
        action="store_true",
        help="Calculate spectra per neuron then average (default is to average spike trains first)",
    )

    # Add test mode flag
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode (process fewer trials)"
    )
    parser.add_argument(
        "--test_trials",
        type=int,
        default=3,
        help="Number of trials to process in test mode (default: 3)",
    )

    args = parser.parse_args()

    # Start timing
    total_start_time = time.time()

    # Define parameters
    basedir = args.basedir
    subdir = args.subdir
    t_start = args.t_start
    t_end = args.t_end
    fs = 1.0  # 1 sample per ms

    # Get normalization setting from args - changed from previous version
    normalize = args.normalized
    normalize_by_neurons_squared = (
        args.normalize_by_neurons_squared
    )  # Now defaults to True

    # Get verbose setting
    verbose = args.verbose

    # Determine whether to use population averaging method
    population_average = not args.neuron_average

    # Create stimulus iterator with test mode handling
    contrast_stim = st.ContrastStimulus()

    # In test mode, limit the number of trials
    if args.test:
        print(f"Running in test mode with {args.test_trials} trials per condition")
        # Extract just the test trials
        all_trials = [
            (angle, contrast, trial) for angle, contrast, trial in contrast_stim
        ]

        # Group trials by angle and contrast
        from collections import defaultdict

        grouped_trials = defaultdict(list)
        for angle, contrast, trial in all_trials:
            grouped_trials[(angle, contrast)].append(trial)

        # For each group, take only the first test_trials trials
        test_trials = []
        for (angle, contrast), trials in grouped_trials.items():
            for trial in trials[: min(args.test_trials, len(trials))]:
                test_trials.append((angle, contrast, trial))

        # Replace the trials with the test subset
        trial_params = test_trials
    else:
        # Create list of parameters for each trial
        trial_params = [
            (angle, contrast, trial) for angle, contrast, trial in contrast_stim
        ]

    # Set output directory for figures
    output_dir = os.path.join(basedir, subdir, "figures")

    if args.plot_only:
        # Load existing combined results
        combined_file = f"{basedir}/{subdir}/combined_spectra_{t_start}to{t_end}.json"
        if os.path.exists(combined_file):
            with open(combined_file, "r") as f:
                combined_results = json.load(f)

            # Set environment variable for plot variant
            if args.plot_variant:
                os.environ["PLOT_VARIANT"] = args.plot_variant

            print(
                f"Loading results from {combined_file} ({len(combined_results['cell_types'])} cell types)"
            )

            # Plot results for each cell type at specified angles
            for cell_type in combined_results["cell_types"]:
                for angle in args.angles:
                    plot_contrast_spectra(
                        combined_results,
                        cell_type,
                        angle=angle,
                        max_freq=args.max_freq,
                        output_dir=output_dir,
                        normalize=normalize,
                        normalize_by_neurons_squared=normalize_by_neurons_squared,
                        verbose=verbose,
                    )

            print(f"Plotting complete. Figures saved to {output_dir}")
            exit(0)  # Exit after plotting to avoid duplicate plotting
        else:
            print(f"Error: Combined results file not found: {combined_file}")
            exit(1)

    # If not plot_only, run the analysis
    if not args.plot_only:
        # Create partial function with fixed parameters
        process_trial_with_params = partial(
            process_trial,
            basedir=basedir,
            subdir=subdir,
            t_start=t_start,
            t_end=t_end,
            fs=fs,
            normalize=normalize,
            normalize_by_neurons_squared=normalize_by_neurons_squared,
            verbose=verbose,
            population_average=population_average,
        )

        # Create list of parameters for each trial
        trial_params = [
            (angle, contrast, trial) for angle, contrast, trial in contrast_stim
        ]

        # Create pool of workers and process trials in parallel
        print(
            f"Processing {len(trial_params)} trials using {args.n_processes} processes"
        )

        # Show a single progress bar for all trials
        # This displays one progress bar for the entire process instead of per trial
        with mp.Pool(processes=args.n_processes) as pool:
            results = list(
                tqdm(
                    pool.starmap(process_trial_with_params, trial_params),
                    total=len(trial_params),
                    desc="Processing trials",
                    unit="trial",
                )
            )

        # Combine results
        print("Combining trial results...")
        combined_results = combine_spectra_results(
            results, basedir, subdir, t_start, t_end, verbose=verbose
        )

        print(f"Total execution time: {time.time() - total_start_time:.2f}s")
    else:
        # Load existing combined results
        combined_file = f"{basedir}/{subdir}/combined_spectra_{t_start}to{t_end}.json"
        if os.path.exists(combined_file):
            with open(combined_file, "r") as f:
                combined_results = json.load(f)

            # Set environment variable for plot variant
            if args.plot_variant:
                os.environ["PLOT_VARIANT"] = args.plot_variant

            # Plot results for each cell type at specified angles
            for cell_type in combined_results["cell_types"]:
                for angle in args.angles:
                    plot_contrast_spectra(
                        combined_results,
                        cell_type,
                        angle=angle,
                        max_freq=args.max_freq,
                        output_dir=output_dir,
                        normalize=normalize,
                        normalize_by_neurons_squared=normalize_by_neurons_squared,
                        verbose=verbose,
                    )

    # Set output directory for figures
    output_dir = os.path.join(basedir, subdir, "figures")

    # Plot results for each cell type at 0 degrees
    print("Creating plots...")
    for cell_type in combined_results["cell_types"]:
        plot_contrast_spectra(
            combined_results,
            cell_type,
            angle=0,
            max_freq=args.max_freq,
            output_dir=output_dir,
            normalize=normalize,
            normalize_by_neurons_squared=normalize_by_neurons_squared,
            verbose=verbose,
        )

        # Plot for other specified angles
        for angle in args.angles:
            if angle != 0:  # Already plotted angle 0
                plot_contrast_spectra(
                    combined_results,
                    cell_type,
                    angle=angle,
                    max_freq=args.max_freq,
                    output_dir=output_dir,
                    normalize=normalize,
                    normalize_by_neurons_squared=normalize_by_neurons_squared,
                    verbose=verbose,
                )

    print(f"Total execution time: {time.time() - total_start_time:.2f}s")
    print(f"Figures saved to {output_dir}")
