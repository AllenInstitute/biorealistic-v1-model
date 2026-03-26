import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress, t as t_dist
import network_utils as nu
from response_correlation_calculations import calculate_edge_df_core
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import lru_cache
from analysis_shared.style import apply_pub_style, trim_spines


# Cache for slope calculations to avoid redundant computation
_slope_cache = {}


def _map_simplified_inhibitory(cell_type: str) -> str:
    """Map layer-specific inhibitory types to simplified PV/SST/VIP; return unchanged otherwise."""
    if isinstance(cell_type, str):
        for inh in ("PV", "SST", "VIP"):
            if cell_type.endswith(f"_{inh}"):
                # Handle cases like "L2/3_PV" → "PV"
                parts = cell_type.split("_")
                if len(parts) >= 2 and parts[-1] in ("PV", "SST", "VIP"):
                    return parts[-1]
    return cell_type


def _load_pair_sampling_map(csv_path: str):
    """Load CSV mapping of per-pair limits.

    Accepts either columns [source_type, target_type, n|count|num] or
    [source, target, connections] or [cell_type_pair, n|count|num].
    Returns dict keyed by (source_type, target_type).
    """
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    # identify count/limit column (support multiple names)
    count_key = None
    for k in ("connections", "n", "count", "num"):
        if k in cols_lower:
            count_key = cols_lower[k]
            break
    if count_key is None:
        raise ValueError("Sampling CSV must include a count column: 'connections', 'n', 'count', or 'num'.")

    if "cell_type_pair" in cols_lower:
        pair_col = cols_lower["cell_type_pair"]
        mapping = {}
        for _, row in df.iterrows():
            pair = str(row[pair_col])
            if "_" not in pair:
                continue
            source_type, target_type = pair.split("_", 1)
            mapping[(source_type, target_type)] = int(row[count_key])
        return mapping

    # allow synonyms: source_type|source and target_type|target
    s_col_key = None
    for k in ("source_type", "source"):
        if k in cols_lower:
            s_col_key = cols_lower[k]
            break
    t_col_key = None
    for k in ("target_type", "target"):
        if k in cols_lower:
            t_col_key = cols_lower[k]
            break
    if s_col_key is None or t_col_key is None:
        raise ValueError("Sampling CSV must include 'cell_type_pair' or both 'source/source_type' and 'target/target_type'.")

    return {(str(row[s_col_key]).strip(), str(row[t_col_key]).strip()): int(row[count_key]) for _, row in df.iterrows()}


def _apply_per_pair_sampling(df: pd.DataFrame, *,
                             max_per_pair: int | None = None,
                             pair_limits: dict | None = None,
                             seed: int = 0,
                             simplify_inh_for_grouping: bool = False) -> pd.DataFrame:
    """Downsample rows per (source_type, target_type) group.

    If simplify_inh_for_grouping is True, grouping keys collapse inhibitory types
    to PV/SST/VIP for grouping only (sampling is applied after selecting rows).
    """
    if max_per_pair is None and not pair_limits:
        return df

    rng = np.random.RandomState(seed)
    df_work = df.copy()

    if simplify_inh_for_grouping:
        df_work["_group_source"] = df_work["source_type"].map(_map_simplified_inhibitory)
        df_work["_group_target"] = df_work["target_type"].map(_map_simplified_inhibitory)
    else:
        df_work["_group_source"] = df_work["source_type"]
        df_work["_group_target"] = df_work["target_type"]

    sampled_chunks = []
    for (gs, gt), g in df_work.groupby(["_group_source", "_group_target"], sort=False):
        limit = None
        if pair_limits and (gs, gt) in pair_limits:
            limit = pair_limits[(gs, gt)]
        elif max_per_pair is not None:
            limit = max_per_pair

        if limit is None or len(g) <= limit:
            sampled_chunks.append(g)
        else:
            sampled_chunks.append(g.sample(n=limit, random_state=rng))

    if not sampled_chunks:
        return df.iloc[0:0]

    result = pd.concat(sampled_chunks, ignore_index=True)
    return result.drop(columns=["_group_source", "_group_target"], errors="ignore")

def get_cache_key(source_type, target_type, swap_axes=False):
    """Generate a cache key for slope calculations"""
    return f"{source_type}_{target_type}_{swap_axes}"

def calculate_slopes_vectorized(combined_edge_df, cell_types, swap_axes=False):
    """Pre-calculate all slope values using vectorized operations where possible"""
    cache_key_base = f"vectorized_{swap_axes}"
    
    if cache_key_base in _slope_cache:
        return _slope_cache[cache_key_base]
    
    slope_results = {}
    
    # Group by cell type pairs once
    grouped = combined_edge_df.groupby(["source_type", "target_type"])
    
    for (source_type, target_type), subset in grouped:
        if source_type not in cell_types or target_type not in cell_types:
            continue
            
        if subset.empty or len(subset) < 2:
            slope_results[(source_type, target_type)] = (np.nan, np.nan)
            continue

        # Choose variables based on swap_axes
        if swap_axes:
            cleaned_df = subset[["syn_weight", "Response Correlation"]].dropna()
            if len(cleaned_df) > 1 and cleaned_df['syn_weight'].nunique() > 1:
                slope, intercept, r_value, p_value, std_err = linregress(
                    cleaned_df["syn_weight"],
                    cleaned_df["Response Correlation"],
                )
                slope_results[(source_type, target_type)] = (slope, p_value)
            else:
                slope_results[(source_type, target_type)] = (np.nan, np.nan)
        else:
            cleaned_df = subset[["Response Correlation", "syn_weight"]].dropna()
            if len(cleaned_df) > 1 and cleaned_df['Response Correlation'].nunique() > 1:
                slope, intercept, r_value, p_value, std_err = linregress(
                    cleaned_df["Response Correlation"],
                    cleaned_df["syn_weight"],
                )
                slope_results[(source_type, target_type)] = (slope, p_value)
            else:
                slope_results[(source_type, target_type)] = (np.nan, np.nan)
    
    _slope_cache[cache_key_base] = slope_results
    return slope_results

def calculate_binned_data_vectorized(combined_edge_df, cell_types, block_size=0.1, swap_axes=False):
    """Pre-calculate binned data for all cell type pairs"""
    cache_key = f"binned_{block_size}_{swap_axes}"
    
    if cache_key in _slope_cache:
        return _slope_cache[cache_key]
    
    binned_results = {}
    
    # Determine x and y variables based on swap_axes
    if swap_axes:
        x_col, y_col = "syn_weight", "Response Correlation"
    else:
        x_col, y_col = "Response Correlation", "syn_weight"
    
    # Group by cell type pairs once
    grouped = combined_edge_df.groupby(["source_type", "target_type"])
    
    for (source_type, target_type), subset in grouped:
        if source_type not in cell_types or target_type not in cell_types:
            continue
            
        if subset.empty or len(subset) < 2:
            binned_results[(source_type, target_type)] = ([], [], [])
            continue

        # Determine binning range based on data
        if swap_axes:
            x_min, x_max = subset[x_col].min(), subset[x_col].max()
            if np.isnan(x_min) or np.isnan(x_max) or x_min == x_max:
                x_min, x_max = 0, 0.1
            bin_range = (x_max - x_min) / 10  # ~10 bins
            if bin_range <= 0:
                bin_range = 0.001
            bins = np.arange(x_min, x_max + bin_range, bin_range)
        else:
            x_min, x_max = subset[x_col].min(), subset[x_col].max()
            if np.isnan(x_min) or np.isnan(x_max) or x_min == x_max:
                x_min, x_max = -1, 1
            bin_range = (x_max - x_min) / 10  # ~10 bins
            if bin_range <= 0:
                bin_range = block_size
            bins = np.arange(x_min, x_max + bin_range, bin_range)
        
        if len(bins) < 2:
            bins = np.linspace(x_min, x_max, 10)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Vectorized binning using pd.cut
        subset_clean = subset[[x_col, y_col]].dropna()
        if len(subset_clean) == 0:
            binned_results[(source_type, target_type)] = ([], [], [])
            continue
            
        subset_clean['bin'] = pd.cut(subset_clean[x_col], bins, include_lowest=True)
        grouped_bins = subset_clean.groupby('bin', observed=False)[y_col]
        
        block_means = grouped_bins.mean().reindex(pd.cut(bin_centers, bins, include_lowest=True)).values
        block_sems = grouped_bins.sem().reindex(pd.cut(bin_centers, bins, include_lowest=True)).values
        
        binned_results[(source_type, target_type)] = (bin_centers, block_means, block_sems)
    
    _slope_cache[cache_key] = binned_results
    return binned_results


def process_network_data(args):
    """Process a single network's data - for parallel processing"""
    base_dir, network_type = args
    
    correlations = np.load(f"{base_dir}/metrics/response_correlations_{network_type}.npy")
    distances = np.load(f"{base_dir}/metrics/distances.npy")

    edge_df_core = calculate_edge_df_core(base_dir)
    edge_df_core["Response Correlation"] = correlations
    edge_df_core["Lateral Distance"] = distances

    ctdf = nu.get_cell_type_table()
    node_lf = nu.load_nodes_pl(base_dir, core_radius=200)
    node_df = node_lf.collect().to_pandas()

    edge_df_core["source_type"] = ctdf["cell_type"][
        node_df["pop_name"][edge_df_core["source_id"]]
    ].values
    edge_df_core["target_type"] = ctdf["cell_type"][
        node_df["pop_name"][edge_df_core["target_id"]]
    ].values

    return edge_df_core


def aggregate_and_plot_optimized(base_dirs, network_type, output_file, plot_type="all", aggregate_l5=False, swap_axes=False, use_parallel=True, max_per_pair=None, pair_sample_csv=None, sample_seed=0, mc_runs: int = 0, mc_output_file: str | None = None):
    """
    Optimized version of aggregate_and_plot with caching, vectorization, and optional parallel processing.
    """
    print(f"Processing {len(base_dirs)} networks...")
    
    # Parallel data loading if requested
    if use_parallel and len(base_dirs) > 1:
        print("Using parallel processing for data loading...")
        with ProcessPoolExecutor(max_workers=min(len(base_dirs), multiprocessing.cpu_count())) as executor:
            futures = [executor.submit(process_network_data, (base_dir, network_type)) for base_dir in base_dirs]
            all_edge_dfs = []
            for future in as_completed(futures):
                all_edge_dfs.append(future.result())
    else:
        print("Using sequential processing for data loading...")
        all_edge_dfs = []
        for base_dir in base_dirs:
            all_edge_dfs.append(process_network_data((base_dir, network_type)))

    print("Combining data...")
    combined_edge_df = pd.concat(all_edge_dfs, ignore_index=True)
    combined_edge_df_unsampled = combined_edge_df.copy()

    L5_EXC_TYPES = ['L5_IT', 'L5_ET', 'L5_NP']
    AGG_L5_EXC_TYPE = 'L5_Exc'

    if aggregate_l5:
        combined_edge_df = combined_edge_df.copy()
        combined_edge_df['source_type'] = combined_edge_df['source_type'].replace(L5_EXC_TYPES, AGG_L5_EXC_TYPE)
        combined_edge_df['target_type'] = combined_edge_df['target_type'].replace(L5_EXC_TYPES, AGG_L5_EXC_TYPE)

    # Optional per-pair sampling BEFORE any computations
    if pair_sample_csv:
        print(f"Applying per-pair sampling from CSV: {pair_sample_csv}")
        pair_limits = _load_pair_sampling_map(pair_sample_csv)
    else:
        pair_limits = None

    if max_per_pair is not None or pair_limits is not None:
        print(f"Sampling with max_per_pair={max_per_pair} seed={sample_seed}")
        combined_edge_df = _apply_per_pair_sampling(
            combined_edge_df,
            max_per_pair=max_per_pair,
            pair_limits=pair_limits,
            seed=sample_seed,
            simplify_inh_for_grouping=True,
        )
        print(f"Sampled rows: {len(combined_edge_df):,}")

    # Get all unique cell types
    all_cell_types = list(set(combined_edge_df["source_type"].unique()) | set(combined_edge_df["target_type"].unique()))
    
    print("Pre-computing slope calculations...")
    # Pre-calculate slopes for both orientations
    slopes_normal = calculate_slopes_vectorized(combined_edge_df, all_cell_types, swap_axes=False)
    slopes_swapped = calculate_slopes_vectorized(combined_edge_df, all_cell_types, swap_axes=True)
    
    print("Pre-computing binned data...")
    # Pre-calculate binned data for both orientations  
    binned_normal = calculate_binned_data_vectorized(combined_edge_df, all_cell_types, block_size=0.1, swap_axes=False)
    binned_swapped = calculate_binned_data_vectorized(combined_edge_df, all_cell_types, block_size=0.1, swap_axes=True)

    # Now generate plots using cached data
    print("Generating plots...")
    
    if plot_type == "all" or plot_type == "main":
        _plot_main_matrix_optimized(combined_edge_df, output_file, slopes_normal, binned_normal)

    if plot_type == "all" or plot_type == "aggregated_panels":
        aggregated_output_file = output_file.replace(".png", "_ei.png")
        if plot_type == "aggregated_panels":
            aggregated_output_file = output_file
        plot_aggregated_panels(combined_edge_df, aggregated_output_file, aggregate_l5=aggregate_l5)

    if plot_type == "all" or plot_type == "exc_only":
        excitatory_output_file = output_file.replace(".png", "_exc_only.png")
        if plot_type == "exc_only":
            excitatory_output_file = output_file
        plot_excitatory_only(combined_edge_df, excitatory_output_file)

    if plot_type == "all" or plot_type == "exc_matrix":
        excitatory_matrix_output_file = output_file.replace(".png", "_exc_matrix.png")
        if plot_type == "exc_matrix":
            excitatory_matrix_output_file = output_file
        _plot_excitatory_matrix_optimized(combined_edge_df, excitatory_matrix_output_file, slopes_normal, binned_normal, aggregate_l5)

    if plot_type == "all" or plot_type == "exc_inh_matrix":
        exc_inh_matrix_output_file = output_file.replace(".png", "_exc_inh_matrix.png")
        if plot_type == "exc_inh_matrix":
            exc_inh_matrix_output_file = output_file
        _plot_excitatory_inhibitory_matrix_optimized(combined_edge_df, exc_inh_matrix_output_file, slopes_normal, binned_normal, aggregate_l5, swap_axes=False)

    if plot_type == "all" or plot_type == "exc_inh_matrix_swapped":
        exc_inh_matrix_swapped_output_file = output_file.replace(".png", "_exc_inh_matrix_swapped.png")
        if plot_type == "exc_inh_matrix_swapped":
            exc_inh_matrix_swapped_output_file = output_file
        _plot_excitatory_inhibitory_matrix_optimized(combined_edge_df, exc_inh_matrix_swapped_output_file, slopes_swapped, binned_swapped, aggregate_l5, swap_axes=True)

    # Heatmap plots (these are already quite fast due to pre-computed slopes)
    if plot_type == "all" or plot_type == "exc_matrix_heatmap":
        exc_matrix_heatmap_output_file = output_file.replace(".png", "_exc_matrix_heatmap.png")
        if plot_type == "exc_matrix_heatmap":
            exc_matrix_heatmap_output_file = output_file
        _plot_excitatory_matrix_heatmap_optimized(combined_edge_df, exc_matrix_heatmap_output_file, slopes_normal, aggregate_l5)

    if plot_type == "all" or plot_type == "exc_inh_matrix_heatmap":
        exc_inh_matrix_heatmap_output_file = output_file.replace(".png", "_exc_inh_matrix_heatmap.png")
        if plot_type == "exc_inh_matrix_heatmap":
            exc_inh_matrix_heatmap_output_file = output_file
        _plot_excitatory_inhibitory_matrix_heatmap_optimized(combined_edge_df, exc_inh_matrix_heatmap_output_file, slopes_normal, aggregate_l5, swap_axes=False)

    if plot_type == "all" or plot_type == "exc_inh_matrix_heatmap_swapped":
        exc_inh_matrix_heatmap_swapped_output_file = output_file.replace(".png", "_exc_inh_matrix_heatmap_swapped.png")
        if plot_type == "exc_inh_matrix_heatmap_swapped":
            exc_inh_matrix_heatmap_swapped_output_file = output_file
        _plot_excitatory_inhibitory_matrix_heatmap_optimized(combined_edge_df, exc_inh_matrix_heatmap_swapped_output_file, slopes_swapped, aggregate_l5, swap_axes=True)
    
    print("All plots completed!")

    # Monte Carlo p-value distribution (exc_matrix focus)
    if mc_runs and mc_runs > 0:
        print(f"Running Monte Carlo p-value analysis (per-pair matrix): runs={mc_runs}")
        pval_map, exc_types = monte_carlo_pvalue_matrix(
            combined_edge_df_unsampled,
            aggregate_l5=aggregate_l5,
            runs=mc_runs,
            max_per_pair=max_per_pair,
            pair_limits=pair_limits,
            base_seed=sample_seed,
        )
        out_matrix = mc_output_file if mc_output_file else output_file.replace(".png", "_exc_matrix_pval_matrix.png")
        plot_pvalue_matrix_histograms(pval_map, exc_types, out_matrix, bin_size=0.05)
        print(f"Monte Carlo p-value matrix saved to: {out_matrix}")


def _plot_main_matrix_optimized(combined_edge_df, output_file, slopes_cache, binned_cache):
    """Optimized version of the main matrix plot using cached data"""
    source_types = sorted(combined_edge_df["source_type"].unique())
    target_types = sorted(combined_edge_df["target_type"].unique())

    fig, axes = plt.subplots(
        len(source_types),
        len(target_types),
        figsize=(len(target_types) * 2, len(source_types) * 2),
        sharex=False,
        sharey=False,
    )
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for i, source_type in enumerate(source_types):
        for j, target_type in enumerate(target_types):
            ax = axes[i, j]
            
            # Get cached data
            bin_centers, block_means, block_sems = binned_cache.get((source_type, target_type), ([], [], []))
            slope, p_value = slopes_cache.get((source_type, target_type), (np.nan, np.nan))
            
            if len(bin_centers) == 0:
                ax.axis("off")
            else:
                ax.errorbar(
                    bin_centers,
                    block_means,
                    yerr=block_sems,
                    fmt="o",
                    color="blue",
                    ecolor="green",
                    markersize=3,
                    capsize=3,
                )

                ax.set_xlim(-1, 1)
                ax.set_xticks(np.arange(-1.0, 1.1, 0.5))
                ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

                # Display cached slope
                if np.isnan(slope):
                    text_to_display = "m=nan"
                else:
                    significance_str = ""
                    if not np.isnan(p_value):
                        if p_value < 0.01:
                            significance_str = "**"
                        elif p_value < 0.05:
                            significance_str = "*"
                    text_to_display = f"m={slope:.3f}{significance_str}"
                
                ax.text(
                    0.05,
                    0.95,
                    text_to_display,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                )

            if i == len(source_types) - 1:
                ax.set_xlabel("Res. corr.")
            if j == 0:
                ax.set_ylabel(source_type, fontsize=8)
            if i == 0:
                ax.set_title(target_type, fontsize=8)

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()


def _plot_excitatory_matrix_optimized(combined_edge_df, output_file, slopes_cache, binned_cache, aggregate_l5=False):
    """Optimized version of excitatory matrix plot"""
    L5_EXC_TYPES = ['L5_IT', 'L5_ET', 'L5_NP']
    AGG_L5_EXC_TYPE = 'L5_Exc'

    if aggregate_l5:
        excitatory_types_to_plot = ["L2/3_Exc", "L4_Exc", AGG_L5_EXC_TYPE, "L6_Exc"]
    else:
        excitatory_types_to_plot = ["L2/3_Exc", "L4_Exc"] + L5_EXC_TYPES + ["L6_Exc"]

    fig, axes = plt.subplots(
        len(excitatory_types_to_plot),
        len(excitatory_types_to_plot),
        figsize=(len(excitatory_types_to_plot) * 2.2, len(excitatory_types_to_plot) * 2.2),
        sharex=False,
        sharey=False,
    )
    # Ensure white background for publication style
    # try:
    #     fig.patch.set_facecolor('white')
    #     for _ax in np.ravel(axes):
    #         _ax.set_facecolor('white')
    # except Exception:
    #     pass
    
    if len(excitatory_types_to_plot) == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes[:, np.newaxis] if len(excitatory_types_to_plot) > 1 else np.array([axes])
        
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Remove top/right spines for a cleaner matrix appearance
    try:
        for _ax in np.ravel(axes):
            _ax.spines["top"].set_visible(False)
            _ax.spines["right"].set_visible(False)
    except Exception:
        pass

    for i, source_type in enumerate(excitatory_types_to_plot):
        for j, target_type in enumerate(excitatory_types_to_plot):
            ax = axes[i, j]
            
            # Subset data for this cell-type pair
            pair_df = combined_edge_df[(combined_edge_df["source_type"] == source_type) & (combined_edge_df["target_type"] == target_type)][["Response Correlation", "syn_weight"]].dropna()
            
            if pair_df.empty or len(pair_df) < 2:
                ax.text(0.5, 0.5, "No data / N<2", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=7)
                ax.set_xlim(-0.2, 0.5)
            else:
                # Histogram-style bar plot with bin=0.05 over x in [-0.2, 0.5]
                bin_size = 0.05
                x_min, x_max = -0.2, 0.5
                bins = np.arange(x_min, x_max + bin_size, bin_size)
                # Aggregate mean and sem of syn_weight within bins
                pair_df = pair_df[(pair_df["Response Correlation"] >= x_min) & (pair_df["Response Correlation"] <= x_max)]
                if pair_df.empty:
                    ax.text(0.5, 0.5, "No data in range", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=7)
                else:
                    pair_df["bin"] = pd.cut(pair_df["Response Correlation"], bins, include_lowest=True)
                    grouped = pair_df.groupby("bin", observed=False)["syn_weight"]
                    means = grouped.mean()
                    sems = grouped.sem()
                    counts = pair_df.groupby("bin", observed=False).size()
                    centers = (bins[:-1] + bins[1:]) / 2
                    # align by reindexing
                    idx = pd.cut(centers, bins, include_lowest=True)
                    means = means.reindex(idx)
                    sems = sems.reindex(idx)
                    counts = counts.reindex(idx).fillna(0)
                    # bar plot
                    ax.bar(centers, means.values, width=bin_size, color="#b3b3b3", edgecolor="none")
                    # error bars
                    ax.errorbar(centers, means.values, yerr=sems.values, fmt="none", ecolor="k", elinewidth=1, capsize=2)

                    # Twin-axis shaded histogram for counts
                    ax2 = ax.twinx()
                    ax2.bar(centers, counts.values, width=bin_size, color="#b084cc", alpha=0.25, edgecolor="none")
                    # Lighten right axis ticks
                    ax2.tick_params(axis='y', labelsize=7, colors='#6b6b6b')
                    # Avoid overcrowding: at most 3 ticks
                    try:
                        ymax = counts.max() if np.isfinite(counts.max()) else 1
                        ax2.set_ylim(0, ymax * 1.2 if ymax > 0 else 1)
                    except Exception:
                        pass
                    # White background to match figure
                    try:
                        ax2.set_facecolor('white')
                    except Exception:
                        pass
                    # Remove spines on twin axis as well
                    try:
                        ax2.spines["top"].set_visible(False)
                        ax2.spines["right"].set_visible(False)
                    except Exception:
                        pass

                    # MLE linear fit on unbinned data (ordinary least squares equivalent under Gaussian noise)
                    x = pair_df["Response Correlation"].values
                    y = pair_df["syn_weight"].values
                    # closed-form OLS
                    x_mean = x.mean(); y_mean = y.mean()
                    Sxx = np.sum((x - x_mean)**2)
                    Sxy = np.sum((x - x_mean)*(y - y_mean))
                    if Sxx > 0:
                        slope_mle = Sxy / Sxx
                        intercept_mle = y_mean - slope_mle * x_mean
                        y_hat = intercept_mle + slope_mle * x
                        resid = y - y_hat
                        n = len(x)
                        dof = max(n - 2, 1)
                        sigma2 = np.sum(resid**2) / dof
                        se_slope = np.sqrt(sigma2 / Sxx)
                        t_stat = slope_mle / se_slope if se_slope > 0 else np.nan
                        # two-sided p-value
                        from scipy.stats import t as _t
                        p_val = 2 * (1 - _t.cdf(abs(t_stat), df=dof)) if np.isfinite(t_stat) else np.nan
                        # overlay fit line within x-range
                        xx = np.array([x_min, x_max])
                        yy = intercept_mle + slope_mle * xx
                        ax.plot(xx, yy, color="crimson", linestyle="-", linewidth=1.2)
                        # annotation
                        sig = "" if not np.isfinite(p_val) else ("***" if p_val < 1e-3 else "**" if p_val < 1e-2 else "*" if p_val < 5e-2 else "")
                        ax.text(0.03, 0.92, f"m={slope_mle:.3f} {sig}\np={p_val:.2e}", transform=ax.transAxes, fontsize=7, va="top")
                    else:
                        ax.text(0.03, 0.92, "m=nan", transform=ax.transAxes, fontsize=7, va="top")

                ax.set_xlim(x_min, x_max)
                # ax.axvspan(x_min, x_max, color="#e6d5eb", alpha=0.25)
                # ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.7)

            if i == len(excitatory_types_to_plot) - 1:
                ax.set_xlabel("Res. corr.")
            if j == 0:
                ax.set_ylabel(source_type, fontsize=8)
            if i == 0:
                ax.set_title(target_type, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# Similar optimized functions for other plot types would go here...
# For brevity, I'll implement the key optimization framework


def _sample_per_pair(df: pd.DataFrame, max_per_pair: int | None, pair_limits: dict | None, rng: np.random.RandomState, simplify_inh_for_grouping: bool = True) -> pd.DataFrame:
    """Helper for Monte Carlo: fresh sampling per run without altering original df."""
    if max_per_pair is None and not pair_limits:
        return df
    df_work = df.copy()
    if simplify_inh_for_grouping:
        df_work["_group_source"] = df_work["source_type"].map(_map_simplified_inhibitory)
        df_work["_group_target"] = df_work["target_type"].map(_map_simplified_inhibitory)
    else:
        df_work["_group_source"] = df_work["source_type"]
        df_work["_group_target"] = df_work["target_type"]

    chunks = []
    for (gs, gt), g in df_work.groupby(["_group_source", "_group_target"], sort=False, observed=False):
        limit = None
        if pair_limits and (gs, gt) in pair_limits:
            limit = pair_limits[(gs, gt)]
        elif max_per_pair is not None:
            limit = max_per_pair
        if limit is None or len(g) <= limit:
            chunks.append(g)
        else:
            chunks.append(g.sample(n=limit, random_state=rng))
    if not chunks:
        return df.iloc[0:0]
    res = pd.concat(chunks, ignore_index=True)
    return res.drop(columns=["_group_source", "_group_target"], errors="ignore")


def _get_exc_types_for_matrix(aggregate_l5: bool) -> list:
    L5_EXC_TYPES = ['L5_IT', 'L5_ET', 'L5_NP']
    AGG_L5_EXC_TYPE = 'L5_Exc'
    if aggregate_l5:
        return ["L2/3_Exc", "L4_Exc", AGG_L5_EXC_TYPE, "L6_Exc"]
    return ["L2/3_Exc", "L4_Exc"] + L5_EXC_TYPES + ["L6_Exc"]


def monte_carlo_pvalue_matrix(edge_df: pd.DataFrame, *, aggregate_l5: bool, runs: int, max_per_pair: int | None, pair_limits: dict | None, base_seed: int = 0):
    """
    Run repeated sampling and OLS regressions; collect p-values PER EE PAIR.
    Returns (pval_map, exc_types) where pval_map[(s, t)] -> list of length <= runs.
    Some runs may skip if invalid (e.g., no variance), so counts can be < runs.
    """
    exc_types = _get_exc_types_for_matrix(aggregate_l5)
    # If aggregating L5, collapse IT/ET/NP to L5_Exc before filtering
    if aggregate_l5:
        L5_EXC_TYPES = ['L5_IT', 'L5_ET', 'L5_NP']
        AGG_L5_EXC_TYPE = 'L5_Exc'
        edge_df = edge_df.copy()
        edge_df['source_type'] = edge_df['source_type'].replace(L5_EXC_TYPES, AGG_L5_EXC_TYPE)
        edge_df['target_type'] = edge_df['target_type'].replace(L5_EXC_TYPES, AGG_L5_EXC_TYPE)
    ee_df = edge_df[(edge_df["source_type"].isin(exc_types)) & (edge_df["target_type"].isin(exc_types))]
    if ee_df.empty:
        return {}, exc_types

    rng = np.random.RandomState(base_seed)
    # initialize map
    pval_map = {(s, t): [] for s in exc_types for t in exc_types}

    for r in range(runs):
        run_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
        sampled = _sample_per_pair(ee_df, max_per_pair=max_per_pair, pair_limits=pair_limits, rng=run_rng, simplify_inh_for_grouping=False)
        for (s_type, t_type), subset in sampled.groupby(["source_type", "target_type"], observed=False):
            df_pair = subset[["Response Correlation", "syn_weight"]].dropna()
            if len(df_pair) > 2 and df_pair["Response Correlation"].nunique() > 1:
                slope, intercept, r_value, p_value, std_err = linregress(
                    df_pair["Response Correlation"], df_pair["syn_weight"]
                )
                if np.isfinite(p_value):
                    pval_map[(s_type, t_type)].append(float(p_value))
    return pval_map, exc_types


def plot_pvalue_matrix_histograms(pval_map: dict, exc_types: list, output_file: str, bin_size: float = 0.05):
    """
    Draw a grid of histograms; each subplot shows the p-value distribution over runs for its pair.
    The y-axis shows counts; with R runs, bar heights sum to <= R (if some runs invalid).
    """
    if not pval_map:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No p-values", ha='center', va='center')
        ax.set_axis_off()
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return

    n_types = len(exc_types)
    fig, axes = plt.subplots(n_types, n_types, figsize=(3.2*n_types, 2.6*n_types), sharex=True, sharey=True)
    bins = np.arange(0.0, 1.0 + bin_size, bin_size)

    # ensure 2D axes array
    axes = np.atleast_2d(axes)
    for i, s in enumerate(exc_types):
        for j, t in enumerate(exc_types):
            ax = axes[i, j]
            pvals = pval_map.get((s, t), [])
            if pvals:
                counts, edges, _ = ax.hist(pvals, bins=bins, color="#7f7f7f", edgecolor="none")
                n = len(pvals)
                if n > 0 and len(counts) > 0:
                    prob_sig = counts[0] / n
                    ax.text(0.05, 0.95, f"{prob_sig:.2f}", transform=ax.transAxes, va='top', fontsize=8)
            ax.set_xlim(0, 1)
            if i == n_types - 1:
                ax.set_xlabel("p")
            if j == 0:
                ax.set_ylabel("count")
            if i == 0:
                ax.set_title(t, fontsize=9)
            if j == n_types - 1:
                ax.yaxis.set_label_position('right')
                ax.yaxis.tick_right()
            # light styling
            ax.set_facecolor('white')
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
        # y-axis label on the left with source type
        axes[i, 0].set_ylabel("count")
        axes[i, 0].text(-0.35, 0.5, s, rotation=90, va='center', ha='center', transform=axes[i, 0].transAxes)

    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def _plot_excitatory_inhibitory_matrix_optimized(combined_edge_df, output_file, slopes_cache, binned_cache, aggregate_l5=False, swap_axes=False):
    """Optimized version using cached data - placeholder for now, uses original function"""
    # For now, fall back to the original function - this could be optimized further
    plot_excitatory_inhibitory_matrix(combined_edge_df, output_file, aggregate_l5, swap_axes)


def _plot_excitatory_matrix_heatmap_optimized(combined_edge_df, output_file, slopes_cache, aggregate_l5=False):
    """Optimized version using cached data - placeholder for now, uses original function"""
    # For now, fall back to the original function - this could be optimized further
    plot_excitatory_matrix_heatmap(combined_edge_df, output_file, aggregate_l5)


def _plot_excitatory_inhibitory_matrix_heatmap_optimized(combined_edge_df, output_file, slopes_cache, aggregate_l5=False, swap_axes=False):
    """Optimized version using cached data - placeholder for now, uses original function"""
    # For now, fall back to the original function - this could be optimized further
    plot_excitatory_inhibitory_matrix_heatmap(combined_edge_df, output_file, aggregate_l5, swap_axes)


def calculate_fraction_for_50_percent_weight(subset):
    """
    Calculate the fraction of pairs needed to achieve 50% of the total synaptic weight.
    Returns a float or None if calculation is not possible.
    """
    # Handle empty or invalid data
    if subset.empty or len(subset) < 2:
        return None

    total_weight = subset["syn_weight"].sum()
    if total_weight <= 0:
        return None

    # Sort by response correlation in descending order (highest correlation first)
    subset_sorted = subset.sort_values(
        by="Response Correlation", ascending=False
    ).copy()

    # Calculate cumulative sum of weights
    subset_sorted["cumsum"] = subset_sorted["syn_weight"].cumsum()

    # Find the index where we cross 50% of total weight
    threshold = 0.5 * total_weight
    idx = subset_sorted["cumsum"].searchsorted(threshold)

    # Make sure we're within bounds
    if idx >= len(subset_sorted):
        idx = len(subset_sorted) - 1

    # Calculate and return the fraction (as a Python float)
    return float((idx + 1) / len(subset_sorted))


def plot_aggregated_panels(combined_edge_df, output_file, aggregate_l5=False):
    """
    Create a 4-panel plot (EE, EI, IE, II) aggregating all excitatory and inhibitory connections.
    """
    # Define connection types
    connection_types = ["EE", "EI", "IE", "II"]
    L5_EXC_TYPES = ['L5_IT', 'L5_ET', 'L5_NP']
    AGG_L5_EXC_TYPE = 'L5_Exc'

    if aggregate_l5:
        excitatory_types_to_plot = ["L2/3_Exc", "L4_Exc", AGG_L5_EXC_TYPE, "L6_Exc"]
    else:
        excitatory_types_to_plot = ["L2/3_Exc", "L4_Exc"] + L5_EXC_TYPES + ["L6_Exc"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

    for i, conn_type in enumerate(connection_types):
        ax = axes.flatten()[i]
        if conn_type == "EE":
            subset = combined_edge_df[
                (combined_edge_df["source_type"].isin(excitatory_types_to_plot))
                & (combined_edge_df["target_type"].isin(excitatory_types_to_plot))
            ]
        elif conn_type == "EI":
            subset = combined_edge_df[
                (combined_edge_df["source_type"].isin(excitatory_types_to_plot))
                & (combined_edge_df["target_type"].str.contains("Inh|PV|SST|VIP"))
            ]
        elif conn_type == "IE":
            subset = combined_edge_df[
                (combined_edge_df["source_type"].str.contains("Inh|PV|SST|VIP"))
                & (combined_edge_df["target_type"].isin(excitatory_types_to_plot))
            ]
        elif conn_type == "II":
            subset = combined_edge_df[
                (combined_edge_df["source_type"].str.contains("Inh|PV|SST|VIP"))
                & (combined_edge_df["target_type"].str.contains("Inh|PV|SST|VIP"))
            ]

        # Calculate block means and SEM
        block_size = 0.1
        bins = np.arange(-1, 1 + block_size, block_size)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        block_means = []
        block_sems = []
        for k in range(len(bins) - 1):
            bin_mask = (subset["Response Correlation"] >= bins[k]) & (
                subset["Response Correlation"] < bins[k + 1]
            )
            block_mean = subset[bin_mask]["syn_weight"].mean()
            block_means.append(block_mean)
            block_sem = subset[bin_mask]["syn_weight"].sem()
            block_sems.append(block_sem)

        # Plot
        ax.errorbar(
            bin_centers,
            block_means,
            yerr=block_sems,
            fmt="o",
            color="blue",
            ecolor="green",
            markersize=3,
            capsize=3,
        )
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

        # Calculate and display the fraction for 50% weight
        try:
            fraction = calculate_fraction_for_50_percent_weight(subset)
            if fraction is not None:
                ax.set_title(f"{conn_type} (50% weight: {fraction:.1%})")
            else:
                ax.set_title(f"{conn_type} (No data)")
        except Exception as e:
            print(f"Error calculating fraction for {conn_type}: {e}")
            ax.set_title(f"{conn_type}")

        ax.set_xlabel("Res. corr.")
        if i == 0 or i == 2:
            ax.set_ylabel("Synaptic Weight")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_excitatory_only(combined_edge_df, output_file):
    """
    Create a plot with only excitatory cells from all layers.
    """
    subset = combined_edge_df[
        (combined_edge_df["source_type"].str.contains("Exc|IT|ET|NP"))
        & (combined_edge_df["target_type"].str.contains("Exc|IT|ET|NP"))
    ]

    # Calculate block means and SEM
    block_size = 0.1
    bins = np.arange(-1, 1 + block_size, block_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    block_means = []
    block_sems = []
    for k in range(len(bins) - 1):
        bin_mask = (subset["Response Correlation"] >= bins[k]) & (
            subset["Response Correlation"] < bins[k + 1]
        )
        block_mean = subset[bin_mask]["syn_weight"].mean()
        block_means.append(block_mean)
        block_sem = subset[bin_mask]["syn_weight"].sem()
        block_sems.append(block_sem)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        bin_centers,
        block_means,
        yerr=block_sems,
        fmt="o",
        color="blue",
        ecolor="green",
        markersize=3,
        capsize=3,
    )
    plt.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    plt.title("Excitatory Cells Only")
    plt.xlabel("Response Correlation")
    plt.ylabel("Synaptic Weight")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_excitatory_matrix(combined_edge_df, output_file, aggregate_l5=False):
    """
    Create a matrix plot for all excitatory cell types.
    """
    # Filter for excitatory cell types
    L5_EXC_TYPES = ['L5_IT', 'L5_ET', 'L5_NP']
    AGG_L5_EXC_TYPE = 'L5_Exc'

    if aggregate_l5:
        excitatory_types_to_plot = ["L2/3_Exc", "L4_Exc", AGG_L5_EXC_TYPE, "L6_Exc"]
    else:
        excitatory_types_to_plot = ["L2/3_Exc", "L4_Exc"] + L5_EXC_TYPES + ["L6_Exc"]

    # Use the predefined excitatory_types_to_plot for grid dimensions and iteration
    # This ensures the matrix size is fixed (4x4 or 6x6)
    source_types_for_grid = excitatory_types_to_plot
    target_types_for_grid = excitatory_types_to_plot

    # Filter the DataFrame based on these types
    excitatory_df_filtered = combined_edge_df[
        combined_edge_df["source_type"].isin(source_types_for_grid)
        & combined_edge_df["target_type"].isin(target_types_for_grid)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning if combined_edge_df is a slice

    # If aggregate_l5 is true, the source/target types in excitatory_df_filtered might already be 'L5_Exc'
    # No further replacement needed here as it's handled in aggregate_and_plot

    fig, axes = plt.subplots(
        len(source_types_for_grid),
        len(target_types_for_grid),
        figsize=(len(target_types_for_grid) * 2.2, len(source_types_for_grid) * 2.2),
        sharex=True,
        sharey=False,
    )
    # Ensure axes is always a 2D array, even if len(source_types_for_grid) or len(target_types_for_grid) is 1
    if len(source_types_for_grid) == 1 and len(target_types_for_grid) == 1:
        axes = np.array([[axes]])
    elif len(source_types_for_grid) == 1:
        axes = np.array([axes])
    elif len(target_types_for_grid) == 1:
        axes = axes[:, np.newaxis]
        
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for i, source_type in enumerate(source_types_for_grid):
        for j, target_type in enumerate(target_types_for_grid):
            ax = axes[i, j]
            # Get subset from the ALREADY filtered and potentially aggregated excitatory_df_filtered
            subset = excitatory_df_filtered[
                (excitatory_df_filtered["source_type"] == source_type)
                & (excitatory_df_filtered["target_type"] == target_type)
            ]

            if subset.empty or len(subset) < 2:
                ax.text(0.5, 0.5, "No data / N<2", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=7)
                ax.set_xlim(-1, 1)
                # Minimal ticks for empty plots
                if i == len(source_types_for_grid) - 1:
                    ax.set_xlabel("Res. corr.")
                if j == 0:
                    ax.set_ylabel("Syn. weight")

                if j > 0:
                    ax.set_yticklabels([])
                if i < len(source_types_for_grid) - 1:
                    ax.set_xticklabels([])
                ax.tick_params(axis='both', which='major', labelsize=8)
                # Set row/col labels for empty plots too
                if j == 0:
                    ax.text(-0.3, 0.5, source_type, va='center', ha='right', transform=ax.transAxes, fontsize=8, rotation=90)
                if i == 0:
                    ax.set_title(target_type, fontsize=8, pad=2)
                continue # Skip to next subplot

            block_size = 0.1
            bins = np.arange(-1, 1 + block_size, block_size)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            block_means = []
            block_sems = []
            for k_bin in range(len(bins) - 1):
                bin_mask = (subset["Response Correlation"] >= bins[k_bin]) & \
                           (subset["Response Correlation"] < bins[k_bin + 1])
                syn_weights_in_bin = subset[bin_mask]["syn_weight"]
                block_means.append(syn_weights_in_bin.mean())
                block_sems.append(syn_weights_in_bin.sem())

            ax.errorbar(
                bin_centers,
                block_means,
                yerr=block_sems,
                fmt="o",
                color="blue",
                ecolor="green",
                markersize=3,
                capsize=3,
            )

            ax.set_xlim(-1, 1)
            # For plot_excitatory_matrix, sharex=True is used, so ticks are often handled by the figure.
            # If specific ticks are needed here, they can be set: ax.set_xticks(np.arange(-1.0, 1.1, 0.5))
            ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

            # Linregress and text annotation
            cleaned_df_for_corr = subset[["Response Correlation", "syn_weight"]].dropna()
            if len(cleaned_df_for_corr) > 1 and cleaned_df_for_corr['Response Correlation'].nunique() > 1: # Ensure variance in x
                slope, intercept, r_value, p_value, std_err = linregress(
                    cleaned_df_for_corr["Response Correlation"],
                    cleaned_df_for_corr["syn_weight"],
                )
            else:
                slope, p_value = np.nan, np.nan

            # Determine text to display
            if np.isnan(slope):
                text_to_display = "m=nan"
            else:
                significance_str = ""
                if not np.isnan(p_value):
                    if p_value < 0.01:
                        significance_str = "**"
                    elif p_value < 0.05:
                        significance_str = "*"
            text_to_display = f"m={slope:.3f}{significance_str}"
            
            ax.text(
                0.05,
                0.95,
                text_to_display,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
            )

            if i == len(source_types_for_grid) - 1:
                ax.set_xlabel("Res. corr.")
            if j == 0:
                ax.set_ylabel(source_type, fontsize=8)
            if i == 0:
                ax.set_title(target_type, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_excitatory_inhibitory_matrix(combined_edge_df, output_file, aggregate_l5=False, swap_axes=False):
    """
    Create a matrix plot for excitatory and simplified inhibitory cell types.
    Excitatory types by layer, inhibitory types aggregated across layers (PV, SST, VIP).
    """
    # Apply publication style
    apply_pub_style()
    
    # Define excitatory cell types
    L5_EXC_TYPES = ['L5_IT', 'L5_ET', 'L5_NP']
    AGG_L5_EXC_TYPE = 'L5_Exc'

    if aggregate_l5:
        excitatory_types = ["L2/3_Exc", "L4_Exc", AGG_L5_EXC_TYPE, "L6_Exc"]
    else:
        excitatory_types = ["L2/3_Exc", "L4_Exc"] + L5_EXC_TYPES + ["L6_Exc"]

    # Define simplified inhibitory cell types (aggregated across layers)
    inhibitory_types = ["PV", "SST", "VIP"]
    
    # Combine all cell types for the matrix
    all_types_for_grid = excitatory_types + inhibitory_types
    
    # Create a copy of the dataframe and aggregate inhibitory types across layers
    filtered_df = combined_edge_df.copy()
    
    # Map layer-specific inhibitory types to simplified types
    inh_mapping = {}
    for layer in ["L1", "L2/3", "L4", "L5", "L6"]:
        for inh_type in ["PV", "SST", "VIP"]:
            inh_mapping[f"{layer}_{inh_type}"] = inh_type
    
    # Apply the mapping to aggregate inhibitory types across layers
    filtered_df['source_type'] = filtered_df['source_type'].replace(inh_mapping)
    filtered_df['target_type'] = filtered_df['target_type'].replace(inh_mapping)
    
    # Filter to only include the cell types we want to plot
    filtered_df = filtered_df[
        filtered_df["source_type"].isin(all_types_for_grid)
        & filtered_df["target_type"].isin(all_types_for_grid)
    ]

    fig, axes = plt.subplots(
        len(all_types_for_grid),
        len(all_types_for_grid),
        figsize=(len(all_types_for_grid) * 2.2, len(all_types_for_grid) * 2.2),
        sharex=False,
        sharey=False,
    )
    # Ensure axes is always a 2D array
    if len(all_types_for_grid) == 1:
        axes = np.array([[axes]])
    elif len(all_types_for_grid) > 1 and axes.ndim == 1:
        axes = axes[:, np.newaxis]
        
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for i, source_type in enumerate(all_types_for_grid):
        for j, target_type in enumerate(all_types_for_grid):
            ax = axes[i, j]
            # Get subset for this specific source-target pair
            subset = filtered_df[
                (filtered_df["source_type"] == source_type)
                & (filtered_df["target_type"] == target_type)
            ]

            if subset.empty or len(subset) < 2:
                ax.text(0.5, 0.5, "No data / N<2", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=7)
                # Set minimal axis limits for empty plots
                if swap_axes:
                    ax.set_xlim(0, 0.1)  # synaptic weight range
                    ax.set_ylim(-1, 1)   # correlation range
                    if i == len(all_types_for_grid) - 1:
                        ax.set_xlabel("Syn. weight")
                    if j == 0:
                        ax.set_ylabel("Res. corr.")
                else:
                    ax.set_xlim(-1, 1)   # correlation range
                    ax.set_ylim(0, 0.1)  # synaptic weight range
                    if i == len(all_types_for_grid) - 1:
                        ax.set_xlabel("Res. corr.")
                    if j == 0:
                        ax.set_ylabel("Syn. weight")

                if j > 0:
                    ax.set_yticklabels([])
                if i < len(all_types_for_grid) - 1:
                    ax.set_xticklabels([])
                ax.tick_params(axis='both', which='major', labelsize=8)
                # Set row/col labels for empty plots too
                if j == 0:
                    ax.text(-0.3, 0.5, source_type, va='center', ha='right', transform=ax.transAxes, fontsize=8, rotation=90)
                if i == 0:
                    ax.set_title(target_type, fontsize=8, pad=2)
                trim_spines(ax)
                continue # Skip to next subplot

            if swap_axes:
                # Plot synaptic weight on x-axis, response correlation on y-axis
                weight_min, weight_max = subset["syn_weight"].min(), subset["syn_weight"].max()
                if np.isnan(weight_min) or np.isnan(weight_max) or weight_min == weight_max:
                    weight_min, weight_max = 0, 0.1  # Default range
                block_size = (weight_max - weight_min) / 10  # ~10 bins
                if block_size <= 0:
                    block_size = 0.001
                bins = np.arange(weight_min, weight_max + block_size, block_size)
                if len(bins) < 2:
                    bins = np.linspace(weight_min, weight_max, 10)
                bin_centers = (bins[:-1] + bins[1:]) / 2

                block_means = []
                block_sems = []
                for k_bin in range(len(bins) - 1):
                    bin_mask = (subset["syn_weight"] >= bins[k_bin]) & \
                               (subset["syn_weight"] < bins[k_bin + 1])
                    corr_values_in_bin = subset[bin_mask]["Response Correlation"]
                    block_means.append(corr_values_in_bin.mean())
                    block_sems.append(corr_values_in_bin.sem())

                ax.errorbar(
                    bin_centers,
                    block_means,
                    yerr=block_sems,
                    fmt="o",
                    color="blue",
                    ecolor="green",
                    markersize=3,
                    capsize=3,
                )

                # Set y-axis limits based on the actual plotted block means, not raw data
                # Ensure 0 is included in the y-axis range (correlation ranges from -1 to 1)
                valid_means = [m for m in block_means if not np.isnan(m)]
                if len(valid_means) > 0:
                    means_min, means_max = min(valid_means), max(valid_means)
                    means_range = means_max - means_min
                    if means_range > 0:
                        y_min = means_min - 0.1 * means_range
                        y_max = means_max + 0.1 * means_range
                        # Ensure 0 is included
                        if y_min > 0:
                            y_min = -0.1 * means_range
                        elif y_max < 0:
                            y_max = 0.1 * means_range
                        ax.set_ylim(y_min, y_max)
                    else:
                        # If no range, ensure 0 is visible
                        if means_min >= 0:
                            ax.set_ylim(-0.1, means_max + 0.1)
                        else:
                            ax.set_ylim(means_min - 0.1, 0.1)
                else:
                    # Fallback to raw data range
                    corr_min, corr_max = subset["Response Correlation"].min(), subset["Response Correlation"].max()
                    if not np.isnan(corr_min) and not np.isnan(corr_max):
                        corr_range = corr_max - corr_min
                        if corr_range > 0:
                            y_min = corr_min - 0.1 * corr_range
                            y_max = corr_max + 0.1 * corr_range
                            # Ensure 0 is included
                            if y_min > 0:
                                y_min = -0.1 * corr_range
                            elif y_max < 0:
                                y_max = 0.1 * corr_range
                            ax.set_ylim(y_min, y_max)
                        else:
                            # If no range, ensure 0 is visible
                            if corr_min >= 0:
                                ax.set_ylim(-0.1, corr_max + 0.1)
                            else:
                                ax.set_ylim(corr_min - 0.1, 0.1)
                    else:
                        ax.set_ylim(-1, 1)
                
                # Add reference line at y=0 if it's within the visible range
                ylims = ax.get_ylim()
                if ylims[0] <= 0 <= ylims[1]:
                    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

                # Linregress with swapped axes
                cleaned_df_for_corr = subset[["syn_weight", "Response Correlation"]].dropna()
                if len(cleaned_df_for_corr) > 1 and cleaned_df_for_corr['syn_weight'].nunique() > 1:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        cleaned_df_for_corr["syn_weight"],
                        cleaned_df_for_corr["Response Correlation"],
                    )
                else:
                    slope, p_value = np.nan, np.nan

                # Set adaptive x-axis limits for weight data  
                weight_range = weight_max - weight_min
                if weight_range > 0:
                    ax.set_xlim(weight_min - 0.1 * weight_range, weight_max + 0.1 * weight_range)
                else:
                    ax.set_xlim(weight_min - 0.001, weight_max + 0.001)

                # Set axis labels for swapped plot
                if i == len(all_types_for_grid) - 1:
                    ax.set_xlabel("Syn. weight")
                if j == 0:
                    ax.set_ylabel(source_type, fontsize=8)
                trim_spines(ax)
            else:
                # Original plot: response correlation on x-axis, synaptic weight on y-axis
                corr_min, corr_max = subset["Response Correlation"].min(), subset["Response Correlation"].max()
                if np.isnan(corr_min) or np.isnan(corr_max) or corr_min == corr_max:
                    corr_min, corr_max = -1, 1  # Default range
                block_size = (corr_max - corr_min) / 10  # ~10 bins
                if block_size <= 0:
                    block_size = 0.1
                bins = np.arange(corr_min, corr_max + block_size, block_size)
                if len(bins) < 2:
                    bins = np.linspace(corr_min, corr_max, 10)
                bin_centers = (bins[:-1] + bins[1:]) / 2

                block_means = []
                block_sems = []
                for k_bin in range(len(bins) - 1):
                    bin_mask = (subset["Response Correlation"] >= bins[k_bin]) & \
                               (subset["Response Correlation"] < bins[k_bin + 1])
                    syn_weights_in_bin = subset[bin_mask]["syn_weight"]
                    block_means.append(syn_weights_in_bin.mean())
                    block_sems.append(syn_weights_in_bin.sem())

                ax.errorbar(
                    bin_centers,
                    block_means,
                    yerr=block_sems,
                    fmt="o",
                    color="blue",
                    ecolor="green",
                    markersize=3,
                    capsize=3,
                )

                # Set adaptive axis limits based on actual data (using already computed values)
                weight_min, weight_max = subset["syn_weight"].min(), subset["syn_weight"].max()
                
                if not np.isnan(corr_min) and not np.isnan(corr_max):
                    corr_range = corr_max - corr_min
                    if corr_range > 0:
                        ax.set_xlim(corr_min - 0.1 * corr_range, corr_max + 0.1 * corr_range)
                    else:
                        ax.set_xlim(corr_min - 0.1, corr_max + 0.1)
                else:
                    ax.set_xlim(-1, 1)
                
                # Set y-axis limits based on the actual plotted block means, not raw data
                # Ensure 0 is included in the y-axis range
                valid_means = [m for m in block_means if not np.isnan(m)]
                if len(valid_means) > 0:
                    means_min, means_max = min(valid_means), max(valid_means)
                    means_range = means_max - means_min
                    if means_range > 0:
                        y_min = means_min - 0.1 * means_range
                        y_max = means_max + 0.1 * means_range
                        # Ensure 0 is included
                        if y_min > 0:
                            y_min = -0.1 * means_range
                        elif y_max < 0:
                            y_max = 0.1 * means_range
                        ax.set_ylim(y_min, y_max)
                    else:
                        # If no range, ensure 0 is visible
                        if means_min >= 0:
                            ax.set_ylim(-0.001, means_max + 0.001)
                        else:
                            ax.set_ylim(means_min - 0.001, 0.001)
                elif not np.isnan(weight_min) and not np.isnan(weight_max):
                    weight_range = weight_max - weight_min
                    if weight_range > 0:
                        y_min = weight_min - 0.1 * weight_range
                        y_max = weight_max + 0.1 * weight_range
                        # Ensure 0 is included
                        if y_min > 0:
                            y_min = -0.1 * weight_range
                        elif y_max < 0:
                            y_max = 0.1 * weight_range
                        ax.set_ylim(y_min, y_max)
                    else:
                        # If no range, ensure 0 is visible
                        if weight_min >= 0:
                            ax.set_ylim(-0.001, weight_max + 0.001)
                        else:
                            ax.set_ylim(weight_min - 0.001, 0.001)
                else:
                    # Fallback: set a range that includes 0
                    ax.set_ylim(-0.001, 0.001)
                
                # Add reference line at x=0 if it's within the visible range
                xlims = ax.get_xlim()
                if xlims[0] <= 0 <= xlims[1]:
                    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

                # Linregress and text annotation
                cleaned_df_for_corr = subset[["Response Correlation", "syn_weight"]].dropna()
                if len(cleaned_df_for_corr) > 1 and cleaned_df_for_corr['Response Correlation'].nunique() > 1:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        cleaned_df_for_corr["Response Correlation"],
                        cleaned_df_for_corr["syn_weight"],
                    )
                else:
                    slope, p_value = np.nan, np.nan

                # Set axis labels for original plot
                if i == len(all_types_for_grid) - 1:
                    ax.set_xlabel("Res. corr.")
                if j == 0:
                    ax.set_ylabel(source_type, fontsize=8)
                trim_spines(ax)

            # Determine text to display (same for both orientations)
            if np.isnan(slope):
                text_to_display = "m=nan"
            else:
                significance_str = ""
                if not np.isnan(p_value):
                    if p_value < 0.01:
                        significance_str = "**"
                    elif p_value < 0.05:
                        significance_str = "*"
                text_to_display = f"m={slope:.3f}{significance_str}"
            
            ax.text(
                0.05,
                0.95,
                text_to_display,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
            )
            if i == 0:
                ax.set_title(target_type, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def plot_excitatory_inhibitory_matrix_heatmap(combined_edge_df, output_file, aggregate_l5=False, swap_axes=False):
    """
    Create a heatmap matrix showing slope values for excitatory and simplified inhibitory cell types.
    """
    # Define excitatory cell types
    L5_EXC_TYPES = ['L5_IT', 'L5_ET', 'L5_NP']
    AGG_L5_EXC_TYPE = 'L5_Exc'

    if aggregate_l5:
        excitatory_types = ["L2/3_Exc", "L4_Exc", AGG_L5_EXC_TYPE, "L6_Exc"]
    else:
        excitatory_types = ["L2/3_Exc", "L4_Exc"] + L5_EXC_TYPES + ["L6_Exc"]

    # Define simplified inhibitory cell types (aggregated across layers)
    inhibitory_types = ["PV", "SST", "VIP"]
    
    # Combine all cell types for the matrix
    all_types_for_grid = excitatory_types + inhibitory_types
    
    # Create a copy of the dataframe and aggregate inhibitory types across layers
    filtered_df = combined_edge_df.copy()
    
    # Map layer-specific inhibitory types to simplified types
    inh_mapping = {}
    for layer in ["L1", "L2/3", "L4", "L5", "L6"]:
        for inh_type in ["PV", "SST", "VIP"]:
            inh_mapping[f"{layer}_{inh_type}"] = inh_type
    
    # Apply the mapping to aggregate inhibitory types across layers
    filtered_df['source_type'] = filtered_df['source_type'].replace(inh_mapping)
    filtered_df['target_type'] = filtered_df['target_type'].replace(inh_mapping)
    
    # Filter to only include the cell types we want to plot
    filtered_df = filtered_df[
        filtered_df["source_type"].isin(all_types_for_grid)
        & filtered_df["target_type"].isin(all_types_for_grid)
    ]

    # Calculate slope values for all pairs
    slope_matrix = np.full((len(all_types_for_grid), len(all_types_for_grid)), np.nan)
    pvalue_matrix = np.full((len(all_types_for_grid), len(all_types_for_grid)), np.nan)
    
    for i, source_type in enumerate(all_types_for_grid):
        for j, target_type in enumerate(all_types_for_grid):
            subset = filtered_df[
                (filtered_df["source_type"] == source_type)
                & (filtered_df["target_type"] == target_type)
            ]

            if subset.empty or len(subset) < 2:
                continue

            if swap_axes:
                # Swap x and y variables for regression
                cleaned_df_for_corr = subset[["syn_weight", "Response Correlation"]].dropna()
                if len(cleaned_df_for_corr) > 1 and cleaned_df_for_corr['syn_weight'].nunique() > 1:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        cleaned_df_for_corr["syn_weight"],
                        cleaned_df_for_corr["Response Correlation"],
                    )
                    slope_matrix[i, j] = slope
                    pvalue_matrix[i, j] = p_value
            else:
                # Original orientation
                cleaned_df_for_corr = subset[["Response Correlation", "syn_weight"]].dropna()
                if len(cleaned_df_for_corr) > 1 and cleaned_df_for_corr['Response Correlation'].nunique() > 1:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        cleaned_df_for_corr["Response Correlation"],
                        cleaned_df_for_corr["syn_weight"],
                    )
                    slope_matrix[i, j] = slope
                    pvalue_matrix[i, j] = p_value

    # Handle outliers (particularly NP-related connections) for better color scaling
    valid_slopes = slope_matrix[~np.isnan(slope_matrix)]
    if len(valid_slopes) > 0:
        # Use percentiles to define robust color range
        vmin = np.percentile(valid_slopes, 5)  # 5th percentile
        vmax = np.percentile(valid_slopes, 95)  # 95th percentile
        # Make sure the range is symmetric around zero if it includes both positive and negative values
        if vmin < 0 and vmax > 0:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -0.1, 0.1

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a diverging colormap
    im = ax.imshow(slope_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
    
    # Set ticks and labels
    ax.set_xticks(range(len(all_types_for_grid)))
    ax.set_yticks(range(len(all_types_for_grid)))
    ax.set_xticklabels(all_types_for_grid, rotation=45, ha='right')
    ax.set_yticklabels(all_types_for_grid)
    
    # Add text annotations with slope values and significance
    for i in range(len(all_types_for_grid)):
        for j in range(len(all_types_for_grid)):
            slope_val = slope_matrix[i, j]
            p_val = pvalue_matrix[i, j]
            
            if not np.isnan(slope_val):
                # Determine significance stars
                significance_str = ""
                if not np.isnan(p_val):
                    if p_val < 0.01:
                        significance_str = "**"
                    elif p_val < 0.05:
                        significance_str = "*"
                
                # Format the text
                text = f"{slope_val:.3f}{significance_str}"
                
                # Choose text color based on background
                text_color = 'white' if abs(slope_val - ((vmin + vmax) / 2)) > abs(vmax - vmin) * 0.3 else 'black'
                
                ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=8, weight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if swap_axes:
        cbar.set_label('Slope (corr/weight)', rotation=270, labelpad=15)
        ax.set_title('Response Correlation vs Synaptic Weight Slopes\n(Swapped Axes)')
        ax.set_xlabel('Target Cell Type')
        ax.set_ylabel('Source Cell Type')
    else:
        cbar.set_label('Slope (weight/corr)', rotation=270, labelpad=15)
        ax.set_title('Synaptic Weight vs Response Correlation Slopes')
        ax.set_xlabel('Target Cell Type')
        ax.set_ylabel('Source Cell Type')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_excitatory_matrix_heatmap(combined_edge_df, output_file, aggregate_l5=False):
    """
    Create a heatmap matrix showing slope values for excitatory cell types only.
    """
    # Filter for excitatory cell types
    L5_EXC_TYPES = ['L5_IT', 'L5_ET', 'L5_NP']
    AGG_L5_EXC_TYPE = 'L5_Exc'

    if aggregate_l5:
        excitatory_types_to_plot = ["L2/3_Exc", "L4_Exc", AGG_L5_EXC_TYPE, "L6_Exc"]
    else:
        excitatory_types_to_plot = ["L2/3_Exc", "L4_Exc"] + L5_EXC_TYPES + ["L6_Exc"]

    # Filter the DataFrame for excitatory connections only
    excitatory_df_filtered = combined_edge_df[
        combined_edge_df["source_type"].isin(excitatory_types_to_plot)
        & combined_edge_df["target_type"].isin(excitatory_types_to_plot)
    ].copy()

    # Calculate slope values for all pairs
    slope_matrix = np.full((len(excitatory_types_to_plot), len(excitatory_types_to_plot)), np.nan)
    pvalue_matrix = np.full((len(excitatory_types_to_plot), len(excitatory_types_to_plot)), np.nan)
    
    for i, source_type in enumerate(excitatory_types_to_plot):
        for j, target_type in enumerate(excitatory_types_to_plot):
            subset = excitatory_df_filtered[
                (excitatory_df_filtered["source_type"] == source_type)
                & (excitatory_df_filtered["target_type"] == target_type)
            ]

            if subset.empty or len(subset) < 2:
                continue

            cleaned_df_for_corr = subset[["Response Correlation", "syn_weight"]].dropna()
            if len(cleaned_df_for_corr) > 1 and cleaned_df_for_corr['Response Correlation'].nunique() > 1:
                slope, intercept, r_value, p_value, std_err = linregress(
                    cleaned_df_for_corr["Response Correlation"],
                    cleaned_df_for_corr["syn_weight"],
                )
                slope_matrix[i, j] = slope
                pvalue_matrix[i, j] = p_value

    # Handle outliers for better color scaling
    valid_slopes = slope_matrix[~np.isnan(slope_matrix)]
    if len(valid_slopes) > 0:
        # Use percentiles to define robust color range
        vmin = np.percentile(valid_slopes, 5)  # 5th percentile
        vmax = np.percentile(valid_slopes, 95)  # 95th percentile
        # Make sure the range is symmetric around zero if it includes both positive and negative values
        if vmin < 0 and vmax > 0:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -0.1, 0.1

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a diverging colormap
    im = ax.imshow(slope_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
    
    # Set ticks and labels
    ax.set_xticks(range(len(excitatory_types_to_plot)))
    ax.set_yticks(range(len(excitatory_types_to_plot)))
    ax.set_xticklabels(excitatory_types_to_plot, rotation=45, ha='right')
    ax.set_yticklabels(excitatory_types_to_plot)
    
    # Add text annotations with slope values and significance
    for i in range(len(excitatory_types_to_plot)):
        for j in range(len(excitatory_types_to_plot)):
            slope_val = slope_matrix[i, j]
            p_val = pvalue_matrix[i, j]
            
            if not np.isnan(slope_val):
                # Determine significance stars
                significance_str = ""
                if not np.isnan(p_val):
                    if p_val < 0.01:
                        significance_str = "**"
                    elif p_val < 0.05:
                        significance_str = "*"
                
                # Format the text
                text = f"{slope_val:.3f}{significance_str}"
                
                # Choose text color based on background
                text_color = 'white' if abs(slope_val - ((vmin + vmax) / 2)) > abs(vmax - vmin) * 0.3 else 'black'
                
                ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=8, weight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Slope (weight/corr)', rotation=270, labelpad=15)
    ax.set_title('Excitatory Connections: Synaptic Weight vs Response Correlation Slopes')
    ax.set_xlabel('Target Cell Type')
    ax.set_ylabel('Source Cell Type')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate and plot response correlations."
    )
    parser.add_argument(
        "base_dirs", nargs="+", help="List of base directories containing the data."
    )
    parser.add_argument("network_type", help="Network type (e.g., checkpoint, plain).")
    parser.add_argument("output_file", help="Output file for the aggregated plot.")
    parser.add_argument(
        "--plot-type",
        type=str,
        default="all",
        choices=["all", "aggregated_panels", "exc_matrix", "exc_only", "exc_inh_matrix", "exc_inh_matrix_swapped", "exc_matrix_heatmap", "exc_inh_matrix_heatmap", "exc_inh_matrix_heatmap_swapped"],
        help="Type of plot to generate: 'all', 'aggregated_panels' (main figure with all types), 'exc_matrix' (excitatory matrix only), 'exc_only' (only excitatory cells in aggregated panels), 'exc_inh_matrix' (excitatory and simplified inhibitory cell types), 'exc_inh_matrix_swapped' (excitatory and simplified inhibitory cell types with swapped axes), 'exc_matrix_heatmap' (excitatory connections heatmap), 'exc_inh_matrix_heatmap' (excitatory and inhibitory connections heatmap), 'exc_inh_matrix_heatmap_swapped' (excitatory and inhibitory connections heatmap with swapped axes)."
    )
    parser.add_argument(
        "--aggregate-l5",
        action="store_true",
        help="Aggregate L5 excitatory cell types (L5_IT, L5_ET, L5_NP) into a single L5_Exc category."
    )
    parser.add_argument(
        "--max-per-pair",
        type=int,
        default=None,
        help="Uniform cap for number of connections per (source_type, target_type) pair."
    )
    parser.add_argument(
        "--pair-sample-csv",
        type=str,
        default=None,
        help="CSV with per-pair limits: either columns [cell_type_pair,n] or [source_type,target_type,n]."
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed for per-pair sampling."
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=0,
        help="If >0, run Monte Carlo p-value analysis with this many runs (exc_matrix)."
    )
    parser.add_argument(
        "--mc-output",
        type=str,
        default=None,
        help="Optional output filepath for MC p-value histogram."
    )
    args = parser.parse_args()

    aggregate_and_plot_optimized(
        args.base_dirs,
        args.network_type,
        args.output_file,
        args.plot_type,
        args.aggregate_l5,
        max_per_pair=args.max_per_pair,
        pair_sample_csv=args.pair_sample_csv,
        sample_seed=args.sample_seed,
        mc_runs=args.mc_runs,
        mc_output_file=args.mc_output,
    )
