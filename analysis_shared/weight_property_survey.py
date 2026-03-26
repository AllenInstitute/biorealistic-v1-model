#!/usr/bin/env python3
"""Generate correlation matrices between synaptic weights and per-cell properties."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import network_utils as nu
from analysis_shared.grouping import apply_inh_simplification
from analysis_shared.style import apply_pub_style, trim_spines

TYPE_ORDER: List[str] = [
    "L2/3_Exc",
    "L4_Exc",
    "L5_IT",
    "L5_ET",
    "L5_NP",
    "L6_Exc",
    "PV",
    "SST",
    "VIP",
    "L1_Inh",
]
N_TYPES = len(TYPE_ORDER)
N_PAIRS = N_TYPES * N_TYPES


@dataclass
class PairStats:
    count: np.ndarray
    sum_w: np.ndarray
    sum_p: np.ndarray
    sum_w2: np.ndarray
    sum_p2: np.ndarray
    sum_wp: np.ndarray

    @classmethod
    def zeros(cls) -> "PairStats":
        zeros = np.zeros(N_PAIRS, dtype=np.float64)
        return cls(
            count=zeros.copy(),
            sum_w=zeros.copy(),
            sum_p=zeros.copy(),
            sum_w2=zeros.copy(),
            sum_p2=zeros.copy(),
            sum_wp=zeros.copy(),
        )

    def update(
        self,
        weights: np.ndarray,
        props: np.ndarray,
        src_codes: np.ndarray,
        tgt_codes: np.ndarray,
    ) -> None:
        mask = np.isfinite(weights) & np.isfinite(props)
        if not np.any(mask):
            return
        w = weights[mask]
        p = props[mask]
        s = src_codes[mask]
        t = tgt_codes[mask]
        key = s * N_TYPES + t
        self.count += np.bincount(key, minlength=N_PAIRS)
        self.sum_w += np.bincount(key, weights=w, minlength=N_PAIRS)
        self.sum_p += np.bincount(key, weights=p, minlength=N_PAIRS)
        self.sum_w2 += np.bincount(key, weights=w * w, minlength=N_PAIRS)
        self.sum_p2 += np.bincount(key, weights=p * p, minlength=N_PAIRS)
        self.sum_wp += np.bincount(key, weights=w * p, minlength=N_PAIRS)

    def correlation(self, min_count: int) -> np.ndarray:
        corr = np.full_like(self.sum_wp, np.nan)
        valid = self.count >= min_count
        if not np.any(valid):
            return corr
        cnt = self.count[valid]
        sum_w = self.sum_w[valid]
        sum_p = self.sum_p[valid]
        sum_w2 = self.sum_w2[valid]
        sum_p2 = self.sum_p2[valid]
        sum_wp = self.sum_wp[valid]
        denom_w = sum_w2 - (sum_w * sum_w) / cnt
        denom_p = sum_p2 - (sum_p * sum_p) / cnt
        good = (denom_w > 0) & (denom_p > 0)
        if np.any(good):
            cov = sum_wp[good] - (sum_w[good] * sum_p[good]) / cnt[good]
            denom = np.sqrt(denom_w[good] * denom_p[good])
            corr_valid = cov / denom
            corr_valid = np.clip(corr_valid, -1.0, 1.0)
            valid_indices = np.flatnonzero(valid)
            corr[valid_indices[good]] = corr_valid
        return corr


@dataclass
class BinStats:
    bin_edges: np.ndarray

    def __post_init__(self) -> None:
        self.bin_edges = np.asarray(self.bin_edges, dtype=float)
        if self.bin_edges.ndim != 1 or self.bin_edges.size < 2:
            raise ValueError("bin_edges must be 1D with >=2 values")
        if not np.all(np.diff(self.bin_edges) > 0):
            raise ValueError("bin_edges must be strictly increasing")
        self.n_bins = self.bin_edges.size - 1
        shape = (N_PAIRS, self.n_bins)
        self.count = np.zeros(shape, dtype=np.float64)
        self.sum_w = np.zeros(shape, dtype=np.float64)
        self.sum_w2 = np.zeros(shape, dtype=np.float64)

    def update(
        self,
        weights: np.ndarray,
        props: np.ndarray,
        src_codes: np.ndarray,
        tgt_codes: np.ndarray,
    ) -> None:
        mask = np.isfinite(weights) & np.isfinite(props)
        if not np.any(mask):
            return
        w = weights[mask]
        p = props[mask]
        s = src_codes[mask]
        t = tgt_codes[mask]
        pair_idx = (s * N_TYPES + t).astype(np.int64, copy=False)
        bins = np.searchsorted(self.bin_edges, p, side="right") - 1
        bins = np.clip(bins, 0, self.n_bins - 1).astype(np.int64, copy=False)
        flat = pair_idx * self.n_bins + bins
        self.count += np.bincount(flat, minlength=N_PAIRS * self.n_bins).reshape(
            N_PAIRS, self.n_bins
        )
        self.sum_w += np.bincount(
            flat, weights=w, minlength=N_PAIRS * self.n_bins
        ).reshape(N_PAIRS, self.n_bins)
        self.sum_w2 += np.bincount(
            flat, weights=w * w, minlength=N_PAIRS * self.n_bins
        ).reshape(N_PAIRS, self.n_bins)

    def mean(self) -> np.ndarray:
        out = np.full_like(self.sum_w, np.nan)
        np.divide(self.sum_w, self.count, out=out, where=self.count > 0)
        return out

    def sem(self) -> np.ndarray:
        sem = np.full_like(self.sum_w, np.nan)
        valid = self.count > 1
        if np.any(valid):
            sum_w = self.sum_w[valid]
            sum_w2 = self.sum_w2[valid]
            cnt = self.count[valid]
            var = (sum_w2 - (sum_w * sum_w) / cnt) / (cnt - 1)
            var = np.clip(var, a_min=0.0, a_max=None)
            sem_vals = np.sqrt(var / cnt)
            sem[valid] = sem_vals
        return sem

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    @property
    def widths(self) -> np.ndarray:
        return self.bin_edges[1:] - self.bin_edges[:-1]


@dataclass
class PropertySpec:
    name: str
    label: str
    column: str | None = None


PROPERTY_SPECS: Sequence[PropertySpec] = (
    PropertySpec("image_selectivity", "Image selectivity"),
    PropertySpec("orientation_selectivity", "DG orientation selectivity", column="OSI"),
    PropertySpec("firing_rate", "Mean firing rate", column="Ave_Rate(Hz)"),
    PropertySpec("oracle_score", "Oracle correlation"),
    PropertySpec("in_degree", "In-degree"),
    PropertySpec("out_degree", "Out-degree"),
    PropertySpec("natural_image_evoked_rate", "Natural-image evoked rate"),
    PropertySpec("dg_spont_rate", "DG spontaneous rate"),
    PropertySpec("dg_evoked_rate", "DG evoked rate"),
    PropertySpec("dg_mean_rate", "DG mean rate"),
    PropertySpec("dg_peak_rate", "DG peak rate"),
    PropertySpec("dg_dsi", "DG direction selectivity"),
)


def discover_bases(bases: Iterable[str] | None) -> List[str]:
    if bases:
        return list(bases)
    found = []
    for path in sorted(Path(".").glob("core_nll_*")):
        if path.is_dir():
            found.append(str(path))
    if not found:
        raise FileNotFoundError("No core_nll_* directories detected; pass --bases.")
    return found


def parse_network_id(base_dir: str) -> int:
    name = Path(base_dir).name
    digits = "".join(ch for ch in name if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse network id from {base_dir}")
    return int(digits)


class PropertyLoader:
    def __init__(self, selectivity_path: Path):
        if not selectivity_path.exists():
            raise FileNotFoundError(selectivity_path)
        sel = pd.read_csv(selectivity_path)
        sel["node_id"] = sel["node_id"].astype(int)
        sel["network"] = sel["network"].astype(int)
        self.selectivity: Dict[tuple[str, int], pd.Series] = {}
        for (nt, net_id), df in sel.groupby(["network_type", "network"]):
            self.selectivity[(nt, int(net_id))] = df.set_index("node_id")[
                "image_selectivity"
            ].astype(float)

        self.osi_cache: Dict[tuple[str, int], pd.DataFrame] = {}
        self.oracle_cache: Dict[tuple[str, int], pd.Series] = {}
        self.dg_cache: Dict[tuple[str, int], Dict[str, pd.Series]] = {}
        self.ni_cache: Dict[tuple[str, int], pd.Series] = {}
        self.dsi_cache: Dict[tuple[str, int], pd.Series] = {}

    def _metrics_path(
        self, base_dir: str, stem: str, network_type: str, ext: str
    ) -> Path:
        metrics_dir = Path(base_dir) / "metrics"
        candidates = []
        if network_type:
            candidates.append(metrics_dir / f"{stem}_{network_type}{ext}")
        candidates.append(metrics_dir / f"{stem}{ext}")
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Unable to find {stem} for {network_type} under {metrics_dir}"
        )

    def selectivity_series(self, network_type: str, net_id: int) -> pd.Series | None:
        return self.selectivity.get((network_type, net_id))

    def osi_df(self, base_dir: str, network_type: str, net_id: int) -> pd.DataFrame:
        key = (network_type, net_id)
        if key not in self.osi_cache:
            path = self._metrics_path(base_dir, "OSI_DSI_DF", network_type, ".csv")
            df = pd.read_csv(path, sep=" ")
            df["node_id"] = df["node_id"].astype(int)
            self.osi_cache[key] = df.set_index("node_id")
        return self.osi_cache[key]

    def dsi_series(self, base_dir: str, network_type: str, net_id: int) -> pd.Series:
        key = (network_type, net_id)
        if key not in self.dsi_cache:
            osi_df = self.osi_df(base_dir, network_type, net_id)
            if "DSI" not in osi_df.columns:
                raise KeyError("DSI column missing in OSI/DSI table")
            self.dsi_cache[key] = osi_df["DSI"].astype(float)
        return self.dsi_cache[key]

    def oracle_series(self, base_dir: str, network_type: str, net_id: int) -> pd.Series:
        key = (network_type, net_id)
        if key not in self.oracle_cache:
            path = self._metrics_path(
                base_dir, "oracle_correlation_output_imagenet", network_type, ".npy"
            )
            arr = np.load(path)
            self.oracle_cache[key] = pd.Series(
                arr.astype(float), index=np.arange(len(arr))
            )
        return self.oracle_cache[key]

    def natural_image_rate(
        self,
        base_dir: str,
        network_type: str,
        net_id: int,
        node_ids: np.ndarray,
    ) -> pd.Series:
        key = (network_type, net_id)
        if key not in self.ni_cache:
            path = self._metrics_path(
                base_dir, "stim_spikes_output_imagenet", network_type, ".npz"
            )
            with np.load(path) as data:
                counts = data["arr_0"].astype(float)
            img_dur = 0.25  # seconds
            rates = counts.mean(axis=1) / img_dur
            self.ni_cache[key] = pd.Series(rates, index=node_ids)
        return self.ni_cache[key]

    def dg_rates(
        self,
        base_dir: str,
        network_type: str,
        net_id: int,
    ) -> Dict[str, pd.Series]:
        key = (network_type, net_id)
        if key not in self.dg_cache:
            path = self._metrics_path(base_dir, "Rates_DF", network_type, ".csv")
            df = pd.read_csv(path, sep=" ")
            df["node_id"] = df["node_id"].astype(int)
            grouped = df.groupby("node_id")
            mean_evoked = grouped["Ave_Rate(Hz)"].mean()
            peak_evoked = grouped["Ave_Rate(Hz)"].max()
            mean_spont = grouped["Spont_rate(Hz)"].mean()
            evoked = mean_evoked - mean_spont
            self.dg_cache[key] = {
                "dg_spont_rate": mean_spont.astype(float),
                "dg_mean_rate": mean_evoked.astype(float),
                "dg_peak_rate": peak_evoked.astype(float),
                "dg_evoked_rate": evoked.astype(float),
            }
        return self.dg_cache[key]


def coarse_types(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_inh_simplification(df)
    return df


def _edge_appendix(network_type: str) -> str:
    if network_type == "bio_trained":
        return "_bio_trained"
    if network_type == "naive":
        return "_naive"
    return ""


def load_typed_edges(
    base_dir: Path, network_type: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    appendix = _edge_appendix(network_type)
    edge_lf = nu.load_edges_pl(str(base_dir), appendix=appendix)
    node_lf = nu.load_nodes_pl(str(base_dir), core_radius=200)

    cores = node_lf.select("core").collect().to_series()
    src_ids = edge_lf.select("source_id").collect().to_series()
    tgt_ids = edge_lf.select("target_id").collect().to_series()
    both_core = cores[src_ids] & cores[tgt_ids]

    edges = edge_lf.filter(both_core).collect().to_pandas()
    node_df = node_lf.collect().to_pandas().set_index("node_id")
    ctdf = nu.get_cell_type_table()["cell_type"]
    src_pop = node_df.loc[edges["source_id"].to_numpy(), "pop_name"].to_numpy()
    tgt_pop = node_df.loc[edges["target_id"].to_numpy(), "pop_name"].to_numpy()
    edges["source_type"] = ctdf.loc[src_pop].to_numpy()
    edges["target_type"] = ctdf.loc[tgt_pop].to_numpy()
    return edges, node_df


def prepare_edge_table(
    base_dir: str, network_type: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges, node_df = load_typed_edges(Path(base_dir), network_type)
    cols = ["source_id", "target_id", "syn_weight", "source_type", "target_type"]
    df = edges.loc[:, cols]
    df = coarse_types(df)
    df = df[df["source_type"].isin(TYPE_ORDER) & df["target_type"].isin(TYPE_ORDER)]
    df = df.dropna(subset=["syn_weight"])
    return df.reset_index(drop=True), node_df


def compute_degrees(edges: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    in_deg = edges.groupby("target_id").size().astype(float)
    out_deg = edges.groupby("source_id").size().astype(float)
    return in_deg, out_deg


def series_or_none(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    if series.empty:
        return None
    return series.astype(float)


def update_property_stats(
    stats: Dict[str, PairStats],
    prop_name: str,
    values: pd.Series | None,
    orientation: str,
    weights: np.ndarray,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    src_codes: np.ndarray,
    tgt_codes: np.ndarray,
) -> None:
    series = series_or_none(values)
    if series is None:
        return
    if orientation == "outgoing":
        prop = series.reindex(source_ids).to_numpy(dtype=float)
    else:
        prop = series.reindex(target_ids).to_numpy(dtype=float)
    stats[prop_name].update(weights, prop, src_codes, tgt_codes)


def to_matrix(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(N_TYPES, N_TYPES)


def save_matrix_csv(matrix: np.ndarray, out_path: Path) -> None:
    df = pd.DataFrame(matrix, index=TYPE_ORDER, columns=TYPE_ORDER)
    df.to_csv(out_path)


def plot_matrix(
    matrix: np.ndarray,
    title: str,
    out_path: Path,
    *,
    orientation: str,
    cmap: str = "RdBu_r",
) -> None:
    apply_pub_style()
    fig_w = 4.6
    fig_h = 4.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    finite_vals = np.abs(matrix[np.isfinite(matrix)])
    if finite_vals.size:
        vmax = float(finite_vals.max())
    else:
        vmax = 0.0
    if vmax <= 0:
        vmax = 0.1
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(N_TYPES))
    ax.set_yticks(range(N_TYPES))
    ax.set_xticklabels(TYPE_ORDER, rotation=45, ha="right")
    ax.set_yticklabels(TYPE_ORDER)
    ax.set_ylabel("Source cell type")
    ax.set_xlabel("Target cell type")
    if orientation == "outgoing":
        ax.set_ylabel("Source cell type (property)")
    else:
        ax.set_xlabel("Target cell type (property)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Pearson r")
    ax.set_title(title, fontsize=10)
    trim_spines(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def append_property_values(
    container: Dict[str, Dict[str, List[np.ndarray]]],
    prop_name: str,
    values: pd.Series | None,
) -> None:
    series = series_or_none(values)
    if series is None:
        return
    arr = series.to_numpy(dtype=float)
    container[prop_name]["outgoing"].append(arr)
    container[prop_name]["incoming"].append(arr)


def append_degree_values(
    container: Dict[str, Dict[str, List[np.ndarray]]],
    in_deg: pd.Series,
    out_deg: pd.Series,
) -> None:
    if not in_deg.empty:
        arr = in_deg.to_numpy(dtype=float)
        container["in_degree"]["outgoing"].append(arr)
        container["in_degree"]["incoming"].append(arr)
    if not out_deg.empty:
        arr = out_deg.to_numpy(dtype=float)
        container["out_degree"]["outgoing"].append(arr)
        container["out_degree"]["incoming"].append(arr)


def compute_bin_edges(values: List[np.ndarray], n_bins: int) -> np.ndarray | None:
    if not values:
        return None
    data = np.concatenate(values)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return None
    v_min = float(np.min(data))
    v_max = float(np.max(data))
    if not np.isfinite(v_min) or not np.isfinite(v_max):
        return None
    if v_max == v_min:
        span = max(abs(v_min), 1.0)
        return np.array([v_min - span * 0.5, v_max + span * 0.5])
    edges = np.linspace(v_min, v_max, n_bins + 1, dtype=float)
    eps = max((v_max - v_min) * 1e-6, 1e-9)
    if v_max == v_min:
        edges = np.array([v_min - eps, v_max + eps], dtype=float)
    else:
        for i in range(1, edges.size):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + eps
        if edges[-1] <= edges[-2]:
            edges[-1] = edges[-2] + eps
    return edges


def update_bin_stats(
    stats: Dict[str, BinStats | None],
    prop_name: str,
    values: pd.Series | None,
    orientation: str,
    weights: np.ndarray,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    src_codes: np.ndarray,
    tgt_codes: np.ndarray,
) -> None:
    if prop_name not in stats:
        return
    stat_obj = stats[prop_name]
    if stat_obj is None:
        return
    series = series_or_none(values)
    if series is None:
        return
    if orientation == "outgoing":
        prop = series.reindex(source_ids).to_numpy(dtype=float)
    else:
        prop = series.reindex(target_ids).to_numpy(dtype=float)
    stat_obj.update(weights, prop, src_codes, tgt_codes)


def plot_block_matrix(
    spec: PropertySpec,
    orientation: str,
    stats: BinStats | None,
    out_path: Path,
) -> None:
    if stats is None:
        return
    apply_pub_style()
    means = stats.mean()
    sems = stats.sem()
    centers = stats.centers
    widths = stats.widths
    vmax = np.nanmax(np.abs(means))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 0.1
    fig, axes = plt.subplots(
        N_TYPES,
        N_TYPES,
        figsize=(7.2, 7.2),
        sharex=True,
        sharey=True,
    )
    xmin = centers[0] - widths[0] * 0.5
    xmax = centers[-1] + widths[-1] * 0.5
    for i, src in enumerate(TYPE_ORDER):
        for j, tgt in enumerate(TYPE_ORDER):
            ax = axes[i, j]
            idx = i * N_TYPES + j
            mu = means[idx]
            se = sems[idx]
            mask = np.isfinite(mu)
            ax.bar(
                centers[mask],
                mu[mask],
                width=widths[mask] * 0.9,
                color="#6baed6",
                edgecolor="none",
            )
            if np.any(mask):
                valid_sem = se[mask]
                valid_sem = np.where(np.isfinite(valid_sem), valid_sem, 0.0)
                ax.errorbar(
                    centers[mask],
                    mu[mask],
                    yerr=valid_sem,
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.5,
                    capsize=1.5,
                )
            if i == 0:
                ax.set_title(tgt, fontsize=7)
            if j == 0:
                ax.set_ylabel(src, fontsize=7)
            else:
                ax.set_yticklabels([])
            if i == N_TYPES - 1:
                ax.set_xticks(centers)
                ax.set_xticklabels(
                    [f"{c:.2f}" for c in centers], fontsize=5, rotation=45
                )
            else:
                ax.set_xticks([])
            ax.set_ylim(-vmax * 1.1, vmax * 1.1)
            ax.set_xlim(xmin, xmax)
            ax.tick_params(axis="y", labelsize=5)
            trim_spines(ax)
    axes[0, 0].set_ylim(-vmax * 1.1, vmax * 1.1)
    fig.suptitle(f"{spec.label} bins ({orientation})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Survey weight correlations with cell properties."
    )
    parser.add_argument(
        "--bases", nargs="*", help="Base directories (default: detect core_nll_*)"
    )
    parser.add_argument(
        "--network-type",
        default="bio_trained",
        help="Network appendix (default: bio_trained)",
    )
    parser.add_argument(
        "--selectivity",
        type=Path,
        default=Path("image_decoding/summary/sparsity_model_by_unit.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("survey"))
    parser.add_argument(
        "--min-count",
        type=int,
        default=200,
        help="Minimum edges per pair to report correlation",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=6,
        help="Number of property bins for block-average plots",
    )
    args = parser.parse_args(argv)

    bases = discover_bases(args.bases)
    loader = PropertyLoader(args.selectivity)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    distributions: Dict[str, Dict[str, List[np.ndarray]]] = {
        spec.name: {"outgoing": [], "incoming": []} for spec in PROPERTY_SPECS
    }

    # Pass 1: collect property distributions for binning
    for base_dir in bases:
        net_id = parse_network_id(base_dir)
        edges, node_df = prepare_edge_table(base_dir, args.network_type)
        if edges.empty:
            continue
        node_ids = node_df.index.to_numpy()

        selectivity = loader.selectivity_series(args.network_type, net_id)
        append_property_values(distributions, "image_selectivity", selectivity)

        osi_df = loader.osi_df(base_dir, args.network_type, net_id)
        append_property_values(
            distributions,
            "orientation_selectivity",
            osi_df.get("OSI") if osi_df is not None else None,
        )
        append_property_values(
            distributions,
            "firing_rate",
            osi_df.get("Ave_Rate(Hz)") if osi_df is not None else None,
        )

        oracle = loader.oracle_series(base_dir, args.network_type, net_id)
        append_property_values(distributions, "oracle_score", oracle)

        ni_rate = loader.natural_image_rate(
            base_dir, args.network_type, net_id, node_ids
        )
        append_property_values(distributions, "natural_image_evoked_rate", ni_rate)

        dg_metrics = loader.dg_rates(base_dir, args.network_type, net_id)
        append_property_values(
            distributions, "dg_spont_rate", dg_metrics.get("dg_spont_rate")
        )
        append_property_values(
            distributions, "dg_evoked_rate", dg_metrics.get("dg_evoked_rate")
        )
        append_property_values(
            distributions, "dg_mean_rate", dg_metrics.get("dg_mean_rate")
        )
        append_property_values(
            distributions, "dg_peak_rate", dg_metrics.get("dg_peak_rate")
        )

        dsi_series = loader.dsi_series(base_dir, args.network_type, net_id)
        append_property_values(distributions, "dg_dsi", dsi_series)

        in_deg, out_deg = compute_degrees(edges)
        append_degree_values(distributions, in_deg, out_deg)

    bin_edges: Dict[str, Dict[str, np.ndarray | None]] = {}
    for spec in PROPERTY_SPECS:
        name = spec.name
        bin_edges[name] = {}
        for orient in ("outgoing", "incoming"):
            edges_arr = compute_bin_edges(distributions[name][orient], args.n_bins)
            bin_edges[name][orient] = edges_arr

    stats_out = {spec.name: PairStats.zeros() for spec in PROPERTY_SPECS}
    stats_in = {spec.name: PairStats.zeros() for spec in PROPERTY_SPECS}

    bin_stats_out: Dict[str, BinStats | None] = {}
    bin_stats_in: Dict[str, BinStats | None] = {}
    for spec in PROPERTY_SPECS:
        out_edges = bin_edges.get(spec.name, {}).get("outgoing")
        in_edges = bin_edges.get(spec.name, {}).get("incoming")
        bin_stats_out[spec.name] = (
            BinStats(out_edges) if out_edges is not None else None
        )
        bin_stats_in[spec.name] = BinStats(in_edges) if in_edges is not None else None

    for base_dir in bases:
        net_id = parse_network_id(base_dir)
        t0 = time.perf_counter()
        print(f"[info] Processing {base_dir} (network {net_id})")
        edges, node_df = prepare_edge_table(base_dir, args.network_type)
        if edges.empty:
            print(f"[warn] {base_dir}: no edges after filtering; skipping")
            continue
        in_deg, out_deg = compute_degrees(edges)

        weights = edges["syn_weight"].to_numpy(dtype=float)
        source_ids = edges["source_id"].to_numpy(dtype=int)
        target_ids = edges["target_id"].to_numpy(dtype=int)
        src_codes = pd.Categorical(edges["source_type"], categories=TYPE_ORDER).codes
        tgt_codes = pd.Categorical(edges["target_type"], categories=TYPE_ORDER).codes
        valid = (src_codes >= 0) & (tgt_codes >= 0) & np.isfinite(weights)
        weights = np.abs(weights[valid])
        source_ids = source_ids[valid]
        target_ids = target_ids[valid]
        src_codes = src_codes[valid]
        tgt_codes = tgt_codes[valid]

        node_ids = node_df.index.to_numpy()
        selectivity = loader.selectivity_series(args.network_type, net_id)
        osi_df = loader.osi_df(base_dir, args.network_type, net_id)
        oracle = loader.oracle_series(base_dir, args.network_type, net_id)
        natural_rate = loader.natural_image_rate(
            base_dir, args.network_type, net_id, node_ids
        )
        dg_metrics = loader.dg_rates(base_dir, args.network_type, net_id)
        dsi_series = loader.dsi_series(base_dir, args.network_type, net_id)

        prop_map = {
            "image_selectivity": selectivity,
            "orientation_selectivity": (
                osi_df.get("OSI") if osi_df is not None else None
            ),
            "firing_rate": osi_df.get("Ave_Rate(Hz)") if osi_df is not None else None,
            "oracle_score": oracle,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "natural_image_evoked_rate": natural_rate,
            "dg_spont_rate": dg_metrics.get("dg_spont_rate"),
            "dg_evoked_rate": dg_metrics.get("dg_evoked_rate"),
            "dg_mean_rate": dg_metrics.get("dg_mean_rate"),
            "dg_peak_rate": dg_metrics.get("dg_peak_rate"),
            "dg_dsi": dsi_series,
        }

        for name, series in prop_map.items():
            update_property_stats(
                stats_out,
                name,
                series,
                "outgoing",
                weights,
                source_ids,
                target_ids,
                src_codes,
                tgt_codes,
            )
            update_property_stats(
                stats_in,
                name,
                series,
                "incoming",
                weights,
                source_ids,
                target_ids,
                src_codes,
                tgt_codes,
            )
            update_bin_stats(
                bin_stats_out,
                name,
                series,
                "outgoing",
                weights,
                source_ids,
                target_ids,
                src_codes,
                tgt_codes,
            )
            update_bin_stats(
                bin_stats_in,
                name,
                series,
                "incoming",
                weights,
                source_ids,
                target_ids,
                src_codes,
                tgt_codes,
            )

        elapsed = time.perf_counter() - t0
        print(
            f"[info] Completed {base_dir} in {elapsed:.1f}s with {len(weights):,} edges"
        )

    for spec in PROPERTY_SPECS:
        out_mat = to_matrix(stats_out[spec.name].correlation(args.min_count))
        in_mat = to_matrix(stats_in[spec.name].correlation(args.min_count))
        out_counts = to_matrix(stats_out[spec.name].count)
        in_counts = to_matrix(stats_in[spec.name].count)

        out_csv = args.out_dir / f"{spec.name}_outgoing.csv"
        in_csv = args.out_dir / f"{spec.name}_incoming.csv"
        save_matrix_csv(out_mat, out_csv)
        save_matrix_csv(in_mat, in_csv)

        out_counts_csv = args.out_dir / f"{spec.name}_outgoing_counts.csv"
        in_counts_csv = args.out_dir / f"{spec.name}_incoming_counts.csv"
        save_matrix_csv(out_counts, out_counts_csv)
        save_matrix_csv(in_counts, in_counts_csv)

        out_fig = args.out_dir / f"{spec.name}_outgoing.png"
        in_fig = args.out_dir / f"{spec.name}_incoming.png"
        plot_matrix(
            out_mat,
            f"Outgoing weights vs {spec.label}",
            out_fig,
            orientation="outgoing",
        )
        plot_matrix(
            in_mat,
            f"Incoming weights vs {spec.label}",
            in_fig,
            orientation="incoming",
        )

        out_block = args.out_dir / f"{spec.name}_outgoing_blocks.png"
        in_block = args.out_dir / f"{spec.name}_incoming_blocks.png"
        plot_block_matrix(spec, "outgoing", bin_stats_out.get(spec.name), out_block)
        plot_block_matrix(spec, "incoming", bin_stats_in.get(spec.name), in_block)


if __name__ == "__main__":
    main()
