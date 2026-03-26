#!/usr/bin/env python3
"""Create node sets for high/low outgoing *weight* cohorts (Figure 5).

This mirrors `analysis_shared/create_highlow_outgoing_synapsecount_nodesets.py` but uses the
sum of absolute synaptic weights per source neuron (total outgoing |syn_weight|).

Outputs are written to `<base_dir>/node_sets/` with the legacy naming used throughout the
repo, e.g.:
- high_outgoing_exc_core_nodes.json
- low_outgoing_inh_periphery_nodes.json
- pv_high_outgoing_core_nodes.json

For Exc/Inh we additionally write "all-region" aliases (core + periphery), matching
existing `core_nll_0/node_sets/high_outgoing_exc_nodes.json` style.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analysis_shared.array_utils import safe_bincount_sum

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORE_RADIUS_UM = 200.0

LAYER_MAP = {
    "1": "L1",
    "23": "L2/3",
    "4": "L4",
    "5": "L5",
    "6": "L6",
}


def _extract_layer_label(pop_name: str) -> str | None:
    match = re.search(r"[ei](\d+)", str(pop_name))
    if not match:
        return None
    return LAYER_MAP.get(match.group(1))


def load_node_table(base_dir: Path) -> pd.DataFrame:
    node_types_file = base_dir / "network" / "v1_node_types.csv"
    node_file = base_dir / "network" / "v1_nodes.h5"

    node_types = pd.read_csv(node_types_file, sep=r"\s+")
    node_types["is_inhibitory"] = node_types["ei"].str.lower().str.startswith("i")
    node_types["cell_marker"] = node_types["pop_name"].str.extract(r"[ei]\d+(\w+)")
    node_types["layer_label"] = node_types["pop_name"].apply(_extract_layer_label)

    type_lookup = node_types.set_index("node_type_id")[
        ["pop_name", "is_inhibitory", "cell_marker", "layer_label"]
    ]

    with h5py.File(node_file, "r") as f:
        node_ids = f["nodes"]["v1"]["node_id"][:].astype(np.int64)
        node_type_ids = f["nodes"]["v1"]["node_type_id"][:].astype(np.int64)
        x = f["nodes"]["v1"]["0"]["x"][:].astype(np.float64)
        z = f["nodes"]["v1"]["0"]["z"][:].astype(np.float64)

    df = pd.DataFrame({"node_type_id": node_type_ids, "x": x, "z": z}, index=node_ids)
    df.index.name = "node_id"
    df = df.join(type_lookup, on="node_type_id", how="left")
    df["is_inhibitory"] = df["is_inhibitory"].fillna(False).astype(bool)
    df["core"] = (df["x"] ** 2 + df["z"] ** 2) < CORE_RADIUS_UM**2
    df = df.rename(columns={"layer_label": "layer"})
    return df


def load_outgoing_abs_weight(edge_file: Path) -> pd.Series:
    with h5py.File(edge_file, "r") as f:
        grp = f["edges"]["v1_to_v1"]
        sources = grp["source_node_id"][:].astype(np.int64)
        weights = grp["0"]["syn_weight"][:].astype(np.float64)

    abs_w = np.abs(weights)
    n_nodes = int(sources.max()) + 1 if sources.size else 0
    sums = safe_bincount_sum(sources, abs_w, minlength=n_nodes)
    return pd.Series(sums, name="outgoing_weight_abs")


def _node_set_filename(base: str, weight: str, region: str) -> str:
    if base in {"pv", "sst", "vip"}:
        return f"{base}_{weight}_outgoing_{region}_nodes.json"
    return f"{weight}_outgoing_{base}_{region}_nodes.json"


def _load_pop_to_cell_type() -> dict[str, str]:
    scheme_path = PROJECT_ROOT / "base_props" / "cell_type_naming_scheme.csv"
    header = scheme_path.read_text().splitlines()[0]
    if "," in header:
        scheme = pd.read_csv(scheme_path)
    else:
        scheme = pd.read_csv(scheme_path, sep=r"\s+", engine="python")
    scheme = scheme.dropna(subset=["pop_name", "cell_type"])
    return dict(zip(scheme["pop_name"].astype(str), scheme["cell_type"].astype(str)))


def _select_high_low_by_cell_type(
    subset: pd.DataFrame,
    metric: pd.Series,
) -> tuple[list[int], list[int], list[dict]]:
    if subset.empty:
        return [], [], []

    high_cells: list[int] = []
    low_cells: list[int] = []
    stats: list[dict] = []

    metric = metric.reindex(subset.index).fillna(0.0)

    for ct, ct_df in subset.groupby("cell_type"):
        ct_ids = [int(nid) for nid in ct_df.index]
        ct_vals = metric.loc[ct_ids].to_dict()
        sorted_ids = sorted(ct_ids, key=lambda nid: ct_vals[nid], reverse=True)
        n_cells = len(sorted_ids)
        n_third = n_cells // 3
        if n_third == 0:
            stats.append(
                {
                    "cell_type": ct,
                    "total": n_cells,
                    "high": 0,
                    "low": 0,
                    "high_mean_weight": np.nan,
                    "low_mean_weight": np.nan,
                }
            )
            continue

        ct_high = sorted_ids[:n_third]
        ct_low = sorted_ids[-n_third:]
        high_cells.extend(ct_high)
        low_cells.extend(ct_low)

        high_vals = [ct_vals[c] for c in ct_high]
        low_vals = [ct_vals[c] for c in ct_low]
        stats.append(
            {
                "cell_type": ct,
                "total": n_cells,
                "high": len(ct_high),
                "low": len(ct_low),
                "high_mean_weight": float(np.mean(high_vals)) if high_vals else np.nan,
                "low_mean_weight": float(np.mean(low_vals)) if low_vals else np.nan,
            }
        )

    return high_cells, low_cells, stats


def _select_high_low_by_layer(
    subset: pd.DataFrame,
    metric: pd.Series,
) -> tuple[list[int], list[int], list[dict]]:
    if subset.empty:
        return [], [], []

    high_cells: list[int] = []
    low_cells: list[int] = []
    stats: list[dict] = []

    metric = metric.reindex(subset.index).fillna(0.0)

    for layer, layer_df in subset.groupby("layer"):
        layer_ids = [int(nid) for nid in layer_df.index]
        layer_vals = metric.loc[layer_ids].to_dict()
        sorted_ids = sorted(layer_ids, key=lambda nid: layer_vals[nid], reverse=True)
        n_cells = len(sorted_ids)
        n_third = n_cells // 3
        if n_third == 0:
            stats.append(
                {
                    "layer": layer,
                    "total": n_cells,
                    "high": 0,
                    "low": 0,
                    "high_mean_weight": np.nan,
                    "low_mean_weight": np.nan,
                }
            )
            continue

        layer_high = sorted_ids[:n_third]
        layer_low = sorted_ids[-n_third:]
        high_cells.extend(layer_high)
        low_cells.extend(layer_low)

        high_vals = [layer_vals[c] for c in layer_high]
        low_vals = [layer_vals[c] for c in layer_low]
        stats.append(
            {
                "layer": layer,
                "total": n_cells,
                "high": len(layer_high),
                "low": len(layer_low),
                "high_mean_weight": float(np.mean(high_vals)) if high_vals else np.nan,
                "low_mean_weight": float(np.mean(low_vals)) if low_vals else np.nan,
            }
        )

    return high_cells, low_cells, stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Create high/low outgoing weight node sets for Fig. 5 cohorts."
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path("core_nll_0"),
        help="Network base directory (default: core_nll_0).",
    )
    ap.add_argument(
        "--edge-file",
        type=Path,
        default=None,
        help="Edge file to compute outgoing weights from (default: <base_dir>/network/v1_v1_edges_bio_trained.h5).",
    )
    ap.add_argument(
        "--mode",
        choices=("by_layer", "by_cell_type"),
        default="by_layer",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir
    node_set_dir = base_dir / "node_sets"
    node_set_dir.mkdir(parents=True, exist_ok=True)

    edge_file = args.edge_file or (base_dir / "network" / "v1_v1_edges_bio_trained.h5")
    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}")

    print(f"Loading node metadata from: {base_dir}")
    nodes = load_node_table(base_dir)

    if args.mode == "by_cell_type":
        pop_to_cell_type = _load_pop_to_cell_type()
        nodes = nodes.copy()
        nodes["cell_type"] = nodes["pop_name"].astype(str).map(pop_to_cell_type)
        nodes = nodes.dropna(subset=["cell_type"])

    print(f"Computing outgoing |weight| from: {edge_file}")
    outgoing = load_outgoing_abs_weight(edge_file)

    if args.mode == "by_cell_type":
        results: list[dict] = []

        exc_core = nodes[(~nodes["is_inhibitory"]) & nodes["core"]].copy()
        inh_core = nodes[(nodes["is_inhibitory"]) & nodes["core"]].copy()

        high_exc, low_exc, exc_stats = _select_high_low_by_cell_type(exc_core, outgoing)
        high_inh, low_inh, inh_stats = _select_high_low_by_cell_type(inh_core, outgoing)
        results.extend({"base": "exc", "region": "core", **rec} for rec in exc_stats)
        results.extend({"base": "inh", "region": "core", **rec} for rec in inh_stats)

        files_to_write = {
            "high_outgoing_exc_core_nodes.json": sorted(set(high_exc)),
            "low_outgoing_exc_core_nodes.json": sorted(set(low_exc)),
            "high_outgoing_inh_core_nodes.json": sorted(set(high_inh)),
            "low_outgoing_inh_core_nodes.json": sorted(set(low_inh)),
        }
        for fname, node_ids in files_to_write.items():
            out_path = node_set_dir / fname
            with open(out_path, "w") as f:
                json.dump({"population": "v1", "node_id": node_ids}, f, indent=2)
            print(f"  → Wrote {out_path} ({len(node_ids)} cells)")

        if results:
            stats_df = pd.DataFrame(results)
            out_stats = node_set_dir / "outgoing_weight_nodeset_stats_by_celltype.csv"
            stats_df.to_csv(out_stats, index=False)
            print(f"\nSaved selection summary: {out_stats}")
        return

    base_defs = {
        "exc": nodes[~nodes["is_inhibitory"]],
        "inh": nodes[nodes["is_inhibitory"]],
        "pv": nodes[nodes["cell_marker"] == "Pvalb"],
        "sst": nodes[nodes["cell_marker"] == "Sst"],
        "vip": nodes[nodes["cell_marker"] == "Vip"],
    }

    results: list[dict] = []

    for base, base_df in base_defs.items():
        base_df = base_df[base_df["layer"].notna()].copy()
        if base_df.empty:
            continue

        selections: dict[str, dict[str, list[int]]] = {"core": {}, "periphery": {}}

        for region, mask in (("core", base_df["core"]), ("periphery", ~base_df["core"])):
            subset = base_df[mask].copy()
            if subset.empty:
                continue

            print(
                f"\nSelecting {base} {region}: n={len(subset)} (per-layer top/bottom third)"
            )
            high_ids, low_ids, stats = _select_high_low_by_layer(subset, outgoing)
            for rec in stats:
                results.append({"base": base, "region": region, **rec})

            selections[region]["high"] = high_ids
            selections[region]["low"] = low_ids

            files_to_write = {
                _node_set_filename(base, "high", region): sorted(set(high_ids)),
                _node_set_filename(base, "low", region): sorted(set(low_ids)),
            }

            for fname, node_ids in files_to_write.items():
                out_path = node_set_dir / fname
                with open(out_path, "w") as f:
                    json.dump({"population": "v1", "node_id": node_ids}, f, indent=2)
                print(f"  → Wrote {out_path} ({len(node_ids)} cells)")

            if base == "exc":
                dup_high = node_set_dir / f"high_outgoing_exc_{region}.json"
                dup_low = node_set_dir / f"low_outgoing_exc_{region}.json"
            elif base == "inh":
                dup_high = node_set_dir / f"high_outgoing_inh_{region}.json"
                dup_low = node_set_dir / f"low_outgoing_inh_{region}.json"
            else:
                dup_high = None
                dup_low = None

            if dup_high is not None and dup_low is not None:
                with open(dup_high, "w") as f:
                    json.dump({"population": "v1", "node_id": sorted(set(high_ids))}, f, indent=2)
                with open(dup_low, "w") as f:
                    json.dump({"population": "v1", "node_id": sorted(set(low_ids))}, f, indent=2)

        if base in {"exc", "inh"}:
            high_all = sorted(
                set(selections["core"].get("high", []))
                | set(selections["periphery"].get("high", []))
            )
            low_all = sorted(
                set(selections["core"].get("low", []))
                | set(selections["periphery"].get("low", []))
            )
            if base == "exc":
                alias_high = node_set_dir / "high_outgoing_exc_nodes.json"
                alias_low = node_set_dir / "low_outgoing_exc_nodes.json"
            else:
                alias_high = node_set_dir / "high_outgoing_inh_nodes.json"
                alias_low = node_set_dir / "low_outgoing_inh_nodes.json"

            with open(alias_high, "w") as f:
                json.dump({"population": "v1", "node_id": high_all}, f, indent=2)
            with open(alias_low, "w") as f:
                json.dump({"population": "v1", "node_id": low_all}, f, indent=2)
            print(f"  → Wrote {alias_high.name} / {alias_low.name} (union core+periphery)")

    if results:
        stats_df = pd.DataFrame(results)
        out_stats = node_set_dir / "outgoing_weight_nodeset_stats.csv"
        stats_df.to_csv(out_stats, index=False)
        print(f"\nSaved selection summary: {out_stats}")


if __name__ == "__main__":
    main()
