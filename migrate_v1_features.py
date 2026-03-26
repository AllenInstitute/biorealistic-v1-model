#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
from typing import Iterable, Tuple

import pandas as pd


JAN2024_ROOT = \
    "/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/tensorflow_bf4/V1_GLIF_model_Jan2024"
SUBDIR_MAP_CSV = os.path.join(JAN2024_ROOT, "weight_change_analysis", "subdirs_map.csv")
THIS_ROOT = \
    "/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/glif_builder_test/biorealistic-v1-model"


def iter_candidates(row: pd.Series) -> Iterable[str]:
    """Yield possible directories to search for v1_features_df.csv for a given mapping row."""
    data_dir = os.path.join(JAN2024_ROOT, str(row["data_dir"]))
    result_dir = os.path.join(JAN2024_ROOT, str(row["result_dir"]))
    sub_dir = str(row.get("sub_dir", "")).strip()
    # Search in result_dir first (most likely), then in data_dir as fallback
    # Common patterns observed: result_dir/<sub_dir>/analysis, result_dir/<sub_dir>, result_dir
    if sub_dir:
        yield os.path.join(result_dir, sub_dir, "analysis")
        yield os.path.join(result_dir, sub_dir)
    yield result_dir
    # Also search under data_dir variants
    if sub_dir:
        yield os.path.join(data_dir, sub_dir, "analysis")
        yield os.path.join(data_dir, sub_dir)
    yield data_dir


def find_features_csv(search_root: str) -> str | None:
    """Find the first v1_features_df.csv under search_root recursively."""
    target_name = "v1_features_df.csv"
    if not os.path.isdir(search_root):
        return None
    for root, _dirs, files in os.walk(search_root):
        if target_name in files:
            return os.path.join(root, target_name)
    return None


def map_label(condition: str, label: str) -> str | None:
    """Normalize to {bio_trained, naive}. Ignore other labels (bio_free, uni_reg)."""
    label = label.strip()
    if label in {"bio_trained", "naive"}:
        return label
    # Fallback mapping following run_weight_change_analysis.sh logic:
    # if condition contains uniform_weights_True -> naive; else -> bio_trained
    if "uniform_weights_True" in condition:
        return "naive"
    if condition == "v1_0":
        return "bio_trained"
    return None


def migrate_one(index: int, label: str, rows: pd.DataFrame, dry_run: bool) -> Tuple[bool, str]:
    """Migrate for a specific dataset index and label. Returns (done, message)."""
    dest_dir = os.path.join(THIS_ROOT, f"core_nll_{index}")
    if not os.path.isdir(dest_dir):
        return False, f"Destination not found: {dest_dir}"
    out_name = f"v1_features_df_{label}.csv"
    out_path = os.path.join(dest_dir, out_name)

    # Prefer rows with explicit label match, then any mapped to that label
    candidate_rows = []
    for _idx, r in rows.iterrows():
        mapped = map_label(str(r["condition"]), str(r["label"]))
        if mapped == label:
            candidate_rows.append(r)
    # Search candidate locations in order and copy the first found
    for r in candidate_rows:
        for cand_dir in iter_candidates(r):
            src = find_features_csv(cand_dir)
            if src:
                if dry_run:
                    return True, f"DRY-RUN: copy {src} -> {out_path}"
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(src, out_path)
                return True, f"Copied {src} -> {out_path}"
    return False, f"No features found for index={index}, label={label}"


def main():
    parser = argparse.ArgumentParser(description="Migrate v1_features_df.csv into current project networks.")
    parser.add_argument("--indices", nargs="*", type=int, default=None, help="Indices to migrate (e.g., 0 1 2). Default: all present in map.")
    parser.add_argument("--labels", nargs="*", default=["bio_trained", "naive"], help="Labels to migrate: bio_trained naive")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying.")
    args = parser.parse_args()

    if not os.path.isfile(SUBDIR_MAP_CSV):
        print(f"Mapping CSV not found: {SUBDIR_MAP_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SUBDIR_MAP_CSV)
    required_cols = {"index", "data_dir", "result_dir", "condition", "sub_dir", "label"}
    if not required_cols.issubset(df.columns):
        print(f"Mapping CSV missing required columns: {required_cols - set(df.columns)}", file=sys.stderr)
        sys.exit(1)

    indices = sorted(df["index"].dropna().unique().astype(int)) if args.indices is None else args.indices

    any_success = False
    for idx in indices:
        rows_idx = df[df["index"] == idx]
        for lab in args.labels:
            ok, msg = migrate_one(idx, lab, rows_idx, args.dry_run)
            print(msg)
            any_success = any_success or ok

    if not any_success:
        sys.exit(2)


if __name__ == "__main__":
    main()

