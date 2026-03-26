from __future__ import annotations
import pandas as pd

L5_EXC = ["L5_IT", "L5_ET", "L5_NP"]
AGG_L5 = "L5_Exc"

INH_SIMPLE_MAP = {
    "PV": {"L2/3_PV", "L4_PV", "L5_PV", "L6_PV"},
    "SST": {"L2/3_SST", "L4_SST", "L5_SST", "L6_SST"},
    "VIP": {"L2/3_VIP", "L4_VIP", "L5_VIP", "L6_VIP"},
}


def aggregate_l5(df: pd.DataFrame) -> pd.DataFrame:
    if "source_type" in df.columns:
        df["source_type"] = df["source_type"].replace(L5_EXC, AGG_L5)
    if "target_type" in df.columns:
        df["target_type"] = df["target_type"].replace(L5_EXC, AGG_L5)
    return df


def simplify_inh(cell_type: str) -> str:
    for simple, fullset in INH_SIMPLE_MAP.items():
        if cell_type in fullset:
            return simple
    return cell_type


def apply_inh_simplification(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "source_type" in df.columns:
        df["source_type"] = df["source_type"].map(simplify_inh)
    if "target_type" in df.columns:
        df["target_type"] = df["target_type"].map(simplify_inh)
    return df


def _extract_layer_series(types: pd.Series) -> pd.Series:
    """
    Extract a coarse cortical layer label from cell-type strings.

    Expected inputs include: 'L2/3_Exc', 'L4_PV', 'L5_IT', 'L5_Exc', 'L6_SST', 'L1_Inh'.
    Returns a Series of layer labels among {'L1','L2/3','L4','L5','L6'} (or None when unknown).
    """
    s = types.astype("string").fillna("")
    out = pd.Series([None] * len(s), index=s.index, dtype="object")

    m = s.str.startswith("L1_")
    out[m] = "L1"
    m = s.str.startswith("L2/3_") | (s == "L2/3_Exc")
    out[m] = "L2/3"
    m = s.str.startswith("L4_") | (s == "L4_Exc")
    out[m] = "L4"
    m = s.str.startswith("L6_") | (s == "L6_Exc")
    out[m] = "L6"
    m = s.str.startswith("L5_") | s.isin(["L5_Exc", "L5_IT", "L5_ET", "L5_NP"])
    out[m] = "L5"
    return out


def filter_inh_respective_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    For Exc↔Inh pairs, keep only inhibitory populations from the corresponding Exc layer.

    Example: if you later simplify inhibitory layers into PV/SST/VIP, then without this
    filter the aggregated label 'PV→L4_Exc' would include L2/3_PV→L4_Exc, L5_PV→L4_Exc, etc.
    With this filter enabled, it keeps only L4_PV→L4_Exc (and similarly for the reverse).

    Notes:
    - L1_Inh is allowed to pair with all layers (it is not layer-specific in the same way).
    - Inh↔Inh and Exc↔Exc pairs are kept unchanged.
    - This should be applied BEFORE `apply_inh_simplification()` (while inhibitory labels
      are still layer-qualified, e.g. 'L4_PV').
    """
    if df.empty:
        return df
    if ("source_type" not in df.columns) or ("target_type" not in df.columns):
        return df

    inh_layered = {ct for fullset in INH_SIMPLE_MAP.values() for ct in fullset} | {
        "L1_Inh"
    }
    src = df["source_type"]
    tgt = df["target_type"]
    src_inh = src.isin(inh_layered)
    tgt_inh = tgt.isin(inh_layered)

    # Only filter when exactly one side is inhibitory (Exc↔Inh)
    ei_pair = src_inh ^ tgt_inh
    if not bool(ei_pair.any()):
        return df

    src_layer = _extract_layer_series(src)
    tgt_layer = _extract_layer_series(tgt)

    keep = pd.Series(True, index=df.index)

    # Inh(source) → non-Inh(target): require same layer (except L1_Inh)
    m = ei_pair & src_inh & (~tgt_inh)
    if bool(m.any()):
        keep[m] = (
            (src_layer[m] == "L1")
            | src_layer[m].isna()
            | tgt_layer[m].isna()
            | (src_layer[m] == tgt_layer[m])
        )

    # non-Inh(source) → Inh(target): require same layer (except L1_Inh)
    m = ei_pair & (~src_inh) & tgt_inh
    if bool(m.any()):
        keep[m] = (
            (tgt_layer[m] == "L1")
            | src_layer[m].isna()
            | tgt_layer[m].isna()
            | (src_layer[m] == tgt_layer[m])
        )

    return df[keep]
