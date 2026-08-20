"""
SentinelOps - Data Cleaning Module
=====================================
Responsibilities:
  - Normalize raw label strings (strip whitespace, fix U+FFFD → " - ")
  - Map raw labels to canonical 8-class taxonomy
  - FAIL LOUDLY if any unknown label is encountered (never silently discard)
  - Replace Inf / -Inf with NaN
  - Fill NaN using the configured strategy (fit on training split only)
  - Detect and optionally drop constant columns
  - Detect and optionally drop duplicate rows
  - Report comprehensive data-quality statistics
  - Produce a clean DataFrame ready for preprocessing

Key design constraint:
  The NaN fill statistics (e.g., medians) are computed on the TRAINING set only
  and applied to test/dev sets. This module provides the fitting step;
  see preprocess.py for the application step.

Usage:
    from src.data.clean import clean_dataframe, build_label_normalizer
    cleaned_df, stats = clean_dataframe(raw_df, params)
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label Normalization
# ---------------------------------------------------------------------------

# U+FFFD is the Unicode replacement character present in WebAttack labels
_UFFFD_PATTERN = re.compile(r"\ufffd")


def normalize_raw_label(raw: str) -> str:
    """
    Normalize a raw label string to a canonical, lookup-safe form.

    Steps:
      1. Strip leading/trailing whitespace.
      2. Replace U+FFFD (Unicode replacement character) with " - ".
         This fixes: "Web Attack \ufffd XSS" → "Web Attack - XSS"
      3. Collapse any internal multiple spaces to single space.
    """
    if not isinstance(raw, str):
        raise TypeError(f"Expected str label, got {type(raw)}: {raw!r}")

    normalized = raw.strip()
    normalized = _UFFFD_PATTERN.sub(" - ", normalized)
    # Collapse multiple internal spaces
    normalized = re.sub(r" {2,}", " ", normalized)
    return normalized


def build_label_normalizer(label_map: dict[str, str]) -> dict[str, str]:
    """
    Pre-normalize all keys in label_map so lookups also go through
    the same normalization pipeline.

    Returns a new dict with normalized keys → canonical class.
    """
    normalized_map = {}
    for raw_key, canonical in label_map.items():
        norm_key = normalize_raw_label(raw_key)
        if norm_key in normalized_map and normalized_map[norm_key] != canonical:
            raise ValueError(
                f"Label map collision after normalization: "
                f"'{raw_key}' → '{norm_key}' maps to both "
                f"'{normalized_map[norm_key]}' and '{canonical}'"
            )
        normalized_map[norm_key] = canonical
    return normalized_map


def map_raw_labels(
    series: pd.Series,
    normalized_label_map: dict[str, str],
) -> pd.Series:
    """
    Map a raw Label Series to canonical class names.

    For every raw label:
      1. Apply normalize_raw_label.
      2. Look up in normalized_label_map.
      3. If not found, raise ValueError with the full list of unknowns.

    Never silently drops or assigns a wrong class.
    """
    # Normalize all raw values
    normalized_series = series.map(normalize_raw_label)

    # Find unknowns before mapping
    unique_normalized = set(normalized_series.unique())
    expected_keys = set(normalized_label_map.keys())
    unknowns = unique_normalized - expected_keys

    if unknowns:
        raise ValueError(
            f"UNKNOWN LABELS FOUND - pipeline cannot proceed safely.\n"
            f"Unknown labels (repr): {sorted(repr(u) for u in unknowns)}\n"
            f"Expected labels: {sorted(repr(e) for e in expected_keys)}\n"
            f"Action required: Add these labels to params.yaml cleaning.label_map."
        )

    # Map to canonical classes
    mapped = normalized_series.map(normalized_label_map)

    # Verify no NaN slipped through (shouldn't happen if unknowns check passed)
    null_count = mapped.isna().sum()
    if null_count > 0:
        raise RuntimeError(
            f"BUG: {null_count} labels became NaN after mapping. "
            f"This should not happen — check build_label_normalizer."
        )

    return mapped


# ---------------------------------------------------------------------------
# Inf / NaN Handling
# ---------------------------------------------------------------------------

def replace_inf(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Replace Inf and -Inf with NaN in numeric columns.

    Returns
    -------
    df : modified DataFrame (copy)
    stats : dict with inf replacement statistics
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    inf_stats = {}
    total_inf = 0

    for col in num_cols:
        col_data = df[col].astype(float)
        pos_inf = int((col_data == np.inf).sum())
        neg_inf = int((col_data == -np.inf).sum())
        total = pos_inf + neg_inf
        if total > 0:
            df[col] = col_data.replace([np.inf, -np.inf], np.nan)
            inf_stats[col] = {"pos_inf": pos_inf, "neg_inf": neg_inf}
            total_inf += total
            logger.debug(f"  Replaced Inf in '{col}': +Inf={pos_inf}, -Inf={neg_inf}")

    if total_inf > 0:
        logger.info(f"Replaced {total_inf:,} Inf values across {len(inf_stats)} columns.")
    else:
        logger.info("No Inf values found.")

    return df, {"total_inf_replaced": total_inf, "columns": inf_stats}


def compute_fill_values(df: pd.DataFrame, strategy: str) -> dict[str, float]:
    """
    Compute per-column fill values from a DataFrame (should be training data only).

    Parameters
    ----------
    df : DataFrame (training data)
    strategy : "median" | "mean" | "zero"

    Returns
    -------
    fill_values : dict mapping column name → scalar fill value
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fill_values = {}

    if strategy == "median":
        for col in num_cols:
            fill_values[col] = float(df[col].median())
    elif strategy == "mean":
        for col in num_cols:
            fill_values[col] = float(df[col].mean())
    elif strategy == "zero":
        for col in num_cols:
            fill_values[col] = 0.0
    else:
        raise ValueError(f"Unknown nan_fill_strategy: '{strategy}'. Use 'median', 'mean', or 'zero'.")

    return fill_values


def apply_fill_values(df: pd.DataFrame, fill_values: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    """
    Fill NaN values using pre-computed fill_values (from training set).
    Only fills columns present in both df and fill_values.

    Returns
    -------
    df : filled DataFrame (copy)
    stats : fill statistics
    """
    df = df.copy()
    filled_cols = {}

    for col, val in fill_values.items():
        if col not in df.columns:
            continue
        n_nan = int(df[col].isna().sum())
        if n_nan > 0:
            df[col] = df[col].fillna(val)
            filled_cols[col] = {"n_filled": n_nan, "fill_value": val}
            logger.debug(f"  Filled {n_nan} NaN in '{col}' with {val:.6g}")

    total_filled = sum(v["n_filled"] for v in filled_cols.values())
    if total_filled > 0:
        logger.info(f"Filled {total_filled:,} NaN values across {len(filled_cols)} columns.")
    else:
        logger.info("No NaN values needed filling.")

    return df, {"total_filled": total_filled, "columns": filled_cols}


# ---------------------------------------------------------------------------
# Structural Cleaning
# ---------------------------------------------------------------------------

def detect_constant_columns(df: pd.DataFrame) -> list[str]:
    """
    Return list of numeric columns where std == 0 (i.e., all values identical).
    """
    num_df = df.select_dtypes(include=[np.number])
    const_cols = num_df.columns[num_df.std() == 0].tolist()
    return const_cols


def drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Detect and drop all-constant numeric columns.
    Returns modified df and list of dropped column names.
    """
    const_cols = detect_constant_columns(df)
    if const_cols:
        df = df.drop(columns=const_cols)
        logger.info(f"Dropped {len(const_cols)} constant columns: {const_cols}")
    else:
        logger.info("No constant columns found.")
    return df, const_cols


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop exact duplicate rows (all columns identical).
    Returns cleaned df and number of dropped rows.
    """
    n_before = len(df)
    # Don't count source_file as part of duplicate detection
    cols_for_dup = [c for c in df.columns if c != "source_file"]
    df = df.drop_duplicates(subset=cols_for_dup).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped:,} duplicate rows.")
    else:
        logger.info("No duplicate rows found.")
    return df, n_dropped


# ---------------------------------------------------------------------------
# Main Cleaning Pipeline
# ---------------------------------------------------------------------------

def clean_dataframe(
    raw_df: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Full cleaning pipeline for a raw ingested DataFrame.

    Steps (in order):
      1. Validate Label column exists
      2. Map raw labels → canonical classes (FAIL on unknown)
      3. Replace Inf → NaN (if configured)
      4. Drop duplicates (if configured)
      5. Detect (and optionally drop) constant columns
      6. Report data quality statistics
      NOTE: NaN filling is done in preprocess.py after train/test split
            to avoid data leakage.

    Parameters
    ----------
    raw_df : combined raw DataFrame from ingest.py
    params : full params.yaml dict

    Returns
    -------
    clean_df : cleaned DataFrame with 'label_raw', 'label_class' columns
    cleaning_stats : dict of statistics for reporting
    """
    cleaning_cfg = params["cleaning"]
    df = raw_df.copy()

    logger.info("=" * 60)
    logger.info("SENTINELOPS - DATA CLEANING")
    logger.info("=" * 60)
    logger.info(f"Input shape: {df.shape}")

    stats = {
        "input_rows": len(df),
        "input_cols": len(df.columns),
    }

    # ----------------------------------------------------------------
    # Step 1: Validate Label column
    # ----------------------------------------------------------------
    if "Label" not in df.columns:
        raise KeyError("'Label' column missing from DataFrame.")

    # ----------------------------------------------------------------
    # Step 2: Map raw labels → canonical 8 classes
    # ----------------------------------------------------------------
    logger.info("Mapping raw labels to canonical 8-class taxonomy...")
    label_map = cleaning_cfg["label_map"]
    normalized_label_map = build_label_normalizer(label_map)

    # Preserve raw label for debugging/audit
    df["label_raw"] = df["Label"].astype(str)

    # Map to canonical class name
    df["label_class"] = map_raw_labels(df["Label"], normalized_label_map)

    # Drop the original Label column (we keep label_raw and label_class)
    df = df.drop(columns=["Label"])

    class_distribution = df["label_class"].value_counts().to_dict()
    stats["class_distribution"] = class_distribution
    logger.info("Canonical class distribution:")
    total = len(df)
    for cls, cnt in sorted(class_distribution.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        logger.info(f"  {cls:15s}  {cnt:>8,}  ({pct:6.3f}%)")

    # ----------------------------------------------------------------
    # Step 3: Replace Inf → NaN
    # ----------------------------------------------------------------
    if cleaning_cfg.get("replace_inf_with_nan", True):
        df, inf_stats = replace_inf(df)
        stats["inf_replacement"] = inf_stats
    else:
        stats["inf_replacement"] = {"total_inf_replaced": 0, "columns": {}}

    # Count NaN after Inf replacement (before filling)
    num_cols = df.select_dtypes(include=[np.number]).columns
    nan_after_inf = int(df[num_cols].isna().sum().sum())
    stats["nan_after_inf_replacement"] = nan_after_inf
    logger.info(f"NaN count after Inf replacement: {nan_after_inf:,}")

    # ----------------------------------------------------------------
    # Step 4: Drop duplicates
    # ----------------------------------------------------------------
    if cleaning_cfg.get("drop_duplicates", True):
        df, n_dropped_dups = drop_duplicate_rows(df)
        stats["duplicates_dropped"] = n_dropped_dups
    else:
        stats["duplicates_dropped"] = 0

    # ----------------------------------------------------------------
    # Step 5: Detect / drop constant columns
    # ----------------------------------------------------------------
    if cleaning_cfg.get("drop_constant_columns", True):
        df, dropped_const = drop_constant_columns(df)
        stats["constant_cols_dropped"] = dropped_const
    else:
        dropped_const = detect_constant_columns(df)
        stats["constant_cols_detected"] = dropped_const
        logger.info(f"Constant columns detected (not dropped): {dropped_const}")

    # ----------------------------------------------------------------
    # Final statistics
    # ----------------------------------------------------------------
    stats["output_rows"] = len(df)
    stats["output_cols"] = len(df.columns)
    stats["rows_removed"] = stats["input_rows"] - stats["output_rows"]

    logger.info(f"Cleaning complete: {stats['input_rows']:,} -> {stats['output_rows']:,} rows")
    logger.info(f"Columns: {stats['input_cols']} -> {stats['output_cols']}")

    return df, stats


def print_cleaning_report(stats: dict) -> None:
    """Pretty-print cleaning statistics."""
    print("\n" + "=" * 65)
    print("CLEANING REPORT")
    print("=" * 65)
    print(f"  Input rows  : {stats['input_rows']:,}")
    print(f"  Output rows : {stats['output_rows']:,}")
    print(f"  Rows removed: {stats['rows_removed']:,}")
    print(f"  Output cols : {stats['output_cols']}")

    inf_rep = stats.get("inf_replacement", {})
    print(f"  Inf replaced: {inf_rep.get('total_inf_replaced', 0):,}")
    print(f"  NaN (post-Inf): {stats.get('nan_after_inf_replacement', 0):,}")
    print(f"  Duplicates  : {stats.get('duplicates_dropped', 0):,}")

    const_dropped = stats.get("constant_cols_dropped", [])
    print(f"  Const cols dropped ({len(const_dropped)}): {const_dropped}")

    print("\n--- Canonical Class Distribution ---")
    dist = stats.get("class_distribution", {})
    total = sum(dist.values())
    for cls, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100 if total > 0 else 0
        print(f"  {cls:15s}  {cnt:>8,}  ({pct:6.3f}%)")
    print(f"  {'TOTAL':15s}  {total:>8,}")
    print("=" * 65)
