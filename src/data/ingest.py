"""
SentinelOps - Data Ingestion Module
====================================
Responsibilities:
  - Discover and validate Parquet files against params.yaml expected list
  - Load each file using PyArrow for memory efficiency
  - Normalize column names (strip whitespace)
  - Attach source_file column for traceability
  - Inspect and report unique raw labels
  - Combine all files into a single DataFrame
  - Report per-file and combined statistics

Usage:
    from src.data.ingest import load_all_parquet, get_ingestion_report
    df, report = load_all_parquet(params)
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def _load_single_parquet(path: str | Path) -> pd.DataFrame:
    """
    Load a single Parquet file using PyArrow.
    Returns a Pandas DataFrame.
    Normalizes column names: strips leading/trailing whitespace.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    logger.info(f"Loading: {path.name}")
    table = pq.read_table(path)
    df = table.to_pandas()

    # Normalize column names - strip whitespace
    original_cols = list(df.columns)
    df.columns = [col.strip() for col in df.columns]
    stripped = [oc for oc, nc in zip(original_cols, df.columns) if oc != nc]
    if stripped:
        logger.warning(f"  Stripped whitespace from {len(stripped)} column names: {stripped}")

    return df


def _inspect_labels(df: pd.DataFrame, source_name: str) -> dict:
    """
    Inspect the Label column of a single file.
    Returns a dict with label counts and repr for unicode-safety.
    """
    if "Label" not in df.columns:
        raise KeyError(f"'Label' column not found in {source_name}. Columns: {list(df.columns)}")

    label_counts = df["Label"].value_counts().to_dict()
    # Build unicode-safe repr for each label
    label_reprs = {lbl: repr(lbl) for lbl in label_counts}
    return {
        "source": source_name,
        "n_rows": len(df),
        "n_labels": len(label_counts),
        "label_counts": label_counts,
        "label_reprs": label_reprs,
    }


def load_all_parquet(params: dict) -> tuple[pd.DataFrame, dict]:
    """
    Main ingestion function.

    Parameters
    ----------
    params : dict
        Full params.yaml contents (loaded by caller).

    Returns
    -------
    combined_df : pd.DataFrame
        All files combined, with 'source_file' column added.

    report : dict
        Ingestion statistics per file and combined.
    """
    parquet_dir = Path(params["paths"]["parquet_dir"])
    expected_files = params["ingestion"]["expected_files"]

    logger.info("=" * 60)
    logger.info("SENTINELOPS - DATA INGESTION")
    logger.info("=" * 60)
    logger.info(f"Parquet directory: {parquet_dir.resolve()}")

    # -----------------------------------------------------------------
    # 1. Validate expected files exist
    # -----------------------------------------------------------------
    missing = []
    for fname in expected_files:
        fpath = parquet_dir / fname
        if not fpath.exists():
            missing.append(fname)

    if missing:
        raise FileNotFoundError(
            f"Missing expected Parquet files:\n" + "\n".join(f"  - {m}" for m in missing)
        )

    # Check for unexpected extra files
    actual_files = set(f for f in os.listdir(parquet_dir) if f.endswith(".parquet"))
    extra = actual_files - set(expected_files)
    if extra:
        logger.warning(f"Unexpected files in parquet_dir (will be ignored): {sorted(extra)}")

    # -----------------------------------------------------------------
    # 2. Load each file and collect statistics
    # -----------------------------------------------------------------
    file_dfs = []
    file_reports = []
    all_columns = None

    for fname in expected_files:
        fpath = parquet_dir / fname
        df = _load_single_parquet(fpath)

        # Verify column consistency across files
        if all_columns is None:
            all_columns = list(df.columns)
        else:
            if list(df.columns) != all_columns:
                diff = set(df.columns).symmetric_difference(set(all_columns))
                raise ValueError(
                    f"Column mismatch in {fname}. "
                    f"Symmetric difference: {diff}"
                )

        # Attach source_file column for traceability
        df["source_file"] = fname

        # Inspect labels
        label_report = _inspect_labels(df, fname)

        # Count Inf and NaN in numeric columns
        num_df = df.select_dtypes(include=[np.number])
        inf_count = int(np.isinf(num_df.values.astype(float)).sum())
        nan_count = int(num_df.isna().sum().sum())
        label_nan_count = int(df["Label"].isna().sum())

        # Constant numeric columns in this file
        const_cols = num_df.columns[num_df.std() == 0].tolist()

        file_report = {
            "file": fname,
            "rows": len(df),
            "columns": len(df.columns) - 1,  # excluding source_file
            "inf_count": inf_count,
            "nan_count_numeric": nan_count,
            "nan_count_label": label_nan_count,
            "constant_cols": const_cols,
            "labels": label_report,
        }
        file_reports.append(file_report)
        file_dfs.append(df)

        logger.info(
            f"  {fname}: {len(df):,} rows | "
            f"Inf={inf_count} | NaN={nan_count} | "
            f"Labels={label_report['n_labels']}"
        )

    # -----------------------------------------------------------------
    # 3. Combine all DataFrames
    # -----------------------------------------------------------------
    logger.info("Combining all files...")
    combined_df = pd.concat(file_dfs, axis=0, ignore_index=True)

    total_rows = len(combined_df)
    logger.info(f"Combined total: {total_rows:,} rows")

    # -----------------------------------------------------------------
    # 4. Collect combined stats
    # -----------------------------------------------------------------
    all_raw_labels = combined_df["Label"].value_counts().to_dict()
    combined_inf = int(
        np.isinf(
            combined_df.select_dtypes(include=[np.number])
            .values.astype(float)
        ).sum()
    )
    combined_nan = int(
        combined_df.select_dtypes(include=[np.number]).isna().sum().sum()
    )

    # Collect constant columns across the combined dataset
    combined_num = combined_df.select_dtypes(include=[np.number])
    combined_const_cols = combined_num.columns[combined_num.std() == 0].tolist()

    report = {
        "parquet_dir": str(parquet_dir.resolve()),
        "files_loaded": expected_files,
        "total_rows": total_rows,
        "total_columns": len(all_columns),
        "combined_inf": combined_inf,
        "combined_nan": combined_nan,
        "combined_constant_cols": combined_const_cols,
        "combined_label_counts": {repr(k): v for k, v in all_raw_labels.items()},
        "per_file": file_reports,
    }

    return combined_df, report


def print_ingestion_report(report: dict) -> None:
    """Pretty-print the ingestion report to stdout (Unicode-safe)."""
    def safe_str(s):
        """Return a display-safe version of a string (replace U+FFFD)."""
        return s.replace('\ufffd', '<UFFFD>')

    print("\n" + "=" * 65)
    print("INGESTION REPORT")
    print("=" * 65)
    print(f"  Parquet dir : {report['parquet_dir']}")
    print(f"  Files loaded: {len(report['files_loaded'])}")
    print(f"  Total rows  : {report['total_rows']:,}")
    print(f"  Total cols  : {report['total_columns']}")
    print(f"  Inf values  : {report['combined_inf']:,}")
    print(f"  NaN (num)   : {report['combined_nan']:,}")
    print(f"  Const cols  : {report['combined_constant_cols']}")

    print("\n--- Per-File Summary ---")
    for fr in report["per_file"]:
        lbl_info = fr["labels"]
        print(f"\n  [{fr['file']}]")
        print(f"    rows={fr['rows']:,}  inf={fr['inf_count']}  nan={fr['nan_count_numeric']}")
        for raw_lbl, cnt in lbl_info["label_counts"].items():
            pct = cnt / fr["rows"] * 100
            display = safe_str(repr(raw_lbl))
            print(f"    {display:55s}  {cnt:>8,}  ({pct:6.3f}%)")

    print("\n--- Combined Raw Label Distribution ---")
    total = report["total_rows"]
    for repr_lbl, cnt in sorted(
        report["combined_label_counts"].items(), key=lambda x: -x[1]
    ):
        pct = cnt / total * 100
        display = safe_str(repr_lbl)
        print(f"  {display:55s}  {cnt:>8,}  ({pct:6.3f}%)")

    print(f"\n  TOTAL: {total:,}")
    print("=" * 65)
