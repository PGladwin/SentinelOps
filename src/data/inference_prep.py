"""
SentinelOps - Inference Preprocessing Module
=============================================
Bridges RAW network-flow input (as produced by CICFlowMeter / shipped in the
CIC-IDS2017 CSV release) to the exact scaled feature vector the promoted
Champion expects, whatever feature set governance approved.

Responsibilities:
  - Normalize incoming column names (strip whitespace, resolve aliases, dedupe)
  - Validate that every required Champion feature is present
  - Coerce to numeric, replace Inf with NaN, fill NaN from TRAINING medians
  - Reorder columns into the exact Champion feature order
  - Apply RobustScaler centering/scaling using training-derived statistics

Why this module exists:
  The Phase 1 scaler is fitted on the full post-correlation feature set while
  the Champion may consume a subset. RobustScaler is column-independent (each
  column uses its own center_/scale_), so a valid subset scaler is obtained by
  slicing the fitted scaler at the Champion's column indices. No retraining
  is required.

Usage:
    from src.data.inference_prep import load_inference_bundle, prepare_for_champion
    bundle = load_inference_bundle("models/champion_preprocessor.json")
    X = prepare_for_champion(raw_df, bundle)
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column Alias Resolution
# ---------------------------------------------------------------------------

# The canonical CIC-IDS2017 CSV release uses different header names than the
# Parquet release the model was trained on. Keys are post-strip CSV headers;
# values are the canonical (Parquet / feature_names.json) names.
COLUMN_ALIASES: dict[str, str] = {
    # Renamed length aggregates
    "Total Length of Fwd Packets": "Fwd Packets Length Total",
    "Total Length of Bwd Packets": "Bwd Packets Length Total",
    # Renamed packet-length extremes (word order reversed)
    "Min Packet Length": "Packet Length Min",
    "Max Packet Length": "Packet Length Max",
    # Renamed average
    "Average Packet Size": "Avg Packet Size",
    # snake_case originals
    "Init_Win_bytes_forward": "Init Fwd Win Bytes",
    "Init_Win_bytes_backward": "Init Bwd Win Bytes",
    "act_data_pkt_fwd": "Fwd Act Data Packets",
    "min_seg_size_forward": "Fwd Seg Size Min",
    # Some redistributions use these spellings
    "Fwd Header Length.1": "Fwd Header Length",
    "CWE Flag Count": "CWE Flag Count",
    "Flow ID": "Flow ID",
}

# Columns that may appear in raw exports but are never model features.
NON_FEATURE_COLUMNS: set[str] = {
    "Flow ID",
    "Source IP",
    "Src IP",
    "Source Port",
    "Src Port",
    "Destination IP",
    "Dst IP",
    "Destination Port",
    "Dst Port",
    "Timestamp",
    "SimillarHTTP",
    "Unnamed: 0",
    "source_file",
    "label_raw",
    "label_class",
    "label_int",
}

# Recognised spellings of the ground-truth column, if the upload happens to
# carry one. Used only for optional accuracy reporting, never for inference.
LABEL_COLUMN_CANDIDATES: tuple[str, ...] = ("Label", "label", "LABEL", "label_class")


def canonical_key(name: str) -> str:
    """
    Reduce a column header to a separator- and case-insensitive lookup key.

    Every exporter spells the same feature differently -- "Flow Duration",
    "flow_duration", "FLOW-DURATION", "Flow.Duration" -- and exact matching
    recognises exactly one of them. Stripping to alphanumerics collapses all of
    them onto one key.

    Word ORDER is deliberately not normalised: "Packet Length Min" and "Min
    Packet Length" are genuinely different strings and are related through
    COLUMN_ALIASES, where the mapping is stated explicitly rather than guessed.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def build_column_resolver(target_features: Optional[list[str]] = None) -> dict[str, str]:
    """
    Build a lookup from any reasonable spelling of a header to its canonical name.

    Seeded from three sources, in increasing priority:
      1. The alias table (raw CSV-release spellings -> canonical names).
      2. The canonical names themselves, so they resolve to themselves.
      3. The target feature list, which wins outright -- if the promoted
         Champion consumes a name, an alias must never redirect away from it.
    """
    resolver: dict[str, str] = {}

    for raw, canonical in COLUMN_ALIASES.items():
        resolver[canonical_key(raw)] = canonical
    for canonical in set(COLUMN_ALIASES.values()):
        resolver.setdefault(canonical_key(canonical), canonical)

    for feature in target_features or []:
        resolver[canonical_key(feature)] = feature

    return resolver


def normalize_columns(
    df: pd.DataFrame,
    target_features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Normalize raw upload column names to the canonical training schema.

    Steps:
      1. Strip leading/trailing whitespace from every column name.
      2. Resolve known aliases via COLUMN_ALIASES.
      3. Resolve remaining headers by canonical key, so case and separator
         variants ("flow_duration", "FLOW-DURATION") match the trained schema.
      4. Drop duplicate column names, keeping the first occurrence.
         (The canonical CIC-IDS2017 CSVs contain 'Fwd Header Length' twice.)

    Parameters
    ----------
    df : pd.DataFrame
        Raw uploaded frame.
    target_features : list[str], optional
        The Champion's feature names. Supplying them lets step 3 match against
        the exact schema being served rather than the alias table alone.

    Returns
    -------
    pd.DataFrame
        Copy with canonical column names and no duplicate columns.
    """
    df = df.copy()

    resolver = build_column_resolver(target_features)
    stripped = [str(c).strip() for c in df.columns]

    resolved = []
    for column in stripped:
        if column in COLUMN_ALIASES:
            resolved.append(COLUMN_ALIASES[column])
        elif target_features and column in target_features:
            resolved.append(column)
        else:
            # Unrecognised headers pass through unchanged rather than being
            # forced onto a near match: a wrong mapping would feed one feature's
            # values into another's slot, which is worse than a missing column
            # because it is silently plausible instead of visibly absent.
            resolved.append(resolver.get(canonical_key(column), column))

    df.columns = resolved

    duplicated_mask = pd.Index(resolved).duplicated(keep="first")
    if duplicated_mask.any():
        dropped = sorted(set(pd.Index(resolved)[duplicated_mask]))
        logger.info(f"Dropped {int(duplicated_mask.sum())} duplicate column(s), keeping first: {dropped}")
        df = df.loc[:, ~duplicated_mask]

    return df


def extract_labels(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Return the ground-truth label column if the upload carries one, else None.

    Used purely for optional accuracy reporting on labelled uploads; the value
    never participates in preprocessing or inference.
    """
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return df[candidate].astype(str)
    return None


def build_truth_resolver(label_map: dict[str, str]) -> dict[str, str]:
    """
    Build a case-insensitive lookup from any label spelling to its canonical class.

    params.yaml:cleaning.label_map lists only the RAW dataset spellings
    ("DoS Hulk", "Web Attack - XSS", "Benign"), because that is all the cleaning
    stage needs. Serving sees a wider range of inputs: a CSV exported *after*
    cleaning carries labels that are already canonical ("BENIGN", "WebAttack",
    "BruteForce"), and none of those appear as keys in the map.

    Resolving through the raw map alone therefore returns nothing for them, and
    because unresolved labels are excluded from scoring, an accuracy figure
    would be computed over only the handful of classes whose canonical name
    happens to coincide with a raw one -- reporting a confident number derived
    from a small, unrepresentative slice of the file.

    This resolver closes both gaps: every canonical class resolves to itself,
    and lookups are case-insensitive.
    """
    from src.data.clean import build_label_normalizer, normalize_raw_label

    normalized = build_label_normalizer(label_map)

    resolver = {normalize_raw_label(k).casefold(): v for k, v in normalized.items()}
    for canonical in set(normalized.values()):
        resolver.setdefault(canonical.casefold(), canonical)
    return resolver


def resolve_truth_labels(truth: pd.Series, label_map: dict[str, str]) -> pd.Series:
    """
    Map a ground-truth column onto the canonical taxonomy.

    Returns a Series aligned to the input, holding the canonical class name or
    NaN where the label could not be resolved. Callers must exclude NaN from
    scoring rather than count it as a miss.
    """
    from src.data.clean import normalize_raw_label

    resolver = build_truth_resolver(label_map)
    resolved = truth.map(lambda v: resolver.get(normalize_raw_label(str(v)).casefold()))

    unresolved = truth[resolved.isna()]
    if not unresolved.empty:
        logger.warning(
            f"{len(unresolved):,} label(s) did not resolve to the taxonomy and are "
            f"excluded from scoring. Unmapped values: {sorted(set(unresolved))[:5]}"
        )
    return resolved


# ---------------------------------------------------------------------------
# Inference Bundle
# ---------------------------------------------------------------------------

def build_inference_bundle(
    champion_features: list[str],
    scaler_path: str | Path,
    fill_values_path: str | Path,
    feature_names_path: str | Path,
) -> dict[str, Any]:
    """
    Build the self-contained raw -> Champion preprocessing bundle.

    Slices the fitted RobustScaler down to the Champion's columns. This is
    exact, not an approximation: RobustScaler transforms each column
    independently as (x - center_[i]) / scale_[i], so selecting a subset of
    columns selects the matching subset of parameters.

    Parameters
    ----------
    champion_features : list[str]
        The Champion's ordered feature names (models/champion_features.json).
    scaler_path : str | Path
        Pickled scaler fitted during Phase 1 preprocessing.
    fill_values_path : str | Path
        Per-column training fill values, in RAW (pre-scaling) units.
    feature_names_path : str | Path
        Ordered feature names the scaler was fitted on.

    Returns
    -------
    dict
        Serializable bundle: feature order, center/scale vectors, fill values,
        and the alias map, so inference needs no other artifact.
    """
    with open(feature_names_path, "r", encoding="utf-8") as f:
        scaler_features: list[str] = json.load(f)

    with open(fill_values_path, "r", encoding="utf-8") as f:
        fill_values: dict[str, float] = json.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    missing_from_scaler = [f for f in champion_features if f not in scaler_features]
    if missing_from_scaler:
        raise ValueError(
            f"Champion features absent from the fitted scaler's feature list: {missing_from_scaler}. "
            f"The scaler and the Champion were built from different Phase 1 runs."
        )

    if not hasattr(scaler, "center_") or not hasattr(scaler, "scale_"):
        raise TypeError(
            f"Expected a fitted RobustScaler with center_/scale_, got {type(scaler).__name__}. "
            f"Update build_inference_bundle if params.yaml:preprocessing.scaler changed."
        )

    if len(scaler.center_) != len(scaler_features):
        raise ValueError(
            f"Scaler was fitted on {len(scaler.center_)} columns but feature_names.json "
            f"lists {len(scaler_features)}. Artifacts are out of sync."
        )

    index_of = {name: i for i, name in enumerate(scaler_features)}
    champion_indices = [index_of[f] for f in champion_features]

    missing_fills = [f for f in champion_features if f not in fill_values]
    if missing_fills:
        raise ValueError(f"Champion features absent from fill_values.json: {missing_fills}")

    bundle = {
        "feature_names": list(champion_features),
        "center": [float(scaler.center_[i]) for i in champion_indices],
        "scale": [float(scaler.scale_[i]) for i in champion_indices],
        "fill_values": {f: float(fill_values[f]) for f in champion_features},
        "aliases": dict(COLUMN_ALIASES),
        "scaler_type": type(scaler).__name__,
        "source_scaler_feature_count": len(scaler_features),
    }

    logger.info(
        f"Built inference bundle: {len(champion_features)} features sliced from a "
        f"{len(scaler_features)}-feature {bundle['scaler_type']}."
    )
    return bundle


def save_inference_bundle(bundle: dict[str, Any], output_path: str | Path) -> Path:
    """Serialize an inference bundle to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    logger.info(f"Saved inference bundle to: {path}")
    return path


def load_inference_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """
    Load and validate an inference bundle from disk.

    Raises
    ------
    FileNotFoundError
        If the bundle has not been generated (run Phase 2 first).
    ValueError
        If the bundle is internally inconsistent.
    """
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Inference bundle not found: {path.resolve()}. "
            "Run run_phase2.py to regenerate models/champion_preprocessor.json."
        )

    with open(path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    n = len(bundle["feature_names"])
    if not (len(bundle["center"]) == len(bundle["scale"]) == n):
        raise ValueError(
            f"Corrupt inference bundle: {n} feature names but "
            f"{len(bundle['center'])} centers and {len(bundle['scale'])} scales."
        )

    return bundle


# ---------------------------------------------------------------------------
# Raw -> Champion Preparation
# ---------------------------------------------------------------------------

def prepare_with_report(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    already_normalized: bool = False,
    allow_missing: bool = False,
    max_missing_fraction: float = 0.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Convert a RAW network-flow frame into the Champion's scaled input matrix,
    reporting exactly what had to be repaired along the way.

    Pipeline: normalize columns -> resolve missing -> coerce numeric ->
    Inf to NaN -> fill from training medians -> reorder -> scale.

    Missing-column policy
    ---------------------
    Real exports vary. The canonical CIC-IDS2017 CSV release, for instance,
    ships 'Destination Port' where the Parquet release has 'Protocol', so a
    Champion trained on the full feature set has no counterpart column.

    With allow_missing=True such columns are imputed from the training median
    and NAMED in the returned report, so the caller can surface the degradation
    to the user rather than presenting an imputed prediction as a clean one.
    Imputation is refused outright beyond max_missing_fraction, where the input
    is no longer meaningfully the same feature space.

    Returns
    -------
    X : np.ndarray
        Shape (n_rows, n_features), float32.
    report : dict
        imputed_columns, n_filled_cells, and the resolved feature count.
    """
    feature_names: list[str] = bundle["feature_names"]
    fill_values: dict[str, float] = bundle["fill_values"]

    if not already_normalized:
        df = normalize_columns(df)

    missing = [f for f in feature_names if f not in df.columns]

    if missing and not allow_missing:
        raise ValueError(
            f"Input is missing {len(missing)} required feature column(s). "
            f"Missing: {missing[:8]}{' ...' if len(missing) > 8 else ''}"
        )

    if missing:
        fraction = len(missing) / len(feature_names)
        if fraction > max_missing_fraction:
            raise ValueError(
                f"Input is missing {len(missing)} of {len(feature_names)} required features "
                f"({fraction:.0%}), exceeding the {max_missing_fraction:.0%} imputation limit. "
                f"Missing: {missing[:8]}{' ...' if len(missing) > 8 else ''}"
            )
        logger.warning(
            f"Imputing {len(missing)} absent feature column(s) from training medians: {missing}"
        )

    present = [f for f in feature_names if f in df.columns]
    X = df.loc[:, present].copy()
    for name in missing:
        X[name] = fill_values[name]
    X = X.loc[:, feature_names]

    # Coerce to numeric; unparseable entries become NaN and are filled below.
    for col in feature_names:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.astype(np.float64)
    X = X.replace([np.inf, -np.inf], np.nan)

    n_filled = int(X.isna().sum().sum())
    if n_filled > 0:
        X = X.fillna(value={f: fill_values[f] for f in feature_names})
        logger.info(f"Filled {n_filled:,} missing/non-finite value(s) using training medians.")

    center = np.asarray(bundle["center"], dtype=np.float64)
    scale = np.asarray(bundle["scale"], dtype=np.float64)

    # RobustScaler sets scale_ to 1.0 for zero-IQR columns; guard defensively.
    safe_scale = np.where(scale == 0.0, 1.0, scale)

    X_scaled = (X.values - center) / safe_scale

    report = {
        "imputed_columns": missing,
        "n_imputed_columns": len(missing),
        "n_filled_cells": n_filled,
        "feature_count": len(feature_names),
    }
    return np.ascontiguousarray(X_scaled, dtype=np.float32), report


def prepare_for_champion(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    already_normalized: bool = False,
    allow_missing: bool = False,
) -> np.ndarray:
    """
    Strict convenience wrapper over prepare_with_report.

    Raises
    ------
    ValueError
        If any required Champion feature is missing and allow_missing is False.
    """
    X, _ = prepare_with_report(df, bundle, already_normalized, allow_missing)
    return X
