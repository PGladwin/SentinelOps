"""
SentinelOps - Data Preprocessing Module
==========================================
Responsibilities:
  - Separate features (X) from labels (y)
  - Encode canonical class names → integer labels
  - Create stratified development sample
  - Stratified train/test split
  - Fit scaler ONLY on training data, transform train and test
  - Fill NaN ONLY from training statistics (prevent leakage)
  - Detect and optionally drop highly-correlated feature pairs
  - Save all preprocessing artifacts (scaler, feature list, label encoder)
  - Save train/test/dev splits as Parquet files
  - Save reproducible metadata

Data leakage prevention:
  - Scaler fitted on X_train only
  - NaN fill values computed on X_train only
  - Correlation analysis done on X_train only
  - All transformations then applied to X_test using training artifacts

Usage:
    from src.data.preprocess import run_preprocessing
    artifacts = run_preprocessing(clean_df, params)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label Encoding
# ---------------------------------------------------------------------------

def encode_labels(
    label_series: pd.Series,
    class_encoding: dict[str, int],
) -> tuple[np.ndarray, dict]:
    """
    Encode canonical class names to integers.

    Parameters
    ----------
    label_series : Series of canonical class names (e.g., 'BENIGN', 'DoS')
    class_encoding : dict mapping canonical name → integer

    Returns
    -------
    y : numpy int array
    stats : encoding statistics
    """
    # Validate no unknown canonical classes
    unique_classes = set(label_series.unique())
    known_classes = set(class_encoding.keys())
    unknowns = unique_classes - known_classes
    if unknowns:
        raise ValueError(
            f"Unknown canonical classes during encoding: {unknowns}\n"
            f"Expected: {known_classes}"
        )

    y = label_series.map(class_encoding).astype(np.int32).values
    stats = {
        "encoding": class_encoding,
        "class_counts": label_series.value_counts().to_dict(),
        "encoded_unique": sorted(np.unique(y).tolist()),
    }
    return y, stats


# ---------------------------------------------------------------------------
# Stratified Sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    df: pd.DataFrame,
    n_samples: int,
    label_col: str,
    random_seed: int,
) -> pd.DataFrame:
    """
    Create a stratified sample of n_samples rows from df.
    Maintains class proportions. If a class has fewer samples than needed,
    all samples from that class are kept.

    Returns sampled DataFrame (shuffled, index reset).
    """
    classes = df[label_col].unique()
    n_total = len(df)
    fraction = n_samples / n_total

    sampled_parts = []
    for cls in classes:
        cls_df = df[df[label_col] == cls]
        n_cls_sample = max(1, int(len(cls_df) * fraction))
        # Don't oversample: cap at class size
        n_cls_sample = min(n_cls_sample, len(cls_df))
        sampled_parts.append(
            cls_df.sample(n=n_cls_sample, random_state=random_seed)
        )

    result = pd.concat(sampled_parts, axis=0).sample(
        frac=1.0, random_state=random_seed
    ).reset_index(drop=True)

    logger.info(
        f"Stratified sample: {n_total:,} → {len(result):,} rows "
        f"({len(result)/n_total*100:.1f}%)"
    )
    return result


# ---------------------------------------------------------------------------
# Correlation Filtering
# ---------------------------------------------------------------------------

def find_correlated_pairs(
    X: pd.DataFrame,
    threshold: float,
) -> list[str]:
    """
    Find features to drop from highly-correlated pairs.

    Strategy:
      - Compute pairwise Pearson correlation on training data
      - For each pair with |corr| > threshold, mark the second column for dropping
      - Return list of columns to drop

    This is O(n_features^2) but fine for ~70 features.
    """
    corr_matrix = X.corr(method="pearson").abs()

    # Only upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    logger.info(
        f"Correlation filter (threshold={threshold}): "
        f"dropping {len(to_drop)} features: {to_drop}"
    )
    return to_drop


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

def build_scaler(scaler_type: str):
    """
    Build an sklearn-compatible scaler object.
    Supported: 'robust', 'standard', 'minmax'
    """
    if scaler_type == "robust":
        from sklearn.preprocessing import RobustScaler
        return RobustScaler()
    elif scaler_type == "standard":
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    elif scaler_type == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: '{scaler_type}'. Use 'robust', 'standard', or 'minmax'.")


# ---------------------------------------------------------------------------
# Main Preprocessing Pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(
    clean_df: pd.DataFrame,
    params: dict,
    use_dev_sample: bool = True,
) -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    clean_df : output of clean_dataframe() — contains 'label_class', 'label_raw',
               'source_file' metadata columns, and all feature columns.
    params : full params.yaml dict
    use_dev_sample : if True and dev_sample_size is set, use the dev sample.

    Returns
    -------
    artifacts : dict containing:
        - 'X_train', 'X_test' : scaled numpy arrays
        - 'y_train', 'y_test' : integer label arrays
        - 'feature_names'     : list of feature columns after filtering
        - 'scaler'            : fitted scaler object
        - 'fill_values'       : dict col→fill_value (from training data)
        - 'class_encoding'    : dict canonical_name → int
        - 'dropped_const'     : list of constant cols (already removed in clean.py)
        - 'dropped_corr'      : list of corr-filtered cols
        - 'metadata'          : full reproducibility metadata dict
    """
    preproc_cfg = params["preprocessing"]
    cleaning_cfg = params["cleaning"]
    general_cfg = params["general"]
    paths_cfg = params["paths"]

    random_seed = general_cfg["random_seed"]
    test_size = preproc_cfg["test_size"]
    scaler_type = preproc_cfg["scaler"]
    dev_sample_size = preproc_cfg.get("dev_sample_size")
    corr_threshold = preproc_cfg.get("correlation_threshold")
    nan_strategy = cleaning_cfg["nan_fill_strategy"]
    class_encoding = cleaning_cfg["class_encoding"]

    processed_dir = Path(paths_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SENTINELOPS - PREPROCESSING")
    logger.info("=" * 60)
    logger.info(f"Input shape: {clean_df.shape}")

    # ----------------------------------------------------------------
    # Step 1: Identify feature columns
    #   Exclude metadata columns: label_raw, label_class, source_file
    # ----------------------------------------------------------------
    META_COLS = {"label_raw", "label_class", "source_file"}
    feature_cols = [c for c in clean_df.columns if c not in META_COLS]
    logger.info(f"Feature columns: {len(feature_cols)}")

    # ----------------------------------------------------------------
    # Step 2: Optional development sample (stratified)
    # ----------------------------------------------------------------
    working_df = clean_df.copy()

    if use_dev_sample and dev_sample_size is not None:
        if len(working_df) > dev_sample_size:
            logger.info(f"Creating stratified dev sample: {dev_sample_size:,} rows...")
            working_df = stratified_sample(
                working_df, dev_sample_size, "label_class", random_seed
            )
        else:
            logger.info(
                f"Dataset ({len(working_df):,} rows) ≤ dev_sample_size "
                f"({dev_sample_size:,}); using full dataset."
            )

    # ----------------------------------------------------------------
    # Step 3: Stratified train/test split
    # ----------------------------------------------------------------
    from sklearn.model_selection import train_test_split

    # float32 rather than the mixed int8/int32/float32 source dtypes: .values
    # on a mixed frame upcasts to float64, which doubles peak memory on the
    # full 2.2M-row corpus. Downstream consumers (RobustScaler, XGBoost,
    # sklearn) all operate in float32 anyway, so no precision is lost that
    # survives to the model.
    X_raw = working_df[feature_cols].astype(np.float32)
    y_encoded, encode_stats = encode_labels(working_df["label_class"], class_encoding)

    logger.info(f"Splitting: test_size={test_size}, seed={random_seed}")

    # Handle extremely rare classes (< 2 samples) that sklearn cannot stratify.
    # These are placed entirely into training to avoid loss, with a warning.
    unique_classes, counts = np.unique(y_encoded, return_counts=True)
    rare_classes = unique_classes[counts < 2]
    if len(rare_classes) > 0:
        inv_enc = {v: k for k, v in class_encoding.items()}
        rare_names = [inv_enc.get(int(c), str(c)) for c in rare_classes]
        logger.warning(
            f"Classes with < 2 samples cannot be stratified: {rare_names}. "
            f"These samples will be placed entirely in the training set."
        )
        # Separate rare-class rows
        rare_mask = np.isin(y_encoded, rare_classes)
        X_rare = X_raw.values[rare_mask]
        y_rare = y_encoded[rare_mask]
        X_common = X_raw.values[~rare_mask]
        y_common = y_encoded[~rare_mask]

        X_train_raw, X_test_raw, y_train_common, y_test = train_test_split(
            X_common, y_common,
            test_size=test_size,
            stratify=y_common,
            random_state=random_seed,
        )
        # Append rare rows to training set
        X_train_raw = np.vstack([X_train_raw, X_rare])
        y_train = np.concatenate([y_train_common, y_rare])
    else:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw.values,
            y_encoded,
            test_size=test_size,
            stratify=y_encoded,
            random_state=random_seed,
        )

    # Keep column names for later correlation analysis
    X_train_df = pd.DataFrame(X_train_raw, columns=feature_cols)
    X_test_df = pd.DataFrame(X_test_raw, columns=feature_cols)
    del X_train_raw, X_test_raw, X_raw

    logger.info(f"  Train: {len(X_train_df):,} rows")
    logger.info(f"  Test : {len(X_test_df):,} rows")

    # ----------------------------------------------------------------
    # Step 4: Fill NaN using training statistics only (prevent leakage)
    # ----------------------------------------------------------------
    from src.data.clean import compute_fill_values, apply_fill_values

    logger.info(f"Computing NaN fill values from training data (strategy='{nan_strategy}')...")
    fill_values = compute_fill_values(X_train_df, nan_strategy)

    X_train_df, fill_train_stats = apply_fill_values(X_train_df, fill_values)
    X_test_df, fill_test_stats = apply_fill_values(X_test_df, fill_values)

    # Verify no NaN remains
    nan_remaining_train = int(X_train_df.isna().sum().sum())
    nan_remaining_test = int(X_test_df.isna().sum().sum())
    if nan_remaining_train > 0 or nan_remaining_test > 0:
        raise RuntimeError(
            f"NaN remain after fill: train={nan_remaining_train}, test={nan_remaining_test}. "
            f"Check fill_values computation."
        )
    logger.info("NaN check passed: 0 NaN remaining in train and test.")

    # ----------------------------------------------------------------
    # Step 5: Correlation filtering (on training data only)
    # ----------------------------------------------------------------
    dropped_corr = []
    if corr_threshold is not None:
        logger.info(f"Running correlation filter (threshold={corr_threshold})...")
        dropped_corr = find_correlated_pairs(X_train_df, corr_threshold)
        if dropped_corr:
            X_train_df = X_train_df.drop(columns=dropped_corr)
            X_test_df = X_test_df.drop(columns=dropped_corr)
            logger.info(f"Dropped {len(dropped_corr)} correlated features.")
        else:
            logger.info("No features dropped by correlation filter.")
    else:
        logger.info("Correlation filtering skipped (correlation_threshold=null).")

    final_feature_cols = list(X_train_df.columns)
    logger.info(f"Final feature count: {len(final_feature_cols)}")

    # ----------------------------------------------------------------
    # Step 6: Scale using training statistics only (prevent leakage)
    # ----------------------------------------------------------------
    logger.info(f"Fitting scaler: {scaler_type}")
    scaler = build_scaler(scaler_type)
    X_train_scaled = scaler.fit_transform(X_train_df.values)
    X_test_scaled = scaler.transform(X_test_df.values)

    # ----------------------------------------------------------------
    # Step 7: Save artifacts
    # ----------------------------------------------------------------
    logger.info(f"Saving preprocessing artifacts to: {processed_dir}")

    # Scaler
    scaler_path = processed_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Fill values
    fill_path = processed_dir / "fill_values.json"
    with open(fill_path, "w") as f:
        json.dump(fill_values, f, indent=2)

    # Feature list (ORDER MATTERS for inference — never change this)
    feature_list_path = processed_dir / "feature_names.json"
    with open(feature_list_path, "w") as f:
        json.dump(final_feature_cols, f, indent=2)

    # Class encoding
    encoding_path = processed_dir / "class_encoding.json"
    with open(encoding_path, "w") as f:
        json.dump(class_encoding, f, indent=2)

    # Inverse encoding (int → class name)
    inverse_encoding = {v: k for k, v in class_encoding.items()}
    inverse_path = processed_dir / "inverse_encoding.json"
    with open(inverse_path, "w") as f:
        json.dump(inverse_encoding, f, indent=2)

    # Save dropped correlated features
    corr_drop_path = processed_dir / "dropped_correlated_features.json"
    with open(corr_drop_path, "w") as f:
        json.dump({"dropped_features": dropped_corr, "threshold": corr_threshold}, f, indent=2)

    # ----------------------------------------------------------------
    # Step 8: Save train/test splits as Parquet (with labels)
    # ----------------------------------------------------------------
    _save_split(
        X_train_scaled, y_train, final_feature_cols, class_encoding,
        processed_dir / "train.parquet"
    )
    _save_split(
        X_test_scaled, y_test, final_feature_cols, class_encoding,
        processed_dir / "test.parquet"
    )

    # ----------------------------------------------------------------
    # Step 9: Build and save metadata
    # ----------------------------------------------------------------
    train_class_dist = _count_classes(y_train, inverse_encoding)
    test_class_dist = _count_classes(y_test, inverse_encoding)

    metadata = {
        "random_seed": random_seed,
        "scaler_type": scaler_type,
        "nan_fill_strategy": nan_strategy,
        "test_size": test_size,
        "dev_sample_size": dev_sample_size,
        "used_dev_sample": use_dev_sample and dev_sample_size is not None,
        "total_samples_before_split": len(X_train_df) + len(X_test_df),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features_before_corr_filter": len(feature_cols),
        "n_features_after_corr_filter": len(final_feature_cols),
        "corr_threshold": corr_threshold,
        "dropped_corr_features": dropped_corr,
        "class_encoding": class_encoding,
        "train_class_distribution": train_class_dist,
        "test_class_distribution": test_class_dist,
        "artifacts": {
            "scaler": str(scaler_path),
            "fill_values": str(fill_path),
            "feature_names": str(feature_list_path),
            "class_encoding": str(encoding_path),
            "inverse_encoding": str(inverse_path),
            "train_parquet": str(processed_dir / "train.parquet"),
            "test_parquet": str(processed_dir / "test.parquet"),
        },
    }

    metadata_path = processed_dir / "preprocessing_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Preprocessing complete.")
    logger.info(f"  Train: {metadata['n_train']:,} rows × {len(final_feature_cols)} features")
    logger.info(f"  Test : {metadata['n_test']:,} rows × {len(final_feature_cols)} features")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": final_feature_cols,
        "scaler": scaler,
        "fill_values": fill_values,
        "class_encoding": class_encoding,
        "inverse_encoding": inverse_encoding,
        "dropped_corr": dropped_corr,
        "metadata": metadata,
    }


def _save_split(
    X_scaled: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    class_encoding: dict[str, int],
    output_path: Path,
) -> None:
    """Save a scaled split as Parquet with integer label column."""
    inverse_encoding = {v: k for k, v in class_encoding.items()}
    df = pd.DataFrame(X_scaled, columns=feature_names)
    df["label_int"] = y.astype(np.int32)
    df["label_class"] = [inverse_encoding[yi] for yi in y]
    df.to_parquet(output_path, index=False)
    logger.debug(f"Saved split: {output_path} ({len(df):,} rows)")


def _count_classes(y: np.ndarray, inverse_encoding: dict) -> dict:
    """Return per-class counts from an integer label array."""
    unique, counts = np.unique(y, return_counts=True)
    return {
        inverse_encoding.get(int(u), f"class_{u}"): int(c)
        for u, c in zip(unique, counts)
    }


def print_preprocessing_report(artifacts: dict) -> None:
    """Pretty-print preprocessing results."""
    meta = artifacts["metadata"]
    print("\n" + "=" * 65)
    print("PREPROCESSING REPORT")
    print("=" * 65)
    print(f"  Dev sample used  : {meta['used_dev_sample']} ({meta['dev_sample_size']})")
    print(f"  Total samples    : {meta['total_samples_before_split']:,}")
    print(f"  Train samples    : {meta['n_train']:,}")
    print(f"  Test samples     : {meta['n_test']:,}")
    print(f"  Features (pre)   : {meta['n_features_before_corr_filter']}")
    print(f"  Features (post)  : {meta['n_features_after_corr_filter']}")
    print(f"  Dropped (corr)   : {len(meta['dropped_corr_features'])}")
    print(f"  Scaler           : {meta['scaler_type']}")
    print(f"  NaN strategy     : {meta['nan_fill_strategy']}")
    print(f"  Random seed      : {meta['random_seed']}")

    print("\n--- Train Class Distribution ---")
    for cls, cnt in sorted(meta["train_class_distribution"].items(), key=lambda x: -x[1]):
        pct = cnt / meta["n_train"] * 100
        print(f"  {cls:15s}  {cnt:>8,}  ({pct:6.3f}%)")

    print("\n--- Test Class Distribution ---")
    for cls, cnt in sorted(meta["test_class_distribution"].items(), key=lambda x: -x[1]):
        pct = cnt / meta["n_test"] * 100
        print(f"  {cls:15s}  {cnt:>8,}  ({pct:6.3f}%)")

    if meta["dropped_corr_features"]:
        print(f"\nDropped correlated features (threshold={meta['corr_threshold']}):")
        for f in meta["dropped_corr_features"]:
            print(f"  - {f}")

    print("\nArtifacts saved:")
    for name, path in meta["artifacts"].items():
        print(f"  {name:20s}: {path}")
    print("=" * 65)
