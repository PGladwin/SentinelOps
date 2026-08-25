"""
SentinelOps - Model Training Module
====================================
Phase 2: Training implementations for Random Forest, XGBoost, and MLP.

Responsibilities:
  - Load Phase 1 processed data and metadata artifacts
  - Compute class and sample weights for extreme class imbalance
  - Train Random Forest with class_weight="balanced"
  - Train multiclass XGBoost with per-sample class weights (sample_weight)
  - Train MLP with early stopping (documenting sklearn MLPClassifier weighting constraints)
  - Save trained models and comprehensive metadata
"""

import json
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import xgboost as xgb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_phase1_data(
    params: dict,
    data_dir: Optional[str | Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int], dict[int, str]]:
    """
    Load Phase 1 train and test Parquet files along with feature names and class encodings.

    Parameters
    ----------
    params : dict
        Loaded params.yaml configuration.
    data_dir : Optional[str | Path]
        Override path to processed data directory. Defaults to params['paths']['processed_dir'].

    Returns
    -------
    X_train : np.ndarray
    y_train : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    feature_names : list[str]
    class_encoding : dict[str, int]
    inverse_encoding : dict[int, str]
    """
    processed_path = Path(data_dir or params["paths"]["processed_dir"])
    train_path = processed_path / "train.parquet"
    test_path = processed_path / "test.parquet"
    feature_names_path = processed_path / "feature_names.json"
    class_encoding_path = processed_path / "class_encoding.json"
    inverse_encoding_path = processed_path / "inverse_encoding.json"

    for required_file in [train_path, test_path, feature_names_path, class_encoding_path]:
        if not required_file.exists():
            raise FileNotFoundError(
                f"Missing Phase 1 artifact: {required_file.resolve()}. "
                f"Please run Phase 1 pipeline (run_phase1.py) first."
            )

    logger.info(f"Loading Phase 1 artifacts from: {processed_path.resolve()}")

    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    with open(class_encoding_path, "r", encoding="utf-8") as f:
        class_encoding = json.load(f)

    if inverse_encoding_path.exists():
        with open(inverse_encoding_path, "r", encoding="utf-8") as f:
            raw_inv = json.load(f)
            inverse_encoding = {int(k): v for k, v in raw_inv.items()}
    else:
        inverse_encoding = {v: k for k, v in class_encoding.items()}

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df[feature_names].values.astype(np.float32)
    y_train = train_df["label_int"].values.astype(np.int32)
    X_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df["label_int"].values.astype(np.int32)

    logger.info(
        f"Data loaded successfully: Train shape=({len(X_train):,}, {len(feature_names)}), "
        f"Test shape=({len(X_test):,}, {len(feature_names)})"
    )
    return X_train, y_train, X_test, y_test, feature_names, class_encoding, inverse_encoding


# ---------------------------------------------------------------------------
# Class & Sample Weighting Utilities
# ---------------------------------------------------------------------------

def compute_sample_weights(y: np.ndarray, method: str = "balanced") -> np.ndarray:
    """
    Compute sample weights for multiclass training given class imbalance.

    Parameters
    ----------
    y : np.ndarray
        Array of integer target labels.
    method : str
        Weighting heuristic ('balanced' or custom).

    Returns
    -------
    sample_weights : np.ndarray of shape (len(y),)
    """
    sample_weights = compute_sample_weight(class_weight=method, y=y)
    return sample_weights.astype(np.float32)


def compute_class_weight_dict(y: np.ndarray) -> dict[int, float]:
    """
    Compute per-class weights dictionary for inspection.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


# ---------------------------------------------------------------------------
# Model Training: Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
    feature_names: Optional[list[str]] = None,
) -> tuple[RandomForestClassifier, float]:
    """
    Train Random Forest baseline classifier with balanced class weighting.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    params : dict
    feature_names : Optional[list[str]]

    Returns
    -------
    model : RandomForestClassifier
    train_time : float (seconds)
    """
    rf_cfg = params.get("training", {}).get("random_forest", {})
    seed = params.get("general", {}).get("random_seed", 42)

    n_estimators = rf_cfg.get("n_estimators", 100)
    max_depth = rf_cfg.get("max_depth", 20)
    min_samples_leaf = rf_cfg.get("min_samples_leaf", 2)
    class_weight = rf_cfg.get("class_weight", "balanced")
    n_jobs = rf_cfg.get("n_jobs", -1)

    logger.info("Initializing Random Forest classifier:")
    logger.info(
        f"  n_estimators={n_estimators}, max_depth={max_depth}, "
        f"min_samples_leaf={min_samples_leaf}, class_weight={class_weight}, seed={seed}"
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=n_jobs,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    logger.info(f"Random Forest trained successfully in {train_time:.2f}s")
    return model, train_time


# ---------------------------------------------------------------------------
# Model Training: XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
    feature_names: Optional[list[str]] = None,
) -> tuple[xgb.XGBClassifier, float]:
    """
    Train Multiclass XGBoost classifier using per-sample class weights.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    params : dict
    feature_names : Optional[list[str]]

    Returns
    -------
    model : xgb.XGBClassifier
    train_time : float (seconds)
    """
    xgb_cfg = params.get("training", {}).get("xgboost", {})
    seed = params.get("general", {}).get("random_seed", 42)

    n_estimators = xgb_cfg.get("n_estimators", 200)
    max_depth = xgb_cfg.get("max_depth", 6)
    learning_rate = xgb_cfg.get("learning_rate", 0.1)
    subsample = xgb_cfg.get("subsample", 0.8)
    colsample_bytree = xgb_cfg.get("colsample_bytree", 0.8)
    eval_metric = xgb_cfg.get("eval_metric", "mlogloss")
    n_jobs = xgb_cfg.get("n_jobs", -1)

    # Calculate sample weights for multiclass imbalance
    sample_weights = compute_sample_weights(y_train, method="balanced")
    num_classes = 8  # 8-class taxonomy

    logger.info("Initializing Multiclass XGBoost classifier:")
    logger.info(
        f"  n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}, "
        f"subsample={subsample}, colsample={colsample_bytree}, num_classes={num_classes}"
    )

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric=eval_metric,
        objective="multi:softprob",
        num_class=num_classes,
        random_state=seed,
        n_jobs=n_jobs,
        tree_method="hist",
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    train_time = time.perf_counter() - t0

    logger.info(f"XGBoost trained successfully with sample weights in {train_time:.2f}s")
    return model, train_time


# ---------------------------------------------------------------------------
# Model Training: MLP Classifier
# ---------------------------------------------------------------------------

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
    feature_names: Optional[list[str]] = None,
) -> tuple[MLPClassifier, float]:
    """
    Train Multi-Layer Perceptron (MLP) classifier with early stopping.

    Note on class weighting:
    scikit-learn's MLPClassifier does not natively support sample_weight or
    class_weight in fit(). We document this limitation explicitly rather than
    introducing unverified approximations.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    params : dict
    feature_names : Optional[list[str]]

    Returns
    -------
    model : MLPClassifier
    train_time : float (seconds)
    """
    mlp_cfg = params.get("training", {}).get("mlp", {})
    seed = params.get("general", {}).get("random_seed", 42)

    hidden_layers = tuple(mlp_cfg.get("hidden_layers", [128, 64, 32]))
    max_epochs = mlp_cfg.get("max_epochs", 50)
    batch_size = mlp_cfg.get("batch_size", 512)
    learning_rate = mlp_cfg.get("learning_rate", 0.001)
    early_stopping_patience = mlp_cfg.get("early_stopping_patience", 5)

    logger.info("Initializing MLP Classifier:")
    logger.info(
        f"  hidden_layers={hidden_layers}, max_iter={max_epochs}, batch_size={batch_size}, "
        f"lr={learning_rate}, early_stopping_patience={early_stopping_patience}"
    )
    logger.info(
        "  [Note] sklearn MLPClassifier does not support sample_weight in fit(); "
        "training with standard cross-entropy and early stopping."
    )

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_epochs,
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=early_stopping_patience,
        random_state=seed,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    logger.info(
        f"MLP trained in {train_time:.2f}s (stopped at epoch {model.n_iter_}/{max_epochs})"
    )
    return model, train_time


# ---------------------------------------------------------------------------
# Artifact Saving & Loading
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    model_name: str,
    models_dir: str | Path,
    metadata: dict,
) -> tuple[Path, Path]:
    """
    Save trained model (.pkl) and associated metadata (.json).

    Parameters
    ----------
    model : Any
        Trained model instance.
    model_name : str
        Identifier (e.g., 'random_forest', 'xgboost', 'mlp').
    models_dir : str | Path
        Directory to save artifacts.
    metadata : dict
        Metadata dictionary to serialize.

    Returns
    -------
    model_path : Path
    meta_path : Path
    """
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{model_name}.pkl"
    meta_path = out_dir / f"{model_name}_metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Ensure creation timestamp
    if "creation_timestamp" not in metadata:
        metadata["creation_timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved model to: {model_path}")
    logger.info(f"Saved model metadata to: {meta_path}")
    return model_path, meta_path


def load_model(model_path: str | Path) -> Any:
    """Load a pickled model from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")
    with open(path, "rb") as f:
        return pickle.load(f)


# Canonical serving artifact names. Whatever the governance gate promotes is
# written under these names, so the API is decoupled from the winning
# algorithm. The legacy xgboost_top40.* names remain a fallback.
CHAMPION_ARTIFACTS = ("champion.pkl", "champion_features.json", "champion_metadata.json")
LEGACY_CHAMPION_ARTIFACTS = (
    "xgboost_top40.pkl",
    "xgboost_top40_features.json",
    "xgboost_top40_metadata.json",
)


def resolve_champion_artifacts(models_dir: str | Path = "models") -> tuple[Path, Path, Path]:
    """
    Locate the serving artifacts for the current champion.

    Prefers the canonical champion.* names written on promotion; falls back to
    the legacy xgboost_top40.* names so existing checkouts keep working.

    Raises
    ------
    FileNotFoundError
        If neither complete set is present.
    """
    dir_path = Path(models_dir)

    for names in (CHAMPION_ARTIFACTS, LEGACY_CHAMPION_ARTIFACTS):
        paths = tuple(dir_path / n for n in names)
        if all(p.exists() for p in paths):
            return paths  # type: ignore[return-value]

    raise FileNotFoundError(
        f"No champion artifacts in {dir_path.resolve()}. Expected either "
        f"{list(CHAMPION_ARTIFACTS)} or {list(LEGACY_CHAMPION_ARTIFACTS)}. "
        "Run run_phase2.py or scripts/run_experiment_sweep.py first."
    )


def load_champion_model(
    models_dir: str | Path = "models",
) -> tuple[Any, list[str], dict]:
    """
    Load the promoted Champion model, its ordered feature list, and metadata.

    Parameters
    ----------
    models_dir : str | Path
        Directory containing champion artifacts.

    Returns
    -------
    model : Any
        Trained Champion model.
    feature_names : list[str]
        Exact ordered feature names the model expects.
    metadata : dict
        Champion metadata dictionary.
    """
    model_path, features_path, meta_path = resolve_champion_artifacts(models_dir)

    model = load_model(model_path)

    with open(features_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(
        f"Loaded Champion model '{metadata.get('model_name', model_path.stem)}' "
        f"({len(feature_names)} features) from {model_path.name}"
    )
    return model, feature_names, metadata


def predict_with_champion(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    input_feature_names: Optional[list[str]] = None,
    champion_features: Optional[list[str]] = None,
) -> np.ndarray:
    """
    Generate predictions using the Champion model, strictly enforcing the 40-feature subset & order.

    Parameters
    ----------
    model : Any
        Loaded Champion model.
    X : pd.DataFrame or np.ndarray
        Input features.
    input_feature_names : Optional[list[str]]
        Names of columns in X if X is a numpy array.
    champion_features : Optional[list[str]]
        Expected 40 features in order. If None, loaded from metadata or model.

    Returns
    -------
    predictions : np.ndarray
    """
    if isinstance(X, pd.DataFrame):
        if champion_features is not None:
            missing = set(champion_features) - set(X.columns)
            if missing:
                raise ValueError(f"Input DataFrame is missing required champion features: {missing}")
            X_aligned = X[champion_features].values
        else:
            X_aligned = X.values
    elif isinstance(X, np.ndarray):
        if X.shape[1] == 40:
            X_aligned = X
        elif input_feature_names is not None and champion_features is not None:
            feat_to_idx = {f: i for i, f in enumerate(input_feature_names)}
            selected_indices = [feat_to_idx[f] for f in champion_features]
            X_aligned = X[:, selected_indices]
        else:
            raise ValueError(
                f"Input array has {X.shape[1]} features, but Champion expects 40 features. "
                "Provide input_feature_names and champion_features to enable automatic column alignment."
            )
    else:
        raise TypeError(f"Expected DataFrame or ndarray, got {type(X)}")

    return model.predict(X_aligned)
