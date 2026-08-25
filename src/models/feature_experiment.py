"""
SentinelOps - Feature Set Experiment Module
===========================================
Phase 2: Controlled feature-set comparison.

Compares:
  A. 48-Feature representation (Phase 1 post-correlation filtering)
  B. 69-Feature representation (Full feature set prior to correlation filter)
  C. Top-K Feature representations (Top-20, Top-30, Top-40 based on training feature importance)

Data Leakage Prevention:
  - Feature importance ranking is computed STRICTLY on X_train.
  - Test set features are subsetted using training-derived feature indices only.
  - Test set is never touched during feature selection or scaling.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.models.evaluate import evaluate_model
from src.models.train import compute_sample_weights

logger = logging.getLogger(__name__)


def compute_feature_importances(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    model_type: str = "xgb",
    params: Optional[dict] = None,
    random_seed: int = 42,
) -> list[tuple[str, float, int]]:
    """
    Compute feature importances using model fitted on TRAINING DATA ONLY.

    Returns
    -------
    ranked_features : list of (feature_name, importance_score, original_index)
        Sorted in descending order of importance.
    """
    logger.info(f"Computing training-data feature importances using {model_type.upper()} (Leakage-free)...")
    if model_type == "xgb":
        import xgboost as xgb
        sample_weights = compute_sample_weights(y_train, method="balanced")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=8,
            random_state=random_seed,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        importances = model.feature_importances_
    else:
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        importances = model.feature_importances_

    ranked = sorted(
        [
            (feature_names[i], float(importances[i]), i)
            for i in range(len(feature_names))
        ],
        key=lambda x: -x[1],
    )
    return ranked


def get_69_feature_splits(
    params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Generate or load the 69-feature representation (no correlation filter).
    Uses the exact same Phase 1 cleaning and splitting logic (seed=42).
    """
    from src.data.clean import clean_dataframe
    from src.data.ingest import load_all_parquet
    from src.data.preprocess import run_preprocessing

    params_69 = json.loads(json.dumps(params))
    params_69["preprocessing"]["correlation_threshold"] = None
    params_69["paths"]["processed_dir"] = "data/processed_69_temp"

    raw_df, _ = load_all_parquet(params_69)
    clean_df, _ = clean_dataframe(raw_df, params_69)
    del raw_df

    artifacts = run_preprocessing(
        clean_df,
        params_69,
        use_dev_sample=(params["preprocessing"].get("dev_sample_size") is not None),
    )
    del clean_df

    return (
        artifacts["X_train"],
        artifacts["y_train"],
        artifacts["X_test"],
        artifacts["y_test"],
        artifacts["feature_names"],
    )


def run_feature_experiment(
    X_train_48: np.ndarray,
    y_train_48: np.ndarray,
    X_test_48: np.ndarray,
    y_test_48: np.ndarray,
    feature_names_48: list[str],
    class_encoding: dict[str, int],
    params: dict,
    evaluator: str = "xgb",
    top_k_values: list[int] = [20, 30, 40],
    include_69_features: bool = True,
    reports_dir: str | Path = "reports",
) -> dict:
    """
    Execute controlled feature-set comparison using the specified model (default: XGBoost).

    Parameters
    ----------
    X_train_48, y_train_48, X_test_48, y_test_48 : 48-feature dataset
    feature_names_48 : list[str]
    class_encoding : dict[str, int]
    params : dict
    evaluator : str, 'xgb' or 'rf' (default: 'xgb')
    top_k_values : list[int]
    include_69_features : bool
    reports_dir : str | Path

    Returns
    -------
    experiment_results : dict
    """
    import xgboost as xgb
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"feature_set_comparison_{evaluator}.json"

    logger.info("=" * 60)
    logger.info(f"STARTING FEATURE SET EXPERIMENT (Evaluator: {evaluator.upper()})")
    logger.info("=" * 60)

    seed = params.get("general", {}).get("random_seed", 42)
    ranked_features = compute_feature_importances(
        X_train_48, y_train_48, feature_names_48, model_type=evaluator, params=params, random_seed=seed
    )

    feature_sets_to_evaluate = {}

    # Top-K feature sets
    for k in top_k_values:
        if k < len(feature_names_48):
            selected_indices = [idx for _, _, idx in ranked_features[:k]]
            selected_names = [name for name, _, _ in ranked_features[:k]]
            feature_sets_to_evaluate[f"Top-{k} Features"] = {
                "X_tr": X_train_48[:, selected_indices],
                "y_tr": y_train_48,
                "X_te": X_test_48[:, selected_indices],
                "y_te": y_test_48,
                "n_features": k,
                "features": selected_names,
            }

    # Baseline 48-feature set
    feature_sets_to_evaluate["48-Feature Set (Phase 1 Baseline)"] = {
        "X_tr": X_train_48,
        "y_tr": y_train_48,
        "X_te": X_test_48,
        "y_te": y_test_48,
        "n_features": 48,
        "features": feature_names_48,
    }

    # Optional 69-feature set
    if include_69_features:
        try:
            logger.info("Loading 69-feature representation (all un-correlated features)...")
            X_tr_69, y_tr_69, X_te_69, y_te_69, feats_69 = get_69_feature_splits(params)
            feature_sets_to_evaluate["69-Feature Set (No Correlation Drop)"] = {
                "X_tr": X_tr_69,
                "y_tr": y_tr_69,
                "X_te": X_te_69,
                "y_te": y_te_69,
                "n_features": len(feats_69),
                "features": feats_69,
            }
        except Exception as e:
            logger.warning(f"Could not load 69-feature dataset: {e}. Skipping 69-feature test.")

    # Evaluate each feature set
    results = []
    xgb_cfg = params.get("training", {}).get("xgboost", {})
    rf_cfg = params.get("training", {}).get("random_forest", {})

    for set_name, f_data in feature_sets_to_evaluate.items():
        logger.info(f"Evaluating feature set with {evaluator.upper()}: '{set_name}' ({f_data['n_features']} features)...")

        t0 = time.perf_counter()
        if evaluator == "xgb":
            sample_weights = compute_sample_weights(f_data["y_tr"], method="balanced")
            model = xgb.XGBClassifier(
                n_estimators=xgb_cfg.get("n_estimators", 200),
                max_depth=xgb_cfg.get("max_depth", 6),
                learning_rate=xgb_cfg.get("learning_rate", 0.1),
                subsample=xgb_cfg.get("subsample", 0.8),
                colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
                eval_metric=xgb_cfg.get("eval_metric", "mlogloss"),
                objective="multi:softprob",
                num_class=8,
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
            )
            model.fit(f_data["X_tr"], f_data["y_tr"], sample_weight=sample_weights)
        else:
            model = RandomForestClassifier(
                n_estimators=rf_cfg.get("n_estimators", 100),
                max_depth=rf_cfg.get("max_depth", 20),
                min_samples_leaf=rf_cfg.get("min_samples_leaf", 2),
                class_weight=rf_cfg.get("class_weight", "balanced"),
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(f_data["X_tr"], f_data["y_tr"])
        train_time = time.perf_counter() - t0

        metrics = evaluate_model(
            model,
            f_data["X_te"],
            f_data["y_te"],
            class_encoding=class_encoding,
            model_name=f"{evaluator}_{set_name}",
            train_time=train_time,
        )

        s = metrics["summary"]
        t = metrics["timing"]

        results.append({
            "feature_set": set_name,
            "feature_count": f_data["n_features"],
            "macro_f1": s["macro_f1"],
            "mean_attack_fnr": s["mean_attack_fnr"],
            "macro_attack_f1": s["macro_attack_f1"],
            "mean_fnr": s["mean_fnr"],
            "weighted_f1": s["weighted_f1"],
            "accuracy": s["accuracy"],
            "mean_minority_f1": s["mean_minority_f1"],
            "training_time_seconds": t["training_time_seconds"],
            "inference_latency_ms_per_1000": t["inference_latency_ms_per_1000"],
            "per_class_metrics": metrics["per_class"],
        })

    # Sort results by Macro F1
    results_sorted = sorted(results, key=lambda r: -r["macro_f1"])

    best_feature_set = results_sorted[0]["feature_set"]
    logger.info(f"Feature set experiment ({evaluator.upper()}) completed. Best feature set: '{best_feature_set}'")

    experiment_report = {
        "experiment": f"Feature Set Dimension Comparison ({evaluator.upper()} Champion Evaluator)",
        "model_evaluator": f"{evaluator.upper()} with balanced sample weights",
        "best_feature_set": best_feature_set,
        "results": results_sorted,
        "top_features_ranked": [
            {"rank": i + 1, "feature": name, "importance": score}
            for i, (name, score, _) in enumerate(ranked_features[:30])
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(experiment_report, f, indent=2)

    # Also update master feature_set_comparison.json
    master_report_path = out_dir / "feature_set_comparison.json"
    with open(master_report_path, "w", encoding="utf-8") as f:
        json.dump(experiment_report, f, indent=2)

    logger.info(f"Saved feature set comparison report to: {report_path}")
    return experiment_report
