"""
SentinelOps - Phase 2 Test Suite
=================================
Unit and integration tests for Phase 2:
  - Class weighting and sample weighting logic
  - Random Forest, XGBoost, and MLP model training
  - Prediction shapes, types, and valid target class IDs
  - Model serialization and deserialization roundtrip
  - Evaluation metric computations (Accuracy, Precision, Recall, F1, FNR)
  - Strict FNR verification: FNR == 1.0 - Recall == FN / (TP + FN)
  - Zero-support class handling (Infiltration with 0 test samples -> None and insufficient_test_support=True)
  - Confusion matrix dimensions and properties
  - Model comparison and provisional champion selection logic
  - Controlled feature set selection and leakage prevention
  - SHAP TreeExplainer generation and output dimensions
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.models.evaluate import (
    compare_models,
    evaluate_model,
    recommend_champion,
    save_confusion_matrix_plot,
    save_metrics,
)
from src.models.explain import run_shap_analysis
from src.models.feature_experiment import compute_feature_importances
from src.models.train import (
    compute_class_weight_dict,
    compute_sample_weights,
    load_model,
    save_model,
    train_mlp,
    train_random_forest,
    train_xgboost,
)

# Canonical 8-class mapping for testing
TEST_CLASS_ENCODING = {
    "BENIGN": 0,
    "DoS": 1,
    "DDoS": 2,
    "PortScan": 3,
    "BruteForce": 4,
    "Botnet": 5,
    "WebAttack": 6,
    "Infiltration": 7,
}

TEST_PARAMS = {
    "general": {"random_seed": 42},
    "training": {
        "random_forest": {
            "n_estimators": 10,
            "max_depth": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "n_jobs": 1,
        },
        "xgboost": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "n_jobs": 1,
        },
        "mlp": {
            "hidden_layers": [32, 16],
            "max_epochs": 10,
            "batch_size": 64,
            "learning_rate": 0.01,
            "early_stopping_patience": 3,
        },
    },
}


@pytest.fixture
def synthetic_multiclass_data():
    """Generate synthetic 8-class imbalanced data for fast deterministic tests."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=8,
        n_clusters_per_class=1,
        weights=[0.60, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01],
        random_state=42,
    )
    feature_names = [f"Feature_{i}" for i in range(20)]
    return X.astype(np.float32), y.astype(np.int32), feature_names


# ---------------------------------------------------------------------------
# Test 1: Weight Calculation Tests
# ---------------------------------------------------------------------------

def test_compute_sample_weights(synthetic_multiclass_data):
    X, y, _ = synthetic_multiclass_data
    sample_weights = compute_sample_weights(y, method="balanced")

    assert len(sample_weights) == len(y)
    assert np.all(sample_weights > 0)
    assert np.all(np.isfinite(sample_weights))

    # Minority class samples should receive strictly higher weight than majority class samples
    majority_class = np.argmax(np.bincount(y))
    minority_class = np.argmin(np.bincount(y))

    majority_sample_idx = np.where(y == majority_class)[0][0]
    minority_sample_idx = np.where(y == minority_class)[0][0]

    assert sample_weights[minority_sample_idx] > sample_weights[majority_sample_idx]


def test_compute_class_weight_dict(synthetic_multiclass_data):
    _, y, _ = synthetic_multiclass_data
    class_weights = compute_class_weight_dict(y)

    assert isinstance(class_weights, dict)
    assert len(class_weights) == len(np.unique(y))
    for cls_id, weight in class_weights.items():
        assert isinstance(cls_id, int)
        assert weight > 0


# ---------------------------------------------------------------------------
# Test 2: Model Training & Prediction Tests
# ---------------------------------------------------------------------------

def test_train_random_forest(synthetic_multiclass_data):
    X, y, feats = synthetic_multiclass_data
    model, train_time = train_random_forest(X, y, TEST_PARAMS, feats)

    assert model is not None
    assert train_time > 0
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset(set(range(8)))


def test_train_xgboost(synthetic_multiclass_data):
    X, y, feats = synthetic_multiclass_data
    model, train_time = train_xgboost(X, y, TEST_PARAMS, feats)

    assert model is not None
    assert train_time > 0
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset(set(range(8)))


def test_train_mlp(synthetic_multiclass_data):
    X, y, feats = synthetic_multiclass_data
    model, train_time = train_mlp(X, y, TEST_PARAMS, feats)

    assert model is not None
    assert train_time > 0
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset(set(range(8)))


# ---------------------------------------------------------------------------
# Test 3: Model Serialization & Deserialization
# ---------------------------------------------------------------------------

def test_model_serialization_roundtrip(synthetic_multiclass_data):
    X, y, feats = synthetic_multiclass_data
    model, _ = train_random_forest(X, y, TEST_PARAMS, feats)

    with tempfile.TemporaryDirectory() as tmp_dir:
        meta = {"model_type": "rf_test", "seed": 42}
        model_path, meta_path = save_model(model, "test_rf", tmp_dir, meta)

        assert model_path.exists()
        assert meta_path.exists()

        # Load back and verify identical predictions
        loaded_model = load_model(model_path)
        orig_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)

        np.testing.assert_array_equal(orig_preds, loaded_preds)


# ---------------------------------------------------------------------------
# Test 4: Rigorous Evaluation Metrics & Zero-Support Handling
# ---------------------------------------------------------------------------

def test_evaluate_model_and_zero_support(synthetic_multiclass_data):
    X, y, _ = synthetic_multiclass_data
    X_train, y_train = X[:350], y[:350]
    X_test, y_test = X[350:], y[350:]

    # Force class 7 (Infiltration) to have 0 samples in test set
    zero_support_class_id = 7
    y_test[y_test == zero_support_class_id] = 0  # Reassign class 7 to BENIGN in test set

    model, train_time = train_random_forest(X_train, y_train, TEST_PARAMS)
    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_encoding=TEST_CLASS_ENCODING,
        model_name="test_rf",
        train_time=train_time,
    )

    summary = metrics["summary"]
    per_class = metrics["per_class"]

    # Verify overall metrics validity
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert 0.0 <= summary["macro_f1"] <= 1.0
    assert 0.0 <= summary["weighted_f1"] <= 1.0
    assert 0.0 <= summary["mean_fnr"] <= 1.0
    assert 0.0 <= summary["mean_attack_fnr"] <= 1.0

    # Verify Infiltration handling: support == 0 -> None / insufficient_test_support
    infil_metrics = per_class["Infiltration"]
    assert infil_metrics["support"] == 0
    assert infil_metrics["precision"] is None
    assert infil_metrics["recall"] is None
    assert infil_metrics["f1"] is None
    assert infil_metrics["fnr"] is None
    assert infil_metrics["insufficient_test_support"] is True

    # Verify valid classes satisfy FNR == 1.0 - Recall
    for cls_name, p in per_class.items():
        if not p["insufficient_test_support"]:
            assert p["recall"] is not None
            assert p["fnr"] is not None
            assert pytest.approx(p["fnr"] + p["recall"], abs=1e-5) == 1.0


def test_confusion_matrix_shape_and_plot(synthetic_multiclass_data):
    X, y, _ = synthetic_multiclass_data
    model, _ = train_random_forest(X, y, TEST_PARAMS)
    metrics = evaluate_model(
        model=model,
        X_test=X,
        y_test=y,
        class_encoding=TEST_CLASS_ENCODING,
        model_name="test_rf",
    )

    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (8, 8)
    assert np.sum(cm) == len(y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        plot_path = save_confusion_matrix_plot(
            cm=cm,
            class_names=list(TEST_CLASS_ENCODING.keys()),
            model_name="test_rf",
            reports_dir=tmp_dir,
        )
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Test 5: Model Comparison & Champion Selection
# ---------------------------------------------------------------------------

def test_model_comparison_and_champion():
    # Construct synthetic metrics for RF and XGBoost
    mock_metrics = {
        "random_forest": {
            "model_name": "random_forest",
            "test_samples": 1000,
            "summary": {
                "accuracy": 0.95,
                "macro_f1": 0.82,
                "macro_precision": 0.85,
                "macro_recall": 0.80,
                "weighted_f1": 0.94,
                "weighted_precision": 0.94,
                "weighted_recall": 0.95,
                "mean_fnr": 0.20,
                "mean_attack_fnr": 0.15,
                "macro_attack_f1": 0.81,
                "macro_attack_recall": 0.85,
                "mean_minority_f1": 0.70,
                "mean_minority_fnr": 0.25,
            },
            "per_class": {},
            "confusion_matrix": [],
            "timing": {
                "training_time_seconds": 5.0,
                "inference_time_seconds": 0.2,
                "inference_latency_ms_per_1000": 2.0,
            },
        },
        "xgboost": {
            "model_name": "xgboost",
            "test_samples": 1000,
            "summary": {
                "accuracy": 0.97,
                "macro_f1": 0.88,
                "macro_precision": 0.90,
                "macro_recall": 0.87,
                "weighted_f1": 0.96,
                "weighted_precision": 0.96,
                "weighted_recall": 0.97,
                "mean_fnr": 0.13,
                "mean_attack_fnr": 0.08,
                "macro_attack_f1": 0.89,
                "macro_attack_recall": 0.92,
                "mean_minority_f1": 0.82,
                "mean_minority_fnr": 0.14,
            },
            "per_class": {},
            "confusion_matrix": [],
            "timing": {
                "training_time_seconds": 8.0,
                "inference_time_seconds": 0.15,
                "inference_latency_ms_per_1000": 1.5,
            },
        },
    }

    comparison = compare_models(mock_metrics)
    assert len(comparison["comparison_table"]) == 2
    assert comparison["comparison_table"][0]["model"] == "xgboost"

    champ = recommend_champion(mock_metrics)
    assert champ["provisional_champion"] == "xgboost"
    assert "reasoning" in champ
    assert len(champ["reasoning"]) > 20


# ---------------------------------------------------------------------------
# Test 6: Feature Importances & Leakage-Free Top-K Selection
# ---------------------------------------------------------------------------

def test_feature_importances_ranking(synthetic_multiclass_data):
    X, y, feats = synthetic_multiclass_data
    ranked = compute_feature_importances(X, y, feats, random_seed=42)

    assert len(ranked) == len(feats)
    # Check descending order
    importances = [imp for _, imp, _ in ranked]
    assert importances == sorted(importances, reverse=True)
    assert all(0.0 <= imp <= 1.0 for imp in importances)


# ---------------------------------------------------------------------------
# Test 7: SHAP TreeExplainer Generation
# ---------------------------------------------------------------------------

def test_shap_analysis_execution(synthetic_multiclass_data):
    X, y, feats = synthetic_multiclass_data
    model, _ = train_xgboost(X, y, TEST_PARAMS, feats)

    with tempfile.TemporaryDirectory() as tmp_dir:
        shap_res = run_shap_analysis(
            model=model,
            X_test=X,
            y_test=y,
            feature_names=feats,
            class_encoding=TEST_CLASS_ENCODING,
            reports_dir=tmp_dir,
            artifact_prefix="test_xgb",
            sample_size=50,
            random_seed=42,
        )

        assert Path(shap_res["summary_plot"]).exists()
        assert Path(shap_res["local_examples_path"]).exists()
        assert len(shap_res["top_features"]) > 0


# ---------------------------------------------------------------------------
# Test 8: Final Top-40 Champion Specific Tests
# ---------------------------------------------------------------------------

def test_champion_top40_model_expects_40_features():
    """Verify that a synthetic 40-feature XGBoost champion trains and enforces 40 features."""
    X_40, y = make_classification(
        n_samples=200,
        n_features=40,
        n_informative=30,
        n_redundant=5,
        n_classes=8,
        random_state=42,
    )
    feature_names_40 = [f"Feature_{i}" for i in range(40)]
    model, train_time = train_xgboost(X_40, y, TEST_PARAMS, feature_names_40)

    assert model is not None
    assert train_time > 0
    assert model.n_features_in_ == 40

    preds = model.predict(X_40)
    assert preds.shape == (200,)
    assert set(np.unique(preds)).issubset(set(range(8)))


def test_champion_top40_serialization_and_metadata():
    """Verify Top-40 Champion serialization, feature list, and metadata integrity."""
    X_40, y = make_classification(
        n_samples=100,
        n_features=40,
        n_informative=30,
        n_redundant=5,
        n_classes=8,
        random_state=42,
    )
    feature_names_40 = [f"Feature_{i}" for i in range(40)]
    model, _ = train_xgboost(X_40, y, TEST_PARAMS, feature_names_40)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        meta = {
            "model_name": "xgboost_top40",
            "feature_count": 40,
            "ordered_feature_names": feature_names_40,
            "status": "provisional_champion",
        }
        # Save model and metadata
        save_model(model, "xgboost_top40", tmp_path, meta)
        # Save features
        with open(tmp_path / "xgboost_top40_features.json", "w") as f:
            json.dump(feature_names_40, f)

        # Test loading via load_champion_model
        from src.models.train import load_champion_model
        loaded_model, loaded_feats, loaded_meta = load_champion_model(tmp_path)

        assert loaded_model.n_features_in_ == 40
        assert len(loaded_feats) == 40
        assert loaded_feats == feature_names_40
        assert loaded_meta["feature_count"] == 40


def test_predict_with_champion_feature_order_consistency():
    """Verify that predict_with_champion enforces strict column ordering from DataFrame or ndarray."""
    import pandas as pd
    from src.models.train import predict_with_champion

    X_40, y = make_classification(
        n_samples=50,
        n_features=40,
        n_informative=30,
        n_redundant=5,
        n_classes=8,
        random_state=42,
    )
    feature_names_40 = [f"Feat_{i}" for i in range(40)]
    model, _ = train_xgboost(X_40, y, TEST_PARAMS, feature_names_40)

    # 1. Direct array prediction
    preds_direct = predict_with_champion(model, X_40)
    assert len(preds_direct) == 50

    # 2. DataFrame with permuted columns -> should align automatically
    df_permuted = pd.DataFrame(X_40, columns=feature_names_40)
    # Shuffle columns
    shuffled_cols = list(reversed(feature_names_40))
    df_shuffled = df_permuted[shuffled_cols]

    preds_df = predict_with_champion(model, df_shuffled, champion_features=feature_names_40)
    np.testing.assert_array_equal(preds_direct, preds_df)

    # 3. Missing column in DataFrame -> should raise ValueError
    df_incomplete = df_permuted.drop(columns=[feature_names_40[0]])
    with pytest.raises(ValueError, match="missing required champion features"):
        predict_with_champion(model, df_incomplete, champion_features=feature_names_40)
