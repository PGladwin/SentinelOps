"""
SentinelOps - Models Package
============================
Phase 2: Feature Engineering & Model Training
"""

from src.models.train import (
    train_random_forest,
    train_xgboost,
    train_mlp,
    compute_sample_weights,
    compute_class_weight_dict,
    save_model,
    load_model,
    load_champion_model,
    predict_with_champion,
    load_phase1_data,
)
from src.models.evaluate import (
    evaluate_model,
    save_metrics,
    save_confusion_matrix_plot,
    compare_models,
    recommend_champion,
)

__all__ = [
    "train_random_forest",
    "train_xgboost",
    "train_mlp",
    "compute_sample_weights",
    "compute_class_weight_dict",
    "save_model",
    "load_model",
    "load_champion_model",
    "predict_with_champion",
    "load_phase1_data",
    "evaluate_model",
    "save_metrics",
    "save_confusion_matrix_plot",
    "compare_models",
    "recommend_champion",
]
