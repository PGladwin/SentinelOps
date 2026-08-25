"""
SentinelOps - Explainability Module
====================================
Phase 2: SHAP (SHapley Additive exPlanations) analysis for XGBoost.

Responsibilities:
  - Generate true TreeExplainer SHAP values on representative test samples
  - Produce global feature importance summary plot (reports/shap_summary_xgboost.png)
  - Generate local explanations for representative attack and benign samples
  - Handle multiclass XGBoost output safely across shap versions
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

logger = logging.getLogger(__name__)


def run_shap_analysis(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    class_encoding: dict[str, int],
    reports_dir: str | Path = "reports",
    artifact_prefix: str = "xgboost",
    title_suffix: str = "",
    sample_size: int = 1000,
    random_seed: int = 42,
) -> dict:
    """
    Run SHAP TreeExplainer analysis on XGBoost multiclass model.

    Parameters
    ----------
    model : trained XGBoost model
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test integer labels.
    feature_names : list[str]
        Ordered feature column names.
    class_encoding : dict[str, int]
        Canonical class mapping.
    reports_dir : str | Path
        Directory to save SHAP artifacts.
    artifact_prefix : str
        Prefix for output filenames (e.g., 'xgboost' -> 'shap_summary_xgboost.png').
    title_suffix : str
        Additional suffix for plot title.
    sample_size : int
        Number of test samples for SHAP analysis (default: 1000 for efficiency).
    random_seed : int
        Reproducibility seed.

    Returns
    -------
    shap_results : dict
        Summary metrics and local explanation examples.
    """
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_plot_path = out_dir / f"shap_summary_{artifact_prefix}.png"
    local_examples_path = out_dir / f"shap_local_examples_{artifact_prefix}.json"

    logger.info(f"Running SHAP TreeExplainer ({artifact_prefix}) on {min(sample_size, len(X_test)):,} test samples...")

    # Subsample test data deterministically for explanation
    rng = np.random.RandomState(random_seed)
    n_samples = min(sample_size, len(X_test))
    sample_indices = rng.choice(len(X_test), size=n_samples, replace=False)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]

    inverse_encoding = {v: k for k, v in class_encoding.items()}

    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # shap_values.values has shape (n_samples, n_features, n_classes) for multiclass
    raw_vals = shap_values.values
    if raw_vals.ndim == 3:
        # Multiclass: compute mean absolute SHAP value across all samples and classes
        mean_abs_shap_per_feature = np.mean(np.abs(raw_vals), axis=(0, 2))
        top_k_indices = np.argsort(mean_abs_shap_per_feature)[::-1][:20]
    elif raw_vals.ndim == 2:
        # Binary / single output
        mean_abs_shap_per_feature = np.mean(np.abs(raw_vals), axis=0)
        top_k_indices = np.argsort(mean_abs_shap_per_feature)[::-1][:20]
    else:
        mean_abs_shap_per_feature = np.zeros(len(feature_names))
        top_k_indices = np.arange(min(20, len(feature_names)))

    # Global Feature Ranking
    top_features = [
        {
            "rank": int(rank + 1),
            "feature": feature_names[idx],
            "mean_abs_shap": float(mean_abs_shap_per_feature[idx]),
        }
        for rank, idx in enumerate(top_k_indices)
    ]

    # Generate and save global SHAP summary bar chart
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    top_feats_ordered = [feature_names[i] for i in top_k_indices[::-1]]
    top_vals_ordered = [mean_abs_shap_per_feature[i] for i in top_k_indices[::-1]]

    bars = ax.barh(range(len(top_feats_ordered)), top_vals_ordered, color="#1f77b4", edgecolor="none")
    ax.set_yticks(range(len(top_feats_ordered)))
    ax.set_yticklabels(top_feats_ordered, fontsize=9)
    ax.set_xlabel("Mean |SHAP Value| (Average impact on model output magnitude)", fontsize=10, fontweight="bold")
    ax.set_title("XGBoost Global Feature Importance (SHAP TreeExplainer)", fontsize=12, fontweight="bold", pad=15)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(summary_plot_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SHAP summary plot saved to: {summary_plot_path}")

    # Generate Local Explanations for Representative Samples
    # Find 1 BENIGN (class 0), 1 DoS/DDoS (class 1 or 2), 1 Minority attack (class 3, 4, 5, or 6)
    local_examples = []
    target_sample_types = [
        {"desc": "BENIGN Baseline Traffic", "class_id": 0},
        {"desc": "High-Volume Attack (DoS/DDoS)", "class_ids": [1, 2]},
        {"desc": "Targeted Attack (BruteForce/WebAttack/PortScan)", "class_ids": [3, 4, 5, 6]},
    ]

    preds = model.predict(X_sample)

    for target in target_sample_types:
        found_idx = None
        if "class_id" in target:
            matches = np.where((y_sample == target["class_id"]) & (preds == target["class_id"]))[0]
            if len(matches) > 0:
                found_idx = matches[0]
        else:
            for cid in target["class_ids"]:
                matches = np.where((y_sample == cid) & (preds == cid))[0]
                if len(matches) > 0:
                    found_idx = matches[0]
                    break

        if found_idx is not None:
            true_cls = inverse_encoding.get(int(y_sample[found_idx]), f"Class_{y_sample[found_idx]}")
            pred_cls = inverse_encoding.get(int(preds[found_idx]), f"Class_{preds[found_idx]}")
            pred_cls_id = int(preds[found_idx])

            # Extract SHAP contributions for the predicted class
            if raw_vals.ndim == 3:
                sample_shap = raw_vals[found_idx, :, pred_cls_id]
            else:
                sample_shap = raw_vals[found_idx, :]

            top_contrib_indices = np.argsort(np.abs(sample_shap))[::-1][:5]

            # SHAP values are signed relative to the PREDICTED class. When the
            # prediction is BENIGN a positive value argues for benign traffic
            # and therefore LOWERS threat, so the sign alone cannot be read as
            # "increases risk" without accounting for which class was predicted.
            is_attack_pred = pred_cls != "BENIGN"

            contributions = []
            for ci in top_contrib_indices:
                s_val = float(sample_shap[ci])
                raises_threat = s_val > 0 if is_attack_pred else s_val < 0
                contributions.append({
                    "feature": feature_names[ci],
                    "feature_value": float(X_sample[found_idx, ci]),
                    "shap_contribution": s_val,
                    "increases_threat": raises_threat,
                    "direction": "increases threat" if raises_threat else "decreases threat",
                })

            local_examples.append({
                "sample_type": target["desc"],
                "true_class": true_cls,
                "predicted_class": pred_cls,
                "top_5_feature_contributions": contributions,
            })

    # Save local explanation artifact
    with open(local_examples_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "XGBoost",
                "explainer": "TreeExplainer",
                "sample_size": n_samples,
                "local_examples": local_examples,
            },
            f,
            indent=2,
        )

    logger.info(f"SHAP local explanations saved to: {local_examples_path}")

    return {
        "summary_plot": str(summary_plot_path),
        "local_examples_path": str(local_examples_path),
        "top_features": top_features,
        "local_examples": local_examples,
    }
