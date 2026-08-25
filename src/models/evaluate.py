"""
SentinelOps - Model Evaluation Module
======================================
Phase 2: Rigorous, security-focused evaluation for multiclass intrusion detection.

Metrics computed:
  - Accuracy, Macro/Weighted Precision, Recall, F1
  - Per-class Precision, Recall, F1, Support, and False Negative Rate (FNR = 1 - Recall)
  - Explicit detection and handling of classes with zero test support (Infiltration)
  - Security-focused aggregates: Mean FNR, Mean Attack FNR, Macro Attack F1
  - Confusion matrix generation and visualization
  - Training and inference latency benchmarking
  - Side-by-side model comparison and provisional champion recommendation
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# Standard 8-class taxonomy
CANONICAL_CLASSES = [
    "BENIGN",
    "DoS",
    "DDoS",
    "PortScan",
    "BruteForce",
    "Botnet",
    "WebAttack",
    "Infiltration",
]


# ---------------------------------------------------------------------------
# Core Evaluation Function
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_encoding: dict[str, int],
    model_name: str = "model",
    train_time: float = 0.0,
) -> dict:
    """
    Perform comprehensive evaluation on the test split.

    Parameters
    ----------
    model : Any
        Trained classifier implementing .predict().
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True integer labels.
    class_encoding : dict[str, int]
        Mapping of canonical class names to integers.
    model_name : str
        Name identifier for the model.
    train_time : float
        Training duration in seconds.

    Returns
    -------
    metrics : dict
        Complete evaluation metrics dictionary.
    """
    logger.info(f"Evaluating {model_name} on {len(X_test):,} test samples...")

    # Benchmark inference time
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - t0
    latency_ms_per_1000 = (inference_time / max(1, len(X_test))) * 1000.0 * 1000.0

    inverse_encoding = {v: k for k, v in class_encoding.items()}
    num_classes = len(class_encoding)
    all_class_indices = list(range(num_classes))

    # Compute full confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=all_class_indices)

    # Calculate per-class metrics with zero-support awareness
    per_class_metrics = {}
    valid_recalls = []
    valid_fnrs = []
    valid_f1s = []
    attack_fnrs = []
    attack_f1s = []
    attack_recalls = []
    minority_f1s = []
    minority_fnrs = []

    for idx in all_class_indices:
        cls_name = inverse_encoding.get(idx, f"Class_{idx}")
        tp = int(cm[idx, idx])
        fn = int(np.sum(cm[idx, :]) - tp)
        fp = int(np.sum(cm[:, idx]) - tp)
        tn = int(np.sum(cm) - tp - fn - fp)
        support = tp + fn

        if support == 0:
            # Explicitly mark zero test support (e.g. Infiltration in dev sample)
            per_class_metrics[cls_name] = {
                "class_id": idx,
                "support": 0,
                "precision": None,
                "recall": None,
                "f1": None,
                "fnr": None,
                "tp": 0,
                "fn": 0,
                "fp": fp,
                "tn": tn,
                "insufficient_test_support": True,
                "note": "Zero test samples present in evaluation split.",
            }
        else:
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn))
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            fnr = float(fn / (tp + fn))  # Equivalent to 1.0 - recall

            per_class_metrics[cls_name] = {
                "class_id": idx,
                "support": support,
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "f1": round(f1, 6),
                "fnr": round(fnr, 6),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "insufficient_test_support": False,
            }

            valid_recalls.append(rec)
            valid_fnrs.append(fnr)
            valid_f1s.append(f1)

            # Attack classes (all non-BENIGN classes with support > 0)
            if cls_name != "BENIGN":
                attack_fnrs.append(fnr)
                attack_f1s.append(f1)
                attack_recalls.append(rec)
                if support < 500:
                    minority_f1s.append(f1)
                    minority_fnrs.append(fnr)

    # Standard global metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Security-centric aggregate metrics
    mean_fnr = float(np.mean(valid_fnrs)) if valid_fnrs else 0.0
    mean_attack_fnr = float(np.mean(attack_fnrs)) if attack_fnrs else 0.0
    macro_attack_f1 = float(np.mean(attack_f1s)) if attack_f1s else 0.0
    macro_attack_recall = float(np.mean(attack_recalls)) if attack_recalls else 0.0
    mean_minority_f1 = float(np.mean(minority_f1s)) if minority_f1s else 0.0
    mean_minority_fnr = float(np.mean(minority_fnrs)) if minority_fnrs else 0.0

    metrics = {
        "model_name": model_name,
        "test_samples": int(len(y_test)),
        "summary": {
            "accuracy": round(acc, 6),
            "macro_f1": round(macro_f1, 6),
            "macro_precision": round(macro_prec, 6),
            "macro_recall": round(macro_rec, 6),
            "weighted_f1": round(weighted_f1, 6),
            "weighted_precision": round(weighted_prec, 6),
            "weighted_recall": round(weighted_rec, 6),
            "mean_fnr": round(mean_fnr, 6),
            "mean_attack_fnr": round(mean_attack_fnr, 6),
            "macro_attack_f1": round(macro_attack_f1, 6),
            "macro_attack_recall": round(macro_attack_recall, 6),
            "mean_minority_f1": round(mean_minority_f1, 6),
            "mean_minority_fnr": round(mean_minority_fnr, 6),
        },
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "timing": {
            "training_time_seconds": round(train_time, 4),
            "inference_time_seconds": round(inference_time, 4),
            "inference_latency_ms_per_1000": round(latency_ms_per_1000, 4),
        },
    }

    logger.info(
        f"{model_name} Results -> Accuracy: {acc*100:.2f}%, Macro F1: {macro_f1:.4f}, "
        f"Mean Attack FNR: {mean_attack_fnr:.4f}, Macro Attack F1: {macro_attack_f1:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Plotting & Reporting
# ---------------------------------------------------------------------------

def save_confusion_matrix_plot(
    cm: list[list[int]] | np.ndarray,
    class_names: list[str],
    model_name: str,
    reports_dir: str | Path,
) -> Path:
    """
    Generate and save a styled confusion matrix heatmap plot.

    Parameters
    ----------
    cm : 2D array / list of lists
    class_names : list[str]
    model_name : str
    reports_dir : str | Path

    Returns
    -------
    plot_path : Path
    """
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"confusion_matrix_{model_name}.png"

    cm_arr = np.array(cm)
    num_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    cax = ax.matshow(cm_arr, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=20)

    # Annotate counts inside cells
    thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_arr[i, j]
            color = "white" if val > thresh else "black"
            ax.text(
                j,
                i,
                f"{val:,}" if val > 0 else "0",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Confusion matrix plot saved: {plot_path}")
    return plot_path


def save_metrics(metrics: dict, model_name: str, metrics_dir: str | Path) -> Path:
    """Save evaluation metrics to JSON."""
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{model_name}_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved metrics for {model_name} to: {metrics_path}")
    return metrics_path


# ---------------------------------------------------------------------------
# Model Comparison & Champion Recommendation
# ---------------------------------------------------------------------------

def compare_models(all_metrics: dict[str, dict]) -> dict:
    """
    Compare multiple models across security and operational metrics.

    Parameters
    ----------
    all_metrics : dict[str, dict]
        Dictionary mapping model_name -> metrics dict.

    Returns
    -------
    comparison : dict
        Comparative summary table and rankings.
    """
    rows = []
    for name, m in all_metrics.items():
        summary = m["summary"]
        timing = m["timing"]
        rows.append({
            "model": name,
            "macro_f1": summary["macro_f1"],
            "mean_attack_fnr": summary["mean_attack_fnr"],
            "macro_attack_f1": summary["macro_attack_f1"],
            "macro_attack_recall": summary["macro_attack_recall"],
            "mean_fnr": summary["mean_fnr"],
            "weighted_f1": summary["weighted_f1"],
            "accuracy": summary["accuracy"],
            "mean_minority_f1": summary["mean_minority_f1"],
            "mean_minority_fnr": summary["mean_minority_fnr"],
            "train_time_sec": timing["training_time_seconds"],
            "inference_latency_ms": timing["inference_latency_ms_per_1000"],
        })

    # Sort primarily by security criteria: High Macro F1, Low Mean Attack FNR
    rows_sorted = sorted(
        rows,
        key=lambda r: (-r["macro_f1"], r["mean_attack_fnr"], -r["macro_attack_f1"])
    )

    return {
        "models_evaluated": list(all_metrics.keys()),
        "comparison_table": rows_sorted,
        "security_metric_rankings": {
            "by_macro_f1": sorted([r["model"] for r in rows], key=lambda x: -all_metrics[x]["summary"]["macro_f1"]),
            "by_lowest_attack_fnr": sorted([r["model"] for r in rows], key=lambda x: all_metrics[x]["summary"]["mean_attack_fnr"]),
            "by_macro_attack_f1": sorted([r["model"] for r in rows], key=lambda x: -all_metrics[x]["summary"]["macro_attack_f1"]),
            "by_lowest_latency": sorted([r["model"] for r in rows], key=lambda x: all_metrics[x]["timing"]["inference_latency_ms_per_1000"]),
        },
    }


def recommend_champion(all_metrics: dict[str, dict]) -> dict:
    """
    Produce a provisional Champion recommendation based on security-first evidence.

    Security criteria prioritization:
      1. Macro F1 & Macro Attack F1 (detecting intrusions across minority classes)
      2. Lowest Mean Attack FNR (minimizing missed cyber threats)
      3. Balance against operational latency and false alarms

    Returns
    -------
    recommendation : dict
    """
    comparison = compare_models(all_metrics)
    table = comparison["comparison_table"]

    if not table:
        return {"provisional_champion": None, "reasoning": "No models evaluated."}

    best_candidate = table[0]
    champion_name = best_candidate["model"]
    champ_metrics = all_metrics[champion_name]["summary"]

    reasons = [
        f"Selected '{champion_name}' as provisional champion based on multi-criteria security analysis.",
        f"Macro F1 score of {champ_metrics['macro_f1']:.4f} across 8 classes.",
        f"Mean Attack FNR of {champ_metrics['mean_attack_fnr']:.4f} (missed intrusion rate).",
        f"Macro Attack F1 of {champ_metrics['macro_attack_f1']:.4f} on attack traffic.",
        f"Weighted F1 of {champ_metrics['weighted_f1']:.4f} and overall Accuracy of {champ_metrics['accuracy']*100:.2f}%.",
        f"Inference latency: {all_metrics[champion_name]['timing']['inference_latency_ms_per_1000']:.2f} ms / 1,000 samples.",
    ]

    return {
        "provisional_champion": champion_name,
        "champion_summary": champ_metrics,
        "champion_timing": all_metrics[champion_name]["timing"],
        "reasoning": " ".join(reasons),
        "comparison_table": table,
        "note": "Provisional recommendation for development phase. Formal champion-challenger promotion occurs in Governance Phase.",
    }


def print_evaluation_summary(metrics: dict) -> None:
    """Pretty-print evaluation metrics for a single model."""
    name = metrics["model_name"]
    s = metrics["summary"]
    t = metrics["timing"]

    print("\n" + "=" * 70)
    print(f"EVALUATION REPORT: {name.upper()}")
    print("=" * 70)
    print(f"  Test samples          : {metrics['test_samples']:,}")
    print(f"  Overall Accuracy      : {s['accuracy']*100:.3f}%")
    print(f"  Macro F1              : {s['macro_f1']:.4f}")
    print(f"  Macro Precision/Recall: {s['macro_precision']:.4f} / {s['macro_recall']:.4f}")
    print(f"  Weighted F1           : {s['weighted_f1']:.4f}")
    print(f"  Mean FNR (All classes): {s['mean_fnr']:.4f}")
    print(f"  Mean Attack FNR       : {s['mean_attack_fnr']:.4f} (Crucial for IDS)")
    print(f"  Macro Attack F1       : {s['macro_attack_f1']:.4f}")
    print(f"  Training Time         : {t['training_time_seconds']:.2f}s")
    print(f"  Inference Latency     : {t['inference_latency_ms_per_1000']:.2f} ms / 1k flows")

    print("\n--- Per-Class Performance Breakdown ---")
    header = f"  {'Class':<14} | {'Support':>7} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'FNR (Miss)':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cls_name, p in metrics["per_class"].items():
        if p["insufficient_test_support"]:
            print(f"  {cls_name:<14} | {p['support']:>7} |       N/A |       N/A |       N/A |       N/A  (No Test Support)")
        else:
            print(
                f"  {cls_name:<14} | {p['support']:>7,} | {p['precision']:>9.4f} | {p['recall']:>9.4f} | "
                f"{p['f1']:>9.4f} | {p['fnr']:>10.4f}"
            )
    print("=" * 70)
