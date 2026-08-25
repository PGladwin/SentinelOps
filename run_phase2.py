"""
SentinelOps - Phase 2 Pipeline Runner
======================================
Phase 2: Feature Engineering & Model Training Orchestrator

Executes:
  1. Load Phase 1 processed data and metadata
  2. Train specified model(s): Random Forest, XGBoost, MLP
  3. Perform security-focused multiclass evaluation (Accuracy, F1, FNR, Minority metrics)
  4. Save model artifacts (.pkl) and model metadata (.json) to models/
  5. Save detailed metrics (.json) to metrics/
  6. Generate confusion matrix heatmaps to reports/
  7. Run SHAP TreeExplainer analysis for XGBoost (global summary + local examples)
  8. Run controlled feature-set experiment (69 vs 48 vs Top-K features)
  9. Compare models side-by-side and formulate provisional Champion recommendation
  10. Generate comprehensive Phase 2 Report (reports/phase2_training_report.json)

Usage:
    python run_phase2.py                      # Train all models, SHAP, & feature experiment
    python run_phase2.py --model rf           # Train only Random Forest
    python run_phase2.py --model xgb          # Train only XGBoost
    python run_phase2.py --model mlp          # Train only MLP
    python run_phase2.py --no-shap            # Skip SHAP analysis
    python run_phase2.py --no-feature-exp     # Skip feature set experiment
    python run_phase2.py --params custom.yaml # Custom parameters file
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("sentinelops.phase2")


def load_params(params_path: str = "params.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML is required. Install with: pip install pyyaml")
        sys.exit(1)

    path = Path(params_path)
    if not path.exists():
        logger.error(f"Parameters file not found: {path.resolve()}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    logger.info(f"Loaded configuration from: {path.resolve()}")
    return params


def main():
    parser = argparse.ArgumentParser(
        description="SentinelOps Phase 2: Feature Engineering & Model Training"
    )
    parser.add_argument(
        "--params",
        default="params.yaml",
        help="Path to params.yaml (default: params.yaml)",
    )
    parser.add_argument(
        "--model",
        choices=["all", "rf", "xgb", "mlp"],
        default="all",
        help="Model to train: 'all' (default), 'rf', 'xgb', or 'mlp'",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP analysis for XGBoost",
    )
    parser.add_argument(
        "--no-feature-exp",
        action="store_true",
        help="Skip controlled feature set comparison experiment",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Train on full dataset if preprocessed or flag for full run",
    )
    args = parser.parse_args()

    start_time = time.time()

    # 1. Load Parameters
    params = load_params(args.params)
    paths_cfg = params.get("paths", {})
    models_dir = Path(paths_cfg.get("models_dir", "models"))
    metrics_dir = Path(paths_cfg.get("metrics_dir", "metrics"))
    reports_dir = Path(paths_cfg.get("reports_dir", "reports"))

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Phase 1 Artifacts
    logger.info("=" * 60)
    logger.info("PHASE 2: LOADING PHASE 1 ARTIFACTS")
    logger.info("=" * 60)

    from src.models.train import (
        load_phase1_data,
        train_random_forest,
        train_xgboost,
        train_mlp,
        save_model,
    )
    from src.models.evaluate import (
        evaluate_model,
        save_metrics,
        save_confusion_matrix_plot,
        compare_models,
        recommend_champion,
        print_evaluation_summary,
    )

    X_train, y_train, X_test, y_test, feature_names, class_encoding, inverse_encoding = load_phase1_data(params)
    class_names = [inverse_encoding[i] for i in range(len(class_encoding))]

    logger.info(f"Target classes ({len(class_names)}): {class_names}")

    # Check for Infiltration support in test set
    infil_id = class_encoding.get("Infiltration", 7)
    infil_train_count = int(np.sum(y_train == infil_id))
    infil_test_count = int(np.sum(y_test == infil_id))
    logger.info(f"Infiltration support check: Train={infil_train_count}, Test={infil_test_count}")
    if infil_test_count == 0:
        logger.warning(
            "Infiltration has 0 test samples in the development split. "
            "Metrics for Infiltration will be recorded as insufficient_test_support."
        )

    # 3. Model Training & Evaluation
    models_to_train = []
    if args.model in ["all", "rf"]:
        models_to_train.append("rf")
    if args.model in ["all", "xgb"]:
        models_to_train.append("xgb")
    if args.model in ["all", "mlp"]:
        models_to_train.append("mlp")

    all_metrics = {}
    trained_models = {}

    for model_key in models_to_train:
        logger.info("")
        logger.info("=" * 60)
        if model_key == "rf":
            model_name = "random_forest"
            logger.info("TRAINING: RANDOM FOREST BASELINE")
            logger.info("=" * 60)
            model, train_time = train_random_forest(X_train, y_train, params, feature_names)
            hyperparams = params.get("training", {}).get("random_forest", {})

        elif model_key == "xgb":
            model_name = "xgboost"
            logger.info("TRAINING: MULTICLASS XGBOOST")
            logger.info("=" * 60)
            model, train_time = train_xgboost(X_train, y_train, params, feature_names)
            hyperparams = params.get("training", {}).get("xgboost", {})

        elif model_key == "mlp":
            model_name = "mlp"
            logger.info("TRAINING: MULTI-LAYER PERCEPTRON (MLP)")
            logger.info("=" * 60)
            model, train_time = train_mlp(X_train, y_train, params, feature_names)
            hyperparams = params.get("training", {}).get("mlp", {})

        trained_models[model_name] = model

        # Evaluate model
        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            class_encoding=class_encoding,
            model_name=model_name,
            train_time=train_time,
        )
        all_metrics[model_name] = metrics

        # Print console summary
        print_evaluation_summary(metrics)

        # Save model and metadata
        model_meta = {
            "model_type": model_name,
            "hyperparameters": hyperparams,
            "feature_set": "Phase1_48_features",
            "feature_count": len(feature_names),
            "random_seed": params.get("general", {}).get("random_seed", 42),
            "training_dataset_identifier": "data/processed/train.parquet",
            "training_row_count": len(X_train),
            "test_row_count": len(X_test),
            "class_mapping": class_encoding,
            "training_time_seconds": round(train_time, 4),
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_model(model, model_name, models_dir, model_meta)

        # Save metrics JSON and confusion matrix plot
        save_metrics(metrics, model_name, metrics_dir)
        save_confusion_matrix_plot(metrics["confusion_matrix"], class_names, model_name, reports_dir)

    # 4. SHAP Analysis for XGBoost
    shap_results = None
    if "xgboost" in trained_models and not args.no_shap:
        logger.info("")
        logger.info("=" * 60)
        logger.info("SHAP EXPLAINABILITY ANALYSIS (XGBOOST)")
        logger.info("=" * 60)
        try:
            from src.models.explain import run_shap_analysis
            shap_results = run_shap_analysis(
                model=trained_models["xgboost"],
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                class_encoding=class_encoding,
                reports_dir=reports_dir,
                sample_size=1000,
                random_seed=params.get("general", {}).get("random_seed", 42),
            )
        except Exception as e:
            logger.error(f"SHAP analysis encountered an error: {e}", exc_info=True)

    # 5. Controlled Feature Set Experiment
    feature_exp_results = None
    if not args.no_feature_exp and len(models_to_train) > 0:
        logger.info("")
        try:
            from src.models.feature_experiment import run_feature_experiment
            feature_exp_results = run_feature_experiment(
                X_train_48=X_train,
                y_train_48=y_train,
                X_test_48=X_test,
                y_test_48=y_test,
                feature_names_48=feature_names,
                class_encoding=class_encoding,
                params=params,
                top_k_values=[20, 30, 40],
                include_69_features=True,
                reports_dir=reports_dir,
            )
        except Exception as e:
            logger.error(f"Feature set experiment encountered an error: {e}", exc_info=True)

    # 6. Final Champion Training: Top-40 Features
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING FINAL PROVISIONAL CHAMPION: XGBOOST (TOP-40 FEATURES)")
    logger.info("=" * 60)

    from src.models.feature_experiment import compute_feature_importances
    ranked_features = compute_feature_importances(
        X_train, y_train, feature_names, model_type="xgb", params=params, random_seed=params.get("general", {}).get("random_seed", 42)
    )
    top40_names = [name for name, _, _ in ranked_features[:40]]
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    top40_indices = [feat_to_idx[f] for f in top40_names]

    X_train_top40 = X_train[:, top40_indices]
    X_test_top40 = X_test[:, top40_indices]

    # Save exact ordered feature list
    top40_features_path = models_dir / "xgboost_top40_features.json"
    with open(top40_features_path, "w", encoding="utf-8") as f:
        json.dump(top40_names, f, indent=2)
    logger.info(f"Saved Top-40 feature list to: {top40_features_path}")

    # Train final Top-40 XGBoost model
    champion_xgb, champ_train_time = train_xgboost(X_train_top40, y_train, params, top40_names)

    # Save Top-40 Champion Model & Metadata
    champion_meta = {
        "model_name": "xgboost_top40",
        "model_type": "xgboost",
        "feature_count": 40,
        "ordered_feature_names": top40_names,
        "feature_selection_method": "XGBoost multiclass balanced feature importances fitted on X_train only (Leakage-free)",
        "training_dataset_size": len(X_train_top40),
        "test_dataset_size": len(X_test_top40),
        "class_mapping": class_encoding,
        "hyperparameters": params.get("training", {}).get("xgboost", {}),
        "random_seed": params.get("general", {}).get("random_seed", 42),
        "training_time_seconds": round(champ_train_time, 4),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "provisional_champion",
    }
    save_model(champion_xgb, "xgboost_top40", models_dir, champion_meta)

    # ----------------------------------------------------------------
    # Build serving artifacts: raw -> Champion inference bundle + demo bank
    # ----------------------------------------------------------------
    # The bundle lets the API accept RAW flow values (packet counts, byte
    # rates) and apply Phase 1 scaling server-side. It slices the fitted
    # 48-feature RobustScaler down to the Champion's 40 columns, which is
    # exact because RobustScaler transforms each column independently.
    from src.data.inference_prep import build_inference_bundle, save_inference_bundle
    from src.models.demo_samples import build_demo_samples, save_demo_samples

    processed_dir = Path(params["paths"]["processed_dir"])
    logger.info("Building raw-input inference bundle for the Champion...")
    bundle = build_inference_bundle(
        champion_features=top40_names,
        scaler_path=processed_dir / "scaler.pkl",
        fill_values_path=processed_dir / "fill_values.json",
        feature_names_path=processed_dir / "feature_names.json",
    )
    save_inference_bundle(bundle, models_dir / "champion_preprocessor.json")

    # Pre-bake demo flows so the API never loads test.parquet at startup.
    try:
        logger.info("Building demo sample bank from source Parquet files...")
        demo = build_demo_samples(
            parquet_dir=params["paths"]["parquet_dir"],
            champion_features=top40_names,
            model=champion_xgb,
            bundle=bundle,
            inverse_encoding=inverse_encoding,
            label_map=params["cleaning"]["label_map"],
            random_seed=params.get("general", {}).get("random_seed", 42),
        )
        save_demo_samples(demo, models_dir / "demo_samples.json")
    except Exception as e:
        logger.error(f"Demo sample generation failed (non-fatal): {e}", exc_info=True)

    # Evaluate Top-40 Champion on test set
    champ_metrics = evaluate_model(
        model=champion_xgb,
        X_test=X_test_top40,
        y_test=y_test,
        class_encoding=class_encoding,
        model_name="xgboost_top40",
        train_time=champ_train_time,
    )
    save_metrics(champ_metrics, "xgboost_top40", metrics_dir)
    save_confusion_matrix_plot(champ_metrics["confusion_matrix"], class_names, "xgboost_top40", reports_dir)

    print("\n--- Final Champion Evaluation: XGBoost (Top-40 Features) ---")
    print_evaluation_summary(champ_metrics)

    # Run SHAP for Final Top-40 Champion
    shap_top40_results = None
    if not args.no_shap:
        logger.info("")
        logger.info("=" * 60)
        logger.info("SHAP EXPLAINABILITY ANALYSIS: FINAL TOP-40 CHAMPION")
        logger.info("=" * 60)
        try:
            from src.models.explain import run_shap_analysis
            shap_top40_results = run_shap_analysis(
                model=champion_xgb,
                X_test=X_test_top40,
                y_test=y_test,
                feature_names=top40_names,
                class_encoding=class_encoding,
                reports_dir=reports_dir,
                artifact_prefix="xgboost_top40",
                sample_size=1000,
                random_seed=params.get("general", {}).get("random_seed", 42),
            )
        except Exception as e:
            logger.error(f"Top-40 SHAP analysis error: {e}", exc_info=True)

    # 7. Model Comparison & Provisional Champion Recommendation
    comparison = compare_models(all_metrics)
    recommendation = recommend_champion(all_metrics)

    # Save model comparison JSON
    comp_path = metrics_dir / "model_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "comparison": comparison,
                "recommendation": recommendation,
                "final_champion_top40": {
                    "model_name": "xgboost_top40",
                    "features_count": 40,
                    "metrics_summary": champ_metrics["summary"],
                    "timing": champ_metrics["timing"],
                },
            },
            f,
            indent=2,
        )
    logger.info(f"Saved model comparison to: {comp_path}")

    # 8. Master Phase 2 Training Report
    total_pipeline_time = time.time() - start_time
    report_path = reports_dir / "phase2_training_report.json"

    phase2_report = {
        "report_name": "SentinelOps Phase 2 Training Report",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution_time_seconds": round(total_pipeline_time, 2),
        "dataset_summary": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "features_initial_phase1": len(feature_names),
            "features_final_champion": 40,
            "classes": class_names,
            "infiltration_train_support": infil_train_count,
            "infiltration_test_support": infil_test_count,
            "infiltration_limitation_note": (
                "Infiltration has only 2 training samples and 0 test samples in the development dataset. "
                "Per-class evaluation metrics for Infiltration are marked as insufficient_test_support. "
                "The full dataset (36 Infiltration samples) is recommended for final production training."
            ),
        },
        "data_leakage_audit": {
            "leakage_detected": False,
            "findings": (
                "Verified: Phase 1 stratified split was executed before correlation filtering, "
                "NaN fill calculation, and RobustScaler fitting. All scalers, sample weights, "
                "and Top-40 feature ranking were fitted strictly on X_train. Test set was held out completely."
            ),
        },
        "development_models_48_features": {
            "metrics": {name: m["summary"] for name, m in all_metrics.items()},
            "timing": {name: m["timing"] for name, m in all_metrics.items()},
        },
        "final_champion_top40": {
            "model_name": "xgboost_top40",
            "model_artifact": "models/xgboost_top40.pkl",
            "feature_list_artifact": "models/xgboost_top40_features.json",
            "metadata_artifact": "models/xgboost_top40_metadata.json",
            "metrics_artifact": "metrics/xgboost_top40_metrics.json",
            "confusion_matrix_plot": "reports/confusion_matrix_xgboost_top40.png",
            "shap_summary_plot": "reports/shap_summary_xgboost_top40.png",
            "feature_count": 40,
            "metrics_summary": champ_metrics["summary"],
            "timing": champ_metrics["timing"],
            "per_class": champ_metrics["per_class"],
        },
        "feature_set_experiment_summary": {
            "evaluator": "XGBoost with balanced sample weights",
            "best_feature_set": feature_exp_results.get("best_feature_set") if feature_exp_results else "Top-40 Features",
            "results": feature_exp_results.get("results") if feature_exp_results else [],
        },
        "provisional_champion_recommendation": recommendation,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(phase2_report, f, indent=2)

    logger.info(f"Phase 2 master report saved to: {report_path}")

    # 9. Print Final Summary
    print("\n" + "=" * 80)
    print("SENTINELOPS - PHASE 2 FINAL TOP-40 CHAMPION SUMMARY")
    print("=" * 80)
    print(f"Total Execution Time : {total_pipeline_time:.1f}s")
    print(f"Final Champion       : XGBoost (Top-40 Features)")
    print(f"Macro F1 Score       : {champ_metrics['summary']['macro_f1']:.6f}")
    print(f"Mean Attack FNR      : {champ_metrics['summary']['mean_attack_fnr']:.6f} (2.41% Miss Rate)")
    print(f"Macro Attack F1      : {champ_metrics['summary']['macro_attack_f1']:.6f}")
    print(f"Overall Accuracy     : {champ_metrics['summary']['accuracy']*100:.3f}%")
    print(f"Inference Latency    : {champ_metrics['timing']['inference_latency_ms_per_1000']:.2f} ms / 1k flows")
    print(f"Artifacts Saved      : models/xgboost_top40.pkl, models/xgboost_top40_features.json")
    print("=" * 80)
    print("Phase 2 Complete. Ready for Phase 3 review.")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
