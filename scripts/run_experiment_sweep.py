"""
SentinelOps - Experiment Sweep & Governance Pipeline
=====================================================
Trains a defined sequence of REAL model configurations, logs each to MLflow,
and puts every one through the champion/challenger governance gate.

This is what produces the MLOps control panel's history. Nothing here is
seeded, back-dated, or hand-written: the runs table, the promote/reject trail,
and the registry state are all consequences of models that actually trained on
the actual dataset.

On promotion the winning model is written to the canonical serving artifacts
(models/champion.*), so the API always serves whatever governance last
approved rather than a hardcoded filename.

Usage:
    python scripts/run_experiment_sweep.py                 # full sweep
    python scripts/run_experiment_sweep.py --only xgb_top40
    python scripts/run_experiment_sweep.py --fast          # smaller configs, smoke test
    python scripts/run_experiment_sweep.py --reset         # clear prior history first
"""

import argparse
import json
import logging
import pickle
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("sentinelops.sweep")

# Quieten MLflow's very chatty model-logging output.
logging.getLogger("mlflow").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------
# Ordered deliberately: a weak-but-valid baseline first so the gate has an
# incumbent to defend, then genuine improvements, then a model that SHOULD be
# rejected, then the feature-reduced candidates.

SWEEP: list[dict[str, Any]] = [
    {
        "name": "rf_baseline",
        "model_type": "random_forest",
        "feature_set": "full",
        "note": "Random Forest baseline on the Phase 1 feature set.",
        "hyperparameters": {
            "n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2,
            "class_weight": "balanced",
            # Bounded rather than -1: on the full 1.78M-row training split a
            # depth-20 forest built 12-wide is the pipeline's memory peak.
            # This caps concurrent tree construction without altering the
            # resulting model in any way.
            "n_jobs": 4,
        },
    },
    {
        "name": "xgb_baseline",
        "model_type": "xgboost",
        "feature_set": "full",
        "note": "Gradient boosting with balanced per-sample weights.",
        "hyperparameters": {
            "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "eval_metric": "mlogloss", "n_jobs": -1,
        },
    },
    {
        "name": "xgb_deep",
        "model_type": "xgboost",
        "feature_set": "full",
        "note": "Deeper trees, lower learning rate, more rounds.",
        "hyperparameters": {
            "n_estimators": 400, "max_depth": 8, "learning_rate": 0.05,
            "subsample": 0.9, "colsample_bytree": 0.8,
            "eval_metric": "mlogloss", "n_jobs": -1,
        },
    },
    {
        "name": "mlp_baseline",
        "model_type": "mlp",
        "feature_set": "full",
        "note": "Neural baseline. sklearn MLPClassifier cannot weight classes.",
        "hyperparameters": {
            "hidden_layers": [128, 64, 32], "max_epochs": 50,
            "batch_size": 512, "learning_rate": 0.001,
            "early_stopping_patience": 5,
        },
    },
    {
        "name": "xgb_top40",
        "model_type": "xgboost",
        "feature_set": "top40",
        "note": "Top-40 features ranked on training data only.",
        "hyperparameters": {
            "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "eval_metric": "mlogloss", "n_jobs": -1,
        },
    },
    {
        "name": "xgb_top40_tuned",
        "model_type": "xgboost",
        "feature_set": "top40",
        "note": "Top-40 with a longer, lower-learning-rate schedule.",
        "hyperparameters": {
            "n_estimators": 400, "max_depth": 7, "learning_rate": 0.06,
            "subsample": 0.9, "colsample_bytree": 0.9,
            "eval_metric": "mlogloss", "n_jobs": -1,
        },
    },
]

# Reduced configs so the whole pipeline can be smoke-tested in ~1 minute.
FAST_OVERRIDES: dict[str, dict[str, Any]] = {
    "random_forest": {"n_estimators": 25, "max_depth": 12},
    "xgboost": {"n_estimators": 50, "max_depth": 5},
    "mlp": {"max_epochs": 8, "hidden_layers": [32, 16]},
}


def load_params(path: str | Path) -> dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_fast_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Shrink a config for smoke testing without changing its identity."""
    config = json.loads(json.dumps(config))
    config["hyperparameters"].update(FAST_OVERRIDES.get(config["model_type"], {}))
    config["note"] = config["note"] + " [fast mode]"
    return config


# ---------------------------------------------------------------------------
# Training dispatch
# ---------------------------------------------------------------------------

def train_one(
    config: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any],
    feature_names: list[str],
) -> tuple[Any, float]:
    """Train a single configuration, reusing the Phase 2 training functions."""
    from src.models.train import train_mlp, train_random_forest, train_xgboost

    # The train_* helpers read hyperparameters out of a params-shaped dict, so
    # overlay this config's values rather than duplicating the training code.
    scoped = json.loads(json.dumps({
        "general": params.get("general", {}),
        "training": {},
    }))
    key = config["model_type"]
    scoped["training"][key] = config["hyperparameters"]

    dispatch = {
        "random_forest": train_random_forest,
        "xgboost": train_xgboost,
        "mlp": train_mlp,
    }
    return dispatch[key](X_train, y_train, scoped, feature_names)


def select_features(
    feature_set: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    seed: int,
    cache: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Resolve a feature-set name to concrete matrices.

    Top-K rankings are computed on X_train ONLY and cached, so the test set
    never participates in feature selection.
    """
    if feature_set == "full":
        return X_train, X_test, feature_names

    if not feature_set.startswith("top"):
        raise ValueError(f"Unknown feature_set: {feature_set}")

    k = int(feature_set[3:])
    if "ranking" not in cache:
        from src.models.feature_experiment import compute_feature_importances
        logger.info("Computing feature importances on TRAINING data only...")
        cache["ranking"] = compute_feature_importances(
            X_train, y_train, feature_names, model_type="xgb", random_seed=seed
        )

    ranked = cache["ranking"][:k]
    names = [n for n, _, _ in ranked]
    indices = [i for _, _, i in ranked]
    return X_train[:, indices], X_test[:, indices], names


# ---------------------------------------------------------------------------
# Serving artifacts
# ---------------------------------------------------------------------------

def write_serving_artifacts(
    model: Any,
    model_name: str,
    selected_features: list[str],
    metadata: dict[str, Any],
    params: dict[str, Any],
    models_dir: Path,
    inverse_encoding: dict[int, str],
) -> None:
    """
    Write the canonical champion.* artifacts plus the raw-input bundle and
    demo bank, so the API immediately serves the newly promoted model.
    """
    from src.data.inference_prep import build_inference_bundle, save_inference_bundle
    from src.models.demo_samples import build_demo_samples, save_demo_samples

    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "champion.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(models_dir / "champion_features.json", "w", encoding="utf-8") as f:
        json.dump(selected_features, f, indent=2)
    with open(models_dir / "champion_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    processed_dir = PROJECT_ROOT / params["paths"]["processed_dir"]
    bundle = build_inference_bundle(
        champion_features=selected_features,
        scaler_path=processed_dir / "scaler.pkl",
        fill_values_path=processed_dir / "fill_values.json",
        feature_names_path=processed_dir / "feature_names.json",
    )
    save_inference_bundle(bundle, models_dir / "champion_preprocessor.json")

    try:
        demo = build_demo_samples(
            parquet_dir=PROJECT_ROOT / params["paths"]["parquet_dir"],
            champion_features=selected_features,
            model=model,
            bundle=bundle,
            inverse_encoding=inverse_encoding,
            label_map=params["cleaning"]["label_map"],
            random_seed=params.get("general", {}).get("random_seed", 42),
        )
        save_demo_samples(demo, models_dir / "demo_samples.json")
    except Exception as e:
        logger.error(f"Demo sample regeneration failed (non-fatal): {e}")

    logger.info(f"Serving artifacts updated for champion '{model_name}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="SentinelOps experiment sweep with governance")
    parser.add_argument("--params", default=str(PROJECT_ROOT / "params.yaml"))
    parser.add_argument("--only", nargs="*", help="Run only these config names")
    parser.add_argument("--fast", action="store_true", help="Shrink configs for a smoke test")
    parser.add_argument("--reset", action="store_true", help="Clear prior governance history and MLflow runs")
    args = parser.parse_args()

    params = load_params(args.params)
    seed = params.get("general", {}).get("random_seed", 42)
    reports_dir = PROJECT_ROOT / params["paths"]["reports_dir"]
    metrics_dir = PROJECT_ROOT / params["paths"]["metrics_dir"]
    models_dir = PROJECT_ROOT / params["paths"]["models_dir"]
    for d in (reports_dir, metrics_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.reset:
        log_path = reports_dir / "governance_log.jsonl"
        if log_path.exists():
            log_path.unlink()
            logger.warning("Cleared governance_log.jsonl")
        # SQLite tracking store plus its artifact directory (see
        # src/mlops/tracking.resolve_tracking_uri for why not ./mlruns).
        for path in (PROJECT_ROOT / "mlflow.db", PROJECT_ROOT / "mlflow.db-journal"):
            if path.exists():
                path.unlink()
                logger.warning(f"Cleared {path.name}")
        artifacts = PROJECT_ROOT / "mlartifacts"
        if artifacts.exists():
            shutil.rmtree(artifacts)
            logger.warning("Cleared mlartifacts/")

    from src.models.evaluate import evaluate_model, save_confusion_matrix_plot, save_metrics
    from src.models.train import load_phase1_data
    from src.mlops import evaluate_challenger, append_decision, current_champion
    from src.mlops.export_state import export_state
    from src.mlops.tracking import (
        get_champion_version, init_tracking, log_training_run,
        mark_rejected, promote_to_champion,
    )

    init_tracking(project_root=PROJECT_ROOT)

    X_train, y_train, X_test, y_test, feature_names, class_encoding, inverse_encoding = load_phase1_data(params)
    class_names = [inverse_encoding[i] for i in range(len(class_encoding))]
    governance_cfg = params.get("governance", {})

    configs = SWEEP
    if args.only:
        configs = [c for c in SWEEP if c["name"] in args.only]
        if not configs:
            logger.error(f"No sweep configs matched {args.only}. Available: {[c['name'] for c in SWEEP]}")
            return 1
    if args.fast:
        configs = [apply_fast_overrides(c) for c in configs]

    # Resume from any existing champion so reruns extend history rather than restarting it.
    champ_record = current_champion(reports_dir)
    champion_name: Optional[str] = champ_record["challenger"] if champ_record else None
    champion_metrics: Optional[dict[str, Any]] = None
    if champion_name:
        champ_path = metrics_dir / f"{champion_name}_metrics.json"
        if champ_path.exists():
            champion_metrics = json.loads(champ_path.read_text(encoding="utf-8"))
            logger.info(f"Resuming with incumbent champion: {champion_name}")

    ranking_cache: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    sweep_start = time.time()

    for config in configs:
        name = config["name"]
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"SWEEP RUN: {name}  ({config['model_type']}, {config['feature_set']})")
        logger.info(f"  {config['note']}")
        logger.info("=" * 70)

        Xtr, Xte, selected = select_features(
            config["feature_set"], X_train, y_train, X_test, feature_names, seed, ranking_cache
        )

        model, train_time = train_one(config, Xtr, y_train, params, selected)

        metrics = evaluate_model(
            model=model, X_test=Xte, y_test=y_test,
            class_encoding=class_encoding, model_name=name, train_time=train_time,
        )
        save_metrics(metrics, name, metrics_dir)
        cm_path = save_confusion_matrix_plot(metrics["confusion_matrix"], class_names, name, reports_dir)

        run_info = log_training_run(
            run_name=name,
            model=model,
            model_type=config["model_type"],
            hyperparameters=config["hyperparameters"],
            metrics=metrics,
            feature_names=selected,
            artifacts=[cm_path, metrics_dir / f"{name}_metrics.json"],
            extra_params={
                "feature_set": config["feature_set"],
                "train_rows": len(Xtr),
                "test_rows": len(Xte),
                "random_seed": seed,
                "note": config["note"],
            },
        )

        decision = evaluate_challenger(
            challenger_name=name,
            challenger_metrics=metrics,
            champion_name=champion_name,
            champion_metrics=champion_metrics,
            governance_cfg=governance_cfg,
        )
        decision["mlflow_run_id"] = run_info["run_id"]
        decision["mlflow_version"] = run_info["version"]
        append_decision(decision, reports_dir)

        reason_text = " ".join(decision["reasons"])[:480]
        if decision["promoted"]:
            if run_info["version"]:
                promote_to_champion(run_info["version"], reason=reason_text)

            metadata = {
                "model_name": name,
                "model_type": config["model_type"],
                "feature_set": config["feature_set"],
                "feature_count": len(selected),
                "ordered_feature_names": selected,
                "hyperparameters": config["hyperparameters"],
                "class_mapping": class_encoding,
                "random_seed": seed,
                "training_dataset_size": int(len(Xtr)),
                "test_dataset_size": int(len(Xte)),
                "training_time_seconds": round(train_time, 4),
                "creation_timestamp": datetime.now(timezone.utc).isoformat(),
                "mlflow_run_id": run_info["run_id"],
                "version": run_info["version"],
                "stage": "Production",
                "governance_decision": decision["decision"],
                "feature_selection_method": (
                    "XGBoost multiclass balanced importances fitted on X_train only"
                    if config["feature_set"] != "full" else "Phase 1 correlation-filtered set"
                ),
            }
            write_serving_artifacts(
                model, name, selected, metadata, params, models_dir, inverse_encoding
            )
            champion_name, champion_metrics = name, metrics
        else:
            mark_rejected(run_info["version"], reason=reason_text)

        results.append({
            "name": name,
            "decision": decision["decision"],
            "macro_f1": metrics["summary"]["macro_f1"],
            "mean_attack_fnr": metrics["summary"]["mean_attack_fnr"],
            "train_time": round(train_time, 2),
        })

    export_state(params, PROJECT_ROOT)

    # ---- Summary --------------------------------------------------
    elapsed = time.time() - sweep_start
    print("\n" + "=" * 78)
    print("SWEEP COMPLETE - GOVERNANCE OUTCOMES")
    print("=" * 78)
    print(f"  {'run':<20} {'decision':<10} {'macro F1':>10} {'attack FNR':>12} {'train s':>9}")
    print("  " + "-" * 66)
    for r in results:
        print(f"  {r['name']:<20} {r['decision']:<10} {r['macro_f1']:>10.4f} "
              f"{r['mean_attack_fnr']:>12.4f} {r['train_time']:>9.1f}")
    print("  " + "-" * 66)
    print(f"  Final champion : {champion_name}")
    print(f"  Registry version: {get_champion_version()}")
    print(f"  Total time      : {elapsed:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
