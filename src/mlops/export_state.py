"""
SentinelOps - MLOps Lifecycle State Export
===========================================
Collapses the live MLOps stack (MLflow store, governance log, DVC lock file,
Evidently summary) into a single static JSON document the API can serve.

Why export instead of querying live:
  The deployed API runs in a small container with no MLflow server and no DVC
  cache. Exporting at pipeline time keeps the serving path dependency-free
  while the numbers stay real -- they are read from the actual MLflow store and
  the actual append-only governance log, never hand-written.

Output: reports/mlops_state.json
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

STATE_FILENAME = "mlops_state.json"


# ---------------------------------------------------------------------------
# MLflow runs & registry
# ---------------------------------------------------------------------------

def collect_runs(experiment_name: str, max_results: int = 200) -> list[dict[str, Any]]:
    """
    Read every run in the experiment from the MLflow store.

    Returns [] when MLflow or the store is unavailable, so export never blocks
    the pipeline.
    """
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"MLflow experiment '{experiment_name}' not found.")
            return []

        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time ASC"],
            max_results=max_results,
        )
    except Exception as e:
        logger.warning(f"Could not read MLflow runs: {e}")
        return []

    collected = []
    for run in runs:
        metrics = run.data.metrics
        start = run.info.start_time
        end = run.info.end_time
        collected.append({
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
            "model_type": run.data.tags.get("model_type", "unknown"),
            "status": run.info.status,
            "started_at": datetime.fromtimestamp(start / 1000, tz=timezone.utc).isoformat() if start else None,
            "duration_seconds": round((end - start) / 1000, 2) if (start and end) else None,
            "macro_f1": metrics.get("macro_f1"),
            "mean_attack_fnr": metrics.get("mean_attack_fnr"),
            "macro_attack_f1": metrics.get("macro_attack_f1"),
            "accuracy": metrics.get("accuracy"),
            "training_time_seconds": metrics.get("timing.training_time_seconds"),
            "inference_latency_ms_per_1000": metrics.get("timing.inference_latency_ms_per_1000"),
            "per_class_fnr": {
                k.split(".")[1].replace("_", " "): v
                for k, v in metrics.items()
                if k.startswith("per_class.") and k.endswith(".fnr")
            },
            "params": dict(run.data.params),
        })
    return collected


def collect_registry(model_name: str) -> dict[str, Any]:
    """Read registered versions and identify the one holding the champion alias."""
    try:
        from mlflow.tracking import MlflowClient

        from src.mlops.tracking import CHAMPION_ALIAS

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        logger.warning(f"Could not read MLflow registry: {e}")
        return {"model_name": model_name, "versions": [], "champion_version": None}

    champion_version: Optional[str] = None
    try:
        champion_version = str(client.get_model_version_by_alias(model_name, CHAMPION_ALIAS).version)
    except Exception:
        champion_version = None

    entries = []
    for mv in sorted(versions, key=lambda v: int(v.version)):
        tags = dict(mv.tags or {})
        entries.append({
            "version": str(mv.version),
            "run_id": mv.run_id,
            "stage": tags.get("stage", "None"),
            "is_champion": str(mv.version) == champion_version,
            "governance": tags.get("governance"),
            "promotion_reason": tags.get("promotion_reason"),
            "rejection_reason": tags.get("rejection_reason"),
            "created_at": datetime.fromtimestamp(
                mv.creation_timestamp / 1000, tz=timezone.utc
            ).isoformat() if mv.creation_timestamp else None,
        })

    return {
        "model_name": model_name,
        "versions": entries,
        "champion_version": champion_version,
    }


# ---------------------------------------------------------------------------
# Governance history
# ---------------------------------------------------------------------------

def collect_governance(reports_dir: str | Path) -> dict[str, Any]:
    """Summarise the append-only champion/challenger decision log."""
    from src.mlops.governance import current_champion, load_decisions

    decisions = load_decisions(reports_dir)
    history = [
        {
            "timestamp": d["timestamp"],
            "challenger": d["challenger"],
            "champion": d.get("champion"),
            "decision": d["decision"],
            "promoted": d["promoted"],
            "challenger_macro_f1": d["challenger_metrics"]["macro_f1"],
            "challenger_attack_fnr": d["challenger_metrics"]["mean_attack_fnr"],
            "champion_macro_f1": (d.get("champion_metrics") or {}).get("macro_f1"),
            "champion_attack_fnr": (d.get("champion_metrics") or {}).get("mean_attack_fnr"),
            "reasons": d["reasons"],
            "gates": d["gates"],
        }
        for d in decisions
    ]

    champ = current_champion(reports_dir)
    return {
        "total_decisions": len(history),
        "promotions": sum(1 for h in history if h["promoted"]),
        "rejections": sum(1 for h in history if not h["promoted"]),
        "thresholds": decisions[-1]["thresholds"] if decisions else None,
        "current_champion": champ["challenger"] if champ else None,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Production model per-class metrics
# ---------------------------------------------------------------------------

def collect_production_metrics(
    reports_dir: str | Path,
    metrics_dir: str | Path,
    max_per_class_fnr: float,
) -> dict[str, Any]:
    """
    Per-class FNR table for the model currently in production.

    Zero-support classes are reported explicitly as such rather than as a
    passing 0.0, so the table cannot overstate coverage.
    """
    from src.mlops.governance import current_champion

    champ = current_champion(reports_dir)
    if champ is None:
        return {"model": None, "classes": []}

    model_name = champ["challenger"]
    metrics_path = Path(metrics_dir) / f"{model_name}_metrics.json"
    if not metrics_path.exists():
        return {"model": model_name, "classes": []}

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    classes = []
    for cls_name, stats in metrics.get("per_class", {}).items():
        if stats.get("insufficient_test_support"):
            classes.append({
                "class": cls_name,
                "support": 0,
                "fnr": None,
                "recall": None,
                "f1": None,
                "status": "NO_TEST_SUPPORT",
            })
            continue

        fnr = stats.get("fnr")
        if cls_name == "BENIGN":
            status = "OK"
        else:
            status = "OK" if fnr is not None and fnr <= max_per_class_fnr else "BREACH"

        classes.append({
            "class": cls_name,
            "support": stats.get("support"),
            "fnr": fnr,
            "recall": stats.get("recall"),
            "f1": stats.get("f1"),
            "status": status,
        })

    return {
        "model": model_name,
        "summary": metrics.get("summary", {}),
        "test_samples": metrics.get("test_samples"),
        "max_per_class_fnr": max_per_class_fnr,
        "classes": classes,
    }


# ---------------------------------------------------------------------------
# DVC pipeline
# ---------------------------------------------------------------------------

def collect_pipeline(project_root: str | Path) -> dict[str, Any]:
    """
    Describe the DVC DAG from dvc.yaml, annotated with dvc.lock freshness.

    Reports availability rather than raising when DVC has not been initialised.
    """
    root = Path(project_root)
    dvc_yaml = root / "dvc.yaml"
    dvc_lock = root / "dvc.lock"

    if not dvc_yaml.exists():
        return {"available": False, "reason": "dvc.yaml not present", "stages": []}

    try:
        import yaml
        pipeline = yaml.safe_load(dvc_yaml.read_text(encoding="utf-8")) or {}
        lock = yaml.safe_load(dvc_lock.read_text(encoding="utf-8")) if dvc_lock.exists() else {}
    except Exception as e:
        return {"available": False, "reason": f"could not parse DVC files: {e}", "stages": []}

    locked_stages = (lock or {}).get("stages", {})

    stages = []
    for name, spec in (pipeline.get("stages") or {}).items():
        stages.append({
            "name": name,
            "cmd": spec.get("cmd"),
            "deps": spec.get("deps", []),
            "outs": [o if isinstance(o, str) else list(o.keys())[0] for o in spec.get("outs", [])],
            "params": spec.get("params", []),
            "locked": name in locked_stages,
        })

    return {
        "available": True,
        "lock_present": dvc_lock.exists(),
        "stages": stages,
    }


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

def collect_drift(reports_dir: str | Path) -> dict[str, Any]:
    """Read the Evidently drift summary if one has been generated."""
    summary_path = Path(reports_dir) / "drift_summary.json"
    if not summary_path.exists():
        return {"available": False, "reason": "no drift report generated yet"}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data["available"] = True
        return data
    except Exception as e:
        return {"available": False, "reason": f"could not read drift summary: {e}"}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_state(params: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Assemble the complete MLOps lifecycle state document."""
    from src.mlops.tracking import DEFAULT_EXPERIMENT, REGISTERED_MODEL_NAME

    root = Path(project_root)
    reports_dir = root / params["paths"]["reports_dir"]
    metrics_dir = root / params["paths"]["metrics_dir"]
    governance_cfg = params.get("governance", {})
    max_per_class_fnr = float(governance_cfg.get("max_per_class_fnr", 0.30))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": params.get("general", {}).get("project_name", "SentinelOps"),
        "runs": collect_runs(DEFAULT_EXPERIMENT),
        "registry": collect_registry(REGISTERED_MODEL_NAME),
        "governance": collect_governance(reports_dir),
        "production_metrics": collect_production_metrics(reports_dir, metrics_dir, max_per_class_fnr),
        "pipeline": collect_pipeline(root),
        "drift": collect_drift(reports_dir),
    }


def export_state(params: dict[str, Any], project_root: str | Path) -> Path:
    """Build and write reports/mlops_state.json."""
    root = Path(project_root)
    state = build_state(params, root)

    out_dir = root / params["paths"]["reports_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / STATE_FILENAME

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    logger.info(
        f"Exported MLOps state to {out_path}: "
        f"{len(state['runs'])} run(s), "
        f"{state['governance']['total_decisions']} governance decision(s), "
        f"champion={state['governance']['current_champion']}"
    )
    return out_path
