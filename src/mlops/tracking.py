"""
SentinelOps - MLflow Experiment Tracking & Model Registry
==========================================================
Thin wrapper over MLflow so training code stays readable and the rest of the
project never imports mlflow directly.

Responsibilities:
  - Point MLflow at a local SQLite store (mlflow.db) with no server required
  - Log params, the full evaluate_model() metric surface (including every
    per-class FNR), and artifacts for each training run
  - Register models and manage the champion pointer

Registry note:
  MLflow 3.x deprecated model version *stages* in favour of *aliases*. This
  module therefore sets the `champion` alias on the serving version and writes
  a human-readable `stage` tag ("Production" / "Archived") for display.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = "SentinelOps"
REGISTERED_MODEL_NAME = "sentinelops-intrusion-detector"
CHAMPION_ALIAS = "champion"

STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"


def _mlflow():
    """Import mlflow lazily so the package stays importable without it."""
    import mlflow
    return mlflow


def _project_root_default() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_tracking_uri(project_root: str | Path | None = None) -> str:
    """
    Resolve the tracking store URI.

    Honours MLFLOW_TRACKING_URI when set (e.g. a remote server in CI);
    otherwise uses a local SQLite store at the project root.

    SQLite rather than the classic ./mlruns file store: MLflow 3.x put the
    filesystem backend into maintenance mode and refuses to open it without an
    explicit opt-out. SQLite is also the backend that actually supports the
    Model Registry, which the governance gate depends on.
    """
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    root = Path(project_root) if project_root else _project_root_default()
    return f"sqlite:///{(root / 'mlflow.db').resolve().as_posix()}"


def resolve_artifact_root(project_root: str | Path | None = None) -> str:
    """Filesystem location for run artifacts under a database backend."""
    env_uri = os.getenv("MLFLOW_ARTIFACT_ROOT")
    if env_uri:
        return env_uri
    root = Path(project_root) if project_root else _project_root_default()
    artifacts = root / "mlartifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts.resolve().as_uri()


def init_tracking(
    experiment_name: str = DEFAULT_EXPERIMENT,
    project_root: str | Path | None = None,
) -> str:
    """
    Configure MLflow and ensure the experiment exists.

    Returns
    -------
    str
        The resolved tracking URI.
    """
    mlflow = _mlflow()
    uri = resolve_tracking_uri(project_root)
    mlflow.set_tracking_uri(uri)

    # Create the experiment with an explicit artifact root the first time;
    # set_experiment alone would default artifacts next to the cwd.
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=resolve_artifact_root(project_root)
        )
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking initialized: experiment='{experiment_name}' uri={uri}")
    return uri


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """
    Flatten an evaluate_model() result into MLflow-loggable scalar metrics.

    Per-class figures become `per_class.<Class>.<metric>`. Classes with no test
    support are skipped rather than logged as 0.0, which would misrepresent a
    class that was never evaluated as one the model failed on.
    """
    flat: dict[str, float] = {}

    for key, value in metrics.get("summary", {}).items():
        if isinstance(value, (int, float)):
            flat[key] = float(value)

    for key, value in metrics.get("timing", {}).items():
        if isinstance(value, (int, float)):
            flat[f"timing.{key}"] = float(value)

    for cls_name, stats in metrics.get("per_class", {}).items():
        if stats.get("insufficient_test_support"):
            continue
        safe = str(cls_name).replace(" ", "_")
        for stat in ("precision", "recall", "f1", "fnr"):
            value = stats.get(stat)
            if isinstance(value, (int, float)):
                flat[f"per_class.{safe}.{stat}"] = float(value)
        if isinstance(stats.get("support"), int):
            flat[f"per_class.{safe}.support"] = float(stats["support"])

    return flat


@contextmanager
def start_run(run_name: str, tags: Optional[dict[str, Any]] = None) -> Iterator[Any]:
    """Context manager wrapping mlflow.start_run with SentinelOps tags."""
    mlflow = _mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})
        yield run


def log_training_run(
    run_name: str,
    model: Any,
    model_type: str,
    hyperparameters: dict[str, Any],
    metrics: dict[str, Any],
    feature_names: list[str],
    artifacts: Optional[list[str | Path]] = None,
    extra_params: Optional[dict[str, Any]] = None,
    register: bool = True,
    registered_model_name: str = REGISTERED_MODEL_NAME,
) -> dict[str, Any]:
    """
    Log one complete training run and optionally register the model.

    Parameters
    ----------
    run_name : str
        Human-readable run identifier shown in the MLflow UI.
    model : Any
        Trained estimator.
    model_type : str
        'xgboost', 'random_forest', or 'mlp'.
    hyperparameters : dict
        Hyperparameters actually used.
    metrics : dict
        evaluate_model() output.
    feature_names : list[str]
        Ordered feature names the model consumes.
    artifacts : list, optional
        Files to attach (confusion matrix PNG, SHAP plot, metrics JSON).
    extra_params : dict, optional
        Additional params (dataset size, seed, feature set name).
    register : bool
        Register the model in the Model Registry.

    Returns
    -------
    dict
        run_id, run_name, and the registered model version when applicable.
    """
    mlflow = _mlflow()

    with start_run(run_name, tags={"model_type": model_type, "project": "SentinelOps"}) as run:
        params: dict[str, Any] = {f"hp.{k}": v for k, v in (hyperparameters or {}).items()}
        params["model_type"] = model_type
        params["feature_count"] = len(feature_names)
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        mlflow.log_metrics(flatten_metrics(metrics))

        # The exact feature contract is part of the model's identity.
        mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")
        mlflow.log_dict(metrics, "evaluation_metrics.json")

        for artifact in artifacts or []:
            path = Path(artifact)
            if path.exists():
                mlflow.log_artifact(str(path))
            else:
                logger.warning(f"Artifact not found, skipping: {path}")

        version: Optional[str] = None
        try:
            flavor = mlflow.xgboost if model_type == "xgboost" else mlflow.sklearn
            info = flavor.log_model(
                model,
                name="model",
                registered_model_name=registered_model_name if register else None,
            )
            if register:
                version = _latest_version_for_run(registered_model_name, run.info.run_id)
        except Exception as e:
            logger.error(f"Model logging/registration failed for '{run_name}': {e}", exc_info=True)

        result = {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "registered_model": registered_model_name if version else None,
            "version": version,
        }

    logger.info(f"MLflow run logged: {run_name} (run_id={result['run_id']}, version={version})")
    return result


def _latest_version_for_run(model_name: str, run_id: str) -> Optional[str]:
    """Find the registered version created by a given run."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        logger.warning(f"Could not search model versions: {e}")
        return None

    for mv in versions:
        if mv.run_id == run_id:
            return str(mv.version)
    return None


# ---------------------------------------------------------------------------
# Registry / champion pointer
# ---------------------------------------------------------------------------

def promote_to_champion(
    version: str,
    model_name: str = REGISTERED_MODEL_NAME,
    reason: str = "",
) -> None:
    """
    Point the `champion` alias at `version` and archive the previous holder.

    Uses aliases rather than the deprecated stage transitions of MLflow 2.x.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    previous: Optional[str] = None
    try:
        current = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
        previous = str(current.version)
    except Exception:
        previous = None  # No champion yet; first promotion.

    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
    client.set_model_version_tag(model_name, version, "stage", STAGE_PRODUCTION)
    if reason:
        client.set_model_version_tag(model_name, version, "promotion_reason", reason[:480])

    if previous and previous != version:
        client.set_model_version_tag(model_name, previous, "stage", STAGE_ARCHIVED)
        logger.info(f"Archived previous champion: {model_name} v{previous}")

    logger.info(f"Promoted to champion: {model_name} v{version}")


def mark_rejected(
    version: Optional[str],
    model_name: str = REGISTERED_MODEL_NAME,
    reason: str = "",
) -> None:
    """Tag a version that failed the governance gate so the registry records why."""
    if not version:
        return
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        client.set_model_version_tag(model_name, version, "stage", STAGE_ARCHIVED)
        client.set_model_version_tag(model_name, version, "governance", "REJECTED")
        if reason:
            client.set_model_version_tag(model_name, version, "rejection_reason", reason[:480])
        logger.info(f"Marked {model_name} v{version} as REJECTED by governance.")
    except Exception as e:
        logger.warning(f"Could not tag rejected version {version}: {e}")


def get_champion_version(model_name: str = REGISTERED_MODEL_NAME) -> Optional[str]:
    """Return the version currently carrying the `champion` alias, if any."""
    from mlflow.tracking import MlflowClient

    try:
        return str(MlflowClient().get_model_version_by_alias(model_name, CHAMPION_ALIAS).version)
    except Exception:
        return None
