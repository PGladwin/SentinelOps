"""
SentinelOps API - MLOps Control Panel Routes
=============================================
Serves the lifecycle state the frontend's MLOps panel renders: experiment
runs, model registry, champion/challenger history, per-class FNR for the
production model, the DVC DAG, and drift status.

Everything is read from reports/mlops_state.json, which the pipeline exports
after each run. The deployed container therefore needs neither an MLflow
server nor a DVC cache -- but the numbers are still real, having been read
out of the actual MLflow store and the append-only governance log at export
time.

CI/CD history is deliberately absent here: the frontend queries the GitHub
Actions REST API directly so that panel is genuinely live rather than a
snapshot.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from api.config import settings

logger = logging.getLogger("sentinelops.api.mlops")

router = APIRouter(prefix="/mlops", tags=["MLOps"])


class MlopsStateCache:
    """
    Caches the exported state, reloading only when the file changes.

    The pipeline rewrites this file on every run, so a long-lived server must
    pick up new state without a restart; an mtime check is enough and avoids
    re-reading a multi-hundred-KB document on every request.
    """

    def __init__(self, path: Path):
        self.path = path
        self._cached: Optional[dict[str, Any]] = None
        self._mtime: Optional[float] = None

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"MLOps state not found at {self.path}. Run the pipeline "
                "(`dvc repro` or scripts/run_experiment_sweep.py) to generate it."
            )

        mtime = self.path.stat().st_mtime
        if self._cached is None or mtime != self._mtime:
            with open(self.path, "r", encoding="utf-8") as f:
                self._cached = json.load(f)
            self._mtime = mtime
            logger.info(f"Loaded MLOps state from {self.path}")

        return self._cached


_cache = MlopsStateCache(settings.mlops_state_path)


def _section(name: str) -> Any:
    """Return one section of the exported state, or 404/503 with a clear reason."""
    try:
        state = _cache.load()
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read MLOps state: {e}",
        )

    if name not in state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Section '{name}' absent from the exported state.",
        )
    return state[name]


@router.get("/state", summary="Complete MLOps lifecycle state")
def get_state() -> dict[str, Any]:
    """Everything the control panel needs, in one request."""
    try:
        return _cache.load()
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@router.get("/runs", summary="MLflow experiment runs")
def get_runs() -> list[dict[str, Any]]:
    """Every training run with its metrics, oldest first."""
    return _section("runs")


@router.get("/registry", summary="Model registry versions and the champion pointer")
def get_registry() -> dict[str, Any]:
    """Registered versions, their lifecycle stage, and which holds the champion alias."""
    return _section("registry")


@router.get("/governance", summary="Champion/challenger decision history")
def get_governance() -> dict[str, Any]:
    """Every promote/reject decision with the gate results that produced it."""
    return _section("governance")


@router.get("/metrics", summary="Per-class metrics for the production model")
def get_production_metrics() -> dict[str, Any]:
    """
    Per-class FNR table for the model currently serving.

    Classes with no test support are reported as NO_TEST_SUPPORT rather than a
    passing 0.0, so the table cannot overstate detection coverage.
    """
    return _section("production_metrics")


@router.get("/pipeline", summary="DVC pipeline DAG and stage freshness")
def get_pipeline() -> dict[str, Any]:
    """Stages, their dependencies and outputs, and whether dvc.lock covers them."""
    return _section("pipeline")


@router.get("/drift", summary="Data drift status")
def get_drift() -> dict[str, Any]:
    """
    Latest drift verdict, or an availability flag when no check has run.

    The verdict comes from src.mlops.drift.detect_drift -- deterministic,
    unit-tested, and dependency-free -- so an Evidently upgrade cannot
    silently change whether the pipeline believes drift occurred. Evidently
    renders the accompanying visual at /mlops/drift/report.
    """
    return _section("drift")


@router.get("/drift/report", summary="Evidently drift report (HTML)")
def get_drift_report() -> FileResponse:
    """
    Serve the rendered Evidently report for embedding in the control panel.

    Returned as a file rather than JSON so the frontend can drop it straight
    into an iframe. Absent until a drift check has been run with --html.
    """
    path = settings.reports_dir / "drift_report.html"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No drift report has been rendered. Run "
                "`python scripts/check_drift.py --simulate --html`."
            ),
        )
    return FileResponse(path, media_type="text/html", filename="drift_report.html")
