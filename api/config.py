"""
SentinelOps API - Runtime Configuration
========================================
Environment-driven settings so the service can run outside the repo root and
be deployed to a container platform without code changes.

Every value has a working local default, so `uvicorn api.main:app` from the
project root continues to behave exactly as before with no env vars set.
"""

import os
from pathlib import Path


def _project_root() -> Path:
    """Repo root, resolved relative to this file rather than the cwd."""
    return Path(__file__).resolve().parent.parent


def _path_from_env(var: str, default_relative: str) -> Path:
    """Read a path from the environment, falling back to a repo-relative default."""
    raw = os.getenv(var)
    if raw:
        return Path(raw).expanduser().resolve()
    return _project_root() / default_relative


class Settings:
    """Container for all runtime-configurable API settings."""

    def __init__(self) -> None:
        self.project_root: Path = _project_root()
        self.models_dir: Path = _path_from_env("SENTINELOPS_MODELS_DIR", "models")
        self.processed_dir: Path = _path_from_env("SENTINELOPS_PROCESSED_DIR", "data/processed")
        self.reports_dir: Path = _path_from_env("SENTINELOPS_REPORTS_DIR", "reports")

        # Upload guardrails for /analyze: cap the file, the rows scanned, and
        # the rows echoed back, so a large CSV cannot exhaust memory or produce
        # an unbounded response body.
        self.max_upload_bytes: int = int(os.getenv("SENTINELOPS_MAX_UPLOAD_MB", "50")) * 1024 * 1024
        self.max_analysis_rows: int = int(os.getenv("SENTINELOPS_MAX_ANALYSIS_ROWS", "100000"))
        self.max_returned_rows: int = int(os.getenv("SENTINELOPS_MAX_RETURNED_ROWS", "20000"))
        self.max_shap_rows: int = int(os.getenv("SENTINELOPS_MAX_SHAP_ROWS", "20000"))

    @property
    def cors_origins(self) -> list[str]:
        """
        Allowed CORS origins.

        Set SENTINELOPS_CORS_ORIGINS to a comma-separated list in deployment
        (e.g. the Render static site URL). Defaults to local dev servers.
        """
        raw = os.getenv("SENTINELOPS_CORS_ORIGINS")
        if raw:
            return [origin.strip() for origin in raw.split(",") if origin.strip()]
        return [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]

    # ---- Derived artifact paths -------------------------------------
    @property
    def champion_model_path(self) -> Path:
        return self.models_dir / "xgboost_top40.pkl"

    @property
    def inference_bundle_path(self) -> Path:
        return self.models_dir / "champion_preprocessor.json"

    @property
    def demo_samples_path(self) -> Path:
        return self.models_dir / "demo_samples.json"

    @property
    def mlops_state_path(self) -> Path:
        return self.reports_dir / "mlops_state.json"


settings = Settings()
