"""
SentinelOps - Local Inference REST API
======================================
FastAPI application exposing real-time inference, model introspection,
SHAP explanations, SOC batch analysis, and MLOps lifecycle state for the
SentinelOps Champion.
"""

import functools
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.mlops_routes import router as mlops_router
from api.model_service import ModelService
from api.schemas import (
    AnalysisResponse,
    DemoSample,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger("sentinelops.api")
logging.basicConfig(level=logging.INFO)


@functools.lru_cache(maxsize=1)
def get_soc_thresholds() -> dict[str, float]:
    """
    Attack-rate cutoffs for the threat-level banner, from params.yaml:soc.

    Falls back to sane defaults if params.yaml is unavailable in the deployed
    image, so the endpoint never fails on a missing config file.
    """
    defaults = {"high": 0.20, "medium": 0.05}
    try:
        import yaml
        params_path = settings.project_root / "params.yaml"
        if not params_path.exists():
            return defaults
        with open(params_path, "r", encoding="utf-8") as f:
            soc = (yaml.safe_load(f) or {}).get("soc", {})
        thresholds = soc.get("threat_level_thresholds", {})
        return {
            "high": float(thresholds.get("high", defaults["high"])),
            "medium": float(thresholds.get("medium", defaults["medium"])),
        }
    except Exception as e:
        logger.warning(f"Falling back to default SOC thresholds: {e}")
        return defaults


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the ModelService at server startup."""
    logger.info("Initializing SentinelOps API service and loading Champion model...")
    try:
        service = ModelService.get_instance()
        logger.info(
            f"SentinelOps API ready: Champion '{service.metadata.get('model_name')}' "
            f"loaded with {len(service.feature_names)} raw features."
        )
    except Exception as e:
        # Logged, not raised: a failed load must surface as 503 on /health
        # rather than preventing the container from binding its port.
        logger.error(f"Failed to load model service on startup: {e}", exc_info=True)
    yield
    logger.info("Shutting down SentinelOps API...")


app = FastAPI(
    title="SentinelOps Inference API",
    description=(
        "Real-Time Network Intrusion Detection & Explainability Service. "
        "Serves whichever model the champion/challenger gate last promoted. "
        "Feature values are RAW; scaling is applied server-side."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MLOps control-panel routes (/mlops/*), served from the exported state file.
app.include_router(mlops_router)


# ---------------------------------------------------------------------------
# Health & Introspection
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Verify backend health and model loading status."""
    try:
        service = ModelService.get_instance()
        return HealthResponse(**service.get_health())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model service unhealthy: {str(e)}",
        )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
def get_model_info() -> ModelInfoResponse:
    """Retrieve metadata, exact feature representation, and class taxonomy."""
    try:
        service = ModelService.get_instance()
        return ModelInfoResponse(**service.get_model_info())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model info: {str(e)}",
        )


@app.get("/demo-samples", response_model=list[DemoSample], tags=["Demo"])
def get_demo_samples() -> list[DemoSample]:
    """Retrieve genuine CIC-IDS2017 traffic flows (RAW values) for instant evaluation."""
    try:
        service = ModelService.get_instance()
        return [DemoSample(**s) for s in service.get_demo_samples()]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load demo samples: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: PredictionRequest) -> PredictionResponse:
    """
    Run prediction on a single RAW network traffic flow.

    Requires every feature the promoted Champion consumes (see GET /model-info).
    Returns classification, confidence, probability distribution, and SHAP explanation.
    """
    try:
        service = ModelService.get_instance()
        return PredictionResponse(**service.predict_single(payload.features))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


def _validate_csv_upload(file: UploadFile) -> None:
    """Reject non-CSV uploads before reading the body."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported.",
        )


@app.post("/analyze", response_model=AnalysisResponse, tags=["SOC"])
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    """
    Full SOC analysis of an uploaded RAW traffic CSV.

    Classifies every flow, computes batched SHAP attributions, and returns
    dashboard aggregates: threat level, attack breakdown, confidence
    histogram, the ten most suspicious flows, global feature importance, and
    per-row detail with the top-3 contributing features.

    Accepts either the Parquet-release column names or the canonical
    CIC-IDS2017 CSV headers; aliases are resolved automatically.
    """
    _validate_csv_upload(file)
    try:
        content = await file.read()
        service = ModelService.get_instance()
        result = service.analyze_csv(content, file.filename, get_soc_thresholds())
        return AnalysisResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )
