"""
SentinelOps - Local Inference REST API
======================================
FastAPI application exposing real-time inference, model introspection,
SHAP explanations, and batch traffic flow evaluation for the SentinelOps Champion.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from api.model_service import ModelService
from api.schemas import (
    BatchPredictionResponse,
    DemoSample,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger("sentinelops.api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure ModelService is pre-loaded at server startup."""
    logger.info("Initializing SentinelOps API service and loading Champion model...")
    try:
        service = ModelService.get_instance()
        logger.info(f"SentinelOps API ready: Champion '{service.metadata.get('model_name')}' loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model service on startup: {e}", exc_info=True)
    yield
    logger.info("Shutting down SentinelOps API...")


app = FastAPI(
    title="SentinelOps Inference API",
    description="Real-Time Network Intrusion Detection & Explainability Service (XGBoost Top-40 Champion)",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Verify backend health and model loading status."""
    try:
        service = ModelService.get_instance()
        info = service.get_health()
        return HealthResponse(
            status=info["status"],
            model_loaded=info["model_loaded"],
            model=info["model"],
            feature_count=info["feature_count"],
        )
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
        info = service.get_model_info()
        return ModelInfoResponse(
            model_name=info["model_name"],
            model_type=info["model_type"],
            feature_count=info["feature_count"],
            features=info["features"],
            class_mapping=info["class_mapping"],
            metadata=info["metadata"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model info: {str(e)}",
        )


@app.get("/demo-samples", response_model=list[DemoSample], tags=["Demo"])
def get_demo_samples() -> list[DemoSample]:
    """Retrieve real CIC-IDS2017 test set traffic samples for instant live evaluation."""
    try:
        service = ModelService.get_instance()
        return service.get_demo_samples()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load demo samples: {str(e)}",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: PredictionRequest) -> PredictionResponse:
    """
    Run prediction on a single network traffic flow (40 features).
    Returns classification, confidence, full probability distribution, and SHAP explanation.
    """
    try:
        service = ModelService.get_instance()
        res = service.predict_single(payload.features)
        return PredictionResponse(**res)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Inference"])
async def batch_predict(file: UploadFile = File(...)) -> BatchPredictionResponse:
    """Upload a CSV of network flows and receive batch predictions and threat distribution."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported for batch inference.",
        )
    try:
        content = await file.read()
        service = ModelService.get_instance()
        res = service.predict_batch(content, file.filename)
        return BatchPredictionResponse(**res)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )
