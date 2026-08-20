"""
SentinelOps API - Pydantic Schemas
==================================
Data validation and response serialization schemas for the inference service.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health & Model Information
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = True
    model: str = "XGBoost Champion"
    feature_count: int = 40


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    feature_count: int
    features: list[str]
    class_mapping: dict[str, int]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Prediction & Explanation
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Dictionary mapping exactly 40 required feature names to numeric values."
    )


class ShapExplanationItem(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str = Field(
        ...,
        description="'increases risk/probability' or 'decreases risk/probability'"
    )


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    is_attack: bool
    probabilities: dict[str, float]
    model: str
    feature_count: int
    explanation: list[ShapExplanationItem]


# ---------------------------------------------------------------------------
# Demo Samples
# ---------------------------------------------------------------------------

class DemoSample(BaseModel):
    id: str
    label: str
    description: str
    features: dict[str, float]


# ---------------------------------------------------------------------------
# Batch Prediction
# ---------------------------------------------------------------------------

class BatchPredictionRow(BaseModel):
    flow_id: int
    prediction: str
    confidence: float
    is_attack: bool


class BatchPredictionResponse(BaseModel):
    total_samples: int
    prediction_counts: dict[str, int]
    predictions: list[BatchPredictionRow]
