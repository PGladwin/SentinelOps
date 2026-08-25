"""
SentinelOps API - Pydantic Schemas
==================================
Data validation and response serialization schemas for the inference service.

Feature values on the wire are RAW (packet counts, byte rates, microsecond
durations). The service applies training-derived scaling internally.
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
    feature_count: int = 0
    model_version: str = "unknown"
    stage: str = "unknown"
    input_contract: str = Field(
        default="raw",
        description="'raw' means the API scales inputs server-side.",
    )


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
        description="Every feature name the promoted Champion requires, "
                    "mapped to RAW numeric values. See GET /model-info.",
    )


class ShapExplanationItem(BaseModel):
    feature: str
    value: float
    shap_value: float
    increases_threat: bool = Field(
        ...,
        description="True when this contribution raises threat. Accounts for "
                    "the predicted class: on a BENIGN prediction a positive "
                    "SHAP value argues for benign traffic and lowers threat.",
    )
    direction: str = Field(..., description="'increases threat' or 'decreases threat'")


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    is_attack: bool
    probabilities: dict[str, float]
    model: str
    feature_count: int
    base_value: float = Field(
        default=0.0,
        description="TreeExplainer expected value for the predicted class; "
                    "SHAP contributions are offsets from this baseline.",
    )
    explanation: list[ShapExplanationItem]


# ---------------------------------------------------------------------------
# Demo Samples
# ---------------------------------------------------------------------------

class DemoSample(BaseModel):
    id: str
    label: str
    description: str
    features: dict[str, float]
    raw_label: Optional[str] = None
    source_file: Optional[str] = None
    champion_predicted: Optional[str] = None
    champion_confidence: Optional[float] = None
    correctly_classified: Optional[bool] = None


# ---------------------------------------------------------------------------
# SOC Analysis
# ---------------------------------------------------------------------------

class AnalysisSummary(BaseModel):
    filename: str
    threat_level: str = Field(..., description="HIGH | MEDIUM | LOW")
    total_connections: int
    total_attacks: int
    total_benign: int
    attack_rate: float
    rows_returned: int
    rows_explained: int
    truncated: bool
    imputed_columns: list[str] = Field(
        default_factory=list,
        description="Champion features absent from the upload and imputed from "
                    "training medians. Non-empty means a degraded analysis.",
    )
    n_imputed_columns: int = 0
    ground_truth_available: bool = False
    ground_truth_accuracy: Optional[float] = Field(
        default=None,
        description="Accuracy against the uploaded Label column, when present.",
    )


class AnalysisRow(BaseModel):
    flow_id: int
    prediction: str
    confidence: float
    is_attack: bool
    top_features: list[ShapExplanationItem]


class ConfidenceBucket(BaseModel):
    bucket: str
    count: int


class GlobalShapItem(BaseModel):
    rank: int
    feature: str
    mean_abs_shap: float


class AnalysisResponse(BaseModel):
    summary: AnalysisSummary
    class_breakdown: dict[str, int]
    confidence_histogram: list[ConfidenceBucket]
    top_suspicious: list[AnalysisRow]
    global_shap: list[GlobalShapItem]
    rows: list[AnalysisRow]
