"""
SentinelOps API Test Suite
==========================
Unit & integration tests for FastAPI backend endpoints:
  - GET /health
  - GET /model-info
  - GET /demo-samples
  - POST /predict (valid input, confidence, probability distribution, SHAP explanation)
  - POST /predict (invalid input: missing features, wrong fields)
  - POST /batch-predict (CSV upload and aggregation)
"""

import io
import pytest
from fastapi.testclient import TestClient
import pandas as pd

from api.main import app
from api.model_service import ModelService

client = TestClient(app)


def test_get_health():
    """Verify health endpoint returns status 200 and indicates model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["feature_count"] == 40
    assert "XGBoost" in data["model"]


def test_get_model_info():
    """Verify model-info endpoint returns 40 features, class mappings, and metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "xgboost_top40"
    assert data["feature_count"] == 40
    assert len(data["features"]) == 40
    assert len(data["class_mapping"]) == 8
    assert "BENIGN" in data["class_mapping"]


def test_get_demo_samples():
    """Verify demo-samples endpoint returns real CIC-IDS2017 samples with 40 features."""
    response = client.get("/demo-samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    assert len(samples) >= 5

    for s in samples:
        assert "id" in s
        assert "label" in s
        assert "description" in s
        assert len(s["features"]) == 40


def test_post_predict_valid_sample():
    """Verify prediction endpoint with real demo traffic returns valid predictions and SHAP explanations."""
    # Fetch a real demo sample
    demo_resp = client.get("/demo-samples")
    samples = demo_resp.json()
    assert len(samples) > 0

    sample = samples[0]
    payload = {"features": sample["features"]}

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()

    assert "prediction" in res_data
    assert "confidence" in res_data
    assert 0.0 <= res_data["confidence"] <= 1.0
    assert "is_attack" in res_data
    assert isinstance(res_data["is_attack"], bool)

    # Check probabilities
    assert len(res_data["probabilities"]) == 8
    assert abs(sum(res_data["probabilities"].values()) - 1.0) < 1e-3

    # Check SHAP explanation items
    assert "explanation" in res_data
    assert len(res_data["explanation"]) > 0
    for exp in res_data["explanation"]:
        assert "feature" in exp
        assert "value" in exp
        assert "shap_value" in exp
        assert "direction" in exp


def test_post_predict_missing_features_raises_422():
    """Verify that omitting required features triggers HTTP 422 Unprocessable Entity."""
    incomplete_features = {"Packet Length Max": 100.0, "Flow Duration": 5000.0}
    payload = {"features": incomplete_features}

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    assert "Missing required feature" in resp.json()["detail"]


def test_post_predict_invalid_payload_schema():
    """Verify invalid payload format triggers 422."""
    resp = client.post("/predict", json={"wrong_key": 123})
    assert resp.status_code == 422


def test_post_batch_predict_csv():
    """Verify batch CSV prediction endpoint."""
    service = ModelService.get_instance()
    feature_names = service.feature_names

    # Construct synthetic CSV in memory with the 40 required columns
    df = pd.DataFrame(
        [[0.5] * 40, [100.0] * 40, [2.0] * 40],
        columns=feature_names,
    )
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    files = {"file": ("test_flows.csv", csv_bytes, "text/csv")}
    resp = client.post("/batch-predict", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_samples"] == 3
    assert len(data["predictions"]) == 3
    assert "prediction_counts" in data
    assert sum(data["prediction_counts"].values()) == 3
