"""
SentinelOps API Test Suite
==========================
Unit & integration tests for FastAPI backend endpoints:
  - GET /health
  - GET /model-info
  - GET /demo-samples
  - POST /predict (valid input, confidence, probability distribution, SHAP explanation)
  - POST /predict (invalid input: missing features, wrong fields)
  - POST /analyze (SOC scan over an uploaded CSV)
"""

import io
import json

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
    assert data["feature_count"] > 0
    assert "XGBoost" in data["model"]


def test_get_model_info():
    """Verify model-info endpoint returns 40 features, class mappings, and metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    # Feature count is whatever governance promoted, not a fixed 40.
    assert data["feature_count"] == len(data["features"])
    assert data["feature_count"] > 0
    assert len(data["class_mapping"]) == 8
    assert "BENIGN" in data["class_mapping"]


def test_get_demo_samples():
    """Verify demo-samples endpoint returns real CIC-IDS2017 samples with 40 features."""
    response = client.get("/demo-samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    assert len(samples) >= 5
    feature_count = client.get("/model-info").json()["feature_count"]

    for s in samples:
        assert "id" in s
        assert "label" in s
        assert "description" in s
        assert len(s["features"]) == feature_count


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


# ---------------------------------------------------------------------------
# SOC /analyze endpoint
# ---------------------------------------------------------------------------

def _demo_csv_bytes(n_rows: int = 5) -> bytes:
    """Build a small RAW-valued CSV from the served demo samples."""
    samples = client.get("/demo-samples").json()
    rows = [s["features"] for s in samples][:n_rows]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def test_analyze_returns_soc_dashboard_payload():
    """Verify /analyze returns threat level, breakdown, histogram and SHAP."""
    files = {"file": ("flows.csv", _demo_csv_bytes(), "text/csv")}
    resp = client.post("/analyze", files=files)
    assert resp.status_code == 200
    data = resp.json()

    summary = data["summary"]
    assert summary["threat_level"] in {"HIGH", "MEDIUM", "LOW"}
    assert summary["total_connections"] == summary["total_attacks"] + summary["total_benign"]
    assert 0.0 <= summary["attack_rate"] <= 1.0

    assert len(data["class_breakdown"]) == 8
    assert sum(data["class_breakdown"].values()) == summary["total_connections"]
    assert sum(b["count"] for b in data["confidence_histogram"]) == summary["total_connections"]
    assert len(data["global_shap"]) > 0

    for row in data["rows"]:
        assert row["prediction"] in data["class_breakdown"]
        assert 0.0 <= row["confidence"] <= 1.0
        assert len(row["top_features"]) <= 3
        for feat in row["top_features"]:
            assert feat["direction"] in {"increases threat", "decreases threat"}


def test_analyze_detects_attacks_in_demo_traffic():
    """The known-attack demo flows must be classified as attacks end to end."""
    samples = client.get("/demo-samples").json()
    attacks = [s for s in samples if s["label"] != "BENIGN"]
    csv_bytes = pd.DataFrame([s["features"] for s in attacks]).to_csv(index=False).encode("utf-8")

    resp = client.post("/analyze", files={"file": ("attacks.csv", csv_bytes, "text/csv")})
    assert resp.status_code == 200
    data = resp.json()

    assert data["summary"]["total_attacks"] >= len(attacks) - 1
    assert data["summary"]["threat_level"] == "HIGH"


def test_analyze_rejects_non_csv():
    resp = client.post("/analyze", files={"file": ("flows.txt", b"nope", "text/plain")})
    assert resp.status_code == 400


def test_analyze_missing_columns_returns_422():
    csv_bytes = b"Flow Duration,Total Fwd Packets\n100,5\n"
    resp = client.post("/analyze", files={"file": ("bad.csv", csv_bytes, "text/csv")})
    assert resp.status_code == 422
    assert "missing" in resp.json()["detail"].lower()


def test_analyze_empty_csv_returns_422():
    samples = client.get("/demo-samples").json()
    header = ",".join(samples[0]["features"].keys()).encode("utf-8") + b"\n"
    resp = client.post("/analyze", files={"file": ("empty.csv", header, "text/csv")})
    assert resp.status_code == 422


def test_predict_rejects_non_finite_values():
    """
    NaN/Inf previously flowed straight into the model unchecked.

    Asserted at the service layer: the JSON encoder already refuses to
    serialize inf, so a standards-compliant HTTP client cannot reach the
    guard. Non-JSON callers and direct library users still can.
    """
    service = ModelService.get_instance()
    features = {f: 1.0 for f in service.feature_names}
    features[service.feature_names[0]] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        service.predict_single(features)


def test_predict_accepts_non_finite_over_http_via_raw_body():
    """FastAPI parses the JSON `Infinity` literal, so the guard returns 422."""
    samples = client.get("/demo-samples").json()
    features = dict(samples[0]["features"])
    key = next(iter(features))
    body = json.dumps({"features": features})
    body = body.replace(f'"{key}": {features[key]}', f'"{key}": Infinity', 1)

    resp = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422


def test_predict_reports_base_value():
    samples = client.get("/demo-samples").json()
    resp = client.post("/predict", json={"features": samples[0]["features"]})
    assert resp.status_code == 200
    assert "base_value" in resp.json()
