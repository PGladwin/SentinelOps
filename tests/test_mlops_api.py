"""
SentinelOps - MLOps Control Panel API Tests
============================================
Verifies the /mlops/* endpoints that back the lifecycle dashboard.

These read reports/mlops_state.json rather than querying MLflow at runtime, so
the tests also pin the contract that the deployed container needs no MLflow
server or DVC cache present.
"""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)

STATE_PATH = Path(__file__).parent.parent / "reports" / "mlops_state.json"

requires_state = pytest.mark.skipif(
    not STATE_PATH.exists(),
    reason="reports/mlops_state.json not generated; run the sweep or `dvc repro`.",
)


@requires_state
def test_state_endpoint_returns_all_sections():
    resp = client.get("/mlops/state")
    assert resp.status_code == 200
    state = resp.json()
    for section in ("runs", "registry", "governance", "production_metrics", "pipeline", "drift"):
        assert section in state, f"missing section: {section}"
    assert "generated_at" in state


@requires_state
def test_runs_carry_the_metrics_the_panel_displays():
    runs = client.get("/mlops/runs").json()
    assert isinstance(runs, list)
    for run in runs:
        for field in ("run_name", "model_type", "macro_f1", "mean_attack_fnr", "started_at"):
            assert field in run


@requires_state
def test_registry_has_at_most_one_champion():
    """The champion alias is a pointer, so exactly one version may hold it."""
    registry = client.get("/mlops/registry").json()
    champions = [v for v in registry["versions"] if v.get("is_champion")]
    assert len(champions) <= 1
    if registry.get("champion_version"):
        assert len(champions) == 1
        assert champions[0]["version"] == registry["champion_version"]


@requires_state
def test_governance_history_is_internally_consistent():
    gov = client.get("/mlops/governance").json()
    assert gov["total_decisions"] == len(gov["history"])
    assert gov["promotions"] + gov["rejections"] == gov["total_decisions"]

    for entry in gov["history"]:
        assert entry["decision"] in {"PROMOTED", "REJECTED"}
        assert entry["promoted"] == (entry["decision"] == "PROMOTED")
        # A decision must be explainable: every gate result is retained.
        assert entry["gates"], "decision recorded with no gate results"
        assert entry["reasons"]


@requires_state
def test_governance_decisions_are_chronological():
    """The log is append-only, so timestamps must not go backwards."""
    history = client.get("/mlops/governance").json()["history"]
    stamps = [h["timestamp"] for h in history]
    assert stamps == sorted(stamps)


@requires_state
def test_current_champion_matches_latest_promotion():
    gov = client.get("/mlops/governance").json()
    promoted = [h for h in gov["history"] if h["promoted"]]
    if promoted:
        assert gov["current_champion"] == promoted[-1]["challenger"]
    else:
        assert gov["current_champion"] is None


@requires_state
def test_production_metrics_flag_zero_support_honestly():
    """
    A class with no test rows must read NO_TEST_SUPPORT, never a passing 0.0 --
    otherwise the panel would overstate detection coverage.
    """
    metrics = client.get("/mlops/metrics").json()
    for cls in metrics.get("classes", []):
        if cls["status"] == "NO_TEST_SUPPORT":
            assert cls["fnr"] is None
            assert cls["support"] == 0
        else:
            assert cls["status"] in {"OK", "BREACH"}
            assert cls["fnr"] is not None


@requires_state
def test_pipeline_reports_availability():
    pipeline = client.get("/mlops/pipeline").json()
    assert "available" in pipeline
    if pipeline["available"]:
        names = [s["name"] for s in pipeline["stages"]]
        assert names, "dvc.yaml present but no stages parsed"
        for stage in pipeline["stages"]:
            assert stage["cmd"]


@requires_state
def test_drift_reports_availability_rather_than_failing():
    drift = client.get("/mlops/drift").json()
    assert "available" in drift
    if not drift["available"]:
        assert "reason" in drift
