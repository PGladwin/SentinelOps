"""
SentinelOps - Governance Gate Tests
====================================
The champion/challenger gate is the safety mechanism that makes automated
retraining survivable: it is the only thing standing between a bad model and
production. These tests pin its decision rule.

Covers promote, reject, bootstrap, tie-breaking, absolute ceilings, and the
append-only decision log.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlops.governance import (
    DECISION_PROMOTED,
    DECISION_REJECTED,
    append_decision,
    current_champion,
    evaluate_challenger,
    extract_gate_metrics,
    load_decisions,
)

GOV = {"min_f1_improvement": 0.001, "max_mean_fnr": 0.10, "max_per_class_fnr": 0.30}


def make_metrics(macro_f1, mean_attack_fnr, per_class_fnr=None, zero_support=()):
    """Build a minimal evaluate_model()-shaped result."""
    per_class_fnr = per_class_fnr or {"DoS": 0.01, "DDoS": 0.01}
    per_class = {
        "BENIGN": {"support": 1000, "precision": 0.99, "recall": 0.99, "f1": 0.99,
                   "fnr": 0.01, "insufficient_test_support": False},
    }
    for cls, fnr in per_class_fnr.items():
        per_class[cls] = {
            "support": 100, "precision": 0.9, "recall": 1 - fnr,
            "f1": 0.9, "fnr": fnr, "insufficient_test_support": False,
        }
    for cls in zero_support:
        per_class[cls] = {
            "support": 0, "precision": None, "recall": None, "f1": None,
            "fnr": None, "insufficient_test_support": True,
        }
    return {
        "summary": {
            "macro_f1": macro_f1, "mean_attack_fnr": mean_attack_fnr,
            "macro_attack_f1": macro_f1, "accuracy": 0.99,
        },
        "per_class": per_class,
    }


class TestGateDecisions:
    def test_bootstrap_promotes_first_model_passing_absolute_gates(self):
        rec = evaluate_challenger("first", make_metrics(0.70, 0.05), None, None, GOV)
        assert rec["decision"] == DECISION_PROMOTED

    def test_bootstrap_rejects_model_breaching_absolute_ceiling(self):
        """No incumbent does not mean anything gets promoted."""
        rec = evaluate_challenger("bad_first", make_metrics(0.90, 0.50), None, None, GOV)
        assert rec["decision"] == DECISION_REJECTED

    def test_promotes_when_better_on_both_axes(self):
        champ = make_metrics(0.90, 0.05)
        chal = make_metrics(0.95, 0.03)
        rec = evaluate_challenger("chal", chal, "champ", champ, GOV)
        assert rec["decision"] == DECISION_PROMOTED

    def test_rejects_when_f1_gain_below_margin(self):
        """A 0.0005 gain does not clear a 0.001 required margin."""
        champ = make_metrics(0.9000, 0.05)
        chal = make_metrics(0.9005, 0.04)
        rec = evaluate_challenger("chal", chal, "champ", champ, GOV)
        assert rec["decision"] == DECISION_REJECTED
        assert any(g["gate"] == "macro_f1_improvement" and not g["passed"] for g in rec["gates"])

    def test_rejects_when_attack_fnr_regresses_despite_higher_f1(self):
        """Higher F1 must not buy a worse miss rate: FNR has veto power."""
        champ = make_metrics(0.90, 0.02)
        chal = make_metrics(0.99, 0.06)
        rec = evaluate_challenger("chal", chal, "champ", champ, GOV)
        assert rec["decision"] == DECISION_REJECTED
        assert any(g["gate"] == "attack_fnr_not_worse" and not g["passed"] for g in rec["gates"])

    def test_equal_attack_fnr_is_allowed(self):
        champ = make_metrics(0.90, 0.02)
        chal = make_metrics(0.95, 0.02)
        rec = evaluate_challenger("chal", chal, "champ", champ, GOV)
        assert rec["decision"] == DECISION_PROMOTED

    def test_rejects_on_per_class_fnr_breach(self):
        champ = make_metrics(0.50, 0.09)
        chal = make_metrics(0.99, 0.05, per_class_fnr={"DoS": 0.01, "Botnet": 0.85})
        rec = evaluate_challenger("chal", chal, "champ", champ, GOV)
        assert rec["decision"] == DECISION_REJECTED
        breach = next(g for g in rec["gates"] if g["gate"] == "absolute_per_class_fnr")
        assert "Botnet" in breach["detail"]

    def test_zero_support_class_does_not_trigger_breach(self):
        """A class with no test rows was never evaluated; it is not a failure."""
        chal = make_metrics(0.90, 0.03, zero_support=("Infiltration",))
        rec = evaluate_challenger("chal", chal, None, None, GOV)
        assert rec["decision"] == DECISION_PROMOTED
        assert "Infiltration" in rec["challenger_metrics"]["classes_without_test_support"]

    def test_reasons_explain_every_failure(self):
        chal = make_metrics(0.10, 0.60, per_class_fnr={"DoS": 0.9})
        rec = evaluate_challenger("chal", chal, None, None, GOV)
        failed = [g for g in rec["gates"] if not g["passed"]]
        assert len(rec["reasons"]) == len(failed) + 1


class TestGateMetricExtraction:
    def test_identifies_worst_attack_class(self):
        m = make_metrics(0.9, 0.2, per_class_fnr={"DoS": 0.05, "Botnet": 0.42, "DDoS": 0.1})
        gm = extract_gate_metrics(m)
        assert gm["worst_attack_class"] == "Botnet"
        assert gm["worst_attack_fnr"] == 0.42

    def test_benign_excluded_from_attack_aggregates(self):
        gm = extract_gate_metrics(make_metrics(0.9, 0.05))
        assert "BENIGN" in gm["per_class_fnr"]
        assert gm["worst_attack_class"] != "BENIGN"


# pytest's log_dir fixture cannot create its base directory on this machine
# (PermissionError under AppData\Local\Temp), so these use tempfile directly,
# matching the pattern already established in tests/test_phase1.py.
@pytest.fixture
def log_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestDecisionLog:
    def test_append_and_reload_roundtrip(self, log_dir):
        rec = evaluate_challenger("m1", make_metrics(0.9, 0.02), None, None, GOV)
        append_decision(rec, log_dir)
        loaded = load_decisions(log_dir)
        assert len(loaded) == 1
        assert loaded[0]["challenger"] == "m1"

    def test_log_is_append_only(self, log_dir):
        for name, f1 in [("m1", 0.90), ("m2", 0.95), ("m3", 0.99)]:
            append_decision(
                evaluate_challenger(name, make_metrics(f1, 0.02), None, None, GOV), log_dir
            )
        assert [d["challenger"] for d in load_decisions(log_dir)] == ["m1", "m2", "m3"]

    def test_current_champion_is_latest_promotion(self, log_dir):
        append_decision(evaluate_challenger("good", make_metrics(0.90, 0.02), None, None, GOV), log_dir)
        append_decision(
            evaluate_challenger("bad", make_metrics(0.99, 0.80), "good", make_metrics(0.90, 0.02), GOV),
            log_dir,
        )
        assert current_champion(log_dir)["challenger"] == "good"

    def test_no_champion_when_nothing_promoted(self, log_dir):
        append_decision(evaluate_challenger("bad", make_metrics(0.5, 0.9), None, None, GOV), log_dir)
        assert current_champion(log_dir) is None

    def test_missing_log_returns_empty(self, log_dir):
        assert load_decisions(log_dir / "nope") == []
