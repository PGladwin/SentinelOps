"""
SentinelOps - Model Governance (Champion / Challenger)
=======================================================
The safety gate that decides whether a newly trained model may replace the one
currently serving production traffic.

Design intent:
  Automated retraining is only safe if something can refuse its output. This
  module is that something. A challenger is promoted ONLY when it clears both
  absolute safety floors and beats the incumbent champion on the metrics that
  matter for intrusion detection.

Decision rule
-------------
Absolute gates (evaluated regardless of any incumbent):
  1. mean attack FNR <= governance.max_mean_fnr
  2. every attack class with test support has FNR <= governance.max_per_class_fnr

Relative gates (only when an incumbent champion exists):
  3. macro F1 >= champion macro F1 + governance.min_f1_improvement
  4. mean attack FNR <= champion mean attack FNR

All four must hold. With no incumbent, clearing the absolute gates bootstraps
the first champion.

Why FNR leads the rule:
  In an IDS a false negative is a missed intrusion. Accuracy is a poor guide
  under this class imbalance -- a model predicting BENIGN for everything scores
  ~89% accuracy on this dataset while detecting nothing. The gate therefore
  refuses any challenger that misses more attacks than the incumbent, even if
  its headline F1 is higher.

Every decision is appended to reports/governance_log.jsonl with a real UTC
timestamp. Nothing is ever back-dated or hand-written.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DECISION_PROMOTED = "PROMOTED"
DECISION_REJECTED = "REJECTED"

GOVERNANCE_LOG_FILENAME = "governance_log.jsonl"


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_gate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Pull the governance-relevant figures out of an evaluate_model() result.

    Parameters
    ----------
    metrics : dict
        Output of src.models.evaluate.evaluate_model.

    Returns
    -------
    dict
        macro_f1, mean_attack_fnr, worst per-class attack FNR and its class,
        and the full per-class FNR map (zero-support classes excluded).
    """
    summary = metrics.get("summary", {})
    per_class = metrics.get("per_class", {})

    per_class_fnr: dict[str, float] = {}
    for cls_name, stats in per_class.items():
        if stats.get("insufficient_test_support") or stats.get("fnr") is None:
            continue
        per_class_fnr[cls_name] = float(stats["fnr"])

    attack_fnr = {c: v for c, v in per_class_fnr.items() if c != "BENIGN"}
    if attack_fnr:
        worst_class = max(attack_fnr, key=lambda c: attack_fnr[c])
        worst_value = attack_fnr[worst_class]
    else:
        worst_class, worst_value = None, None

    return {
        "macro_f1": float(summary.get("macro_f1", 0.0)),
        "mean_attack_fnr": float(summary.get("mean_attack_fnr", 1.0)),
        "macro_attack_f1": float(summary.get("macro_attack_f1", 0.0)),
        "accuracy": float(summary.get("accuracy", 0.0)),
        "worst_attack_class": worst_class,
        "worst_attack_fnr": worst_value,
        "per_class_fnr": per_class_fnr,
        "classes_without_test_support": [
            c for c, s in per_class.items() if s.get("insufficient_test_support")
        ],
    }


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def evaluate_challenger(
    challenger_name: str,
    challenger_metrics: dict[str, Any],
    champion_name: Optional[str],
    champion_metrics: Optional[dict[str, Any]],
    governance_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Decide whether a challenger should replace the incumbent champion.

    Parameters
    ----------
    challenger_name : str
        Identifier of the newly trained model.
    challenger_metrics : dict
        evaluate_model() output for the challenger.
    champion_name : str, optional
        Incumbent identifier, or None when bootstrapping.
    champion_metrics : dict, optional
        evaluate_model() output for the incumbent, or None.
    governance_cfg : dict
        params.yaml:governance block.

    Returns
    -------
    dict
        Decision record: decision, reasons, gate-by-gate results, and both
        models' gate metrics. Suitable for direct JSON serialization.
    """
    min_f1_improvement = float(governance_cfg.get("min_f1_improvement", 0.001))
    max_mean_fnr = float(governance_cfg.get("max_mean_fnr", 0.10))
    max_per_class_fnr = float(governance_cfg.get("max_per_class_fnr", 0.30))

    chal = extract_gate_metrics(challenger_metrics)
    champ = extract_gate_metrics(champion_metrics) if champion_metrics else None

    gates: list[dict[str, Any]] = []

    # --- Gate 1: absolute mean attack FNR ceiling --------------------
    g1_pass = chal["mean_attack_fnr"] <= max_mean_fnr
    gates.append({
        "gate": "absolute_mean_attack_fnr",
        "passed": g1_pass,
        "detail": (
            f"mean attack FNR {chal['mean_attack_fnr']:.4f} "
            f"{'<=' if g1_pass else '>'} ceiling {max_mean_fnr:.4f}"
        ),
    })

    # --- Gate 2: absolute per-class FNR ceiling ----------------------
    breaches = {
        c: v for c, v in chal["per_class_fnr"].items()
        if c != "BENIGN" and v > max_per_class_fnr
    }
    g2_pass = not breaches
    gates.append({
        "gate": "absolute_per_class_fnr",
        "passed": g2_pass,
        "detail": (
            f"all attack classes within per-class FNR ceiling {max_per_class_fnr:.2f}"
            if g2_pass else
            "per-class FNR ceiling breached by " + ", ".join(
                f"{c} ({v:.4f})" for c, v in sorted(breaches.items(), key=lambda kv: -kv[1])
            )
        ),
    })

    # --- Gates 3 & 4: relative to the incumbent ----------------------
    if champ is None:
        gates.append({
            "gate": "relative_to_champion",
            "passed": True,
            "detail": "No incumbent champion; absolute gates alone govern the first promotion.",
        })
        g3_pass = g4_pass = True
    else:
        required_f1 = champ["macro_f1"] + min_f1_improvement
        g3_pass = chal["macro_f1"] >= required_f1
        gates.append({
            "gate": "macro_f1_improvement",
            "passed": g3_pass,
            "detail": (
                f"macro F1 {chal['macro_f1']:.4f} vs champion {champ['macro_f1']:.4f} "
                f"(required >= {required_f1:.4f}, margin {min_f1_improvement})"
            ),
        })

        g4_pass = chal["mean_attack_fnr"] <= champ["mean_attack_fnr"]
        gates.append({
            "gate": "attack_fnr_not_worse",
            "passed": g4_pass,
            "detail": (
                f"mean attack FNR {chal['mean_attack_fnr']:.4f} vs champion "
                f"{champ['mean_attack_fnr']:.4f} (must not regress)"
            ),
        })

    promoted = all(g["passed"] for g in gates)
    decision = DECISION_PROMOTED if promoted else DECISION_REJECTED

    failed = [g for g in gates if not g["passed"]]
    if promoted:
        reasons = [f"All {len(gates)} governance gates passed."] + [g["detail"] for g in gates]
    else:
        reasons = [
            f"Rejected: {len(failed)} of {len(gates)} governance gates failed."
        ] + [f"FAILED [{g['gate']}]: {g['detail']}" for g in failed]

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "challenger": challenger_name,
        "champion": champion_name,
        "decision": decision,
        "promoted": promoted,
        "gates": gates,
        "reasons": reasons,
        "thresholds": {
            "min_f1_improvement": min_f1_improvement,
            "max_mean_fnr": max_mean_fnr,
            "max_per_class_fnr": max_per_class_fnr,
        },
        "challenger_metrics": chal,
        "champion_metrics": champ,
    }

    log_fn = logger.info if promoted else logger.warning
    log_fn(f"GOVERNANCE {decision}: challenger '{challenger_name}' vs champion '{champion_name}'")
    for reason in reasons:
        log_fn(f"  {reason}")

    return record


# ---------------------------------------------------------------------------
# Append-only decision log
# ---------------------------------------------------------------------------

def append_decision(record: dict[str, Any], reports_dir: str | Path) -> Path:
    """
    Append a decision to the governance log.

    The log is append-only JSONL so the promotion history is auditable and
    cannot be silently rewritten by a later run.
    """
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / GOVERNANCE_LOG_FILENAME

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(f"Governance decision appended to: {log_path}")
    return log_path


def load_decisions(reports_dir: str | Path) -> list[dict[str, Any]]:
    """Read the full governance history, oldest first. Returns [] if absent."""
    log_path = Path(reports_dir) / GOVERNANCE_LOG_FILENAME
    if not log_path.exists():
        return []

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed governance log line {line_no}.")
    return records


def current_champion(reports_dir: str | Path) -> Optional[dict[str, Any]]:
    """
    Return the most recent promotion record, i.e. the model currently serving.

    Returns None when nothing has ever been promoted.
    """
    for record in reversed(load_decisions(reports_dir)):
        if record.get("promoted"):
            return record
    return None
