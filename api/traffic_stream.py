"""
SentinelOps API - Live Traffic Replay
=====================================
Turns a static traffic CSV into a paced, classified flow feed so the console
shows detection happening rather than a report about detection that already
happened.

How it works
------------
A scenario file is loaded once and preprocessed once: column aliases resolved,
the champion's feature matrix built, ground truth extracted. Replay then walks
that matrix in small windows, scoring each window through the live Champion at
the moment it is due. Nothing is precomputed at load time, so starting a stream
is instant regardless of file size, and the verdicts are produced by the same
code path that serves ``/predict``.

Ground truth travels with each flow. The console can therefore show live
running accuracy against the real CIC-IDS2017 labels instead of asking the
audience to take the confidence score on faith.

The replay is stateless per subscriber: two browsers streaming at once each get
their own cursor over the same shared, read-only matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd

from api.config import settings
from src.data.inference_prep import (
    extract_labels,
    normalize_columns,
    prepare_with_report,
    resolve_truth_labels,
)

logger = logging.getLogger("sentinelops.api.stream")

# Display-envelope columns written by scripts/generate_demo_traffic.py. They are
# presentation metadata, never model input -- see that script's docstring.
ENVELOPE_COLUMNS = ["src_ip", "dst_ip", "dst_port", "protocol_name"]

# Fallbacks so a scenario CSV without an envelope still renders a usable log.
ENVELOPE_DEFAULTS = {
    "src_ip": "10.0.0.0",
    "dst_ip": "10.0.0.1",
    "dst_port": 0,
    "protocol_name": "TCP",
}


@dataclass
class ReplayStats:
    """Running totals for one subscriber's session."""

    flows: int = 0
    attacks: int = 0
    correct: int = 0
    labelled: int = 0
    loops: int = 0
    by_class: dict[str, int] = field(default_factory=dict)

    def observe(self, prediction: str, is_attack: bool, actual: Optional[str]) -> None:
        self.flows += 1
        self.attacks += int(is_attack)
        self.by_class[prediction] = self.by_class.get(prediction, 0) + 1
        if actual is not None:
            self.labelled += 1
            self.correct += int(actual == prediction)

    def snapshot(self, position: int, total: int, thresholds: dict[str, float]) -> dict[str, Any]:
        rate = self.attacks / self.flows if self.flows else 0.0
        return {
            "flows": self.flows,
            "attacks": self.attacks,
            "benign": self.flows - self.attacks,
            "attack_rate": round(rate, 6),
            "threat_level": threat_level(rate, thresholds),
            "by_class": dict(self.by_class),
            "accuracy": round(self.correct / self.labelled, 6) if self.labelled else None,
            "labelled": self.labelled,
            "position": position,
            "total": total,
            "loops": self.loops,
        }


def normalize_truth(truth: pd.Series) -> Optional[np.ndarray]:
    """
    Map a scenario's ground-truth column onto the canonical 8-class taxonomy.

    Scenario files vary: one may carry cleaned labels ("WebAttack"), another the
    raw dataset strings ("Web Attack - Brute Force"). Comparing a prediction
    against an unmapped raw label is always unequal, which would have the
    console report roughly chance accuracy on a model that is in fact correct.

    Returns None if the labels cannot be mapped, so the caller reports "no
    ground truth" rather than a fabricated score.
    """
    try:
        import yaml

        params_path = settings.project_root / "params.yaml"
        if not params_path.exists():
            return None
        with open(params_path, "r", encoding="utf-8") as f:
            label_map = yaml.safe_load(f)["cleaning"]["label_map"]

        mapped = resolve_truth_labels(truth, label_map)
        if not mapped.notna().any():
            return None

        # Rows whose label resolves to nothing are excluded from scoring by
        # carrying None, rather than being counted as misclassifications.
        return np.array([None if pd.isna(v) else str(v) for v in mapped], dtype=object)
    except Exception as e:
        logger.warning(f"Could not normalize scenario ground truth: {e}")
        return None


def threat_level(attack_rate: float, thresholds: dict[str, float]) -> str:
    """Map a rolling attack rate onto the SOC banner, using the params.yaml cutoffs."""
    if attack_rate >= thresholds.get("high", 0.20):
        return "HIGH"
    if attack_rate >= thresholds.get("medium", 0.05):
        return "MEDIUM"
    return "LOW"


class TrafficScenario:
    """One preprocessed traffic file, shared read-only across all subscribers."""

    def __init__(self, name: str, path: Path, service: Any):
        self.name = name
        self.path = path

        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            raise ValueError(f"Scenario '{name}' contains no rows.")

        df = normalize_columns(df)
        truth = extract_labels(df)

        # allow_missing mirrors /analyze: a scenario assembled from a different
        # export still replays, with any absent column imputed and reported.
        self.matrix, self.report = prepare_with_report(
            df, service.bundle, already_normalized=True, allow_missing=True
        )

        self.total = len(self.matrix)
        self.truth = normalize_truth(truth) if truth is not None else None
        self.envelope = self._build_envelope(df)
        # Raw values are what an analyst reads in an explanation; the matrix
        # holds scaled values, which are meaningless on screen.
        self.raw_features = self._build_raw_features(df, service)

        logger.info(
            f"Scenario '{name}': {self.total:,} flows, "
            f"{'labelled' if self.truth is not None else 'unlabelled'}, "
            f"{self.report['n_imputed_columns']} imputed column(s)"
        )

    def _build_envelope(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Extract the display columns, substituting defaults where absent."""
        envelope = {}
        for column, default in ENVELOPE_DEFAULTS.items():
            if column in df.columns:
                envelope[column] = df[column].to_numpy()
            else:
                envelope[column] = np.full(len(df), default)
        return envelope

    def _build_raw_features(self, df: pd.DataFrame, service: Any) -> np.ndarray:
        """Unscaled feature values, aligned to the champion's feature order."""
        display = df.reindex(columns=service.feature_names).apply(pd.to_numeric, errors="coerce")
        display = display.fillna(value={f: service.bundle["fill_values"][f] for f in service.feature_names})
        return display.to_numpy(dtype=np.float64, na_value=np.nan)

    def describe(self) -> dict[str, Any]:
        """Metadata for the scenario picker."""
        classes: dict[str, int] = {}
        if self.truth is not None:
            resolved = pd.Series([v for v in self.truth if v is not None], dtype=str)
            classes = {str(n): int(c) for n, c in resolved.value_counts().items()}

        hostile = sum(c for n, c in classes.items() if n != "BENIGN")
        return {
            "name": self.name,
            "filename": self.path.name,
            "total_flows": self.total,
            "labelled": self.truth is not None,
            "class_counts": classes,
            "attack_rate": round(hostile / self.total, 6) if self.total and classes else None,
            "imputed_columns": self.report["n_imputed_columns"],
        }


class TrafficReplayEngine:
    """Loads scenarios on demand and scores replay windows through the Champion."""

    def __init__(self, service: Any):
        self.service = service
        self._scenarios: dict[str, TrafficScenario] = {}

    # -- scenario discovery -------------------------------------------

    def available(self) -> list[str]:
        """Scenario names discoverable in the samples directory, preferred first."""
        directory = settings.samples_dir
        if not directory.exists():
            return []
        names = sorted(p.stem for p in directory.glob("*.csv"))
        default = settings.default_scenario
        # Surface the purpose-built stream first; the picker shows this order.
        return [default] + [n for n in names if n != default] if default in names else names

    def get(self, name: Optional[str] = None) -> TrafficScenario:
        """Return a scenario, loading and caching it on first use."""
        name = name or settings.default_scenario

        if name not in self._scenarios:
            path = settings.samples_dir / f"{name}.csv"
            # Guard against a crafted name escaping the samples directory.
            if not path.resolve().is_relative_to(settings.samples_dir.resolve()) or not path.exists():
                raise FileNotFoundError(
                    f"Scenario '{name}' not found. Available: {', '.join(self.available()) or 'none'}"
                )
            self._scenarios[name] = TrafficScenario(name, path, self.service)

        return self._scenarios[name]

    # -- scoring -------------------------------------------------------

    def score_window(self, scenario: TrafficScenario, start: int, end: int) -> list[dict[str, Any]]:
        """
        Classify and explain rows [start, end) of a scenario.

        Runs the full Champion path -- probabilities plus a TreeExplainer pass --
        on the window only, which keeps a live stream's per-tick cost flat no
        matter how long the session has been running.
        """
        window = scenario.matrix[start:end]
        if len(window) == 0:
            return []

        probabilities = self.service.model.predict_proba(window)
        predicted = probabilities.argmax(axis=1)
        confidences = probabilities.max(axis=1)

        names = [self.service.inverse_encoding.get(int(p), f"Class_{p}") for p in predicted]

        shap_window: Optional[np.ndarray] = None
        if self.service.explainer is not None:
            values, _ = self.service._shap_for_matrix(window)
            shap_window = (
                values[np.arange(len(window)), :, predicted] if values.ndim == 3 else values
            )

        flows = []
        for offset in range(len(window)):
            index = start + offset
            prediction = names[offset]
            is_attack = prediction != "BENIGN"
            # None here means either an unlabelled scenario or a label that did
            # not resolve to the taxonomy; both are excluded from live accuracy.
            actual = scenario.truth[index] if scenario.truth is not None else None

            explanation = []
            if shap_window is not None:
                explanation = self.service._explain_row(
                    shap_window[offset], scenario.raw_features[index], is_attack, top_k=3
                )

            # Full distribution, not just the winning confidence: on a close
            # call the runner-up class is the most informative thing an analyst
            # can see, and eight floats per flow is negligible on the wire.
            distribution = {
                self.service.inverse_encoding.get(i, f"Class_{i}"): round(float(p), 6)
                for i, p in enumerate(probabilities[offset])
            }

            flows.append({
                "index": index,
                "probabilities": distribution,
                "src_ip": str(scenario.envelope["src_ip"][index]),
                "dst_ip": str(scenario.envelope["dst_ip"][index]),
                "dst_port": int(scenario.envelope["dst_port"][index]),
                "protocol": str(scenario.envelope["protocol_name"][index]),
                "prediction": prediction,
                "confidence": round(float(confidences[offset]), 4),
                "is_attack": is_attack,
                "actual": actual,
                "correct": (actual == prediction) if actual is not None else None,
                "top_features": explanation,
            })

        return flows


def paced_windows(
    total: int,
    flows_per_tick: int,
    start: int,
    loop: bool,
    limit: Optional[int],
) -> Iterator[tuple[int, int, bool]]:
    """
    Yield (start, end, wrapped) windows over a scenario, honouring loop and limit.

    Separated from the async transport so the pacing arithmetic -- the part that
    is easy to get subtly wrong at a file boundary -- is straightforward to test
    without a running event loop.
    """
    cursor = start
    emitted = 0

    while limit is None or emitted < limit:
        if cursor >= total:
            if not loop:
                return
            # Advance the cursor past the wrap window as well. Resetting it to
            # 0 and yielding without moving it made the next iteration re-emit
            # the same opening window, so every loop replayed its first flows
            # twice -- visible in a long demo as a stutter at each restart.
            end = min(flows_per_tick, total)
            yield 0, end, True
            cursor = end
        else:
            end = min(cursor + flows_per_tick, total)
            yield cursor, end, False
            cursor = end

        emitted += flows_per_tick


def synthetic_timestamps(count: int, seconds_per_tick: float) -> list[str]:
    """
    Wall-clock stamps for one emitted window.

    The corpus records capture-relative timings only, so the log view needs a
    clock. Stamps are spread across the tick interval rather than sharing one
    instant, which is what makes the feed read as a stream instead of a batch.
    """
    now = datetime.now(timezone.utc)
    if count <= 0:
        return []
    step = seconds_per_tick / count
    return [
        (now + timedelta(seconds=step * i)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        for i in range(count)
    ]
