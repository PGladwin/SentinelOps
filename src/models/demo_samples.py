"""
SentinelOps - Demo Sample Bank
===============================
Builds a small, self-contained bank of genuine CIC-IDS2017 traffic flows for
the Live Analysis demo, stored as RAW (unscaled) feature values.

Why this exists:
  The API previously loaded the entire test Parquet at startup just to pull a
  handful of demo rows. At full-dataset scale that is ~446k rows and will
  exhaust a 512 MB container. Baking the samples into a small JSON artifact
  removes the pandas/Parquet dependency from the serving hot path entirely.

Design notes:
  - Values are RAW, matching the /predict contract, so the UI shows real packet
    counts and byte rates rather than scaler outputs.
  - Each sample comes from a DIFFERENT source row. The previous implementation
    took .iloc[0] per spec, so both BENIGN specs resolved to the same flow and
    only 7 distinct samples were ever returned.
  - Descriptions are derived from each row's actual raw Label, so nothing is
    asserted about a flow that the data does not support.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Canonical class -> the Parquet file that carries it, so we never load all 8.
CLASS_SOURCE_FILE: dict[str, str] = {
    "BENIGN": "Benign-Monday-no-metadata.parquet",
    "DoS": "DoS-Wednesday-no-metadata.parquet",
    "DDoS": "DDoS-Friday-no-metadata.parquet",
    "PortScan": "Portscan-Friday-no-metadata.parquet",
    "BruteForce": "Bruteforce-Tuesday-no-metadata.parquet",
    "Botnet": "Botnet-Friday-no-metadata.parquet",
    "WebAttack": "WebAttacks-Thursday-no-metadata.parquet",
    "Infiltration": "Infiltration-Thursday-no-metadata.parquet",
}

# What each canonical class represents. Describes the CLASS, not a specific
# tool; the per-sample raw label supplies the specific attack name.
CLASS_DESCRIPTIONS: dict[str, str] = {
    "BENIGN": "Legitimate enterprise traffic (normal background activity)",
    "DoS": "Denial of Service - single-source resource exhaustion",
    "DDoS": "Distributed Denial of Service - volumetric multi-source flood",
    "PortScan": "Reconnaissance - port sweep enumerating open services",
    "BruteForce": "Credential attack - automated authentication guessing",
    "Botnet": "Botnet command-and-control communication",
    "WebAttack": "Web application layer exploit attempt",
    "Infiltration": "Internal infiltration following host compromise",
}

# Order the samples are presented in the UI.
DEMO_CLASS_ORDER: list[str] = [
    "BENIGN",
    "DoS",
    "DDoS",
    "PortScan",
    "BruteForce",
    "Botnet",
    "WebAttack",
    "Infiltration",
]


def _normalize_raw_label(value: Any) -> str:
    """Render a raw dataset label safely for display (U+FFFD -> ' - ')."""
    text = str(value).strip().replace("�", " - ")
    return " ".join(text.split())


def build_demo_samples(
    parquet_dir: str | Path,
    champion_features: list[str],
    model: Any,
    bundle: dict[str, Any],
    inverse_encoding: dict[int, str],
    label_map: dict[str, str],
    extra_benign: int = 1,
    scan_rows: int = 60000,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Select one genuine, distinct demo flow per canonical class.

    Preference order within a class: correctly classified with the highest
    confidence. If the Champion cannot classify any row of a class correctly
    (expected for Infiltration, which has almost no training support), the
    highest-confidence available row is used and flagged rather than skipped.

    Parameters
    ----------
    parquet_dir : str | Path
        Directory holding the source CIC-IDS2017 Parquet files.
    champion_features : list[str]
        Ordered Champion feature names.
    model : Any
        Trained Champion classifier.
    bundle : dict
        Inference bundle from src.data.inference_prep.
    inverse_encoding : dict[int, str]
        Integer class id -> canonical class name.
    label_map : dict[str, str]
        Raw dataset label -> canonical class (params.yaml:cleaning.label_map).
    extra_benign : int
        Additional distinct BENIGN flows beyond the first.
    scan_rows : int
        Rows sampled per source file when searching for candidates.
    random_seed : int
        Reproducibility seed.

    Returns
    -------
    list[dict]
        Demo sample records with raw feature values.
    """
    from src.data.clean import build_label_normalizer, normalize_raw_label
    from src.data.inference_prep import normalize_columns, prepare_for_champion

    parquet_dir = Path(parquet_dir)
    normalized_map = build_label_normalizer(label_map)
    rng = np.random.RandomState(random_seed)

    samples: list[dict[str, Any]] = []

    for cls in DEMO_CLASS_ORDER:
        source = parquet_dir / CLASS_SOURCE_FILE[cls]
        if not source.exists():
            logger.warning(f"Source Parquet missing for {cls}: {source}. Skipping.")
            continue

        df = pd.read_parquet(source)
        df = normalize_columns(df)

        if "Label" not in df.columns:
            logger.warning(f"No Label column in {source.name}; skipping {cls}.")
            continue

        canonical = df["Label"].map(lambda v: normalized_map.get(normalize_raw_label(str(v))))
        subset = df[canonical == cls]
        if subset.empty:
            logger.warning(f"No rows of class '{cls}' found in {source.name}.")
            continue

        if len(subset) > scan_rows:
            subset = subset.sample(n=scan_rows, random_state=random_seed)

        X = prepare_for_champion(subset, bundle, already_normalized=True)
        probs = model.predict_proba(X)
        preds = probs.argmax(axis=1)
        pred_names = np.array([inverse_encoding.get(int(p), f"Class_{p}") for p in preds])
        confidence = probs.max(axis=1)

        wanted = 1 + (extra_benign if cls == "BENIGN" else 0)
        correct_idx = np.where(pred_names == cls)[0]

        if len(correct_idx) >= wanted:
            chosen = correct_idx[np.argsort(-confidence[correct_idx])][:wanted]
            verified = True
        else:
            chosen = np.argsort(-confidence)[:wanted]
            verified = False
            logger.warning(
                f"Champion does not correctly classify any sampled '{cls}' flow; "
                f"using highest-confidence rows and flagging them."
            )

        for rank, pos in enumerate(chosen):
            row = subset.iloc[int(pos)]
            raw_label = _normalize_raw_label(row["Label"])
            suffix = f"_{rank + 1:02d}"
            samples.append({
                "id": f"{cls.lower()}{suffix}",
                "label": cls,
                "raw_label": raw_label,
                "description": f"{CLASS_DESCRIPTIONS[cls]} - dataset label: {raw_label}",
                "source_file": source.name,
                "champion_predicted": str(pred_names[int(pos)]),
                "champion_confidence": round(float(confidence[int(pos)]), 4),
                "correctly_classified": bool(verified and pred_names[int(pos)] == cls),
                "features": {f: float(row[f]) for f in champion_features},
            })

        del df, subset, X, probs

    logger.info(f"Built {len(samples)} distinct demo samples across {len(DEMO_CLASS_ORDER)} classes.")
    return samples


def save_demo_samples(samples: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Serialize the demo sample bank to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    logger.info(f"Saved {len(samples)} demo samples to: {path}")
    return path


def load_demo_samples(path: str | Path) -> list[dict[str, Any]]:
    """
    Load the demo sample bank, returning [] when it has not been generated.

    Absence is non-fatal: the API still serves predictions, the Live Analysis
    dropdown is simply empty.
    """
    p = Path(path)
    if not p.exists():
        logger.warning(f"Demo sample bank not found at {p}; /demo-samples will be empty.")
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
