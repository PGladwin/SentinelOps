"""
SentinelOps - Data Drift Monitoring
====================================
Maintains the reference distribution the production model was trained against,
and compares incoming traffic batches to it.

Why it matters here:
  Attack tooling and normal traffic both change. A model trained on 2017 flows
  will silently degrade as the traffic it sees stops resembling its training
  data. Drift detection is what converts that silent failure into a signal that
  can trigger retraining -- which then has to clear the governance gate before
  reaching production.

Configuration comes from params.yaml:drift (stattest, drift_share_threshold),
which existed as dead config before this module.

Reference construction lives here and needs no third-party dependency;
the Evidently comparison is imported lazily so the baseline can always be
rebuilt even if Evidently is unavailable.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REFERENCE_FILENAME = "reference.parquet"
REFERENCE_META_FILENAME = "reference_metadata.json"
DRIFT_SUMMARY_FILENAME = "drift_summary.json"
DRIFT_REPORT_FILENAME = "drift_report.html"


# ---------------------------------------------------------------------------
# Reference snapshot
# ---------------------------------------------------------------------------

def build_reference(
    train_path: str | Path,
    feature_names: list[str],
    output_dir: str | Path,
    max_rows: int = 50000,
    random_seed: int = 42,
) -> Path:
    """
    Snapshot the training distribution as the drift baseline.

    A stratified subsample rather than the whole training split: the baseline
    is compared against every incoming batch, so it needs to be small enough to
    load cheaply while still representing each class. Stratifying matters here
    because the rare attack classes are exactly the ones whose disappearance
    from production traffic would signal drift.

    Parameters
    ----------
    train_path : str | Path
        data/processed/train.parquet (scaled features plus label columns).
    feature_names : list[str]
        Feature columns to retain.
    output_dir : str | Path
        Destination directory (params.yaml:paths.reference_dir).
    max_rows : int
        Upper bound on the snapshot size.
    random_seed : int
        Reproducibility seed.

    Returns
    -------
    Path
        Path to the written reference Parquet.
    """
    train_path = Path(train_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(train_path)
    logger.info(f"Loaded training split for reference: {len(df):,} rows")

    keep = [c for c in feature_names if c in df.columns]
    missing = set(feature_names) - set(keep)
    if missing:
        logger.warning(f"Reference is missing {len(missing)} feature column(s): {sorted(missing)[:5]}")

    label_col = "label_class" if "label_class" in df.columns else None

    if len(df) > max_rows and label_col:
        fraction = max_rows / len(df)
        parts = []
        for cls, group in df.groupby(label_col, observed=True):
            n = max(1, min(len(group), int(round(len(group) * fraction))))
            parts.append(group.sample(n=n, random_state=random_seed))
        sample = pd.concat(parts).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    elif len(df) > max_rows:
        sample = df.sample(n=max_rows, random_state=random_seed).reset_index(drop=True)
    else:
        sample = df.reset_index(drop=True)

    columns = keep + ([label_col] if label_col else [])
    sample = sample.loc[:, columns]

    ref_path = out_dir / REFERENCE_FILENAME
    sample.to_parquet(ref_path, index=False)

    distribution = (
        sample[label_col].value_counts().to_dict() if label_col else {}
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(train_path),
        "source_rows": int(len(df)),
        "reference_rows": int(len(sample)),
        "feature_count": len(keep),
        "features": keep,
        "class_distribution": {str(k): int(v) for k, v in distribution.items()},
        "random_seed": random_seed,
    }
    with open(out_dir / REFERENCE_META_FILENAME, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Wrote drift reference: {ref_path} "
        f"({len(sample):,} rows x {len(keep)} features)"
    )
    return ref_path


def load_reference(reference_dir: str | Path) -> Optional[pd.DataFrame]:
    """Load the reference snapshot, or None when it has not been built."""
    path = Path(reference_dir) / REFERENCE_FILENAME
    if not path.exists():
        logger.warning(f"No drift reference at {path}; run the drift_reference stage.")
        return None
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Statistical drift detection
# ---------------------------------------------------------------------------

# Above this many rows a KS test becomes uninformative: with tens of thousands
# of samples it returns p ~ 0 for differences far too small to matter
# operationally, so every feature reads as "drifted". Evidently switches
# strategy on the same principle. Above the threshold we use a normalized
# Wasserstein distance, which measures the SIZE of the shift rather than our
# confidence that a shift exists.
LARGE_SAMPLE_THRESHOLD = 1000

# Wasserstein distance in units of the reference standard deviation.
DEFAULT_WASSERSTEIN_THRESHOLD = 0.10

# p-value cutoff for the KS test on small samples.
DEFAULT_PVALUE_THRESHOLD = 0.05


def _feature_drift(
    reference: np.ndarray,
    current: np.ndarray,
    stattest: str,
    pvalue_threshold: float,
    distance_threshold: float,
) -> dict[str, Any]:
    """
    Test a single feature for distribution shift.

    Parameters
    ----------
    reference, current : np.ndarray
        1-D finite samples.
    stattest : str
        'ks', 'wasserstein', or 'auto' (size-dependent, recommended).

    Returns
    -------
    dict
        Chosen test, score, threshold, and the drift verdict.
    """
    from scipy.stats import ks_2samp, wasserstein_distance

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) < 2 or len(current) < 2:
        return {
            "test": "none", "score": None, "threshold": None,
            "drifted": False, "note": "insufficient data",
        }

    chosen = stattest
    if stattest == "auto":
        n = min(len(reference), len(current))
        chosen = "ks" if n <= LARGE_SAMPLE_THRESHOLD else "wasserstein"

    if chosen == "ks":
        result = ks_2samp(reference, current)
        return {
            "test": "ks",
            "score": float(result.pvalue),
            "statistic": float(result.statistic),
            "threshold": float(pvalue_threshold),
            "drifted": bool(result.pvalue < pvalue_threshold),
        }

    # Normalize by the reference spread so the threshold is scale-free and
    # comparable across features measured in different units.
    spread = float(np.std(reference))
    raw = float(wasserstein_distance(reference, current))
    normalized = raw / spread if spread > 0 else (0.0 if raw == 0 else float("inf"))

    return {
        "test": "wasserstein",
        "score": float(normalized),
        "raw_distance": raw,
        "threshold": float(distance_threshold),
        "drifted": bool(normalized > distance_threshold),
    }


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Optional[list[str]] = None,
    stattest: str = "auto",
    drift_share_threshold: float = 0.20,
    pvalue_threshold: float = DEFAULT_PVALUE_THRESHOLD,
    distance_threshold: float = DEFAULT_WASSERSTEIN_THRESHOLD,
) -> dict[str, Any]:
    """
    Compare a current batch against the reference distribution, feature by feature.

    Both frames must be in the SAME representation. The reference is stored
    scaled (it is a snapshot of train.parquet), so a production batch must go
    through prepare_for_champion first. Comparison runs on the intersection of
    columns present in both, since the Champion may consume a subset of the
    features the reference carries.

    A dataset is considered drifted when the share of drifted features exceeds
    drift_share_threshold -- one shifted column is noise, a third of them is a
    changed world.

    Returns
    -------
    dict
        Per-feature results plus the overall verdict, ready to serialize.
    """
    if features is None:
        features = [c for c in reference.columns if c in current.columns]
    else:
        features = [f for f in features if f in reference.columns and f in current.columns]

    if not features:
        raise ValueError(
            "No overlapping feature columns between reference and current batch."
        )

    results: dict[str, Any] = {}
    drifted: list[str] = []

    for feature in features:
        ref_values = pd.to_numeric(reference[feature], errors="coerce").to_numpy(dtype=np.float64)
        cur_values = pd.to_numeric(current[feature], errors="coerce").to_numpy(dtype=np.float64)

        outcome = _feature_drift(
            ref_values, cur_values, stattest, pvalue_threshold, distance_threshold
        )
        results[feature] = outcome
        if outcome["drifted"]:
            drifted.append(feature)

    share = len(drifted) / len(features)
    dataset_drift = share > drift_share_threshold

    # Rank by how far each feature moved, so the report leads with the
    # features actually responsible rather than an arbitrary column order.
    def _magnitude(name: str) -> float:
        entry = results[name]
        if entry["test"] == "wasserstein":
            return entry["score"]
        if entry["test"] == "ks":
            return entry.get("statistic", 0.0)
        return 0.0

    top_drifted = sorted(drifted, key=_magnitude, reverse=True)[:15]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "drift_detected": bool(dataset_drift),
        "n_features": len(features),
        "n_drifted_features": len(drifted),
        "drift_share": round(share, 6),
        "drift_share_threshold": drift_share_threshold,
        "stattest": stattest,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "drifted_features": drifted,
        "top_drifted_features": [
            {
                "feature": name,
                "test": results[name]["test"],
                "score": round(results[name]["score"], 6) if results[name]["score"] is not None else None,
                "threshold": results[name]["threshold"],
            }
            for name in top_drifted
        ],
        "per_feature": results,
    }

    logger.info(
        f"Drift check: {len(drifted)}/{len(features)} features drifted "
        f"(share {share:.1%}, threshold {drift_share_threshold:.0%}) -> "
        f"{'DRIFT DETECTED' if dataset_drift else 'no dataset drift'}"
    )
    return summary


def save_drift_summary(summary: dict[str, Any], reports_dir: str | Path) -> Path:
    """Write drift_summary.json, which the API serves at GET /mlops/drift."""
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / DRIFT_SUMMARY_FILENAME

    # per_feature can run to hundreds of entries; keep it out of the served
    # document and leave the ranked subset the panel actually renders.
    served = {k: v for k, v in summary.items() if k != "per_feature"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(served, f, indent=2)

    logger.info(f"Wrote drift summary: {path}")
    return path


# ---------------------------------------------------------------------------
# Evidently HTML report
# ---------------------------------------------------------------------------

def render_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Optional[list[str]] = None,
    reports_dir: str | Path = "reports",
    max_rows: int = 20000,
) -> Optional[Path]:
    """
    Render the interactive Evidently drift report.

    Presentation only. The verdict the system acts on comes from detect_drift()
    above, which is deterministic, unit-tested, and has no third-party
    dependency -- so an Evidently upgrade can change this visual without
    silently changing whether the pipeline thinks drift occurred.

    Evidently reorganised its public API between 0.4 and 0.7 (evidently.report
    .Report + metric_preset became evidently.Report + presets), so both import
    paths are attempted.

    Returns
    -------
    Path | None
        Path to the written HTML, or None if Evidently is not installed.
    """
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / DRIFT_REPORT_FILENAME

    if features:
        columns = [c for c in features if c in reference.columns and c in current.columns]
        reference = reference.loc[:, columns]
        current = current.loc[:, columns]

    # Evidently renders every row into the HTML payload; cap it so the report
    # stays openable in a browser during a live demo.
    if len(reference) > max_rows:
        reference = reference.sample(n=max_rows, random_state=42)
    if len(current) > max_rows:
        current = current.sample(n=max_rows, random_state=42)

    try:  # Evidently >= 0.7
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset()])
        snapshot = report.run(current_data=current, reference_data=reference)
        snapshot.save_html(str(out_path))
        logger.info(f"Evidently report written: {out_path}")
        return out_path
    except ImportError:
        pass

    try:  # Evidently 0.4.x
        from evidently.metric_preset import DataDriftPreset as LegacyPreset
        from evidently.report import Report as LegacyReport

        report = LegacyReport(metrics=[LegacyPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(out_path))
        logger.info(f"Evidently report written (legacy API): {out_path}")
        return out_path
    except ImportError:
        logger.warning("Evidently is not installed; skipping the HTML report.")
        return None
