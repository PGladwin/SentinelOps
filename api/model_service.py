"""
SentinelOps API - Model Inference & Explanation Service
=======================================================
Singleton model service that:
  - Loads whichever Champion governance promoted, its metadata, and the
    raw -> scaled inference bundle
  - Accepts RAW network-flow features and preprocesses them exactly as Phase 1
    did (alias resolution, median fill, RobustScaler) before inference
  - Computes multiclass probabilities
  - Generates true SHAP local explanations via TreeExplainer
  - Serves a pre-baked bank of genuine CIC-IDS2017 demo flows
  - Powers the SOC /analyze endpoint over uploaded CSV files

Input contract:
  All feature values are RAW, in the units of the source dataset (packet
  counts, byte rates, microsecond durations). Scaling happens server-side.
"""

import io
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap

from api.config import settings
from src.data.inference_prep import (
    LABEL_COLUMN_CANDIDATES,
    NON_FEATURE_COLUMNS,
    extract_labels,
    load_inference_bundle,
    normalize_columns,
    prepare_for_champion,
    prepare_with_report,
    resolve_truth_labels,
)
from src.models.demo_samples import load_demo_samples
from src.models.train import load_champion_model

logger = logging.getLogger("sentinelops.api")

# Confidence histogram buckets used by the SOC dashboard.
CONFIDENCE_BINS: list[tuple[float, float]] = [
    (0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001),
]


class ModelService:
    _instance: Optional["ModelService"] = None

    def __init__(
        self,
        models_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
    ):
        self.models_dir = Path(models_dir) if models_dir else settings.models_dir
        self.data_dir = Path(data_dir) if data_dir else settings.processed_dir
        self.model: Any = None
        self.feature_names: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.class_encoding: dict[str, int] = {}
        self.inverse_encoding: dict[int, str] = {}
        self.explainer: Optional[shap.TreeExplainer] = None
        self.bundle: dict[str, Any] = {}
        self.demo_samples: list[dict[str, Any]] = []

        self._load_artifacts()
        self._init_explainer()
        self._load_demo_samples()

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the cached singleton. Used by tests that reload artifacts."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _load_artifacts(self) -> None:
        """Load Champion model, ordered features, metadata, and the raw-input bundle."""
        logger.info(f"Loading Champion artifacts from: {self.models_dir.resolve()}")
        self.model, self.feature_names, self.metadata = load_champion_model(self.models_dir)

        if not self.feature_names:
            raise ValueError("Champion feature list is empty; artifacts are corrupt.")

        # Deliberately no hardcoded feature count: governance may promote a
        # model built on any feature set, and the serving layer must follow
        # whatever was approved rather than assume the Top-40 variant won.
        self.bundle = load_inference_bundle(self.models_dir / "champion_preprocessor.json")
        if self.bundle["feature_names"] != self.feature_names:
            raise ValueError(
                "Inference bundle feature order does not match the Champion feature list. "
                "Regenerate models/champion_preprocessor.json via run_phase2.py."
            )

        self.class_encoding = self.metadata.get("class_mapping", {})
        if not self.class_encoding:
            enc_path = self.data_dir / "class_encoding.json"
            if enc_path.exists():
                with open(enc_path, "r", encoding="utf-8") as f:
                    self.class_encoding = json.load(f)

        self.inverse_encoding = {int(v): k for k, v in self.class_encoding.items()}
        logger.info(
            f"Model service initialized: {self.metadata.get('model_name', 'xgboost_top40')} "
            f"({len(self.feature_names)} raw features, {len(self.class_encoding)} classes)"
        )

    def _init_explainer(self) -> None:
        """Initialize TreeExplainer for SHAP on the promoted Champion."""
        logger.info("Initializing SHAP TreeExplainer for real-time inference explanations...")
        self.explainer = shap.TreeExplainer(self.model)

    def _load_demo_samples(self) -> None:
        """
        Load the pre-baked demo bank.

        Deliberately does NOT read test.parquet: at full-dataset scale that is
        ~446k rows and would exhaust a small container at startup.
        """
        self.demo_samples = load_demo_samples(self.models_dir / "demo_samples.json")
        logger.info(f"Loaded {len(self.demo_samples)} genuine demo traffic samples.")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_health(self) -> dict[str, Any]:
        """Return service health and loaded model status."""
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "model": "XGBoost Champion",
            "feature_count": len(self.feature_names),
            "model_version": str(self.metadata.get("version", self.metadata.get("model_name", "xgboost_top40"))),
            "stage": str(self.metadata.get("stage", self.metadata.get("status", "provisional_champion"))),
            "input_contract": "raw",
        }

    def get_model_info(self) -> dict[str, Any]:
        """Return detailed metadata about the Champion and feature representation."""
        return {
            "model_name": self.metadata.get("model_name", "xgboost_top40"),
            "model_type": self.metadata.get("model_type", "XGBoost (Multiclass Tree Ensemble)"),
            "feature_count": len(self.feature_names),
            "features": self.feature_names,
            "class_mapping": self.class_encoding,
            "metadata": self.metadata,
        }

    def get_demo_samples(self) -> list[dict[str, Any]]:
        """Return the pre-loaded genuine demo samples."""
        return self.demo_samples

    # ------------------------------------------------------------------
    # Explanation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _increases_threat(shap_value: float, is_attack: bool) -> bool:
        """
        Decide whether a contribution raises threat, not merely whether it
        supports the predicted class.

        SHAP values are signed relative to the PREDICTED class. When the
        prediction is BENIGN, a positive value argues for benign traffic and
        therefore lowers threat. The previous implementation reported any
        positive value as 'increases risk', which inverted the meaning of
        every benign explanation.
        """
        return shap_value > 0 if is_attack else shap_value < 0

    def _explain_row(
        self,
        shap_row: np.ndarray,
        raw_values: dict[str, float] | np.ndarray,
        is_attack: bool,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Build the top-k ranked SHAP contribution records for one flow."""
        top_indices = np.argsort(np.abs(shap_row))[::-1][:top_k]

        explanation = []
        for idx in top_indices:
            idx = int(idx)
            feat_name = self.feature_names[idx]
            if isinstance(raw_values, dict):
                feat_val = float(raw_values[feat_name])
            else:
                feat_val = float(raw_values[idx])

            s_val = float(shap_row[idx])
            raises = self._increases_threat(s_val, is_attack)

            explanation.append({
                "feature": feat_name,
                "value": round(feat_val, 4),
                "shap_value": round(s_val, 6),
                "increases_threat": raises,
                "direction": "increases threat" if raises else "decreases threat",
            })
        return explanation

    def _shap_for_matrix(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute SHAP values for a scaled matrix.

        Returns
        -------
        values : np.ndarray
            Shape (n_rows, n_features, n_classes) for multiclass, or
            (n_rows, n_features) when the explainer collapses the class axis.
        base_values : np.ndarray
            Per-class expected values.
        """
        explanation = self.explainer(X)
        base = np.asarray(explanation.base_values)
        return np.asarray(explanation.values), base

    # ------------------------------------------------------------------
    # Single-flow prediction
    # ------------------------------------------------------------------

    def predict_single(self, features: dict[str, float]) -> dict[str, Any]:
        """
        Run prediction, probability estimation, and SHAP explanation on one RAW flow.

        Parameters
        ----------
        features : dict[str, float]
            Mapping of the 40 Champion feature names to RAW values.

        Returns
        -------
        dict[str, Any]
        """
        missing = set(self.feature_names) - set(features.keys())
        if missing:
            raise ValueError(
                f"Missing required feature(s): {sorted(missing)[:5]} ({len(missing)} missing)"
            )

        non_finite = [
            f for f in self.feature_names
            if not np.isfinite(np.asarray(features[f], dtype=np.float64))
        ]
        if non_finite:
            raise ValueError(
                f"Non-finite value(s) supplied for: {non_finite[:5]} "
                f"({len(non_finite)} total). Provide finite numeric values."
            )

        raw_df = pd.DataFrame([{f: features[f] for f in self.feature_names}])
        feature_vector = prepare_for_champion(raw_df, self.bundle, already_normalized=True)

        probabilities = self.model.predict_proba(feature_vector)[0]
        pred_int = int(np.argmax(probabilities))

        pred_class = self.inverse_encoding.get(pred_int, f"Class_{pred_int}")
        confidence = float(probabilities[pred_int])
        is_attack = pred_class != "BENIGN"

        prob_dict = {
            self.inverse_encoding.get(i, f"Class_{i}"): float(round(prob, 6))
            for i, prob in enumerate(probabilities)
        }

        shap_explanation: list[dict[str, Any]] = []
        base_value = 0.0
        if self.explainer is not None:
            raw_vals, base_vals = self._shap_for_matrix(feature_vector)
            if raw_vals.ndim == 3:
                shap_row = raw_vals[0, :, pred_int]
                base_value = float(np.ravel(base_vals)[pred_int]) if base_vals.size > 1 else float(np.ravel(base_vals)[0])
            else:
                shap_row = raw_vals[0, :]
                base_value = float(np.ravel(base_vals)[0])

            shap_explanation = self._explain_row(shap_row, features, is_attack, top_k=7)

        return {
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "is_attack": is_attack,
            "probabilities": prob_dict,
            "model": "XGBoost Champion (Top-40)",
            "feature_count": len(self.feature_names),
            "base_value": round(base_value, 6),
            "explanation": shap_explanation,
        }

    # ------------------------------------------------------------------
    # SOC analysis
    # ------------------------------------------------------------------

    def _read_upload(self, file_content: bytes, filename: str) -> pd.DataFrame:
        """
        Parse an uploaded traffic export into a DataFrame.

        Accepts the formats network tooling actually emits -- delimited text
        (CSV/TSV, with the delimiter sniffed rather than assumed), JSON records,
        newline-delimited JSON, and Parquet -- so a user is not forced to
        convert an export before they can scan it.

        Size and row guardrails are enforced here, before any parsing that
        would materialize the whole file.
        """
        if len(file_content) > settings.max_upload_bytes:
            raise ValueError(
                f"Uploaded file is {len(file_content) / 1024 / 1024:.1f} MB, exceeding the "
                f"{settings.max_upload_bytes / 1024 / 1024:.0f} MB limit."
            )
        if not file_content.strip():
            raise ValueError("Uploaded file is empty.")

        suffix = Path(filename or "").suffix.lower()
        buffer = io.BytesIO(file_content)

        try:
            if suffix == ".parquet":
                df = pd.read_parquet(buffer)
            elif suffix == ".jsonl" or suffix == ".ndjson":
                df = pd.read_json(buffer, lines=True)
            elif suffix == ".json":
                df = self._read_json(file_content)
            else:
                # The delimiter is sniffed from the header rather than passed as
                # sep=None: that option forces pandas onto the Python engine,
                # which is far slower on a large export and rejects low_memory.
                df = pd.read_csv(
                    buffer,
                    sep=self._sniff_delimiter(file_content),
                    low_memory=False,
                )
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Could not parse {suffix or 'the uploaded file'}: {e}")

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Uploaded file contains no data rows.")

        if len(df) > settings.max_analysis_rows:
            logger.warning(
                f"Upload has {len(df):,} rows; truncating to {settings.max_analysis_rows:,}."
            )
            df = df.head(settings.max_analysis_rows)
        return df

    @staticmethod
    def _sniff_delimiter(file_content: bytes) -> str:
        """
        Infer the delimiter from the header line.

        Picks whichever candidate appears most often, which is reliable here
        because a flow export's header carries dozens of feature names and
        therefore dozens of separators -- the true delimiter wins by a wide
        margin over a stray character inside a name. Falls back to a comma when
        nothing separates anything, so a single-column file still parses.
        """
        header = file_content.split(b"\n", 1)[0].decode("utf-8", errors="replace")
        counts = {d: header.count(d) for d in (",", "\t", ";", "|")}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","

    @staticmethod
    def _read_json(file_content: bytes) -> pd.DataFrame:
        """
        Parse JSON that may be a bare array, or records nested under a key.

        Exporters commonly wrap the rows -- {"flows": [...]} or {"data": [...]}
        -- and pd.read_json turns that into a one-row frame of lists rather than
        failing, which would then look like a file whose every column is missing.
        """
        payload = json.loads(file_content)

        if isinstance(payload, dict):
            for key in ("flows", "data", "records", "rows", "results"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
            else:
                # A dict of equal-length columns is also valid JSON tabular data.
                payload = [payload]

        return pd.DataFrame(payload)

    def _coverage_report(self, prep_report: dict[str, Any]) -> dict[str, Any]:
        """
        Describe how much of the Champion's schema the upload actually supplied.

        Missing features are filled from training medians, which lets an
        analysis complete on a partial export -- but a file that matched almost
        nothing would still produce confident-looking verdicts driven entirely
        by imputed constants. Quantifying the match lets the caller refuse, and
        lets the UI qualify what it shows.
        """
        total = len(self.feature_names)
        imputed = prep_report["n_imputed_columns"]
        matched = total - imputed
        return {
            "features_expected": total,
            "features_matched": matched,
            "features_imputed": imputed,
            "coverage": round(matched / total, 6) if total else 0.0,
            "imputed_columns": prep_report["imputed_columns"],
        }

    def _threat_level(self, attack_rate: float, thresholds: dict[str, float]) -> str:
        """Map an attack rate to a HIGH / MEDIUM / LOW banner."""
        if attack_rate >= thresholds.get("high", 0.20):
            return "HIGH"
        if attack_rate >= thresholds.get("medium", 0.05):
            return "MEDIUM"
        return "LOW"

    def analyze_csv(
        self,
        file_content: bytes,
        filename: str,
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Full SOC analysis of an uploaded RAW traffic CSV.

        Classifies every flow, computes batched SHAP attributions, and returns
        dashboard-ready aggregates alongside per-row detail.

        Parameters
        ----------
        file_content : bytes
            Raw CSV bytes.
        filename : str
            Original filename, echoed back in the summary.
        thresholds : dict, optional
            Attack-rate cutoffs for the threat-level banner.

        Returns
        -------
        dict[str, Any]
        """
        thresholds = thresholds or {"high": 0.20, "medium": 0.05}

        df = self._read_upload(file_content, filename)
        n_supplied_columns = len(df.columns)

        # Passing the Champion's feature names lets the resolver match case and
        # separator variants against the exact schema being served.
        df = normalize_columns(df, target_features=self.feature_names)
        truth = extract_labels(df)

        # Columns present in the upload that this Champion does not consume.
        #
        # Deliberately NOT called "unrecognized": most entries here are genuine
        # CIC-IDS2017 features the correlation filter dropped during training,
        # so a well-formed 70-column export legitimately leaves ~23 unused.
        # Flagging those as unknown would alarm a user whose file is perfect.
        # Identifier and label columns are excluded because they are expected.
        #
        # Computed AFTER normalization: "flow_duration" resolves to
        # "Flow Duration", so diffing the supplied spelling would report
        # correctly-matched columns as unused.
        unused = sorted(
            {str(c) for c in df.columns}
            - set(self.feature_names)
            - NON_FEATURE_COLUMNS
            - set(LABEL_COLUMN_CANDIDATES)
        )

        # allow_missing: real exports vary in schema. Absent columns are
        # imputed from training medians and reported back, so a degraded
        # analysis is visible rather than silently presented as clean.
        # prepare_with_report enforces the coverage floor itself and raises when
        # too much of the schema is missing: past that point most values the
        # model sees are training medians, so the verdicts would describe the
        # defaults rather than the upload. One threshold, enforced in one place.
        X, prep_report = prepare_with_report(
            df,
            self.bundle,
            already_normalized=True,
            allow_missing=True,
            max_missing_fraction=settings.max_missing_fraction,
        )
        n_rows = len(X)

        coverage = self._coverage_report(prep_report)

        probs_all = self.model.predict_proba(X)
        preds_int = probs_all.argmax(axis=1)
        confidences = probs_all.max(axis=1)
        pred_names = np.array(
            [self.inverse_encoding.get(int(p), f"Class_{p}") for p in preds_int]
        )
        is_attack_arr = pred_names != "BENIGN"

        # ---- SHAP over the batch (capped) ----------------------------
        shap_rows = min(n_rows, settings.max_shap_rows)
        global_shap: list[dict[str, Any]] = []
        per_row_shap: Optional[np.ndarray] = None

        if self.explainer is not None and shap_rows > 0:
            values, _ = self._shap_for_matrix(X[:shap_rows])
            if values.ndim == 3:
                mean_abs = np.mean(np.abs(values), axis=(0, 2))
                per_row_shap = values[np.arange(shap_rows), :, preds_int[:shap_rows]]
            else:
                mean_abs = np.mean(np.abs(values), axis=0)
                per_row_shap = values

            order = np.argsort(mean_abs)[::-1][:20]
            global_shap = [
                {
                    "rank": int(r + 1),
                    "feature": self.feature_names[int(i)],
                    "mean_abs_shap": float(round(mean_abs[int(i)], 6)),
                }
                for r, i in enumerate(order)
            ]

        # ---- Per-row records -----------------------------------------
        returned = min(n_rows, settings.max_returned_rows)
        # Raw (unscaled) values for display alongside SHAP contributions.
        # reindex rather than .loc so imputed columns do not raise; they are
        # backfilled with the same training medians used for inference.
        raw_display = df.reindex(columns=self.feature_names)
        raw_display = raw_display.apply(pd.to_numeric, errors="coerce")
        raw_display = raw_display.fillna(
            value={f: self.bundle["fill_values"][f] for f in self.feature_names}
        )
        raw_features = raw_display.to_numpy(dtype=np.float64, na_value=np.nan)

        rows = []
        for i in range(returned):
            explanation = []
            if per_row_shap is not None and i < shap_rows:
                explanation = self._explain_row(
                    per_row_shap[i], raw_features[i], bool(is_attack_arr[i]), top_k=3
                )
            rows.append({
                "flow_id": i + 1,
                "prediction": str(pred_names[i]),
                "confidence": round(float(confidences[i]), 4),
                "is_attack": bool(is_attack_arr[i]),
                "top_features": explanation,
            })

        # ---- Aggregates ----------------------------------------------
        class_breakdown = {self.inverse_encoding[i]: 0 for i in sorted(self.inverse_encoding)}
        names, counts = np.unique(pred_names, return_counts=True)
        for name, count in zip(names, counts):
            class_breakdown[str(name)] = int(count)

        total_attacks = int(is_attack_arr.sum())
        attack_rate = total_attacks / n_rows if n_rows else 0.0

        histogram = [
            {
                "bucket": f"{lo:.0%}-{min(hi, 1.0):.0%}",
                "count": int(((confidences >= lo) & (confidences < hi)).sum()),
            }
            for lo, hi in CONFIDENCE_BINS
        ]

        top_suspicious = sorted(
            [r for r in rows if r["is_attack"]],
            key=lambda r: -r["confidence"],
        )[:10]

        summary: dict[str, Any] = {
            "filename": filename,
            "threat_level": self._threat_level(attack_rate, thresholds),
            "total_connections": n_rows,
            "total_attacks": total_attacks,
            "total_benign": n_rows - total_attacks,
            "attack_rate": round(attack_rate, 6),
            "rows_returned": returned,
            "rows_explained": shap_rows,
            "truncated": returned < n_rows,
            "imputed_columns": prep_report["imputed_columns"],
            "n_imputed_columns": prep_report["n_imputed_columns"],
            # Schema match, so the UI can qualify a partial analysis rather than
            # presenting a degraded scan with the same confidence as a clean one.
            "features_expected": coverage["features_expected"],
            "features_matched": coverage["features_matched"],
            "schema_coverage": coverage["coverage"],
            "columns_supplied": n_supplied_columns,
            "unused_columns": unused[:20],
            "n_unused_columns": len(unused),
        }

        # Optional: if the upload carried ground truth, report real accuracy.
        if truth is not None:
            accuracy, scored = self._score_against_truth(truth, pred_names)
            summary["ground_truth_available"] = True
            summary["ground_truth_accuracy"] = accuracy
            # Rows whose label did not resolve are excluded from the score.
            # Publishing the denominator keeps a partial-coverage accuracy from
            # reading as a whole-file one.
            summary["ground_truth_scored"] = scored
        else:
            summary["ground_truth_available"] = False

        return {
            "summary": summary,
            "class_breakdown": class_breakdown,
            "confidence_histogram": histogram,
            "top_suspicious": top_suspicious,
            "global_shap": global_shap,
            "rows": rows,
        }

    def _score_against_truth(
        self, truth: pd.Series, predictions: np.ndarray
    ) -> tuple[Optional[float], int]:
        """
        Accuracy against an uploaded Label column, mapped to the canonical taxonomy.

        Returns (accuracy, rows_scored). Accuracy is None when no label could be
        mapped, rather than reporting a misleading score; rows_scored is the
        denominator so the caller can show what the figure actually covers.
        """
        try:
            import yaml

            params_path = settings.project_root / "params.yaml"
            if not params_path.exists():
                return None, 0
            with open(params_path, "r", encoding="utf-8") as f:
                label_map = yaml.safe_load(f)["cleaning"]["label_map"]

            mapped = resolve_truth_labels(truth, label_map)
            valid = mapped.notna().to_numpy()
            if not valid.any():
                return None, 0
            accuracy = float((mapped[valid].to_numpy() == predictions[valid]).mean())
            return round(accuracy, 6), int(valid.sum())
        except Exception as e:
            logger.warning(f"Could not score against uploaded ground truth: {e}")
            return None, 0
