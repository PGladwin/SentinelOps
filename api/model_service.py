"""
SentinelOps API - Model Inference & Explanation Service
=======================================================
Singleton model service that:
  - Loads the Phase 2 Top-40 XGBoost Champion model and its metadata
  - Verifies exact 40-feature ordering
  - Computes multiclass probabilities
  - Generates true SHAP local explanations via TreeExplainer
  - Serves genuine CIC-IDS2017 test dataset demo samples
"""

import io
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap

from src.models.train import load_champion_model

logger = logging.getLogger("sentinelops.api")


class ModelService:
    _instance: Optional["ModelService"] = None

    def __init__(self, models_dir: str = "models", data_dir: str = "data/processed"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.model: Any = None
        self.feature_names: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.class_encoding: dict[str, int] = {}
        self.inverse_encoding: dict[int, str] = {}
        self.explainer: Optional[shap.TreeExplainer] = None
        self.demo_samples: list[dict[str, Any]] = []

        self._load_artifacts()
        self._init_explainer()
        self._load_demo_samples()

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def _load_artifacts(self) -> None:
        """Load Champion model, ordered 40 features, metadata, and class mappings."""
        logger.info(f"Loading Champion artifacts from: {self.models_dir.resolve()}")
        self.model, self.feature_names, self.metadata = load_champion_model(self.models_dir)

        if len(self.feature_names) != 40:
            raise ValueError(
                f"Expected exactly 40 features for Top-40 Champion, but found {len(self.feature_names)}"
            )

        self.class_encoding = self.metadata.get("class_mapping", {})
        if not self.class_encoding:
            # Fallback to processed dir
            enc_path = self.data_dir / "class_encoding.json"
            if enc_path.exists():
                with open(enc_path, "r", encoding="utf-8") as f:
                    self.class_encoding = json.load(f)

        self.inverse_encoding = {int(v): k for k, v in self.class_encoding.items()}
        logger.info(f"Model service initialized: {self.metadata.get('model_name', 'xgboost_top40')}")

    def _init_explainer(self) -> None:
        """Initialize TreeExplainer for SHAP on the Top-40 XGBoost Champion."""
        logger.info("Initializing SHAP TreeExplainer for real-time inference explanations...")
        self.explainer = shap.TreeExplainer(self.model)

    def _load_demo_samples(self) -> None:
        """Extract genuine representative traffic flows from the test dataset."""
        test_path = self.data_dir / "test.parquet"
        if not test_path.exists():
            logger.warning(f"Test parquet not found at {test_path}; demo samples will be empty.")
            return

        test_df = pd.read_parquet(test_path)
        samples = []

        # Available classes in test set: BENIGN, DoS, DDoS, PortScan, BruteForce, Botnet, WebAttack
        demo_specs = [
            {
                "id": "benign_flow_01",
                "target_class": "BENIGN",
                "description": "Legitimate HTTPS browsing & DNS query traffic (Standard enterprise baseline)",
            },
            {
                "id": "dos_hulk_01",
                "target_class": "DoS",
                "description": "DoS HTTP Flood Attack (High-frequency GET requests with randomized headers)",
            },
            {
                "id": "ddos_loic_01",
                "target_class": "DDoS",
                "description": "Volumetric DDoS Attack (Distributed packet flood targeting network gateway)",
            },
            {
                "id": "portscan_syn_01",
                "target_class": "PortScan",
                "description": "Reconnaissance Port Scan (Sequential SYN probes sweeping open service ports)",
            },
            {
                "id": "bruteforce_ssh_01",
                "target_class": "BruteForce",
                "description": "SSH Authentication Brute-Force (Automated credential dictionary attack)",
            },
            {
                "id": "botnet_ares_01",
                "target_class": "Botnet",
                "description": "Botnet C2 Beaconing (Periodic command-and-control heartbeat communication)",
            },
            {
                "id": "webattack_xss_01",
                "target_class": "WebAttack",
                "description": "Web Application Exploit / XSS Injection (Cross-site scripting payload delivery)",
            },
            {
                "id": "benign_flow_02",
                "target_class": "BENIGN",
                "description": "Standard FTP File Transfer session (Benign internal file server communication)",
            },
        ]

        for spec in demo_specs:
            match = test_df[test_df["label_class"] == spec["target_class"]]
            if len(match) > 0:
                row = match.iloc[0]
                # Extract only the 40 required features
                feat_dict = {f: float(row[f]) for f in self.feature_names}
                samples.append({
                    "id": spec["id"],
                    "label": spec["target_class"],
                    "description": spec["description"],
                    "features": feat_dict,
                })

        self.demo_samples = samples
        logger.info(f"Loaded {len(self.demo_samples)} genuine demo traffic samples from CIC-IDS2017 test set.")

    def get_health(self) -> dict[str, Any]:
        """Return service health and loaded model status."""
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "model": "XGBoost Champion",
            "feature_count": len(self.feature_names),
        }

    def get_model_info(self) -> dict[str, Any]:
        """Return detailed metadata about the Champion model and feature representation."""
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

    def predict_single(self, features: dict[str, float]) -> dict[str, Any]:
        """
        Run prediction, probability estimation, and SHAP explanation for a single flow.

        Parameters
        ----------
        features : dict[str, float]
            Mapping of 40 feature names to values.

        Returns
        -------
        result : dict[str, Any]
        """
        # Validate that all 40 required features are present
        missing = set(self.feature_names) - set(features.keys())
        if missing:
            raise ValueError(f"Missing required feature(s): {list(missing)[:5]} ({len(missing)} missing)")

        # Construct 1D vector in exact feature order
        feature_vector = np.array(
            [[features[f] for f in self.feature_names]],
            dtype=np.float32,
        )

        # Run model inference
        pred_int = int(self.model.predict(feature_vector)[0])
        probabilities = self.model.predict_proba(feature_vector)[0]

        pred_class = self.inverse_encoding.get(pred_int, f"Class_{pred_int}")
        confidence = float(probabilities[pred_int])
        is_attack = pred_class != "BENIGN"

        prob_dict = {
            self.inverse_encoding.get(i, f"Class_{i}"): float(round(prob, 6))
            for i, prob in enumerate(probabilities)
        }

        # Calculate SHAP explanation for this sample
        shap_explanation = []
        if self.explainer is not None:
            shap_vals = self.explainer(feature_vector)
            raw_vals = shap_vals.values  # (1, 40, 8) or (1, 40)

            if raw_vals.ndim == 3:
                sample_shap = raw_vals[0, :, pred_int]
            else:
                sample_shap = raw_vals[0, :]

            # Select top 7 influential features by magnitude
            top_indices = np.argsort(np.abs(sample_shap))[::-1][:7]

            for idx in top_indices:
                feat_name = self.feature_names[idx]
                feat_val = float(features[feat_name])
                s_val = float(sample_shap[idx])
                direction = (
                    "increases risk/probability"
                    if s_val > 0
                    else "decreases risk/probability"
                )
                shap_explanation.append({
                    "feature": feat_name,
                    "value": round(feat_val, 4),
                    "shap_value": round(s_val, 6),
                    "direction": direction,
                })

        return {
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "is_attack": is_attack,
            "probabilities": prob_dict,
            "model": "XGBoost Champion (Top-40)",
            "feature_count": len(self.feature_names),
            "explanation": shap_explanation,
        }

    def predict_batch(self, file_content: bytes, filename: str) -> dict[str, Any]:
        """
        Run batch prediction on uploaded CSV file.

        Parameters
        ----------
        file_content : bytes
            Raw CSV bytes.
        filename : str
            Name of uploaded file.

        Returns
        -------
        batch_results : dict[str, Any]
        """
        try:
            df = pd.read_csv(io.BytesIO(file_content))
        except Exception as e:
            raise ValueError(f"Could not parse CSV: {e}")

        # Check required columns
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing {len(missing)} required feature columns. "
                f"Missing examples: {list(missing)[:5]}"
            )

        # Align columns
        X_batch = df[self.feature_names].values.astype(np.float32)

        preds_int = self.model.predict(X_batch)
        probs_all = self.model.predict_proba(X_batch)

        prediction_counts: dict[str, int] = {
            self.inverse_encoding[i]: 0 for i in range(len(self.inverse_encoding))
        }

        row_results = []
        for idx, (p_int, probs) in enumerate(zip(preds_int, probs_all)):
            p_class = self.inverse_encoding.get(int(p_int), f"Class_{p_int}")
            conf = float(probs[p_int])
            prediction_counts[p_class] = prediction_counts.get(p_class, 0) + 1

            row_results.append({
                "flow_id": idx + 1,
                "prediction": p_class,
                "confidence": round(conf, 4),
                "is_attack": p_class != "BENIGN",
            })

        return {
            "total_samples": len(df),
            "prediction_counts": prediction_counts,
            "predictions": row_results,
        }
