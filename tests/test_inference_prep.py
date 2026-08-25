"""
SentinelOps - Inference Preprocessing Tests
============================================
Guards the raw -> Champion serving path.

Regression context: the API previously required callers to submit
already-RobustScaler-transformed values while advertising plain feature names,
and the saved scaler covered 48 features while the Champion consumed 40. Any
genuine raw CSV therefore produced confidently wrong predictions. These tests
lock that path down.

Covers:
  - Column normalization: whitespace, aliases, duplicate columns
  - Alias coverage for the canonical CIC-IDS2017 CSV release
  - Exactness of the 48 -> 40 RobustScaler slice
  - Missing-feature and non-finite handling
  - Champion feature list is a subset of the Phase 1 feature list
  - Demo sample bank is distinct and covers the taxonomy
  - SHAP threat-direction semantics (the BENIGN inversion)
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.inference_prep import (
    COLUMN_ALIASES,
    build_inference_bundle,
    load_inference_bundle,
    normalize_columns,
    prepare_for_champion,
)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
CSV_DIR = ROOT / "datasets" / "CIC-IDS2017-csv"

BUNDLE_PATH = MODELS_DIR / "champion_preprocessor.json"
FEATURES_40_PATH = MODELS_DIR / "xgboost_top40_features.json"
FEATURES_48_PATH = PROCESSED_DIR / "feature_names.json"
SCALER_PATH = PROCESSED_DIR / "scaler.pkl"
FILL_PATH = PROCESSED_DIR / "fill_values.json"
DEMO_PATH = MODELS_DIR / "demo_samples.json"

requires_artifacts = pytest.mark.skipif(
    not (BUNDLE_PATH.exists() and FEATURES_48_PATH.exists() and SCALER_PATH.exists()),
    reason="Phase 1/2 artifacts not present; run run_phase1.py and run_phase2.py first.",
)


@pytest.fixture(scope="module")
def bundle():
    if not BUNDLE_PATH.exists():
        pytest.skip("Inference bundle not generated.")
    return load_inference_bundle(BUNDLE_PATH)


# ===========================================================================
# Column normalization
# ===========================================================================

class TestNormalizeColumns:
    def test_strips_whitespace(self):
        df = pd.DataFrame({" Flow Duration": [1], "Total Fwd Packets ": [2]})
        out = normalize_columns(df)
        assert list(out.columns) == ["Flow Duration", "Total Fwd Packets"]

    def test_resolves_known_aliases(self):
        df = pd.DataFrame({
            " Total Length of Fwd Packets": [1],
            " Min Packet Length": [2],
            "Init_Win_bytes_forward": [3],
            " act_data_pkt_fwd": [4],
            " min_seg_size_forward": [5],
        })
        out = normalize_columns(df)
        assert list(out.columns) == [
            "Fwd Packets Length Total",
            "Packet Length Min",
            "Init Fwd Win Bytes",
            "Fwd Act Data Packets",
            "Fwd Seg Size Min",
        ]

    def test_drops_duplicate_columns_keeping_first(self):
        """Canonical CIC-IDS2017 CSVs carry 'Fwd Header Length' twice."""
        df = pd.DataFrame([[1, 2, 3]], columns=["Fwd Header Length", "Flow Duration", " Fwd Header Length"])
        out = normalize_columns(df)
        assert list(out.columns) == ["Fwd Header Length", "Flow Duration"]
        assert out["Fwd Header Length"].iloc[0] == 1

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({" Flow Duration": [1]})
        original = list(df.columns)
        normalize_columns(df)
        assert list(df.columns) == original


# ===========================================================================
# Alias coverage against the real CSV release
# ===========================================================================

@pytest.mark.skipif(not CSV_DIR.exists(), reason="Canonical CSV release not available.")
@requires_artifacts
def test_alias_map_covers_canonical_csv_headers():
    """Every Champion feature must be reachable from the canonical CSV headers."""
    csv_file = next(CSV_DIR.glob("*.csv"), None)
    if csv_file is None:
        pytest.skip("No CSV files found.")

    header = pd.read_csv(csv_file, nrows=0)
    normalized = set(normalize_columns(header).columns)

    champion_path = MODELS_DIR / "champion_features.json"
    champion_features = json.loads(
        (champion_path if champion_path.exists() else FEATURES_40_PATH).read_text(encoding="utf-8")
    )
    # 'Protocol' has no counterpart in the canonical CSV release (which ships
    # 'Destination Port' instead), so it is excluded from alias coverage.
    missing = [f for f in champion_features if f not in normalized and f != "Protocol"]

    assert not missing, (
        f"{len(missing)} Champion feature(s) unreachable from canonical CSV headers "
        f"after alias resolution: {missing}"
    )


# ===========================================================================
# Scaler slicing exactness
# ===========================================================================

@requires_artifacts
def test_sliced_scaler_matches_full_scaler():
    """
    The 40-feature slice must equal scaling all 48 columns and then selecting
    the Champion's 40. This is the property that makes the bundle valid without
    retraining, so it is asserted directly rather than assumed.
    """
    features_40 = json.loads(FEATURES_40_PATH.read_text(encoding="utf-8"))
    features_48 = json.loads(FEATURES_48_PATH.read_text(encoding="utf-8"))
    fill = json.loads(FILL_PATH.read_text(encoding="utf-8"))
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    rng = np.random.default_rng(0)
    raw = pd.DataFrame(
        rng.normal(500, 2500, size=(256, len(features_48))), columns=features_48
    )

    reference = scaler.transform(raw[features_48].values)
    reference_40 = reference[:, [features_48.index(f) for f in features_40]]

    bundle = build_inference_bundle(features_40, SCALER_PATH, FILL_PATH, FEATURES_48_PATH)
    produced = prepare_for_champion(raw, bundle, already_normalized=True)

    np.testing.assert_allclose(reference_40, produced, rtol=1e-4, atol=1e-4)


@requires_artifacts
def test_champion_features_are_subset_of_phase1_features():
    features_48 = json.loads(FEATURES_48_PATH.read_text(encoding="utf-8"))

    champion_path = MODELS_DIR / "champion_features.json"
    champion = json.loads(
        (champion_path if champion_path.exists() else FEATURES_40_PATH).read_text(encoding="utf-8")
    )

    # Whatever governance promoted must be expressible in Phase 1 features,
    # otherwise the inference bundle cannot scale it.
    assert champion, "Champion feature list is empty"
    assert set(champion).issubset(set(features_48))


@requires_artifacts
def test_bundle_is_internally_consistent(bundle):
    n = len(bundle["feature_names"])
    assert n > 0
    assert len(bundle["center"]) == n
    assert len(bundle["scale"]) == n
    assert set(bundle["fill_values"]) == set(bundle["feature_names"])


# ===========================================================================
# prepare_for_champion behaviour
# ===========================================================================

@requires_artifacts
class TestPrepareForChampion:
    def test_missing_feature_raises(self, bundle):
        df = pd.DataFrame([{f: 1.0 for f in bundle["feature_names"][:-1]}])
        with pytest.raises(ValueError, match="missing"):
            prepare_for_champion(df, bundle)

    def test_extra_columns_ignored(self, bundle):
        row = {f: 1.0 for f in bundle["feature_names"]}
        row["Destination Port"] = 443
        row["Timestamp"] = "2017-07-07 09:00:00"
        out = prepare_for_champion(pd.DataFrame([row]), bundle)
        assert out.shape == (1, len(bundle["feature_names"]))

    def test_inf_and_nan_are_filled(self, bundle):
        row = {f: 1.0 for f in bundle["feature_names"]}
        row[bundle["feature_names"][0]] = np.inf
        row[bundle["feature_names"][1]] = np.nan
        out = prepare_for_champion(pd.DataFrame([row]), bundle)
        assert np.isfinite(out).all()

    def test_output_is_float32_and_ordered(self, bundle):
        df = pd.DataFrame([{f: float(i) for i, f in enumerate(bundle["feature_names"])}])
        shuffled = df[list(reversed(bundle["feature_names"]))]
        out_a = prepare_for_champion(df, bundle)
        out_b = prepare_for_champion(shuffled, bundle)
        assert out_a.dtype == np.float32
        np.testing.assert_array_equal(out_a, out_b), "Column order must not affect output"

    def test_string_numerics_are_coerced(self, bundle):
        row = {f: "1.5" for f in bundle["feature_names"]}
        out = prepare_for_champion(pd.DataFrame([row]), bundle)
        assert np.isfinite(out).all()


# ===========================================================================
# Demo sample bank
# ===========================================================================

@pytest.mark.skipif(not DEMO_PATH.exists(), reason="Demo sample bank not generated.")
def test_demo_samples_are_distinct_and_cover_taxonomy():
    """
    Regression: the previous implementation took .iloc[0] per spec, so both
    BENIGN entries resolved to the same flow and only 7 distinct samples were
    ever served despite 8 being declared.
    """
    samples = json.loads(DEMO_PATH.read_text(encoding="utf-8"))

    assert len(samples) >= 8
    assert len({s["id"] for s in samples}) == len(samples), "Duplicate sample ids"

    fingerprints = {tuple(sorted(s["features"].items())) for s in samples}
    assert len(fingerprints) == len(samples), "Duplicate feature vectors across demo samples"

    expected = {"BENIGN", "DoS", "DDoS", "PortScan", "BruteForce", "Botnet", "WebAttack", "Infiltration"}
    assert expected.issubset({s["label"] for s in samples})

    champion_features = json.loads(
        (MODELS_DIR / "champion_features.json").read_text(encoding="utf-8")
    ) if (MODELS_DIR / "champion_features.json").exists() else json.loads(
        FEATURES_40_PATH.read_text(encoding="utf-8")
    )
    for s in samples:
        assert set(s["features"]) == set(champion_features)


# ===========================================================================
# SHAP threat-direction semantics
# ===========================================================================

class TestThreatDirection:
    """
    Regression: direction was previously read straight off the SHAP sign, so a
    feature supporting a BENIGN prediction was labelled 'increases risk'. Every
    benign explanation in the UI was therefore inverted.
    """

    @staticmethod
    def _increases_threat(shap_value, is_attack):
        from api.model_service import ModelService
        return ModelService._increases_threat(shap_value, is_attack)

    def test_positive_shap_on_attack_increases_threat(self):
        assert self._increases_threat(0.8, is_attack=True) is True

    def test_negative_shap_on_attack_decreases_threat(self):
        assert self._increases_threat(-0.8, is_attack=True) is False

    def test_positive_shap_on_benign_decreases_threat(self):
        assert self._increases_threat(0.8, is_attack=False) is False

    def test_negative_shap_on_benign_increases_threat(self):
        assert self._increases_threat(-0.8, is_attack=False) is True
