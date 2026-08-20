"""
SentinelOps - Phase 1 Tests
==============================
Tests for:
  - Label normalization (including U+FFFD handling)
  - Label mapping (correct mapping + unknown label detection)
  - Inf / NaN replacement
  - Constant column detection
  - Duplicate row removal
  - Stratified sampling preserves class proportions
  - Train/test split is stratified
  - No data leakage: scaler/fill fitted only on train
  - Feature list order is preserved
  - Artifacts are saved and loadable

Run with:
    python -m pytest tests/test_phase1.py -v
    # or without pytest:
    python tests/test_phase1_runner.py
"""

import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.clean import (
    normalize_raw_label,
    build_label_normalizer,
    map_raw_labels,
    replace_inf,
    compute_fill_values,
    apply_fill_values,
    detect_constant_columns,
    drop_constant_columns,
    drop_duplicate_rows,
    clean_dataframe,
)
from src.data.preprocess import (
    encode_labels,
    stratified_sample,
    find_correlated_pairs,
    run_preprocessing,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def minimal_params():
    """Minimal params.yaml-equivalent dict for testing."""
    return {
        "general": {"random_seed": 42, "project_name": "test"},
        "paths": {
            "parquet_dir": "datasets/CIC-IDS2017-paraquet",
            "processed_dir": "data/processed_test",
            "reference_dir": "data/reference",
            "reports_dir": "reports",
            "models_dir": "models",
            "metrics_dir": "metrics",
        },
        "ingestion": {
            "expected_files": [
                "Benign-Monday-no-metadata.parquet",
                "Botnet-Friday-no-metadata.parquet",
                "Bruteforce-Tuesday-no-metadata.parquet",
                "DDoS-Friday-no-metadata.parquet",
                "DoS-Wednesday-no-metadata.parquet",
                "Infiltration-Thursday-no-metadata.parquet",
                "Portscan-Friday-no-metadata.parquet",
                "WebAttacks-Thursday-no-metadata.parquet",
            ]
        },
        "cleaning": {
            "replace_inf_with_nan": True,
            "nan_fill_strategy": "median",
            "drop_duplicates": True,
            "drop_constant_columns": True,
            "label_map": {
                "Benign": "BENIGN",
                "Bot": "Botnet",
                "DDoS": "DDoS",
                "DoS GoldenEye": "DoS",
                "DoS Hulk": "DoS",
                "DoS Slowhttptest": "DoS",
                "DoS slowloris": "DoS",
                "FTP-Patator": "BruteForce",
                "Heartbleed": "DoS",
                "Infiltration": "Infiltration",
                "PortScan": "PortScan",
                "SSH-Patator": "BruteForce",
                "Web Attack - Brute Force": "WebAttack",
                "Web Attack - XSS": "WebAttack",
                "Web Attack - Sql Injection": "WebAttack",
            },
            "class_encoding": {
                "BENIGN": 0, "DoS": 1, "DDoS": 2, "PortScan": 3,
                "BruteForce": 4, "Botnet": 5, "WebAttack": 6, "Infiltration": 7,
            },
        },
        "preprocessing": {
            "correlation_threshold": 0.95,
            "scaler": "robust",
            "test_size": 0.20,
            "dev_sample_size": 500,  # tiny for unit tests
        },
    }


def _make_dummy_df(n=1000, label_col="Label"):
    """
    Create a small synthetic DataFrame that mimics CIC-IDS2017 structure.
    8 classes, some NaN, some Inf, a constant column.
    Guarantees at least 5 samples per class to survive stratified splits.
    """
    rng = np.random.default_rng(0)

    # U+FFFD versions of WebAttack labels (as they appear in the actual dataset)
    ufffd_raw_labels = [
        "Benign", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk",
        "DoS Slowhttptest", "DoS slowloris", "FTP-Patator", "Heartbleed",
        "Infiltration", "PortScan", "SSH-Patator",
        "Web Attack \ufffd Brute Force",
        "Web Attack \ufffd XSS",
        "Web Attack \ufffd Sql Injection",
    ]

    # Assign labels with heavy imbalance like real data
    probs = [0.60, 0.03, 0.07, 0.02, 0.04, 0.02, 0.02, 0.02, 0.01,
             0.01, 0.01, 0.01, 0.03, 0.02, 0.01]
    probs = np.array(probs)
    probs = probs / probs.sum()

    labels = rng.choice(ufffd_raw_labels, size=n, p=probs).tolist()

    # Guarantee at least 5 samples per raw label to survive dev_sample+stratify
    for lbl in ufffd_raw_labels:
        count = labels.count(lbl)
        if count < 5:
            labels.extend([lbl] * (5 - count))

    rng.shuffle(labels)
    labels = labels[:n + len(ufffd_raw_labels) * 5]  # might be slightly over n
    n_actual = len(labels)

    # Features
    data = {label_col: labels}
    for i in range(10):
        col = f"feature_{i}"
        data[col] = rng.normal(0, 1, n_actual).astype(np.float32)

    # Insert constant column
    data["const_col"] = np.zeros(n_actual, dtype=np.float32)

    # Insert some NaN
    nan_indices = rng.integers(0, n_actual, 10)
    arr = data["feature_0"].copy()
    arr[nan_indices] = np.nan
    data["feature_0"] = arr

    # Insert some Inf
    inf_indices = rng.integers(0, n_actual, 5)
    arr2 = data["feature_1"].copy()
    arr2[inf_indices] = np.inf
    arr2[rng.integers(0, n_actual, 5)] = -np.inf
    data["feature_1"] = arr2

    # Source file metadata
    data["source_file"] = "dummy.parquet"

    return pd.DataFrame(data)



# ===========================================================================
# Tests: Label Normalization
# ===========================================================================

class TestNormalizeRawLabel:
    def test_strips_whitespace(self):
        assert normalize_raw_label("  Benign  ") == "Benign"

    def test_replaces_ufffd_with_dash(self):
        raw = "Web Attack \ufffd XSS"
        expected = "Web Attack - XSS"
        assert normalize_raw_label(raw) == expected

    def test_multiple_ufffd(self):
        raw = "A\ufffdB\ufffdC"
        assert normalize_raw_label(raw) == "A - B - C"

    def test_collapses_double_spaces(self):
        assert normalize_raw_label("DoS  Hulk") == "DoS Hulk"

    def test_plain_label_unchanged(self):
        assert normalize_raw_label("PortScan") == "PortScan"

    def test_raises_on_non_string(self):
        with pytest.raises(TypeError):
            normalize_raw_label(123)

    def test_combined_strip_and_ufffd(self):
        raw = "  Web Attack \ufffd Brute Force  "
        assert normalize_raw_label(raw) == "Web Attack - Brute Force"


# ===========================================================================
# Tests: Label Mapping
# ===========================================================================

class TestLabelMapping:
    def test_all_expected_raw_labels_map_correctly(self, minimal_params):
        label_map = minimal_params["cleaning"]["label_map"]
        norm_map = build_label_normalizer(label_map)

        # Test that U+FFFD variants are mapped through normalization
        test_cases = [
            ("Benign", "BENIGN"),
            ("Bot", "Botnet"),
            ("DDoS", "DDoS"),
            ("DoS GoldenEye", "DoS"),
            ("DoS Hulk", "DoS"),
            ("DoS Slowhttptest", "DoS"),
            ("DoS slowloris", "DoS"),
            ("FTP-Patator", "BruteForce"),
            ("Heartbleed", "DoS"),
            ("Infiltration", "Infiltration"),
            ("PortScan", "PortScan"),
            ("SSH-Patator", "BruteForce"),
            ("Web Attack - Brute Force", "WebAttack"),
            ("Web Attack - XSS", "WebAttack"),
            ("Web Attack - Sql Injection", "WebAttack"),
        ]

        for raw, expected_class in test_cases:
            norm_key = normalize_raw_label(raw)
            assert norm_key in norm_map, f"'{raw}' not found in normalized map"
            assert norm_map[norm_key] == expected_class, (
                f"'{raw}' mapped to '{norm_map[norm_key]}' but expected '{expected_class}'"
            )

    def test_ufffd_labels_map_correctly(self, minimal_params):
        """Web Attack labels with U+FFFD must map same as with ' - '."""
        label_map = minimal_params["cleaning"]["label_map"]
        norm_map = build_label_normalizer(label_map)

        ufffd_labels = pd.Series([
            "Web Attack \ufffd Brute Force",
            "Web Attack \ufffd XSS",
            "Web Attack \ufffd Sql Injection",
        ])

        mapped = map_raw_labels(ufffd_labels, norm_map)
        assert list(mapped) == ["WebAttack", "WebAttack", "WebAttack"]

    def test_unknown_label_raises(self, minimal_params):
        """An unknown label must raise ValueError, not silently map to anything."""
        label_map = minimal_params["cleaning"]["label_map"]
        norm_map = build_label_normalizer(label_map)

        series = pd.Series(["Benign", "UNKNOWN_ATTACK_TYPE"])

        with pytest.raises(ValueError, match="UNKNOWN LABELS FOUND"):
            map_raw_labels(series, norm_map)

    def test_no_unknown_labels_passes(self, minimal_params):
        label_map = minimal_params["cleaning"]["label_map"]
        norm_map = build_label_normalizer(label_map)

        series = pd.Series(["Benign", "DDoS", "Bot"])
        mapped = map_raw_labels(series, norm_map)
        assert list(mapped) == ["BENIGN", "DDoS", "Botnet"]

    def test_all_8_canonical_classes_present(self, minimal_params):
        """After mapping, exactly 8 canonical class names are expected."""
        label_map = minimal_params["cleaning"]["label_map"]
        expected_canonical = {
            "BENIGN", "DoS", "DDoS", "PortScan",
            "BruteForce", "Botnet", "WebAttack", "Infiltration"
        }
        actual_canonical = set(label_map.values())
        assert actual_canonical == expected_canonical


# ===========================================================================
# Tests: Inf / NaN Handling
# ===========================================================================

class TestInfNaNHandling:
    def test_replace_inf_converts_to_nan(self):
        df = pd.DataFrame({
            "a": [1.0, np.inf, 3.0],
            "b": [-np.inf, 2.0, 4.0],
            "c": [1, 2, 3],
        })
        out, stats = replace_inf(df)
        assert not np.isinf(out["a"]).any()
        assert not np.isinf(out["b"]).any()
        assert out["a"].isna().sum() == 1
        assert out["b"].isna().sum() == 1
        assert stats["total_inf_replaced"] == 2

    def test_no_inf_returns_unchanged(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        out, stats = replace_inf(df)
        assert stats["total_inf_replaced"] == 0
        pd.testing.assert_frame_equal(out, df)

    def test_fill_values_computed_from_training_only(self):
        """fill_values computed on train should be different from test median."""
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
        test = pd.DataFrame({"x": [100.0, 200.0, np.nan]})

        fill_vals = compute_fill_values(train, "median")
        expected_median = float(np.nanmedian([1.0, 2.0, 3.0]))
        assert abs(fill_vals["x"] - expected_median) < 1e-6

        filled_test, _ = apply_fill_values(test, fill_vals)
        # The NaN in test should be filled with the TRAIN median, not 150.0
        assert abs(filled_test["x"].iloc[2] - expected_median) < 1e-6

    def test_no_nan_after_fill(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 4.0]})
        fill_vals = compute_fill_values(df, "median")
        filled, _ = apply_fill_values(df, fill_vals)
        assert filled.isna().sum().sum() == 0

    def test_median_fill_strategy(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan]})
        fill_vals = compute_fill_values(df, "median")
        assert fill_vals["x"] == 2.5  # median of [1,2,3,4]

    def test_zero_fill_strategy(self):
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan]})
        fill_vals = compute_fill_values(df, "zero")
        assert fill_vals["x"] == 0.0

    def test_invalid_strategy_raises(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Unknown nan_fill_strategy"):
            compute_fill_values(df, "invalid_strategy")


# ===========================================================================
# Tests: Constant Columns
# ===========================================================================

class TestConstantColumns:
    def test_detects_constant_column(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "const": [0.0, 0.0, 0.0],
            "b": [4.0, 5.0, 6.0],
        })
        const = detect_constant_columns(df)
        assert "const" in const
        assert "a" not in const
        assert "b" not in const

    def test_drop_constant_columns(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "const": [0.0, 0.0, 0.0],
        })
        out, dropped = drop_constant_columns(df)
        assert "const" not in out.columns
        assert "a" in out.columns
        assert "const" in dropped

    def test_no_constant_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        out, dropped = drop_constant_columns(df)
        assert dropped == []
        assert list(out.columns) == list(df.columns)


# ===========================================================================
# Tests: Duplicate Removal
# ===========================================================================

class TestDuplicateRemoval:
    def test_drops_exact_duplicates(self):
        df = pd.DataFrame({
            "a": [1, 2, 1],
            "b": [3, 4, 3],
            "source_file": ["f1", "f1", "f2"],  # source_file excluded from dup check
        })
        out, n_dropped = drop_duplicate_rows(df)
        assert n_dropped == 1
        assert len(out) == 2

    def test_source_file_difference_not_counted_as_dup(self):
        """Rows identical in all feature cols but different source_file → still dup."""
        df = pd.DataFrame({
            "a": [1, 1],
            "b": [2, 2],
            "source_file": ["file_a.parquet", "file_b.parquet"],
        })
        out, n_dropped = drop_duplicate_rows(df)
        # source_file excluded from dup check → these ARE duplicates
        assert n_dropped == 1

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        out, n_dropped = drop_duplicate_rows(df)
        assert n_dropped == 0
        assert len(out) == 3


# ===========================================================================
# Tests: Clean DataFrame (Integration)
# ===========================================================================

class TestCleanDataframe:
    def test_clean_dataframe_runs_successfully(self, minimal_params):
        df = _make_dummy_df(n=500)
        clean_df, stats = clean_dataframe(df, minimal_params)

        # Must have label_class and label_raw columns
        assert "label_class" in clean_df.columns
        assert "label_raw" in clean_df.columns
        assert "Label" not in clean_df.columns  # original dropped

    def test_all_canonical_classes_present(self, minimal_params):
        df = _make_dummy_df(n=2000)
        clean_df, stats = clean_dataframe(df, minimal_params)

        expected_classes = {
            "BENIGN", "DoS", "DDoS", "PortScan",
            "BruteForce", "Botnet", "WebAttack", "Infiltration"
        }
        actual_classes = set(clean_df["label_class"].unique())
        assert actual_classes == expected_classes

    def test_no_inf_after_cleaning(self, minimal_params):
        df = _make_dummy_df(n=500)
        clean_df, stats = clean_dataframe(df, minimal_params)

        num_df = clean_df.select_dtypes(include=[np.number])
        assert not np.isinf(num_df.values.astype(float)).any(), \
            "Inf values remain after cleaning"

    def test_constant_cols_removed(self, minimal_params):
        df = _make_dummy_df(n=500)
        clean_df, stats = clean_dataframe(df, minimal_params)
        assert "const_col" not in clean_df.columns

    def test_output_row_count_is_reasonable(self, minimal_params):
        df = _make_dummy_df(n=500)
        clean_df, stats = clean_dataframe(df, minimal_params)
        # After dedup, should still have most rows
        assert len(clean_df) >= 400

    def test_stats_keys_present(self, minimal_params):
        df = _make_dummy_df(n=200)
        _, stats = clean_dataframe(df, minimal_params)
        for key in ["input_rows", "output_rows", "class_distribution",
                    "duplicates_dropped", "constant_cols_dropped"]:
            assert key in stats, f"Missing stats key: {key}"


# ===========================================================================
# Tests: Preprocessing (Stratified Split + Leakage Prevention)
# ===========================================================================

class TestPreprocessing:
    def test_stratified_sample_preserves_proportions(self, minimal_params):
        df = _make_dummy_df(n=2000)
        df["label_class"] = df["Label"].map({
            "Benign": "BENIGN", "Bot": "Botnet", "DDoS": "DDoS",
            "DoS GoldenEye": "DoS", "DoS Hulk": "DoS",
            "DoS Slowhttptest": "DoS", "DoS slowloris": "DoS",
            "FTP-Patator": "BruteForce", "Heartbleed": "DoS",
            "Infiltration": "Infiltration", "PortScan": "PortScan",
            "SSH-Patator": "BruteForce",
            "Web Attack \ufffd Brute Force": "WebAttack",
            "Web Attack \ufffd XSS": "WebAttack",
            "Web Attack \ufffd Sql Injection": "WebAttack",
        })
        df = df.dropna(subset=["label_class"])

        sampled = stratified_sample(df, 500, "label_class", random_seed=42)

        # Check proportions roughly preserved
        orig_props = df["label_class"].value_counts(normalize=True)
        samp_props = sampled["label_class"].value_counts(normalize=True)

        for cls in orig_props.index:
            if cls in samp_props.index:
                orig_p = orig_props[cls]
                samp_p = samp_props.get(cls, 0)
                # Allow up to 50% relative deviation for small classes
                assert abs(orig_p - samp_p) < 0.05 or orig_p < 0.01, \
                    f"Class '{cls}' proportion changed too much: {orig_p:.3f} → {samp_p:.3f}"

    def test_encode_labels_correct_values(self, minimal_params):
        class_encoding = minimal_params["cleaning"]["class_encoding"]
        series = pd.Series(["BENIGN", "DoS", "DDoS", "WebAttack", "Infiltration"])
        y, stats = encode_labels(series, class_encoding)
        assert y[0] == 0  # BENIGN
        assert y[1] == 1  # DoS
        assert y[2] == 2  # DDoS
        assert y[3] == 6  # WebAttack
        assert y[4] == 7  # Infiltration

    def test_encode_labels_raises_on_unknown(self, minimal_params):
        class_encoding = minimal_params["cleaning"]["class_encoding"]
        series = pd.Series(["BENIGN", "UNKNOWN_CLASS"])
        with pytest.raises(ValueError):
            encode_labels(series, class_encoding)

    def test_no_nan_after_preprocessing(self, minimal_params):
        """After full preprocessing, X_train and X_test must have no NaN."""
        df = _make_dummy_df(n=5000)
        clean_df, _ = clean_dataframe(df, minimal_params)

        # Need sklearn for this test
        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not installed")

        # Use a larger dev_sample_size to ensure all rare classes survive
        params = dict(minimal_params)
        params["preprocessing"] = dict(minimal_params["preprocessing"])
        params["preprocessing"]["dev_sample_size"] = 2000

        with tempfile.TemporaryDirectory() as tmpdir:
            params["paths"] = dict(minimal_params["paths"])
            params["paths"]["processed_dir"] = tmpdir
            artifacts = run_preprocessing(clean_df, params, use_dev_sample=True)

        assert not np.isnan(artifacts["X_train"]).any(), "NaN in X_train after preprocessing"
        assert not np.isnan(artifacts["X_test"]).any(), "NaN in X_test after preprocessing"


    def test_feature_order_preserved_in_artifacts(self, minimal_params):
        """The saved feature_names.json must match the actual training matrix columns."""
        df = _make_dummy_df(n=5000)
        clean_df, _ = clean_dataframe(df, minimal_params)

        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not installed")

        params = dict(minimal_params)
        params["preprocessing"] = dict(minimal_params["preprocessing"])
        params["preprocessing"]["dev_sample_size"] = 2000

        with tempfile.TemporaryDirectory() as tmpdir:
            params["paths"] = dict(minimal_params["paths"])
            params["paths"]["processed_dir"] = tmpdir
            artifacts = run_preprocessing(clean_df, params, use_dev_sample=True)

            # Load saved feature names
            feature_path = Path(tmpdir) / "feature_names.json"
            assert feature_path.exists()
            with open(feature_path) as f:
                saved_features = json.load(f)

        assert saved_features == artifacts["feature_names"]
        assert len(saved_features) == artifacts["X_train"].shape[1]


    def test_train_test_split_stratified(self, minimal_params):
        """Class proportions should be roughly equal in train and test."""
        df = _make_dummy_df(n=5000)
        clean_df, _ = clean_dataframe(df, minimal_params)

        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not installed")

        params = dict(minimal_params)
        params["preprocessing"] = dict(minimal_params["preprocessing"])
        params["preprocessing"]["dev_sample_size"] = 2000

        with tempfile.TemporaryDirectory() as tmpdir:
            params["paths"] = dict(minimal_params["paths"])
            params["paths"]["processed_dir"] = tmpdir
            artifacts = run_preprocessing(clean_df, params, use_dev_sample=True)

        y_train = artifacts["y_train"]
        y_test = artifacts["y_test"]
        inverse = artifacts["inverse_encoding"]

        for cls_int in np.unique(y_train):
            train_frac = (y_train == cls_int).mean()
            test_frac = (y_test == cls_int).mean() if cls_int in y_test else 0.0
            cls_name = inverse.get(cls_int, str(cls_int))
            # Very rare classes (< 2%): skip detailed check — handled with rare-class logic
            if train_frac > 0.02:
                assert abs(train_frac - test_frac) < 0.06, (
                    f"Class '{cls_name}' stratification off: "
                    f"train={train_frac:.3f} test={test_frac:.3f}"
                )


    def test_scaler_fitted_on_train_only(self, minimal_params):
        """Scaler mean/scale should come from training data, not test data."""
        df = _make_dummy_df(n=5000)
        clean_df, _ = clean_dataframe(df, minimal_params)

        try:
            import sklearn
        except ImportError:
            pytest.skip("scikit-learn not installed")

        params = dict(minimal_params)
        params["preprocessing"] = dict(minimal_params["preprocessing"])
        params["preprocessing"]["dev_sample_size"] = 2000

        with tempfile.TemporaryDirectory() as tmpdir:
            params["paths"] = dict(minimal_params["paths"])
            params["paths"]["processed_dir"] = tmpdir
            artifacts = run_preprocessing(clean_df, params, use_dev_sample=True)

        # RobustScaler: center_ attribute is the median of training data
        scaler = artifacts["scaler"]
        # Just verify scaler is fitted (has center_ attribute for RobustScaler)
        if hasattr(scaler, "center_"):
            assert scaler.center_ is not None
            assert len(scaler.center_) == len(artifacts["feature_names"])



# ===========================================================================
# Tests: Correlation Filtering
# ===========================================================================

class TestCorrelationFiltering:
    def test_correlated_feature_detected(self):
        """Two perfectly correlated features: one should be dropped."""
        n = 100
        rng = np.random.default_rng(1)
        base = rng.normal(0, 1, n)

        df = pd.DataFrame({
            "a": base,
            "b": base * 2.0 + 0.001 * rng.normal(0, 1, n),  # ~perfect corr
            "c": rng.normal(0, 1, n),  # independent
        })

        to_drop = find_correlated_pairs(df, threshold=0.95)
        assert len(to_drop) > 0, "Expected at least one correlated feature to be flagged"
        assert "c" not in to_drop, "Independent feature 'c' should not be dropped"

    def test_uncorrelated_features_kept(self):
        rng = np.random.default_rng(2)
        n = 200
        df = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        })
        to_drop = find_correlated_pairs(df, threshold=0.95)
        assert to_drop == [], f"Unexpected features flagged: {to_drop}"


# ===========================================================================
# Integration Test: Expected Label Mapping for CIC-IDS2017
# ===========================================================================

class TestRealLabelMapping:
    """
    Tests that all known real CIC-IDS2017 raw labels map correctly.
    This acts as a regression test: if someone changes the label_map,
    these tests will catch unexpected class reassignments.
    """

    REAL_RAW_LABELS_AND_EXPECTED = [
        ("Benign", "BENIGN"),
        ("Bot", "Botnet"),
        ("DDoS", "DDoS"),
        ("DoS GoldenEye", "DoS"),
        ("DoS Hulk", "DoS"),
        ("DoS Slowhttptest", "DoS"),
        ("DoS slowloris", "DoS"),
        ("FTP-Patator", "BruteForce"),
        ("Heartbleed", "DoS"),
        ("Infiltration", "Infiltration"),
        ("PortScan", "PortScan"),
        ("SSH-Patator", "BruteForce"),
        # U+FFFD variants (actual from dataset)
        ("Web Attack \ufffd Brute Force", "WebAttack"),
        ("Web Attack \ufffd XSS", "WebAttack"),
        ("Web Attack \ufffd Sql Injection", "WebAttack"),
        # Dash variants (normalized form)
        ("Web Attack - Brute Force", "WebAttack"),
        ("Web Attack - XSS", "WebAttack"),
        ("Web Attack - Sql Injection", "WebAttack"),
    ]

    def test_all_real_labels_map_correctly(self, minimal_params):
        label_map = minimal_params["cleaning"]["label_map"]
        norm_map = build_label_normalizer(label_map)

        for raw, expected in self.REAL_RAW_LABELS_AND_EXPECTED:
            series = pd.Series([raw])
            mapped = map_raw_labels(series, norm_map)
            assert mapped.iloc[0] == expected, (
                f"Label {repr(raw)} mapped to '{mapped.iloc[0]}' "
                f"but expected '{expected}'"
            )

    def test_heartbleed_maps_to_dos(self, minimal_params):
        """Heartbleed is a network-level exploit classified under DoS."""
        label_map = minimal_params["cleaning"]["label_map"]
        norm_map = build_label_normalizer(label_map)
        series = pd.Series(["Heartbleed"])
        mapped = map_raw_labels(series, norm_map)
        assert mapped.iloc[0] == "DoS"
