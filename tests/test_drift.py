"""
SentinelOps - Drift Detection Tests
====================================
Pins the behaviour of the drift verdict, which is what would trigger
retraining in production.

The tests build synthetic reference/current frames rather than depending on
pipeline artifacts, so they run on a fresh clone and assert the statistics
rather than a particular dataset's numbers.

Particular attention to the large-sample trap: a KS test over tens of
thousands of rows reports p ~ 0 for differences far too small to act on, so
identical-distribution batches would be flagged as drifted. The "auto"
strategy exists to prevent exactly that, and is tested for it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mlops.drift import (
    DEFAULT_WASSERSTEIN_THRESHOLD,
    LARGE_SAMPLE_THRESHOLD,
    detect_drift,
)

FEATURES = [f"feature_{i}" for i in range(10)]


def make_frame(n, shift=0.0, scale=1.0, seed=0, n_shifted=10):
    """Build a frame whose first `n_shifted` columns are offset by `shift`."""
    rng = np.random.default_rng(seed)
    data = {}
    for i, name in enumerate(FEATURES):
        offset = shift if i < n_shifted else 0.0
        spread = scale if i < n_shifted else 1.0
        data[name] = rng.normal(offset, spread, n)
    return pd.DataFrame(data)


class TestNoDriftCases:
    def test_identical_distributions_report_no_drift(self):
        ref = make_frame(5000, seed=1)
        cur = make_frame(5000, seed=2)
        result = detect_drift(ref, cur, FEATURES, stattest="auto")
        assert result["drift_detected"] is False
        assert result["n_drifted_features"] == 0

    def test_large_identical_samples_do_not_false_positive(self):
        """
        The large-sample trap: with 40k rows per side a KS test returns
        p ~ 0 on noise alone. 'auto' must not report drift here.
        """
        n = 40_000
        assert n > LARGE_SAMPLE_THRESHOLD
        ref = make_frame(n, seed=11)
        cur = make_frame(n, seed=12)
        result = detect_drift(ref, cur, FEATURES, stattest="auto")
        assert result["drift_detected"] is False, (
            f"auto strategy false-positived on identical distributions: "
            f"{result['n_drifted_features']}/{result['n_features']} flagged"
        )
        assert all(r["test"] == "wasserstein" for r in result["per_feature"].values())

    def test_single_shifted_feature_is_below_share_threshold(self):
        """One moved column out of ten is noise, not a changed world."""
        ref = make_frame(5000, seed=3)
        cur = make_frame(5000, shift=3.0, seed=4, n_shifted=1)
        result = detect_drift(ref, cur, FEATURES, stattest="auto", drift_share_threshold=0.20)
        assert result["n_drifted_features"] == 1
        assert result["drift_share"] == 0.1
        assert result["drift_detected"] is False


class TestDriftCases:
    def test_shifted_distribution_is_detected(self):
        ref = make_frame(5000, seed=5)
        cur = make_frame(5000, shift=2.0, seed=6)
        result = detect_drift(ref, cur, FEATURES, stattest="auto")
        assert result["drift_detected"] is True
        assert result["n_drifted_features"] == len(FEATURES)

    def test_variance_change_is_detected(self):
        ref = make_frame(5000, seed=7)
        cur = make_frame(5000, scale=5.0, seed=8)
        result = detect_drift(ref, cur, FEATURES, stattest="auto")
        assert result["drift_detected"] is True

    def test_majority_shifted_crosses_share_threshold(self):
        ref = make_frame(5000, seed=9)
        cur = make_frame(5000, shift=3.0, seed=10, n_shifted=5)
        result = detect_drift(ref, cur, FEATURES, stattest="auto", drift_share_threshold=0.20)
        assert result["n_drifted_features"] == 5
        assert result["drift_share"] == 0.5
        assert result["drift_detected"] is True

    def test_top_drifted_features_ranked_by_magnitude(self):
        """The report must lead with the features actually responsible."""
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({f: rng.normal(0, 1, 4000) for f in FEATURES})
        cur = pd.DataFrame({
            f: rng.normal(6.0 if f == "feature_3" else 2.0, 1, 4000) for f in FEATURES
        })
        result = detect_drift(ref, cur, FEATURES, stattest="auto")
        assert result["top_drifted_features"][0]["feature"] == "feature_3"


class TestStattestSelection:
    def test_auto_uses_ks_on_small_samples(self):
        ref = make_frame(200, seed=20)
        cur = make_frame(200, seed=21)
        result = detect_drift(ref, cur, FEATURES, stattest="auto")
        assert all(r["test"] == "ks" for r in result["per_feature"].values())

    def test_auto_uses_wasserstein_on_large_samples(self):
        ref = make_frame(5000, seed=22)
        cur = make_frame(5000, seed=23)
        result = detect_drift(ref, cur, FEATURES, stattest="auto")
        assert all(r["test"] == "wasserstein" for r in result["per_feature"].values())

    def test_explicit_ks_is_honoured_regardless_of_size(self):
        ref = make_frame(5000, seed=24)
        cur = make_frame(5000, seed=25)
        result = detect_drift(ref, cur, FEATURES, stattest="ks")
        assert all(r["test"] == "ks" for r in result["per_feature"].values())

    def test_wasserstein_score_is_scale_free(self):
        """
        Normalizing by reference spread means the same relative shift scores
        the same whether the feature is measured in bytes or microseconds.
        """
        rng = np.random.default_rng(30)
        small_ref = pd.DataFrame({"a": rng.normal(0, 1, 4000)})
        small_cur = pd.DataFrame({"a": rng.normal(1, 1, 4000)})
        big_ref = pd.DataFrame({"a": rng.normal(0, 1000, 4000)})
        big_cur = pd.DataFrame({"a": rng.normal(1000, 1000, 4000)})

        s = detect_drift(small_ref, small_cur, ["a"], stattest="wasserstein")
        b = detect_drift(big_ref, big_cur, ["a"], stattest="wasserstein")
        assert abs(s["per_feature"]["a"]["score"] - b["per_feature"]["a"]["score"]) < 0.1


class TestRobustness:
    def test_non_overlapping_columns_raise(self):
        ref = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        cur = pd.DataFrame({"b": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="No overlapping"):
            detect_drift(ref, cur)

    def test_compares_only_shared_columns(self):
        """The Champion may consume a subset of the reference's features."""
        ref = make_frame(2000, seed=40)
        cur = make_frame(2000, seed=41).loc[:, FEATURES[:4]]
        result = detect_drift(ref, cur, stattest="auto")
        assert result["n_features"] == 4

    def test_nan_values_are_ignored_not_fatal(self):
        ref = make_frame(2000, seed=42)
        cur = make_frame(2000, seed=43)
        cur.loc[:200, "feature_0"] = np.nan
        result = detect_drift(ref, cur, FEATURES, stattest="auto")
        assert result["per_feature"]["feature_0"]["test"] != "none"

    def test_constant_feature_does_not_divide_by_zero(self):
        ref = pd.DataFrame({"a": np.zeros(3000)})
        cur = pd.DataFrame({"a": np.zeros(3000)})
        result = detect_drift(ref, cur, ["a"], stattest="wasserstein")
        assert result["per_feature"]["a"]["drifted"] is False
        assert np.isfinite(result["per_feature"]["a"]["score"])

    def test_summary_is_json_serializable(self):
        import json
        ref = make_frame(2000, seed=50)
        cur = make_frame(2000, shift=2.0, seed=51)
        result = detect_drift(ref, cur, FEATURES, stattest="auto")
        json.dumps({k: v for k, v in result.items() if k != "per_feature"})
