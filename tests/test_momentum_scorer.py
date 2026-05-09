"""Tests for ``model/momentum_scorer.py`` — the deterministic momentum L1.

Coverage migrated from ``test_subsample_validator.py``'s ``TestMomentumBaseline``
class when the same formula stopped being a "baseline a GBM must beat" and
became the L1 component itself (2026-05-09).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model import momentum_scorer


# ── predict_array (vectorized, raw-feature matrix) ───────────────────────────


class TestPredictArray:
    def test_clean_features_match_canonical_formula(self):
        """0.4·m5 + 0.3·m20 + 0.2·ma50 + 0.1·(rsi-50)/100"""
        feature_names = ["momentum_5d", "momentum_20d", "price_vs_ma50", "rsi_14"]
        X = np.array([[0.05, 0.10, 0.02, 70.0]])
        preds = momentum_scorer.predict_array(X, feature_names)
        expected = 0.4 * 0.05 + 0.3 * 0.10 + 0.2 * 0.02 + 0.1 * (70 - 50) / 100
        assert preds[0] == pytest.approx(expected)

    def test_nan_features_use_neutral_defaults(self):
        feature_names = ["momentum_5d", "momentum_20d", "price_vs_ma50", "rsi_14"]
        X = np.array([[np.nan, np.nan, np.nan, np.nan]])
        preds = momentum_scorer.predict_array(X, feature_names)
        # All NaN → m5=0, m20=0, ma50=0, rsi=50 → score = 0
        assert preds[0] == pytest.approx(0.0)

    def test_partial_nan_uses_neutral_for_missing_only(self):
        feature_names = ["momentum_5d", "momentum_20d", "price_vs_ma50", "rsi_14"]
        X = np.array([[0.05, np.nan, np.nan, np.nan]])
        preds = momentum_scorer.predict_array(X, feature_names)
        assert preds[0] == pytest.approx(0.02)  # 0.4 * 0.05

    def test_missing_feature_name_uses_default(self):
        feature_names = ["momentum_5d"]
        X = np.array([[0.10]])
        preds = momentum_scorer.predict_array(X, feature_names)
        # m5=0.10, others default → 0.4 * 0.10 = 0.04
        assert preds[0] == pytest.approx(0.04)

    def test_batch_shape_preserved(self):
        feature_names = ["momentum_5d", "momentum_20d", "price_vs_ma50", "rsi_14"]
        X = np.random.default_rng(0).normal(0, 0.1, size=(50, 4))
        preds = momentum_scorer.predict_array(X, feature_names)
        assert preds.shape == (50,)


# ── predict_dict (per-row, inference-time) ──────────────────────────────────


class TestPredictDict:
    def test_clean_dict_matches_array_path(self):
        """predict_dict and predict_array must agree on the same row."""
        features = {
            "momentum_5d": 0.05,
            "momentum_20d": 0.10,
            "price_vs_ma50": 0.02,
            "rsi_14": 70.0,
        }
        score_dict = momentum_scorer.predict_dict(features)
        score_array = momentum_scorer.predict_array(
            np.array([[0.05, 0.10, 0.02, 70.0]]),
            ["momentum_5d", "momentum_20d", "price_vs_ma50", "rsi_14"],
        )[0]
        assert score_dict == pytest.approx(score_array)

    def test_missing_keys_use_neutral_defaults(self):
        # Empty dict → all defaults (m5=0, m20=0, ma50=0, rsi=50) → score = 0
        assert momentum_scorer.predict_dict({}) == pytest.approx(0.0)

    def test_none_values_use_neutral_defaults(self):
        features = {"momentum_5d": None, "momentum_20d": None, "price_vs_ma50": None, "rsi_14": None}
        assert momentum_scorer.predict_dict(features) == pytest.approx(0.0)

    def test_nan_values_use_neutral_defaults(self):
        features = {"momentum_5d": float("nan"), "rsi_14": float("nan")}
        # m5=0, m20=0, ma50=0 (missing), rsi=50 → score = 0
        assert momentum_scorer.predict_dict(features) == pytest.approx(0.0)

    def test_string_values_use_default(self):
        features = {"momentum_5d": "garbage", "momentum_20d": 0.10}
        # m5 → default 0, m20 → 0.10, others default → 0.3 * 0.10 = 0.03
        assert momentum_scorer.predict_dict(features) == pytest.approx(0.03)

    def test_pandas_series_input(self):
        """Inference passes a pd.Series of features for one ticker."""
        latest = pd.Series({
            "momentum_5d": 0.05,
            "momentum_20d": 0.10,
            "price_vs_ma50": 0.02,
            "rsi_14": 70.0,
            "extra_unused_feature": 99.0,
        })
        score = momentum_scorer.predict_dict(latest)
        expected = 0.4 * 0.05 + 0.3 * 0.10 + 0.2 * 0.02 + 0.1 * (70 - 50) / 100
        assert score == pytest.approx(expected)
