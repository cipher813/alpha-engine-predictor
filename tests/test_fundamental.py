"""
tests/test_fundamental.py — Tests for fundamental features.

Tests the fundamental fetcher (with mocked FMP responses) and backward
compatibility of compute_features() with and without fundamental_data.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_store.fundamental_fetcher import (
    fetch_fundamental_data,
    _fetch_single_ticker,
    _NEUTRAL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n_rows: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV data (same helper as test_feature_engineer.py)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2024-06-01", periods=n_rows)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n_rows))
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0, 0.02, n_rows))
    low = close * (1 - rng.uniform(0, 0.02, n_rows))
    volume = rng.integers(1_000_000, 10_000_000, n_rows).astype(float)
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# Mock FMP responses
_MOCK_KEY_METRICS = [{
    "peRatioTTM": 25.5,
    "pbRatioTTM": 4.2,
    "roeTTM": 0.35,
    "freeCashFlowYieldTTM": 0.04,
    "currentRatioTTM": 1.8,
    "debtToEquityTTM": 0.95,
}]

_MOCK_INCOME_5Q = [
    {"revenue": 120_000_000, "grossProfit": 50_000_000},  # most recent
    {"revenue": 115_000_000, "grossProfit": 47_000_000},
    {"revenue": 110_000_000, "grossProfit": 45_000_000},
    {"revenue": 105_000_000, "grossProfit": 42_000_000},
    {"revenue": 100_000_000, "grossProfit": 40_000_000},  # year-ago quarter
]


# ---------------------------------------------------------------------------
# Fundamental fetcher tests
# ---------------------------------------------------------------------------

class TestFundamentalFetcher:
    @patch("feature_store.fundamental_fetcher._fmp_get")
    def test_fetch_single_ticker_normalizes(self, mock_fmp):
        mock_fmp.side_effect = [
            _MOCK_KEY_METRICS,
            _MOCK_INCOME_5Q,
        ]

        result = _fetch_single_ticker("AAPL")

        assert abs(result["pe_ratio"] - 25.5 / 30.0) < 0.01
        assert abs(result["pb_ratio"] - 4.2 / 5.0) < 0.01
        assert abs(result["debt_to_equity"] - 0.95 / 2.0) < 0.01
        assert abs(result["revenue_growth_yoy"] - 0.20) < 0.01  # 120M/100M - 1
        assert abs(result["gross_margin"] - 50 / 120) < 0.01
        assert abs(result["roe"] - 0.35) < 0.01
        assert abs(result["fcf_yield"] - 0.04) < 0.01
        assert abs(result["current_ratio"] - 1.8 / 3.0) < 0.01

    @patch("feature_store.fundamental_fetcher._fmp_get")
    def test_fetch_returns_neutral_on_api_failure(self, mock_fmp):
        mock_fmp.side_effect = Exception("API timeout")

        result = fetch_fundamental_data(["AAPL"])

        assert result["AAPL"] == _NEUTRAL

    @patch("feature_store.fundamental_fetcher._fmp_get")
    def test_fetch_returns_neutral_on_empty_response(self, mock_fmp):
        mock_fmp.return_value = []

        result = _fetch_single_ticker("AAPL")

        assert result == _NEUTRAL

    @patch("feature_store.fundamental_fetcher._fmp_get")
    def test_fetch_clips_extreme_values(self, mock_fmp):
        mock_fmp.side_effect = [
            [{"peRatioTTM": 500.0, "pbRatioTTM": -10.0, "roeTTM": 5.0,
              "freeCashFlowYieldTTM": 2.0, "currentRatioTTM": 20.0,
              "debtToEquityTTM": 100.0}],
            _MOCK_INCOME_5Q,
        ]

        result = _fetch_single_ticker("WILD")

        assert result["pe_ratio"] == 3.0      # clipped at 3.0
        assert result["pb_ratio"] == -2.0      # -10/5 = -2.0, clipped at -3.0? No, -2.0 is within range
        assert result["roe"] == 1.0            # clipped at 1.0
        assert result["fcf_yield"] == 0.5      # clipped at 0.5

    @patch("feature_store.fundamental_fetcher._fmp_get")
    def test_fetch_multiple_tickers(self, mock_fmp):
        mock_fmp.side_effect = [
            _MOCK_KEY_METRICS, _MOCK_INCOME_5Q,  # AAPL
            _MOCK_KEY_METRICS, _MOCK_INCOME_5Q,  # MSFT
        ]

        result = fetch_fundamental_data(["AAPL", "MSFT"])

        assert "AAPL" in result
        assert "MSFT" in result
        assert len(result["AAPL"]) == 8
        assert len(result["MSFT"]) == 8


# ---------------------------------------------------------------------------
# compute_features() backward compatibility tests
# ---------------------------------------------------------------------------

class TestComputeFeaturesWithFundamentals:
    def test_without_fundamental_data_still_works(self):
        """Existing callers that don't pass fundamental_data should still work."""
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        result = compute_features(df)

        assert not result.empty
        # Fundamental columns should exist with neutral values
        assert "pe_ratio" in result.columns
        assert (result["pe_ratio"] == 0.0).all()
        assert "gross_margin" in result.columns
        assert (result["gross_margin"] == 0.0).all()

    def test_with_fundamental_data_populates_columns(self):
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        fund = {
            "pe_ratio": 0.85,
            "pb_ratio": 0.84,
            "debt_to_equity": 0.475,
            "revenue_growth_yoy": 0.20,
            "fcf_yield": 0.04,
            "gross_margin": 0.42,
            "roe": 0.35,
            "current_ratio": 0.6,
        }

        result = compute_features(df, fundamental_data=fund)

        assert not result.empty
        assert (result["pe_ratio"] == 0.85).all()
        assert (result["gross_margin"] == 0.42).all()
        assert (result["roe"] == 0.35).all()

    def test_feature_count_is_49(self):
        """Verify the total feature count after adding fundamentals."""
        from config import FEATURES, N_FEATURES

        assert len(FEATURES) == 49
        assert N_FEATURES == 49

    def test_features_to_array_shape(self):
        """features_to_array should produce 49 columns."""
        from data.feature_engineer import compute_features, features_to_array

        df = _make_ohlcv(300)
        result = compute_features(df)
        arr = features_to_array(result)

        assert arr.shape[1] == 49

    def test_gbm_features_excludes_macro_and_fundamental(self):
        """GBM features should exclude both macro and fundamental (until validated)."""
        from config import GBM_FEATURES, MACRO_FEATURES, FUNDAMENTAL_FEATURES

        for f in FUNDAMENTAL_FEATURES:
            assert f not in GBM_FEATURES, f"Fundamental feature {f} should not be in GBM_FEATURES yet"
        for f in MACRO_FEATURES:
            assert f not in GBM_FEATURES

    def test_fundamental_features_in_full_features_list(self):
        """Fundamental features should be in FEATURES (computed and stored) even if not in GBM."""
        from config import FEATURES, FUNDAMENTAL_FEATURES

        for f in FUNDAMENTAL_FEATURES:
            assert f in FEATURES, f"Fundamental feature {f} missing from FEATURES"
