"""
tests/test_feature_engineer.py — Unit tests for data/feature_engineer.py.

Tests feature computation, cross-sectional rank normalization, missing data
handling, and macro feature forward-fill behavior. No S3 or network calls.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n_rows: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic OHLCV DataFrame with a DatetimeIndex.
    300 rows is enough to pass the 252-row warmup requirement.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2024-06-01", periods=n_rows)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n_rows))
    close = np.maximum(close, 1.0)  # avoid zero/negative
    high = close * (1 + rng.uniform(0, 0.02, n_rows))
    low = close * (1 - rng.uniform(0, 0.02, n_rows))
    volume = rng.integers(1_000_000, 10_000_000, n_rows).astype(float)

    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _make_spy_series(index: pd.DatetimeIndex, seed: int = 99) -> pd.Series:
    """Create a synthetic SPY price series aligned to the given index."""
    rng = np.random.default_rng(seed)
    prices = 400.0 + np.cumsum(rng.normal(0, 0.3, len(index)))
    return pd.Series(prices, index=index, name="SPY")


# ---------------------------------------------------------------------------
# Tests: compute_features
# ---------------------------------------------------------------------------

class TestComputeFeatures:
    """Tests for data.feature_engineer.compute_features()."""

    def test_produces_expected_feature_count(self):
        """compute_features should produce exactly N_FEATURES feature columns."""
        from config import FEATURES, N_FEATURES
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        result = compute_features(df)

        # All expected feature columns must be present
        for feat in FEATURES:
            assert feat in result.columns, f"Missing feature column: {feat}"

        # Verify the constant matches the list length
        assert len(FEATURES) == N_FEATURES
        assert N_FEATURES == 41

    def test_output_rows_fewer_than_input(self):
        """Warmup rows (252+) should be dropped, so output has fewer rows."""
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        result = compute_features(df)

        assert len(result) < len(df)
        assert len(result) > 0

    def test_no_nan_in_feature_columns(self):
        """After compute_features, feature columns should contain no NaN."""
        from config import FEATURES
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        result = compute_features(df)

        nan_counts = result[FEATURES].isna().sum()
        assert nan_counts.sum() == 0, f"NaN found in features: {nan_counts[nan_counts > 0]}"

    def test_empty_dataframe_returns_empty(self):
        """An empty input DataFrame should return an empty DataFrame."""
        from data.feature_engineer import compute_features

        df = pd.DataFrame(columns=["Close", "Volume"])
        result = compute_features(df)
        assert result.empty

    def test_rsi_range(self):
        """RSI values should be in the [0, 100] range."""
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        result = compute_features(df)

        assert result["rsi_14"].min() >= 0
        assert result["rsi_14"].max() <= 100

    def test_macd_cross_values(self):
        """macd_cross should only contain -1, 0, or 1."""
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        result = compute_features(df)

        unique_vals = set(result["macd_cross"].unique())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_macd_above_zero_binary(self):
        """macd_above_zero should be binary (0.0 or 1.0)."""
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        result = compute_features(df)

        unique_vals = set(result["macd_above_zero"].unique())
        assert unique_vals.issubset({0.0, 1.0})

    def test_with_spy_series(self):
        """return_vs_spy_5d should be non-zero when SPY series is provided."""
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        spy = _make_spy_series(df.index)
        result = compute_features(df, spy_series=spy)

        # With a real SPY series, return_vs_spy_5d should vary
        assert not (result["return_vs_spy_5d"] == 0.0).all()

    def test_without_spy_series_defaults_to_zero(self):
        """return_vs_spy_5d should be 0.0 when SPY series is None."""
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        result = compute_features(df, spy_series=None)

        assert (result["return_vs_spy_5d"] == 0.0).all()

    def test_macro_forward_fill(self):
        """
        Macro series (VIX, TNX, etc.) should be forward-filled on alignment
        so that weekends/holidays don't create NaNs.
        """
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)

        # Create a VIX series with some dates missing (simulating holidays)
        vix_index = df.index[::2]  # every other day
        vix_vals = np.full(len(vix_index), 25.0)
        vix_series = pd.Series(vix_vals, index=vix_index)

        result = compute_features(df, vix_series=vix_series)

        # vix_level should have no NaN — forward fill should handle gaps
        assert not result["vix_level"].isna().any()

    def test_vix_neutral_default(self):
        """Without VIX series, vix_level should default to 1.0 (neutral)."""
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        result = compute_features(df, vix_series=None)

        assert (result["vix_level"] == 1.0).all()

    def test_earnings_data_defaults(self):
        """Without earnings_data, alternative features should default to neutral."""
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        result = compute_features(df, earnings_data=None)

        assert (result["earnings_surprise_pct"] == 0.0).all()
        assert (result["days_since_earnings"] == 1.0).all()

    def test_earnings_data_custom(self):
        """With earnings_data, values should be reflected in the output."""
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        earnings = {"surprise_pct": 5.2, "days_since_earnings": 10}
        result = compute_features(df, earnings_data=earnings)

        assert (result["earnings_surprise_pct"] == 5.2).all()
        assert np.isclose(result["days_since_earnings"].iloc[0], 10 / 90.0)

    def test_volume_price_div_values(self):
        """volume_price_div should only contain -1, 0, or 1."""
        from data.feature_engineer import compute_features

        df = _make_ohlcv(300)
        result = compute_features(df)

        unique_vals = set(result["volume_price_div"].unique())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})


# ---------------------------------------------------------------------------
# Tests: features_to_array
# ---------------------------------------------------------------------------

class TestFeaturesToArray:
    """Tests for data.feature_engineer.features_to_array()."""

    def test_output_shape(self):
        """features_to_array should return (N, N_FEATURES) array."""
        from config import N_FEATURES
        from data.feature_engineer import compute_features, features_to_array

        df = _make_ohlcv(300)
        featured = compute_features(df)
        arr = features_to_array(featured)

        assert arr.shape == (len(featured), N_FEATURES)
        assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: cross_sectional_rank_normalize
# ---------------------------------------------------------------------------

class TestCrossSectionalRankNormalize:
    """Tests for data.dataset.cross_sectional_rank_normalize()."""

    def test_output_in_zero_one_range(self):
        """Rank-normalized output should be in [0, 1]."""
        from data.dataset import cross_sectional_rank_normalize

        rng = np.random.default_rng(42)
        n_tickers = 20
        n_features = 5
        n_dates = 10

        dates = []
        for d in range(n_dates):
            dates.extend([f"2024-01-{d+1:02d}"] * n_tickers)

        X = rng.standard_normal((n_tickers * n_dates, n_features)).astype(np.float32)
        X_ranked = cross_sectional_rank_normalize(X, dates)

        assert X_ranked.min() >= 0.0
        assert X_ranked.max() <= 1.0

    def test_single_ticker_per_date_gets_midpoint(self):
        """A single ticker on a date should get rank 0.5."""
        from data.dataset import cross_sectional_rank_normalize

        X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        dates = ["2024-01-01"]
        X_ranked = cross_sectional_rank_normalize(X, dates)

        np.testing.assert_array_equal(X_ranked[0], [0.5, 0.5, 0.5])

    def test_preserves_relative_order(self):
        """Higher raw values should get higher ranks within the same date."""
        from data.dataset import cross_sectional_rank_normalize

        X = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ], dtype=np.float32)
        dates = ["2024-01-01"] * 3
        X_ranked = cross_sectional_rank_normalize(X, dates)

        # Row with highest values should have rank 1.0
        assert X_ranked[2, 0] == 1.0
        assert X_ranked[0, 0] == 0.0

    def test_tied_values_get_averaged_rank(self):
        """Tied values within a date should receive the average rank."""
        from data.dataset import cross_sectional_rank_normalize

        X = np.array([
            [5.0],
            [5.0],
            [10.0],
        ], dtype=np.float32)
        dates = ["2024-01-01"] * 3
        X_ranked = cross_sectional_rank_normalize(X, dates)

        # Tied values (rows 0 and 1) should have averaged rank
        assert X_ranked[0, 0] == X_ranked[1, 0]
        assert X_ranked[2, 0] == 1.0  # highest

    def test_different_dates_are_independent(self):
        """Normalization should be per-date, not across dates."""
        from data.dataset import cross_sectional_rank_normalize

        X = np.array([
            [100.0],   # date 1 — this is highest within its date
            [1.0],     # date 1 — lowest
            [1000.0],  # date 2 — only ticker
        ], dtype=np.float32)
        dates = ["2024-01-01", "2024-01-01", "2024-01-02"]
        X_ranked = cross_sectional_rank_normalize(X, dates)

        # Date 1: row 0 should be 1.0 (highest), row 1 should be 0.0
        assert X_ranked[0, 0] == 1.0
        assert X_ranked[1, 0] == 0.0
        # Date 2: single ticker → 0.5
        assert X_ranked[2, 0] == 0.5

    def test_output_shape_matches_input(self):
        """Output shape should match input shape exactly."""
        from data.dataset import cross_sectional_rank_normalize

        X = np.random.randn(50, 10).astype(np.float32)
        dates = [f"2024-01-{(i // 10) + 1:02d}" for i in range(50)]
        X_ranked = cross_sectional_rank_normalize(X, dates)

        assert X_ranked.shape == X.shape
