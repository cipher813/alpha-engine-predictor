"""
tests/test_features.py — Unit tests for data/feature_engineer.py.

Tests that compute_features() produces the correct column names, handles
sufficient history, produces no NaN values after warmup, and keeps feature
values within expected ranges.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the repo root is on sys.path when running pytest from the root dir
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.feature_engineer import compute_features, features_to_array
from config import FEATURES, N_FEATURES


def _make_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame with n rows.
    Prices follow a random walk; volume is random positive integers.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")  # business days

    # Random-walk log returns, then cumulative price
    log_returns = rng.normal(0.0, 0.01, size=n)
    price = 100.0 * np.exp(np.cumsum(log_returns))

    # Plausible OHLCV structure
    noise = rng.uniform(0.99, 1.01, size=n)
    df = pd.DataFrame(
        {
            "Open": price * rng.uniform(0.995, 1.005, size=n),
            "High": price * rng.uniform(1.001, 1.015, size=n),
            "Low": price * rng.uniform(0.985, 0.999, size=n),
            "Close": price * noise,
            "Volume": rng.integers(1_000_000, 10_000_000, size=n).astype(float),
        },
        index=dates,
    )
    return df


class TestComputeFeatures:
    """Tests for compute_features()."""

    def test_output_has_all_feature_columns(self):
        """compute_features() must return a DataFrame with all feature columns."""
        df = _make_ohlcv(250)
        result = compute_features(df)

        for col in FEATURES:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_column_count(self):
        """All config.FEATURES must be present."""
        df = _make_ohlcv(250)
        result = compute_features(df)
        for feat in FEATURES:
            assert feat in result.columns

    def test_no_nan_in_features_after_warmup(self):
        """After dropping warmup rows, no NaN should remain in any feature column."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        assert len(result) > 0, "Expected at least some rows after feature computation"

        nan_counts = result[FEATURES].isnull().sum()
        for col, count in nan_counts.items():
            assert count == 0, f"Column {col} has {count} NaN values"

    def test_minimum_rows_for_ma200(self):
        """MA200 requires 200+ rows; fewer than 200 should yield empty output."""
        df_small = _make_ohlcv(150)  # not enough for MA200
        result = compute_features(df_small)
        # All rows should be dropped because price_vs_ma200 will be NaN
        assert len(result) == 0, (
            f"Expected 0 rows for 150-row input (MA200 requires 200), got {len(result)}"
        )

    def test_rsi_in_valid_range(self):
        """RSI values must be between 0 and 100 (inclusive)."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        assert result["rsi_14"].min() >= 0.0, "RSI below 0"
        assert result["rsi_14"].max() <= 100.0, "RSI above 100"

    def test_macd_above_zero_is_binary(self):
        """macd_above_zero must only contain 0.0 or 1.0."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        unique_vals = set(result["macd_above_zero"].unique())
        assert unique_vals.issubset({0.0, 1.0}), (
            f"macd_above_zero contains unexpected values: {unique_vals}"
        )

    def test_macd_cross_in_valid_range(self):
        """macd_cross must only contain -1.0, 0.0, or 1.0."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        unique_vals = set(result["macd_cross"].unique())
        assert unique_vals.issubset({-1.0, 0.0, 1.0}), (
            f"macd_cross contains unexpected values: {unique_vals}"
        )

    def test_avg_volume_20d_positive(self):
        """Normalized average volume must be positive."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        assert (result["avg_volume_20d"] > 0).all(), "avg_volume_20d has non-positive values"

    def test_preserves_original_columns(self):
        """Output must include the original OHLCV columns (not just features)."""
        df = _make_ohlcv(250)
        result = compute_features(df)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns, f"Original column {col} missing from output"

    def test_empty_input_returns_empty(self):
        """Empty input DataFrame should return empty output."""
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        result = compute_features(df)
        assert result.empty

    def test_features_to_array_shape(self):
        """features_to_array() must return shape (N, N_FEATURES)."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        arr = features_to_array(result)
        assert arr.ndim == 2
        assert arr.shape[1] == N_FEATURES
        assert arr.shape[0] == len(result)

    def test_deterministic_output(self):
        """Same input must produce identical output (no randomness)."""
        df = _make_ohlcv(300, seed=7)
        r1 = compute_features(df)
        r2 = compute_features(df)
        pd.testing.assert_frame_equal(r1, r2)

    def test_unsorted_index_handled(self):
        """compute_features() should sort by index if not already sorted."""
        df = _make_ohlcv(300, seed=1)
        df_shuffled = df.sample(frac=1, random_state=99)  # shuffle rows
        result = compute_features(df_shuffled)
        # Should produce the same number of rows as sorted input
        result_sorted = compute_features(df)
        assert len(result) == len(result_sorted)

    def test_sufficient_rows_after_warmup(self):
        """With 300 rows, there should be some usable rows after 252-day warmup."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        # 252-day warmup for 52w rolling windows leaves ~48 rows from 300 input
        assert len(result) >= 30, f"Expected >=30 rows, got {len(result)}"

    def test_price_vs_ma50_range(self):
        """price_vs_ma50 should be a small decimal ratio, not a percent like ±100."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        # For a random walk around 100, should be within ±50% = ±0.50 ratio
        assert result["price_vs_ma50"].abs().max() < 1.0, (
            "price_vs_ma50 seems to be in percent (×100) rather than ratio"
        )

    # ── v1.4 feature tests ──────────────────────────────────────────────────

    def test_atr_14_pct_positive(self):
        """ATR is always positive (it's an average of absolute ranges)."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        assert (result["atr_14_pct"] > 0).all(), "atr_14_pct has non-positive values"
        # For a random walk with 1% daily vol, ATR/price should be < 10%
        assert result["atr_14_pct"].max() < 0.20, "atr_14_pct unreasonably large"

    def test_realized_vol_20d_positive(self):
        """Realized volatility must be positive."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        assert (result["realized_vol_20d"] > 0).all(), "realized_vol_20d has non-positive values"
        # Annualized vol should typically be < 100% for a random walk
        assert result["realized_vol_20d"].max() < 1.0, "realized_vol_20d unreasonably large"

    def test_volume_trend_positive(self):
        """Volume trend (SMA5/SMA20 ratio) must be positive."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        assert (result["volume_trend"] > 0).all(), "volume_trend has non-positive values"

    def test_volume_price_div_in_valid_range(self):
        """volume_price_div must be in {-1.0, 0.0, 1.0}."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        unique_vals = set(result["volume_price_div"].unique())
        assert unique_vals.issubset({-1.0, 0.0, 1.0}), (
            f"volume_price_div has unexpected values: {unique_vals}"
        )

    def test_rsi_slope_5d_reasonable(self):
        """RSI slope should be a small number (RSI range 0-100, divided by 5)."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        # Max RSI change over 5 days / 5 should be < 20 (100/5)
        assert result["rsi_slope_5d"].abs().max() < 20, "rsi_slope_5d unreasonably large"

    def test_new_features_no_nan(self):
        """All v1.4 features must have no NaN after warmup."""
        df = _make_ohlcv(300)
        result = compute_features(df)
        v14_features = [
            "price_accel", "ema_cross_8_21", "atr_14_pct", "realized_vol_20d",
            "volume_trend", "obv_slope_10d", "rsi_slope_5d", "volume_price_div",
        ]
        for col in v14_features:
            assert col in result.columns, f"Missing v1.4 column: {col}"
            nan_count = result[col].isnull().sum()
            assert nan_count == 0, f"v1.4 column {col} has {nan_count} NaN values"

    def test_n_features_matches_config(self):
        """The FEATURES list length must match N_FEATURES constant."""
        assert len(FEATURES) == N_FEATURES, (
            f"FEATURES list has {len(FEATURES)} entries but N_FEATURES={N_FEATURES}"
        )
