"""Tests for ``training.meta_trainer._build_macro_array``.

Stage 1b helper of the regime-conditioning rebuild — broadcasts per-date
macro values from ``regime_features_df`` to every per-(date, ticker)
training row. Output feeds ``time_series_zscore_normalize`` (Stage 1a)
and then concatenates with the rank-normed volatility features for the
parallel macro-augmented volatility GBM training.

Critical correctness properties:
- Cross-ticker consistency: every row on date d gets identical macro
  values (macros are constant cross-sectionally).
- Missing-date handling: rows whose date isn't in the macro DataFrame
  return NaN (LightGBM handles NaN natively).
- Column-order preservation: the output columns match the
  ``macro_features`` argument order exactly.
- Empty regime_features_df: all-NaN output (training cycle proceeds
  with macro signal disabled rather than failing).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.meta_trainer import _build_macro_array


def _build_synthetic_regime_df(
    dates: list,
    feature_names: list[str],
    seed: int = 0,
) -> pd.DataFrame:
    """Build a date-indexed regime_features_df mimicking the production shape."""
    rng = np.random.default_rng(seed)
    data = {
        name: rng.normal(0, 1, len(dates))
        for name in feature_names
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates))


class TestBuildMacroArray:

    def test_cross_ticker_consistency(self):
        # 50 dates × 5 tickers per date — every row on date d should
        # carry identical macro values.
        unique_dates = pd.date_range("2018-01-01", periods=50, freq="B")
        macro_features = ["spy_20d_return", "vix_level", "yield_curve_slope"]
        regime_df = _build_synthetic_regime_df(unique_dates, macro_features)

        all_dates = []
        for d in unique_dates:
            for _ in range(5):
                all_dates.append(d)

        X = _build_macro_array(all_dates, regime_df, macro_features)
        assert X.shape == (250, 3)

        # For each unique date, the 5 corresponding rows must be identical.
        date_arr = np.array(all_dates)
        for d in unique_dates:
            mask = date_arr == d
            rows = X[mask]
            assert (rows == rows[0]).all(), (
                f"Date {d}: rows differ across tickers: {rows}"
            )

    def test_missing_date_returns_nan(self):
        # regime_df has 30 dates but all_dates includes a 31st that's
        # missing — that row should be NaN.
        unique_dates = pd.date_range("2018-01-01", periods=30, freq="B")
        missing_date = pd.Timestamp("2030-01-01")
        macro_features = ["spy_20d_return", "vix_level"]
        regime_df = _build_synthetic_regime_df(unique_dates, macro_features)

        all_dates = list(unique_dates) + [missing_date]
        X = _build_macro_array(all_dates, regime_df, macro_features)

        assert X.shape == (31, 2)
        # The missing-date row should be all NaN.
        assert np.isnan(X[-1]).all()
        # Other rows should be finite.
        assert not np.isnan(X[:30]).any()

    def test_column_order_matches_argument(self):
        unique_dates = pd.date_range("2018-01-01", periods=10, freq="B")
        # Build with one column order
        df_features = ["a", "b", "c"]
        regime_df = _build_synthetic_regime_df(unique_dates, df_features)

        # Request a different order
        requested_order = ["c", "a", "b"]
        all_dates = list(unique_dates)
        X = _build_macro_array(all_dates, regime_df, requested_order)

        # Column 0 of X should match df["c"]
        assert np.allclose(X[:, 0], regime_df["c"].to_numpy())
        # Column 1 of X should match df["a"]
        assert np.allclose(X[:, 1], regime_df["a"].to_numpy())
        # Column 2 of X should match df["b"]
        assert np.allclose(X[:, 2], regime_df["b"].to_numpy())

    def test_empty_regime_df_returns_all_nan(self):
        # Production safety: if upstream macro feature build fails and
        # regime_features_df comes back empty, the training cycle proceeds
        # with NaN macros (LightGBM tolerant) rather than crashing.
        all_dates = list(pd.date_range("2018-01-01", periods=10, freq="B"))
        macro_features = ["spy_20d_return", "vix_level"]
        empty_df = pd.DataFrame()  # no columns, no index
        X = _build_macro_array(all_dates, empty_df, macro_features)
        assert X.shape == (10, 2)
        assert np.isnan(X).all()

    def test_output_dtype_is_float64(self):
        unique_dates = pd.date_range("2018-01-01", periods=10, freq="B")
        macro_features = ["spy_20d_return"]
        regime_df = _build_synthetic_regime_df(unique_dates, macro_features)
        all_dates = list(unique_dates)
        X = _build_macro_array(all_dates, regime_df, macro_features)
        # float64 (not float32) so the downstream z-score normalization
        # can compute high-precision rolling stats. Caller can downcast.
        assert X.dtype == np.float64

    def test_subset_macro_features(self):
        # regime_df has 6 columns; caller only requests 2.
        unique_dates = pd.date_range("2018-01-01", periods=20, freq="B")
        df_features = [
            "spy_20d_return", "spy_20d_vol", "vix_level",
            "vix_term_slope", "yield_curve_slope", "market_breadth",
        ]
        regime_df = _build_synthetic_regime_df(unique_dates, df_features)

        # Request only 2 of the 6 columns
        requested = ["vix_level", "yield_curve_slope"]
        all_dates = list(unique_dates)
        X = _build_macro_array(all_dates, regime_df, requested)
        assert X.shape == (20, 2)
        assert np.allclose(X[:, 0], regime_df["vix_level"].to_numpy())
        assert np.allclose(X[:, 1], regime_df["yield_curve_slope"].to_numpy())

    def test_dates_sorted_or_not_handled_correctly(self):
        # all_dates may not be perfectly sorted (some test fixtures shuffle).
        # The function should still produce per-row correct broadcasts.
        unique_dates = pd.date_range("2018-01-01", periods=10, freq="B")
        macro_features = ["spy_20d_return"]
        regime_df = _build_synthetic_regime_df(unique_dates, macro_features)

        # Reverse the date order
        reversed_dates = list(reversed(unique_dates))
        X = _build_macro_array(reversed_dates, regime_df, macro_features)

        # Each row's value should match the regime_df value for that date
        for i, d in enumerate(reversed_dates):
            expected = float(regime_df.at[d, "spy_20d_return"])
            assert X[i, 0] == expected
