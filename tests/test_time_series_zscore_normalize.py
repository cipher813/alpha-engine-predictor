"""Tests for ``data.dataset.time_series_zscore_normalize``.

Stage 1 substrate of the regime-conditioning rebuild — point-in-time
rolling z-score for time-series-only features (macros). Cross-sectional
rank-norm degenerates these to 0.5 for every row because macros are
constant across tickers on a given date; time-series z-score preserves
the regime signal.

Critical correctness properties to verify:
- No look-ahead leakage: z-score at date d uses ONLY data prior to d.
- Cross-ticker consistency: every row on date d gets the same z-score
  (since macros are constant within a date).
- min_periods bootstrap: dates with insufficient trailing history return
  NaN, not z-scores against partial samples.
- Output dtype: float32 (matches GBM input expectation).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import time_series_zscore_normalize


def _build_synthetic_macro_array(
    n_dates: int,
    n_tickers_per_date: int,
    seed: int = 0,
):
    """Build (N, 1) macro array + dates list mimicking real training shape.

    Each date has the same value across all tickers (macros are constant
    cross-sectionally). Dates are unique and sorted ascending.
    """
    rng = np.random.default_rng(seed)
    base_dates = pd.date_range("2018-01-01", periods=n_dates, freq="B")
    # Per-date macro value — random walk so values change over time but
    # have a meaningful rolling mean/std.
    per_date_value = np.cumsum(rng.normal(0, 1.0, n_dates)) * 0.1 + 20.0
    # Replicate each date n_tickers_per_date times.
    dates = []
    values = []
    for i, d in enumerate(base_dates):
        for _ in range(n_tickers_per_date):
            dates.append(d)
            values.append(per_date_value[i])
    X = np.array(values, dtype=np.float64).reshape(-1, 1)
    return X, dates, per_date_value, base_dates


# ── Cross-ticker consistency ────────────────────────────────────────────


class TestCrossTickerConsistency:

    def test_all_rows_on_same_date_get_same_zscore(self):
        # 100 dates × 5 tickers — every row on date d should get the same
        # z-score because macros are constant cross-sectionally.
        X, dates, _, _ = _build_synthetic_macro_array(n_dates=100, n_tickers_per_date=5)
        z = time_series_zscore_normalize(X, dates, window=20, min_periods=5)

        # Group rows by date; every group should have a single unique
        # z-score (or NaN, which we treat as a single value).
        date_arr = np.array(dates)
        unique_dates = np.unique(date_arr)
        for d in unique_dates:
            mask = date_arr == d
            vals = z[mask, 0]
            # NaN-aware uniqueness: all NaN OR all the same finite value.
            if np.isnan(vals).all():
                continue
            finite_vals = vals[~np.isnan(vals)]
            assert np.allclose(finite_vals, finite_vals[0]), (
                f"Date {d}: z-scores diverge across tickers: {vals}"
            )


# ── Point-in-time correctness (no look-ahead) ──────────────────────────


class TestPointInTime:

    def test_zscore_at_date_uses_only_past_data(self):
        # Build a series with a clear regime shift in the middle. If the
        # z-score uses past-only data, the post-shift values should
        # appear as positive z-scores (relative to past-low regime).
        n_dates = 200
        per_date = np.array(
            [10.0] * 100 + [50.0] * 100,  # step change at index 100
            dtype=np.float64,
        )
        dates = pd.date_range("2018-01-01", periods=n_dates, freq="B").tolist()
        X = per_date.reshape(-1, 1)

        z = time_series_zscore_normalize(X, dates, window=50, min_periods=20)

        # Pre-shift z-scores (after window bootstrap): regime is steady at
        # 10, std is zero, z-score is NaN or undefined.
        early_window = z[20:99, 0]  # past warmup, before shift
        # After warmup, with constant value 10 and std=0, z = (10 - 10) / 0
        # which is NaN (or inf); both should be filterable.

        # Post-shift z-scores: at index 100, the rolling window includes
        # all 10's in the past → mean=10, std=0; z is NaN/inf.
        # As the window moves forward and includes 50's, std becomes
        # finite and post-shift values get HIGH positive z-scores.
        late = z[150, 0]  # well past the shift
        # The z-score at index 150 should be positive (50 is high relative
        # to a window that includes a mix of 10's and 50's, but the mean
        # is around 35 and std is large enough to produce a positive z).
        if not np.isnan(late):
            assert late > 0, f"expected positive z-score post-regime-shift, got {late}"

    def test_no_leakage_from_future(self):
        # Constructive test: feed a constant series followed by a single
        # huge spike. The pre-spike rows must NOT see the spike.
        n_dates = 100
        per_date = np.full(n_dates, 5.0)
        per_date[80] = 1000.0  # spike at index 80
        dates = pd.date_range("2018-01-01", periods=n_dates, freq="B").tolist()
        X = per_date.reshape(-1, 1)

        z = time_series_zscore_normalize(X, dates, window=30, min_periods=10)

        # Pre-spike (index 30-79): constant 5.0. Rolling mean=5, std=0,
        # z is NaN (divide by zero). NOT contaminated by future spike.
        pre_spike = z[30:80, 0]
        # All pre-spike z-scores must be NaN (constant input, std=0).
        assert np.isnan(pre_spike).all(), (
            f"Pre-spike z-scores must not see future spike: {pre_spike}"
        )

    def test_at_index_0_returns_nan(self):
        # No trailing window for the very first date.
        X = np.array([[1.0], [2.0], [3.0]])
        dates = pd.date_range("2018-01-01", periods=3, freq="B").tolist()
        z = time_series_zscore_normalize(X, dates, window=10, min_periods=2)
        assert np.isnan(z[0, 0])


# ── min_periods bootstrap ──────────────────────────────────────────────


class TestMinPeriodsBootstrap:

    def test_dates_below_min_periods_return_nan(self):
        n_dates = 30
        rng = np.random.default_rng(0)
        per_date = rng.normal(20, 5, n_dates).astype(np.float64)
        dates = pd.date_range("2018-01-01", periods=n_dates, freq="B").tolist()
        X = per_date.reshape(-1, 1)

        z = time_series_zscore_normalize(X, dates, window=50, min_periods=20)

        # First 20 dates (indices 0-19): min_periods=20 not yet met.
        # At index 0, no past data; index 19, 19 past data points.
        # z must be NaN until index 20+ (when shift(1) + window-window
        # has min_periods satisfied).
        assert np.isnan(z[:20, 0]).all()
        # By index 25+ the window has 25 past values (>= min_periods 20).
        # z should be finite.
        assert not np.isnan(z[25, 0])


# ── NaN handling ───────────────────────────────────────────────────────


class TestNaNHandling:

    def test_nan_in_input_propagates_at_that_date(self):
        # If a macro is NaN at some date, the z-score at that date is
        # NaN. Future dates can still compute z-scores using the past
        # non-NaN window (rolling skips NaN by default in pandas).
        n_dates = 100
        rng = np.random.default_rng(0)
        per_date = rng.normal(20, 5, n_dates).astype(np.float64)
        per_date[50] = np.nan  # NaN macro at one date
        dates = pd.date_range("2018-01-01", periods=n_dates, freq="B").tolist()
        X = per_date.reshape(-1, 1)

        z = time_series_zscore_normalize(X, dates, window=30, min_periods=10)

        # At index 50 (NaN input), z must be NaN.
        assert np.isnan(z[50, 0])
        # At index 80 (well past the NaN), z should be finite (pandas
        # rolling skips NaN by default).
        assert not np.isnan(z[80, 0])


# ── Output shape + dtype ────────────────────────────────────────────────


class TestOutputContract:

    def test_output_shape_matches_input(self):
        X, dates, _, _ = _build_synthetic_macro_array(n_dates=50, n_tickers_per_date=3)
        z = time_series_zscore_normalize(X, dates, window=20, min_periods=5)
        assert z.shape == X.shape

    def test_output_dtype_is_float32(self):
        X, dates, _, _ = _build_synthetic_macro_array(n_dates=50, n_tickers_per_date=3)
        z = time_series_zscore_normalize(X, dates, window=20, min_periods=5)
        assert z.dtype == np.float32

    def test_empty_input_returns_empty(self):
        X = np.zeros((0, 1))
        z = time_series_zscore_normalize(X, [], window=20, min_periods=5)
        assert z.shape == (0, 1)
        assert z.dtype == np.float32

    def test_dates_length_mismatch_raises(self):
        X = np.zeros((5, 1))
        bad_dates = [pd.Timestamp("2018-01-01")] * 3  # length 3, X has 5 rows
        try:
            time_series_zscore_normalize(X, bad_dates)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "does not match" in str(e)

    def test_multi_feature_array(self):
        # 2-feature array — both columns should be z-scored independently.
        n_dates = 100
        rng = np.random.default_rng(0)
        f1 = rng.normal(20, 5, n_dates).astype(np.float64)
        f2 = rng.normal(0, 0.01, n_dates).astype(np.float64)
        per_date = np.column_stack([f1, f2])
        dates = pd.date_range("2018-01-01", periods=n_dates, freq="B").tolist()
        z = time_series_zscore_normalize(per_date, dates, window=30, min_periods=10)

        # Both features finite after warmup
        assert not np.isnan(z[50, 0])
        assert not np.isnan(z[50, 1])
        # Independent z-scores: f2 has tiny variance, so its z-score
        # magnitude can differ from f1's.


# ── Cross-feature: integrates cleanly with cross_sectional_rank_normalize ─


class TestPerFeatureNormalizationDispatch:
    """Validates the institutional pattern: rank-norm for ticker features,
    z-score for macros. Separate calls compose; results don't conflict."""

    def test_separate_calls_compose(self):
        # 50 dates × 5 tickers. Build a (N, 2) array with column 0 a
        # ticker-varying feature and column 1 a date-constant macro.
        from data.dataset import cross_sectional_rank_normalize

        n_dates, n_tickers = 50, 5
        rng = np.random.default_rng(0)
        rows = []
        dates_list = []
        base_dates = pd.date_range("2018-01-01", periods=n_dates, freq="B")
        macro_per_date = rng.normal(20, 5, n_dates)
        for i, d in enumerate(base_dates):
            for t in range(n_tickers):
                rows.append([rng.normal(), macro_per_date[i]])
                dates_list.append(d)
        X = np.array(rows, dtype=np.float64)

        # Apply rank-norm to ticker column, z-score to macro column.
        X_ticker = cross_sectional_rank_normalize(
            X[:, [0]].astype(np.float64), dates_list,
        )
        X_macro = time_series_zscore_normalize(
            X[:, [1]], dates_list, window=20, min_periods=5,
        )

        # Per-date rank-norm: values in [0, 1]
        assert (X_ticker >= 0).all() and (X_ticker <= 1).all()
        # Per-date z-score: cross-ticker consistent (all rows on same
        # date have same macro z-score). Use the property already tested.
        date_arr = np.array(dates_list)
        for d in np.unique(date_arr):
            mask = date_arr == d
            vals = X_macro[mask, 0]
            if not np.isnan(vals).all():
                finite = vals[~np.isnan(vals)]
                assert np.allclose(finite, finite[0])
