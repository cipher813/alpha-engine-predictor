"""Tests for data.label_generator.compute_triple_barrier_touch_labels (Task B).

Verifies the touch-order meta-label column is appended with the same vol-scaled
barriers as the alpha label, on sector-neutral residual log-returns.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.label_generator import compute_triple_barrier_touch_labels


def _price_df(returns: np.ndarray) -> pd.DataFrame:
    close = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2024-01-01", periods=len(returns), freq="B")
    return pd.DataFrame({"Close": close}, index=idx)


class TestComputeTouchLabels:
    def test_column_appended(self):
        df = _price_df(np.full(120, 0.005))
        out = compute_triple_barrier_touch_labels(df, forward_window=21)
        assert "triple_barrier_touch_21d" in out.columns
        assert len(out) == len(df)

    def test_uptrend_labels_one(self):
        # Steady absolute uptrend (no benchmark) → up barrier first → 1.0.
        df = _price_df(np.full(150, 0.01))
        out = compute_triple_barrier_touch_labels(
            df, forward_window=21, vol_window=20, min_periods=10
        )
        finite = out["triple_barrier_touch_21d"].dropna()
        assert len(finite) > 0
        assert (finite == 1.0).all()

    def test_downtrend_labels_zero(self):
        df = _price_df(np.full(150, -0.01))
        out = compute_triple_barrier_touch_labels(
            df, forward_window=21, vol_window=20, min_periods=10
        )
        finite = out["triple_barrier_touch_21d"].dropna()
        assert len(finite) > 0
        assert (finite == 0.0).all()

    def test_front_history_nan(self):
        # First `min_periods` rows have no vol estimate → NaN barrier → NaN.
        df = _price_df(np.full(150, 0.01))
        out = compute_triple_barrier_touch_labels(
            df, forward_window=21, vol_window=20, min_periods=10
        )
        assert out["triple_barrier_touch_21d"].iloc[:5].isna().all()

    def test_empty_df(self):
        out = compute_triple_barrier_touch_labels(
            pd.DataFrame({"Close": pd.Series(dtype=float)}), forward_window=21
        )
        assert "triple_barrier_touch_21d" in out.columns
        assert len(out) == 0

    def test_benchmark_neutralization_runs(self):
        # With a benchmark, labels are on residual returns — just verify it
        # runs and produces in-range labels.
        df = _price_df(np.full(150, 0.01))
        bench = pd.Series(
            100.0 * np.exp(np.cumsum(np.full(150, 0.008))), index=df.index
        )
        out = compute_triple_barrier_touch_labels(
            df, benchmark_returns=bench, forward_window=21, min_periods=10
        )
        finite = out["triple_barrier_touch_21d"].dropna()
        assert finite.isin([0.0, 1.0]).all()
