"""Tests for data/dataset.py — array building and normalization."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.dataset import (
    _load_ticker_parquet,
    cross_sectional_rank_normalize,
    load_norm_stats,
)


class TestLoadTickerParquet:
    def test_loads_valid_parquet(self):
        df = pd.DataFrame({
            "Open": [150.0, 151.0],
            "High": [155.0, 156.0],
            "Low": [148.0, 149.0],
            "Close": [152.0, 153.0],
            "Volume": [1000000, 1100000],
        }, index=pd.to_datetime(["2026-04-07", "2026-04-08"]))

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name)
            result = _load_ticker_parquet(Path(f.name))
            assert len(result) == 2
            assert "Close" in result.columns

    def test_returns_empty_on_missing(self):
        result = _load_ticker_parquet(Path("/nonexistent/AAPL.parquet"))
        assert result.empty

    def test_sorts_by_date(self):
        df = pd.DataFrame({
            "Close": [153.0, 150.0, 152.0],
        }, index=pd.to_datetime(["2026-04-08", "2026-04-06", "2026-04-07"]))

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name)
            result = _load_ticker_parquet(Path(f.name))
            assert result.index[0] <= result.index[-1]


class TestCrossSectionalRankNormalize:
    def test_basic_normalization(self):
        # 3 dates, 4 tickers each = 12 rows
        dates = ["2026-04-06"] * 4 + ["2026-04-07"] * 4 + ["2026-04-08"] * 4
        features = np.random.randn(12, 3)
        result = cross_sectional_rank_normalize(features, dates)
        assert result.shape == (12, 3)
        # Rank-normalized values should be in [-1, 1] approximately
        assert result.min() >= -2.0
        assert result.max() <= 2.0

    def test_preserves_shape(self):
        dates = ["2026-04-08"] * 10
        features = np.random.randn(10, 5)
        result = cross_sectional_rank_normalize(features, dates)
        assert result.shape == (10, 5)

    def test_single_date(self):
        dates = ["2026-04-08"] * 5
        features = np.random.randn(5, 3)
        result = cross_sectional_rank_normalize(features, dates)
        assert result.shape == (5, 3)

    def test_constant_feature_handled(self):
        dates = ["2026-04-08"] * 5
        features = np.ones((5, 2))  # constant — std=0
        result = cross_sectional_rank_normalize(features, dates)
        assert not np.isnan(result).any()

    def test_nan_input_gets_neutral_rank_05(self):
        """Regression test for the 2026-05-09 momentum subsample IC bug.

        Pre-fix: numpy ``argsort`` placed NaN at the end, so NaN inputs
        got rank=n-1 → percentile 1.0 (top of the cross-section). The
        GBM then learned a spurious "rank=1.0 cluster" pattern across
        every NaN feature that short-history tickers shared. Subsample
        IC gate caught this as -0.17 component_ic vs +0.32 baseline_ic.

        Post-fix: NaN entries get 0.5 (cross-sectional median rank),
        leaving them on the neutral line where they belong. The non-NaN
        rows still rank-normalize to (0, 1) over the non-NaN subset.
        """
        # 1 date, 5 tickers: third ticker has NaN in feature 0
        dates = ["2026-04-08"] * 5
        features = np.array(
            [[1.0, 1.0],
             [2.0, 2.0],
             [np.nan, 3.0],   # NaN row — must get 0.5 in feature 0
             [4.0, 4.0],
             [5.0, 5.0]],
            dtype=np.float32,
        )
        result = cross_sectional_rank_normalize(features, dates)
        # NaN row in feature 0 → 0.5 (neutral median rank), NOT 1.0 (top)
        assert result[2, 0] == 0.5, (
            f"NaN should rank to 0.5 (median), got {result[2, 0]} — "
            "the legacy argsort path mapped NaN to rank=n-1 / (n-1) = 1.0"
        )
        # Feature 1 has no NaNs — must rank-normalize to (0, 1) over the
        # full 5-ticker cross-section (verifying the non-NaN path is
        # intact).
        assert result[0, 1] == 0.0  # smallest
        assert result[4, 1] == 1.0  # largest

        # Feature 0 non-NaN ranks: 1.0→0, 2.0→1, 4.0→2, 5.0→3 over 4
        # non-NaN rows → scaled by (4-1)=3 → [0, 1/3, 2/3, 1]
        assert result[0, 0] == 0.0
        assert abs(result[1, 0] - 1.0 / 3.0) < 1e-6
        assert abs(result[3, 0] - 2.0 / 3.0) < 1e-6
        assert result[4, 0] == 1.0

    def test_all_nan_feature_assigns_05_to_everyone(self):
        """If every value for a feature on a given date is NaN (e.g. an
        alt-data feature missing universe-wide), every row gets 0.5 —
        no rank computable, neutral assignment is the only sensible
        behavior."""
        dates = ["2026-04-08"] * 4
        features = np.array(
            [[np.nan, 1.0],
             [np.nan, 2.0],
             [np.nan, 3.0],
             [np.nan, 4.0]],
            dtype=np.float32,
        )
        result = cross_sectional_rank_normalize(features, dates)
        assert (result[:, 0] == 0.5).all()
        # Other feature unaffected
        assert result[0, 1] == 0.0
        assert result[3, 1] == 1.0

    def test_single_non_nan_row_neutral(self):
        """Edge case: only one non-NaN value for a feature on a date.
        Cannot rank a 1-element set meaningfully — assign 0.5 to all
        rows (NaN and the lone non-NaN alike). Better than crashing or
        forcing the lone row to rank=0."""
        dates = ["2026-04-08"] * 4
        features = np.array(
            [[np.nan, 1.0],
             [np.nan, 2.0],
             [3.0,    3.0],
             [np.nan, 4.0]],
            dtype=np.float32,
        )
        result = cross_sectional_rank_normalize(features, dates)
        assert (result[:, 0] == 0.5).all()

    def test_existing_non_nan_behavior_unchanged(self):
        """Ensure the NaN-aware refactor doesn't drift the all-non-NaN
        path. Cross-sectional ranks must still be (0, 1) percentile
        with deterministic ordering."""
        dates = ["2026-04-08"] * 5
        features = np.array(
            [[10.0],
             [20.0],
             [30.0],
             [40.0],
             [50.0]],
            dtype=np.float32,
        )
        result = cross_sectional_rank_normalize(features, dates)
        expected = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-6)


class TestLoadNormStats:
    def test_loads_valid_json(self):
        stats = {
            "mean": [0.1, 0.2, 0.3],
            "std": [1.0, 1.1, 1.2],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(stats, f)
            f.flush()
            mean, std = load_norm_stats(f.name)
            assert len(mean) == 3
            assert len(std) == 3
            assert mean[0] == pytest.approx(0.1)

    def test_returns_arrays(self):
        stats = {"mean": [0.0, 0.0], "std": [1.0, 1.0]}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(stats, f)
            f.flush()
            mean, std = load_norm_stats(f.name)
            assert isinstance(mean, np.ndarray)
            assert isinstance(std, np.ndarray)
