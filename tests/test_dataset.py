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
