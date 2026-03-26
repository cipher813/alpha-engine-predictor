"""
tests/test_feature_store.py — Unit tests for the feature store.

Tests registry generation, writer/reader round-trip, and group splitting.
No S3 or network calls — uses moto for S3 mocking.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_store.registry import (
    CATALOG,
    GROUPS,
    FeatureEntry,
    generate_registry_json,
    get_feature,
    get_group_features,
)
from feature_store.writer import write_feature_snapshot
from feature_store.reader import read_feature_snapshot, read_feature_range


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_df(n_tickers: int = 5, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic feature DataFrame matching the registry schema."""
    rng = np.random.default_rng(seed)
    tickers = [f"TICK{i}" for i in range(n_tickers)]

    data = {"ticker": tickers}
    for entry in CATALOG:
        data[entry.name] = rng.normal(0, 1, n_tickers).astype(np.float32)

    return pd.DataFrame(data)


def _make_mock_s3():
    """Create a mock S3 client that stores objects in a dict."""
    store = {}

    mock = MagicMock()

    def put_object(Bucket, Key, Body, **kwargs):
        store[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(Bucket, Key):
        if (Bucket, Key) not in store:
            error = MagicMock()
            error.response = {"Error": {"Code": "NoSuchKey"}}
            raise mock.exceptions.NoSuchKey(error)
        body = MagicMock()
        body.read.return_value = store[(Bucket, Key)]
        return {"Body": body}

    # Wire up NoSuchKey exception class
    class NoSuchKey(Exception):
        pass

    mock.exceptions = MagicMock()
    mock.exceptions.NoSuchKey = NoSuchKey

    # Override get_object to raise our custom exception
    def get_object_with_exc(Bucket, Key):
        if (Bucket, Key) not in store:
            raise NoSuchKey(f"Key not found: {Key}")
        body = MagicMock()
        body.read.return_value = store[(Bucket, Key)]
        return {"Body": body}

    mock.put_object = put_object
    mock.get_object = get_object_with_exc
    mock._store = store  # expose for inspection
    return mock


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_catalog_not_empty(self):
        assert len(CATALOG) >= 49

    def test_all_features_have_required_fields(self):
        for f in CATALOG:
            assert f.name, f"Feature missing name: {f}"
            assert f.group in ("technical", "macro", "interaction", "alternative", "fundamental")
            assert f.description
            assert f.dtype == "float32"

    def test_no_duplicate_names(self):
        names = [f.name for f in CATALOG]
        assert len(names) == len(set(names)), f"Duplicate feature names: {set(n for n in names if names.count(n) > 1)}"

    def test_groups_cover_all_features(self):
        all_from_groups = set()
        for features in GROUPS.values():
            all_from_groups.update(features)
        all_from_catalog = {f.name for f in CATALOG}
        assert all_from_groups == all_from_catalog

    def test_group_counts(self):
        assert len(GROUPS["technical"]) == 24
        assert len(GROUPS["macro"]) == 5
        assert len(GROUPS["interaction"]) == 5
        assert len(GROUPS["alternative"]) == 7
        assert len(GROUPS["fundamental"]) == 8

    def test_macro_features_not_per_ticker(self):
        for name in GROUPS["macro"]:
            f = get_feature(name)
            assert not f.per_ticker, f"Macro feature {name} should have per_ticker=False"

    def test_get_feature_lookup(self):
        f = get_feature("rsi_14")
        assert f.group == "technical"
        assert f.per_ticker is True

    def test_get_group_features(self):
        tech = get_group_features("technical")
        assert "rsi_14" in tech
        assert "vix_level" not in tech

    def test_generate_registry_json(self):
        raw = generate_registry_json()
        data = json.loads(raw)
        assert "features" in data
        assert len(data["features"]) == len(CATALOG)
        # Verify structure of first entry
        first = data["features"][0]
        assert "name" in first
        assert "group" in first
        assert "description" in first
        assert "per_ticker" in first


# ---------------------------------------------------------------------------
# Writer tests
# ---------------------------------------------------------------------------

class TestWriter:
    def test_write_splits_by_group(self):
        df = _make_feature_df(n_tickers=3)
        s3 = _make_mock_s3()

        result = write_feature_snapshot("2026-03-26", df, "test-bucket", s3_client=s3)

        assert "technical" in result
        assert "macro" in result
        assert "interaction" in result
        assert "alternative" in result
        assert "fundamental" in result

    def test_write_technical_has_ticker_column(self):
        df = _make_feature_df(n_tickers=3)
        s3 = _make_mock_s3()

        write_feature_snapshot("2026-03-26", df, "test-bucket", s3_client=s3)

        key = ("test-bucket", "features/2026-03-26/technical.parquet")
        assert key in s3._store
        tech_df = pd.read_parquet(pd.io.common.BytesIO(s3._store[key]))
        assert "ticker" in tech_df.columns
        assert "date" in tech_df.columns
        assert len(tech_df) == 3

    def test_write_macro_has_one_row(self):
        df = _make_feature_df(n_tickers=10)
        s3 = _make_mock_s3()

        write_feature_snapshot("2026-03-26", df, "test-bucket", s3_client=s3)

        key = ("test-bucket", "features/2026-03-26/macro.parquet")
        macro_df = pd.read_parquet(pd.io.common.BytesIO(s3._store[key]))
        assert len(macro_df) == 1  # macro is one row, not per-ticker
        assert "ticker" not in macro_df.columns
        assert "date" in macro_df.columns

    def test_write_idempotent(self):
        df = _make_feature_df(n_tickers=3)
        s3 = _make_mock_s3()

        r1 = write_feature_snapshot("2026-03-26", df, "test-bucket", s3_client=s3)
        r2 = write_feature_snapshot("2026-03-26", df, "test-bucket", s3_client=s3)
        assert r1 == r2

    def test_write_skips_missing_columns(self):
        # DataFrame with only technical features
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "rsi_14": [50.0, 60.0],
            "macd_cross": [1.0, -1.0],
        })
        s3 = _make_mock_s3()

        result = write_feature_snapshot("2026-03-26", df, "test-bucket", s3_client=s3)

        assert "technical" in result
        assert "fundamental" not in result  # no fundamental columns present


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------

class TestReader:
    def test_round_trip(self):
        """Write features then read them back — values should match."""
        df = _make_feature_df(n_tickers=5)
        s3 = _make_mock_s3()

        write_feature_snapshot("2026-03-26", df, "test-bucket", s3_client=s3)
        tech_df = read_feature_snapshot("2026-03-26", "technical", "test-bucket", s3_client=s3)

        assert tech_df is not None
        assert len(tech_df) == 5
        assert "ticker" in tech_df.columns
        # Verify a feature value round-trips correctly
        original_rsi = df.set_index("ticker")["rsi_14"]
        read_rsi = tech_df.set_index("ticker")["rsi_14"]
        pd.testing.assert_series_equal(original_rsi, read_rsi, check_names=False)

    def test_read_nonexistent_returns_none(self):
        s3 = _make_mock_s3()
        result = read_feature_snapshot("2026-01-01", "technical", "test-bucket", s3_client=s3)
        assert result is None

    def test_read_range(self):
        df = _make_feature_df(n_tickers=3)
        s3 = _make_mock_s3()

        # Write 3 days
        write_feature_snapshot("2026-03-24", df, "test-bucket", s3_client=s3)
        write_feature_snapshot("2026-03-25", df, "test-bucket", s3_client=s3)
        write_feature_snapshot("2026-03-26", df, "test-bucket", s3_client=s3)

        result = read_feature_range("2026-03-24", "2026-03-26", "technical", "test-bucket", s3_client=s3)
        assert len(result) == 9  # 3 tickers × 3 days

    def test_read_range_skips_missing_dates(self):
        df = _make_feature_df(n_tickers=2)
        s3 = _make_mock_s3()

        # Only write for 2026-03-24 (skip 25, 26)
        write_feature_snapshot("2026-03-24", df, "test-bucket", s3_client=s3)

        result = read_feature_range("2026-03-24", "2026-03-26", "technical", "test-bucket", s3_client=s3)
        assert len(result) == 2  # only the one date that exists
