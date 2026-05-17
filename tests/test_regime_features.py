"""Tests for regime/features.py — macro feature fetcher."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from regime.features import (
    DEFAULT_PRICE_CACHE_PREFIX,
    SOURCE_TICKERS,
    current_features_from_history,
    fetch_macro_feature_history,
)


pytest.importorskip("pyarrow")


class _FakeS3:
    """In-memory S3 with put/get of parquet bodies keyed by (bucket, key)."""

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}

    def put_parquet(self, bucket: str, key: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=True)
        self._objects[(bucket, key)] = buf.getvalue()

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self._objects:
            raise KeyError(f"no object at {Bucket}/{Key}")
        return {"Body": io.BytesIO(self._objects[(Bucket, Key)])}


def _make_close_parquet(start: str, n_days: int, base: float, drift: float, seed: int) -> pd.DataFrame:
    """Synthesize a per-ticker price_cache parquet with a Close column."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_days, freq="B")
    closes = base + drift * np.arange(n_days) + rng.normal(0, base * 0.005, n_days)
    return pd.DataFrame({"Close": closes}, index=dates)


def _seed_full_price_cache(s3: _FakeS3, bucket: str, prefix: str) -> None:
    """Populate the FakeS3 with all 6 source tickers worth of synthetic
    daily Close data spanning ~3 years.

    Unit conventions match what FRED publishes (and what the data side
    stores at face value):
      - HYOAS: percent (FRED BAMLH0A0HYM2 publishes e.g. 4.10 meaning 410 bps)
      - TNX / TWO: percent (4.0 = 4.0% annualized yield)
      - VIX / VIX3M: index points
      - SPY: dollars
    """
    cfg = {
        "SPY":   (400.0, 0.05, 1),
        "VIX":   (18.0,  0.0,  2),
        "VIX3M": (19.0,  0.0,  3),
        "TNX":   (4.0,   0.0,  4),   # percent
        "TWO":   (3.5,   0.0,  5),   # percent
        "HYOAS": (4.0,   0.0,  6),   # percent → × 100 = 400 bps
    }
    for ticker, (base, drift, seed) in cfg.items():
        df = _make_close_parquet("2024-01-01", n_days=750, base=base, drift=drift, seed=seed)
        s3.put_parquet(bucket, f"{prefix}{ticker}.parquet", df)


def test_fetch_returns_weekly_dataframe_with_required_columns() -> None:
    s3 = _FakeS3()
    _seed_full_price_cache(s3, "test-bucket", DEFAULT_PRICE_CACHE_PREFIX)
    df = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket")
    expected_cols = {
        "spy_20d_return", "vix_level", "vix_term_slope",
        "hy_oas_bps", "yield_curve_slope", "market_breadth",
    }
    assert expected_cols <= set(df.columns)
    # Weekly cadence — Friday-anchored
    assert all(d.dayofweek == 4 for d in df.index)


def test_fetch_hy_oas_converted_to_basis_points() -> None:
    """FRED publishes HYOAS in percent units (4.0 = 4% spread).
    Fetcher multiplies by 100 to land in basis points (4% → 400 bps).
    Synthetic data sets HYOAS Close = 4.0; expected output mean ~400."""
    s3 = _FakeS3()
    _seed_full_price_cache(s3, "test-bucket", DEFAULT_PRICE_CACHE_PREFIX)
    df = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket")
    assert 350 < df["hy_oas_bps"].mean() < 450, (
        f"expected ~400 bps (4.0% × 100), got {df['hy_oas_bps'].mean():.1f}"
    )


def test_fetch_yield_curve_slope_is_difference() -> None:
    s3 = _FakeS3()
    _seed_full_price_cache(s3, "test-bucket", DEFAULT_PRICE_CACHE_PREFIX)
    df = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket")
    # TNX synthetic mean ≈ 4.0, TWO ≈ 3.5 → slope ≈ 0.5
    assert 0.3 < df["yield_curve_slope"].mean() < 0.7


def test_fetch_vix_term_slope_is_difference() -> None:
    s3 = _FakeS3()
    _seed_full_price_cache(s3, "test-bucket", DEFAULT_PRICE_CACHE_PREFIX)
    df = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket")
    # VIX3M ≈ 19, VIX ≈ 18 → term slope ≈ 1.0
    assert 0.5 < df["vix_term_slope"].mean() < 1.5


def test_fetch_market_breadth_is_nan_pending_source() -> None:
    """Breadth source lands in a follow-up PR; column must be NaN today
    so downstream consumers see absence explicitly, not a spurious 0."""
    s3 = _FakeS3()
    _seed_full_price_cache(s3, "test-bucket", DEFAULT_PRICE_CACHE_PREFIX)
    df = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket")
    assert df["market_breadth"].isna().all()


def test_fetch_returns_empty_when_no_parquets_exist() -> None:
    s3 = _FakeS3()  # no objects seeded
    df = fetch_macro_feature_history(s3_client=s3, bucket="empty-bucket")
    assert df.empty


def test_fetch_resilient_to_missing_single_ticker() -> None:
    """Missing one of the 6 tickers must not crash the fetch — the
    feature derived from it goes NaN, others still populate."""
    s3 = _FakeS3()
    _seed_full_price_cache(s3, "test-bucket", DEFAULT_PRICE_CACHE_PREFIX)
    # Remove VIX3M — term slope should become NaN, other features OK
    del s3._objects[("test-bucket", f"{DEFAULT_PRICE_CACHE_PREFIX}VIX3M.parquet")]
    df = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket")
    assert df["vix_term_slope"].isna().all()
    assert not df["vix_level"].isna().all()
    assert not df["spy_20d_return"].isna().all()


def test_fetch_lookback_weeks_truncates() -> None:
    s3 = _FakeS3()
    _seed_full_price_cache(s3, "test-bucket", DEFAULT_PRICE_CACHE_PREFIX)
    df_full = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket")
    df_tail = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket", lookback_weeks=20)
    assert len(df_tail) == 20
    assert len(df_full) > 20


def test_fetch_as_of_truncates_right_edge() -> None:
    """as_of bounds the maximum date in the output — used by the
    backfill script to reproduce historical snapshots."""
    s3 = _FakeS3()
    _seed_full_price_cache(s3, "test-bucket", DEFAULT_PRICE_CACHE_PREFIX)
    as_of = pd.Timestamp("2025-06-01")
    df = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket", as_of=as_of)
    assert df.index.max() <= as_of


def test_current_features_returns_finite_floats_only() -> None:
    """current_features_from_history drops NaN-valued keys so consumers
    can use np.isfinite() checks without manual filtering."""
    history = pd.DataFrame({
        "spy_20d_return": [0.01, 0.02, 0.03],
        "vix_level": [15.0, 16.0, 17.0],
        "market_breadth": [np.nan, np.nan, np.nan],
    })
    current = current_features_from_history(history)
    assert "market_breadth" not in current  # all-NaN column dropped
    assert current["spy_20d_return"] == pytest.approx(0.03)
    assert current["vix_level"] == pytest.approx(17.0)


def test_current_features_merges_extra() -> None:
    """The optional ``extra`` mapping is layered on top — used by the
    handler to inject derived guardrail inputs like spy_30d_return."""
    history = pd.DataFrame({"vix_level": [18.0]})
    current = current_features_from_history(history, extra={"spy_30d_return": -0.08})
    assert current["spy_30d_return"] == pytest.approx(-0.08)
    assert current["vix_level"] == pytest.approx(18.0)


def test_current_features_raises_on_empty_history() -> None:
    with pytest.raises(ValueError, match="Empty feature history"):
        current_features_from_history(pd.DataFrame())


def test_source_tickers_match_data_side_convention() -> None:
    """Source ticker names must NOT include the yfinance caret prefix —
    pins the contract that aligns with the alpha-engine-data price
    collector (collectors/prices.py) caret-stripped parquet storage
    convention."""
    for ticker in SOURCE_TICKERS:
        assert not ticker.startswith("^"), f"{ticker} should not have ^ prefix"
