"""Tests for regime/features.py — macro feature fetcher."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from regime.features import (
    DEFAULT_PRICE_CACHE_PREFIX,
    SOURCE_TICKERS,
    _PRICE_CACHE_NEW_PREFIX,
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
    _seed_full_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
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
    _seed_full_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
    df = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket")
    assert 350 < df["hy_oas_bps"].mean() < 450, (
        f"expected ~400 bps (4.0% × 100), got {df['hy_oas_bps'].mean():.1f}"
    )


def test_fetch_yield_curve_slope_is_difference() -> None:
    s3 = _FakeS3()
    _seed_full_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
    df = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket")
    # TNX synthetic mean ≈ 4.0, TWO ≈ 3.5 → slope ≈ 0.5
    assert 0.3 < df["yield_curve_slope"].mean() < 0.7


def test_fetch_vix_term_slope_is_difference() -> None:
    s3 = _FakeS3()
    _seed_full_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
    df = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket")
    # VIX3M ≈ 19, VIX ≈ 18 → term slope ≈ 1.0
    assert 0.5 < df["vix_term_slope"].mean() < 1.5


def test_fetch_market_breadth_is_nan_pending_source() -> None:
    """Breadth source lands in a follow-up PR; column must be NaN today
    so downstream consumers see absence explicitly, not a spurious 0."""
    s3 = _FakeS3()
    _seed_full_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
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
    _seed_full_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
    # Remove VIX3M — term slope should become NaN, other features OK
    del s3._objects[("test-bucket", f"{_PRICE_CACHE_NEW_PREFIX}VIX3M.parquet")]
    df = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket")
    assert df["vix_term_slope"].isna().all()
    assert not df["vix_level"].isna().all()
    assert not df["spy_20d_return"].isna().all()


def test_fetch_lookback_weeks_truncates() -> None:
    s3 = _FakeS3()
    _seed_full_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
    df_full = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket")
    df_tail = fetch_macro_feature_history(s3_client=s3, bucket="test-bucket", lookback_weeks=20)
    assert len(df_tail) == 20
    assert len(df_full) > 20


def test_fetch_as_of_truncates_right_edge() -> None:
    """as_of bounds the maximum date in the output — used by the
    backfill script to reproduce historical snapshots."""
    s3 = _FakeS3()
    _seed_full_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
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


# ── Wave-3 reader migration (ROADMAP L1401) ─────────────────────────────────


def test_wave3_reader_resolves_reference_and_ignores_legacy() -> None:
    """Wave-3 PR4 cutover (DONE): the reader resolves the price_cache parquet
    from ``reference/price_cache/`` ONLY. The legacy ``predictor/price_cache/``
    prefix is dropped from the fallback chain and must NOT be consulted, even
    if a stale copy still lingers under it pre-``aws s3 rm``."""
    from regime.features import _PRICE_CACHE_NEW_PREFIX, _read_parquet_close

    s3 = _FakeS3()
    # Seed the NEW prefix with one Close value, LEGACY with a clearly-
    # different stale one. If the reader ever picked legacy, we'd see 999.
    new_df = pd.DataFrame(
        {"Close": [100.0]},
        index=pd.to_datetime(["2026-05-19"]),
    )
    legacy_df = pd.DataFrame(
        {"Close": [999.0]},
        index=pd.to_datetime(["2026-05-19"]),
    )
    s3.put_parquet("bkt", f"{_PRICE_CACHE_NEW_PREFIX}SPY.parquet", new_df)
    s3.put_parquet("bkt", f"{DEFAULT_PRICE_CACHE_PREFIX}SPY.parquet", legacy_df)

    s = _read_parquet_close(
        "SPY", s3_client=s3, bucket="bkt",
        prefix=DEFAULT_PRICE_CACHE_PREFIX,
    )
    assert s.iloc[0] == 100.0, "reader must resolve reference/, never legacy"


def test_wave3_reader_does_not_fall_back_to_legacy_when_reference_missing() -> None:
    """Wave-3 PR4 cutover: with the legacy fallback dropped, a miss on
    ``reference/price_cache/`` is the end of the chain. A parquet present
    ONLY under the legacy prefix must NOT be found — the reader returns an
    empty series (callers treat absence as non-fatal)."""
    from regime.features import _read_parquet_close

    s3 = _FakeS3()
    legacy_df = pd.DataFrame(
        {"Close": [123.0]},
        index=pd.to_datetime(["2026-05-19"]),
    )
    # Only legacy seeded (no reference/ key) — post-cutover this is invisible.
    s3.put_parquet("bkt", f"{DEFAULT_PRICE_CACHE_PREFIX}SPY.parquet", legacy_df)

    s = _read_parquet_close(
        "SPY", s3_client=s3, bucket="bkt",
        prefix=DEFAULT_PRICE_CACHE_PREFIX,
    )
    assert s.empty, "legacy-only parquet must be invisible post-cutover"


def test_wave3_reader_explicit_custom_prefix_does_not_fall_back() -> None:
    """Test/config-override callers that pass a non-default prefix opt
    out of the fallback chain — single-prefix semantics. Mirrors the
    write-side ``price_cache_write_prefixes`` opt-out."""
    from regime.features import _read_parquet_close

    s3 = _FakeS3()
    # Seed only the legacy default — the custom prefix has nothing.
    s3.put_parquet(
        "bkt", f"{DEFAULT_PRICE_CACHE_PREFIX}SPY.parquet",
        pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2026-05-19"]),
        ),
    )
    s = _read_parquet_close(
        "SPY", s3_client=s3, bucket="bkt",
        prefix="custom/explicit/",  # NOT the default
    )
    # Custom prefix had no SPY.parquet → empty (no fallback to legacy).
    assert s.empty
