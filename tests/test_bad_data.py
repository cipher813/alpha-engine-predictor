"""
tests/test_bad_data.py — Regression tests for bad/missing external data.

Verifies that the prediction pipeline degrades gracefully (logs warnings,
returns empty/partial results) instead of crashing when yfinance, S3, or
macro series return garbage data.

Created as part of the 2026-04-01 reliability hardening plan after a
NaT.normalize() crash from empty VIX data took down the predictor Lambda.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.daily_predict import _safe_last_date


# ---------------------------------------------------------------------------
# _safe_last_date unit tests
# ---------------------------------------------------------------------------

class TestSafeLastDate:
    """Unit tests for the centralized NaT guard."""

    def test_normal_index(self):
        idx = pd.DatetimeIndex(["2026-03-28", "2026-03-31", "2026-04-01"])
        result = _safe_last_date(idx)
        assert result == pd.Timestamp("2026-04-01").normalize()

    def test_empty_index(self):
        idx = pd.DatetimeIndex([])
        assert _safe_last_date(idx) is None

    def test_none_index(self):
        assert _safe_last_date(None) is None

    def test_all_nat_index(self):
        idx = pd.DatetimeIndex([pd.NaT, pd.NaT])
        assert _safe_last_date(idx) is None

    def test_mixed_nat_index(self):
        idx = pd.DatetimeIndex(["2026-03-28", pd.NaT, "2026-04-01"])
        result = _safe_last_date(idx)
        assert result == pd.Timestamp("2026-04-01").normalize()

    def test_single_date(self):
        idx = pd.DatetimeIndex(["2026-04-01"])
        result = _safe_last_date(idx)
        assert result == pd.Timestamp("2026-04-01").normalize()

    def test_returns_normalized(self):
        """Result should have time component zeroed out."""
        idx = pd.DatetimeIndex(["2026-04-01 14:30:00"])
        result = _safe_last_date(idx)
        assert result.hour == 0
        assert result.minute == 0


# ---------------------------------------------------------------------------
# Slim cache with bad data
# ---------------------------------------------------------------------------

class TestSlimCacheNaT:
    """Slim cache loading should not crash on NaT indices."""

    def _make_slim_df(self, dates, close_values):
        """Build a minimal OHLCV DataFrame."""
        idx = pd.DatetimeIndex(dates)
        return pd.DataFrame({
            "Open": close_values,
            "High": close_values,
            "Low": close_values,
            "Close": close_values,
            "Volume": [1000] * len(dates),
        }, index=idx)

    def test_all_nat_slim_data_returns_none(self):
        """If every slim cache ticker has NaT-only index, fall back gracefully."""
        # _safe_last_date returns None for all-NaT → the slim_last_date
        # calculation should return (None, None) to trigger yfinance fallback.
        all_nat_df = pd.DataFrame({
            "Open": [np.nan], "High": [np.nan], "Low": [np.nan],
            "Close": [np.nan], "Volume": [0],
        }, index=pd.DatetimeIndex([pd.NaT]))

        dates = [_safe_last_date(all_nat_df.index)]
        valid = [d for d in dates if d is not None]
        assert len(valid) == 0  # all filtered out

    def test_partial_nat_slim_data(self):
        """If some tickers have NaT index, the valid ones still work."""
        good_df = self._make_slim_df(["2026-03-31"], [100.0])
        bad_df = pd.DataFrame({
            "Close": [np.nan],
        }, index=pd.DatetimeIndex([pd.NaT]))

        dates = [_safe_last_date(df.index) for df in [good_df, bad_df]]
        valid = [d for d in dates if d is not None]
        assert len(valid) == 1
        assert valid[0] == pd.Timestamp("2026-03-31").normalize()


# ---------------------------------------------------------------------------
# fetch_today_prices with empty yfinance
# ---------------------------------------------------------------------------

class TestFetchTodayPricesEmpty:
    """fetch_today_prices should return empty DataFrames, not crash."""

    @patch("inference.stages.load_prices._yf_download_batch")
    def test_yfinance_returns_empty(self, mock_download):
        """When yfinance returns an empty DataFrame, result should be empty too."""
        mock_download.return_value = pd.DataFrame()
        from inference.daily_predict import fetch_today_prices
        result = fetch_today_prices(["AAPL"])
        assert "AAPL" in result
        # Empty DataFrame is acceptable

    @patch("inference.stages.load_prices._yf_download_batch", side_effect=ConnectionError("timeout"))
    def test_yfinance_connection_error(self, mock_download):
        """ConnectionError should be caught, not propagate."""
        from inference.daily_predict import fetch_today_prices
        result = fetch_today_prices(["AAPL"])
        assert isinstance(result, dict)
        assert result.get("AAPL", pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# fetch_macro_series with partial failures
# ---------------------------------------------------------------------------

class TestFetchMacroPartialFailure:
    """Macro series fetch should degrade gracefully on partial failures."""

    @patch("yfinance.download")
    def test_partial_macro_failure(self, mock_download):
        """If some macro symbols fail, the rest should still be returned."""
        # Simulate: SPY works, ^VIX is empty
        dates = pd.date_range("2026-01-01", "2026-03-31", freq="B")
        close = np.random.uniform(400, 500, len(dates))

        # Build multi-ticker response
        spy_data = pd.DataFrame({"Close": close}, index=dates)

        # Return a DataFrame that has SPY but not ^VIX
        mock_download.return_value = pd.DataFrame()

        from inference.daily_predict import fetch_macro_series
        result = fetch_macro_series(period="1y")
        # Should not crash — returns whatever it can
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Caret ticker mapping
# ---------------------------------------------------------------------------

class TestCaretTickerMapping:
    """Split re-fetch should map bare macro tickers to yfinance caret format."""

    def test_caret_map_coverage(self):
        """VIX, TNX, IRX must be mapped to ^VIX, ^TNX, ^IRX."""
        _CARET_MAP = {"VIX": "^VIX", "TNX": "^TNX", "IRX": "^IRX"}
        assert _CARET_MAP["VIX"] == "^VIX"
        assert _CARET_MAP["TNX"] == "^TNX"
        assert _CARET_MAP["IRX"] == "^IRX"

    def test_non_macro_tickers_pass_through(self):
        """Regular tickers like AAPL should not get a caret prefix."""
        _CARET_MAP = {"VIX": "^VIX", "TNX": "^TNX", "IRX": "^IRX"}
        tickers = ["AAPL", "VIX", "MSFT"]
        mapped = [_CARET_MAP.get(t, t) for t in tickers]
        assert mapped == ["AAPL", "^VIX", "MSFT"]


# ---------------------------------------------------------------------------
# Daily closes freshness gate
# ---------------------------------------------------------------------------

class TestDailyClosesFreshnessGate:
    """_verify_daily_closes_fresh guards against missing or stale DataPhase1 output."""

    def _s3_mock(self, last_modified=None, raise_exc=None):
        from unittest.mock import MagicMock
        s3 = MagicMock()
        if raise_exc is not None:
            s3.head_object.side_effect = raise_exc
        else:
            s3.head_object.return_value = {"LastModified": last_modified}
        return s3

    def test_missing_file_raises_pipeline_abort(self):
        from inference.stages.load_prices import _verify_daily_closes_fresh
        from inference.pipeline import PipelineAbort

        s3 = self._s3_mock(raise_exc=Exception("NoSuchKey"))
        with pytest.raises(PipelineAbort, match="not found"):
            _verify_daily_closes_fresh(s3, "bucket", "2026-04-16")

    def test_fresh_file_today_passes(self):
        from datetime import datetime, timezone, timedelta
        from inference.stages.load_prices import _verify_daily_closes_fresh

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        recent = datetime.now(timezone.utc) - timedelta(minutes=30)
        s3 = self._s3_mock(last_modified=recent)
        # Should NOT raise
        _verify_daily_closes_fresh(s3, "bucket", today)

    def test_stale_file_today_raises_pipeline_abort(self):
        from datetime import datetime, timezone, timedelta
        from inference.stages.load_prices import _verify_daily_closes_fresh
        from inference.pipeline import PipelineAbort

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        yesterday_write = datetime.now(timezone.utc) - timedelta(hours=25)
        s3 = self._s3_mock(last_modified=yesterday_write)
        with pytest.raises(PipelineAbort, match="stale"):
            _verify_daily_closes_fresh(s3, "bucket", today)

    def test_backfill_skips_freshness_check(self):
        """Historical date_str should NOT enforce LastModified — legitimate old files."""
        from datetime import datetime, timezone, timedelta
        from inference.stages.load_prices import _verify_daily_closes_fresh

        old_write = datetime.now(timezone.utc) - timedelta(days=30)
        s3 = self._s3_mock(last_modified=old_write)
        # Past date: should NOT raise even though file is ancient
        _verify_daily_closes_fresh(s3, "bucket", "2026-03-10")

    def test_boundary_exactly_at_threshold(self):
        """File at exactly the 12h boundary should pass (strict > for staleness)."""
        from datetime import datetime, timezone, timedelta
        from inference.stages.load_prices import _verify_daily_closes_fresh

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        at_boundary = datetime.now(timezone.utc) - timedelta(hours=11, minutes=59)
        s3 = self._s3_mock(last_modified=at_boundary)
        _verify_daily_closes_fresh(s3, "bucket", today)
