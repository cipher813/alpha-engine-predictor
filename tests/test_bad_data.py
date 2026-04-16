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
# ArcticDB freshness gate (Phase 7a — replaces legacy daily_closes gate)
# ---------------------------------------------------------------------------

class TestArcticFreshnessGate:
    """_verify_arctic_fresh guards against stale/missing SPY in ArcticDB."""

    def _macro_lib_mock(self, last_date=None, raise_exc=None):
        from unittest.mock import MagicMock
        lib = MagicMock()
        if raise_exc is not None:
            lib.read.side_effect = raise_exc
            return lib
        if last_date is None:
            lib.read.return_value.data = pd.DataFrame(columns=["Close"])
        else:
            idx = pd.DatetimeIndex([pd.Timestamp(last_date)])
            lib.read.return_value.data = pd.DataFrame({"Close": [500.0]}, index=idx)
        return lib

    def test_missing_spy_raises_pipeline_abort(self):
        from inference.stages.load_prices import _verify_arctic_fresh
        from inference.pipeline import PipelineAbort

        lib = self._macro_lib_mock(raise_exc=Exception("SymbolNotFound"))
        with pytest.raises(PipelineAbort, match="unreadable"):
            _verify_arctic_fresh(lib, "2026-04-16")

    def test_empty_spy_raises_pipeline_abort(self):
        from inference.stages.load_prices import _verify_arctic_fresh
        from inference.pipeline import PipelineAbort

        lib = self._macro_lib_mock(last_date=None)
        with pytest.raises(PipelineAbort, match="no rows"):
            _verify_arctic_fresh(lib, "2026-04-16")

    def test_fresh_spy_passes(self):
        from inference.stages.load_prices import _verify_arctic_fresh

        lib = self._macro_lib_mock(last_date="2026-04-16")
        _verify_arctic_fresh(lib, "2026-04-16")  # should not raise

    def test_stale_spy_raises_pipeline_abort(self):
        from inference.stages.load_prices import _verify_arctic_fresh
        from inference.pipeline import PipelineAbort

        lib = self._macro_lib_mock(last_date="2026-04-15")
        with pytest.raises(PipelineAbort, match="stale"):
            _verify_arctic_fresh(lib, "2026-04-16")
