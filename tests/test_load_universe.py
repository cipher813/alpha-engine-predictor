"""Tests for inference/stages/load_universe.py — local file loading."""

import json
import tempfile
from pathlib import Path

import pytest

from inference.stages.load_universe import load_watchlist, get_universe_tickers


class TestLoadWatchlistLocal:
    def test_signals_file(self):
        data = {
            "date": "2026-04-08",
            "universe": [
                {"ticker": "AAPL", "score": 82},
                {"ticker": "MSFT", "score": 75},
                {"ticker": "NVDA", "score": 90},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            f.flush()
            tickers, sources, raw = load_watchlist(f.name)
            assert len(tickers) == 3
            assert "AAPL" in tickers
            assert sources["AAPL"] == "tracked"

    def test_population_file(self):
        data = {
            "population": [
                {"ticker": "GOOG", "sector": "Technology"},
                {"ticker": "AMZN", "sector": "Consumer"},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            f.flush()
            tickers, sources, raw = load_watchlist(f.name)
            assert len(tickers) == 2
            assert sources["GOOG"] == "population"

    def test_empty_universe(self):
        data = {"date": "2026-04-08", "universe": []}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            f.flush()
            tickers, sources, raw = load_watchlist(f.name)
            assert tickers == []

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_watchlist("/nonexistent/signals.json")

    def test_tickers_uppercased(self):
        data = {"universe": [{"ticker": "aapl"}, {"ticker": "msft"}]}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            f.flush()
            tickers, _, _ = load_watchlist(f.name)
            assert all(t.isupper() for t in tickers)

    def test_tickers_sorted(self):
        data = {"universe": [{"ticker": "NVDA"}, {"ticker": "AAPL"}, {"ticker": "MSFT"}]}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(data, f)
            f.flush()
            tickers, _, _ = load_watchlist(f.name)
            assert tickers == sorted(tickers)

    def test_auto_without_bucket_raises(self):
        with pytest.raises(ValueError, match="s3_bucket"):
            load_watchlist("auto")
