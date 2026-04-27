"""Tests for inference/stages/load_universe.py — local file loading."""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ── 2026-04-27: main pass auto-includes signals.buy_candidates ──────────────


def _mock_s3_with_keys(payloads_by_key: dict[str, dict]):
    """Build a MagicMock boto3 s3 client whose get_object returns the given
    JSON payloads keyed by S3 Key. Missing keys raise NoSuchKey ClientError.
    """
    from botocore.exceptions import ClientError

    def get_object(Bucket, Key):
        payload = payloads_by_key.get(Key)
        if payload is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )
        body = json.dumps(payload).encode("utf-8")
        return {"Body": io.BytesIO(body)}

    s3 = MagicMock()
    s3.get_object.side_effect = get_object
    return s3


class TestLoadWatchlistAutoUnionsBuyCandidates:
    """The 2026-04-27 incident exposed that the predictor's main pass scored
    only `population/latest.json` tickers, leaving any `signals.buy_candidates`
    not in population for a separate supplemental Lambda invocation. After this
    change the main pass unions both, so the SF coverage-gap re-invoke becomes
    a no-op safety net on the happy path.
    """

    def test_main_pass_unions_population_and_buy_candidates(self):
        population = {"population": [
            {"ticker": "AAPL"}, {"ticker": "MSFT"}, {"ticker": "NVDA"},
        ]}
        signals = {"buy_candidates": [
            {"ticker": "ABT"}, {"ticker": "DVA"}, {"ticker": "AAPL"},  # one overlap
        ]}
        s3 = _mock_s3_with_keys({
            "population/latest.json": population,
            "signals/2026-04-27/signals.json": signals,
        })
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-04-27",
            )
        # Union: population ∪ buy_candidates
        assert set(tickers) == {"AAPL", "MSFT", "NVDA", "ABT", "DVA"}
        # Sources reflect the union: overlap → "both", new → "buy_candidate"
        assert sources["AAPL"] == "both"
        assert sources["MSFT"] == "population"
        assert sources["NVDA"] == "population"
        assert sources["ABT"] == "buy_candidate"
        assert sources["DVA"] == "buy_candidate"

    def test_main_pass_falls_through_when_no_signals_payload(self):
        """No signals/*.json available — population alone is returned (matches
        prior behavior)."""
        population = {"population": [{"ticker": "AAPL"}, {"ticker": "MSFT"}]}
        s3 = _mock_s3_with_keys({"population/latest.json": population})
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-04-27",
            )
        assert set(tickers) == {"AAPL", "MSFT"}
        assert sources == {"AAPL": "population", "MSFT": "population"}

    def test_main_pass_walks_back_to_signals_latest(self):
        """When signals/{date}/signals.json is absent (weekday — research
        runs Saturdays), the helper falls back through prior weekdays and
        ultimately signals/latest.json."""
        population = {"population": [{"ticker": "AAPL"}]}
        # Only signals/latest.json exists — all dated keys missing
        signals_latest = {"buy_candidates": [{"ticker": "ABT"}]}
        s3 = _mock_s3_with_keys({
            "population/latest.json": population,
            "signals/latest.json": signals_latest,
        })
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-04-27",
            )
        assert set(tickers) == {"AAPL", "ABT"}
        assert sources["ABT"] == "buy_candidate"

    def test_main_pass_handles_buy_candidates_as_strings(self):
        """`buy_candidates` may be a list of strings, not just dicts. Both
        shapes appear in the wild; the helper must handle either."""
        population = {"population": [{"ticker": "AAPL"}]}
        signals = {"buy_candidates": ["abt", "DVA"]}  # string entries, mixed case
        s3 = _mock_s3_with_keys({
            "population/latest.json": population,
            "signals/2026-04-27/signals.json": signals,
        })
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-04-27",
            )
        assert "ABT" in tickers
        assert "DVA" in tickers
        assert sources["ABT"] == "buy_candidate"

    def test_main_pass_empty_buy_candidates_returns_population_only(self):
        """Empty buy_candidates list → no union, population alone returned."""
        population = {"population": [{"ticker": "AAPL"}, {"ticker": "MSFT"}]}
        signals = {"buy_candidates": []}
        s3 = _mock_s3_with_keys({
            "population/latest.json": population,
            "signals/2026-04-27/signals.json": signals,
        })
        with patch("boto3.client", return_value=s3):
            tickers, sources, _ = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-04-27",
            )
        assert set(tickers) == {"AAPL", "MSFT"}
        assert sources["AAPL"] == "population"
