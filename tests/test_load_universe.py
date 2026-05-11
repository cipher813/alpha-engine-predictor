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


# ── 2026-05-11: macro-merge fix for the "NEUTRAL" regime brief defect ──────


class TestLoadWatchlistMergesCanonicalMacro:
    """Two research producers disagree on macro fields:

    * `population/latest.json` carries a pre-critic `market_regime` (and
      typically no `sector_modifiers`).
    * `signals/latest.json` carries the post-critic `market_regime` plus
      `sector_modifiers` / `sector_ratings` written by macro_agent.

    Pre-fix the predictor's morning brief and veto-threshold logic saw
    whichever value population happened to ship — neutral while research's
    brief loudly proclaimed BULL. Post-fix the population payload is
    overlaid with canonical macro fields from signals' fallback chain.
    """

    def test_signals_market_regime_overrides_population(self):
        """Population says neutral, signals says bull → returned data
        carries bull (signals wins by design)."""
        population = {
            "population": [{"ticker": "AAPL"}],
            "market_regime": "neutral",  # stale pre-critic value
        }
        signals = {
            "market_regime": "bull",  # canonical post-critic value
        }
        s3 = _mock_s3_with_keys({
            "population/latest.json": population,
            "signals/2026-04-27/signals.json": signals,
        })
        with patch("boto3.client", return_value=s3):
            _, _, data = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-04-27",
            )
        assert data["market_regime"] == "bull"

    def test_signals_populates_missing_macro_fields_on_population(self):
        """Population lacks sector_modifiers / sector_ratings; signals
        has them. Returned data carries the signals values."""
        population = {"population": [{"ticker": "AAPL"}]}
        signals = {
            "market_regime": "bull",
            "sector_modifiers": {"Technology": 1.1, "Energy": 1.05},
            "sector_ratings": {"Technology": "OVERWEIGHT"},
        }
        s3 = _mock_s3_with_keys({
            "population/latest.json": population,
            "signals/2026-04-27/signals.json": signals,
        })
        with patch("boto3.client", return_value=s3):
            _, _, data = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-04-27",
            )
        assert data["market_regime"] == "bull"
        assert data["sector_modifiers"] == {"Technology": 1.1, "Energy": 1.05}
        assert data["sector_ratings"] == {"Technology": "OVERWEIGHT"}

    def test_missing_signals_field_preserves_population_value(self):
        """If signals.json is missing a macro field that population HAS,
        do NOT blank the population value with None. Absence ≠ override."""
        population = {
            "population": [{"ticker": "AAPL"}],
            "market_regime": "neutral",
            "sector_ratings": {"Energy": "OVERWEIGHT"},
        }
        signals = {
            "market_regime": "bull",
            # sector_ratings absent in signals
        }
        s3 = _mock_s3_with_keys({
            "population/latest.json": population,
            "signals/2026-04-27/signals.json": signals,
        })
        with patch("boto3.client", return_value=s3):
            _, _, data = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-04-27",
            )
        assert data["market_regime"] == "bull"
        # population's sector_ratings retained because signals didn't carry it
        assert data["sector_ratings"] == {"Energy": "OVERWEIGHT"}

    def test_macro_merge_walks_back_to_signals_latest(self):
        """Today's signals/{date}/signals.json is absent (typical weekday);
        the merge helper falls back through prior weekdays to
        signals/latest.json — same chain `_read_buy_candidates_from_signals`
        uses, now shared via `_load_signals_payload_with_fallback`."""
        population = {
            "population": [{"ticker": "AAPL"}],
            "market_regime": "neutral",
        }
        signals_latest = {"market_regime": "bull"}
        s3 = _mock_s3_with_keys({
            "population/latest.json": population,
            "signals/latest.json": signals_latest,
            # All dated signals/{date}/signals.json keys missing
        })
        with patch("boto3.client", return_value=s3):
            _, _, data = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-04-27",
            )
        assert data["market_regime"] == "bull"

    def test_no_signals_payload_leaves_population_macro_intact(self):
        """Defensive: when no signals key in the entire fallback chain
        resolves, population's macro fields are returned untouched.
        This preserves prior behavior when research is genuinely down."""
        population = {
            "population": [{"ticker": "AAPL"}],
            "market_regime": "neutral",
        }
        s3 = _mock_s3_with_keys({
            "population/latest.json": population,
            # NO signals/* anywhere
        })
        with patch("boto3.client", return_value=s3):
            _, _, data = load_watchlist(
                "auto", s3_bucket="b", date_str="2026-04-27",
            )
        # Without signals to overlay, population's value stays
        assert data["market_regime"] == "neutral"


class TestMergeCanonicalMacroHelper:
    """Direct unit tests for the overlay helper. Pin the
    'non-None-only' semantics so callers can rely on it as a one-shot
    safe overlay primitive."""

    def test_only_non_none_fields_overlay(self):
        from inference.stages.load_universe import _merge_canonical_macro_into
        data = {
            "market_regime": "neutral",
            "sector_modifiers": {"Energy": 1.05},
        }
        signals = {
            "market_regime": "bull",
            "sector_modifiers": None,  # explicit None must NOT blank data
        }
        out = _merge_canonical_macro_into(data, signals)
        assert out["market_regime"] == "bull"
        # Explicit None left the existing dict intact
        assert out["sector_modifiers"] == {"Energy": 1.05}

    def test_returns_same_dict_for_caller_chaining(self):
        """Helper mutates `data` in place and returns it so the caller
        can write `_merge_canonical_macro_into(data, signals)` without
        a separate assignment."""
        from inference.stages.load_universe import _merge_canonical_macro_into
        data = {}
        out = _merge_canonical_macro_into(data, {"market_regime": "bull"})
        assert out is data


class TestGetUniverseTickersFallback:
    """Regression: get_universe_tickers used to hardcode
    `signals/{date}/signals.json` and return `{}` signals_data on miss.
    Post-fix it walks the same fallback chain everywhere else in this
    module uses."""

    def test_falls_back_to_signals_latest_when_today_missing(self):
        """No dated key → uses signals/latest.json universe + signals_data."""
        signals_latest = {
            "signals": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
            "market_regime": "bull",
        }
        s3 = _mock_s3_with_keys({"signals/latest.json": signals_latest})
        with patch("boto3.client", return_value=s3):
            tickers, data = get_universe_tickers("b", date_str="2026-05-11")
        assert set(tickers) == {"AAPL", "MSFT"}
        # Critical regression: signals_data is NOT empty when only the
        # dated key is missing. Pre-fix this returned ({fallback_tickers}, {}).
        assert data["market_regime"] == "bull"

    def test_fallback_universe_when_no_signals_anywhere(self):
        """All signals keys missing → falls back to _FALLBACK_TICKERS,
        signals_data is `{}` (the only place where empty data is correct)."""
        from inference.stages.load_universe import _FALLBACK_TICKERS
        s3 = _mock_s3_with_keys({})
        with patch("boto3.client", return_value=s3):
            tickers, data = get_universe_tickers("b", date_str="2026-05-11")
        assert tickers == _FALLBACK_TICKERS
        assert data == {}
