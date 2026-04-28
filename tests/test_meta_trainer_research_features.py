"""
tests/test_meta_trainer_research_features.py — Unit tests for the
research-signal join helpers added to meta_trainer 2026-04-28.

Background: the predictor's meta-model + isotonic calibrator collapsed
all 27 daily predictions to ~p_up=0.5119 because the four per-ticker
research features had ZERO variance in training (hardcoded constants).
This test file locks the new helpers that pull real values from
historical signals.json snapshots so a future refactor cannot silently
revert to the old behavior.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from training.meta_trainer import (
    _SECTOR_NAME_CANONICAL,
    _build_signals_lookup_by_test_date,
    _extract_research_features,
    _load_signals_history,
)


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def signals_payload():
    """A minimal signals.json snapshot with the fields run_inference reads."""
    return {
        "date": "2026-04-12",
        "market_regime": "neutral",
        "sector_modifiers": {
            "Technology": 1.25,
            "Healthcare": 1.0,
            "Financial": 1.18,
            "Industrials": 1.08,
            "Energy": 1.20,
        },
        "universe": [
            {
                "ticker": "AAPL",
                "score": 75.0,
                "conviction": "rising",
                "sector": "Information Technology",  # canonicalizes to "Technology"
            },
            {
                "ticker": "JNJ",
                "score": 60.0,
                "conviction": "stable",
                "sector": "Health Care",  # canonicalizes to "Healthcare"
            },
            {
                "ticker": "JPM",
                "score": 45.0,
                "conviction": "declining",
                "sector": "Financials",  # canonicalizes to "Financial"
            },
            {
                "ticker": "NEW_LISTING",
                "score": None,  # missing score → row should drop
                "conviction": "stable",
                "sector": "Industrials",
            },
        ],
    }


@pytest.fixture
def fitted_calibrator():
    """A minimal stub matching ResearchCalibrator's interface."""
    cal = MagicMock()
    cal.is_fitted = True
    cal.predict = MagicMock(side_effect=lambda score: 0.55 if score >= 70 else 0.48)
    return cal


# ── _extract_research_features ──────────────────────────────────────────────
class TestExtractResearchFeatures:
    def test_happy_path_uses_calibrator_when_fitted(
        self, signals_payload, fitted_calibrator
    ):
        out = _extract_research_features(signals_payload, "AAPL", fitted_calibrator)
        assert out is not None
        # score=75 → calibrator returns 0.55 (>=70 branch)
        assert out["research_calibrator_prob"] == pytest.approx(0.55)
        assert out["research_composite_score"] == pytest.approx(0.75)
        assert out["research_conviction"] == 1.0  # rising
        # Information Technology → Technology (modifier=1.25), -1.0 → 0.25
        assert out["sector_macro_modifier"] == pytest.approx(0.25)

    def test_falls_back_to_score_norm_when_calibrator_unfitted(
        self, signals_payload
    ):
        cal = MagicMock()
        cal.is_fitted = False
        out = _extract_research_features(signals_payload, "JNJ", cal)
        assert out is not None
        # No fitted calibrator → calibrator_prob == score / 100 == 0.60
        assert out["research_calibrator_prob"] == pytest.approx(0.60)
        assert out["research_composite_score"] == pytest.approx(0.60)
        assert out["research_conviction"] == 0.0  # stable

    def test_falls_back_when_calibrator_is_none(self, signals_payload):
        out = _extract_research_features(signals_payload, "JNJ", None)
        assert out is not None
        assert out["research_calibrator_prob"] == pytest.approx(0.60)

    def test_sector_canonicalization_health_care(self, signals_payload, fitted_calibrator):
        out = _extract_research_features(signals_payload, "JNJ", fitted_calibrator)
        # "Health Care" → canonical "Healthcare" → modifier 1.0 → -1.0 → 0.0
        assert out["sector_macro_modifier"] == pytest.approx(0.0)

    def test_sector_canonicalization_financials(self, signals_payload, fitted_calibrator):
        out = _extract_research_features(signals_payload, "JPM", fitted_calibrator)
        # "Financials" → "Financial" → modifier 1.18 → 0.18
        assert out["sector_macro_modifier"] == pytest.approx(0.18)

    def test_declining_conviction_encoded_as_negative_one(
        self, signals_payload, fitted_calibrator
    ):
        out = _extract_research_features(signals_payload, "JPM", fitted_calibrator)
        assert out["research_conviction"] == -1.0

    def test_returns_none_when_payload_is_none(self, fitted_calibrator):
        assert _extract_research_features(None, "AAPL", fitted_calibrator) is None

    def test_returns_none_when_ticker_absent_from_universe(
        self, signals_payload, fitted_calibrator
    ):
        assert _extract_research_features(signals_payload, "MSFT", fitted_calibrator) is None

    def test_returns_none_when_score_missing(self, signals_payload, fitted_calibrator):
        # "NEW_LISTING" entry has score=None → must drop
        assert _extract_research_features(
            signals_payload, "NEW_LISTING", fitted_calibrator
        ) is None

    def test_unknown_sector_falls_back_to_modifier_one(self, fitted_calibrator):
        payload = {
            "sector_modifiers": {"Technology": 1.25},
            "universe": [
                {
                    "ticker": "FOO",
                    "score": 70.0,
                    "conviction": "stable",
                    "sector": "Aerospace & Defense",  # not in modifier map
                },
            ],
        }
        out = _extract_research_features(payload, "FOO", fitted_calibrator)
        assert out is not None
        # Falls through to default 1.0 → -1.0 → 0.0
        assert out["sector_macro_modifier"] == pytest.approx(0.0)


# ── _build_signals_lookup_by_test_date ──────────────────────────────────────
class TestBuildSignalsLookup:
    def test_most_recent_prior_signals_picked_for_each_test_date(self):
        history = {
            "2026-03-05": {"id": "s1"},
            "2026-03-12": {"id": "s2"},
            "2026-03-19": {"id": "s3"},
        }
        # Test dates: before any snapshot, between snapshots, after last snapshot
        test_dates = [
            "2026-03-04",  # before any → None
            "2026-03-05",  # exactly first → s1
            "2026-03-08",  # after s1, before s2 → s1
            "2026-03-12",  # exactly s2 → s2
            "2026-03-15",  # after s2, before s3 → s2
            "2026-03-25",  # after last → s3
        ]
        out = _build_signals_lookup_by_test_date(history, test_dates)
        assert out["2026-03-04"] is None
        assert out["2026-03-05"]["id"] == "s1"
        assert out["2026-03-08"]["id"] == "s1"
        assert out["2026-03-12"]["id"] == "s2"
        assert out["2026-03-15"]["id"] == "s2"
        assert out["2026-03-25"]["id"] == "s3"

    def test_empty_history_returns_all_none(self):
        out = _build_signals_lookup_by_test_date({}, ["2026-04-01", "2026-04-02"])
        assert out == {"2026-04-01": None, "2026-04-02": None}

    def test_handles_duplicate_test_dates(self):
        history = {"2026-03-05": {"id": "s1"}}
        out = _build_signals_lookup_by_test_date(
            history, ["2026-03-10", "2026-03-10", "2026-03-10"]
        )
        # Only one entry per unique date
        assert len(out) == 1
        assert out["2026-03-10"]["id"] == "s1"


# ── _load_signals_history ───────────────────────────────────────────────────
class TestLoadSignalsHistory:
    def test_loads_all_dated_prefixes_from_s3(self):
        # Mock S3 paginator returning a few date prefixes
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "signals/2026-03-05/"},
                    {"Prefix": "signals/2026-03-12/"},
                    {"Prefix": "signals/notadate/"},     # filtered out
                    {"Prefix": "signals/latest.json"},   # not a date dir
                ]
            }
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        def _get_object(Bucket, Key):
            payload = {"date": Key.split("/")[-2]}
            body = MagicMock()
            body.read.return_value = b'{"date":"' + payload["date"].encode() + b'"}'
            return {"Body": body}

        mock_s3.get_object.side_effect = _get_object

        history = _load_signals_history(mock_s3, "test-bucket")
        assert set(history.keys()) == {"2026-03-05", "2026-03-12"}

    def test_raises_on_empty_history(self):
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"CommonPrefixes": []}]
        mock_s3.get_paginator.return_value = mock_paginator

        with pytest.raises(RuntimeError, match="No signals.json snapshots"):
            _load_signals_history(mock_s3, "test-bucket")

    def test_per_date_failures_are_skipped_not_fatal(self):
        # One date succeeds, one fails — function should skip the failure
        # and still return the successful one.
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "signals/2026-03-05/"},
                    {"Prefix": "signals/2026-03-12/"},
                ]
            }
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        def _get_object(Bucket, Key):
            if "2026-03-12" in Key:
                raise RuntimeError("simulated S3 error")
            body = MagicMock()
            body.read.return_value = b'{"date":"2026-03-05"}'
            return {"Body": body}

        mock_s3.get_object.side_effect = _get_object

        history = _load_signals_history(mock_s3, "test-bucket")
        assert "2026-03-05" in history
        assert "2026-03-12" not in history


# ── Sector-name map invariants ──────────────────────────────────────────────
class TestSectorCanonicalMap:
    def test_known_gics_to_modifier_keys(self):
        """The canonical map must include the GICS → modifier-key translations
        that broke run_inference's sector_modifier lookup pre-fix. Adding new
        sectors requires updating both research's sector_modifiers writer and
        this map in the same change."""
        assert _SECTOR_NAME_CANONICAL["Health Care"] == "Healthcare"
        assert _SECTOR_NAME_CANONICAL["Information Technology"] == "Technology"
        assert _SECTOR_NAME_CANONICAL["Financials"] == "Financial"
