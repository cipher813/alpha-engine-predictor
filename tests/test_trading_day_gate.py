"""Tests for inference.trading_day_gate + its handler dispatch.

The gate is pure NYSE-calendar math (config#1430). Tests use KNOWN fixed
dates — never "today" — so they are deterministic. The handler-dispatch test
asserts the gate returns BEFORE PredictorPreflight is ever constructed, which
is the whole point of the fix: the gate cannot be broken by infra/env issues.
"""
import sys
from unittest.mock import MagicMock

import pytest

from inference.trading_day_gate import check_trading_day


# ── Direct gate tests (fixed dates) ─────────────────────────────────────────


def test_known_weekday_trading_day():
    """2026-06-30 is a Tuesday and an NYSE trading day."""
    r = check_trading_day("2026-06-30")
    assert r["is_trading_day"] is True
    assert r["check_date"] == "2026-06-30"
    assert r["day_name"] == "Tuesday"
    assert r["marker"] == "TRADING DAY"
    # A trading day carries no reason / next_trading_day fields.
    assert "reason" not in r
    assert "next_trading_day" not in r


def test_saturday_is_weekend():
    """2026-06-27 is a Saturday — closed, reason weekend, has next_trading_day."""
    r = check_trading_day("2026-06-27")
    assert r["is_trading_day"] is False
    assert r["day_name"] == "Saturday"
    assert r["marker"] == "MARKET_CLOSED"
    assert r["reason"] == "weekend"
    # Next trading day is a real future date, not a weekend.
    assert r["next_trading_day"] > "2026-06-27"


def test_new_years_day_is_holiday():
    """2026-01-01 is a Thursday New Year's Day — closed as an NYSE holiday."""
    r = check_trading_day("2026-01-01")
    assert r["is_trading_day"] is False
    assert r["day_name"] == "Thursday"
    assert r["marker"] == "MARKET_CLOSED"
    assert r["reason"] == "NYSE holiday"
    assert "next_trading_day" in r


def test_none_returns_expected_keys():
    """date=None resolves to today (NYSE) and returns the standard shape."""
    r = check_trading_day(None)
    assert set(["is_trading_day", "check_date", "day_name", "marker"]).issubset(r)
    assert isinstance(r["is_trading_day"], bool)
    assert r["marker"] in ("TRADING DAY", "MARKET_CLOSED")


# ── Handler dispatch test (gate returns BEFORE preflight) ────────────────────


def test_handler_dispatches_before_preflight(monkeypatch):
    """handler(action=check_trading_day) returns the gate result WITHOUT ever
    constructing PredictorPreflight — proving zero preflight/infra dependency."""

    class _ExplodingPreflight:
        def __init__(self, *a, **k):
            raise AssertionError("PredictorPreflight must NOT be constructed for check_trading_day")

    fake_module = MagicMock(PredictorPreflight=_ExplodingPreflight)
    monkeypatch.setitem(sys.modules, "inference.preflight", fake_module)

    import inference.handler as h
    result = h.handler({"action": "check_trading_day", "date": "2026-06-30"}, None)

    assert result["is_trading_day"] is True
    assert result["check_date"] == "2026-06-30"
    assert result["marker"] == "TRADING DAY"
