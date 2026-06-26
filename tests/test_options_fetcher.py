"""Tests for data.options_fetcher — S3 archive options read (yfinance retired).

PR4b (config#874): the predictor's options feature read is a hard cutover off
the live ``yfinance.Ticker().option_chain()`` fetch and onto the S3 archive
snapshot the upstream collector writes at ``archive/options/{date}.json``.
These tests stub the S3 read with a canned producer-shaped payload, assert the
reader consumes it in predictor units, and assert NO yfinance import/fetch
occurs anywhere in the options read path.
"""

import json
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from data import options_fetcher


def _s3_body(payload: dict):
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode()
    return {"Body": body}


@pytest.fixture
def fake_s3(monkeypatch):
    """Stub boto3.client('s3') so load_historical_options never hits AWS."""
    s3 = MagicMock(name="s3")
    fake_boto3 = MagicMock(name="boto3")
    fake_boto3.client.return_value = s3
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    return s3


@pytest.fixture
def no_yfinance(monkeypatch):
    """Make ANY ``import yfinance`` in the options path explode — proves the
    archive read never falls through to a live fetch (PR4b cutover)."""
    real_import = __import__

    def guard(name, *args, **kwargs):
        if name == "yfinance" or name.startswith("yfinance."):
            raise AssertionError("yfinance was imported — PR4b cutover violated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guard)


# ── archive read happy path ──────────────────────────────────────────────────


def test_load_historical_options_reads_archive_key(fake_s3):
    """Reads archive/options/{date}.json from the predictor bucket root."""
    fake_s3.get_object.return_value = _s3_body(
        {"AAPL": {"put_call_ratio": 1.0, "iv_rank": 50.0, "atm_iv": 0.25}}
    )
    out = options_fetcher.load_historical_options("2026-06-26", bucket="bkt")

    fake_s3.get_object.assert_called_once_with(
        Bucket="bkt", Key="archive/options/2026-06-26.json"
    )
    assert set(out) == {"AAPL"}
    feat = out["AAPL"]
    # put_call_ratio is log-transformed: log(1.0) == 0.0
    assert feat["put_call_ratio"] == pytest.approx(0.0)
    # iv_rank normalized 0-100 -> [0,1]
    assert feat["iv_rank"] == pytest.approx(0.5)
    assert feat["atm_iv"] == pytest.approx(0.25)


def test_load_historical_options_units_match_producer_shape(fake_s3):
    """Producer emits raw P/C ratio + 0-100 iv_rank; reader log-transforms /
    divides by 100 (verified vs nousergon-data _build_predictor_options_mirror)."""
    fake_s3.get_object.return_value = _s3_body(
        {"MSFT": {"put_call_ratio": 2.0, "iv_rank": 80.0, "atm_iv": 0.0}}
    )
    out = options_fetcher.load_historical_options("2026-06-26", bucket="bkt")
    assert out["MSFT"]["put_call_ratio"] == pytest.approx(float(np.log(2.0)))
    assert out["MSFT"]["iv_rank"] == pytest.approx(0.8)


def test_load_historical_options_defaults_on_partial_payload(fake_s3):
    """Producer omits atm_iv; reader defaults it to 0.0. Missing pc/iv_rank
    fall back to neutral (log(1.0)=0, 50/100=0.5)."""
    fake_s3.get_object.return_value = _s3_body({"NVDA": {}})
    out = options_fetcher.load_historical_options("2026-06-26", bucket="bkt")
    assert out["NVDA"] == {
        "put_call_ratio": pytest.approx(0.0),
        "iv_rank": pytest.approx(0.5),
        "atm_iv": 0.0,
    }


def test_load_historical_options_does_not_use_yfinance(fake_s3, no_yfinance):
    """The archive read path must never import yfinance."""
    fake_s3.get_object.return_value = _s3_body(
        {"AAPL": {"put_call_ratio": 1.0, "iv_rank": 50.0}}
    )
    out = options_fetcher.load_historical_options("2026-06-26", bucket="bkt")
    assert "AAPL" in out


# ── archive miss → None (caller neutral-fills) ───────────────────────────────


def test_load_historical_options_returns_none_on_missing_key(fake_s3):
    fake_s3.get_object.side_effect = RuntimeError("NoSuchKey")
    assert options_fetcher.load_historical_options("2026-06-26", bucket="bkt") is None


# ── yfinance fully retired from the module ───────────────────────────────────


def test_live_yfinance_fetch_symbol_removed():
    """PR4b retires the live fetch — the symbol and its yfinance-only helpers
    must be gone so a regression that re-adds them fails loudly."""
    for gone in (
        "fetch_options_features",
        "_select_expiry",
        "_get_atm_iv",
        "_compute_iv_rank",
    ):
        assert not hasattr(options_fetcher, gone), f"{gone} should be removed in PR4b"
