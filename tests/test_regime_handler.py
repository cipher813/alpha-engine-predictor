"""Tests for regime/handler.py — Lambda entry point + orchestration.

Exercises the full pipeline (fetch → fit → assemble → write) against
in-memory S3 stubs. No real boto3 calls.
"""
from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from regime.features import _PRICE_CACHE_NEW_PREFIX
from regime.handler import (
    DEFAULT_HMM_FEATURE_COLUMNS,
    produce_regime_substrate,
)
from regime.substrate import read_regime_substrate


pytest.importorskip("hmmlearn")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")


class _FakeS3:
    """In-memory S3 stub — supports put_object, get_object, parquet seeding."""

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}

    def put_parquet(self, bucket: str, key: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=True)
        self._objects[(bucket, key)] = buf.getvalue()

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None) -> dict:
        self._objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode("utf-8")
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self._objects:
            raise KeyError(f"no object at {Bucket}/{Key}")
        return {"Body": io.BytesIO(self._objects[(Bucket, Key)])}


def _seed_two_regime_price_cache(s3: _FakeS3, bucket: str, prefix: str) -> None:
    """Two-regime synthetic price cache: first half calm, second half
    stressed. Gives the HMM enough separation to discover meaningful
    states across the full fit window."""
    rng = np.random.default_rng(7)
    dates = pd.date_range(start="2024-01-01", periods=750, freq="B")
    half = len(dates) // 2
    cfg = {
        "SPY":   (lambda i: 400 + 0.05 * i + rng.normal(0, 2.0)),
        "VIX":   (lambda i: 14 if i < half else 28),
        "VIX3M": (lambda i: 16 if i < half else 26),
        "TNX":   (lambda i: 4.0 + rng.normal(0, 0.05)),
        "TWO":   (lambda i: 3.5 + rng.normal(0, 0.05)),
        "HYOAS": (lambda i: 3.5 if i < half else 6.0),  # percent
    }
    for ticker, fn in cfg.items():
        closes = [fn(i) + rng.normal(0, abs(fn(i)) * 0.005) for i in range(len(dates))]
        df = pd.DataFrame({"Close": closes}, index=dates)
        s3.put_parquet(bucket, f"{prefix}{ticker}.parquet", df)


def test_produce_writes_substrate_artifact_and_sidecar() -> None:
    s3 = _FakeS3()
    _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
    result = produce_regime_substrate(
        s3_client=s3, bucket="test-bucket", write=True,
    )
    assert result["wrote"] is True
    assert "artifact_key" in result and result["artifact_key"].startswith("regime/")
    assert result["latest_key"] == "regime/latest.json"
    # Round-trip: consumer-side read resolves latest.json → artifact
    payload = read_regime_substrate(s3_client=s3, bucket="test-bucket", prefix="regime")
    assert payload is not None
    assert payload["run_id"] == result["payload"]["run_id"]
    assert payload["hmm"]["argmax"] in {"bear", "neutral", "bull"}


def test_produce_dry_run_does_not_write() -> None:
    s3 = _FakeS3()
    _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
    result = produce_regime_substrate(
        s3_client=s3, bucket="test-bucket", write=False,
    )
    assert result["wrote"] is False
    # No regime artifact was written
    assert ("test-bucket", "regime/latest.json") not in s3._objects


def test_produce_hmm_probs_sum_to_one() -> None:
    s3 = _FakeS3()
    _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
    result = produce_regime_substrate(s3_client=s3, bucket="test-bucket", write=False)
    probs = result["payload"]["hmm"]["probs"]
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_produce_raises_when_no_price_cache() -> None:
    s3 = _FakeS3()  # no objects seeded
    with pytest.raises(RuntimeError, match="No macro feature history"):
        produce_regime_substrate(s3_client=s3, bucket="empty-bucket", write=False)


def test_produce_drops_unavailable_hmm_features_before_fit() -> None:
    """If an HMM feature column comes back all-NaN (e.g., breadth),
    the handler must drop it before calling fit() — otherwise the
    HMM will raise on the NaN column. Pins graceful-degradation."""
    s3 = _FakeS3()
    _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
    # Request market_breadth as an HMM feature even though no breadth
    # source is wired yet — handler should drop it from the fit set.
    result = produce_regime_substrate(
        s3_client=s3,
        bucket="test-bucket",
        write=False,
        hmm_feature_columns=("spy_20d_return", "vix_level", "market_breadth"),
    )
    # Fit succeeded — payload returned with valid HMM block
    assert result["payload"]["hmm"]["argmax"] in {"bear", "neutral", "bull"}
    # market_breadth was dropped from the HMM feature column list
    assert "market_breadth" not in result["payload"]["model_metadata"]["hmm_feature_columns"]


def test_produce_raises_when_pin_feature_unavailable() -> None:
    """spy_20d_return is required for HMM state pinning — handler must
    fail loudly when it's missing rather than silently producing a
    label-switching-prone fit. Without SPY, the entire history becomes
    empty (every row drops on the spy_20d_return-NaN dropna), so the
    handler trips the upstream-FRED-failure check before reaching the
    pin-feature check; either error is acceptable."""
    s3 = _FakeS3()
    _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
    # Remove SPY parquet so spy_20d_return goes all-NaN
    del s3._objects[("test-bucket", f"{_PRICE_CACHE_NEW_PREFIX}SPY.parquet")]
    with pytest.raises(RuntimeError, match="(spy_20d_return is required|No macro feature history)"):
        produce_regime_substrate(s3_client=s3, bucket="test-bucket", write=False)


def test_produce_fit_window_metadata_populated() -> None:
    """Fit-window dates are surfaced in model_metadata so the dashboard
    + forensic replay scripts can verify which observations went in."""
    s3 = _FakeS3()
    _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
    result = produce_regime_substrate(s3_client=s3, bucket="test-bucket", write=False)
    md = result["payload"]["model_metadata"]
    assert md["fit_window_start"] is not None
    assert md["fit_window_end"] is not None
    # End date should be reasonably recent (within the synthetic window)
    assert md["fit_window_end"] >= "2024-01-01"


def test_default_hmm_features_include_pin_feature() -> None:
    """spy_20d_return must be in DEFAULT_HMM_FEATURE_COLUMNS — otherwise
    every default-invocation Lambda call would fail the pin check."""
    assert "spy_20d_return" in DEFAULT_HMM_FEATURE_COLUMNS


def test_produce_assembles_weekly_drawdown_block() -> None:
    """The weekly handler now fills the additive drawdown block (PR:
    weekly-handler wiring) so the substrate the macro agent reads
    carries `drawdown` + the composed `effective_regime`."""
    s3 = _FakeS3()
    _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
    result = produce_regime_substrate(
        s3_client=s3, bucket="test-bucket", write=False,
    )
    payload = result["payload"]
    assert "drawdown" in payload
    assert payload["drawdown"]["spy"]["tier"] in {
        "risk_on", "caution", "risk_off",
    }
    # No eod_pnl.csv seeded ⇒ excess leg gracefully unavailable.
    assert payload["drawdown"]["excess"]["available"] is False
    eff = payload["effective_regime"]["effective_regime"]
    assert eff in {"bull", "neutral", "caution", "bear"}


def test_produce_substrate_survives_drawdown_block_failure(monkeypatch) -> None:
    """A drawdown-assembly failure must NOT block the substrate — the
    key is simply omitted (additive, S3-contract-safe)."""
    import regime.drawdown as dd

    s3 = _FakeS3()
    _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
    monkeypatch.setattr(
        dd, "block_from_history",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = produce_regime_substrate(
        s3_client=s3, bucket="test-bucket", write=False,
    )
    payload = result["payload"]
    assert "drawdown" not in payload          # omitted, not crashed
    assert "effective_regime" not in payload
    assert payload["hmm"]["argmax"] in {"bear", "neutral", "bull"}
