"""Tests for regime/retrospective_eval_handler.py — Lambda entry point
for the weekly T1 retrospective HMM smoothing eval.

Exercises the full pipeline (fetch macro features → load agent calls
from signals/ archive → build eval indices → smooth → pair → score →
write canonical artifact) against in-memory S3 stubs. No real boto3.
"""
from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from regime.features import _PRICE_CACHE_NEW_PREFIX
from regime.retrospective_eval_handler import (
    DEFAULT_RETROSPECTIVE_PREFIX,
    DEFAULT_SIGNALS_PREFIX,
    _build_eval_indices,
    load_agent_calls_history,
    produce_t1_eval,
    lambda_handler,
)


pytest.importorskip("hmmlearn")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")


class _FakeS3:
    """In-memory S3 stub — supports put_parquet, put_object, get_object,
    list_objects_v2 pagination."""

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}

    def put_parquet(self, bucket: str, key: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=True)
        self._objects[(bucket, key)] = buf.getvalue()

    def put_json(self, bucket: str, key: str, payload: dict) -> None:
        self._objects[(bucket, key)] = json.dumps(payload).encode("utf-8")

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None) -> dict:
        self._objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode("utf-8")
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self._objects:
            from botocore.exceptions import ClientError
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
                "GetObject",
            )
        return {"Body": io.BytesIO(self._objects[(Bucket, Key)])}

    def get_paginator(self, name: str):
        if name != "list_objects_v2":
            raise NotImplementedError(name)

        class _Paginator:
            def __init__(self, parent):
                self.parent = parent

            def paginate(self, *, Bucket: str, Prefix: str):
                contents = []
                for (b, k) in self.parent._objects.keys():
                    if b == Bucket and k.startswith(Prefix):
                        contents.append({"Key": k})
                yield {"Contents": contents}

        return _Paginator(self)


def _seed_two_regime_price_cache(s3: _FakeS3, bucket: str, prefix: str) -> None:
    """Two-regime synthetic price cache: first half calm, second half
    stressed. Mirrors test_regime_handler.py but extends to 2500 daily
    rows (~500 weekly) so the HMM fit clears its 60-observation minimum
    after resampling + dropna under varied as_of cutoffs."""
    rng = np.random.default_rng(7)
    dates = pd.date_range(start="2020-01-01", periods=2500, freq="B")
    half = len(dates) // 2
    cfg = {
        "SPY":   (lambda i: 400 + 0.05 * i + rng.normal(0, 2.0)),
        "VIX":   (lambda i: 14 if i < half else 28),
        "VIX3M": (lambda i: 16 if i < half else 26),
        "TNX":   (lambda i: 4.0 + rng.normal(0, 0.05)),
        "TWO":   (lambda i: 3.5 + rng.normal(0, 0.05)),
        "HYOAS": (lambda i: 3.5 if i < half else 6.0),
    }
    for ticker, fn in cfg.items():
        closes = [fn(i) + rng.normal(0, abs(fn(i)) * 0.005) for i in range(len(dates))]
        df = pd.DataFrame({"Close": closes}, index=dates)
        s3.put_parquet(bucket, f"{prefix}{ticker}.parquet", df)


def _seed_signals_archive(
    s3: _FakeS3,
    bucket: str,
    dates: list[pd.Timestamp],
    regimes: list[str],
) -> None:
    """Seed signals/{YYYY-MM-DD}/signals.json artifacts with the
    supplied market_regime per Saturday."""
    assert len(dates) == len(regimes)
    for d, r in zip(dates, regimes):
        key = f"signals/{d.date().isoformat()}/signals.json"
        s3.put_json(bucket, key, {
            "date": d.date().isoformat(),
            "market_regime": r,
            "universe": [],
            "buy_candidates": [],
            "sector_ratings": {},
            "architecture_version": "sector_teams",
        })


# ─────────────────────────────────────────────────────────────────────
# load_agent_calls_history
# ─────────────────────────────────────────────────────────────────────


class TestLoadAgentCallsHistory:
    def test_empty_archive_returns_empty_frame(self):
        s3 = _FakeS3()
        result = load_agent_calls_history(s3, bucket="test-bucket")
        assert result.empty
        assert list(result.columns) == ["trading_day", "market_regime"]

    def test_extracts_market_regime_per_saturday(self):
        s3 = _FakeS3()
        sats = pd.date_range("2026-01-03", periods=4, freq="W-SAT")
        _seed_signals_archive(s3, "test-bucket", list(sats), ["bull", "neutral", "bear", "caution"])
        result = load_agent_calls_history(
            s3, bucket="test-bucket",
            as_of=pd.Timestamp("2026-02-01"),
            window_weeks=10,
        )
        assert len(result) == 4
        assert set(result["market_regime"]) == {"bull", "neutral", "bear", "caution"}

    def test_window_filters_older_calls(self):
        s3 = _FakeS3()
        sats = pd.date_range("2024-01-06", periods=20, freq="W-SAT")
        _seed_signals_archive(s3, "test-bucket", list(sats), ["bull"] * 20)
        # 5-week window from a date inside the seeded range
        result = load_agent_calls_history(
            s3, bucket="test-bucket",
            as_of=pd.Timestamp("2024-05-20"),
            window_weeks=5,
        )
        # Older calls filtered out; recent retained
        assert len(result) < 20
        assert (result["trading_day"] >= pd.Timestamp("2024-04-15")).all()

    def test_malformed_signals_json_is_skipped(self):
        s3 = _FakeS3()
        sats = pd.date_range("2026-01-03", periods=3, freq="W-SAT")
        _seed_signals_archive(s3, "test-bucket", list(sats[:2]), ["bull", "neutral"])
        # Inject a malformed signals.json for the 3rd Saturday
        s3._objects[("test-bucket", f"signals/{sats[2].date()}/signals.json")] = b"NOT JSON"
        result = load_agent_calls_history(
            s3, bucket="test-bucket",
            as_of=pd.Timestamp("2026-02-01"),
            window_weeks=10,
        )
        # Good entries preserved; bad one skipped
        assert len(result) == 2

    def test_signals_without_market_regime_field_skipped(self):
        s3 = _FakeS3()
        # Seed a valid signals.json missing the market_regime field
        s3.put_json("test-bucket", "signals/2026-01-03/signals.json", {
            "date": "2026-01-03",
            "universe": [],
        })
        # Plus a valid one
        s3.put_json("test-bucket", "signals/2026-01-10/signals.json", {
            "date": "2026-01-10",
            "market_regime": "bull",
        })
        result = load_agent_calls_history(
            s3, bucket="test-bucket",
            as_of=pd.Timestamp("2026-02-01"),
            window_weeks=10,
        )
        assert len(result) == 1
        assert result.iloc[0]["market_regime"] == "bull"

    def test_skips_keys_not_matching_pattern(self):
        s3 = _FakeS3()
        # Real signals
        s3.put_json("test-bucket", "signals/2026-01-03/signals.json", {
            "date": "2026-01-03", "market_regime": "bull",
        })
        # Garbage under signals/ prefix
        s3._objects[("test-bucket", "signals/random.txt")] = b"x"
        s3._objects[("test-bucket", "signals/2026-01-10/something_else.json")] = b"{}"
        result = load_agent_calls_history(
            s3, bucket="test-bucket",
            as_of=pd.Timestamp("2026-02-01"),
            window_weeks=10,
        )
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────
# _build_eval_indices
# ─────────────────────────────────────────────────────────────────────


class TestBuildEvalIndices:
    def _panel(self) -> pd.DataFrame:
        idx = pd.date_range("2024-01-05", periods=20, freq="W-FRI")
        return pd.DataFrame({
            "spy_20d_return": np.linspace(-0.1, 0.1, 20),
            "vix_level": np.linspace(10, 30, 20),
        }, index=idx)

    def test_empty_panel_returns_empty(self):
        empty = pd.DataFrame()
        date_map, indices = _build_eval_indices(
            empty, ["spy_20d_return"], [pd.Timestamp("2024-01-05")],
            lag_weeks=8,
        )
        assert date_map == {}
        assert indices == []

    def test_panel_too_short_for_lag_returns_empty(self):
        panel = self._panel()
        # lag=25 exceeds panel length 20 → no valid positions
        date_map, indices = _build_eval_indices(
            panel, ["spy_20d_return"],
            [pd.Timestamp("2024-01-05")],
            lag_weeks=25,
        )
        assert indices == []

    def test_lag_filter_excludes_recent_calls(self):
        """Agent calls within lag_weeks of panel tail are excluded —
        smoother needs future observations to refine the posterior."""
        panel = self._panel()
        # Panel ends ~2024-05-17 (week 20); call at week 18 (within 2-week lag)
        recent_call = panel.index[18]
        date_map, indices = _build_eval_indices(
            panel, ["spy_20d_return"], [recent_call], lag_weeks=8,
        )
        assert indices == []

    def test_valid_call_maps_to_position(self):
        panel = self._panel()
        call = panel.index[5]  # week 5, comfortably before lag tail
        date_map, indices = _build_eval_indices(
            panel, ["spy_20d_return"], [call], lag_weeks=8,
        )
        assert indices == [5]
        assert date_map[call.normalize()] == 5

    def test_call_between_panel_rows_falls_back_to_prior(self):
        """Saturday agent call should map to the prior Friday panel row."""
        panel = self._panel()  # Fridays
        friday = panel.index[5]
        saturday = friday + pd.Timedelta(days=1)
        date_map, indices = _build_eval_indices(
            panel, ["spy_20d_return"], [saturday], lag_weeks=8,
        )
        assert indices == [5]

    def test_call_before_panel_start_skipped(self):
        panel = self._panel()
        too_early = panel.index[0] - pd.Timedelta(weeks=5)
        date_map, indices = _build_eval_indices(
            panel, ["spy_20d_return"], [too_early], lag_weeks=8,
        )
        assert indices == []


# ─────────────────────────────────────────────────────────────────────
# produce_t1_eval — end-to-end
# ─────────────────────────────────────────────────────────────────────


class TestProduceT1Eval:
    def _setup(self):
        s3 = _FakeS3()
        _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
        # Seed agent calls spanning the calm + stress halves
        sats = pd.date_range("2024-01-06", periods=40, freq="W-SAT")
        # Match the synthetic price-cache regime structure: first 20
        # calls neutral, last 20 bear (matching the stressed half)
        regimes = ["neutral"] * 20 + ["bear"] * 20
        _seed_signals_archive(s3, "test-bucket", list(sats), regimes)
        return s3, sats

    def test_produces_canonical_eval_artifact(self):
        s3, sats = self._setup()
        result = produce_t1_eval(
            s3_client=s3,
            bucket="test-bucket",
            as_of=sats[-1] + pd.Timedelta(weeks=10),
            agent_calls_window_weeks=60,
            lag_weeks=4,
        )
        assert result["wrote"] is True
        assert result["artifact_key"].startswith("regime/retrospective/")
        assert result["artifact_key"].endswith(".json")
        assert result["latest_key"] == "regime/retrospective/latest.json"

    def test_payload_contains_t1_schema_fields(self):
        s3, sats = self._setup()
        result = produce_t1_eval(
            s3_client=s3,
            bucket="test-bucket",
            as_of=sats[-1] + pd.Timedelta(weeks=10),
            agent_calls_window_weeks=60,
            lag_weeks=4,
        )
        payload = result["payload"]
        assert payload["schema_version"] == 1
        assert payload["eval_tier"] == "T1_retrospective_hmm_smoothing"
        assert payload["lag_weeks"] == 4
        assert "score" in payload
        assert "pairings" in payload
        assert "model_metadata" in payload
        score = payload["score"]
        assert "asymmetric_weighted_agreement_rate" in score
        assert "symmetric_agreement_rate" in score
        assert "rolling_window_score" in score
        assert score["n_pairings"] > 0

    def test_dry_run_skips_write(self):
        s3, sats = self._setup()
        result = produce_t1_eval(
            s3_client=s3,
            bucket="test-bucket",
            as_of=sats[-1] + pd.Timedelta(weeks=10),
            agent_calls_window_weeks=60,
            lag_weeks=4,
            write=False,
        )
        assert result["wrote"] is False
        # No artifact written to S3
        assert ("test-bucket", "regime/retrospective/latest.json") not in s3._objects

    def test_latest_sidecar_carries_summary_fields(self):
        s3, sats = self._setup()
        result = produce_t1_eval(
            s3_client=s3,
            bucket="test-bucket",
            as_of=sats[-1] + pd.Timedelta(weeks=10),
            agent_calls_window_weeks=60,
            lag_weeks=4,
        )
        sidecar_bytes = s3._objects[("test-bucket", "regime/retrospective/latest.json")]
        sidecar = json.loads(sidecar_bytes.decode("utf-8"))
        assert sidecar["run_id"] == result["payload"]["run_id"]
        assert sidecar["artifact_key"] == result["artifact_key"]
        assert sidecar["eval_tier"] == "T1_retrospective_hmm_smoothing"
        assert sidecar["lag_weeks"] == 4
        assert "n_pairings" in sidecar
        assert "asymmetric_weighted_agreement_rate" in sidecar

    def test_cold_start_no_agent_calls_emits_empty_score(self):
        """Lambda runs on a fresh bucket with no signals/ archive yet.
        Should not crash; should write artifact with n_pairings=0."""
        s3 = _FakeS3()
        _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
        # No agent-call archive seeded
        result = produce_t1_eval(
            s3_client=s3,
            bucket="test-bucket",
            as_of=pd.Timestamp("2026-01-01"),
        )
        assert result["wrote"] is True
        assert result["payload"]["score"]["n_pairings"] == 0
        assert result["payload"]["pairings"] == []

    def test_missing_spy_raises(self):
        """spy_20d_return is required for HMM state pinning."""
        s3 = _FakeS3()
        # Seed only VIX (no SPY)
        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=500, freq="B")
        s3.put_parquet("test-bucket", f"{_PRICE_CACHE_NEW_PREFIX}VIX.parquet",
                       pd.DataFrame({"Close": rng.normal(15, 3, 500)}, index=dates))
        with pytest.raises(RuntimeError):
            produce_t1_eval(s3_client=s3, bucket="test-bucket")


# ─────────────────────────────────────────────────────────────────────
# lambda_handler — entry point
# ─────────────────────────────────────────────────────────────────────


class TestLambdaHandler:
    def test_unknown_action_returns_400(self, monkeypatch):
        result = lambda_handler({"action": "nonsense"}, None)
        assert result["statusCode"] == 400
        assert "unknown action" in result["error"]

    def test_produce_action_response_shape(self, monkeypatch):
        """Patch boto3 to return our fake S3, verify the produce path
        returns the expected response keys."""
        s3 = _FakeS3()
        _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
        sats = pd.date_range("2024-01-06", periods=20, freq="W-SAT")
        _seed_signals_archive(s3, "test-bucket", list(sats), ["bull"] * 20)

        # Patch boto3.client to return our fake
        import boto3 as _boto3
        original = _boto3.client
        monkeypatch.setattr(_boto3, "client", lambda *_, **__: s3)
        try:
            result = lambda_handler({
                "action": "produce",
                "bucket": "test-bucket",
                "lag_weeks": 4,
                "agent_calls_window_weeks": 30,
            }, None)
        finally:
            monkeypatch.setattr(_boto3, "client", original)

        assert result["statusCode"] == 200
        assert result["action"] == "produce"
        assert "run_id" in result
        assert "artifact_key" in result
        assert "latest_key" in result
        assert "n_pairings" in result
        assert "asymmetric_weighted_agreement_rate" in result
        assert "rolling_window_score" in result

    def test_dry_run_action_response_shape(self, monkeypatch):
        s3 = _FakeS3()
        _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)

        import boto3 as _boto3
        monkeypatch.setattr(_boto3, "client", lambda *_, **__: s3)
        result = lambda_handler({
            "action": "dry_run",
            "bucket": "test-bucket",
            "lag_weeks": 4,
        }, None)
        assert result["statusCode"] == 200
        assert result["action"] == "dry_run"
        assert "payload" in result
        # No write happened
        assert ("test-bucket", "regime/retrospective/latest.json") not in s3._objects

    def test_constants(self):
        """Pin the S3 prefix conventions — Lambda CMD targets land
        the artifact under this prefix; SF state + dashboard reader
        depend on it."""
        assert DEFAULT_RETROSPECTIVE_PREFIX == "regime/retrospective"
        assert DEFAULT_SIGNALS_PREFIX == "signals/"
