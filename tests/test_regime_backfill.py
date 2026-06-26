"""Tests for scripts/backfill_regime_substrate.py — historical backfill.

Covers:
- Saturday generation across a date range
- Run-ID encoding (YYMMDDHHMM, 02:00 UTC anchor)
- HeadObject idempotency check (skip existing unless --force)
- Single-Saturday processing happy path + per-date error capture
- Range driver: status summary + non-blocking failures

Also pins the substrate writer's ``write_latest`` flag — backfill MUST
NOT clobber the live ``regime/latest.json`` sidecar.
"""
from __future__ import annotations

import io
import json
from datetime import date as Date
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("hmmlearn")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")


from regime.features import DEFAULT_PRICE_CACHE_PREFIX, _PRICE_CACHE_NEW_PREFIX
from regime.substrate import DEFAULT_S3_BUCKET, DEFAULT_S3_PREFIX, write_regime_substrate

# Import script as a module (it has a `if __name__ == "__main__"` guard).
import scripts.backfill_regime_substrate as bf


class _FakeS3:
    """In-memory S3 stub — supports put_object, head_object, get_object,
    parquet seeding."""

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

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self._objects:
            raise Exception("NoSuchKey")
        return {}


def _seed_two_regime_price_cache(s3: _FakeS3, bucket: str, prefix: str) -> None:
    """Two-regime synthetic price cache with calm + stressed halves."""
    rng = np.random.default_rng(7)
    dates = pd.date_range(start="2020-01-01", periods=1500, freq="B")
    half = len(dates) // 2
    cfg = {
        "SPY":   (lambda i: 350 + 0.05 * i + rng.normal(0, 2.0)),
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


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

class TestSaturdayGeneration:
    def test_generates_all_saturdays_in_range(self) -> None:
        sats = bf._generate_saturdays(Date(2024, 1, 1), Date(2024, 1, 31))
        # 2024-01-06, 13, 20, 27 — Saturdays in January 2024
        assert sats == [Date(2024, 1, 6), Date(2024, 1, 13), Date(2024, 1, 20), Date(2024, 1, 27)]

    def test_empty_when_range_inverted(self) -> None:
        assert bf._generate_saturdays(Date(2024, 6, 1), Date(2024, 1, 1)) == []

    def test_single_saturday_when_range_is_one_week(self) -> None:
        sats = bf._generate_saturdays(Date(2024, 1, 6), Date(2024, 1, 6))
        assert sats == [Date(2024, 1, 6)]


class TestRunIdEncoding:
    def test_encodes_saturday_at_02_00_utc(self) -> None:
        rid = bf._saturday_run_id(Date(2024, 5, 11))
        assert rid == "2405110200"
        assert len(rid) == 10

    def test_lexicographic_sort_matches_chronological(self) -> None:
        """YYMMDDHHMM must sort chronologically for the dashboard's
        history loader (which lex-sorts run_ids)."""
        rids = [
            bf._saturday_run_id(Date(2020, 1, 4)),
            bf._saturday_run_id(Date(2024, 5, 11)),
            bf._saturday_run_id(Date(2022, 6, 18)),
        ]
        assert sorted(rids) == [
            bf._saturday_run_id(Date(2020, 1, 4)),
            bf._saturday_run_id(Date(2022, 6, 18)),
            bf._saturday_run_id(Date(2024, 5, 11)),
        ]


# ---------------------------------------------------------------------------
# Writer write_latest flag — pins backfill safety
# ---------------------------------------------------------------------------

class TestWriterRespectsWriteLatest:
    def _make_payload(self, run_id: str = "2401060200") -> dict:
        return {
            "calendar_date": "2024-01-06",
            "trading_day": "2024-01-05",
            "run_id": run_id,
            "schema_version": 1,
            "hmm": {"argmax": "neutral", "probs": {"bear": 0.2, "neutral": 0.6, "bull": 0.2}},
            "composite": {"intensity_z": 0.1, "per_feature_z": {}, "features_used": [], "implied_severity": "neutral"},
            "bocpd": {"change_signal": False, "max_runlength_prob": 0.8, "change_confidence": 0.05},
            "guardrails": {},
            "features": {},
            "model_metadata": {},
        }

    def test_write_latest_true_writes_both_keys(self) -> None:
        s3 = _FakeS3()
        keys = write_regime_substrate(
            self._make_payload(), s3_client=s3, bucket="test-bucket", prefix="regime",
            write_latest=True,
        )
        assert "artifact_key" in keys
        assert "latest_key" in keys
        # Both objects present
        assert ("test-bucket", "regime/2401060200.json") in s3._objects
        assert ("test-bucket", "regime/latest.json") in s3._objects

    def test_write_latest_false_skips_sidecar(self) -> None:
        """Backfill must never clobber the live latest.json."""
        s3 = _FakeS3()
        keys = write_regime_substrate(
            self._make_payload(), s3_client=s3, bucket="test-bucket", prefix="regime",
            write_latest=False,
        )
        assert "artifact_key" in keys
        assert "latest_key" not in keys
        assert ("test-bucket", "regime/2401060200.json") in s3._objects
        assert ("test-bucket", "regime/latest.json") not in s3._objects


# ---------------------------------------------------------------------------
# Idempotency probe
# ---------------------------------------------------------------------------

class TestArtifactExists:
    def test_returns_true_when_object_present(self) -> None:
        s3 = _FakeS3()
        s3.put_object(Bucket="b", Key="regime/2401060200.json", Body=b"{}")
        assert bf._artifact_exists(s3, "b", "regime", "2401060200") is True

    def test_returns_false_when_object_missing(self) -> None:
        s3 = _FakeS3()
        assert bf._artifact_exists(s3, "b", "regime", "2401060200") is False


# ---------------------------------------------------------------------------
# Single-Saturday processing
# ---------------------------------------------------------------------------

class TestBackfillOne:
    def test_writes_artifact_for_new_saturday(self) -> None:
        s3 = _FakeS3()
        _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
        result = bf.backfill_one(
            saturday=Date(2024, 5, 11),
            s3_client=s3,
            bucket="test-bucket",
            price_cache_prefix=DEFAULT_PRICE_CACHE_PREFIX,
            regime_prefix="regime",
            hmm_feature_columns=("spy_20d_return", "vix_level", "hy_oas_bps"),
            force=False, dry_run=False,
        )
        assert result["status"] == "wrote"
        assert result["run_id"] == "2405110200"
        assert ("test-bucket", "regime/2405110200.json") in s3._objects

    def test_skips_when_artifact_exists(self) -> None:
        s3 = _FakeS3()
        _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
        # Pre-seed the artifact
        s3.put_object(Bucket="test-bucket", Key="regime/2405110200.json", Body=b"existing")
        result = bf.backfill_one(
            saturday=Date(2024, 5, 11),
            s3_client=s3,
            bucket="test-bucket",
            price_cache_prefix=DEFAULT_PRICE_CACHE_PREFIX,
            regime_prefix="regime",
            hmm_feature_columns=("spy_20d_return", "vix_level", "hy_oas_bps"),
            force=False, dry_run=False,
        )
        assert result["status"] == "skipped_existing"
        # Existing artifact untouched
        assert s3._objects[("test-bucket", "regime/2405110200.json")] == b"existing"

    def test_force_overwrites_existing(self) -> None:
        s3 = _FakeS3()
        _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
        s3.put_object(Bucket="test-bucket", Key="regime/2405110200.json", Body=b"stale")
        result = bf.backfill_one(
            saturday=Date(2024, 5, 11),
            s3_client=s3,
            bucket="test-bucket",
            price_cache_prefix=DEFAULT_PRICE_CACHE_PREFIX,
            regime_prefix="regime",
            hmm_feature_columns=("spy_20d_return", "vix_level", "hy_oas_bps"),
            force=True, dry_run=False,
        )
        assert result["status"] == "wrote"
        # Was overwritten — payload is now valid JSON
        body = s3._objects[("test-bucket", "regime/2405110200.json")]
        payload = json.loads(body)
        assert payload["run_id"] == "2405110200"

    def test_dry_run_does_not_write(self) -> None:
        s3 = _FakeS3()
        _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
        result = bf.backfill_one(
            saturday=Date(2024, 5, 11),
            s3_client=s3,
            bucket="test-bucket",
            price_cache_prefix=DEFAULT_PRICE_CACHE_PREFIX,
            regime_prefix="regime",
            hmm_feature_columns=("spy_20d_return", "vix_level", "hy_oas_bps"),
            force=False, dry_run=True,
        )
        assert result["status"] == "would_write"
        assert ("test-bucket", "regime/2405110200.json") not in s3._objects

    def test_does_not_touch_latest_json(self) -> None:
        """The Stage A safety contract — backfill MUST NEVER clobber the
        live latest.json sidecar pointer. Pinned at the script layer too,
        not just the writer layer, so a refactor that bypasses the
        writer-side flag still can't drift."""
        s3 = _FakeS3()
        _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
        # Pre-seed a known latest.json sentinel
        sentinel = b'{"run_id": "LIVE_DO_NOT_OVERWRITE"}'
        s3.put_object(Bucket="test-bucket", Key="regime/latest.json", Body=sentinel)

        bf.backfill_one(
            saturday=Date(2024, 5, 11),
            s3_client=s3,
            bucket="test-bucket",
            price_cache_prefix=DEFAULT_PRICE_CACHE_PREFIX,
            regime_prefix="regime",
            hmm_feature_columns=("spy_20d_return", "vix_level", "hy_oas_bps"),
            force=False, dry_run=False,
        )

        # latest.json is untouched
        assert s3._objects[("test-bucket", "regime/latest.json")] == sentinel

    def test_error_captured_not_raised(self) -> None:
        """A bad date that can't be processed must return status=error,
        not blow up the whole backfill driver."""
        s3 = _FakeS3()  # empty — no price cache
        result = bf.backfill_one(
            saturday=Date(2024, 5, 11),
            s3_client=s3,
            bucket="test-bucket",
            price_cache_prefix=DEFAULT_PRICE_CACHE_PREFIX,
            regime_prefix="regime",
            hmm_feature_columns=("spy_20d_return", "vix_level", "hy_oas_bps"),
            force=False, dry_run=False,
        )
        assert result["status"] == "error"
        assert "error_type" in result and "error_message" in result


# ---------------------------------------------------------------------------
# Range driver
# ---------------------------------------------------------------------------

class TestRunBackfill:
    def test_processes_all_saturdays_in_range(self) -> None:
        s3 = _FakeS3()
        _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)
        results = bf.run_backfill(
            start=Date(2024, 5, 1),
            end=Date(2024, 5, 31),
            s3_client=s3,
            bucket="test-bucket",
            price_cache_prefix=DEFAULT_PRICE_CACHE_PREFIX,
            regime_prefix="regime",
            hmm_feature_columns=("spy_20d_return", "vix_level", "hy_oas_bps"),
        )
        # May 2024 Saturdays: 4, 11, 18, 25
        assert len(results) == 4
        statuses = [r["status"] for r in results]
        assert all(s == "wrote" for s in statuses), statuses

    def test_continues_past_per_date_failures(self) -> None:
        """One bad date must not abort the rest. We patch backfill_one
        to fail on a specific date and verify the others still process."""
        s3 = _FakeS3()
        _seed_two_regime_price_cache(s3, "test-bucket", _PRICE_CACHE_NEW_PREFIX)

        real_backfill_one = bf.backfill_one

        def _wrapped(*, saturday, **kwargs):
            if saturday == Date(2024, 5, 11):
                return {"saturday": saturday.isoformat(), "run_id": "2405110200", "status": "error",
                        "error_type": "Synthetic", "error_message": "test"}
            return real_backfill_one(saturday=saturday, **kwargs)

        with patch.object(bf, "backfill_one", side_effect=_wrapped):
            results = bf.run_backfill(
                start=Date(2024, 5, 1),
                end=Date(2024, 5, 31),
                s3_client=s3,
                bucket="test-bucket",
                price_cache_prefix=DEFAULT_PRICE_CACHE_PREFIX,
                regime_prefix="regime",
                hmm_feature_columns=("spy_20d_return", "vix_level", "hy_oas_bps"),
            )
        statuses = [r["status"] for r in results]
        assert statuses.count("wrote") == 3
        assert statuses.count("error") == 1

    def test_dry_run_returns_would_write_for_every_saturday(self) -> None:
        s3 = _FakeS3()
        results = bf.run_backfill(
            start=Date(2024, 5, 1), end=Date(2024, 5, 31),
            s3_client=s3, bucket="test-bucket",
            dry_run=True,
        )
        assert len(results) == 4
        assert all(r["status"] == "would_write" for r in results)
        # No artifacts written
        assert not any("regime/" in k[1] for k in s3._objects)
