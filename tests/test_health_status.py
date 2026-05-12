"""Tests for health_status — S3 health JSON write/read + upstream-staleness check."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from health_status import (
    check_upstream_health,
    read_health,
    write_data_manifest,
    write_health,
)


def _captured_put_object():
    """Return (s3_mock, capture_dict) where capture_dict gets populated on put."""
    capture = {}
    s3 = MagicMock()

    def fake_put(**kwargs):
        capture.update(kwargs)
        return {}

    s3.put_object.side_effect = fake_put
    return s3, capture


# ── write_health ───────────────────────────────────────────────────────────


def test_write_health_ok_writes_to_expected_key():
    s3, capture = _captured_put_object()
    with patch("health_status.boto3.client", return_value=s3):
        write_health(
            bucket="bucket-x", module_name="predictor", status="ok",
            run_date="2026-04-15", duration_seconds=12.345,
            summary={"n_predictions": 480},
        )

    assert capture["Bucket"] == "bucket-x"
    assert capture["Key"] == "health/predictor.json"
    assert capture["ContentType"] == "application/json"
    payload = json.loads(capture["Body"].decode())
    assert payload["module"] == "predictor"
    assert payload["status"] == "ok"
    assert payload["run_date"] == "2026-04-15"
    assert payload["duration_seconds"] == 12.3
    assert payload["summary"] == {"n_predictions": 480}
    assert payload["last_success"] is not None
    assert payload["warnings"] == []
    assert payload["error"] is None


def test_write_health_failed_sets_last_success_null():
    s3, capture = _captured_put_object()
    with patch("health_status.boto3.client", return_value=s3):
        write_health(
            bucket="bucket-x", module_name="predictor", status="failed",
            run_date="2026-04-15", duration_seconds=0.1,
            error="something blew up",
        )
    payload = json.loads(capture["Body"].decode())
    assert payload["status"] == "failed"
    assert payload["last_success"] is None
    assert payload["error"] == "something blew up"


def test_write_health_swallows_s3_failure_silently():
    s3 = MagicMock()
    s3.put_object.side_effect = RuntimeError("S3 down")
    with patch("health_status.boto3.client", return_value=s3):
        # Must not raise even though S3 is unavailable
        write_health(
            bucket="bucket-x", module_name="predictor", status="ok",
            run_date="2026-04-15", duration_seconds=1.0,
        )


# ── write_data_manifest ────────────────────────────────────────────────────


def test_write_data_manifest_writes_dated_key():
    s3, capture = _captured_put_object()
    with patch("health_status.boto3.client", return_value=s3):
        write_data_manifest(
            bucket="bucket-x", module_name="data", run_date="2026-04-15",
            manifest={"tickers_processed": 920, "ok_ratio": 0.985},
        )

    assert capture["Key"] == "data_manifest/data/2026-04-15.json"
    payload = json.loads(capture["Body"].decode())
    assert payload["module"] == "data"
    assert payload["run_date"] == "2026-04-15"
    assert payload["tickers_processed"] == 920
    assert payload["ok_ratio"] == 0.985
    assert "written_at" in payload


def test_write_data_manifest_swallows_s3_failure():
    s3 = MagicMock()
    s3.put_object.side_effect = RuntimeError("S3 down")
    with patch("health_status.boto3.client", return_value=s3):
        write_data_manifest("bucket-x", "data", "2026-04-15", {})


# ── read_health ────────────────────────────────────────────────────────────


def test_read_health_returns_payload():
    s3 = MagicMock()
    body = MagicMock()
    body.read.return_value = json.dumps({"status": "ok", "last_success": "x"}).encode()
    s3.get_object.return_value = {"Body": body}
    with patch("health_status.boto3.client", return_value=s3):
        result = read_health("bucket-x", "predictor")

    assert result == {"status": "ok", "last_success": "x"}
    s3.get_object.assert_called_once_with(Bucket="bucket-x", Key="health/predictor.json")


def test_read_health_missing_returns_none():
    s3 = MagicMock()
    s3.get_object.side_effect = RuntimeError("NoSuchKey")
    with patch("health_status.boto3.client", return_value=s3):
        assert read_health("bucket-x", "predictor") is None


# ── check_upstream_health ──────────────────────────────────────────────────


def test_check_upstream_health_all_unknown_when_missing(monkeypatch):
    monkeypatch.setattr("health_status.read_health", lambda b, m: None)
    result = check_upstream_health("bucket-x", ["a", "b"])
    for mod in ("a", "b"):
        assert result[mod]["status"] == "unknown"
        assert result[mod]["age_hours"] == -1
        assert result[mod]["stale"] is True


def test_check_upstream_health_fresh_module_not_stale(monkeypatch):
    fresh = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    monkeypatch.setattr(
        "health_status.read_health",
        lambda b, m: {"status": "ok", "last_success": fresh},
    )
    result = check_upstream_health("bucket-x", ["predictor"], max_age_hours=48)
    assert result["predictor"]["status"] == "ok"
    assert result["predictor"]["stale"] is False
    assert 1.5 < result["predictor"]["age_hours"] < 2.5


def test_check_upstream_health_stale_module(monkeypatch):
    stale = (datetime.now(timezone.utc) - timedelta(hours=100)).isoformat()
    monkeypatch.setattr(
        "health_status.read_health",
        lambda b, m: {"status": "ok", "last_success": stale},
    )
    result = check_upstream_health("bucket-x", ["data"], max_age_hours=48)
    assert result["data"]["stale"] is True
    assert result["data"]["age_hours"] > 48


def test_check_upstream_health_malformed_last_success(monkeypatch):
    """Malformed ISO date should be caught — age_hours stays -1, treated stale."""
    monkeypatch.setattr(
        "health_status.read_health",
        lambda b, m: {"status": "degraded", "last_success": "not-a-date"},
    )
    result = check_upstream_health("bucket-x", ["data"])
    assert result["data"]["age_hours"] == -1
    assert result["data"]["stale"] is True
    assert result["data"]["status"] == "degraded"


def test_check_upstream_health_no_last_success_field(monkeypatch):
    monkeypatch.setattr(
        "health_status.read_health",
        lambda b, m: {"status": "ok"},  # no last_success key
    )
    result = check_upstream_health("bucket-x", ["data"])
    assert result["data"]["age_hours"] == -1
    assert result["data"]["stale"] is True
