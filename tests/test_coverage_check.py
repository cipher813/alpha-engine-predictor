"""Unit tests for inference/coverage_check.py — Step Function coverage delta."""

import json
from unittest.mock import MagicMock, patch

import pytest

from inference.coverage_check import compute_coverage_delta


# ── Helpers ──────────────────────────────────────────────────────────────────

def _signals_json(buy_tickers: list[str]) -> bytes:
    return json.dumps({
        "date": "2026-04-20",
        "buy_candidates": [{"ticker": t, "signal": "ENTER"} for t in buy_tickers],
    }).encode()


def _predictions_json(pred_tickers: list[str]) -> bytes:
    return json.dumps({
        "date": "2026-04-20",
        "predictions": [{"ticker": t, "predicted_alpha": 0.01} for t in pred_tickers],
    }).encode()


def _mock_s3_client(key_to_payload: dict):
    """Build a boto3 mock whose get_object returns the payload for each key,
    or raises NoSuchKey if a key is mapped to None.
    """
    from botocore.exceptions import ClientError
    nosuchkey = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "not found"}}, "GetObject"
    )

    def get_object(Bucket, Key):
        if Key not in key_to_payload:
            raise nosuchkey
        payload = key_to_payload[Key]
        if payload is None:
            raise nosuchkey
        body = MagicMock()
        body.read.return_value = payload
        return {"Body": body}

    s3 = MagicMock()
    s3.get_object.side_effect = get_object
    boto3_mod = MagicMock()
    boto3_mod.client.return_value = s3
    return boto3_mod


# ── Happy path ───────────────────────────────────────────────────────────────

def test_no_gap_when_predictions_cover_all_buy_candidates():
    mock_boto3 = _mock_s3_client({
        "signals/2026-04-20/signals.json":              _signals_json(["AAPL", "MSFT"]),
        "predictor/predictions/2026-04-20.json":        _predictions_json(["AAPL", "MSFT", "GOOG"]),
    })
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = compute_coverage_delta("bucket", "2026-04-20")
    assert result["has_gap"] is False
    assert result["missing_count"] == 0
    assert result["missing_tickers"] == []
    assert result["n_buy_candidates"] == 2
    assert result["n_predictions"] == 3
    assert result["signals_present"] and result["predictions_present"]


def test_gap_detected_with_sorted_missing_list():
    # Today's bug reproduced
    mock_boto3 = _mock_s3_client({
        "signals/2026-04-20/signals.json": _signals_json(
            ["AAPL", "SNDK", "WDC", "BIIB", "XEL", "CTAS"]
        ),
        "predictor/predictions/2026-04-20.json": _predictions_json(["AAPL", "CTAS"]),
    })
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = compute_coverage_delta("bucket", "2026-04-20")
    assert result["has_gap"] is True
    assert result["missing_count"] == 4
    assert result["missing_tickers"] == ["BIIB", "SNDK", "WDC", "XEL"]  # sorted
    assert result["n_buy_candidates"] == 6
    assert result["n_predictions"] == 2


# ── Missing artifacts ────────────────────────────────────────────────────────

def test_no_predictions_file_means_all_buy_candidates_missing():
    # SF Choice state: if predictions.json doesn't exist, re-invoke with
    # tickers=buy_candidates (full set). has_gap should be True.
    mock_boto3 = _mock_s3_client({
        "signals/2026-04-20/signals.json": _signals_json(["AAPL", "MSFT"]),
        # predictions key intentionally absent
    })
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = compute_coverage_delta("bucket", "2026-04-20")
    assert result["has_gap"] is True
    assert result["missing_count"] == 2
    assert result["missing_tickers"] == ["AAPL", "MSFT"]
    assert result["predictions_present"] is False


def test_no_signals_file_is_not_a_gap():
    # If both date-specific AND latest signals paths are absent we can't
    # measure coverage. Return has_gap=False so the SF Choice state proceeds
    # to PredictorHealthCheck, which will catch the upstream problem (no
    # fresh signals). We don't want to double-alert.
    mock_boto3 = _mock_s3_client({
        "predictor/predictions/2026-04-20.json": _predictions_json(["AAPL"]),
        # both signals keys intentionally absent
    })
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = compute_coverage_delta("bucket", "2026-04-20")
    assert result["has_gap"] is False
    assert result["signals_present"] is False
    assert result["n_buy_candidates"] == 0


def test_falls_back_to_signals_latest_when_date_specific_missing():
    # 2026-04-23 regression: research writes signals/{SAT-date}/signals.json
    # once per week, so on weekdays the date-specific path is absent. Without
    # this fallback, check_coverage always returned 0 buy_candidates on
    # weekdays → SF self-heal never fired → executor hit coverage gap on its
    # own and halted. This test encodes the fix.
    mock_boto3 = _mock_s3_client({
        # date-specific signals path absent (weekday)
        "signals/latest.json": _signals_json(
            ["AAPL", "BIIB", "LLY", "ROST", "SNDK", "VLO", "WDC", "XEL"]
        ),
        "predictor/predictions/2026-04-23.json": _predictions_json(["AAPL"]),
    })
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = compute_coverage_delta("bucket", "2026-04-23")
    assert result["has_gap"] is True
    assert result["signals_present"] is True
    assert result["n_buy_candidates"] == 8
    assert result["missing_tickers"] == [
        "BIIB", "LLY", "ROST", "SNDK", "VLO", "WDC", "XEL",
    ]


def test_both_absent_is_no_gap():
    mock_boto3 = _mock_s3_client({})
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = compute_coverage_delta("bucket", "2026-04-20")
    assert result["has_gap"] is False
    assert result["missing_count"] == 0
    assert result["signals_present"] is False
    assert result["predictions_present"] is False


# ── Date defaulting ──────────────────────────────────────────────────────────

def test_date_defaults_to_today():
    # When date_str is None, today's date string is used. We verify by
    # observing the constructed S3 key includes a date shape — the actual
    # today's value depends on clock, so just check the result contains a date.
    mock_boto3 = _mock_s3_client({})  # returns empty state
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = compute_coverage_delta("bucket")
    assert result["date"]
    assert len(result["date"]) == 10  # YYYY-MM-DD
    assert result["date"][4] == "-" and result["date"][7] == "-"
