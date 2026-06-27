"""Tests for inference.handler — Lambda action dispatch."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def stubbed_preflight(monkeypatch):
    """Replace PredictorPreflight with a no-op so tests don't hit S3."""
    pf = MagicMock()
    pf.return_value.run = MagicMock()
    pf.return_value.run_for_drift_gate = MagicMock()
    fake_module = MagicMock(PredictorPreflight=pf)
    monkeypatch.setitem(sys.modules, "inference.preflight", fake_module)
    return pf


def _fake_context(function_name="local", arn=""):
    ctx = MagicMock()
    ctx.function_name = function_name
    ctx.invoked_function_arn = arn
    return ctx


# ── action="train" (deprecated) ─────────────────────────────────────────────


def test_handler_train_returns_400_deprecated(stubbed_preflight):
    import inference.handler as h
    result = h.handler({"action": "train"}, _fake_context())
    assert result["statusCode"] == 400
    assert "Training has moved" in result["body"]


# ── action="check_coverage" ─────────────────────────────────────────────────


def test_handler_check_coverage_returns_delta(stubbed_preflight, monkeypatch):
    fake_delta = {
        "n_buy_candidates": 25,
        "n_predictions": 20,
        "missing_count": 5,
        "missing_tickers": ["A", "B", "C", "D", "E"],
    }
    fake_cov = MagicMock()
    fake_cov.compute_coverage_delta = MagicMock(return_value=fake_delta)
    monkeypatch.setitem(sys.modules, "inference.coverage_check", fake_cov)

    import inference.handler as h
    result = h.handler({"action": "check_coverage", "date": "2026-04-15"}, _fake_context())

    assert result == fake_delta
    fake_cov.compute_coverage_delta.assert_called_once()


# ── action="check_drift" ────────────────────────────────────────────────────


def test_handler_check_drift_writes_artifact(stubbed_preflight, monkeypatch):
    """check_drift dispatches to monitoring.drift_detector.check_drift, returns
    its structured result, and uses the light env+S3 preflight only — NOT the
    full predict preflight and NOT the deploy-drift gate (config#1282)."""
    fake_result = {
        "date": "2026-06-26",
        "status": "ok",
        "severity": None,
        "alerts": [],
        "alert_details": [],
        "skipped_checks": [],
        "n_alerts": 0,
    }
    fake_dd = MagicMock()
    fake_dd.check_drift = MagicMock(return_value=fake_result)
    monkeypatch.setitem(sys.modules, "monitoring.drift_detector", fake_dd)

    import inference.handler as h
    result = h.handler({"action": "check_drift", "date": "2026-06-26"}, _fake_context())

    assert result == fake_result
    fake_dd.check_drift.assert_called_once()
    call_kwargs = fake_dd.check_drift.call_args.kwargs
    assert call_kwargs["date_str"] == "2026-06-26"
    # Light preflight: env + S3 only, no full predict bootstrap, no drift gate.
    stubbed_preflight.return_value.check_env_vars.assert_called_once_with("AWS_REGION")
    stubbed_preflight.return_value.check_s3_bucket.assert_called_once()
    stubbed_preflight.return_value.run.assert_not_called()
    stubbed_preflight.return_value.run_for_drift_gate.assert_not_called()


# ── action="check_deploy_drift" ─────────────────────────────────────────────


def test_handler_check_deploy_drift_uses_run_for_drift_gate(
    stubbed_preflight, monkeypatch,
):
    fake_drift = MagicMock()
    fake_drift.check_deploy_drift = MagicMock(return_value={
        "upstream_sha": "abc1234567",
        "sf_sha": "abc1234567",
        "sf_drift": False,
        "stack_sha": "abc1234567",
        "cf_drift": False,
    })
    monkeypatch.setitem(sys.modules, "inference.deploy_drift", fake_drift)

    import inference.handler as h
    ctx = _fake_context(arn="arn:aws:lambda:us-east-1:123456789012:function:foo")
    result = h.handler({"action": "check_deploy_drift"}, ctx)

    assert result["sf_drift"] is False
    fake_drift.check_deploy_drift.assert_called_once_with(
        region="us-east-1", account_id="123456789012",
    )
    # Drift gate should use the lighter preflight path
    stubbed_preflight.return_value.run_for_drift_gate.assert_called_once()
    stubbed_preflight.return_value.run.assert_not_called()


def test_handler_check_deploy_drift_falls_back_to_env_account(
    stubbed_preflight, monkeypatch,
):
    monkeypatch.setenv("AWS_ACCOUNT_ID", "999999999999")
    fake_drift = MagicMock()
    fake_drift.check_deploy_drift = MagicMock(return_value={
        "upstream_sha": "abc",
        "sf_sha": "abc",
        "sf_drift": False,
        "stack_sha": "abc",
        "cf_drift": False,
    })
    monkeypatch.setitem(sys.modules, "inference.deploy_drift", fake_drift)

    import inference.handler as h
    result = h.handler({"action": "check_deploy_drift"}, None)
    fake_drift.check_deploy_drift.assert_called_once_with(
        region="us-east-1", account_id="999999999999",
    )


# ── action="predict" (default) ─────────────────────────────────────────────


def test_handler_predict_default_path_invokes_main(
    stubbed_preflight, monkeypatch,
):
    fake_daily = MagicMock()
    fake_daily.main = MagicMock()
    monkeypatch.setitem(sys.modules, "inference.daily_predict", fake_daily)

    import inference.handler as h
    result = h.handler({"date": "2026-04-15"}, _fake_context())

    assert result["statusCode"] == 200
    assert "Predictions written for 2026-04-15" in result["body"]
    fake_daily.main.assert_called_once()
    call_kwargs = fake_daily.main.call_args.kwargs
    assert call_kwargs["date_str"] == "2026-04-15"
    assert call_kwargs["dry_run"] is False
    assert call_kwargs["local"] is False
    assert call_kwargs["explicit_tickers"] == []
    # Default predict path uses full preflight, drift-check enabled
    stubbed_preflight.return_value.run.assert_called_once()
    assert (
        stubbed_preflight.return_value.run.call_args.kwargs["skip_deploy_drift"]
        is False
    )


def test_handler_predict_supplemental_tickers_split_csv(
    stubbed_preflight, monkeypatch,
):
    fake_daily = MagicMock()
    fake_daily.main = MagicMock()
    monkeypatch.setitem(sys.modules, "inference.daily_predict", fake_daily)

    import inference.handler as h
    result = h.handler(
        {"action": "predict", "tickers": "aapl, msft , googl"},
        _fake_context(),
    )

    assert result["statusCode"] == 200
    assert "Supplemental predictions written" in result["body"]
    assert "3 tickers" in result["body"]
    call_kwargs = fake_daily.main.call_args.kwargs
    assert call_kwargs["explicit_tickers"] == ["AAPL", "MSFT", "GOOGL"]


def test_handler_predict_supplemental_tickers_list(
    stubbed_preflight, monkeypatch,
):
    fake_daily = MagicMock()
    fake_daily.main = MagicMock()
    monkeypatch.setitem(sys.modules, "inference.daily_predict", fake_daily)

    import inference.handler as h
    result = h.handler({"tickers": ["aapl", "msft"], "dry_run": True}, _fake_context())

    call_kwargs = fake_daily.main.call_args.kwargs
    assert call_kwargs["explicit_tickers"] == ["AAPL", "MSFT"]
    assert call_kwargs["dry_run"] is True
    # Canary (dry_run=true) must skip the deploy-drift check (config#1073)
    assert (
        stubbed_preflight.return_value.run.call_args.kwargs["skip_deploy_drift"]
        is True
    )


