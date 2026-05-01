"""
Tests for PredictorPreflight.run() mode composition.

BasePreflight primitives are tested in alpha-engine-lib. The deploy-drift
check has its own test file (test_preflight_drift.py). These tests verify
that ``run()`` composes the expected primitive calls in the expected order
— and that ``run_for_drift_gate()`` is a strict subset for the SF gate.

Universe-freshness now reads the producer-side receipt
(s3://{bucket}/health/universe_freshness.json) emitted by
alpha-engine-data/builders/daily_append.py instead of running a 200s
~900-ticker scan on every Lambda invocation. See alpha-engine-data #119.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.preflight import PredictorPreflight


def _make() -> PredictorPreflight:
    return PredictorPreflight(bucket="test-bucket")


def test_run_composes_full_check_sequence():
    """Verify the full primitive call sequence: env + S3 + drift +
    weights + 5 macro symbols + universe-freshness receipt read."""
    pf = _make()
    with patch.object(pf, "check_env_vars") as env, \
         patch.object(pf, "check_s3_bucket") as s3, \
         patch.object(pf, "check_deploy_drift") as drift, \
         patch.object(pf, "check_s3_key") as s3_key, \
         patch.object(pf, "check_arcticdb_fresh") as fresh, \
         patch.object(pf, "check_universe_freshness_receipt") as universe:
        pf.run()

    env.assert_called_once_with("AWS_REGION")
    s3.assert_called_once()
    drift.assert_called_once()
    s3_key.assert_called_once()
    assert s3_key.call_args.args[0] == "predictor/weights/meta/meta_model.pkl"

    # 5 macro symbols
    assert fresh.call_count == 5
    macro_symbols = [call.args[1] for call in fresh.call_args_list]
    assert macro_symbols == ["SPY", "VIX", "VIX3M", "TNX", "IRX"]
    for call in fresh.call_args_list:
        assert call.args[0] == "macro"

    # Universe-freshness via receipt read (NOT the per-ticker scan)
    universe.assert_called_once()


def test_run_for_drift_gate_is_strict_subset():
    """The drift gate runs ONLY env + S3 + image-SHA. No model weights,
    no macro reads, no universe work — those are predict-path concerns
    and running them on the gate caused the 2026-05-01 SF timeout
    cascade."""
    pf = _make()
    with patch.object(pf, "check_env_vars") as env, \
         patch.object(pf, "check_s3_bucket") as s3, \
         patch.object(pf, "check_deploy_drift") as drift, \
         patch.object(pf, "check_s3_key") as s3_key, \
         patch.object(pf, "check_arcticdb_fresh") as fresh, \
         patch.object(pf, "check_universe_freshness_receipt") as universe:
        pf.run_for_drift_gate()

    env.assert_called_once_with("AWS_REGION")
    s3.assert_called_once()
    drift.assert_called_once()
    s3_key.assert_not_called()
    fresh.assert_not_called()
    universe.assert_not_called()


def test_macro_freshness_threshold_is_4_days():
    """Macro symbols use the canonical 4-day threshold (covers Fri→Tue
    long weekends + 1d buffer)."""
    pf = _make()
    with patch.object(pf, "check_env_vars"), \
         patch.object(pf, "check_s3_bucket"), \
         patch.object(pf, "check_deploy_drift"), \
         patch.object(pf, "check_s3_key"), \
         patch.object(pf, "check_arcticdb_fresh") as fresh, \
         patch.object(pf, "check_universe_freshness_receipt"):
        pf.run()

    for call in fresh.call_args_list:
        assert call.kwargs.get("max_stale_days") == 4


def test_universe_receipt_check_uses_max_age_1_day():
    """Receipt must be ≤1 day old. A receipt older than that means the
    producer hasn't run today and the consumer can't trust the data."""
    pf = _make()
    with patch.object(pf, "check_s3_key") as s3_key, \
         patch("inference.preflight._json.loads", return_value={"all_fresh": True}) if False else patch("boto3.client") as boto_client:
        boto_client.return_value.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({"all_fresh": True}).encode())
        }
        pf.check_universe_freshness_receipt()

    s3_key.assert_called_once()
    assert s3_key.call_args.args[0] == "health/universe_freshness.json"
    assert s3_key.call_args.kwargs.get("max_age_days") == 1


def test_universe_receipt_raises_when_all_fresh_false():
    """Defensive: if the producer somehow wrote a receipt with all_fresh
    other than True, the consumer must not silently proceed. Producer
    contract is "only written on success" but defense in depth."""
    import pytest

    pf = _make()
    with patch.object(pf, "check_s3_key"), \
         patch("boto3.client") as boto_client:
        boto_client.return_value.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({"all_fresh": False, "stale_symbols": ["AAPL"]}).encode())
        }
        with pytest.raises(RuntimeError, match="all_fresh"):
            pf.check_universe_freshness_receipt()


def test_first_failure_short_circuits():
    """If env vars fail, no S3 / ArcticDB / GitHub calls happen."""
    import pytest

    pf = _make()
    with patch.object(
        pf, "check_env_vars", side_effect=RuntimeError("AWS_REGION missing")
    ), \
         patch.object(pf, "check_s3_bucket") as s3, \
         patch.object(pf, "check_deploy_drift") as drift, \
         patch.object(pf, "check_arcticdb_fresh") as fresh, \
         patch.object(pf, "check_universe_freshness_receipt") as universe:
        with pytest.raises(RuntimeError, match="AWS_REGION"):
            pf.run()

    s3.assert_not_called()
    drift.assert_not_called()
    fresh.assert_not_called()
    universe.assert_not_called()
