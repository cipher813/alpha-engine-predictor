"""
Tests for PredictorPreflight.run() mode composition.

BasePreflight primitives are tested in alpha-engine-lib. The deploy-drift
check has its own test file (test_preflight_drift.py). These tests verify
that ``run()`` composes the expected primitive calls in the expected order
— and that ``run_for_drift_gate()`` is a strict subset for the SF gate.

Data-freshness assertions (universe + macro/SPY + inference-macro symbols)
moved upstream to ``alpha-engine-data``'s preflight 2026-05-05; the data
step in every Step Function hard-fails on staleness before any consumer
runs, so re-checking here is redundant.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.preflight import PredictorPreflight


def _make() -> PredictorPreflight:
    return PredictorPreflight(bucket="test-bucket")


def test_run_composes_full_check_sequence():
    """Verify the full primitive call sequence: env + S3 + drift + weights."""
    pf = _make()
    with patch.object(pf, "check_env_vars") as env, \
         patch.object(pf, "check_s3_bucket") as s3, \
         patch.object(pf, "check_deploy_drift") as drift, \
         patch.object(pf, "check_s3_key") as s3_key:
        pf.run()

    env.assert_called_once_with("AWS_REGION")
    s3.assert_called_once()
    drift.assert_called_once()
    s3_key.assert_called_once()
    assert s3_key.call_args.args[0] == "predictor/weights/meta/meta_model.pkl"


def test_run_does_not_call_data_freshness_primitives():
    """Data-freshness checks moved upstream to alpha-engine-data preflight.
    The inference Lambda must not re-check macro/universe freshness — the
    SF data step already gated on it."""
    pf = _make()
    with patch.object(pf, "check_env_vars"), \
         patch.object(pf, "check_s3_bucket"), \
         patch.object(pf, "check_deploy_drift"), \
         patch.object(pf, "check_s3_key"), \
         patch.object(pf, "check_arcticdb_fresh") as fresh:
        pf.run()

    fresh.assert_not_called()


def test_run_for_drift_gate_is_strict_subset():
    """The drift gate runs ONLY env + S3 + image-SHA. No model weights,
    no macro reads — those are predict-path concerns and running them on
    the gate caused the 2026-05-01 SF timeout cascade."""
    pf = _make()
    with patch.object(pf, "check_env_vars") as env, \
         patch.object(pf, "check_s3_bucket") as s3, \
         patch.object(pf, "check_deploy_drift") as drift, \
         patch.object(pf, "check_s3_key") as s3_key, \
         patch.object(pf, "check_arcticdb_fresh") as fresh:
        pf.run_for_drift_gate()

    env.assert_called_once_with("AWS_REGION")
    s3.assert_called_once()
    drift.assert_called_once()
    s3_key.assert_not_called()
    fresh.assert_not_called()


def test_first_failure_short_circuits():
    """If env vars fail, no S3 / ArcticDB / GitHub calls happen."""
    import pytest

    pf = _make()
    with patch.object(
        pf, "check_env_vars", side_effect=RuntimeError("AWS_REGION missing")
    ), \
         patch.object(pf, "check_s3_bucket") as s3, \
         patch.object(pf, "check_deploy_drift") as drift, \
         patch.object(pf, "check_arcticdb_fresh") as fresh:
        with pytest.raises(RuntimeError, match="AWS_REGION"):
            pf.run()

    s3.assert_not_called()
    drift.assert_not_called()
    fresh.assert_not_called()
