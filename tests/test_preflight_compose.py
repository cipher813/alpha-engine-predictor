"""
Tests for PredictorPreflight.run() mode composition.

BasePreflight primitives are tested in alpha-engine-lib. The deploy-drift
check has its own test file (test_preflight_drift.py). These tests verify
that ``run()`` composes the expected primitive calls in the expected order
— including the per-ticker universe-freshness scan added 2026-04-30.
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
    """Verify the full primitive call sequence: env + S3 + drift +
    weights + 5 macro symbols + per-ticker universe scan."""
    pf = _make()
    with patch.object(pf, "check_env_vars") as env, \
         patch.object(pf, "check_s3_bucket") as s3, \
         patch.object(pf, "check_deploy_drift") as drift, \
         patch.object(pf, "check_s3_key") as s3_key, \
         patch.object(pf, "check_arcticdb_fresh") as fresh, \
         patch.object(pf, "check_arcticdb_universe_fresh") as universe:
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

    # Per-ticker universe scan
    universe.assert_called_once()
    assert universe.call_args.args[0] == "universe"


def test_macro_freshness_threshold_is_4_days():
    """Macro symbols use the canonical 4-day threshold (covers Fri→Tue
    long weekends + 1d buffer)."""
    pf = _make()
    with patch.object(pf, "check_env_vars"), \
         patch.object(pf, "check_s3_bucket"), \
         patch.object(pf, "check_deploy_drift"), \
         patch.object(pf, "check_s3_key"), \
         patch.object(pf, "check_arcticdb_fresh") as fresh, \
         patch.object(pf, "check_arcticdb_universe_fresh"):
        pf.run()

    for call in fresh.call_args_list:
        assert call.kwargs.get("max_stale_days") == 4


def test_universe_freshness_threshold_is_5_days():
    """Per-ticker scan is one day more permissive (5d) to absorb
    legitimate per-ticker lag (DST/cross-listing edge cases)."""
    pf = _make()
    with patch.object(pf, "check_env_vars"), \
         patch.object(pf, "check_s3_bucket"), \
         patch.object(pf, "check_deploy_drift"), \
         patch.object(pf, "check_s3_key"), \
         patch.object(pf, "check_arcticdb_fresh"), \
         patch.object(pf, "check_arcticdb_universe_fresh") as universe:
        pf.run()

    assert universe.call_args.kwargs.get("max_stale_days") == 5


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
         patch.object(pf, "check_arcticdb_universe_fresh") as universe:
        with pytest.raises(RuntimeError, match="AWS_REGION"):
            pf.run()

    s3.assert_not_called()
    drift.assert_not_called()
    fresh.assert_not_called()
    universe.assert_not_called()
