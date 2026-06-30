"""Tests for training.io_spec.TrainingIOSpec — the single source of truth for
the predictor's training data basis + output paths + promotion eligibility.

PR7-step-7b (epic config#1433/#1434): a CRSP total-return SHADOW run must be
hard-isolated from the live champion. These tests pin that contract at the spec
layer (where it is enforced) — the live spec is byte-identical to today, and
the shadow spec's prefixes are provably disjoint from the live champion paths
with promotion + model-zoo registration disabled.
"""

from __future__ import annotations

import sys
import types

import pytest

from training.io_spec import TrainingIOSpec


# The live champion artifact paths a shadow run must NEVER write.
_LIVE_WEIGHTS_PREFIX = "predictor/weights/meta/"
_LIVE_MANIFEST = "predictor/weights/meta/manifest.json"
_LIVE_FEATURE_LIST = "predictor/weights/meta/feature_list.json"
_LIVE_SUMMARY_LATEST = "predictor/metrics/training_summary_latest.json"


@pytest.fixture
def fake_config(monkeypatch):
    """Inject a minimal fake ``config`` module so ``TrainingIOSpec.live()`` (which
    reads the live S3 keys from config) resolves without a real predictor.yaml."""
    mod = types.ModuleType("config")
    mod.META_WEIGHTS_PREFIX = _LIVE_WEIGHTS_PREFIX
    mod.META_MANIFEST_KEY = _LIVE_MANIFEST
    mod.META_FEATURE_LIST_KEY = _LIVE_FEATURE_LIST
    monkeypatch.setitem(sys.modules, "config", mod)
    return mod


# ── Live spec — must equal today's behaviour ────────────────────────────────
def test_live_reads_canonical_universe_and_close(fake_config):
    io = TrainingIOSpec.live()
    assert io.universe_lib == "universe"
    assert io.close_col == "Close"
    assert io.is_shadow is False
    assert io.shadow_basis is None


def test_live_writes_live_champion_paths(fake_config):
    io = TrainingIOSpec.live()
    assert io.weights_prefix == _LIVE_WEIGHTS_PREFIX
    assert io.manifest_key == _LIVE_MANIFEST
    assert io.feature_list_key == _LIVE_FEATURE_LIST
    assert io.summary_key("2026-06-30") == (
        "predictor/metrics/training_summary_2026-06-30.json"
    )
    assert io.summary_latest_key == _LIVE_SUMMARY_LATEST
    assert io.oos_rows_key("2026-06-30") == (
        "predictor/diagnostics/oos_rows/2026-06-30.parquet"
    )
    assert io.oos_rows_latest_key == "predictor/diagnostics/oos_rows/latest.parquet"


def test_live_is_promotable(fake_config):
    io = TrainingIOSpec.live()
    assert io.allow_live_promote is True
    assert io.register_in_zoo is True
    assert io.write_side_artifacts is True


# ── Shadow spec (crsp) — isolated + non-promotable ──────────────────────────
def test_shadow_crsp_reads_scratch_lib_and_total_return_close():
    io = TrainingIOSpec.shadow("crsp")
    assert io.universe_lib == "universe_crsp"
    assert io.close_col == "total_return_close"
    assert io.is_shadow is True
    assert io.shadow_basis == "crsp"


def test_shadow_crsp_outputs_are_isolated_under_shadow_prefixes():
    io = TrainingIOSpec.shadow("crsp")
    assert io.weights_prefix == "predictor/weights_shadow/crsp/"
    assert io.manifest_key == "predictor/weights_shadow/crsp/manifest.json"
    assert io.feature_list_key == "predictor/weights_shadow/crsp/feature_list.json"
    assert io.summary_key("2026-06-30") == (
        "predictor/metrics_shadow/crsp/training_summary_2026-06-30.json"
    )
    assert io.summary_latest_key == (
        "predictor/metrics_shadow/crsp/training_summary_latest.json"
    )
    assert io.oos_rows_key("2026-06-30") == (
        "predictor/diagnostics_shadow/crsp/oos_rows/2026-06-30.parquet"
    )
    assert io.predictions_prefix == "predictor/predictions_shadow/crsp/"


def test_shadow_crsp_cannot_promote_or_register():
    """The core guarantee: an evidence-only run can never touch the champion."""
    io = TrainingIOSpec.shadow("crsp")
    assert io.allow_live_promote is False
    assert io.register_in_zoo is False
    assert io.write_side_artifacts is False


def test_shadow_paths_are_disjoint_from_live_champion():
    """No shadow output key may equal — or be nested under — any live champion
    artifact path. This is the structural 'shadow never writes the live
    champion path' assertion at the path layer."""
    io = TrainingIOSpec.shadow("crsp")
    live_paths = {
        _LIVE_WEIGHTS_PREFIX,
        _LIVE_MANIFEST,
        _LIVE_FEATURE_LIST,
        _LIVE_SUMMARY_LATEST,
        "predictor/diagnostics/oos_rows/",
        "predictor/predictions/",
    }
    shadow_paths = [
        io.weights_prefix,
        io.manifest_key,
        io.feature_list_key,
        io.summary_key("2026-06-30"),
        io.summary_latest_key,
        io.oos_rows_key("2026-06-30"),
        io.oos_rows_latest_key,
        io.predictions_prefix,
    ]
    for sp in shadow_paths:
        assert sp not in live_paths
        # No shadow key starts with the live weights/diagnostics/predictions
        # prefixes (which would mean writing inside the live tree).
        assert not sp.startswith(_LIVE_WEIGHTS_PREFIX)
        assert not sp.startswith("predictor/diagnostics/oos_rows/")
        assert not sp.startswith("predictor/predictions/")
        assert not sp.startswith("predictor/metrics/")


def test_shadow_unknown_basis_raises():
    with pytest.raises(ValueError, match="unknown shadow basis"):
        TrainingIOSpec.shadow("bogus")


# ── resolve() — arg + env precedence ────────────────────────────────────────
def test_resolve_none_and_empty_env_is_live(fake_config):
    io = TrainingIOSpec.resolve(None, env={})
    assert io.is_shadow is False
    assert io.universe_lib == "universe"


def test_resolve_explicit_arg_selects_shadow():
    io = TrainingIOSpec.resolve("crsp", env={})
    assert io.is_shadow is True
    assert io.shadow_basis == "crsp"


def test_resolve_env_flag_selects_shadow():
    io = TrainingIOSpec.resolve(None, env={"CRSP_SHADOW_ENABLED": "true"})
    assert io.is_shadow is True
    assert io.shadow_basis == "crsp"  # SHADOW_BASIS defaults to crsp


def test_resolve_env_flag_honours_explicit_basis():
    io = TrainingIOSpec.resolve(
        None, env={"CRSP_SHADOW_ENABLED": "1", "SHADOW_BASIS": "crsp"}
    )
    assert io.shadow_basis == "crsp"


def test_resolve_explicit_arg_wins_over_absent_env(fake_config):
    # falsey env flag → live unless arg selects shadow
    io_live = TrainingIOSpec.resolve(None, env={"CRSP_SHADOW_ENABLED": "off"})
    assert io_live.is_shadow is False
    io_shadow = TrainingIOSpec.resolve("crsp", env={"CRSP_SHADOW_ENABLED": "off"})
    assert io_shadow.is_shadow is True
