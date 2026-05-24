"""Tests for `_annotate_subsample_noise_floor` — the L980 WARN helper.

Pins:
  1. Annotates `n_pct_change_vs_prior` + `prior_n` when prior summary present.
  2. WARNs on >|30%| swing.
  3. Does not WARN below threshold.
  4. Gracefully no-ops when prior absent (best-effort, no raise).
  5. Skips components with missing n / prior_n / non-numeric values.
"""
from __future__ import annotations

import io
import json
import logging
from unittest.mock import MagicMock

import pytest

from training.train_handler import _annotate_subsample_noise_floor


def _make_s3_client_with_prior(prior_summary: dict) -> MagicMock:
    """Build a mock S3 client whose `get_object` returns the given prior summary."""
    s3 = MagicMock()
    body = io.BytesIO(json.dumps(prior_summary).encode())
    s3.get_object.return_value = {"Body": body}
    return s3


def _make_s3_client_no_prior() -> MagicMock:
    s3 = MagicMock()
    s3.get_object.side_effect = Exception("NoSuchKey")
    return s3


def test_annotates_pct_change_and_prior_n_when_prior_present():
    prior = {"short_history_subsample": {"volatility": {"n": 821}}}
    current = {"short_history_subsample": {"volatility": {"n": 263}}}
    s3 = _make_s3_client_with_prior(prior)
    _annotate_subsample_noise_floor(current, s3, "alpha-engine-research")
    vol = current["short_history_subsample"]["volatility"]
    # 263 - 821 = -558; -558 / 821 * 100 ≈ -67.96%
    assert vol["n_pct_change_vs_prior"] == pytest.approx(-67.96, abs=0.05)
    assert vol["prior_n"] == 821


def test_warns_when_swing_exceeds_30_pct(caplog):
    prior = {"short_history_subsample": {"volatility": {"n": 821}}}
    current = {"short_history_subsample": {"volatility": {"n": 263}}}
    s3 = _make_s3_client_with_prior(prior)
    with caplog.at_level(logging.WARNING, logger="training.train_handler"):
        _annotate_subsample_noise_floor(current, s3, "alpha-engine-research")
    warning_lines = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("subsample noise-floor" in msg for msg in warning_lines)
    assert any("volatility.n changed" in msg for msg in warning_lines)


def test_no_warn_when_swing_within_noise_floor(caplog):
    # 821 → 800 = -2.56% — well within the 30% floor
    prior = {"short_history_subsample": {"volatility": {"n": 821}}}
    current = {"short_history_subsample": {"volatility": {"n": 800}}}
    s3 = _make_s3_client_with_prior(prior)
    with caplog.at_level(logging.WARNING, logger="training.train_handler"):
        _annotate_subsample_noise_floor(current, s3, "alpha-engine-research")
    warning_lines = [
        r.message for r in caplog.records
        if r.levelno == logging.WARNING and "noise-floor" in r.message
    ]
    assert len(warning_lines) == 0
    # But pct_change should still be annotated
    assert (
        current["short_history_subsample"]["volatility"]["n_pct_change_vs_prior"]
        == pytest.approx(-2.56, abs=0.05)
    )


def test_no_op_when_prior_missing():
    current = {"short_history_subsample": {"volatility": {"n": 263}}}
    s3 = _make_s3_client_no_prior()
    _annotate_subsample_noise_floor(current, s3, "alpha-engine-research")
    # No n_pct_change_vs_prior should be added (prior unavailable).
    assert "n_pct_change_vs_prior" not in current["short_history_subsample"]["volatility"]


def test_skips_component_with_missing_n():
    # `momentum` is a deterministic_baseline with no `n` field — must skip
    # without raising.
    prior = {"short_history_subsample": {"momentum": {"kind": "deterministic_baseline"}}}
    current = {"short_history_subsample": {"momentum": {"kind": "deterministic_baseline"}}}
    s3 = _make_s3_client_with_prior(prior)
    _annotate_subsample_noise_floor(current, s3, "alpha-engine-research")
    assert "n_pct_change_vs_prior" not in current["short_history_subsample"]["momentum"]


def test_no_short_history_subsample_block_is_noop():
    current = {"some_other_field": 123}
    s3 = MagicMock()
    _annotate_subsample_noise_floor(current, s3, "alpha-engine-research")
    # Nothing should change; no raise.
    assert current == {"some_other_field": 123}
    # s3 should not even be hit since the early-return fires before the read.
    assert not s3.get_object.called


def test_division_by_zero_safe():
    # prior_n=0 must not raise ZeroDivisionError.
    prior = {"short_history_subsample": {"volatility": {"n": 0}}}
    current = {"short_history_subsample": {"volatility": {"n": 100}}}
    s3 = _make_s3_client_with_prior(prior)
    _annotate_subsample_noise_floor(current, s3, "alpha-engine-research")
    assert (
        "n_pct_change_vs_prior"
        not in current["short_history_subsample"]["volatility"]
    )
