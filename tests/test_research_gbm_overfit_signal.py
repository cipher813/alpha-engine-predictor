"""ROADMAP L1816 (2026-05-19) — ResearchGBM overfit signal regression suite.

Covers ``_compute_overfit_signal`` (the train/val IC ratio +
warn-threshold helper) and ``_emit_research_gbm_overfit_metrics``
(the CloudWatch gauge dispatcher).

The overfit surface is observability — the Ridge regularization
absorbs the bounded overfit today — so the tests pin the contract
shape rather than asserting on alpha-impacting behavior. A future
``train_val_ic_ratio`` drift past 3.0 should raise the gauge alarm,
not silently fall through.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from training.meta_trainer import (
    _compute_overfit_signal,
    _emit_research_gbm_overfit_metrics,
)


# ---------------------------------------------------------------------------
# _compute_overfit_signal — ratio + warn flag
# ---------------------------------------------------------------------------


class TestComputeOverfitSignal:

    def test_healthy_ratio_below_threshold_no_warn(self):
        """train_ic 0.20 / val_ic 0.15 → ratio 1.33 < 3.0 → no warn."""
        ratio, warn = _compute_overfit_signal(train_ic=0.20, val_ic=0.15)
        assert ratio == pytest.approx(0.20 / 0.15)
        assert warn is False

    def test_at_2026_05_09_retrain_observation(self):
        """The historical observation that motivated this PR:
        train_ic=0.4328 / val_ic=0.0857 → ratio 5.05 → warn."""
        ratio, warn = _compute_overfit_signal(
            train_ic=0.4328, val_ic=0.0857, warn_threshold=3.0,
        )
        assert ratio == pytest.approx(0.4328 / 0.0857, rel=1e-4)
        assert warn is True  # 5.05 > 3.0

    def test_threshold_boundary_strictly_greater_than(self):
        """Ratio exactly at threshold → no warn (strict >)."""
        ratio, warn = _compute_overfit_signal(
            train_ic=0.30, val_ic=0.10, warn_threshold=3.0,
        )
        # 0.30 / 0.10 = 3.0 exactly
        assert ratio == pytest.approx(3.0)
        assert warn is False

    def test_train_ic_none_returns_none_no_warn(self):
        """Missing train_ic → cannot measure, no warn fired."""
        ratio, warn = _compute_overfit_signal(train_ic=None, val_ic=0.10)
        assert ratio is None
        assert warn is False

    def test_val_ic_near_zero_returns_none_no_warn(self):
        """|val_ic| < 1e-3 noise floor → ratio undefined → None, no warn.

        Prevents a spurious warn fire when the holdout IC is noise-level
        rather than an honest "model learned nothing" signal.
        """
        ratio, warn = _compute_overfit_signal(train_ic=0.30, val_ic=0.0001)
        assert ratio is None
        assert warn is False

    def test_negative_val_ic_uses_abs_in_denominator(self):
        """train_ic 0.30 / val_ic -0.10 → ratio 3.0 (magnitude-based).
        Negative val_ic is its own pathology; the ratio still measures
        train-fit dominance and the warn rule still applies on magnitude."""
        ratio, warn = _compute_overfit_signal(
            train_ic=0.30, val_ic=-0.10, warn_threshold=2.0,
        )
        assert ratio == pytest.approx(3.0)
        assert warn is True

    def test_custom_threshold_overrides_default(self):
        """Threshold is a parameter so the test (and a future re-tune
        in the calling site) can dial sensitivity without touching the
        helper."""
        ratio, warn_at_default = _compute_overfit_signal(
            train_ic=0.20, val_ic=0.10, warn_threshold=3.0,
        )
        _, warn_at_low = _compute_overfit_signal(
            train_ic=0.20, val_ic=0.10, warn_threshold=1.5,
        )
        assert ratio == pytest.approx(2.0)
        assert warn_at_default is False
        assert warn_at_low is True


# ---------------------------------------------------------------------------
# _emit_research_gbm_overfit_metrics — CloudWatch gauge dispatcher
# ---------------------------------------------------------------------------


class TestEmitResearchGbmOverfitMetrics:

    def test_emits_full_metric_set_when_warn(self):
        """All four gauges land when ratio is measured + warn fires."""
        mock_cw = MagicMock()
        with patch("boto3.client", return_value=mock_cw):
            _emit_research_gbm_overfit_metrics(
                train_ic=0.4328, val_ic=0.0857, ratio=5.05, warn=True,
            )
        assert mock_cw.put_metric_data.call_count == 1
        call = mock_cw.put_metric_data.call_args
        assert call.kwargs["Namespace"] == "AlphaEngine/Predictor"
        names = {m["MetricName"] for m in call.kwargs["MetricData"]}
        assert names == {
            "research_gbm_train_ic",
            "research_gbm_val_ic",
            "research_gbm_train_val_ic_ratio",
            "research_gbm_overfit_warn",
        }
        # Warn → 1.0 Count
        warn_metric = next(
            m for m in call.kwargs["MetricData"]
            if m["MetricName"] == "research_gbm_overfit_warn"
        )
        assert warn_metric["Value"] == 1.0

    def test_emits_zero_warn_when_healthy(self):
        """Ratio under threshold → warn gauge emits 0.0 (not absent),
        so the alarm can distinguish "healthy" from "couldn't measure"."""
        mock_cw = MagicMock()
        with patch("boto3.client", return_value=mock_cw):
            _emit_research_gbm_overfit_metrics(
                train_ic=0.20, val_ic=0.15, ratio=1.33, warn=False,
            )
        call = mock_cw.put_metric_data.call_args
        warn_metric = next(
            m for m in call.kwargs["MetricData"]
            if m["MetricName"] == "research_gbm_overfit_warn"
        )
        assert warn_metric["Value"] == 0.0

    def test_skips_emit_when_ratio_none(self):
        """Couldn't measure → no datapoints. The absence of a datapoint
        is the right signal for "missing measurement" — emitting zeros
        would mislead the alarm."""
        mock_cw = MagicMock()
        with patch("boto3.client", return_value=mock_cw):
            _emit_research_gbm_overfit_metrics(
                train_ic=None, val_ic=0.0001, ratio=None, warn=False,
            )
        assert mock_cw.put_metric_data.call_count == 0

    def test_skips_train_ic_metric_when_train_ic_none_but_ratio_present(self):
        """Defensive: in the unusual state where ratio was computed but
        train_ic is None (shouldn't happen given the helper contract,
        but coverage-safe), the train_ic gauge is omitted rather than
        emitting a None-valued datapoint."""
        # This state is contradictory with the helper's own contract
        # but the emit fn is defensive enough to handle it cleanly.
        mock_cw = MagicMock()
        with patch("boto3.client", return_value=mock_cw):
            _emit_research_gbm_overfit_metrics(
                train_ic=None, val_ic=0.10, ratio=2.0, warn=False,
            )
        call = mock_cw.put_metric_data.call_args
        names = {m["MetricName"] for m in call.kwargs["MetricData"]}
        assert "research_gbm_train_ic" not in names
        assert names == {
            "research_gbm_val_ic",
            "research_gbm_train_val_ic_ratio",
            "research_gbm_overfit_warn",
        }

    def test_cloudwatch_failure_does_not_raise(self):
        """Observability is best-effort. A CloudWatch ClientError must
        surface as a WARN log + continue training, never raise."""
        mock_cw = MagicMock()
        mock_cw.put_metric_data.side_effect = RuntimeError("CW throttled")
        with patch("boto3.client", return_value=mock_cw):
            # Should not raise
            _emit_research_gbm_overfit_metrics(
                train_ic=0.40, val_ic=0.08, ratio=5.0, warn=True,
            )
