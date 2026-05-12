"""Tests for the CloudWatch ``research_join_*`` gauges emitted by
``training/meta_trainer.py`` after the per-fold research-signal join.

ROADMAP L1465 (added 2026-05-09, closed 2026-05-12 as deliberate-with-
monitoring). The investigation confirmed that dropping rows where the
ticker isn't in the snapshot's ``universe`` is intentional — research
ranks only the tracked-population universe (~25 tickers), so non-
population rows have no real research signal. Filling with neutral
defaults would reproduce the pre-2026-04-28 meta-model collapse (per
``feedback_no_silent_fails``).

These tests guard the monitoring half — gauges + ratio math — so a
future change to the join logic doesn't silently grow the drop rate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.meta_trainer import _emit_research_join_coverage_metrics


_REPO = Path(__file__).parent.parent


def _src(rel_path: str) -> str:
    return (_REPO / rel_path).read_text()


# ── CloudWatch emit contract ──────────────────────────────────────────


class TestResearchJoinCoverageMetrics:

    def test_emits_four_metrics(self):
        mock_cw = MagicMock()
        with patch("boto3.client", return_value=mock_cw):
            _emit_research_join_coverage_metrics(
                kept=496,
                dropped_no_snapshot=1720831,
                dropped_ticker_missing=22054,
            )
        mock_cw.put_metric_data.assert_called_once()
        kwargs = mock_cw.put_metric_data.call_args.kwargs
        assert kwargs["Namespace"] == "AlphaEngine/Predictor"
        names = {m["MetricName"] for m in kwargs["MetricData"]}
        assert names == {
            "research_join_kept_count",
            "research_join_dropped_no_snapshot_count",
            "research_join_dropped_ticker_missing_count",
            "research_join_kept_ratio",
        }

    def test_metric_values_match_inputs(self):
        mock_cw = MagicMock()
        with patch("boto3.client", return_value=mock_cw):
            _emit_research_join_coverage_metrics(
                kept=496,
                dropped_no_snapshot=1720831,
                dropped_ticker_missing=22054,
            )
        kwargs = mock_cw.put_metric_data.call_args.kwargs
        by_name = {m["MetricName"]: m for m in kwargs["MetricData"]}
        assert by_name["research_join_kept_count"]["Value"] == 496.0
        assert (
            by_name["research_join_dropped_no_snapshot_count"]["Value"]
            == 1720831.0
        )
        assert (
            by_name["research_join_dropped_ticker_missing_count"]["Value"]
            == 22054.0
        )

    def test_kept_ratio_excludes_no_snapshot_drops(self):
        # The ratio is meaningful only inside "we had a snapshot for the
        # date" — pre-corpus dates aren't a coverage issue. Denominator
        # is kept + ticker_missing, NOT kept + ticker_missing + no_snapshot.
        mock_cw = MagicMock()
        with patch("boto3.client", return_value=mock_cw):
            _emit_research_join_coverage_metrics(
                kept=496,
                dropped_no_snapshot=1720831,
                dropped_ticker_missing=22054,
            )
        kwargs = mock_cw.put_metric_data.call_args.kwargs
        ratio = next(
            m["Value"] for m in kwargs["MetricData"]
            if m["MetricName"] == "research_join_kept_ratio"
        )
        expected = 496 / (496 + 22054)
        assert abs(ratio - expected) < 1e-9

    def test_zero_denominator_safe(self):
        # No real signals AND no ticker-missing drops — every row dropped
        # for "no snapshot". The ratio should not raise; it returns 0.0.
        mock_cw = MagicMock()
        with patch("boto3.client", return_value=mock_cw):
            _emit_research_join_coverage_metrics(
                kept=0,
                dropped_no_snapshot=1000,
                dropped_ticker_missing=0,
            )
        kwargs = mock_cw.put_metric_data.call_args.kwargs
        ratio = next(
            m["Value"] for m in kwargs["MetricData"]
            if m["MetricName"] == "research_join_kept_ratio"
        )
        assert ratio == 0.0

    def test_full_coverage_ratio_is_one(self):
        mock_cw = MagicMock()
        with patch("boto3.client", return_value=mock_cw):
            _emit_research_join_coverage_metrics(
                kept=500,
                dropped_no_snapshot=0,
                dropped_ticker_missing=0,
            )
        kwargs = mock_cw.put_metric_data.call_args.kwargs
        ratio = next(
            m["Value"] for m in kwargs["MetricData"]
            if m["MetricName"] == "research_join_kept_ratio"
        )
        assert ratio == 1.0

    def test_iam_failure_does_not_propagate(self):
        # Mirror ``_emit_nan_feature_tickers_metric`` — observability
        # must never block training.
        mock_cw = MagicMock()
        mock_cw.put_metric_data.side_effect = Exception("AccessDenied")
        with patch("boto3.client", return_value=mock_cw):
            # Must not raise.
            _emit_research_join_coverage_metrics(
                kept=496,
                dropped_no_snapshot=1720831,
                dropped_ticker_missing=22054,
            )

    def test_boto3_import_failure_does_not_propagate(self):
        # If boto3 itself is unavailable (e.g. local dev without AWS
        # SDK), the WARN-and-continue path must still hold.
        def _raise(*_a, **_kw):
            raise Exception("boto3 not installed")
        with patch("boto3.client", _raise):
            _emit_research_join_coverage_metrics(
                kept=496,
                dropped_no_snapshot=1720831,
                dropped_ticker_missing=22054,
            )


# ── Source-level wiring contract ──────────────────────────────────────


class TestEmitWiredAtJoinLogSite:

    def test_emit_called_after_research_join_log(self):
        src = _src("training/meta_trainer.py")
        # The emit must be wired at the existing post-fold-loop log site
        # so kept/dropped counts go to CW from the same code path that
        # already logs them.
        assert "_emit_research_join_coverage_metrics(" in src
        assert "kept=n_rows_with_real_signals" in src
        assert (
            "dropped_no_snapshot=n_rows_dropped_no_signals_snapshot"
            in src
        )
        assert (
            "dropped_ticker_missing=n_rows_dropped_ticker_missing" in src
        )

    def test_helper_is_module_level(self):
        # Module-level so it's directly importable from tests + doesn't
        # close over loop locals.
        src = _src("training/meta_trainer.py")
        assert (
            "def _emit_research_join_coverage_metrics(" in src
        )
