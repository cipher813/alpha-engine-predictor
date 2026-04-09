"""Tests for inference/pipeline.py — PipelineContext and stage orchestration."""

import time
from unittest.mock import MagicMock, patch

import pytest

from inference.pipeline import PipelineContext, PipelineAbort, run_pipeline


class TestPipelineContext:
    def test_defaults(self):
        ctx = PipelineContext()
        assert ctx.model_type == "gbm"
        assert ctx.dry_run is False
        assert ctx.predictions == []
        assert ctx.n_skipped == 0

    def test_near_timeout_false(self):
        ctx = PipelineContext(start_ts=time.monotonic(), soft_timeout_s=600)
        assert ctx.near_timeout() is False

    def test_near_timeout_true(self):
        ctx = PipelineContext(start_ts=time.monotonic() - 1000, soft_timeout_s=600)
        assert ctx.near_timeout() is True

    def test_elapsed_seconds(self):
        ctx = PipelineContext(start_ts=time.monotonic() - 5.0)
        assert ctx.elapsed_seconds() >= 4.5

    def test_predictions_are_independent(self):
        ctx1 = PipelineContext()
        ctx2 = PipelineContext()
        ctx1.predictions.append({"ticker": "AAPL"})
        assert len(ctx2.predictions) == 0


class TestPipelineAbort:
    def test_is_exception(self):
        with pytest.raises(PipelineAbort):
            raise PipelineAbort("test abort")


class TestRunPipeline:
    @patch("importlib.import_module")
    def test_runs_all_stages(self, mock_import):
        mock_mod = MagicMock()
        mock_import.return_value = mock_mod

        ctx = PipelineContext(date_str="2026-04-08", start_ts=time.monotonic())
        run_pipeline(ctx)

        assert mock_import.call_count == 6

    @patch("importlib.import_module")
    def test_critical_failure_raises(self, mock_import):
        mock_mod = MagicMock()
        mock_mod.run.side_effect = RuntimeError("model not found")
        mock_import.return_value = mock_mod

        ctx = PipelineContext(date_str="2026-04-08", start_ts=time.monotonic())
        with pytest.raises(RuntimeError, match="model not found"):
            run_pipeline(ctx)

    @patch("importlib.import_module")
    def test_noncritical_failure_continues(self, mock_import):
        call_count = {"n": 0}

        def side_effect(path):
            mod = MagicMock()
            call_count["n"] += 1
            if path == "inference.stages.fetch_alt_data":
                mod.run.side_effect = RuntimeError("alt data unavailable")
            return mod

        mock_import.side_effect = side_effect

        ctx = PipelineContext(date_str="2026-04-08", start_ts=time.monotonic())
        run_pipeline(ctx)

        assert call_count["n"] == 6

    @patch("importlib.import_module")
    def test_pipeline_abort_stops_cleanly(self, mock_import):
        call_count = {"n": 0}

        def side_effect(path):
            mod = MagicMock()
            call_count["n"] += 1
            if path == "inference.stages.load_universe":
                mod.run.side_effect = PipelineAbort("no tickers")
            return mod

        mock_import.side_effect = side_effect

        ctx = PipelineContext(date_str="2026-04-08", start_ts=time.monotonic())
        run_pipeline(ctx)

        assert call_count["n"] == 2
