"""Tests for inference/stages/write_output.py — predictions writing + email builder."""

import json
from unittest.mock import MagicMock, patch

import pytest

import inference.stages.write_output as wo
from inference.stages.write_output import write_predictions, get_veto_threshold


class TestWritePredictionsDryRun:
    """Test write_predictions in dry-run mode (no S3)."""

    def test_dry_run_prints(self, capsys):
        predictions = [
            {"ticker": "AAPL", "predicted_direction": "UP", "prediction_confidence": 0.75},
            {"ticker": "MSFT", "predicted_direction": "DOWN", "prediction_confidence": 0.68},
        ]
        metrics = {"model_version": "test-v1", "hit_rate_30d_rolling": 0.55}

        write_predictions(predictions, "2026-04-08", "bucket", metrics, dry_run=True)

        captured = capsys.readouterr()
        assert "PREDICTIONS (dry-run)" in captured.out
        assert "AAPL" in captured.out
        assert "METRICS (dry-run)" in captured.out

    def test_dry_run_counts_high_confidence(self, capsys):
        predictions = [
            {"ticker": "A", "prediction_confidence": 0.80},
            {"ticker": "B", "prediction_confidence": 0.50},
            {"ticker": "C", "prediction_confidence": 0.70},
        ]
        write_predictions(predictions, "2026-04-08", "bucket", {}, dry_run=True, veto_threshold=0.65)

        captured = capsys.readouterr()
        output = json.loads(captured.out.split("=== PREDICTIONS (dry-run) ===\n")[1].split("\n=== METRICS")[0])
        assert output["n_high_confidence"] == 2  # A (0.80) and C (0.70) >= 0.65

    def test_dry_run_includes_date(self, capsys):
        write_predictions([], "2026-04-08", "bucket", {}, dry_run=True)
        captured = capsys.readouterr()
        assert "2026-04-08" in captured.out


class TestWritePredictionsS3:
    """Test write_predictions with mocked S3."""

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    @patch("inference.stages.write_output._s3_put_json")
    def test_writes_three_keys(self, mock_put):
        predictions = [{"ticker": "AAPL", "prediction_confidence": 0.75}]
        write_predictions(predictions, "2026-04-08", "bucket", {"model_version": "v1"})
        assert mock_put.call_count == 3  # dated, latest, metrics

    @patch.dict("sys.modules", {"boto3": MagicMock()})
    @patch("inference.stages.write_output._s3_put_json", side_effect=Exception("S3 error"))
    def test_handles_write_failure(self, mock_put):
        # Should not raise — failures are logged but not propagated
        write_predictions([{"ticker": "AAPL"}], "2026-04-08", "bucket", {})


class TestGetVetoThresholdExtended:
    """Additional veto threshold tests."""

    def setup_method(self):
        wo._predictor_params_cache = None
        wo._predictor_params_loaded = False

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.65})
    def test_case_insensitive_regime(self, _mock):
        result = get_veto_threshold("bucket", "  BEAR  ")
        assert result == pytest.approx(0.55)

    @patch.object(wo, "_load_predictor_params_from_s3", return_value={"veto_confidence": 0.65})
    def test_none_regime(self, _mock):
        result = get_veto_threshold("bucket", None)
        assert result == pytest.approx(0.65)
