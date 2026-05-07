"""Tests for the inference-time gate wired into write_predictions.

Per the 2026-05-07 predictor audit Phase 2a-INFER (companion to
test_output_distribution_gate_live_batch.py): the gate is called from
``inference/stages/write_output.py:write_predictions`` before any S3
write. Behind a feature flag (default OFF). When the flag is True and
the gate fails, write_predictions raises RuntimeError without writing
to S3 — fail-closed semantics.
"""
from __future__ import annotations

import os
import sys

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _healthy_predictions(n: int = 27):
    """Diverse p_up values across [0.10, 0.85] — gate should pass."""
    preds = []
    for i in range(n):
        p_up = 0.10 + (i / (n - 1)) * 0.75
        preds.append({
            "ticker": f"T{i:02d}",
            "p_up": round(p_up, 4),
            "p_down": round(1 - p_up, 4),
            "p_flat": 0.0,
            "predicted_alpha": (p_up - 0.5) * 0.05,
            "predicted_direction": "UP" if p_up >= 0.5 else "DOWN",
            "prediction_confidence": round(max(p_up, 1 - p_up), 4),
        })
    return preds


def _degenerate_predictions():
    """The 2026-05-07 incident shape: 16 at p_up=0.458, 5 at 0.010, etc."""
    from collections import Counter
    dist = Counter({0.458: 16, 0.010: 5, 0.521: 2, 0.459: 2, 0.517: 1, 0.429: 1})
    preds = []
    i = 0
    for p_up, count in dist.items():
        for _ in range(count):
            direction = "UP" if p_up >= 0.5 else "DOWN"
            preds.append({
                "ticker": f"T{i:02d}",
                "p_up": p_up,
                "p_down": round(1 - p_up, 4),
                "p_flat": 0.0,
                "predicted_alpha": -0.001,
                "predicted_direction": direction,
                "prediction_confidence": max(p_up, 1 - p_up),
            })
            i += 1
    return preds


# ── Observe-only mode (default) ──────────────────────────────────────────

class TestGateObserveOnly:

    def test_healthy_batch_writes_normally_observe_only(self):
        from inference.stages.write_output import write_predictions
        with patch("inference.stages.write_output._s3_put_json") as mock_put:
            write_predictions(
                _healthy_predictions(),
                "2026-05-07",
                "test-bucket",
                {"model_version": "test"},
            )
        # All three S3 puts should have been attempted.
        assert mock_put.call_count == 3

    def test_degenerate_batch_observe_only_still_writes(self):
        # Default mode: gate runs, logs failure, but does NOT block.
        from inference.stages.write_output import write_predictions
        with patch("inference.stages.write_output._s3_put_json") as mock_put:
            # Don't patch the cfg flag; default is False (observe-only).
            write_predictions(
                _degenerate_predictions(),
                "2026-05-07",
                "test-bucket",
                {"model_version": "test"},
            )
        # S3 writes still happen in observe-only mode.
        assert mock_put.call_count == 3

    def test_metrics_carry_gate_result_on_observe_only(self):
        from inference.stages import write_output
        captured_metrics_body = []

        def fake_put(s3, bucket, key, body):
            if "metrics" in key:
                captured_metrics_body.append(body)

        with patch.object(write_output, "_s3_put_json", side_effect=fake_put):
            write_output.write_predictions(
                _degenerate_predictions(),
                "2026-05-07",
                "test-bucket",
                {"model_version": "test"},
            )

        import json
        assert captured_metrics_body, "no metrics write captured"
        metrics = json.loads(captured_metrics_body[0])
        assert "output_distribution_gate" in metrics
        assert metrics["output_distribution_gate"]["passed"] is False
        assert metrics["output_distribution_gate"]["blocking"] is False
        assert metrics["output_distribution_gate"]["would_have_blocked_if_blocking"] is True


# ── Blocking mode (feature-flagged on) ───────────────────────────────────

class TestGateBlockingMode:

    def test_healthy_batch_writes_normally_under_blocking(self):
        from inference.stages.write_output import write_predictions
        from inference.stages import write_output as wo_module
        with patch("inference.stages.write_output._s3_put_json") as mock_put, \
             patch.object(wo_module.cfg, "OUTPUT_DISTRIBUTION_GATE_INFERENCE_BLOCKING", True, create=True):
            write_predictions(
                _healthy_predictions(),
                "2026-05-07",
                "test-bucket",
                {"model_version": "test"},
            )
        # Healthy batch passes the gate; all writes happen.
        assert mock_put.call_count == 3

    def test_degenerate_batch_blocked_under_blocking(self):
        from inference.stages.write_output import write_predictions
        from inference.stages import write_output as wo_module
        with patch("inference.stages.write_output._s3_put_json") as mock_put, \
             patch.object(wo_module.cfg, "OUTPUT_DISTRIBUTION_GATE_INFERENCE_BLOCKING", True, create=True):
            with pytest.raises(RuntimeError) as excinfo:
                write_predictions(
                    _degenerate_predictions(),
                    "2026-05-07",
                    "test-bucket",
                    {"model_version": "test"},
                )
        assert "refused write at inference time" in str(excinfo.value)
        # NO S3 writes should have happened — fail-closed.
        assert mock_put.call_count == 0


# ── Edge cases ───────────────────────────────────────────────────────────

class TestGateEdgeCases:

    def test_dry_run_skips_gate_via_normal_dry_run_path(self):
        # Dry-run path returns before the S3 writes — but gate runs first.
        from inference.stages.write_output import write_predictions
        with patch("inference.stages.write_output._s3_put_json") as mock_put:
            write_predictions(
                _healthy_predictions(),
                "2026-05-07",
                "test-bucket",
                {"model_version": "test"},
                dry_run=True,
            )
        # No S3 writes in dry-run.
        assert mock_put.call_count == 0

    def test_dry_run_under_blocking_with_degenerate_batch_still_raises(self):
        # Even in dry-run, fail-closed should fire — operator wants to
        # know about a degenerate batch BEFORE simulating a release.
        # Check current order: dry-run check happens AFTER metrics build
        # but BEFORE the S3 writes. Gate raise happens BEFORE dry-run
        # check, so it fires regardless of dry_run.
        from inference.stages.write_output import write_predictions
        from inference.stages import write_output as wo_module
        with patch.object(wo_module.cfg, "OUTPUT_DISTRIBUTION_GATE_INFERENCE_BLOCKING", True, create=True):
            with pytest.raises(RuntimeError):
                write_predictions(
                    _degenerate_predictions(),
                    "2026-05-07",
                    "test-bucket",
                    {"model_version": "test"},
                    dry_run=True,
                )
