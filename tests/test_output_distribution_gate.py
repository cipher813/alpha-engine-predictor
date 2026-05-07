"""Tests for the output-distribution gate (audit Phase 2a).

Validates calibrator output-shape invariants on a synthetic alpha sweep.
Catches the calibrator-plateau / saturation / direction-skew failure
class that bit production 2026-04-28, 2026-05-04, 2026-05-07.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.output_distribution_gate import (
    OutputDistributionGateResult,
    validate_calibrator_distribution,
)


# ── Fake calibrators for test isolation (no sklearn dependency) ──────────

class _LinearCalibrator:
    """Healthy calibrator: linear monotonic mapping alpha → p_up."""
    def __init__(self):
        self._fitted = True

    def calibrate_prediction(self, raw_alpha, label_clip=0.15):
        alpha = float(np.clip(raw_alpha, -label_clip, label_clip))
        p_up = float(np.clip(0.5 + alpha / (2.0 * label_clip), 0.0, 1.0))
        direction = "UP" if alpha >= 0 else "DOWN"
        return {
            "p_up": round(p_up, 4),
            "predicted_direction": direction,
            "prediction_confidence": round(p_up if direction == "UP" else 1 - p_up, 4),
        }


class _PlateauCalibrator:
    """Degenerate calibrator: maps everything to p_up=0.458 (2026-05-07 mode)."""
    def __init__(self):
        self._fitted = True

    def calibrate_prediction(self, raw_alpha, label_clip=0.15):
        return {"p_up": 0.458, "predicted_direction": "DOWN", "prediction_confidence": 0.542}


class _SaturatedCalibrator:
    """Mostly-saturated calibrator: maps to floor 0.01 except at extremes."""
    def __init__(self):
        self._fitted = True

    def calibrate_prediction(self, raw_alpha, label_clip=0.15):
        if raw_alpha > 0.04:
            return {"p_up": 0.99, "predicted_direction": "UP", "prediction_confidence": 0.99}
        return {"p_up": 0.01, "predicted_direction": "DOWN", "prediction_confidence": 0.99}


class _CompressedCalibrator:
    """Tight cluster around 0.50 — compressed output (2026-04-28 mode)."""
    def __init__(self):
        self._fitted = True

    def calibrate_prediction(self, raw_alpha, label_clip=0.15):
        # All outputs in narrow band [0.499, 0.502]
        p_up = 0.500 + 0.0005 * np.sign(raw_alpha)
        direction = "UP" if raw_alpha >= 0 else "DOWN"
        return {"p_up": round(p_up, 4), "predicted_direction": direction,
                "prediction_confidence": 0.502}


class _SkewedCalibrator:
    """All-DOWN bias — false-collapse mirror (2026-05-04 mode)."""
    def __init__(self):
        self._fitted = True

    def calibrate_prediction(self, raw_alpha, label_clip=0.15):
        # Always returns p_up < 0.5 → direction = DOWN, regardless of alpha
        p_up = 0.30 + raw_alpha * 0.5  # vary by alpha but always < 0.5
        p_up = max(0.05, min(0.49, p_up))  # clip into [0.05, 0.49]
        return {
            "p_up": round(p_up, 4), "predicted_direction": "DOWN",
            "prediction_confidence": round(1 - p_up, 4),
        }


class _UnfittedCalibrator:
    """Calibrator that hasn't been fitted yet — gate should pass-by-default."""
    def __init__(self):
        self._fitted = False

    def calibrate_prediction(self, raw_alpha, label_clip=0.15):
        # Linear fallback — but gate shouldn't even call this since unfitted.
        raise RuntimeError("calibrate_prediction shouldn't be called on unfitted")


# ── Healthy calibrator passes ────────────────────────────────────────────

class TestHealthyCalibrator:

    def test_linear_calibrator_passes_all_checks(self):
        result = validate_calibrator_distribution(_LinearCalibrator())
        assert result.passed is True
        assert result.failed_check is None
        assert "all checks passed" in result.reason

    def test_metrics_populated_on_pass(self):
        result = validate_calibrator_distribution(_LinearCalibrator())
        m = result.metrics
        assert m["calibrator_fitted"] is True
        assert m["n_synthetic"] == 25
        assert m["n_unique_p_up"] >= 8
        assert m["saturation_rate"] <= 0.25
        assert m["stdev_p_up"] >= 0.005


# ── Plateau (uniqueness) failure ─────────────────────────────────────────

class TestPlateauFailure:

    def test_plateau_calibrator_fails_uniqueness(self):
        result = validate_calibrator_distribution(_PlateauCalibrator())
        assert result.passed is False
        assert result.failed_check == "unique_p_up"
        assert "1 unique p_up values" in result.reason
        assert "2026-05-07-class plateau" in result.reason

    def test_uniqueness_failure_records_metrics(self):
        result = validate_calibrator_distribution(_PlateauCalibrator())
        # n_unique should be 1 (every input maps to 0.458)
        assert result.metrics["n_unique_p_up"] == 1


# ── Saturation failure ───────────────────────────────────────────────────

class TestSaturationFailure:

    def test_saturated_calibrator_fails_saturation(self):
        result = validate_calibrator_distribution(_SaturatedCalibrator())
        assert result.passed is False
        # Either uniqueness or saturation could fire first depending on
        # check order; saturated calibrator only has 2 unique values
        # which fails uniqueness check first (priority order in the gate).
        assert result.failed_check in ("unique_p_up", "saturation_rate")


# ── Compressed-output failure ────────────────────────────────────────────

class TestCompressedFailure:

    def test_compressed_calibrator_fails_stdev(self):
        # Need higher uniqueness threshold relaxation since compressed has
        # 2 unique values (sign-based). But pass uniqueness via custom params.
        result = validate_calibrator_distribution(
            _CompressedCalibrator(),
            min_unique_p_up=2,  # allow the 2-bucket sign-based output
        )
        assert result.passed is False
        # Compressed output fails on stdev (output too tight).
        assert result.failed_check == "stdev"
        assert "compressed" in result.reason or "below threshold" in result.reason


# ── Direction skew failure ───────────────────────────────────────────────

class TestDirectionSkewFailure:

    def test_skewed_calibrator_fails_direction_skew(self):
        result = validate_calibrator_distribution(
            _SkewedCalibrator(),
            min_unique_p_up=2,  # allow varied but all-DOWN output
        )
        assert result.passed is False
        assert result.failed_check == "direction_skew"
        # All 25 synthetic alphas → DOWN → skew = 1.0
        assert result.metrics["direction_skew"] == 1.0
        assert result.metrics["n_up"] == 0
        assert result.metrics["n_down"] == 25


# ── Unfitted calibrator passes by default ────────────────────────────────

class TestUnfittedCalibrator:

    def test_unfitted_calibrator_passes_with_explanation(self):
        result = validate_calibrator_distribution(_UnfittedCalibrator())
        assert result.passed is True
        assert "not fitted" in result.reason
        assert result.metrics["calibrator_fitted"] is False


# ── Threshold customization ──────────────────────────────────────────────

class TestThresholdCustomization:

    def test_tighter_min_unique_blocks_borderline(self):
        # Linear calibrator yields ~25 unique values — tighten threshold to 30.
        result = validate_calibrator_distribution(
            _LinearCalibrator(), min_unique_p_up=30,
        )
        assert result.passed is False
        assert result.failed_check == "unique_p_up"

    def test_looser_min_unique_allows_borderline(self):
        # Plateau calibrator has 1 unique — relax threshold to 1.
        # But it should still fail on saturation or stdev or skew.
        result = validate_calibrator_distribution(
            _PlateauCalibrator(), min_unique_p_up=1,
        )
        assert result.passed is False
        # 0.458 isn't saturated (between 0.011 and 0.989), so saturation passes.
        # Stdev is 0 (all the same value) → fails on stdev.
        assert result.failed_check == "stdev"

    def test_tighter_alpha_range_changes_results(self):
        # With a tighter alpha sweep (-0.01, +0.01), linear calibrator's
        # 25 outputs span a smaller p_up range, lower stdev. Confirm metrics
        # reflect the narrower input range.
        result = validate_calibrator_distribution(
            _LinearCalibrator(), alpha_range=(-0.01, 0.01),
        )
        # Should still pass with default min_stdev=0.005, since linear over
        # ±0.01 range yields ~0.034..0.067 around 0.5 → stdev ~0.02.
        assert result.passed is True
        assert result.metrics["alpha_range"] == [-0.01, 0.01]


# ── Result dataclass ─────────────────────────────────────────────────────

class TestResultDataclass:

    def test_result_is_frozen(self):
        result = validate_calibrator_distribution(_LinearCalibrator())
        import pytest
        with pytest.raises(Exception):
            result.passed = False  # frozen dataclass — mutation should raise

    def test_result_has_all_fields(self):
        result = validate_calibrator_distribution(_LinearCalibrator())
        assert hasattr(result, "passed")
        assert hasattr(result, "failed_check")
        assert hasattr(result, "reason")
        assert hasattr(result, "metrics")
