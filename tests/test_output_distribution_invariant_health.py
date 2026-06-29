"""Tests for the recalibration-invariant inference-time output-health gate.

config#1373 redesign: the inference-time circuit-breaker must halt ONLY on
recalibration-invariant *broken-output* signals (non-finite predicted_alpha,
predicted_alpha collapse, sign skew, p_up saturation) and must NOT halt on
benign isotonic-p_up uniqueness/modal-fraction (a calibrator staircase
artifact that false-halted a healthy low-dispersion book on 2026-06-29).
"""
from __future__ import annotations

import math

from model.output_distribution_gate import validate_live_batch_invariant_health


def _row(alpha, p_up):
    return {
        "ticker": "X",
        "predicted_alpha": alpha,
        "p_up": p_up,
        "predicted_direction": "UP" if (p_up is not None and p_up >= 0.5) else "DOWN",
    }


class TestRegressionLowDispersionPasses:
    """The 2026-06-29 GE-drop false-halt: low isotonic-p_up dispersion but a
    cleanly differentiated tradable signal MUST pass (this is the whole point)."""

    def test_low_p_up_dispersion_but_differentiated_alpha_passes(self):
        # 26 tickers: predicted_alpha cleanly differentiated (all distinct,
        # small magnitudes, both signs — a low-conviction-but-healthy day),
        # yet the isotonic map collapses them onto only ~5 p_up staircase steps.
        preds = []
        staircase = [0.46, 0.47, 0.48, 0.49, 0.50]
        for i in range(26):
            alpha = (i - 12) * 0.0008  # -0.0096 .. +0.0104, all distinct
            preds.append(_row(alpha, staircase[i % len(staircase)]))
        result = validate_live_batch_invariant_health(preds)
        assert result.passed is True, result.reason
        assert result.failed_check is None
        # Legacy p_up shape stats are recorded but observe-only.
        assert result.metrics["n_unique_p_up"] <= 6
        assert result.metrics["n_unique_alpha"] == 26

    def test_isotonic_p_up_uniqueness_alone_never_halts(self):
        # All 20 tickers on the SAME p_up step (extreme staircase collapse) but
        # distinct, healthy predicted_alpha — must still pass: p_up uniqueness/
        # modal-fraction is explicitly demoted to observe-only.
        preds = [_row((i - 10) * 0.001, 0.483) for i in range(20)]
        result = validate_live_batch_invariant_health(preds)
        assert result.passed is True, result.reason
        assert result.metrics["modal_fraction"] == 1.0  # all on one p_up
        assert result.metrics["n_unique_alpha"] == 20


class TestInvariantBrokennessHalts:
    def test_nonfinite_alpha_halts(self):
        preds = [_row(float("nan"), 0.5) for _ in range(8)]
        preds += [_row(0.01 * i, 0.5 + 0.01 * i) for i in range(4)]
        result = validate_live_batch_invariant_health(preds)
        assert result.passed is False
        assert result.failed_check == "alpha_nonfinite_rate"

    def test_constant_alpha_collapse_halts(self):
        # 19 of 20 tickers share one predicted_alpha value -> essentially
        # constant tradable signal (2026-04-28-class compression).
        preds = [_row(0.002, 0.5) for _ in range(19)] + [_row(0.05, 0.6)]
        result = validate_live_batch_invariant_health(preds)
        assert result.passed is False
        assert result.failed_check == "alpha_collapse"

    def test_sign_skew_halts(self):
        # All-positive predicted_alpha (degenerate one-sided), but each value
        # distinct so it is NOT a modal collapse — isolates the skew check.
        preds = [_row(0.001 * (i + 1), 0.55) for i in range(20)]
        result = validate_live_batch_invariant_health(preds)
        assert result.passed is False
        assert result.failed_check == "alpha_sign_skew"

    def test_saturation_halts(self):
        # Differentiated alpha (no alpha-side failure) but most p_up clamped to
        # the calibrator floor -> wide band crushed to the bound.
        preds = [_row((i - 10) * 0.001, 0.010) for i in range(16)]
        preds += [_row((i + 1) * 0.001, 0.6) for i in range(4)]
        result = validate_live_batch_invariant_health(preds)
        assert result.passed is False
        assert result.failed_check == "saturation_rate"


class TestEdgeCases:
    def test_thin_batch_passes(self):
        result = validate_live_batch_invariant_health([_row(0.01, 0.5)] * 3)
        assert result.passed is True
        assert "below min_batch_size" in result.reason

    def test_empty_batch_passes(self):
        result = validate_live_batch_invariant_health([])
        assert result.passed is True

    def test_missing_predicted_alpha_degrades_gracefully(self):
        # No predicted_alpha key at all -> alpha checks skip (not treated as
        # broken); healthy non-saturated p_up -> pass.
        preds = [{"ticker": f"T{i}", "p_up": 0.4 + i * 0.02,
                  "predicted_direction": "UP"} for i in range(12)]
        result = validate_live_batch_invariant_health(preds)
        assert result.passed is True
        assert result.metrics["n_alpha_present"] == 0

    def test_healthy_full_batch_passes(self):
        preds = [_row((i - 13) * 0.003, 0.30 + (i / 26) * 0.45) for i in range(27)]
        result = validate_live_batch_invariant_health(preds)
        assert result.passed is True
        assert result.metrics["alpha_sign_skew"] < 0.97

    def test_metrics_record_observe_only_p_up_stats(self):
        preds = [_row((i - 10) * 0.001, 0.48) for i in range(20)]
        result = validate_live_batch_invariant_health(preds)
        # Observe-only isotonic shape stats present regardless of verdict.
        for k in ("n_unique_p_up", "modal_fraction", "stdev_p_up"):
            assert k in result.metrics
        assert math.isclose(result.metrics["alpha_modal_fraction"], 0.05, abs_tol=0.01)
